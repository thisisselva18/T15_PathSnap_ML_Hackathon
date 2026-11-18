"""Train + evaluate the selected MLP candidate classifier.

Owner: P2 (run by everyone)

Usage (from repo root):
    python scripts/train_models.py --model logreg --laptop L3
    python scripts/train_models.py --model lgbm   --laptop L1
    python scripts/train_models.py --model rf     --laptop L4
    python scripts/train_models.py --model mlp --laptop L2 --max-rows 150000

What it does
  1. Loads data/processed/candidates_train.parquet and candidates_student_test.parquet
     (one row per (GPS point, candidate arc); label = 1 if the candidate is the reference arc).
  2. Training rows only: keeps every 3rd GPS point (1 Hz neighbours are near-duplicates) and drops
     points whose aligned reference arc is > 50 m away (unreliable labels, e.g. trip 54).
  3. Group-aware 5-fold CV on the 80 train trips using the SHARED fold file artifacts/folds.json
     (whole trips per fold, so no trip leaks between train and validation).
  4. Refits on all train rows, evaluates on ALL student-test rows (no thinning, no filtering).
  5. Appends one row to results/experiments_<laptop>.csv, saves predictions, model,
     confusion matrix and (MLP only) epoch-wise loss curves.

Primary metric: observation-level Top-1 accuracy (argmax candidate per GPS point vs ref_arc_id).
Points whose true arc is not among the candidates count as wrong.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
SEED = 42
KEY_COLS = ["trip_id", "seg_id", "obs_idx", "t", "cand_rank", "arc_id", "label", "ref_arc_id", "ref_dist", "t_raw"]
# KNN and the other classifiers remain in build_model as historical implementations;
# the active training entry point is intentionally MLP-only.
MODELS = ["mlp"]

COLUMNS = ["model", "family", "laptop", "params_json", "seed", "n_train_rows", "cv_folds",
           "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "cv_top1_std", "cv_log_loss_mean",
           "train_top1", "test_top1", "test_accuracy", "test_precision", "test_recall", "test_f1",
           "test_macro_f1", "test_log_loss", "train_time_s", "infer_time_s_per_1k_rows", "test_candidate_miss_rate",
           "top1_population", "cv_top1_n_observations", "mlp_selected_iterations", "machine", "notes"]

def read_table(stem, data_dir=DATA):
    p=Path(data_dir)/f"{stem}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(Path(data_dir)/f"{stem}.csv",dtype={"trip_id":str})

def save_table(df,path):
    try:
        df.to_parquet(path,index=False); return path
    except (ImportError,ModuleNotFoundError):
        p=path.with_suffix('.csv'); df.to_csv(p,index=False); return p

def positive_probability(model,X):
    probabilities=model.predict_proba(X)
    estimator=model[-1] if hasattr(model,"steps") else model
    labels=list(estimator.classes_)
    return probabilities[:,labels.index(1)] if 1 in labels else np.zeros(len(X))


# ---------------------------------------------------------------- models
def build_model(name, params=None):
    """Return (family, unfitted estimator). params overrides the defaults (used by tune_bayes.py)."""
    p = params or {}
    if name == "logreg":
        return "Linear", make_pipeline(StandardScaler(), LogisticRegression(
            C=p.get("C", 1.0), max_iter=2000))
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return "Boosting", LGBMClassifier(
            n_estimators=p.get("n_estimators", 400), learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 63), max_depth=p.get("max_depth", -1),
            min_child_samples=p.get("min_child_samples", 20), subsample=p.get("subsample", 1.0),
            subsample_freq=1 if p.get("subsample", 1.0) < 1 else 0,
            colsample_bytree=p.get("colsample_bytree", 1.0), reg_lambda=p.get("reg_lambda", 0.0),
            n_jobs=-1, random_state=SEED, verbose=-1)
    if name == "rf":
        return "Ensemble (bagging)", RandomForestClassifier(
            n_estimators=p.get("n_estimators", 150), max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 10), max_features=p.get("max_features", "sqrt"),
            max_samples=0.5, n_jobs=-1, random_state=SEED)
    if name == "mlp":
        return "Neural network", make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=p.get("hidden_layer_sizes", (64, 32)), alpha=p.get("alpha", 1e-4),
            learning_rate_init=p.get("learning_rate_init", 1e-3), early_stopping=False,
            max_iter=p.get("max_iter", 60), random_state=SEED))
    if name == "knn":
        return "Instance-based", make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=p.get("n_neighbors", 25), weights=p.get("weights", "distance"), n_jobs=-1))
    raise ValueError(f"unknown model {name}")


# ---------------------------------------------------------------- metrics
def top1(df, proba, observations=None):
    """Observation-level Top-1; supplied manifest defines denominator; missing candidates are wrong."""
    if observations is not None and len(observations) == 0:
        return float("nan")
    if len(df) == 0:
        return float("nan") if observations is None or len(observations) == 0 else 0.0
    d = df[["trip_id", "obs_idx", "arc_id", "ref_arc_id"]].copy()
    d["p"] = proba
    best = d.loc[d.groupby(["trip_id", "obs_idx"], sort=False)["p"].idxmax()]
    if observations is None:
        return float((best["arc_id"].to_numpy() == best["ref_arc_id"].to_numpy()).mean())
    predicted={(r.trip_id,int(r.obs_idx)):int(r.arc_id==r.ref_arc_id) for r in best.itertuples()}
    return float(np.mean([predicted.get((r.trip_id,int(r.obs_idx)),0) for r in observations.itertuples()]))


def row_metrics(y, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
    }


def grouped_validation_indices(groups, validation_fraction=0.2, seed=SEED):
    """Split candidate rows by trip so a trip cannot straddle fit and validation sets."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0 or len(np.unique(groups)) < 2:
        raise ValueError("grouped validation requires rows from at least two distinct groups")
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    fit_idx, val_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    if set(groups[fit_idx]) & set(groups[val_idx]):
        raise RuntimeError("grouped validation split contains overlapping groups")
    return fit_idx, val_idx


def select_mlp_iterations(X, y, groups, seed=SEED, max_iter=60, patience=8):
    """Choose MLP iterations by validation log loss on held-out trips, never random rows."""
    fit_idx, val_idx = grouped_validation_indices(groups, seed=seed)
    scaler = StandardScaler().fit(X[fit_idx])
    x_fit, x_val = scaler.transform(X[fit_idx]), scaler.transform(X[val_idx])
    estimator = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                              alpha=1e-4, learning_rate_init=1e-3, batch_size="auto",
                              max_iter=1, early_stopping=False, random_state=seed)
    best_loss, best_iter, rounds_without_improvement = np.inf, 1, 0
    classes = np.array([0, 1])
    for iteration in range(1, max_iter + 1):
        estimator.partial_fit(x_fit, y[fit_idx], classes=classes if iteration == 1 else None)
        probabilities = estimator.predict_proba(x_val)
        loss = log_loss(y[val_idx], probabilities, labels=classes)
        if loss < best_loss - 1e-6:
            best_loss, best_iter, rounds_without_improvement = loss, iteration, 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= patience:
                break
    return int(best_iter), float(best_loss)


def fit_mlp_epochs(model, X, y, epochs):
    """Fit an MLP for an explicitly selected epoch count without max_iter warnings.

    `partial_fit` performs one epoch per call. This lets grouped validation select a
    checkpoint without sklearn treating the intentional stopping point as failed
    convergence. The scaler remains fitted only on the supplied training rows.
    """
    if not hasattr(model, "steps") or len(model.steps) != 2:
        raise TypeError("expected the StandardScaler + MLP pipeline")
    scaler, estimator = model.steps[0][1], model.steps[1][1]
    x_scaled = scaler.fit_transform(X)
    classes = np.array([0, 1])
    losses = []
    for epoch in range(int(epochs)):
        estimator.partial_fit(x_scaled, y, classes=classes if epoch == 0 else None)
        losses.append(float(estimator.loss_))
    estimator.loss_curve_ = losses
    estimator.n_iter_ = int(epochs)
    return model


# ---------------------------------------------------------------- data + folds
def get_folds(train_trips, n_folds=5):
    """Shared, deterministic trip -> fold map. Created once, then committed and reused by everyone."""
    path = ARTIFACTS / "folds.json"
    if path.exists():
        folds = json.loads(path.read_text())
    else:
        trips = sorted(map(str, train_trips))
        rng = np.random.default_rng(SEED)
        rng.shuffle(trips)
        folds = {t: i % n_folds for i, t in enumerate(trips)}
        path.write_text(json.dumps(folds, indent=1))
        print(f"created {path} - COMMIT THIS FILE so every laptop uses the same folds")
    missing = set(map(str, train_trips)) - set(folds)
    if missing:
        raise SystemExit(f"folds.json is missing trips {sorted(missing)[:5]} - delete it, regenerate once, recommit")
    return folds


def load_train_test(stride=3, max_ref_dist=50.0, max_rows=0, data_dir=DATA, return_manifest=False):
    """Shared by train_models.py and tune_bayes.py so both see exactly the same rows."""
    data_dir = Path(data_dir)
    tr = read_table("candidates_train", data_dir)
    te = read_table("candidates_student_test", data_dir)
    train_manifest = read_table("observations_train", data_dir)
    tr["trip_id"], te["trip_id"] = tr["trip_id"].astype(str), te["trip_id"].astype(str)
    train_manifest["trip_id"] = train_manifest["trip_id"].astype(str)
    feature_path = data_dir / "feature_list.json"
    if not feature_path.exists():
        feature_path = ARTIFACTS / "feature_list.json"
    flist = json.loads(feature_path.read_text())
    feats = flist or [c for c in tr.columns if c not in KEY_COLS and pd.api.types.is_numeric_dtype(tr[c])]
    tr = tr[(tr["obs_idx"] % stride == 0) & (tr["ref_dist"] <= max_ref_dist)].reset_index(drop=True)
    eligible_manifest = train_manifest[
        (train_manifest["obs_idx"] % stride == 0) & (train_manifest["ref_dist"] <= max_ref_dist)
    ].copy()
    eligible_candidates = tr.copy()
    candidate_sizes = tr.groupby(["trip_id", "obs_idx"], sort=False).size().to_dict()
    if max_rows:
        shuffled = eligible_manifest.sample(frac=1, random_state=SEED)
        picked, total = [], 0
        for row in shuffled.itertuples():
            key = (row.trip_id, int(row.obs_idx))
            count = int(candidate_sizes.get(key, 0))
            if total + count <= max_rows:
                picked.append(key)
                total += count
        selected = pd.DataFrame(picked, columns=["trip_id", "obs_idx"])
    else:
        selected = eligible_manifest[["trip_id", "obs_idx"]]
    tr = tr.merge(selected, on=["trip_id", "obs_idx"], how="inner", validate="many_to_one").reset_index(drop=True)
    train_manifest = eligible_manifest.merge(selected, on=["trip_id", "obs_idx"], how="inner", validate="one_to_one")
    save_table(tr, data_dir / "candidates_train_sample.parquet")
    save_table(train_manifest, data_dir / "observations_train_sample.parquet")
    (data_dir / "training_sample_config.json").write_text(json.dumps({
        "seed": SEED, "row_cap": max_rows or None, "actual_rows": len(tr),
        "actual_observations": len(train_manifest), "training_stride": stride,
        "max_reference_distance_m": max_ref_dist,
        "sampling_unit": "whole eligible (trip_id, obs_idx) groups, including selected observations with no candidates",
        "top1_population": "selected eligible training observation manifest; observations with no candidates count incorrect",
    }, indent=2) + "\n")
    if return_manifest:
        return tr, te, feats, eligible_manifest, eligible_candidates, train_manifest
    return tr, te, feats


def evaluate_on_test(model, te, feats, data_dir=DATA):
    Xte, yte = te[feats].to_numpy(np.float32), te["label"].to_numpy()
    t0 = time.perf_counter()
    p = positive_probability(model,Xte)
    infer = (time.perf_counter() - t0) / len(te) * 1000
    obs_path=Path(data_dir)/"observations_student_test.parquet"
    manifest=read_table("observations_student_test", data_dir) if obs_path.exists() or (Path(data_dir)/"observations_student_test.csv").exists() else None
    if manifest is None:
        raise FileNotFoundError("student-test observation manifest is required for Top-1 evaluation")
    miss=float('nan')
    if manifest is not None:
        candidate_sets={(tid,int(obs)):set(g.arc_id.astype(int)) for (tid,obs),g in te.groupby(['trip_id','obs_idx'])}
        missed=[]
        for r in manifest.itertuples():
            arcs=candidate_sets.get((r.trip_id,int(r.obs_idx)),set())
            missed.append(int(int(r.ref_arc_id) not in arcs))
        miss=float(np.mean(missed)) if missed else float('nan')
    return p, {**row_metrics(yte, p), "top1": top1(te, p, manifest), "candidate_miss_rate":miss, "infer_per_1k": infer}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--laptop", required=True, help="L1/L2/L3/L4 - picks the results file")
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth GPS point for training")
    ap.add_argument("--max-rows", type=int, default=0, help="cap training rows (0 = no cap)")
    ap.add_argument("--max-ref-dist", type=float, default=50.0,
                    help="drop TRAINING points whose reference arc is further than this (unreliable labels)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--data-dir", default=str(DATA), help="processed dataset directory")
    ap.add_argument("--run-tag", default="", help="versioned output subdirectory; e.g. post_correction")
    a = ap.parse_args()

    run_dir = RESULTS / a.run_tag if a.run_tag else RESULTS
    model_dir = ARTIFACTS / "models" / a.run_tag if a.run_tag else ARTIFACTS / "models"
    data_dir = Path(a.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    model_path = model_dir / f"{a.model}.joblib"
    pred_path = run_dir / "preds" / f"{a.model}_student_test.parquet"
    if model_path.exists() or pred_path.exists() or pred_path.with_suffix(".csv").exists():
        raise FileExistsError(f"Refusing to overwrite existing {a.model} outputs for run {a.run_tag or 'default'}")

    tr, te, feats, train_manifest, cv_candidates, sample_manifest = load_train_test(
        a.stride, a.max_ref_dist, a.max_rows, data_dir=data_dir, return_manifest=True)
    print(f"{len(feats)} features: {feats}")
    print(f"train rows {len(tr):,} | selected train observations {len(sample_manifest):,} | "
          f"test rows {len(te):,} | test observations {len(read_table('observations_student_test', data_dir)):,}")

    folds = get_folds(train_manifest["trip_id"].unique())
    fold_of = tr["trip_id"].map(folds).to_numpy()
    manifest_fold = train_manifest["trip_id"].map(folds)
    cv_candidate_fold = cv_candidates["trip_id"].map(folds).to_numpy()
    X, y = tr[feats].to_numpy(np.float32), tr["label"].to_numpy()

    # ---- group-aware CV (whole trips per fold)
    cv = {"macro_f1": [], "top1": [], "log_loss": [], "selected_mlp_iterations": [], "top1_n_observations": []}
    for k in sorted(set(manifest_fold)):
        trn = fold_of != k
        val = cv_candidate_fold == k
        fold_params = {}
        if a.model == "mlp":
            chosen_iter, inner_loss = select_mlp_iterations(
                X[trn], y[trn], tr.loc[trn, "trip_id"].to_numpy(), seed=SEED + int(k))
            fold_params["max_iter"] = chosen_iter
            cv["selected_mlp_iterations"].append(chosen_iter)
            print(f"  fold {k}: grouped stopping selected {chosen_iter} iterations (inner log loss {inner_loss:.5f})")
        m = build_model(a.model, fold_params)[1]
        fit_mlp_epochs(m, X[trn], y[trn], fold_params["max_iter"])
        X_val = cv_candidates.loc[val, feats].to_numpy(np.float32)
        y_val = cv_candidates.loc[val, "label"].to_numpy()
        p = positive_probability(m, X_val)
        rm = row_metrics(y_val, p)
        cv["macro_f1"].append(rm["macro_f1"])
        cv["log_loss"].append(rm["log_loss"])
        val_manifest = train_manifest.loc[manifest_fold == k]
        cv["top1"].append(top1(cv_candidates.loc[val], p, val_manifest))
        cv["top1_n_observations"].append(len(val_manifest))
        print(f"  fold {k}: macroF1 {rm['macro_f1']:.4f}  top1 {cv['top1'][-1]:.4f} "
              f"(n={len(val_manifest)} observations; no-candidate observations count wrong) "
              f"logloss {rm['log_loss']:.4f}")

    # ---- refit on all train, evaluate on student-test
    final_params = {"max_iter": int(np.median(cv["selected_mlp_iterations"]))} if a.model == "mlp" else {}
    family, model = build_model(a.model, final_params)
    t0 = time.perf_counter()
    fit_mlp_epochs(model, X, y, final_params["max_iter"])
    train_time = time.perf_counter() - t0
    train_top1 = top1(tr, positive_probability(model,X), sample_manifest)
    pte, tm = evaluate_on_test(model, te, feats, data_dir=data_dir)

    # ---- save everything
    for d in ["preds", "confusion", "curves"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_table(te[["trip_id", "obs_idx", "arc_id", "ref_arc_id", "label"]].assign(score=pte),
               run_dir / "preds" / f"{a.model}_student_test.parquet")
    cm = confusion_matrix(te["label"].to_numpy(), (pte >= 0.5).astype(int), labels=[0, 1])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        run_dir / "confusion" / f"{a.model}.csv")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(cm,cmap='Blues')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['incorrect','correct'],yticklabels=['incorrect','correct'],xlabel='Predicted',ylabel='Actual',title=f'{a.model.upper()} candidate confusion matrix')
    for (i,j),value in np.ndenumerate(cm): ax.text(j,i,str(value),ha='center',va='center')
    fig.tight_layout(); fig.savefig(run_dir/'confusion'/f'{a.model}.png',dpi=140); plt.close(fig)
    joblib.dump({"model": model, "features": feats, "run_tag": a.run_tag,
                 "data_dir": str(data_dir),
                 "top1_population": "observation manifest; no candidate counts wrong",
                 "mlp_iteration_selection": "nested held-trip validation" if a.model == "mlp" else None}, model_path)
    est = model[-1] if hasattr(model, "steps") else model
    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "model": a.model, "family": family, "seed": SEED,
        "parameters": {key: str(value) for key, value in est.get_params().items()},
        "features": feats, "data_dir": str(data_dir), "training_rows": len(tr),
        "training_observations_in_sample": len(sample_manifest),
        "outer_cv_folds": 5, "group": "trip_id",
        "cv_top1_population": "all strided training observations with ref_dist <= max_ref_dist in each held-out fold; no candidates count wrong",
        "student_test_top1_population": "all observations in student-test manifest; no candidates count wrong",
        "max_reference_distance_m": a.max_ref_dist, "training_stride": a.stride,
        "row_cap": a.max_rows or None, "selected_mlp_iterations_by_fold": cv["selected_mlp_iterations"],
        "final_mlp_iterations": final_params.get("max_iter"),
    }
    (config_dir / f"{a.model}.json").write_text(json.dumps(model_config, indent=2) + "\n")
    if hasattr(est, "loss_curve_"):  # epoch-wise history: MLP only
        json.dump({"loss_curve": list(map(float, est.loss_curve_)),
                   "selected_iterations_by_outer_fold": cv["selected_mlp_iterations"],
                   "final_iterations": final_params["max_iter"],
                   "stopping_method": "inner validation grouped by trip; log loss; no random row split"},
                  open(run_dir / "curves" / f"{a.model}.json", "w"))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(est.loss_curve_)
        ax.set(title="MLP training loss", xlabel="iteration", ylabel="loss")
        fig.tight_layout()
        fig.savefig(run_dir / "curves" / f"{a.model}.png", dpi=140)
        plt.close(fig)

    row = {
        "model": a.model, "family": family, "laptop": a.laptop,
        "params_json": json.dumps({k: str(v) for k, v in est.get_params().items()}),
        "seed": SEED, "n_train_rows": len(tr), "cv_folds": len(cv["top1"]),
        "cv_macro_f1_mean": np.mean(cv["macro_f1"]), "cv_macro_f1_std": np.std(cv["macro_f1"]),
        "cv_top1_mean": np.mean(cv["top1"]), "cv_top1_std": np.std(cv["top1"]),
        "cv_log_loss_mean": np.mean(cv["log_loss"]),
        "train_top1": train_top1, "test_top1": tm["top1"],
        **{f"test_{k}": tm[k] for k in ["accuracy", "precision", "recall", "f1", "macro_f1", "log_loss"]},
        "train_time_s": train_time, "infer_time_s_per_1k_rows": tm["infer_per_1k"],
        "test_candidate_miss_rate": tm["candidate_miss_rate"],
        "top1_population": "CV: all filtered/strided train-manifest observations in held-out trips; test: full student-test manifest; train: selected sample manifest; no candidates count wrong",
        "cv_top1_n_observations": json.dumps(cv["top1_n_observations"]),
        "mlp_selected_iterations": json.dumps(cv["selected_mlp_iterations"]),
        "machine": f"{platform.node()} | {platform.processor()}", "notes": a.notes,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"experiments_{a.laptop}.csv"
    pd.DataFrame([row])[COLUMNS].to_csv(out, mode="a", header=not out.exists(), index=False)
    pd.read_csv(out).query("model == 'mlp'").to_csv(run_dir/f"mlp_experiment_summary_{a.laptop}.csv",index=False)
    legacy=run_dir/'experiments_legacy.csv'
    legacy_row={"model":a.model,"family":family,"params_json":row['params_json'],"seed":SEED,
                "cv_macro_f1_mean":row['cv_macro_f1_mean'],"cv_macro_f1_std":row['cv_macro_f1_std'],
                "test_top1":tm['top1'],"test_macro_f1":tm['macro_f1'],"test_precision":tm['precision'],
                "test_recall":tm['recall'],"test_log_loss":tm['log_loss'],"route_continuity":"",
                "train_time_s":train_time,"infer_time_s_per_1k":tm['infer_per_1k'],
                "notes":f"{a.laptop}; candidate macro F1; Top-1 includes observations without candidates"}
    pd.DataFrame([legacy_row]).to_csv(legacy,mode='a',header=not legacy.exists(),index=False)
    print(f"\n{a.model}: CV top1 {row['cv_top1_mean']:.4f}±{row['cv_top1_std']:.4f} | "
          f"train top1 {train_top1:.4f} | test top1 {tm['top1']:.4f} | test macroF1 {tm['macro_f1']:.4f} | "
          f"logloss {tm['log_loss']:.4f} | train {train_time:.1f}s -> {out}")


if __name__ == "__main__":
    main()
