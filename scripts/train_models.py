"""Train + evaluate ONE candidate classifier. Same script, same folds, same data on every laptop.

Owner: P2 (run by everyone)

Usage (from repo root):
    python scripts/train_models.py --model logreg --laptop L3
    python scripts/train_models.py --model lgbm   --laptop L1
    python scripts/train_models.py --model rf     --laptop L4
    python scripts/train_models.py --model mlp    --laptop L2 --max-rows 150000
    python scripts/train_models.py --model knn    --laptop L2 --max-rows 150000

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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
SEED = 42
KEY_COLS = ["trip_id", "seg_id", "obs_idx", "t", "cand_rank", "arc_id", "label", "ref_arc_id", "ref_dist", "t_raw"]
MODELS = ["logreg", "lgbm", "rf", "mlp", "knn"]

COLUMNS = ["model", "family", "laptop", "params_json", "seed", "n_train_rows", "cv_folds",
           "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "cv_top1_std", "cv_log_loss_mean",
           "train_top1", "test_top1", "test_accuracy", "test_precision", "test_recall", "test_f1",
           "test_macro_f1", "test_log_loss", "train_time_s", "infer_time_s_per_1k_rows", "machine", "notes"]


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
            learning_rate_init=p.get("learning_rate_init", 1e-3), early_stopping=True,
            max_iter=60, random_state=SEED))
    if name == "knn":
        return "Instance-based", make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=p.get("n_neighbors", 25), weights=p.get("weights", "distance"), n_jobs=-1))
    raise ValueError(f"unknown model {name}")


# ---------------------------------------------------------------- metrics
def top1(df, proba):
    """Observation-level Top-1: argmax candidate per (trip_id, obs_idx) == ref_arc_id."""
    d = df[["trip_id", "obs_idx", "arc_id", "ref_arc_id"]].copy()
    d["p"] = proba
    best = d.loc[d.groupby(["trip_id", "obs_idx"], sort=False)["p"].idxmax()]
    return float((best["arc_id"].to_numpy() == best["ref_arc_id"].to_numpy()).mean())


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


def load_train_test(stride=3, max_ref_dist=50.0, max_rows=0):
    """Shared by train_models.py and tune_bayes.py so both see exactly the same rows."""
    tr = pd.read_parquet(DATA / "candidates_train.parquet")
    te = pd.read_parquet(DATA / "candidates_student_test.parquet")
    tr["trip_id"], te["trip_id"] = tr["trip_id"].astype(str), te["trip_id"].astype(str)
    flist = json.loads((ARTIFACTS / "feature_list.json").read_text())
    feats = flist or [c for c in tr.columns if c not in KEY_COLS and pd.api.types.is_numeric_dtype(tr[c])]
    tr = tr[(tr["obs_idx"] % stride == 0) & (tr["ref_dist"] <= max_ref_dist)].reset_index(drop=True)
    if max_rows and len(tr) > max_rows:
        # sample whole GPS points (all their candidates) so Top-1 stays well defined
        obs = tr[["trip_id", "obs_idx"]].drop_duplicates()
        keep = obs.sample(frac=max_rows / len(tr), random_state=SEED)
        tr = tr.merge(keep, on=["trip_id", "obs_idx"]).reset_index(drop=True)
    return tr, te, feats


def evaluate_on_test(model, te, feats):
    Xte, yte = te[feats].to_numpy(np.float32), te["label"].to_numpy()
    t0 = time.perf_counter()
    p = model.predict_proba(Xte)[:, 1]
    infer = (time.perf_counter() - t0) / len(te) * 1000
    return p, {**row_metrics(yte, p), "top1": top1(te, p), "infer_per_1k": infer}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--laptop", required=True, help="L1/L2/L3/L4 - picks the results file")
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth GPS point for training")
    ap.add_argument("--max-rows", type=int, default=0, help="cap training rows (0 = no cap); use for mlp/knn")
    ap.add_argument("--max-ref-dist", type=float, default=50.0,
                    help="drop TRAINING points whose reference arc is further than this (unreliable labels)")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    tr, te, feats = load_train_test(a.stride, a.max_ref_dist, a.max_rows)
    print(f"{len(feats)} features: {feats}")
    print(f"train rows {len(tr):,} | test rows {len(te):,} | positives {tr['label'].mean():.3f}")

    folds = get_folds(tr["trip_id"].unique())
    fold_of = tr["trip_id"].map(folds).to_numpy()
    X, y = tr[feats].to_numpy(np.float32), tr["label"].to_numpy()

    # ---- group-aware CV (whole trips per fold)
    cv = {"macro_f1": [], "top1": [], "log_loss": []}
    for k in sorted(set(fold_of)):
        trn, val = fold_of != k, fold_of == k
        m = build_model(a.model)[1].fit(X[trn], y[trn])
        p = m.predict_proba(X[val])[:, 1]
        rm = row_metrics(y[val], p)
        cv["macro_f1"].append(rm["macro_f1"])
        cv["log_loss"].append(rm["log_loss"])
        cv["top1"].append(top1(tr[val], p))
        print(f"  fold {k}: macroF1 {rm['macro_f1']:.4f}  top1 {cv['top1'][-1]:.4f}  logloss {rm['log_loss']:.4f}")

    # ---- refit on all train, evaluate on student-test
    family, model = build_model(a.model)
    t0 = time.perf_counter()
    model.fit(X, y)
    train_time = time.perf_counter() - t0
    train_top1 = top1(tr, model.predict_proba(X)[:, 1])
    pte, tm = evaluate_on_test(model, te, feats)

    # ---- save everything
    for d in ["preds", "confusion", "curves"]:
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "models").mkdir(parents=True, exist_ok=True)
    te[["trip_id", "obs_idx", "arc_id", "ref_arc_id", "label"]].assign(score=pte).to_parquet(
        RESULTS / "preds" / f"{a.model}_student_test.parquet", index=False)
    cm = confusion_matrix(te["label"].to_numpy(), (pte >= 0.5).astype(int), labels=[0, 1])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        RESULTS / "confusion" / f"{a.model}.csv")
    joblib.dump({"model": model, "features": feats}, ARTIFACTS / "models" / f"{a.model}.joblib")
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "loss_curve_"):  # epoch-wise history: MLP only
        json.dump({"loss_curve": list(map(float, est.loss_curve_)),
                   "validation_scores": list(map(float, getattr(est, "validation_scores_", None) or []))},
                  open(RESULTS / "curves" / f"{a.model}.json", "w"))

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
        "machine": f"{platform.node()} | {platform.processor()}", "notes": a.notes,
    }
    out = RESULTS / f"experiments_{a.laptop}.csv"
    pd.DataFrame([row])[COLUMNS].to_csv(out, mode="a", header=not out.exists(), index=False)
    print(f"\n{a.model}: CV top1 {row['cv_top1_mean']:.4f}±{row['cv_top1_std']:.4f} | "
          f"train top1 {train_top1:.4f} | test top1 {tm['top1']:.4f} | test macroF1 {tm['macro_f1']:.4f} | "
          f"logloss {tm['log_loss']:.4f} | train {train_time:.1f}s -> {out.name}")


if __name__ == "__main__":
    main()
