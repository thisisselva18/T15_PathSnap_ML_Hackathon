"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model, then refit + export.

Owner: P2
Usage (repo root, after scripts/merge_results.py has written results/experiments.csv):
    python scripts/tune_bayes.py                      # tunes the top model in results/experiments.csv
    python scripts/tune_bayes.py --model lgbm --timeout 600 --n-trials 40

Method
  * Sampler: Optuna TPE (Tree-structured Parzen Estimator) - a sequential model-based (Bayesian)
    optimiser: it fits density models of good vs bad trials and proposes the next point accordingly.
  * Objective: mean observation-level Top-1 accuracy over 3-fold GROUP cross-validation on the 80
    train trips (3 folds = the shared 5-fold trip assignment mod 3, so whole trips stay together).
    Macro F1 and log loss of every trial are stored as user attributes for the report.
  * Student-test and evaluator data are NOT used to choose anything here; student-test is only used
    once at the end to compare tuned vs untuned.
  * Trials are stored in results/optuna.db (survives crashes; re-running continues the same study).
Outputs
  results/optuna_trials_<model>.csv, results/figures/optuna_history_<model>.png,
  results/figures/optuna_importance_<model>.png, results/tuned_vs_untuned.csv,
  results/experiments_tuned.csv, artifacts/best_params.json, artifacts/model.joblib (final model)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_models import (ARTIFACTS, COLUMNS, RESULTS, SEED, build_model, evaluate_on_test,  # noqa: E402
                          get_folds, load_train_test, row_metrics, top1)

MLP_SHAPES = {"64": (64,), "128": (128,), "64-32": (64, 32), "128-64": (128, 64)}

SPACES = {  # justified ranges: around common defaults, log scale for scale-type parameters
    "lgbm": lambda t: dict(
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
        max_depth=t.suggest_int("max_depth", 3, 12),
        min_child_samples=t.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        n_estimators=t.suggest_int("n_estimators", 100, 800, step=50)),
    "rf": lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 50, 300, step=50),
        max_depth=t.suggest_int("max_depth", 5, 30),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 50, log=True),
        max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.5])),
    "logreg": lambda t: dict(C=t.suggest_float("C", 1e-3, 1e2, log=True)),
    "mlp": lambda t: dict(
        hidden_layer_sizes=MLP_SHAPES[t.suggest_categorical("hidden", list(MLP_SHAPES))],
        alpha=t.suggest_float("alpha", 1e-6, 1e-2, log=True),
        learning_rate_init=t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)),
    "knn": lambda t: dict(
        n_neighbors=t.suggest_int("n_neighbors", 5, 100, log=True),
        weights=t.suggest_categorical("weights", ["uniform", "distance"])),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="default: top row of results/experiments.csv")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=600, help="seconds; stops early even if n-trials not reached")
    ap.add_argument("--max-rows", type=int, default=0, help="cap rows (use 150000 for mlp/knn)")
    a = ap.parse_args()

    exp_path = RESULTS / "experiments.csv"
    if a.model is None:
        a.model = pd.read_csv(exp_path).iloc[0]["model"]
    print(f"Tuning: {a.model}")

    tr, te, feats = load_train_test(max_rows=a.max_rows)
    fold3 = tr["trip_id"].map(get_folds(tr["trip_id"].unique())).to_numpy() % 3
    X, y = tr[feats].to_numpy(np.float32), tr["label"].to_numpy()

    def objective(trial):
        params = SPACES[a.model](trial)
        scores, f1s, lls = [], [], []
        for k in range(3):
            trn, val = fold3 != k, fold3 == k
            m = build_model(a.model, params)[1].fit(X[trn], y[trn])
            p = m.predict_proba(X[val])[:, 1]
            scores.append(top1(tr[val], p))
            rm = row_metrics(y[val], p)
            f1s.append(rm["macro_f1"])
            lls.append(rm["log_loss"])
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1_std", float(np.std(scores)))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        study_name=f"{a.model}_top1", direction="maximize", load_if_exists=True,
        storage=f"sqlite:///{(RESULTS / 'optuna.db').as_posix()}",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=a.n_trials, timeout=a.timeout)
    tune_time = time.perf_counter() - t0

    # ---- record the search
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(RESULTS / f"optuna_trials_{a.model}.csv", index=False)
    best = dict(study.best_params)
    params = SPACES[a.model](optuna.trial.FixedTrial(best))
    (ARTIFACTS / "best_params.json").write_text(json.dumps(
        {"model": a.model, "best_params": best, "best_cv3_top1": study.best_value,
         "n_trials": len(study.trials), "tuning_time_s": tune_time}, indent=1, default=str))
    try:
        import matplotlib
        matplotlib.use("Agg")
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        plot_optimization_history(study).figure.savefig(RESULTS / "figures" / f"optuna_history_{a.model}.png",
                                                        dpi=150, bbox_inches="tight")
        plot_param_importances(study).figure.savefig(RESULTS / "figures" / f"optuna_importance_{a.model}.png",
                                                     dpi=150, bbox_inches="tight")
    except Exception as e:  # plots are nice-to-have; never lose the run over them
        print("plot skipped:", e)

    # ---- refit tuned model on all train rows, evaluate once on student-test
    family, model = build_model(a.model, params)
    t1 = time.perf_counter()
    model.fit(X, y)
    train_time = time.perf_counter() - t1
    _, tm = evaluate_on_test(model, te, feats)

    tuned = {"model": f"{a.model}_tuned", "family": family, "laptop": "L1",
             "params_json": json.dumps(best, default=str), "seed": SEED, "n_train_rows": len(tr),
             "cv_folds": 3, "cv_macro_f1_mean": study.best_trial.user_attrs["cv_macro_f1"], "cv_macro_f1_std": np.nan,
             "cv_top1_mean": study.best_value, "cv_top1_std": study.best_trial.user_attrs["cv_top1_std"],
             "cv_log_loss_mean": study.best_trial.user_attrs["cv_log_loss"],
             "train_top1": top1(tr, model.predict_proba(X)[:, 1]), "test_top1": tm["top1"],
             **{f"test_{k}": tm[k] for k in ["accuracy", "precision", "recall", "f1", "macro_f1", "log_loss"]},
             "train_time_s": train_time, "infer_time_s_per_1k_rows": tm["infer_per_1k"], "machine": "",
             "notes": f"Optuna TPE, {len(study.trials)} trials, {tune_time:.0f}s"}
    out = RESULTS / "experiments_tuned.csv"
    pd.DataFrame([tuned])[COLUMNS].to_csv(out, mode="a", header=not out.exists(), index=False)

    cmp_cols = ["model", "cv_macro_f1_mean", "cv_top1_mean", "test_top1", "test_macro_f1",
                "test_log_loss", "infer_time_s_per_1k_rows"]
    rows = [pd.DataFrame([tuned])[cmp_cols]]
    if exp_path.exists():
        untuned = pd.read_csv(exp_path)
        rows.insert(0, untuned[untuned["model"] == a.model][cmp_cols])
    cmp = pd.concat(rows, ignore_index=True)
    cmp.to_csv(RESULTS / "tuned_vs_untuned.csv", index=False)

    joblib.dump({"model": model, "features": feats, "model_name": a.model, "params": best},
                ARTIFACTS / "model.joblib")
    print(f"\n{len(study.trials)} trials in {tune_time:.0f}s | best 3-fold CV top1 {study.best_value:.4f}")
    print("best params:", best)
    print(cmp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("saved artifacts/model.joblib")


if __name__ == "__main__":
    main()
