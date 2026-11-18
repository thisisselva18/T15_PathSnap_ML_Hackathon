"""Bayesian optimisation only (TPE): trial 0 = defaults, trials 1–3 = initial design, every later trial proposed by the TPE model.

MLP-only, trip-grouped tuning for the corrected PathSnap training pipeline.
Student-test data is opened only after the study is complete, for final comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import classification_report, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline as sklearn_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_models import (ARTIFACTS, RESULTS, SEED, fit_mlp_epochs,
                          get_folds, load_train_test, positive_probability,
                          read_table, select_mlp_iterations, top1)
from evaluate_models import compute_metrics, plot_confusion, plot_roc

DEFAULTS = {
    "mlp": {"hidden": "64-32", "alpha": 1e-4, "learning_rate_init": 1e-3},
    # Historical baseline metadata only; KNN is not eligible for tuning.
    "knn": {"n_neighbors": 25, "weights": "distance"},
}
MLP_SHAPES = {"64": (64,), "128": (128,), "64-32": (64, 32), "128-64": (128, 64)}
BASELINE_METRICS = RESULTS / "mlp_baseline_verified" / "metrics" / "model_metrics.csv"
BASELINE_EXPERIMENT = RESULTS / "mlp_baseline_verified" / "experiments_L2.csv"


def make_pipeline(params, seed=SEED):
    return make_pipeline_fn(params, seed)


def make_pipeline_fn(params, seed):
    return sklearn_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=MLP_SHAPES[params["hidden"]], alpha=float(params["alpha"]),
        learning_rate_init=float(params["learning_rate_init"]), early_stopping=False,
        max_iter=1, random_state=seed))


def parameter_space(trial):
    return {
        "hidden": trial.suggest_categorical("hidden", list(MLP_SHAPES)),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    }


def cv_population_top1(df, probabilities, manifest):
    return top1(df, probabilities, manifest)


def main():
    parser = argparse.ArgumentParser(description="TPE tuning for the selected MLP only")
    parser.add_argument("--model", choices=["mlp"], default="mlp")
    parser.add_argument("--max-rows", type=int, default=150000)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--data-dir", default="data/processed/post_correction")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.max_rows <= 0 or args.n_trials < 1 or args.timeout <= 0:
        parser.error("--max-rows, --n-trials and --timeout must be positive")

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    smoke = args.max_rows < 150000 or args.n_trials < 8
    out = Path(args.output_dir) if args.output_dir else ROOT / "results" / ("mlp_tuning_smoke" if smoke else "mlp_tuning")
    if not out.is_absolute():
        out = ROOT / out
    model_path = ARTIFACTS / "models" / ("mlp_tuned_smoke.joblib" if smoke else "mlp_tuned.joblib")
    if (out / "optuna_mlp.db").exists() or (out / "best_params_mlp.json").exists() or model_path.exists():
        raise FileExistsError(f"Refusing to overwrite a prior tuning run: {out} or {model_path}")

    tr, _student_test, features, eligible_manifest, eligible_candidates, sample_manifest = load_train_test(
        max_rows=args.max_rows, data_dir=data_dir, return_manifest=True)
    if len(tr) != args.max_rows:
        print(f"Requested row cap {args.max_rows:,}; deterministic whole-observation sample has {len(tr):,} rows.")
    expected_features = json.loads((data_dir / "feature_list.json").read_text())
    if list(features) != expected_features:
        raise ValueError("training feature schema differs from the corrected dataset feature_list.json")
    fold_map = get_folds(eligible_manifest.trip_id.astype(str).unique())
    outer_fold = tr.trip_id.astype(str).map(fold_map).to_numpy() % 3
    all_candidate_fold = eligible_candidates.trip_id.astype(str).map(fold_map).to_numpy() % 3
    manifest_fold = eligible_manifest.trip_id.astype(str).map(fold_map).to_numpy() % 3
    x = tr[features].to_numpy(np.float32)
    y = tr.label.to_numpy(int)

    print(f"Optuna {optuna.__version__}; sampler args: seed={SEED}, n_startup_trials=4, multivariate=True")
    print(f"Corrected data: {data_dir}; selected rows={len(tr):,}; selected observations={len(sample_manifest):,}; "
          f"CV population=all eligible training observations by held-out trip; student-test labels not loaded yet.")

    out.mkdir(parents=True, exist_ok=True)
    database = out / "optuna_mlp.db"
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4, multivariate=True)
    study = optuna.create_study(study_name="mlp_top1", direction="maximize",
                                storage=f"sqlite:///{database.as_posix()}", sampler=sampler)
    study.enqueue_trial(DEFAULTS["mlp"], user_attrs={"trial_role": "untuned_default_baseline"})

    def objective(trial):
        params = parameter_space(trial)
        if trial.number == 0 and params != DEFAULTS["mlp"]:
            raise RuntimeError(f"trial 0 is not baseline defaults: {params}")
        trial.set_user_attr("trial_role", "untuned_default_baseline" if trial.number == 0 else
                            ("tpe_startup_design" if trial.number <= 3 else "tpe_proposal"))
        trial.set_user_attr("n_training_rows", int(len(tr)))
        trial.set_user_attr("cv_folds", 3)
        started = time.perf_counter()
        top_scores, f1_scores, losses, chosen_iterations = [], [], [], []
        try:
            for fold in range(3):
                fit_mask = outer_fold != fold
                fit_groups = tr.loc[fit_mask, "trip_id"].astype(str).to_numpy()
                chosen, _ = select_mlp_iterations(
                    x[fit_mask], y[fit_mask], fit_groups, seed=SEED + fold,
                    max_iter=60, patience=8, hidden_layer_sizes=MLP_SHAPES[params["hidden"]],
                    alpha=float(params["alpha"]), learning_rate_init=float(params["learning_rate_init"]))
                model = make_pipeline(params, seed=SEED + fold)
                fit_mlp_epochs(model, x[fit_mask], y[fit_mask], chosen)
                val_mask = all_candidate_fold == fold
                val_df = eligible_candidates.loc[val_mask]
                val_manifest = eligible_manifest.loc[manifest_fold == fold]
                probabilities = positive_probability(model, val_df[features].to_numpy(np.float32))
                top_scores.append(cv_population_top1(val_df, probabilities, val_manifest))
                binary_preds = (probabilities >= .5).astype(int)
                from sklearn.metrics import f1_score
                f1_scores.append(float(f1_score(val_df.label.to_numpy(), binary_preds,
                                                average="macro", zero_division=0)))
                losses.append(float(log_loss(val_df.label.to_numpy(),
                                             np.column_stack([1-probabilities, probabilities]), labels=[0, 1])))
                chosen_iterations.append(int(chosen))
            trial.set_user_attr("cv_macro_f1_mean", float(np.mean(f1_scores)))
            trial.set_user_attr("cv_log_loss_mean", float(np.mean(losses)))
            trial.set_user_attr("cv_top1_std", float(np.std(top_scores)))
            trial.set_user_attr("cv_top1_by_fold", [float(v) for v in top_scores])
            trial.set_user_attr("selected_iterations_by_fold", chosen_iterations)
            return float(np.mean(top_scores))
        except Exception as exc:
            trial.set_user_attr("failure", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            trial.set_user_attr("duration_s", float(time.perf_counter() - started))

    started = time.perf_counter()
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, catch=(Exception,))
    tune_seconds = time.perf_counter() - started
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No tuning trial completed; inspect the Optuna trial log.")
    if completed[0].number != 0 or completed[0].params != DEFAULTS["mlp"]:
        raise RuntimeError("The first completed trial was not the enqueued untuned MLP defaults.")

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state", "duration"))
    trials_df.to_csv(out / "optuna_trials_mlp.csv", index=False)
    best = study.best_trial
    best_params = dict(best.params)
    (out / "best_params_mlp.json").write_text(json.dumps({
        "model": "mlp", "best_params": best_params, "best_cv3_top1": best.value,
        "n_trials_attempted": len(study.trials), "n_trials_complete": len(completed),
        "n_trials_failed": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
        "tuning_time_s": tune_seconds, "sampler": {"name": "TPESampler", "seed": SEED,
        "n_startup_trials": 4, "multivariate": True}, "data_dir": str(data_dir),
        "n_training_rows": len(tr), "feature_schema": features,
        "cv_top1_population": "All eligible training observation manifest in each held-out trip fold; observations without candidates count incorrect.",
        "iteration_selection": "Grouped held-trip inner validation, minimum log loss; final epoch count is median across best trial's three outer folds.",
        "student_test_used_for_optimization": False,
    }, indent=2) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        ax = plot_optimization_history(study)
        ax.figure.savefig(out / "optuna_history_mlp.png", dpi=240, bbox_inches="tight")
        plt.close(ax.figure)
        if len(completed) >= 2 and len({tuple(sorted(t.params.items())) for t in completed}) > 1:
            ax = plot_param_importances(study)
            ax.figure.savefig(out / "optuna_importance_mlp.png", dpi=240, bbox_inches="tight")
            plt.close(ax.figure)
    except Exception as exc:
        (out / "plot_generation_note.txt").write_text(f"Optuna plot generation unavailable: {type(exc).__name__}: {exc}\n")

    # Refit with median grouped-validation iteration count for the best parameters.
    best_iterations = best.user_attrs.get("selected_iterations_by_fold", [])
    if not best_iterations:
        raise RuntimeError("Best trial did not save grouped iteration selections.")
    final_epochs = max(1, int(np.median(best_iterations)))
    model = make_pipeline(best_params)
    fit_start = time.perf_counter()
    fit_mlp_epochs(model, x, y, final_epochs)
    train_seconds = time.perf_counter() - fit_start
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"model": model, "features": list(features), "model_name": "mlp",
                "params": best_params, "run_tag": "mlp_tuning_smoke" if smoke else "mlp_tuned",
                "training_rows": len(tr), "selected_iterations": final_epochs,
                "data_dir": str(data_dir), "sampler": "TPE", "student_test_used_for_optimization": False}
    joblib.dump(artifact, model_path)

    # Only now load and score the student-test partition, exactly once for selection.
    student_test = read_table("candidates_student_test", data_dir)
    student_test["trip_id"] = student_test.trip_id.astype(str)
    test_manifest = read_table("observations_student_test", data_dir)
    test_manifest["trip_id"] = test_manifest.trip_id.astype(str)
    test_features = student_test[features].to_numpy(np.float32)
    test_y = student_test.label.to_numpy(int)
    infer_start = time.perf_counter()
    raw = model.predict_proba(test_features)
    inference_seconds = time.perf_counter() - infer_start
    estimator = model[-1]
    classes = np.asarray(estimator.classes_)
    pred = classes[np.argmax(raw, axis=1)]
    metrics = compute_metrics(test_y, pred, raw, classes)
    positive = positive_probability(model, test_features)
    metrics["top1_accuracy"] = top1(student_test, positive, test_manifest)
    metrics["n_train_samples"] = int(len(tr))
    metrics["n_test_candidate_rows"] = int(len(student_test))
    metrics["n_test_observations"] = int(len(test_manifest))
    metrics["inference_time_s"] = float(inference_seconds)
    metrics["inference_time_s_per_1000_rows"] = float(inference_seconds * 1000 / len(student_test))
    metrics["training_time_s"] = float(train_seconds)
    metrics["cv_top1_mean"] = float(best.value)
    metrics["cv_top1_std"] = float(best.user_attrs["cv_top1_std"])
    candidate_sets = {(str(tid), int(obs)): set(g.arc_id.astype(int))
                      for (tid, obs), g in student_test.groupby(["trip_id", "obs_idx"], sort=False)}
    candidate_misses = [
        int(r.ref_arc_id) not in candidate_sets.get((str(r.trip_id), int(r.obs_idx)), set())
        for r in test_manifest.itertuples()
    ]
    metrics["candidate_miss_rate"] = float(np.mean(candidate_misses))
    metrics["candidate_accuracy"] = metrics["accuracy"]
    metrics["top1_population"] = "Full student-test observation manifest; observations without candidates count incorrect."
    metrics["candidate_accuracy_population"] = "All candidate rows from the student-test partition."
    (out / "model_metrics_mlp.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame([{"model": "mlp_tuned", **metrics}]).to_csv(out / "model_metrics_mlp.csv", index=False)
    pd.DataFrame([{
        "n_student_test_observations": len(test_manifest),
        "candidate_miss_rate": metrics["candidate_miss_rate"],
        "candidate_coverage": 1.0 - metrics["candidate_miss_rate"],
        "observations_without_candidates": sum(
            (str(r.trip_id), int(r.obs_idx)) not in candidate_sets for r in test_manifest.itertuples()),
        "mean_candidates_per_observation": len(student_test) / max(len(test_manifest), 1),
    }]).to_csv(out / "candidate_generation_metrics_mlp.csv", index=False)
    pd.DataFrame({"trip_id": student_test.trip_id, "obs_idx": student_test.obs_idx,
                  "arc_id": student_test.arc_id, "ref_arc_id": student_test.ref_arc_id,
                  "label": test_y, "score": positive}).to_csv(out / "predictions_mlp_student_test.csv", index=False)
    report = classification_report(test_y, pred, labels=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.rename_axis("class_or_average").to_csv(out / "classification_report_mlp.csv")
    cm = plot_confusion("mlp_tuned", test_y, pred, classes, out / "confusion_matrix_mlp.png")
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]).to_csv(out / "confusion_matrix_mlp.csv")
    plot_roc("mlp_tuned", test_y, raw, classes, out / "roc_curve_mlp.png")

    # Baseline is the corrected 150k MLP evaluation on this same student-test partition.
    base = pd.read_csv(BASELINE_METRICS).iloc[0].to_dict()
    base_exp = pd.read_csv(BASELINE_EXPERIMENT).iloc[-1].to_dict()
    baseline_row = {"model": "mlp_untuned", "top1_accuracy": float(base["top1_accuracy"]),
                    "candidate_accuracy": float(base["accuracy"]),
                    "macro_precision": float(base["macro_precision"]), "macro_recall": float(base["macro_recall"]),
                    "macro_f1": float(base["macro_f1"]), "weighted_f1": float(base["weighted_f1"]),
                    "roc_auc": float(base["roc_auc"]), "log_loss": float(base["log_loss"]),
                    "inference_time_s": float(base["inference_time_s"]),
                    "training_time_s": float(base_exp["train_time_s"]),
                    "cv_top1_mean": float(study.trials[0].value),
                    "cv_top1_std": float(study.trials[0].user_attrs["cv_top1_std"]),
                    "n_train_samples": int(base["n_train_samples"]), "n_test_observations": int(base["n_test_observations"])}
    tuned_row = {"model": "mlp_tuned", "top1_accuracy": metrics["top1_accuracy"],
                 "candidate_accuracy": metrics["accuracy"], "macro_precision": metrics["macro_precision"],
                 "macro_recall": metrics["macro_recall"], "macro_f1": metrics["macro_f1"],
                 "weighted_f1": metrics["weighted_f1"], "roc_auc": metrics["roc_auc"],
                 "log_loss": metrics["log_loss"], "inference_time_s": inference_seconds,
                 "training_time_s": train_seconds, "cv_top1_mean": best.value,
                 "cv_top1_std": best.user_attrs["cv_top1_std"], "n_train_samples": len(tr),
                 "n_test_observations": len(test_manifest)}
    comparison = pd.DataFrame([baseline_row, tuned_row])
    comparison.to_csv(out / "tuned_vs_untuned_mlp.csv", index=False)
    pd.DataFrame([{"model": "mlp_tuned", "best_params": json.dumps(best_params),
                   "cv_top1_mean": best.value, "cv_top1_std": best.user_attrs["cv_top1_std"],
                   "test_top1": metrics["top1_accuracy"], "test_accuracy": metrics["accuracy"],
                   "test_macro_f1": metrics["macro_f1"], "test_log_loss": metrics["log_loss"],
                   "n_train_rows": len(tr), "cv_folds": 3, "train_time_s": train_seconds,
                   "infer_time_s_per_1k_rows": metrics["inference_time_s_per_1000_rows"],
                   "selected_iterations": final_epochs, "n_trials_attempted": len(study.trials),
                   "n_trials_complete": len(completed), "student_test_used_for_optimization": False}]
                  ).to_csv(out / "experiments_tuned_mlp.csv", index=False)
    (out / "selection.json").write_text(json.dumps({
        "selected_model": "mlp_tuned" if tuned_row["top1_accuracy"] >= baseline_row["top1_accuracy"] else "mlp_untuned",
        "rule": "select tuned MLP iff full student-test observation Top-1 is at least untuned MLP Top-1",
        "tuned_top1": tuned_row["top1_accuracy"], "untuned_top1": baseline_row["top1_accuracy"],
        "uses_evaluator_test": False,
    }, indent=2) + "\n")
    print(f"Trials: attempted={len(study.trials)}, completed={len(completed)}, "
          f"failed={len(study.trials)-len(completed)}, elapsed={tune_seconds:.1f}s")
    print(f"Best params: {best_params}; CV Top-1={best.value:.6f}; iterations={best_iterations}")
    print(comparison.to_string(index=False))
    print(f"Saved tuned pipeline: {model_path}; results: {out}")


if __name__ == "__main__":
    main()
