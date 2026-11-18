"""Merge results/experiments_L*.csv from all laptops into results/experiments.csv, ranked for model selection.

Owner: P2
Usage:
    python scripts/merge_results.py
    python scripts/merge_results.py --run-dir results/v19 --extra results/mlp_baseline_verified/experiments_L2.csv
        (19-feature comparison: the four re-run models plus P4's 19-feature MLP)
Selection rule: highest student-test Top-1, ties broken by lower log loss.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default="results", help="folder holding experiments_L*.csv; output is written there")
ap.add_argument("--extra", nargs="*", default=[], help="additional experiments CSV files to include")
a = ap.parse_args()

run_dir = Path(a.run_dir) if Path(a.run_dir).is_absolute() else ROOT / a.run_dir
# laptop files only (L1, L2, ...), never experiments_legacy.csv
files = sorted(run_dir.glob("experiments_L[0-9]*.csv"))
files += [ROOT / f if not Path(f).is_absolute() else Path(f) for f in a.extra]
if not files:
    raise SystemExit(f"no experiments_L*.csv found in {run_dir} - pull everyone's results first")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset=["model"], keep="last")  # latest run of each model wins
df = df.sort_values(["test_top1", "test_log_loss"], ascending=[False, True])
df.to_csv(run_dir / "experiments.csv", index=False)

cols = ["model", "family", "laptop", "cv_macro_f1_mean", "cv_top1_mean", "cv_top1_std",
        "train_top1", "test_top1", "test_macro_f1", "test_log_loss", "train_time_s", "infer_time_s_per_1k_rows"]
print(df[[c for c in cols if c in df]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\n{len(df)} models from {len(files)} files -> {run_dir / 'experiments.csv'}")
print(f"Selected for Bayesian optimisation: {df.iloc[0]['model']}")
