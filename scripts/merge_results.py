"""Merge results/experiments_L*.csv from all laptops into results/experiments.csv, ranked for model selection.

Owner: P2
Usage: python scripts/merge_results.py
Selection rule: highest student-test Top-1, ties broken by lower log loss.
"""
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"

files = sorted(RESULTS.glob("experiments_L*.csv"))
if not files:
    raise SystemExit("no results/experiments_L*.csv found - pull everyone's results first")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset=["model"], keep="last")  # latest run of each model wins
df = df.sort_values(["test_top1", "test_log_loss"], ascending=[False, True])
df.to_csv(RESULTS / "experiments.csv", index=False)

cols = ["model", "family", "laptop", "cv_macro_f1_mean", "cv_top1_mean", "cv_top1_std",
        "test_top1", "test_macro_f1", "test_log_loss", "train_time_s", "infer_time_s_per_1k_rows"]
print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSelected for Bayesian optimisation: {df.iloc[0]['model']}")
