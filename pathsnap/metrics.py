"""Metrics: row-level (acc, P, R, F1, macro F1, log loss), observation Top-1, candidate miss rate, route continuity, sequence-level.

Owner: P2
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


def top1_accuracy(df, score_col="score"):
    """Argmax candidate per (trip_id, obs_idx) vs ref_arc_id; points with no correct candidate count as wrong."""
    d = df[["trip_id", "obs_idx", "arc_id", "ref_arc_id", score_col]].copy()
    best = d.loc[d.groupby(["trip_id", "obs_idx"], sort=False)[score_col].idxmax()]
    return float((best["arc_id"].to_numpy() == best["ref_arc_id"].to_numpy()).mean())


def candidate_miss_rate(df):
    """Fraction of GPS observations whose ref_arc_id is not in their candidate set."""
    obs = df[["trip_id", "obs_idx", "arc_id", "ref_arc_id"]].drop_duplicates()
    hit = (obs["arc_id"] == obs["ref_arc_id"]).groupby([obs["trip_id"], obs["obs_idx"]]).any()
    return float(1.0 - hit.mean())


def route_continuity(pred_arcs, arcs):
    """Fraction of consecutive transitions in pred_arcs that are directly connected in road network."""
    if len(pred_arcs) <= 1:
        return 1.0
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    out_map = set(zip(u.tolist(), v.tolist()))
    consecutive_valid = 0
    total = len(pred_arcs) - 1
    for a1, a2 in zip(pred_arcs[:-1], pred_arcs[1:]):
        if a1 == a2 or (v[a1], u[a2]) in out_map:
            consecutive_valid += 1
    return float(consecutive_valid / total) if total > 0 else 1.0

