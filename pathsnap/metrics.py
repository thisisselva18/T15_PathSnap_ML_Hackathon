"""Metrics: row-level (acc, P, R, F1, macro F1, log loss), observation Top-1, candidate miss rate, route continuity, sequence-level.

Owner: P2
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


def top1_accuracy(df, score_col="score", observations=None):
    """Observation Top-1; optional manifest includes observations with no candidates as wrong."""
    if df.empty:
        return float("nan") if observations is None or observations.empty else 0.0
    scored = df.sort_values(score_col, ascending=False).drop_duplicates(["trip_id", "obs_idx"])
    if observations is None:
        return float((scored["arc_id"].to_numpy() == scored["ref_arc_id"].to_numpy()).mean())
    predicted = {
        (row.trip_id, int(row.obs_idx)): int(row.arc_id == row.ref_arc_id)
        for row in scored.itertuples()
    }
    return float(
        np.mean(
            [
                predicted.get((row.trip_id, int(row.obs_idx)), 0)
                for row in observations.itertuples()
            ]
        )
    ) if len(observations) else float("nan")


def candidate_miss_rate(df, observations=None):
    """Fraction of observations whose candidate set omits its reference arc."""
    if df.empty:
        return float("nan") if observations is None or observations.empty else 1.0
    observed = {
        (trip_id, int(obs_idx)): set(group["arc_id"].astype(int))
        for (trip_id, obs_idx), group in df.groupby(["trip_id", "obs_idx"])
    }
    if observations is None:
        return float(
            np.mean(
                [
                    int(int(group["ref_arc_id"].iloc[0]) not in observed[(trip_id, int(obs_idx))])
                    for (trip_id, obs_idx), group in df.groupby(["trip_id", "obs_idx"])
                ]
            )
        )
    return float(
        np.mean(
            [
                int(
                    int(row.ref_arc_id)
                    not in observed.get((row.trip_id, int(row.obs_idx)), set())
                )
                for row in observations.itertuples()
            ]
        )
    ) if len(observations) else float("nan")


def route_continuity(pred_arcs, arcs):
    """Fraction of consecutive predicted arcs whose directed endpoints connect."""
    pred = np.asarray(pred_arcs, dtype=int)
    if len(pred) <= 1:
        return 1.0

    if hasattr(arcs, "columns") and {"u", "v"}.issubset(arcs.columns):
        u = arcs["u"].to_numpy()
        v = arcs["v"].to_numpy()
        valid = [
            0 <= first < len(u)
            and 0 <= second < len(u)
            and (
                first == second
                or (v[first], u[second]) in set(zip(u, v))
            )
            for first, second in zip(pred[:-1], pred[1:])
        ]
    else:
        endpoints = np.asarray(arcs)
        valid = [
            0 <= first < len(endpoints)
            and 0 <= second < len(endpoints)
            and (
                first == second
                or endpoints[first, 1] == endpoints[second, 0]
            )
            for first, second in zip(pred[:-1], pred[1:])
        ]
    return float(np.mean(valid))
