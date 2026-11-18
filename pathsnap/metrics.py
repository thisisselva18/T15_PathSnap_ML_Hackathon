"""Metrics: row-level (acc, P, R, F1, macro F1, log loss), observation Top-1, candidate miss rate, route continuity, sequence-level.

Owner: P2
"""

def top1_accuracy(df, score_col="score"):
    """Argmax candidate per (trip_id, obs_idx) vs ref_arc_id; points with no correct candidate count as wrong."""
    raise NotImplementedError


def candidate_miss_rate(df):
    raise NotImplementedError


def route_continuity(pred_arcs, arcs):
    raise NotImplementedError
