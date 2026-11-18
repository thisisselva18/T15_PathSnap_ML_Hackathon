"""Candidate generation: the K nearest arcs within R metres of every GPS point.

Owner: P1
"""
import numpy as np
import pandas as pd
import shapely

from .geo import point_segment


def generate_candidates(txy, A, B, arc_ids, radius_m=50.0, k=8):
    """arc_ids: arcs to search (cropped network). A, B: start/end xy for ALL arcs (indexed by arc id).
    Returns DataFrame obs_idx, arc_id, dist, t_frac, t_raw, cand_rank (0 = nearest)."""
    lines = shapely.linestrings(np.stack([A[arc_ids], B[arc_ids]], axis=1))
    tree = shapely.STRtree(lines)
    pi, li = tree.query(shapely.points(txy), predicate="dwithin", distance=radius_m)
    arc = arc_ids[li]
    d, tf, tr = point_segment(txy[pi, 0], txy[pi, 1], A[arc, 0], A[arc, 1], B[arc, 0], B[arc, 1])
    df = pd.DataFrame({"obs_idx": pi, "arc_id": arc, "dist": d, "t_frac": tf, "t_raw": tr})
    df = df.sort_values(["obs_idx", "dist", "arc_id"], kind="stable")
    df["cand_rank"] = df.groupby("obs_idx").cumcount()
    return df[df["cand_rank"] < k].reset_index(drop=True)
