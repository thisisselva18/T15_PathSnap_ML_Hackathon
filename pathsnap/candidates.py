"""Candidate generation: the K nearest arcs within R metres of every GPS point.

Owner: P1
"""
import numpy as np
import pandas as pd
try:
    import shapely
except ImportError:
    shapely = None

from .geo import point_segment


def generate_candidates(txy, A, B, arc_ids, radius_m=50.0, k=8):
    """arc_ids: arcs to search (cropped network). A, B: start/end xy for ALL arcs (indexed by arc id).
    Returns DataFrame obs_idx, arc_id, dist, t_frac, t_raw, cand_rank (0 = nearest).

    Shapely's STRtree is the production spatial index. If unavailable, the fallback evaluates
    exact point-to-segment distances against every requested arc. It is intentionally exhaustive:
    midpoint-neighbor shortlists are not safe for long segments.
    """
    txy = np.asarray(txy, dtype=float)
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    arc_ids = np.asarray(arc_ids, dtype=np.int64)
    if txy.ndim != 2 or txy.shape[1] != 2 or A.ndim != 2 or A.shape[1] != 2 or B.shape != A.shape:
        raise ValueError("txy, A, and B must have shape (n, 2)")
    if not np.isfinite(txy).all() or not np.isfinite(A).all() or not np.isfinite(B).all():
        raise ValueError("candidate-search coordinates must be finite")
    if arc_ids.ndim != 1 or len(np.unique(arc_ids)) != len(arc_ids):
        raise ValueError("arc_ids must be a one-dimensional list of unique IDs")
    if np.any(arc_ids < 0) or np.any(arc_ids >= len(A)):
        raise ValueError("arc_ids must index A and B")
    if not np.isfinite(radius_m) or radius_m < 0 or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("radius_m must be non-negative and k must be a positive integer")
    columns = ["obs_idx", "arc_id", "dist", "t_frac", "t_raw", "cand_rank"]
    if len(txy) == 0 or len(arc_ids) == 0:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})

    if shapely is None:
        pi, arc, d, tf, tr = [], [], [], [], []
        for oi, point in enumerate(txy):
            ids = arc_ids
            dd, tt, rr = point_segment(point[0], point[1], A[ids, 0], A[ids, 1], B[ids, 0], B[ids, 1])
            keep = np.flatnonzero(dd <= radius_m)
            pi.extend([oi] * len(keep))
            arc.extend(ids[keep])
            d.extend(dd[keep])
            tf.extend(tt[keep])
            tr.extend(rr[keep])
    else:
        lines = shapely.linestrings(np.stack([A[arc_ids], B[arc_ids]], axis=1))
        tree = shapely.STRtree(lines)
        pi, li = tree.query(shapely.points(txy), predicate="dwithin", distance=radius_m)
        arc = arc_ids[li]
        d, tf, tr = point_segment(txy[pi, 0], txy[pi, 1], A[arc, 0], A[arc, 1], B[arc, 0], B[arc, 1])
    df = pd.DataFrame({"obs_idx": pi, "arc_id": arc, "dist": d, "t_frac": tf, "t_raw": tr})
    df = df.sort_values(["obs_idx", "dist", "arc_id"], kind="stable")
    df["cand_rank"] = df.groupby("obs_idx").cumcount()
    return df[df["cand_rank"] < k].reset_index(drop=True)
