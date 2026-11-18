"""Feature engineering per (GPS point, candidate arc). Uses GPS + road network ONLY, never .route.

Owner: P1
"""
import numpy as np

from .geo import point_segment

FEATURES = [
    "dist", "cand_rank", "dist_minus_min", "dist_ratio", "n_cands",          # geometry vs other candidates
    "t_frac", "past_end", "arc_len",                                         # position along the arc
    "head_diff", "head_cos", "head_valid", "speed", "dt_prev",               # direction / motion
    "ctx_mean_dist", "dist_prev",                                             # causal temporal context
    "out_deg_v", "in_deg_u", "has_twin", "conn_prev_top3",                   # topology
]


def add_features(c, txy, t, seg_id, A, B, u, v, out_deg, in_deg, twin):
    """c: candidate DataFrame from generate_candidates (modified in place and returned)."""
    N = len(txy)
    i = c["obs_idx"].to_numpy()
    a = c["arc_id"].to_numpy()
    ax, ay, bx, by = A[a, 0], A[a, 1], B[a, 0], B[a, 1]

    # --- geometry relative to the other candidates of the same point
    g = c.groupby("obs_idx")["dist"]
    dmin = g.transform("min").to_numpy()
    c["dist_minus_min"] = c["dist"] - dmin
    c["dist_ratio"] = c["dist"] / (dmin + 1.0)
    c["n_cands"] = g.transform("size")
    c["past_end"] = ((c["t_raw"] < 0) | (c["t_raw"] > 1)).astype(np.int8)
    c["arc_len"] = np.hypot(bx - ax, by - ay)

    # --- neighbours inside the same segment (clipped at segment edges)
    idx = np.arange(N)
    prv = np.where((idx > 0) & (seg_id[np.maximum(idx - 1, 0)] == seg_id), np.maximum(idx - 1, 0), idx)
    vec = txy - txy[prv]
    move = np.hypot(vec[:, 0], vec[:, 1])
    head = np.arctan2(vec[:, 1], vec[:, 0])
    span = t - t[prv]
    speed = np.where(span > 0, move / np.where(span > 0, span, 1), 0.0)
    head_valid = (move >= 2.0).astype(np.int8)         # heading is noise when (nearly) stopped
    bearing = np.arctan2(by - ay, bx - ax)
    diff = np.abs((head[i] - bearing + np.pi) % (2 * np.pi) - np.pi)
    diff = np.where(head_valid[i] == 1, diff, np.pi / 2)
    c["head_diff"] = diff
    c["head_cos"] = np.cos(diff)
    c["head_valid"] = head_valid[i]
    c["speed"] = speed[i]
    c["dt_prev"] = np.minimum(np.diff(t, prepend=t[0]), 600.0)[i]

    # --- causal context: current and up to two previous points within this segment
    prev1 = prv
    prev2 = np.where((prev1 > 0) & (seg_id[np.maximum(prev1 - 1, 0)] == seg_id), np.maximum(prev1 - 1, 0), prev1)
    dprev = point_segment(txy[prev1[i], 0], txy[prev1[i], 1], ax, ay, bx, by)[0]
    dprev2 = point_segment(txy[prev2[i], 0], txy[prev2[i], 1], ax, ay, bx, by)[0]
    c["dist_prev"] = dprev
    c["ctx_mean_dist"] = (dprev2 + dprev + c["dist"].to_numpy()) / 3.0

    # --- topology
    c["out_deg_v"] = out_deg[v[a]]
    c["in_deg_u"] = in_deg[u[a]]
    c["has_twin"] = twin[a].astype(np.int8)

    # connected to (or same as) one of the previous point's 3 nearest candidates?
    BIG = np.int64(10_000_000)
    top3 = c[c["cand_rank"] < 3]
    po = top3["obs_idx"].to_numpy().astype(np.int64) + 1
    pa = top3["arc_id"].to_numpy().astype(np.int64)
    same = np.isin(i.astype(np.int64) * BIG + a, po * BIG + pa)
    follows = np.isin(i.astype(np.int64) * BIG + u[a], po * BIG + v[pa])
    first_of_seg = np.diff(seg_id, prepend=-1)[i] != 0
    c["conn_prev_top3"] = ((same | follows) & ~first_of_seg).astype(np.int8)
    return c
