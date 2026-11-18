"""Per-point reference labels from the .route arc sequence (TRAINING / EVALUATION ONLY).

.route gives the ordered list of arcs the vehicle drove, not a label per GPS point. We align the N GPS
points to the M route positions with a monotone dynamic programme (like DTW):

    D[i, j] = dist(point_i, route_arc_j) + min_k D[i-1, j-k],   k = 0..max_skip

i.e. the vehicle either stays on the same route arc or moves forward (skipping up to max_skip short
arcs in one step). After a time gap (dt > free_jump_dt or a new segment) any forward jump is allowed.
Backtracking the cheapest path gives ref_arc_id for every point. "Nearest route arc" is NOT used
because it breaks on loops and out-and-back trips.

Owner: P3
"""
import numpy as np

from .geo import point_segment


def align_points_to_route(txy, t, seg_id, route, A, B, max_skip=8, free_jump_dt=5.0,
                          return_diagnostics=False):
    """Align observations monotonically to route arc positions.

    The existing alignment rule is preserved. Inputs are checked explicitly, and optional
    diagnostics expose route positions and distance/jump summaries without changing labels.
    """
    txy = np.asarray(txy, dtype=float)
    t = np.asarray(t, dtype=float)
    seg_id = np.asarray(seg_id)
    route_raw = np.asarray(route)
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if txy.ndim != 2 or txy.shape[1] != 2 or t.ndim != 1 or seg_id.ndim != 1:
        raise ValueError("txy must be (N,2); t and seg_id must be 1-D")
    if len(txy) == 0 or len(route_raw) == 0:
        raise ValueError("track observations and route must both be non-empty")
    if len(t) != len(txy) or len(seg_id) != len(txy):
        raise ValueError("txy, t, and seg_id must have the same observation count")
    if not np.issubdtype(seg_id.dtype, np.integer) or np.any(seg_id < 0) or np.any(np.diff(seg_id) < 0):
        raise ValueError("seg_id must be non-negative, integer, and non-decreasing")
    if A.ndim != 2 or A.shape[1] != 2 or B.shape != A.shape:
        raise ValueError("A and B must have shape (n_arcs, 2)")
    if not all(np.isfinite(x).all() for x in (txy, t, A, B)):
        raise ValueError("track, time, and route geometry inputs must be finite")
    if np.any(np.diff(t) < 0):
        raise ValueError("timestamps must be non-decreasing after cleaning")
    if not np.issubdtype(route_raw.dtype, np.integer):
        if not np.issubdtype(route_raw.dtype, np.number) or not np.equal(route_raw, np.floor(route_raw)).all():
            raise ValueError("route arc IDs must be integers")
    route = route_raw.astype(np.int64)
    if np.any(route < 0) or np.any(route >= len(A)):
        raise ValueError("route contains arc IDs outside the arc geometry table")
    if not isinstance(max_skip, (int, np.integer)) or max_skip < 1 or max_skip > 255:
        raise ValueError("max_skip must be an integer between 1 and 255")
    if not np.isfinite(free_jump_dt) or free_jump_dt < 0:
        raise ValueError("free_jump_dt must be finite and non-negative")
    N, M = len(txy), len(route)
    ax, ay = A[route, 0], A[route, 1]
    bx, by = B[route, 0], B[route, 1]
    dist = lambda i: point_segment(txy[i, 0], txy[i, 1], ax, ay, bx, by)[0]
    INF = np.inf

    D = dist(0)
    D[min(3, M):] = INF                      # start on one of the first 3 route arcs
    move = np.zeros((N, M), dtype=np.uint8)  # how many arcs we advanced at step i
    jumps = {}                               # rows where any forward jump was allowed: i -> source j
    idx = np.arange(M)
    dt = np.diff(t, prepend=t[0])
    newseg = np.diff(seg_id, prepend=seg_id[0]) != 0

    for i in range(1, N):
        d = dist(i)
        if newseg[i] or dt[i] > free_jump_dt:
            pm = np.minimum.accumulate(D)
            src = np.maximum.accumulate(np.where(D == pm, idx, 0))
            jumps[i] = src
            D = d + pm
        else:
            best, bm = D.copy(), np.zeros(M, dtype=np.uint8)
            for k in range(1, max_skip + 1):
                if k >= M:
                    break
                sh = np.full(M, INF)
                sh[k:] = D[:-k]
                better = sh < best
                best[better] = sh[better]
                bm[better] = k
            move[i] = bm
            D = d + best

    j = int(np.argmin(D))                    # end anywhere (GPS may stop before the route ends)
    path = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        path[i] = j
        if i == 0:
            break
        j = int(jumps[i][j]) if i in jumps else j - int(move[i, j])

    ref_arc = route[path]
    ref_dist = point_segment(txy[:, 0], txy[:, 1], A[ref_arc, 0], A[ref_arc, 1], B[ref_arc, 0], B[ref_arc, 1])[0]
    if not return_diagnostics:
        return ref_arc, ref_dist
    position_steps = np.diff(path)
    diagnostics = {
        "route_positions": path,
        "position_monotone": bool(np.all(position_steps >= 0)),
        "n_position_jumps_over_max_skip": int(np.sum(position_steps > max_skip)),
        "n_ref_dist_gt_50m": int(np.sum(ref_dist > 50.0)),
        "n_ref_dist_gt_100m": int(np.sum(ref_dist > 100.0)),
        "ref_dist_p95_m": float(np.percentile(ref_dist, 95)),
        "ref_dist_max_m": float(np.max(ref_dist)),
    }
    return ref_arc, ref_dist, diagnostics
