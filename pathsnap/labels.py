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


def align_points_to_route(txy, t, seg_id, route, A, B, max_skip=8, free_jump_dt=5.0):
    """Return (ref_arc_id[N], ref_dist[N]). A, B: start/end xy of every arc (indexed by arc id)."""
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
    return ref_arc, ref_dist
