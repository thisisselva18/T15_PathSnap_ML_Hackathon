"""HMM map matching: emissions from classifier probabilities, transitions from network vs great-circle distance,
Viterbi decoding with restarts at GPS segments and at unreachable steps.

Hidden states   the candidate arcs of each GPS point (up to K=8 within 50 m, from candidates.py).
Emission        log E_i(b) = log max(p_i(b), p_floor) - w * dist_i(b)^2 / (2 sigma^2)
                p = the classifier's probability that b is the right arc; dist = GPS-to-arc distance (m).
Transition      d_route(a->b) = rest of a + shortest path(v[a] -> u[b]) + offset on b
                (same arc: offset_b - offset_a, signed, so going backwards costs; U-turn onto a's
                reverse twin: + uturn_penalty_m)
                d_gc = straight-line distance between the two GPS points
                log T(a,b) = -|d_route - d_gc| / beta; no path within max_route_m -> -inf.
Decoding        delta_i(b) = log E_i(b) + max_a [delta_{i-1}(a) + log T(a,b)], with back-pointers.
                Restart at every new GPS segment and whenever every transition is -inf (counted as a break).
                Points without candidates are skipped (they keep -1) and do not cut the chain.

Interface used by pathsnap/pipeline.py and scripts/eval_hmm.py:
    load_hmm_cfg() -> dict
    RoadGraph(u, v, A, B, arc_ids, max_route_m)
    decode(cands, proba, txy, seg_id, graph, cfg) -> (arc per GPS point, -1 = none; number of breaks)
    route_from_points(point_arcs, seg_id, graph) -> list of continuous arc-id lists
    continuity(point_arcs, seg_id, graph, txy) -> float in [0, 1] (None if no pairs)

Owner: P3
"""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

CFG_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "hmm_config.json"
DEFAULTS = {
    "sigma_m": 10.0,          # GPS noise scale in the distance term of the emission
    "beta_m": 10.0,           # tolerance for |route length - straight-line length| in the transition
    "dist_weight": 0.0,       # w: 0 = emission is the classifier probability alone (it already sees dist)
    "uturn_penalty_m": 50.0,  # extra metres for turning onto the reverse twin of the current arc
    "max_route_m": 2000.0,    # Dijkstra search limit; longer moves between two points count as unreachable
    "p_floor": 1e-6,          # probability floor so one confident-but-wrong zero cannot veto a path
    "min_move_m": 0.0,        # fixes closer than this to the last kept fix inherit its arc (0 = keep every fix)
}
TUNABLE = ("sigma_m", "beta_m", "dist_weight", "uturn_penalty_m")
_DIJKSTRA_CHUNK = 64


def load_hmm_cfg(path=CFG_PATH):
    """Defaults, overridden by any non-null value in artifacts/hmm_config.json."""
    cfg = dict(DEFAULTS)
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text())
        cfg.update({k: float(saved[k]) for k in DEFAULTS if saved.get(k) is not None})
    return cfg


class RoadGraph:
    """Directed graph over the cropped arcs with cached, distance-limited shortest paths.

    u, v: from/to node id of EVERY arc; A, B: their start/end xy (m); arc_ids: the arcs to include (ctx["keep"]).
    """

    def __init__(self, u, v, A, B, arc_ids, max_route_m=DEFAULTS["max_route_m"]):
        self.u = np.asarray(u, dtype=np.int64)
        self.v = np.asarray(v, dtype=np.int64)
        A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
        self.length = np.hypot(*(B - A).T)                       # every arc, indexed by arc id
        self.max_route_m = float(max_route_m)
        arcs = np.asarray(arc_ids, dtype=np.int64)
        self.nodes = np.unique(np.concatenate([self.u[arcs], self.v[arcs]]))   # sorted global node ids
        su, sv = self.local(self.u[arcs]), self.local(self.v[arcs])
        w = np.maximum(self.length[arcs], 0.01)                  # zero-length arcs must stay edges

        # parallel arcs between the same node pair: keep the shortest (csr_matrix would ADD duplicates)
        order = np.lexsort((w, sv, su))
        su, sv, w, arcs = su[order], sv[order], w[order], arcs[order]
        first = np.r_[True, (su[1:] != su[:-1]) | (sv[1:] != sv[:-1])]
        su, sv, w, arcs = su[first], sv[first], w[first], arcs[first]
        n = len(self.nodes)
        self.G = csr_matrix((w, (su, sv)), shape=(n, n))
        self.edge_arc = dict(zip(zip(su.tolist(), sv.tolist()), arcs.tolist()))   # (local u, local v) -> arc id
        self._sp = {}                                            # local src -> (reachable local ids, dist, pred)

    # ---------------------------------------------------------------- node ids
    def local(self, global_nodes):
        """Global node id(s) -> local index, -1 if the node is not in the cropped graph."""
        g = np.asarray(global_nodes, dtype=np.int64)
        if not len(self.nodes):
            return np.full(g.shape, -1, dtype=np.int64)
        idx = np.clip(np.searchsorted(self.nodes, g), 0, len(self.nodes) - 1)
        return np.where(self.nodes[idx] == g, idx, -1)

    # ---------------------------------------------------------------- shortest paths
    def _run(self, srcs):
        """Dijkstra from the given local sources (chunked), caching only reachable entries."""
        srcs = [s for s in dict.fromkeys(int(s) for s in srcs) if s >= 0 and s not in self._sp]
        for i in range(0, len(srcs), _DIJKSTRA_CHUNK):
            chunk = srcs[i:i + _DIJKSTRA_CHUNK]
            dist, pred = dijkstra(self.G, directed=True, indices=chunk, limit=self.max_route_m,
                                  return_predecessors=True)
            for k, s in enumerate(chunk):
                reach = np.flatnonzero(np.isfinite(dist[k]))
                self._sp[s] = (reach, dist[k][reach], pred[k][reach])

    def prefetch(self, global_nodes):
        self._run(self.local(np.unique(global_nodes)))

    def sp(self, src_node, dst_nodes):
        """Network distance (m) from global node src_node to each global node in dst_nodes; inf if unreachable."""
        dst = np.atleast_1d(np.asarray(dst_nodes, dtype=np.int64))
        out = np.full(len(dst), np.inf)
        s = int(self.local(src_node))
        if s < 0:
            return out
        if s not in self._sp:
            self._run([s])
        reach, dist, _ = self._sp[s]
        d = self.local(dst)
        pos = np.clip(np.searchsorted(reach, d), 0, max(len(reach) - 1, 0))
        hit = (d >= 0) & (len(reach) > 0)
        if len(reach):
            hit &= reach[pos] == d
        out[hit] = dist[pos[hit]]
        return out

    def path_arcs(self, src_node, dst_node):
        """Arc ids of the shortest path between two global nodes ([] if same node, None if unreachable)."""
        s, d = int(self.local(src_node)), int(self.local(dst_node))
        if s < 0 or d < 0:
            return None
        if s == d:
            return []
        if s not in self._sp:
            self._run([s])
        reach, _, pred = self._sp[s]
        pos = np.searchsorted(reach, d)
        if pos >= len(reach) or reach[pos] != d:
            return None
        arcs, cur = [], d
        while cur != s:
            p = int(pred[np.searchsorted(reach, cur)])
            arcs.append(self.edge_arc[(p, cur)])
            cur = p
        return arcs[::-1]


# ==================================================================== lattice (parameter-free part)
def build_lattice(cands, proba, txy, seg_id, graph, min_move_m=0.0):
    """Everything Viterbi needs that does not depend on the tunable parameters, so tuning only re-runs Viterbi.

    Returns a list of steps, one per kept GPS point that has candidates, in time order:
        obs, arcs, dist, proba, new_chain (first step of a GPS segment),
        D (Ka x Kb route distance from the previous step, inf = unreachable), U (U-turn mask), dgc,
        followers (later fixes within min_move_m of this one: a stopped vehicle's jitter; they take its arc).
    """
    obs = cands["obs_idx"].to_numpy(np.int64)
    arc = cands["arc_id"].to_numpy(np.int64)
    dist = cands["dist"].to_numpy(float)
    off = cands["t_frac"].to_numpy(float) * graph.length[arc]      # metres from the arc's start
    p = np.asarray(proba, dtype=float)
    txy = np.asarray(txy, dtype=float)
    seg_id = np.asarray(seg_id)
    if len(obs) == 0:
        return []
    order = np.argsort(obs, kind="stable")
    obs, arc, dist, off, p = obs[order], arc[order], dist[order], off[order], p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    ends = np.r_[starts[1:], len(obs)]

    graph.prefetch(graph.v[np.unique(arc)])                        # all Dijkstra sources in a few batched calls
    u, v, length = graph.u, graph.v, graph.length
    steps, prev = [], None
    for s, e in zip(starts, ends):
        i = int(obs[s])
        st = {"obs": i, "arcs": arc[s:e], "dist": dist[s:e], "proba": p[s:e], "off": off[s:e], "followers": []}
        if prev is not None and seg_id[i] == seg_id[prev["obs"]] and                 np.hypot(*(txy[i] - txy[prev["obs"]])) < min_move_m:
            prev["followers"].append(i)
            continue
        if prev is None or seg_id[i] != seg_id[prev["obs"]]:
            st["new_chain"] = True
        else:
            st["new_chain"] = False
            a_arcs, a_off, b_arcs, b_off = prev["arcs"], prev["off"], st["arcs"], st["off"]
            D = np.empty((len(a_arcs), len(b_arcs)))
            U = np.zeros(D.shape, dtype=bool)
            for k, a in enumerate(a_arcs):
                D[k] = (length[a] - a_off[k]) + graph.sp(v[a], u[b_arcs]) + b_off
                same = b_arcs == a
                D[k, same] = b_off[same] - a_off[k]      # signed: moving backwards along an arc costs 2x
                U[k] = (u[b_arcs] == v[a]) & (v[b_arcs] == u[a]) & ~same    # reverse twin of a
            st["D"], st["U"] = D, U
            st["dgc"] = float(np.hypot(*(txy[i] - txy[prev["obs"]])))
        steps.append(st)
        prev = st
    return steps


def viterbi(steps, n_points, cfg, use_classifier=True):
    """Decode a lattice. use_classifier=False gives the classic geometry-only HMM (emission from distance alone)."""
    sigma, beta = float(cfg["sigma_m"]), float(cfg["beta_m"])
    w, pen, floor = float(cfg["dist_weight"]), float(cfg["uturn_penalty_m"]), float(cfg["p_floor"])
    if not use_classifier:
        w = 1.0
    out = np.full(n_points, -1, dtype=np.int64)
    breaks = 0
    chain, deltas, backs = [], [], []

    def flush():
        if not chain:
            return
        k = int(np.argmax(deltas[-1]))
        for j in range(len(chain) - 1, -1, -1):
            out[chain[j]["obs"]] = chain[j]["arcs"][k]
            out[chain[j]["followers"]] = chain[j]["arcs"][k]
            if j:
                k = int(backs[j][k])
        chain.clear(), deltas.clear(), backs.clear()

    for st in steps:
        logE = -w * st["dist"] ** 2 / (2.0 * sigma ** 2)
        if use_classifier:
            logE = logE + np.log(np.maximum(st["proba"], floor))
        if not st["new_chain"]:
            D = st["D"] + pen * st["U"]
            with np.errstate(invalid="ignore"):
                logT = np.where(np.isfinite(D), -np.abs(D - st["dgc"]) / beta, -np.inf)
            score = deltas[-1][:, None] + logT
            best = np.argmax(score, axis=0)
            bscore = score[best, np.arange(score.shape[1])]
            if np.isfinite(bscore).any():
                chain.append(st), deltas.append(bscore + logE), backs.append(best)
                continue
            breaks += 1                                             # nothing reachable: restart here
        flush()
        chain.append(st), deltas.append(logE), backs.append(None)
    flush()
    return out, breaks


def decode(cands, proba, txy, seg_id, graph, cfg):
    """(arc per GPS point with -1 where a point has no candidates, number of HMM breaks)."""
    steps = build_lattice(cands, proba, txy, seg_id, graph, cfg.get("min_move_m", 0.0))
    return viterbi(steps, len(txy), cfg)


# ==================================================================== route output and continuity
def route_from_points(point_arcs, seg_id, graph):
    """Continuous arc-id lists: per GPS segment, the distinct matched arcs with shortest-path gaps filled in.
    A new list starts where two consecutive arcs are not connected within max_route_m."""
    point_arcs, seg_id = np.asarray(point_arcs), np.asarray(seg_id)
    routes = []
    for s in np.unique(seg_id):
        arcs = point_arcs[(seg_id == s) & (point_arcs >= 0)]
        if not len(arcs):
            continue
        arcs = arcs[np.r_[True, arcs[1:] != arcs[:-1]]]
        cur = [int(arcs[0])]
        for b in arcs[1:]:
            a = cur[-1]
            gap = [] if graph.v[a] == graph.u[b] else graph.path_arcs(graph.v[a], graph.u[b])
            if gap is None:
                routes.append(cur)
                cur = [int(b)]
            else:
                cur.extend(gap)
                cur.append(int(b))
        routes.append(cur)
    return routes


def continuity(point_arcs, seg_id, graph, txy):
    """Share of consecutive matched points (same segment) whose arcs are the same, directly connected, or joined
    by a network path of at most 2 x their GPS displacement + 50 m."""
    point_arcs, seg_id, txy = np.asarray(point_arcs), np.asarray(seg_id), np.asarray(txy, dtype=float)
    idx = np.flatnonzero(point_arcs >= 0)
    if len(idx) < 2:
        return None
    i, j = idx[:-1], idx[1:]
    keep = seg_id[i] == seg_id[j]
    i, j = i[keep], j[keep]
    if not len(i):
        return None
    a, b = point_arcs[i], point_arcs[j]
    ok = (a == b) | (graph.v[a] == graph.u[b])
    disp = np.hypot(*(txy[j] - txy[i]).T)
    graph.prefetch(graph.v[a[~ok]])
    for k in np.flatnonzero(~ok):
        ok[k] = graph.sp(graph.v[a[k]], graph.u[b[k]])[0] <= 2.0 * disp[k] + 50.0
    return float(ok.mean())
