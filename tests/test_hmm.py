import numpy as np
import pandas as pd

from pathsnap import hmm

# Synthetic network (metres). Nodes 0-1-2 along y=0 (a two-way street), node 3-4 a far-away street.
NODES = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [0.0, 5000.0], [100.0, 5000.0]])
U = np.array([0, 1, 1, 2, 3])          # arc 0: 0->1, 1: 1->0 (twin), 2: 1->2, 3: 2->1 (twin), 4: 3->4 (disconnected)
V = np.array([1, 0, 2, 1, 4])
A, B = NODES[U], NODES[V]
CFG = dict(hmm.DEFAULTS)


def graph(max_route_m=2000.0):
    return hmm.RoadGraph(U, V, A, B, np.arange(len(U)), max_route_m)


def cands_for(txy, arcs_per_point, proba_per_point):
    rows, proba = [], []
    for i, (arcs, ps) in enumerate(zip(arcs_per_point, proba_per_point)):
        for a, p in zip(arcs, ps):
            ax, ay = A[a]
            bx, by = B[a]
            vx, vy = bx - ax, by - ay
            t = np.clip(((txy[i, 0] - ax) * vx + (txy[i, 1] - ay) * vy) / (vx * vx + vy * vy), 0, 1)
            d = np.hypot(txy[i, 0] - (ax + t * vx), txy[i, 1] - (ay + t * vy))
            rows.append({"obs_idx": i, "arc_id": a, "dist": d, "t_frac": t})
            proba.append(p)
    return pd.DataFrame(rows), np.array(proba)


def test_shortest_paths_and_path_arcs():
    g = graph()
    assert g.sp(0, [2])[0] == 200.0
    assert g.path_arcs(0, 2) == [0, 2]
    assert g.path_arcs(2, 0) == [3, 1]
    assert np.isinf(g.sp(0, [4])[0])
    assert g.path_arcs(0, 4) is None


def test_load_cfg_ignores_nulls(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"beta_m": 7, "sigma_m": null, "_tuning": {"x": 1}}')
    cfg = hmm.load_hmm_cfg(p)
    assert cfg["beta_m"] == 7.0 and cfg["sigma_m"] == hmm.DEFAULTS["sigma_m"]


def test_direction_follows_motion_despite_noisy_classifier():
    # driving east along y=1; the classifier slightly prefers the westbound twins at every point
    txy = np.column_stack([np.arange(10, 200, 20.0), np.ones(10)])
    arcs = [[0, 1] if x < 100 else [2, 3] for x in txy[:, 0]]
    probs = [[0.45, 0.55]] * len(txy)
    c, p = cands_for(txy, arcs, probs)
    out, breaks = hmm.decode(c, p, txy, np.zeros(len(txy), np.int32), graph(), CFG)
    expected = np.where(txy[:, 0] < 100, 0, 2)
    assert breaks == 0
    assert (out == expected).all()


def test_unreachable_step_restarts_chain_and_counts_break():
    txy = np.array([[10.0, 1.0], [30.0, 1.0], [10.0, 5001.0], [40.0, 5001.0]])
    c, p = cands_for(txy, [[0], [0], [4], [4]], [[0.9], [0.9], [0.9], [0.9]])
    out, breaks = hmm.decode(c, p, txy, np.zeros(4, np.int32), graph(), CFG)
    assert breaks == 1
    assert out.tolist() == [0, 0, 4, 4]


def test_new_segment_is_not_a_break_and_missing_candidates_stay_minus_one():
    txy = np.array([[10.0, 1.0], [50.0, 1.0], [10.0, 5001.0], [40.0, 5001.0]])
    c, p = cands_for(txy, [[0], [], [4], [4]], [[0.9], [], [0.9], [0.9]])
    seg = np.array([0, 0, 1, 1], np.int32)
    out, breaks = hmm.decode(c, p, txy, seg, graph(), CFG)
    assert breaks == 0
    assert out.tolist() == [0, -1, 4, 4]


def test_route_from_points_fills_gaps_and_splits_disconnected():
    g = graph()
    routes = hmm.route_from_points(np.array([0, 0, -1, 2, 4]), np.zeros(5, np.int32), g)
    assert routes == [[0, 2], [4]]


def test_continuity():
    g = graph()
    txy = np.array([[10.0, 0], [90.0, 0], [150.0, 0], [180.0, 0]])
    assert hmm.continuity(np.array([0, 0, 2, 2]), np.zeros(4, np.int32), g, txy) == 1.0
    assert hmm.continuity(np.array([0, 0, 4, 2]), np.zeros(4, np.int32), g, txy) == 1 / 3
