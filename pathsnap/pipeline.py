"""Single prediction entry point used by the API (and maps/experiments). Reads only .track/.nodes/.arcs - NEVER .route.

Owner: P2 (written by P1)

    bundle = load_bundle()                      # artifacts/model.joblib: {"model", "features", "model_name", ...}
    pred, ctx = predict_trip(trip_dir, bundle)  # per-point arcs (classifier-only and decoded), route, continuity, timings

Decoding: if pathsnap.hmm provides the agreed interface (RoadGraph, decode, load_hmm_cfg), the HMM/Viterbi decoder
is used. Until then the decoder falls back to the classifier's best candidate per point, and pred["decoder"] says so.
"""
import time
from pathlib import Path

import joblib
import numpy as np

from .prepare import prepare_trip

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
MAX_GPS_POINTS = 2000


def load_bundle(path=MODEL_PATH):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError(f"{path} must be a dict with 'model' and 'features'")
    bundle.setdefault("model_name", type(bundle["model"]).__name__)
    return bundle


def positive_proba(model, X):
    p = model.predict_proba(X)
    classes = list((model[-1] if hasattr(model, "steps") else model).classes_)
    return p[:, classes.index(1)]


def classifier_best(c, proba, n_points):
    """Highest-probability candidate arc per GPS point; -1 where a point has no candidates."""
    best = np.full(n_points, -1, dtype=np.int64)
    if len(c):
        order = np.lexsort((-proba, c["obs_idx"].to_numpy()))   # by point, then descending probability
        obs = c["obs_idx"].to_numpy()[order]
        first = np.r_[True, obs[1:] != obs[:-1]]
        best[obs[first]] = c["arc_id"].to_numpy()[order][first]
    return best


def decode(c, proba, ctx, clf_arcs):
    """HMM/Viterbi decoding via pathsnap.hmm when it is implemented; otherwise classifier-only fallback.
    Returns (arc per point, number of HMM breaks, decoder name)."""
    try:
        from . import hmm
        if not all(hasattr(hmm, f) for f in ("RoadGraph", "decode", "load_hmm_cfg")):
            raise NotImplementedError("pathsnap.hmm does not provide RoadGraph/decode/load_hmm_cfg yet")
        cfg = hmm.load_hmm_cfg()
        graph = hmm.RoadGraph(ctx["u"], ctx["v"], ctx["A"], ctx["B"], ctx["keep"], cfg["max_route_m"])
        arcs, breaks = hmm.decode(c, proba, ctx["txy"], ctx["seg_id"], graph, cfg)
        return np.asarray(arcs, dtype=np.int64), int(breaks), "hmm_viterbi"
    except Exception as e:  # noqa: BLE001 - any HMM failure falls back, and is reported, never hidden
        return clf_arcs.copy(), 0, f"classifier_only (HMM unavailable: {type(e).__name__})"


def continuity(point_arcs, seg_id, u, v):
    """Share of consecutive point pairs (same segment, both matched) on the same arc or on directly connected arcs."""
    a, b = point_arcs[:-1], point_arcs[1:]
    ok_pair = (a >= 0) & (b >= 0) & (seg_id[:-1] == seg_id[1:])
    if not ok_pair.any():
        return None
    a, b = a[ok_pair], b[ok_pair]
    connected = (a == b) | (v[a] == u[b])
    return float(connected.mean())


def route_segments(point_arcs, seg_id):
    """One arc list per GPS segment, consecutive repeats removed and unmatched points skipped."""
    out = []
    for s in np.unique(seg_id):
        arcs = point_arcs[(seg_id == s) & (point_arcs >= 0)]
        if len(arcs):
            out.append(arcs[np.r_[True, arcs[1:] != arcs[:-1]]].tolist())
    return out


def predict_trip(trip_dir, bundle):
    """Full prediction for one trip folder. Never opens .route."""
    timings = {}
    t0 = time.perf_counter()
    c, ctx = prepare_trip(trip_dir)
    timings["prepare"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    X = c[bundle["features"]].to_numpy(np.float32)
    proba = positive_proba(bundle["model"], X) if len(c) else np.zeros(0)
    n = len(ctx["txy"])
    clf_arcs = classifier_best(c, proba, n)
    timings["classify"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    hmm_arcs, breaks, decoder = decode(c, proba, ctx, clf_arcs)
    timings["hmm"] = time.perf_counter() - t2
    timings["predict_total"] = time.perf_counter() - t0

    seg_id, u, v = ctx["seg_id"], ctx["u"], ctx["v"]
    pred = {
        "trip_id": ctx["trip_id"],
        "model": bundle["model_name"],
        "decoder": decoder,
        "n_points": int(n),
        "n_with_candidates": int(c["obs_idx"].nunique()) if len(c) else 0,
        "n_segments": int(len(np.unique(seg_id))),
        "n_hmm_breaks": breaks,
        "clf_arcs": clf_arcs,
        "hmm_arcs": hmm_arcs,
        "route_segments": route_segments(hmm_arcs, seg_id),
        "continuity_clf": continuity(clf_arcs, seg_id, u, v),
        "continuity_hmm": continuity(hmm_arcs, seg_id, u, v),
        "timings_s": {k: round(x, 3) for k, x in timings.items()},
    }
    return pred, ctx


# ---------------------------------------------------------------- helpers for the UI response ([lat, lon] order)
def gps_latlon(ctx, max_points=MAX_GPS_POINTS):
    track = ctx["track"]
    step = max(1, len(track) // max_points)
    return track[["lat", "lon"]].to_numpy()[::step].round(7).tolist()


def arcs_to_latlon(route_segments, ctx):
    """Polylines for the predicted route: consecutive connected arcs are joined; a new polyline starts at a break."""
    u, v, ll = ctx["u"], ctx["v"], ctx["nodes_ll"]
    lines = []
    for seg in route_segments:
        cur = []
        for k, a in enumerate(seg):
            if k > 0 and v[seg[k - 1]] != u[a]:
                if len(cur) > 1:
                    lines.append(cur)
                cur = []
            if not cur:
                cur.append([round(float(ll[u[a], 1]), 7), round(float(ll[u[a], 0]), 7)])
            cur.append([round(float(ll[v[a], 1]), 7), round(float(ll[v[a], 0]), 7)])
        if len(cur) > 1:
            lines.append(cur)
    return lines
