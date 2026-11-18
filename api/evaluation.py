"""The ONLY API module that reads .route. Called after predict_trip() to compute metrics server-side;
ground truth is never returned to the UI (only the scores below are).

Owner: P2 (written by P1)
"""
from pathlib import Path

import numpy as np

from pathsnap.io import load_trip
from pathsnap.labels import align_points_to_route

RELIABLE_M = 50.0  # reference arcs further than this from the GPS point are unreliable labels (e.g. trip 54)


def reference(trip_dir, ctx):
    """Per-point reference arc from .route, aligned exactly like the training labels. None if no .route."""
    trip_dir = Path(trip_dir)
    if not (trip_dir / f"{trip_dir.name}.route").exists():
        return None
    route = load_trip(trip_dir, with_route=True)["route"]
    ref_arc, ref_dist = align_points_to_route(ctx["txy"], ctx["t"], ctx["seg_id"], route, ctx["A"], ctx["B"])
    return {"route": np.asarray(route), "ref_arc": np.asarray(ref_arc), "ref_dist": np.asarray(ref_dist)}


def _top1(pred_arcs, ref_arc, mask=None):
    m = np.ones(len(ref_arc), bool) if mask is None else mask
    return float(np.mean(pred_arcs[m] == ref_arc[m])) if m.any() else None


def metrics(pred, ref):
    """Top-1 over ALL points (no prediction counts as wrong), Top-1 on reliable points, route precision/recall."""
    out = {"ground_truth_available": ref is not None,
           "continuity_hmm": pred["continuity_hmm"], "continuity_clf": pred["continuity_clf"]}
    if ref is None:
        return out
    hmm, clf, ref_arc = pred["hmm_arcs"], pred["clf_arcs"], ref["ref_arc"]
    reliable = ref["ref_dist"] <= RELIABLE_M
    pred_set = {a for seg in pred["route_segments"] for a in seg}
    ref_set = set(ref["route"].tolist())
    hit = len(pred_set & ref_set)
    out.update({
        "hmm_top1": _top1(hmm, ref_arc),
        "clf_top1": _top1(clf, ref_arc),
        "hmm_top1_reliable": _top1(hmm, ref_arc, reliable),
        "clf_top1_reliable": _top1(clf, ref_arc, reliable),
        "reliable_share": float(reliable.mean()),
        "route_precision": hit / len(pred_set) if pred_set else None,
        "route_recall": hit / len(ref_set) if ref_set else None,
    })
    return out
