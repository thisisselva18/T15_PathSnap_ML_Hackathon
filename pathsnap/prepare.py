"""Turn ONE trip folder into the candidate/feature table. Shared by build_dataset.py and the API,
so training and inference use exactly the same preprocessing. Never reads .route.

Owner: P1 (+ P3 for segments)
"""
import json
from pathlib import Path

import numpy as np

from .candidates import generate_candidates
from .features import FEATURES, add_features
from .geo import crop_arc_ids, project_trip
from .io import clean_track, load_trip
from .segment import split_segments

CFG_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "candidate_config.json"


def load_cfg():
    return json.loads(CFG_PATH.read_text())


def prepare_trip(trip_dir, cfg=None):
    """Return (cand_df, ctx). ctx holds projected geometry for labels / HMM / maps."""
    cfg = cfg or load_cfg()
    raw = load_trip(trip_dir)
    track = clean_track(raw["track"])
    txy, nxy, epsg = project_trip(track, raw["nodes"])
    t = track["t"].to_numpy(float)
    u, v = raw["arcs"]["u"].to_numpy(), raw["arcs"]["v"].to_numpy()
    A, B = nxy[u], nxy[v]
    seg_id = split_segments(txy, t, cfg["gap_s"], cfg["max_speed_ms"])

    keep = crop_arc_ids(nxy, u, v, txy, cfg.get("crop_margin_m", 200))
    n_nodes = len(nxy)
    out_deg = np.bincount(u, minlength=n_nodes)
    in_deg = np.bincount(v, minlength=n_nodes)
    pair = set(zip(u.tolist(), v.tolist()))
    twin = np.fromiter(((b, a) in pair for a, b in zip(u.tolist(), v.tolist())), bool, len(u))

    c = generate_candidates(txy, A, B, keep, cfg["radius_m"], cfg["k"])
    c = add_features(c, txy, t, seg_id, A, B, u, v, out_deg, in_deg, twin)
    c.insert(0, "trip_id", raw["trip_id"])
    c.insert(1, "seg_id", seg_id[c["obs_idx"].to_numpy()])
    c.insert(3, "t", t[c["obs_idx"].to_numpy()])
    c[FEATURES] = c[FEATURES].astype(np.float32)

    ctx = dict(trip_id=raw["trip_id"], track=track, txy=txy, nxy=nxy, t=t, seg_id=seg_id, u=u, v=v,
               A=A, B=B, epsg=epsg, n_arcs=len(u), n_arcs_cropped=len(keep))
    return c, ctx
