"""Folium maps: GPS, network, reference route, predicted route, LayerControl + legend -> results/maps/.

Owner: P1

Usage (from repo root):
    python scripts/make_maps.py --raw 00000052                  # EDA map: GPS points + nearby road network
    python scripts/make_maps.py --trip 00000077                 # + reference route, classifier-only, HMM route
    python scripts/make_maps.py --trip 00000077 --preds results/preds/logreg_student_test.parquet

Layers
    Raw GPS points           blue dots          (both modes)
    Road network near track  thin grey lines    (both modes; arcs within --near-m metres of the track)
    Reference route (.route) green              (--trip only)
    Predicted route (HMM)    red                (--trip only, when pathsnap.pipeline.predict_trip works)
    Classifier-only choices  orange, hidden     (--trip only, from a saved predictions file)

The reference layer reads .route, so these maps are offline report material only. They never go into the UI.
Folium expects [lat, lon]; the raw files store lon, lat.
"""
import argparse
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap.geo import project_trip  # noqa: E402
from pathsnap.io import clean_track, load_trip  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "maps"
MAX_GPS_MARKERS = 3000  # beyond this the HTML gets slow; points are thinned evenly

COLOURS = {"gps": "#1f5fd6", "network": "#8c8c8c", "reference": "#1a9641", "hmm": "#d7191c", "clf": "#fd8d3c"}


def find_trip(trip_id):
    """Return (trip_dir, partition) for a trip in data/raw/<partition>/<trip_id>."""
    for part in ("train", "student_test"):
        d = RAW / part / trip_id
        if d.is_dir():
            return d, part
    raise SystemExit(f"trip {trip_id} not found under {RAW}\\train or {RAW}\\student_test")


def arc_latlon(arc_ids, u, v, nodes_ll):
    """One [[lat, lon], [lat, lon]] line per arc."""
    return [[[nodes_ll[u[a], 1], nodes_ll[u[a], 0]], [nodes_ll[v[a], 1], nodes_ll[v[a], 0]]] for a in arc_ids]


def chain_latlon(arc_seq, u, v, nodes_ll):
    """Join consecutive arcs into polylines; start a new polyline wherever the sequence is not connected."""
    lines, cur = [], []
    for k, a in enumerate(arc_seq):
        if k > 0 and v[arc_seq[k - 1]] != u[a]:
            lines.append(cur)
            cur = []
        if not cur:
            cur.append([nodes_ll[u[a], 1], nodes_ll[u[a], 0]])
        cur.append([nodes_ll[v[a], 1], nodes_ll[v[a], 0]])
    if cur:
        lines.append(cur)
    return lines


def dedupe(seq):
    """Drop consecutive repeats: [5, 5, 5, 9, 9, 5] -> [5, 9, 5]."""
    seq = np.asarray(seq)
    return seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq


def nearby_arcs(txy, nxy, u, v, near_m):
    """Arcs whose segment lies within near_m metres of any (thinned) GPS point, via an STRtree."""
    lines = shapely.linestrings(np.stack([nxy[u], nxy[v]], axis=1))
    tree = shapely.STRtree(lines)
    pts = shapely.points(txy[:: max(1, len(txy) // 5000)])
    _, arc_idx = tree.query(pts, predicate="dwithin", distance=near_m)
    return np.unique(arc_idx)


def add_legend(fmap, title, rows):
    items = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{name}</div>' for name, c in rows)
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
            f'padding:8px 10px;border:1px solid #bbb;border-radius:4px;font:12px sans-serif">'
            f'<b>{title}</b>{items}</div>')
    fmap.get_root().html.add_child(folium.Element(html))


def classifier_only_arcs(preds_path, trip_id):
    """Highest-scoring candidate per GPS point, in time order, from a saved predictions file."""
    p = pd.read_parquet(preds_path)
    p = p[p["trip_id"].astype(str) == trip_id]
    if p.empty:
        return None
    best = p.loc[p.groupby("obs_idx")["score"].idxmax()].sort_values("obs_idx")
    return best["arc_id"].to_numpy()


def hmm_route(trip_dir):
    """HMM route segments from the shared prediction pipeline, or None if it isn't available yet."""
    try:
        from pathsnap.pipeline import load_bundle, predict_trip
        from pathsnap.hmm import load_hmm_cfg
        pred, _, _ = predict_trip(trip_dir, load_bundle(), load_hmm_cfg())
        return pred["route_segments"]
    except Exception as e:  # pipeline/HMM still being built: skip the layer rather than fake it
        print(f"  HMM route layer skipped ({type(e).__name__}: {e})")
        return None


def make_map(trip_id, with_reference, preds_path, near_m):
    trip_dir, part = find_trip(trip_id)
    raw = load_trip(trip_dir, with_route=with_reference and (trip_dir / f"{trip_id}.route").exists())
    track = clean_track(raw["track"])
    txy, nxy, epsg = project_trip(track, raw["nodes"])
    u, v = raw["arcs"]["u"].to_numpy(), raw["arcs"]["v"].to_numpy()
    nodes_ll = raw["nodes"][["lon", "lat"]].to_numpy()
    gps = track[["lat", "lon"]].to_numpy()

    fmap = folium.Map(tiles="OpenStreetMap", control_scale=True)
    legend = []

    # road network near the track
    near = nearby_arcs(txy, nxy, u, v, near_m)
    net = folium.FeatureGroup(name=f"Road network within {near_m:.0f} m ({len(near)} arcs)", show=True)
    folium.PolyLine(arc_latlon(near, u, v, nodes_ll), color=COLOURS["network"], weight=1.5, opacity=0.7).add_to(net)
    net.add_to(fmap)
    legend.append(("Road network", COLOURS["network"]))

    # reference route (offline report maps only)
    if with_reference and "route" in raw:
        ref = folium.FeatureGroup(name=f"Reference route ({len(raw['route'])} arcs)", show=True)
        folium.PolyLine(chain_latlon(raw["route"], u, v, nodes_ll), color=COLOURS["reference"],
                        weight=6, opacity=0.6).add_to(ref)
        ref.add_to(fmap)
        legend.append(("Reference route (.route)", COLOURS["reference"]))

    # predicted route (classifier + HMM)
    if with_reference:
        segs = hmm_route(trip_dir)
        if segs:
            hmm = folium.FeatureGroup(name="Predicted route (classifier + HMM)", show=True)
            for seg in segs:
                folium.PolyLine(chain_latlon(seg, u, v, nodes_ll), color=COLOURS["hmm"], weight=3).add_to(hmm)
            hmm.add_to(fmap)
            legend.append(("Predicted route (HMM)", COLOURS["hmm"]))

    # classifier-only choices
    if preds_path:
        arcs = classifier_only_arcs(preds_path, trip_id)
        if arcs is None:
            print(f"  no rows for trip {trip_id} in {preds_path}; classifier-only layer skipped")
        else:
            clf = folium.FeatureGroup(name=f"Classifier-only choices ({Path(preds_path).stem})", show=False)
            folium.PolyLine(arc_latlon(dedupe(arcs), u, v, nodes_ll), color=COLOURS["clf"],
                            weight=3, opacity=0.9).add_to(clf)
            clf.add_to(fmap)
            legend.append(("Classifier-only (toggle)", COLOURS["clf"]))

    # raw GPS points, drawn last so they sit on top
    step = max(1, len(gps) // MAX_GPS_MARKERS)
    pts = folium.FeatureGroup(name=f"Raw GPS points ({len(gps)}, every {step} shown)", show=True)
    for lat, lon in gps[::step]:
        folium.CircleMarker([lat, lon], radius=2, color=COLOURS["gps"], fill=True, fill_opacity=0.8,
                            weight=0).add_to(pts)
    pts.add_to(fmap)
    folium.Marker(gps[0].tolist(), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker(gps[-1].tolist(), tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    legend.insert(0, ("Raw GPS points", COLOURS["gps"]))

    fmap.fit_bounds([gps.min(axis=0).tolist(), gps.max(axis=0).tolist()])
    folium.LayerControl(collapsed=False).add_to(fmap)
    add_legend(fmap, f"Trip {trip_id} ({part})", legend)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (f"{trip_id}.html" if with_reference else f"{trip_id}_raw.html")
    fmap.save(str(out))
    print(f"trip {trip_id} ({part}): {len(gps)} GPS points, {len(u)} arcs, {len(near)} near the track, "
          f"UTM EPSG:{epsg} -> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--raw", metavar="TRIP", help="EDA map: GPS points + nearby road network")
    g.add_argument("--trip", metavar="TRIP", help="result map: + reference, HMM and classifier-only layers")
    ap.add_argument("--preds", default=None,
                    help="predictions parquet for the classifier-only layer (default: results/preds/logreg_student_test.parquet if present)")
    ap.add_argument("--near-m", type=float, default=100.0, help="draw network arcs within this many metres of the track")
    a = ap.parse_args()

    preds = a.preds
    if a.trip and preds is None:
        default = ROOT / "results" / "preds" / "logreg_student_test.parquet"
        preds = default if default.exists() else None
    make_map(a.raw or a.trip, with_reference=bool(a.trip), preds_path=preds if a.trip else None, near_m=a.near_m)


if __name__ == "__main__":
    main()
