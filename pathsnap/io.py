"""Load raw trip files into DataFrames.

.track: lon, lat, t (s) | .nodes: lon, lat (row = node_id) | .arcs: from_node, to_node (row = arc_id)
.route: ground-truth arc_id sequence (only read when with_route=True - training/evaluation only)

Owner: P1
"""
from pathlib import Path

import pandas as pd


def load_trip(trip_dir, with_route=False):
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    rd = lambda ext, names: pd.read_csv(trip_dir / f"{tid}.{ext}", sep=r"\s+", header=None, names=names)
    out = {
        "trip_id": tid,
        "track": rd("track", ["lon", "lat", "t"]),
        "nodes": rd("nodes", ["lon", "lat"]),
        "arcs": rd("arcs", ["u", "v"]),
    }
    if with_route:
        out["route"] = rd("route", ["arc_id"])["arc_id"].to_numpy()
    return out


def clean_track(track):
    """Drop rows with missing/invalid coordinates, sort by time, drop duplicate timestamps."""
    t = track.dropna()
    t = t[(t.lon.between(-180, 180)) & (t.lat.between(-90, 90))]
    t = t.sort_values("t", kind="stable").drop_duplicates("t", keep="first")
    return t.reset_index(drop=True)
