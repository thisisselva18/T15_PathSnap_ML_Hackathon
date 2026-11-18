"""Load raw trip files into DataFrames.

.track: lon, lat, t (s) | .nodes: lon, lat (row = node_id) | .arcs: from_node, to_node (row = arc_id) | .route: arc_id sequence

Owner: P1
"""

from pathlib import Path
import pandas as pd


def load_trip(trip_dir, with_route=False):
    """Return dict with 'track', 'nodes', 'arcs' (and 'route' only if with_route=True)."""
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
