"""Geometry: per-trip UTM projection, network cropping, point-to-segment distance.

Why UTM: lon/lat are angles, a degree of longitude shrinks with latitude, so Euclidean distance in
degrees is wrong. Trips are city-scale, so projecting each trip into its own UTM zone (chosen from the
track centroid) gives metre coordinates with negligible distortion. Works for both hemispheres.

Owner: P1
"""
import numpy as np
try:
    from pyproj import Transformer
except ImportError as exc:  # pragma: no cover - exercised in dependency-free environments
    Transformer = None
    _PYPROJ_IMPORT_ERROR = exc
else:
    _PYPROJ_IMPORT_ERROR = None


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def project_trip(track, nodes):
    """Return (track_xy, nodes_xy, epsg) in metres."""
    if Transformer is None:
        raise ImportError(
            "PyProj is required for metric map-matching geometry; install the project requirements."
        ) from _PYPROJ_IMPORT_ERROR
    for name, frame in (("track", track), ("nodes", nodes)):
        coords = frame[["lon", "lat"]].to_numpy(dtype=float)
        if not np.isfinite(coords).all():
            raise ValueError(f"{name} coordinates must be finite")
        if ((coords[:, 0] < -180) | (coords[:, 0] > 180) |
                (coords[:, 1] < -80) | (coords[:, 1] > 84)).any():
            raise ValueError(f"{name} coordinates are outside valid UTM longitude/latitude bounds")
    if track.empty:
        raise ValueError("track must contain at least one valid coordinate")
    epsg = utm_epsg(float(track.lon.median()), float(track.lat.median()))
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    txy = np.column_stack(tf.transform(track.lon.to_numpy(dtype=float), track.lat.to_numpy(dtype=float)))
    nxy = np.column_stack(tf.transform(nodes.lon.to_numpy(dtype=float), nodes.lat.to_numpy(dtype=float)))
    if not np.isfinite(txy).all() or not np.isfinite(nxy).all():
        raise ValueError("projected coordinates must be finite")
    return txy, nxy, epsg


def crop_arc_ids(nxy, u, v, txy, margin_m=200.0):
    """Conservatively keep every segment whose bounding box meets the expanded track box.

    Testing segment envelopes (rather than endpoints alone) retains long roads that cross the
    search area with both endpoint nodes outside it. Envelope overlap is conservative: it can
    retain extra arcs but cannot exclude a segment that intersects the expanded rectangle.
    """
    if margin_m < 0 or not np.isfinite(margin_m):
        raise ValueError("margin_m must be a finite non-negative distance")
    nxy = np.asarray(nxy, dtype=float)
    u, v = np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64)
    txy = np.asarray(txy, dtype=float)
    if nxy.ndim != 2 or nxy.shape[1] != 2 or txy.ndim != 2 or txy.shape[1] != 2:
        raise ValueError("nxy and txy must have shape (n, 2)")
    if not np.isfinite(nxy).all() or not np.isfinite(txy).all():
        raise ValueError("network and track coordinates must be finite")
    if u.shape != v.shape or np.any(u < 0) or np.any(v < 0) or np.any(u >= len(nxy)) or np.any(v >= len(nxy)):
        raise ValueError("arc endpoint IDs must index the node table")
    if len(txy) == 0:
        return np.empty(0, dtype=np.int64)
    lo, hi = txy.min(0) - margin_m, txy.max(0) + margin_m
    seg_lo = np.minimum(nxy[u], nxy[v])
    seg_hi = np.maximum(nxy[u], nxy[v])
    overlaps = np.all((seg_hi >= lo) & (seg_lo <= hi), axis=1)
    return np.flatnonzero(overlaps)


def point_segment(px, py, ax, ay, bx, by):
    """Distance from point(s) to segment(s) A->B (broadcasts). Returns (dist, t_clipped, t_raw)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t_raw = ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1.0)
    t = np.clip(t_raw, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t, t_raw
