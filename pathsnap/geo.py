"""Geometry: per-trip UTM projection, network cropping, point-to-segment distance.

Why UTM: lon/lat are angles, a degree of longitude shrinks with latitude, so Euclidean distance in
degrees is wrong. Trips are city-scale, so projecting each trip into its own UTM zone (chosen from the
track centroid) gives metre coordinates with negligible distortion. Works for both hemispheres.

Owner: P1
"""
import numpy as np
from pyproj import Transformer


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def project_trip(track, nodes):
    """Return (track_xy, nodes_xy, epsg) in metres."""
    epsg = utm_epsg(float(track.lon.median()), float(track.lat.median()))
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    txy = np.column_stack(tf.transform(track.lon.to_numpy(), track.lat.to_numpy()))
    nxy = np.column_stack(tf.transform(nodes.lon.to_numpy(), nodes.lat.to_numpy()))
    return txy, nxy, epsg


def crop_arc_ids(nxy, u, v, txy, margin_m=200.0):
    """Arc ids with at least one end node inside the track bounding box + margin."""
    lo, hi = txy.min(0) - margin_m, txy.max(0) + margin_m
    inside = np.all((nxy >= lo) & (nxy <= hi), axis=1)
    return np.flatnonzero(inside[u] | inside[v])


def point_segment(px, py, ax, ay, bx, by):
    """Distance from point(s) to segment(s) A->B (broadcasts). Returns (dist, t_clipped, t_raw)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t_raw = ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1.0)
    t = np.clip(t_raw, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t, t_raw
