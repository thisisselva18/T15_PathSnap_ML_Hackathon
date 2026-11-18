"""Per-point reference labels: monotone alignment DP of GPS points to the .route arc sequence. TRAINING/EVAL ONLY.

Owner: P3
"""

def align_points_to_route(points_xy, route_arcs, arc_geom):
    """Return ref_arc_id for each GPS point (and its distance in metres)."""
    raise NotImplementedError
