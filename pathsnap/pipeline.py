"""Single prediction entry point used by experiments AND the API. Reads only .track/.nodes/.arcs - NEVER .route.

Owner: P4
"""

def predict_trip(trip_dir):
    """Return {'obs_arcs': [...], 'route_arcs': [...], 'segments': [...]}."""
    raise NotImplementedError
