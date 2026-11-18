"""Feature engineering per (GPS point, candidate arc). Uses GPS + network only, never .route.

Owner: P1
"""

FEATURES = []  # keep in sync with artifacts/feature_list.json


def build_features(cands, track_xy, t, arc_geom, graph):
    raise NotImplementedError
