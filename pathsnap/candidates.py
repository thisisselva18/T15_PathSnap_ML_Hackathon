"""Candidate generation: K nearest arcs within R metres of each GPS point.

Owner: P1
"""

import json
from pathlib import Path

CFG = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "candidate_config.json").read_text())


def generate_candidates(points_xy, arc_geom, radius_m=CFG["radius_m"], k=CFG["k"]):
    raise NotImplementedError
