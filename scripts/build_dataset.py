"""Build the candidate/feature tables + per-point labels for the train and student-test partitions.

Owner: P1 + P3
Usage (repo root):  python scripts/build_dataset.py              # both partitions, all CPU cores
                    python scripts/build_dataset.py --jobs 4 --parts student_test

Outputs
  data/processed/candidates_<part>.parquet   one row per (GPS point, candidate arc):
        keys    trip_id, seg_id, obs_idx, t, cand_rank, arc_id
        labels  label (1 = this candidate is the aligned reference arc), ref_arc_id, ref_dist
        features  see pathsnap/features.py FEATURES (also written to artifacts/feature_list.json)
  results/candidate_stats_<part>.csv         per-trip: points, segments, network size, candidate miss
                                             rate, points with no candidate, label quality
Evaluator-test data is never touched by this script.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.io import load_trip  # noqa: E402
from pathsnap.labels import align_points_to_route  # noqa: E402
from pathsnap.prepare import load_cfg, prepare_trip  # noqa: E402


def process(trip_dir, cfg):
    t0 = time.perf_counter()
    c, ctx = prepare_trip(trip_dir, cfg)
    route = load_trip(trip_dir, with_route=True)["route"]          # labels only - never a feature
    ref_arc, ref_dist = align_points_to_route(ctx["txy"], ctx["t"], ctx["seg_id"], route, ctx["A"], ctx["B"])
    obs = c["obs_idx"].to_numpy()
    c["ref_arc_id"] = ref_arc[obs]
    c["ref_dist"] = ref_dist[obs].astype(np.float32)
    c["label"] = (c["arc_id"].to_numpy() == c["ref_arc_id"].to_numpy()).astype(np.int8)

    n = len(ctx["txy"])
    has_cand = np.zeros(n, bool)
    has_cand[obs] = True
    hit = np.zeros(n, bool)
    hit[c.loc[c["label"] == 1, "obs_idx"].to_numpy()] = True
    stats = dict(trip_id=ctx["trip_id"], n_points=n, n_segments=int(ctx["seg_id"].max()) + 1,
                 n_arcs=ctx["n_arcs"], n_arcs_cropped=ctx["n_arcs_cropped"], route_len=len(route),
                 pct_no_candidate=100 * (1 - has_cand.mean()),
                 candidate_miss_rate=100 * (1 - hit.mean()),
                 mean_ref_dist_m=float(ref_dist.mean()), pct_ref_dist_gt50=100 * float((ref_dist > 50).mean()),
                 rows=len(c), seconds=time.perf_counter() - t0)
    return c, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["train", "student_test"])
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    cfg = load_cfg()
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    for part in a.parts:
        trips = sorted(p for p in (ROOT / "data" / "raw" / part).iterdir() if p.is_dir())
        print(f"[{part}] {len(trips)} trips, config {cfg}")
        t0 = time.perf_counter()
        res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(process)(p, cfg) for p in trips)
        df = pd.concat([r[0] for r in res], ignore_index=True)
        stats = pd.DataFrame([r[1] for r in res])
        df.to_parquet(out_dir / f"candidates_{part}.parquet", index=False)
        stats.to_csv(ROOT / "results" / f"candidate_stats_{part}.csv", index=False)
        pts = stats["n_points"].sum()
        miss = (stats["candidate_miss_rate"] * stats["n_points"]).sum() / pts
        print(f"[{part}] {len(df):,} rows, {pts:,} points, positives {df['label'].mean():.3f}, "
              f"candidate miss rate {miss:.2f}%, {time.perf_counter() - t0:.0f}s")

    (ROOT / "artifacts" / "feature_list.json").write_text(json.dumps(FEATURES, indent=1))
    print("wrote artifacts/feature_list.json")


if __name__ == "__main__":
    main()
