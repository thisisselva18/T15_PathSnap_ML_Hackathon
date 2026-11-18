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
import importlib.metadata
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
    raw=load_trip(trip_dir)
    c, ctx = prepare_trip(trip_dir, cfg)
    route = load_trip(trip_dir, with_route=True)["route"]          # labels only - never a feature
    if np.any((route<0)|(route>=ctx["n_arcs"])):
        raise ValueError(f"Trip {ctx['trip_id']} has route arc IDs outside the .arcs table")
    ref_arc, ref_dist, alignment = align_points_to_route(
        ctx["txy"], ctx["t"], ctx["seg_id"], route, ctx["A"], ctx["B"], return_diagnostics=True)
    route_breaks = np.flatnonzero(ctx["v"][route[:-1]] != ctx["u"][route[1:]])
    transition_diagnostics = pd.DataFrame({
        "trip_id": ctx["trip_id"],
        "route_position": route_breaks,
        "from_arc_id": route[route_breaks],
        "from_node_id": ctx["v"][route[route_breaks]],
        "to_arc_id": route[route_breaks + 1],
        "to_node_id": ctx["u"][route[route_breaks + 1]],
    })
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
                 n_obs_raw=len(raw["track"]), n_obs_dropped=len(raw["track"])-n,
                 n_timestamp_nonincreasing=int((np.diff(raw["track"]["t"].to_numpy(float))<=0).sum()),
                 n_gaps_over_60s=int((np.diff(ctx["t"])>60).sum()),
                 route_discontinuities=len(route_breaks),
                 alignment_position_monotone=alignment["position_monotone"],
                 alignment_jumps_over_max_skip=alignment["n_position_jumps_over_max_skip"],
                 alignment_ref_dist_p95_m=alignment["ref_dist_p95_m"],
                 alignment_ref_dist_max_m=alignment["ref_dist_max_m"],
                 alignment_obs_over_50m=alignment["n_ref_dist_gt_50m"],
                 alignment_obs_over_100m=alignment["n_ref_dist_gt_100m"],
                 rows=len(c), seconds=time.perf_counter() - t0)
    c["obs_idx"] = c["orig_obs_idx"].astype(np.int64)
    c.drop(columns=["orig_obs_idx"],inplace=True)
    observations=pd.DataFrame({"trip_id":ctx["trip_id"],"seg_id":ctx["seg_id"],"obs_idx":ctx["original_obs_idx"],"ref_arc_id":ref_arc,"ref_dist":ref_dist})
    route_positions = alignment["route_positions"]
    steps = np.diff(route_positions, prepend=route_positions[0])
    questionable = (ref_dist > 50.0) | (steps > 8)
    alignment_diagnostics = pd.DataFrame({
        "trip_id": ctx["trip_id"], "obs_idx": ctx["original_obs_idx"][questionable],
        "route_position": route_positions[questionable], "ref_arc_id": ref_arc[questionable],
        "ref_dist_m": ref_dist[questionable],
        "distance_over_50m": ref_dist[questionable] > 50.0,
        "distance_over_100m": ref_dist[questionable] > 100.0,
        "route_position_jump_over_max_skip": steps[questionable] > 8,
    })
    return c, stats, observations, alignment_diagnostics, transition_diagnostics


def save_table(df, path):
    try:
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ModuleNotFoundError):
        csv_path=path.with_suffix('.csv')
        df.to_csv(csv_path,index=False)
        return csv_path


def verify_geometry_dependencies():
    """Fail before building data if declared exact-geometry libraries are unusable."""
    try:
        import shapely
        from shapely.geometry import LineString, Point
        from shapely.strtree import STRtree
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError("Dataset construction requires working Shapely 2 and PyProj installations") from exc
    line = LineString([(0.0, 0.0), (1000.0, 0.0)])
    if not Point(500.0, 1.0).dwithin(line, 2.0):
        raise RuntimeError("Shapely dwithin geometry self-check failed")
    tree = STRtree([line])
    if len(tree.query(Point(500.0, 1.0), predicate="dwithin", distance=2.0)) != 1:
        raise RuntimeError("Shapely STRtree dwithin self-check failed")
    tf = Transformer.from_crs(4326, 32643, always_xy=True)
    x, y = tf.transform(77.0, 13.0)
    lon, lat = Transformer.from_crs(32643, 4326, always_xy=True).transform(x, y)
    if abs(lon - 77.0) > 1e-7 or abs(lat - 13.0) > 1e-7:
        raise RuntimeError("PyProj axis-order / round-trip self-check failed")
    return {"pyproj": importlib.metadata.version("pyproj"), "shapely": importlib.metadata.version("shapely")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["train", "student_test"])
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--output-dir", default="data/processed",
                    help="output directory; use a versioned path to preserve previous datasets")
    a = ap.parse_args()
    geometry_versions = verify_geometry_dependencies()
    cfg = load_cfg()
    out_dir = Path(a.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for part in a.parts:
        part_dir=ROOT/"data"/"raw"/part
        if not part_dir.exists(): part_dir=ROOT/"data"/part
        if not part_dir.exists(): raise FileNotFoundError(f"No data directory for {part}: checked data/raw/{part} and data/{part}")
        trips = sorted(p for p in part_dir.iterdir() if p.is_dir())
        print(f"[{part}] {len(trips)} trips, config {cfg}")
        t0 = time.perf_counter()
        res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(process)(p, cfg) for p in trips)
        df = pd.concat([r[0] for r in res], ignore_index=True)
        stats = pd.DataFrame([r[1] for r in res])
        observations=pd.concat([r[2] for r in res],ignore_index=True)
        alignment_diagnostics=pd.concat([r[3] for r in res],ignore_index=True)
        transition_diagnostics=pd.concat([r[4] for r in res],ignore_index=True)
        candidate_path=save_table(df,out_dir / f"candidates_{part}.parquet")
        save_table(observations,out_dir / f"observations_{part}.parquet")
        stats.to_csv(out_dir / f"candidate_stats_{part}.csv", index=False)
        alignment_diagnostics.to_csv(out_dir / f"route_alignment_questionable_{part}.csv", index=False)
        transition_diagnostics.to_csv(out_dir / f"route_transition_diagnostics_{part}.csv", index=False)
        pts = stats["n_points"].sum()
        miss = (stats["candidate_miss_rate"] * stats["n_points"]).sum() / pts
        print(f"[{part}] {len(df):,} candidate rows at {candidate_path.name}, {pts:,} points, positives {df['label'].mean():.3f}, "
              f"candidate miss rate {miss:.2f}%, {time.perf_counter() - t0:.0f}s")

    (out_dir / "feature_list.json").write_text(json.dumps(FEATURES, indent=1))
    (out_dir / "dataset_config.json").write_text(json.dumps({
        "candidate_config": cfg, "features": FEATURES,
        "projection": "per-trip UTM via PyProj; missing dependency is an error",
        "candidate_search": "Shapely 2 STRtree dwithin; exact exhaustive segment-distance fallback",
        "geometry_dependencies": geometry_versions,
        "label_alignment": "existing monotone route dynamic program; questionable distances and disconnected route transitions are diagnosed, not relabeled",
        "alignment_questionable_distance_threshold_m": 50.0,
        "evaluator_test_touched": False,
    }, indent=2) + "\n")
    print(f"wrote {out_dir / 'feature_list.json'}")


if __name__ == "__main__":
    main()
