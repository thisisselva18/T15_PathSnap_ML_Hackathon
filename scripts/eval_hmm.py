"""Classifier-only vs classifier + HMM/Viterbi -> results/hmm_comparison.csv (+ summary JSON and a figure).

    python scripts/eval_hmm.py                      # student-test, config from artifacts/hmm_config.json
    python scripts/eval_hmm.py --defaults           # same, with the untuned decoder defaults
    python scripts/eval_hmm.py --tune               # Bayesian (TPE) decoder tuning on 12 TRAIN trips, then
                                                    # writes artifacts/hmm_config.json and re-runs student-test

Prediction goes through pathsnap.pipeline.predict_trip, exactly like the API. The reference per point comes from
pathsnap.labels.align_points_to_route (the same alignment as the training labels) and is used only for scoring.

Top-1: over ALL GPS points; a point with no prediction counts as wrong.
Continuity: share of consecutive matched points whose arcs are the same, directly connected, or joined by a network
path of at most 2 x GPS displacement + 50 m (pathsnap.hmm.continuity).

Owner: P3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap import hmm  # noqa: E402
from pathsnap.io import load_trip  # noqa: E402
from pathsnap.labels import align_points_to_route  # noqa: E402
from pathsnap.pipeline import load_bundle, predict_trip  # noqa: E402

SEED = 42
RELIABLE_M = 50.0
RESULTS = ROOT / "results"
MIN_MOVE_CHOICES = [0.0, 2.0, 5.0, 10.0]


def trip_dirs(split):
    for base in (ROOT / "data" / "raw" / split, ROOT / "data" / split):
        if base.is_dir():
            return sorted(d for d in base.iterdir() if (d / f"{d.name}.track").exists())
    raise FileNotFoundError(f"no trip folders for split {split!r} under data/raw or data/")


def reference(trip_dir, ctx):
    route = load_trip(trip_dir, with_route=True)["route"]
    ref_arc, ref_dist = align_points_to_route(ctx["txy"], ctx["t"], ctx["seg_id"], route, ctx["A"], ctx["B"])
    return np.asarray(route), ref_arc, ref_dist


def candidate_hit(ctx, ref_arc):
    """Per point: is the reference arc among its candidates? (the ceiling for any decoder)"""
    c = ctx["cands"]
    hit = np.zeros(len(ref_arc), dtype=bool)
    m = c["arc_id"].to_numpy() == ref_arc[c["obs_idx"].to_numpy()]
    hit[c["obs_idx"].to_numpy()[m]] = True
    return hit


def route_pr(routes, route):
    pred, ref = {a for r in routes for a in r}, set(route.tolist())
    hit = len(pred & ref)
    return (hit / len(pred) if pred else np.nan), (hit / len(ref) if ref else np.nan)


def evaluate_trip(trip_dir, bundle, cfg):
    t0 = time.perf_counter()
    pred, ctx = predict_trip(trip_dir, bundle, cfg)
    wall = time.perf_counter() - t0
    if not pred["decoder"].startswith("hmm"):
        raise RuntimeError(f"{trip_dir.name}: HMM did not run ({pred['decoder']})")
    route, ref, ref_dist = reference(trip_dir, ctx)
    clf, dec, g, seg = pred["clf_arcs"], pred["hmm_arcs"], ctx["graph"], ctx["seg_id"]

    # ablation: the same Viterbi with the classic geometry-only emission (no classifier)
    steps = hmm.build_lattice(ctx["cands"], ctx["proba"], ctx["txy"], seg, g, cfg.get("min_move_m", 0.0))
    geo, _ = hmm.viterbi(steps, len(ref), cfg, use_classifier=False)

    ok_clf, ok_hmm, ok_geo = clf == ref, dec == ref, geo == ref
    rel = ref_dist <= RELIABLE_M
    p_clf, r_clf = route_pr(hmm.route_from_points(clf, seg, g), route)
    p_hmm, r_hmm = route_pr(pred["route_segments"], route)
    return {
        "trip": trip_dir.name, "points": pred["n_points"], "segments": pred["n_segments"],
        "breaks": pred["n_hmm_breaks"], "points_no_cands": pred["n_points"] - pred["n_with_candidates"],
        "cand_hit_rate": float(candidate_hit(ctx, ref).mean()),
        "top1_clf": float(ok_clf.mean()), "top1_hmm": float(ok_hmm.mean()), "top1_hmm_geom_only": float(ok_geo.mean()),
        "top1_clf_reliable": float(ok_clf[rel].mean()) if rel.any() else np.nan,
        "top1_hmm_reliable": float(ok_hmm[rel].mean()) if rel.any() else np.nan,
        "fixed_by_hmm": int((~ok_clf & ok_hmm).sum()), "broken_by_hmm": int((ok_clf & ~ok_hmm).sum()),
        "continuity_clf": pred["continuity_clf"], "continuity_hmm": pred["continuity_hmm"],
        "route_precision_clf": p_clf, "route_recall_clf": r_clf,
        "route_precision_hmm": p_hmm, "route_recall_hmm": r_hmm,
        "n_route_pieces_hmm": len(pred["route_segments"]),
        "time_prepare_s": pred["timings_s"]["prepare"], "time_classify_s": pred["timings_s"]["classify"],
        "time_hmm_s": pred["timings_s"]["hmm"], "time_total_s": round(wall, 3),
    }


def summarise(df):
    w = df["points"]
    pw = lambda col: float(np.average(df[col], weights=w))
    out = {"trips": len(df), "points": int(w.sum()),
           "breaks": int(df["breaks"].sum()), "fixed_by_hmm": int(df["fixed_by_hmm"].sum()),
           "broken_by_hmm": int(df["broken_by_hmm"].sum()),
           "trips_improved": int((df["top1_hmm"] > df["top1_clf"]).sum()),
           "trips_worse": int((df["top1_hmm"] < df["top1_clf"]).sum())}
    for col in ["cand_hit_rate", "top1_clf", "top1_hmm", "top1_hmm_geom_only", "top1_clf_reliable",
                "top1_hmm_reliable", "continuity_clf", "continuity_hmm", "route_precision_clf", "route_recall_clf",
                "route_precision_hmm", "route_recall_hmm"]:
        out[f"{col}_pointweighted"] = round(pw(col), 4)
        out[f"{col}_tripmean"] = round(float(df[col].mean()), 4)
    out["time_total_s_mean"] = round(float(df["time_total_s"].mean()), 2)
    out["time_hmm_s_mean"] = round(float(df["time_hmm_s"].mean()), 2)
    return out


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.sort_values("top1_clf")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - 0.2, d["top1_clf"], 0.4, label="Classifier only (MLP)", color="#9aa5b1")
    ax.bar(x + 0.2, d["top1_hmm"], 0.4, label="Classifier + HMM/Viterbi", color="#2563eb")
    ax.scatter(x, d["cand_hit_rate"], marker="_", s=300, color="black", label="Candidate ceiling", zorder=3)
    ax.set_xticks(x, d["trip"].str[-2:])
    ax.set_xlabel("Student-test trip (last two digits)")
    ax.set_ylabel("Top-1 accuracy (all points)")
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_eval(split, bundle, cfg, out_csv, tag):
    rows = []
    for d in trip_dirs(split):
        r = evaluate_trip(d, bundle, cfg)
        rows.append(r)
        print(f"{r['trip']}  pts {r['points']:5d}  seg {r['segments']}  breaks {r['breaks']}  "
              f"top1 clf {r['top1_clf']:.3f} -> hmm {r['top1_hmm']:.3f} (geom-only {r['top1_hmm_geom_only']:.3f}, "
              f"ceiling {r['cand_hit_rate']:.3f})  cont {r['continuity_clf']:.3f} -> {r['continuity_hmm']:.3f}  "
              f"{r['time_total_s']:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.4f")
    summary = summarise(df)
    summary.update(split=split, config=tag, model=bundle["model_name"], hmm_cfg=cfg)
    out_csv.with_name(out_csv.stem + "_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "hmm_cfg"}, indent=2))
    return df


# ==================================================================== decoder tuning (train trips only)
def tune(bundle, n_trials, timeout, n_trips):
    import optuna
    dirs = trip_dirs("train")
    pick = np.random.default_rng(SEED).choice(len(dirs), size=min(n_trips, len(dirs)), replace=False)
    dirs = [dirs[i] for i in sorted(pick)]
    print("tuning trips:", [d.name for d in dirs], flush=True)

    base = hmm.load_hmm_cfg(path=Path("__defaults__"))       # pure defaults, ignoring any saved config
    cache = []
    for d in dirs:                                           # prepare + classify + graph ONCE per trip
        _, ctx = predict_trip(d, bundle, base)
        _, ref, _ = reference(d, ctx)
        cache.append({"ctx": ctx, "ref": ref, "lattice": {}})

    def score(cfg):
        accs = []
        for item in cache:
            ctx, mm = item["ctx"], float(cfg["min_move_m"])
            if mm not in item["lattice"]:
                item["lattice"][mm] = hmm.build_lattice(ctx["cands"], ctx["proba"], ctx["txy"], ctx["seg_id"],
                                                        ctx["graph"], mm)
            arcs, _ = hmm.viterbi(item["lattice"][mm], len(item["ref"]), cfg)
            accs.append(float((arcs == item["ref"]).mean()))
        return float(np.mean(accs)), accs

    def objective(trial):
        cfg = dict(base,
                   beta_m=trial.suggest_float("beta_m", 1.0, 50.0, log=True),
                   sigma_m=trial.suggest_float("sigma_m", 3.0, 30.0, log=True),
                   dist_weight=trial.suggest_float("dist_weight", 0.0, 1.0),
                   uturn_penalty_m=trial.suggest_float("uturn_penalty_m", 0.0, 200.0),
                   min_move_m=trial.suggest_categorical("min_move_m", MIN_MOVE_CHOICES))
        mean, accs = score(cfg)
        trial.set_user_attr("per_trip_top1", [round(a, 4) for a in accs])
        return mean

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="hmm_decoder")
    study.enqueue_trial({k: base[k] for k in ("beta_m", "sigma_m", "dist_weight", "uturn_penalty_m", "min_move_m")},
                        user_attrs={"trial_role": "untuned_default_baseline"})
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   callbacks=[lambda s, t: print(f"trial {t.number:3d}  top1 {t.value:.4f}  best {s.best_value:.4f}",
                                                 flush=True)])
    tune_s = time.perf_counter() - t0

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration", "state"))
    trials.to_csv(RESULTS / "optuna_trials_hmm.csv", index=False)
    default_value = study.trials[0].value
    best = dict(base, **study.best_params)
    saved = {k: best[k] for k in hmm.DEFAULTS}
    saved["_tuning"] = {
        "objective": "mean per-trip HMM Top-1 on train trips (all points)", "trips": [d.name for d in dirs],
        "trip_selection": f"numpy default_rng({SEED}).choice, {len(dirs)} of 80 train trips",
        "sampler": {"name": "TPESampler", "seed": SEED, "n_startup_trials": 4, "multivariate": True},
        "trial0": "untuned defaults", "n_trials": len(study.trials), "tuning_time_s": round(tune_s, 1),
        "default_top1": round(default_value, 4), "best_top1": round(study.best_value, 4),
        "best_trial": study.best_trial.number, "model": bundle["model_name"],
        "student_test_used_for_tuning": False,
    }
    hmm.CFG_PATH.write_text(json.dumps(saved, indent=2))
    print(f"defaults {default_value:.4f} -> best {study.best_value:.4f} (trial {study.best_trial.number}); "
          f"wrote {hmm.CFG_PATH.relative_to(ROOT)}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="student_test")
    ap.add_argument("--defaults", action="store_true", help="ignore artifacts/hmm_config.json")
    ap.add_argument("--tune", action="store_true", help="tune the decoder on train trips first")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--tune-trips", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    bundle = load_bundle()
    print(f"model: {bundle['model_name']}", flush=True)
    if args.tune:
        tune(bundle, args.n_trials, args.timeout, args.tune_trips)
    if args.defaults:
        cfg, tag = hmm.load_hmm_cfg(path=Path("__defaults__")), "defaults"
    else:
        cfg, tag = hmm.load_hmm_cfg(), "artifacts/hmm_config.json"
    out = args.out or RESULTS / ("hmm_comparison_defaults.csv" if args.defaults else "hmm_comparison.csv")
    df = run_eval(args.split, bundle, cfg, out, tag)
    if not args.defaults and args.split == "student_test":
        plot(df, RESULTS / "figures" / "hmm_vs_classifier_top1.png")


if __name__ == "__main__":
    main()
