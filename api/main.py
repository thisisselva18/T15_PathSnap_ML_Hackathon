"""FastAPI app: GET /health, GET /trips (lists EVAL_DATA_DIR), POST /predict {trip_id}. Logs every request.

Owner: P2 (written by P1)

Run on the API laptop, from the repo root:
    $env:EVAL_DATA_DIR = "data\\eval"        # folder with one sub-folder per trip (.track/.nodes/.arcs/.route)
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Ground truth: predict_trip() never reads .route. api/evaluation.py reads it AFTER prediction to score the result,
and the response carries only scores, never the reference route.
"""
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.evaluation import metrics, reference
from pathsnap.pipeline import arcs_to_latlon, gps_latlon, load_bundle, predict_trip

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(os.environ.get("EVAL_DATA_DIR", ROOT / "data" / "eval"))
if not EVAL_DIR.is_absolute():
    EVAL_DIR = ROOT / EVAL_DIR
LOG_FILE = ROOT / "results" / "api_log.txt"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
if not log.handlers:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)

BUNDLE = load_bundle()
log.info("startup: model=%s features=%d eval_dir=%s", BUNDLE["model_name"], len(BUNDLE["features"]), EVAL_DIR)

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def list_trips():
    """Trip folders in EVAL_DIR that contain a .track file. Nothing is hardcoded."""
    if not EVAL_DIR.is_dir():
        return []
    return sorted(d.name for d in EVAL_DIR.iterdir() if d.is_dir() and (d / f"{d.name}.track").exists())


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health(request: Request):
    trips = list_trips()
    log.info("GET /health from %s -> model=%s trips=%d", request.client.host, BUNDLE["model_name"], len(trips))
    return {"status": "ok", "model": BUNDLE["model_name"], "n_trips": len(trips), "eval_dir": str(EVAL_DIR)}


@app.get("/trips")
def trips(request: Request):
    t = list_trips()
    log.info("GET /trips from %s -> %d trips", request.client.host, len(t))
    return {"trips": t}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    caller = request.client.host
    if req.trip_id not in list_trips():  # also blocks path tricks like ../train/...
        log.warning("POST /predict from %s -> unknown trip %r (404)", caller, req.trip_id)
        raise HTTPException(status_code=404, detail=f"unknown trip {req.trip_id}")
    trip_dir = EVAL_DIR / req.trip_id
    t0 = time.perf_counter()
    try:
        pred, ctx = predict_trip(trip_dir, BUNDLE)        # prediction: no .route involved
        t1 = time.perf_counter()
        m = metrics(pred, reference(trip_dir, ctx))      # scoring: server-side only
        pred["timings_s"]["scoring"] = round(time.perf_counter() - t1, 3)
        pred["timings_s"]["with_scoring"] = round(time.perf_counter() - t0, 3)
    except Exception as e:  # noqa: BLE001
        log.exception("POST /predict from %s trip %s failed", caller, req.trip_id)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    fmt = lambda x: "n/a" if x is None else f"{x:.4f}"  # noqa: E731
    log.info("POST /predict from %s trip=%s points=%d segments=%d breaks=%d decoder=%s timings=%s "
             "top1_hmm=%s top1_clf=%s continuity_hmm=%s",
             caller, req.trip_id, pred["n_points"], pred["n_segments"], pred["n_hmm_breaks"], pred["decoder"],
             pred["timings_s"], fmt(m.get("hmm_top1")), fmt(m.get("clf_top1")), fmt(m.get("continuity_hmm")))
    return {
        "trip_id": pred["trip_id"], "model": pred["model"], "decoder": pred["decoder"],
        "n_points": pred["n_points"], "n_with_candidates": pred["n_with_candidates"],
        "n_segments": pred["n_segments"], "n_hmm_breaks": pred["n_hmm_breaks"],
        "timings_s": pred["timings_s"], "metrics": m,
        "gps": gps_latlon(ctx),
        "predicted_route": arcs_to_latlon(pred["route_segments"], ctx),
    }
