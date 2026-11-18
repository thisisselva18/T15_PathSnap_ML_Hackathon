"""Streamlit UI: API URL in sidebar, trip dropdown from /trips, run -> Folium map (GPS blue, predicted red) + metric cards.

Owner: P1

Run from the repo root on the UI laptop:
    streamlit run ui/app.py

The UI talks ONLY to the API. It holds no model, no trip data and no ground truth: the trip list comes from
GET /trips, every prediction and metric from POST /predict, and the response never contains the reference route.

Expected /predict response (agreed with P2):
    trip_id, model, n_points, n_with_candidates, n_segments, n_hmm_breaks,
    timings_s {prepare, classify, hmm, predict_total, with_scoring},
    metrics {ground_truth_available, hmm_top1, clf_top1, continuity_hmm, continuity_clf,
             route_precision, route_recall},
    gps [[lat, lon], ...], predicted_route [[[lat, lon], ...], ...]
"""
import json
import time
from pathlib import Path

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

SAMPLE = Path(__file__).with_name("sample_response.json")
MAX_GPS_MARKERS = 2000

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: map-matching evaluation")


# ---------------------------------------------------------------- API helpers
def api_get(base, path):
    r = requests.get(f"{base}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def trip_list(payload):
    """/trips may return a plain list or {"trips": [...]}."""
    return payload.get("trips", []) if isinstance(payload, dict) else list(payload)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Connection")
    base = st.text_input("API URL", value="http://127.0.0.1:8000").rstrip("/")
    demo = st.checkbox("Use demo data (UI development only)", value=False,
                       help="Loads ui/sample_response.json instead of calling the API. Never use during evaluation.")

    if not demo:
        try:
            health = api_get(base, "/health")
            st.success(f"API reachable. Model: {health.get('model', '?')}")
            trips = trip_list(api_get(base, "/trips"))
            st.caption(f"{len(trips)} trips available")
        except Exception as e:
            st.error(f"Cannot reach the API at {base}\n\n{type(e).__name__}: {e}")
            st.stop()
    else:
        trips = []
        st.warning("Demo mode: showing a saved sample response, not a live prediction.")

# ---------------------------------------------------------------- trip choice + run
if demo:
    if not SAMPLE.exists():
        st.error(f"{SAMPLE.name} not found next to app.py.")
        st.stop()
    if st.button("Load demo response", type="primary"):
        st.session_state["result"] = json.loads(SAMPLE.read_text())
        st.session_state["round_trip_s"] = None
else:
    if not trips:
        st.warning("The API returned no trips. Check EVAL_DATA_DIR on the API laptop.")
        st.stop()
    trip_id = st.selectbox("Evaluation trip", trips)
    if st.button("Run map-matching", type="primary"):
        with st.spinner(f"Running trip {trip_id} on the API..."):
            t0 = time.perf_counter()
            try:
                r = requests.post(f"{base}/predict", json={"trip_id": trip_id}, timeout=600)
                r.raise_for_status()
            except Exception as e:
                st.error(f"/predict failed for trip {trip_id}: {type(e).__name__}: {e}")
                st.stop()
            st.session_state["result"] = r.json()
            st.session_state["round_trip_s"] = time.perf_counter() - t0

res = st.session_state.get("result")
if not res:
    st.info("Pick a trip and press **Run map-matching**.")
    st.stop()

# ---------------------------------------------------------------- metric cards
m = res.get("metrics", {})
rt = st.session_state.get("round_trip_s")


def pct(x):
    return "N/A" if x is None else f"{100 * x:.1f}%"


st.subheader(f"Trip {res.get('trip_id')} - model {res.get('model', '?')}")
decoder = res.get("decoder", "hmm_viterbi")
if not decoder.startswith("hmm"):
    st.info(f"Decoder: {decoder}. The 'classifier + HMM' figures equal classifier-only until the HMM is deployed.")
if not m.get("ground_truth_available", False):
    st.warning("No reference route on the server for this trip: accuracy metrics are not available.")

row1 = st.columns(4)
row1[0].metric("Top-1 (classifier + HMM)", pct(m.get("hmm_top1")))
row1[1].metric("Top-1 (classifier only)", pct(m.get("clf_top1")),
               delta=None if m.get("hmm_top1") is None or m.get("clf_top1") is None
               else f"HMM {100 * (m['hmm_top1'] - m['clf_top1']):+.1f} pts", delta_color="off")
row1[2].metric("Route continuity (HMM)", pct(m.get("continuity_hmm")))
row1[3].metric("Route continuity (classifier)", pct(m.get("continuity_clf")))

row2 = st.columns(4)
row2[0].metric("Route precision / recall", f"{pct(m.get('route_precision'))} / {pct(m.get('route_recall'))}")
row2[1].metric("GPS points", f"{res.get('n_points', 0):,}",
               help=f"{res.get('n_with_candidates', '?')} with candidates")
row2[2].metric("Segments / HMM breaks", f"{res.get('n_segments', '?')} / {res.get('n_hmm_breaks', '?')}")
row2[3].metric("Response time", "N/A (demo)" if rt is None else f"{rt:.2f} s",
               help=f"Server-side predict: {res.get('timings_s', {}).get('predict_total', '?')} s")

# ---------------------------------------------------------------- map
gps = res.get("gps") or []
route = res.get("predicted_route") or []
if gps:
    fmap = folium.Map(tiles="OpenStreetMap", control_scale=True)

    pts = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(gps) // MAX_GPS_MARKERS)
    for lat, lon in gps[::step]:
        folium.CircleMarker([lat, lon], radius=2, color="#1f5fd6", fill=True, fill_opacity=0.8, weight=0).add_to(pts)
    pts.add_to(fmap)

    if route:
        pr = folium.FeatureGroup(name="Predicted route", show=True)
        folium.PolyLine(route, color="#d7191c", weight=4, opacity=0.85).add_to(pr)
        pr.add_to(fmap)

    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl(collapsed=False).add_to(fmap)

    st_folium(fmap, height=620, use_container_width=True, returned_objects=[])
    st.caption("Blue dots: raw GPS observations. Red line: predicted route (classifier + HMM/Viterbi). "
               "The reference route stays on the server and is never sent to the UI.")
else:
    st.warning("The response contained no GPS coordinates to draw.")

with st.expander("Raw timings and counts"):
    st.json({k: v for k, v in res.items() if k not in ("gps", "predicted_route")})
