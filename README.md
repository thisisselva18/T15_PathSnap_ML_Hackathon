# PathSnap - ML Hackathon

Map-matching noisy GPS trajectories to a road network (Team T15).

## Setup
```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Data placement
- `data/raw/train/<trip_id>/` : 80 training trips (`.track .nodes .arcs .route`)
- `data/raw/student_test/<trip_id>/` : 10 student-testing trips
- `data/eval/<trip_id>/` : evaluator trips, placed on Laptop 1 on demo day only (never committed)

## Pipeline
1. `python scripts/build_dataset.py` -> `data/processed/candidates_*.parquet`
2. `python scripts/train_models.py --model <name>` -> `results/experiments.csv`
3. `python scripts/tune_bayes.py` -> `results/optuna.db`, `artifacts/model.joblib`
4. `python scripts/eval_hmm.py` -> `results/hmm_comparison.csv`
5. `python scripts/make_plots.py` and `python scripts/make_maps.py` -> `results/figures`, `results/maps`

## Run the demo
Laptop 1 (API):
```
set EVAL_DATA_DIR=data\eval
uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info
```
Laptop 2 (UI):
```
streamlit run ui/app.py
```
Set the API URL in the UI sidebar to `http://<laptop1-ip>:8000`.

## AI assistance
(Disclose AI tool usage here as required by the hackathon policy.)
