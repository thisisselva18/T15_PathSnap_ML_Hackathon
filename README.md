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

Processed datasets are generated locally under `data/processed/` and are not
committed to Git. Rebuild them with the dataset command below when needed.

## Pipeline
1. `python scripts/build_dataset.py` -> candidate-level train and student-test tables in `data/processed/`
2. `python scripts/train_models.py --model <name> --laptop L2` -> grouped CV and evaluation outputs
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

## Selected classifier: MLP

The processing code accepts `data/raw/<partition>/<trip_id>/` and, for the current
checkout layout, `data/<partition>/<trip_id>/`. It preserves source observation
indices after dropping invalid GPS rows, projects each trip to metric coordinates,
segments large gaps, crops the network, generates up to the configured number of
nearby arc candidates, engineers the shared causal feature schema, and aligns
observations monotonically to the ordered `.route` arc IDs. Candidate labels are
binary: `1` for the aligned reference arc, `0` for other candidates. Training rows
more than 50 m from their aligned reference are filtered as unreliable; student
test rows are not filtered for model selection.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/build_dataset.py
python scripts/train_models.py --model mlp --laptop L2 --max-rows 150000
python scripts/evaluate_models.py
python -m pytest -q
```

MLP is the selected classifier. KNN was discontinued by the team. Its source,
saved models, predictions, metrics, plots, and experiment rows are retained as
historical results, but KNN is excluded from the active training/evaluation CLI.

The first model run creates the deterministic shared trip-fold map in
`artifacts/folds.json`; keep it for subsequent runs. Sampling is reproducible and
selects complete observation candidate groups, so the actual row count can be
slightly below the requested cap. The exact sample is saved as
`data/processed/candidates_train_sample.parquet` (CSV if no Parquet engine is
installed), with its seed and row count in `training_sample_config.json`.

The builder writes `candidates_train` and `candidates_student_test`, per-observation
label manifests, and per-trip validation/candidate statistics under
`data/processed/` and `results/`. Model artifacts are saved in
`artifacts/models/<run-tag>/`; predictions, confusion matrices, MLP training
history, and experiment rows are saved under `results/<run-tag>/`.

The feature builder uses only the current and prior observations; it does not use
future observations or route labels as prediction features. Route labels are used
only by the offline builder and evaluator. The corrected MLP disables
scikit-learn's row-random early stopping and selects its iteration count from
held-out training trips. The corrected builder requires PyProj for metric
projection and fails clearly when it is unavailable. Shapely 2 STRtree is used
for exact radius queries; the dependency-free candidate fallback checks exact
segment distances exhaustively. CSV output remains available without a Parquet
engine.

`scripts/evaluate_models.py` does not retrain. It loads the tagged MLP artifact,
recomputes probabilities on the current student-test candidate table, and checks
that they agree with the saved `results/<run-tag>/preds/mlp_student_test` file
before scoring. It writes classifier metrics and per-class reports under
`results/<run-tag>/metrics/`, OvR ROC and confusion-matrix PNGs under
`results/<run-tag>/plots/`, and candidate miss/coverage statistics separately.
The classifier uses binary candidate
correctness; their macro ROC-AUC is the mean of defined per-class OvR AUCs and
micro ROC-AUC pools the one-hot OvR decisions. If a test split lacks a class,
that class's ROC is undefined and omitted from the macro average.

### Corrected MLP baseline

Corrected datasets and outputs are versioned separately so the earlier experiment
remains intact. From the repository root:

```powershell
python scripts/build_dataset.py --jobs 4 --output-dir data/processed/post_correction
python scripts/train_models.py --model mlp --laptop L2 --max-rows 150000 --data-dir data/processed/post_correction --run-tag mlp_baseline
python scripts/evaluate_models.py --run-tag mlp_baseline --data-dir data/processed/post_correction
python -m pytest -q
```

The grouped CV Top-1 denominator is every strided training observation with
reference distance at most 50 m in that held-out trip fold; observations with no
candidates count as incorrect. Student-test Top-1 uses the complete student-test
observation manifest and also counts no-candidate observations as incorrect.
MLP iteration count is selected using held-out training trips inside each outer
fold, then the final iteration count is the median selected count. Per-trip route
alignment and disconnected route-transition diagnostics are saved beside each
versioned dataset. Model artifacts go to `artifacts/models/<run-tag>/`, while
predictions, metrics, and plots go to `results/<run-tag>/`. Use an unused run tag
because training refuses to overwrite model or prediction artifacts.

MLP iteration counts are selected by minimum log loss on held-out trips inside
each outer CV training partition. The final epoch count is the median of the five
selected counts. These can be short when validation loss is best early; they are
not a convergence target and should not be increased without a separate tuning
decision. Training executes exactly the selected number of epochs with
`partial_fit`, avoiding sklearn's `max_iter` warning for an intentional stopping
checkpoint. The selected fold counts and final epoch loss history are saved with
the run artifacts.

### MLP Bayesian tuning and final selection

The team selected MLP as the sole active classifier. KNN remains in the
five-model untuned comparison as a historical result and is not tuned. The
LightGBM row and its earlier exploratory tuning outputs are historical; no new
LightGBM tuning was performed for this selection.

Reproduce the full search from the repository root with:

```powershell
python scripts/tune_bayes.py --model mlp --max-rows 150000 --n-trials 30 --timeout 900
```

The tuner refuses to overwrite an existing study or model. To repeat the run
while preserving the recorded result, use fresh output paths:

```powershell
python scripts/tune_bayes.py --model mlp --max-rows 150000 --n-trials 30 --timeout 900 --output-dir results/mlp_tuning_rerun --model-output artifacts/models/mlp_tuned_rerun.joblib
```

The tuner reads `data/processed/post_correction/`, samples complete eligible
observation groups up to 150,000 candidate rows using the shared seed and feature
schema, and optimizes mean observation-manifest Top-1 over three trip-grouped
folds. The inner epoch selection also holds out whole trips and minimizes
validation log loss. Observations without candidates count as incorrect. Student
test and evaluator-test data are excluded from the optimization objective; the
student-test partition is scored once after tuning for the predeclared MLP
selection rule. The evaluator-test partition is never loaded.

Optuna uses `TPESampler(seed=42, n_startup_trials=4, multivariate=True)`. Trial 0
is enqueued with the untuned defaults (`64-32`, `alpha=0.0001`,
`learning_rate_init=0.001`); trials 1–3 are startup samples and subsequent trials
use TPE proposals. The search space is hidden layer size `{64, 128, 64-32,
128-64}`, `alpha` in `[1e-6, 1e-2]` log-uniform, and initial learning rate in
`[1e-4, 1e-2]` log-uniform. SQLite study, trial log, best parameters, metrics,
student-test predictions, reports and plots are saved under
`results/mlp_tuning/`; the separate fitted pipeline is
`artifacts/models/mlp_tuned.joblib`. The selected classifier is installed at
`artifacts/model.joblib`. The prior LightGBM artifact is preserved locally at
`artifacts/models/lgbm_exploratory_preserved.joblib` and in Git history.

The recorded search completed 11 trials (0 failed). Best parameters were
`hidden=128`, `alpha=0.0006637202`, and `learning_rate_init=0.0001587732`.
On the same corrected 150k training sample and student-test population, untuned
MLP Top-1 was `0.821503` and tuned MLP Top-1 was `0.825783`; therefore tuned MLP
was selected. Candidate accuracy was `0.939541` vs `0.940069`, macro F1 was
`0.890912` vs `0.890138`, weighted F1 was `0.939286` vs `0.939328`, ROC-AUC was
`0.974966` vs `0.975345`, and log loss was `0.147097` vs `0.145167` (untuned vs
tuned). Train time was `4.78s` vs `46.10s`; student-test inference time was
`0.105s` vs `0.115s`. On the comparable three-fold
grouped CV, untuned-default vs best-trial Top-1 was `0.791490 ± 0.027450` vs
`0.792075 ± 0.028264`. The tuned run's 900-second Optuna
timeout was exceeded by an in-flight trial; total recorded search time was
999.7 seconds. The slowest individual trial took `205.91s`; 11 trials completed
and none failed. The saved database preserves all 11 trials and their durations.
Historical untuned KNN Top-1 in the five-model comparison is `0.814301`; no KNN
tuning was performed.

The MLP artifact follows the repository's `{model, features, ...}` joblib
wrapper and has been checked for load, prediction shape and probability output.
The current API and HMM modules are placeholders and contain no active model
loader, so end-to-end API/HMM integration is not verified by this repository
checkout.
