import numpy as np
import optuna

from scripts.tune_bayes import DEFAULTS, MLP_SHAPES, make_pipeline, parameter_space
from scripts.train_models import fit_mlp_epochs, top1


def test_tpe_supports_required_startup_and_multivariate_options():
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=4, multivariate=True)
    assert sampler._n_startup_trials == 4
    assert sampler._multivariate is True


def test_default_trial_parameters_match_untuned_mlp():
    trial = optuna.trial.FixedTrial(DEFAULTS["mlp"])
    assert parameter_space(trial) == DEFAULTS["mlp"]
    model = make_pipeline(DEFAULTS["mlp"])
    estimator = model[-1]
    assert estimator.hidden_layer_sizes == MLP_SHAPES["64-32"]
    assert estimator.alpha == 1e-4
    assert estimator.learning_rate_init == 1e-3
    assert estimator.early_stopping is False


def test_tuned_pipeline_fits_and_returns_probability_scores():
    params = {"hidden": "64", "alpha": 1e-4, "learning_rate_init": 1e-3}
    model = make_pipeline(params)
    x = np.array([[0.0], [0.2], [0.8], [1.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    fit_mlp_epochs(model, x, y, 2)
    assert model.predict(x).shape == (4,)
    assert model.predict_proba(x).shape == (4, 2)


def test_top1_manifest_counts_no_candidate_observations_as_incorrect():
    import pandas as pd

    candidates = pd.DataFrame({
        "trip_id": ["a"], "obs_idx": [0], "arc_id": [11], "ref_arc_id": [11],
    })
    manifest = pd.DataFrame({"trip_id": ["a", "a"], "obs_idx": [0, 1]})
    assert top1(candidates, np.array([0.9]), manifest) == 0.5
