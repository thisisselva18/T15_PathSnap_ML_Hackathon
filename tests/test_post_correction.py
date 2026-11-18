import numpy as np
import pandas as pd
import pytest
import warnings
from sklearn.exceptions import ConvergenceWarning

from pathsnap import candidates as candidate_module
from pathsnap.candidates import generate_candidates
from pathsnap.geo import crop_arc_ids, project_trip, point_segment
from pathsnap.labels import align_points_to_route
from scripts.evaluate_models import validate_saved_predictions
from scripts.train_models import build_model, fit_mlp_epochs, grouped_validation_indices, select_mlp_iterations, top1


def _exhaustive_expected(point, starts, ends, arc_ids, radius, k):
    distance, fraction, raw_fraction = point_segment(
        point[0], point[1], starts[arc_ids, 0], starts[arc_ids, 1], ends[arc_ids, 0], ends[arc_ids, 1]
    )
    order = np.lexsort((arc_ids, distance))
    keep = order[distance[order] <= radius][:k]
    return arc_ids[keep], distance[keep], fraction[keep], raw_fraction[keep]


def test_exact_fallback_finds_long_segment_missed_by_midpoint_shortlist(monkeypatch):
    # The long segment is 100 m from the observation but its midpoint is >5 km away.
    # Sixty non-candidate segment midpoints are closer, which defeated the old 64-midpoint query.
    starts = [[100.0, 0.0]]
    ends = [[10100.0, 0.0]]
    for i in range(5):
        starts.append([-0.5, 20.0 + i])
        ends.append([0.5, 20.0 + i])
    for i in range(60):
        starts.append([-0.5, 200.0 + i])
        ends.append([0.5, 200.0 + i])
    starts, ends = np.asarray(starts), np.asarray(ends)
    monkeypatch.setattr(candidate_module, "shapely", None)
    found = generate_candidates(np.asarray([[0.0, 0.0]]), starts, ends, np.arange(len(starts)), radius_m=150, k=8)
    expected = _exhaustive_expected(np.asarray([0.0, 0.0]), starts, ends, np.arange(len(starts)), 150, 8)
    assert found.arc_id.to_numpy().tolist() == expected[0].tolist()
    assert 0 in found.arc_id.to_numpy()


@pytest.mark.parametrize("use_fallback", [False, True])
def test_candidate_search_matches_exhaustive_for_crossing_nearby_and_edge_segments(monkeypatch, use_fallback):
    starts = np.asarray([[-1000.0, 0.0], [0.0, 4.0], [0.0, 5.0], [10.0, 10.0]])
    ends = np.asarray([[1000.0, 0.0], [10.0, 4.0], [10.0, 5.0], [10.0, 10.0]])
    points = np.asarray([[0.0, 1.0], [5.0, 4.0], [10.0, 10.0]])
    if use_fallback:
        monkeypatch.setattr(candidate_module, "shapely", None)
    found = generate_candidates(points, starts, ends, np.arange(4), radius_m=5, k=3)
    for obs_idx, point in enumerate(points):
        expected = _exhaustive_expected(point, starts, ends, np.arange(4), 5, 3)
        actual = found.loc[found.obs_idx == obs_idx]
        assert actual.arc_id.to_numpy().tolist() == expected[0].tolist()
        assert np.allclose(actual.dist, expected[1])


def test_candidate_search_empty_and_invalid_inputs(monkeypatch):
    monkeypatch.setattr(candidate_module, "shapely", None)
    empty = generate_candidates(np.asarray([[0.0, 0.0]]), np.empty((0, 2)), np.empty((0, 2)), np.asarray([], dtype=int))
    assert empty.empty
    with pytest.raises(ValueError, match="positive integer"):
        generate_candidates(np.asarray([[0.0, 0.0]]), np.zeros((1, 2)), np.ones((1, 2)), np.asarray([0]), k=0)


def test_projection_uses_pyproj_axis_order_and_metric_units():
    track = pd.DataFrame({"lon": [77.0, 77.001], "lat": [13.0, 13.0]})
    nodes = pd.DataFrame({"lon": [77.0], "lat": [13.0]})
    txy, nxy, epsg = project_trip(track, nodes)
    assert epsg == 32643
    assert np.allclose(txy[0], nxy[0])
    assert 100 < np.linalg.norm(txy[1] - txy[0]) < 120


def test_arc_crop_keeps_long_segment_crossing_expanded_track_box():
    nodes = np.asarray([[-1000.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [1100.0, 500.0]])
    keep = crop_arc_ids(nodes, np.asarray([0, 2]), np.asarray([1, 3]), np.asarray([[0.0, 0.0]]), margin_m=50)
    assert keep.tolist() == [0]


def test_route_alignment_validates_ids_and_returns_distance_diagnostics():
    points = np.asarray([[1.0, 1.0], [11.0, 1.0]])
    starts = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    ends = np.asarray([[10.0, 0.0], [20.0, 0.0]])
    labels, distances, diagnostics = align_points_to_route(
        points, np.asarray([0.0, 1.0]), np.asarray([0, 0]), np.asarray([0, 1]), starts, ends,
        return_diagnostics=True,
    )
    assert labels.tolist() == [0, 1]
    assert diagnostics["position_monotone"]
    assert diagnostics["n_ref_dist_gt_50m"] == 0
    with pytest.raises(ValueError, match="outside the arc geometry"):
        align_points_to_route(points, np.asarray([0.0, 1.0]), np.asarray([0, 0]), np.asarray([0, 2]), starts, ends)


def test_grouped_mlp_stopping_split_never_separates_same_trip():
    groups = np.repeat(np.asarray([f"trip_{i}" for i in range(10)]), 4)
    fit_idx, val_idx = grouped_validation_indices(groups, seed=17)
    assert set(groups[fit_idx]).isdisjoint(set(groups[val_idx]))
    rng = np.random.default_rng(7)
    X = rng.normal(size=(len(groups), 4))
    y = np.tile(np.asarray([0, 0, 0, 1]), 10)
    epochs, loss = select_mlp_iterations(X, y, groups, seed=17, max_iter=2, patience=1)
    assert 1 <= epochs <= 2
    assert np.isfinite(loss)


def test_mlp_selected_epochs_fit_without_false_max_iter_warning():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    y = np.tile(np.asarray([0, 0, 0, 1]), 20)
    model = build_model("mlp", {"max_iter": 3})[1]
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        fit_mlp_epochs(model, X, y, 3)
    assert not any(issubclass(item.category, ConvergenceWarning) for item in emitted)
    assert model[-1].n_iter_ == 3
    assert len(model[-1].loss_curve_) == 3
    assert np.isfinite(model.predict_proba(X)).all()


def test_top1_denominator_is_observation_manifest_including_missing_candidates():
    candidates = pd.DataFrame({"trip_id": ["a"], "obs_idx": [0], "arc_id": [7], "ref_arc_id": [7]})
    manifest = pd.DataFrame({"trip_id": ["a", "a"], "obs_idx": [0, 1]})
    assert top1(candidates, np.asarray([0.9]), manifest) == 0.5


def test_saved_prediction_consistency_check_rejects_changed_score():
    test = pd.DataFrame({"trip_id": ["a"], "obs_idx": [0], "arc_id": [2], "ref_arc_id": [2], "label": [1]})
    saved = test.assign(score=[0.75])
    validate_saved_predictions(saved, test, np.asarray([0.75]))
    with pytest.raises(ValueError, match="probabilities do not match"):
        validate_saved_predictions(saved, test, np.asarray([0.25]))
