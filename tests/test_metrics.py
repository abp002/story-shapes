import numpy as np
import pytest

from story_shapes.metrics import direction_accuracy, loo_bandwidth, nearest_shape, pearson, spearman

SHAPES = {
    "rise": [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
    "fall": [4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 0, 0],
    "icarus": [1, 1, 2, 3, 4, 4, 4, 3, 2, 1, 0, 0],
}


def test_direction_counts_agreeing_signs():
    assert direction_accuracy([0.5, -0.2, 0.1, -0.9], [1, -1, -1, -0.5]) == 0.75


def test_a_neutral_prediction_is_never_a_hit():
    assert direction_accuracy([0.0, 0.0], [1, -1]) == 0


def test_spearman_is_one_for_any_increasing_relation():
    assert spearman([1, 2, 3, 4], [10, 20, 400, 800]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)


def test_spearman_uses_average_ranks_for_ties():
    # ranks with ties: x -> [1.5, 1.5, 3, 4]; y -> [1, 2, 3, 4]
    expected = np.corrcoef([1.5, 1.5, 3, 4], [1, 2, 3, 4])[0, 1]
    assert spearman([0, 0, 1, 2], [5, 6, 7, 8]) == pytest.approx(expected)


def test_pearson_matches_numpy():
    x, y = [1, 2, 3, 5], [2, 1, 4, 3]
    assert pearson(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_correlation_with_a_flat_series_is_zero_not_nan():
    assert pearson([1, 1, 1], [1, 2, 3]) == 0
    assert spearman([2, 2, 2], [1, 2, 3]) == 0


def test_nearest_shape_ignores_scale_and_offset():
    noisy_icarus = [0.1 + 0.3 * v for v in SHAPES["icarus"]]
    assert nearest_shape(noisy_icarus, SHAPES) == "icarus"
    assert nearest_shape(SHAPES["fall"], SHAPES) == "fall"


def test_bandwidth_for_pure_noise_is_the_widest():
    rng = np.random.default_rng(0)
    best, _ = loo_bandwidth(rng.normal(size=200), sigmas=[1, 3, 10, 30])
    assert best == 30


def test_bandwidth_for_a_clean_wave_is_the_narrowest():
    wave = np.sin(np.linspace(0, 6 * np.pi, 200))
    best, _ = loo_bandwidth(wave, sigmas=[1, 3, 10, 30])
    assert best == 1


def test_bandwidth_for_a_noisy_wave_is_in_between():
    rng = np.random.default_rng(1)
    wave = np.sin(np.linspace(0, 4 * np.pi, 300)) + rng.normal(scale=0.5, size=300)
    best, errors = loo_bandwidth(wave, sigmas=[0.5, 2, 6, 60])
    assert best in (2, 6)
    assert set(errors) == {0.5, 2, 6, 60}
