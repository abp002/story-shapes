import numpy as np
import pytest

from story_shapes.curve import mean_level, smooth, spread, to_unit


def test_mean_level_of_a_certain_answer_is_that_level():
    assert mean_level({"0": 0.0, "1": 0.0, "2": 1.0}) == 2


def test_mean_level_does_not_depend_on_key_order():
    assert mean_level({"2": 0.5, "0": 0.5}) == mean_level({"0": 0.5, "2": 0.5}) == 1


def test_mean_level_renormalises_rounded_probabilities():
    # the API rounds probabilities, so they may not add up to exactly 1
    assert mean_level({"0": 1, "1": 1}) == 0.5


def test_spread_is_zero_when_certain_and_grows_when_torn():
    assert spread({"0": 0, "1": 1, "2": 0}) == 0
    assert spread({"0": 0.5, "1": 0, "2": 0.5}) == pytest.approx(1)
    assert spread({"0": 0.25, "1": 0.5, "2": 0.25}) < spread({"0": 0.5, "1": 0, "2": 0.5})


def test_to_unit_maps_the_scale_onto_minus_one_to_one():
    assert [to_unit(level, 5) for level in (0, 2, 4)] == [-1, 0, 1]


def test_smooth_keeps_a_flat_curve_flat():
    assert np.allclose(smooth([0.3] * 25), 0.3)


def test_a_passage_with_zero_weight_does_not_pull_the_curve():
    values = [0.0] * 21
    values[10] = 100.0
    weights = [1.0] * 21
    weights[10] = 0.0
    assert np.allclose(smooth(values, weights), 0.0)


def test_smooth_pulls_neighbours_towards_a_spike():
    values = [0.0] * 21
    values[10] = 1.0
    curve = smooth(values, sigma=2)
    assert len(curve) == 21
    assert curve[10] > curve[9] > curve[5] > 0
