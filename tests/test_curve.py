import numpy as np
import pytest

from story_shapes.curve import death_cut, mean_level, smooth, spread, to_unit


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



def test_a_lasting_death_ends_the_curve_where_it_starts():
    assert death_cut([0.0] * 10 + [0.9, 0.95, 0.7, 0.8, 0.9]) == 10


def test_a_single_passage_of_death_that_does_not_last_is_ignored():
    # Scrooge sees his own grave, then wakes up
    assert death_cut([0.0] * 10 + [0.95] + [0.05] * 6) is None


def test_a_faked_death_followed_by_a_real_one_cuts_at_the_real_one():
    dead = [0.0] * 5 + [0.85, 0.3, 0.1, 0.0, 0.0] + [0.0] * 5 + [0.9, 0.95, 0.99]
    assert death_cut(dead) == 15


def test_a_death_in_the_last_passage_counts():
    assert death_cut([0.0] * 8 + [0.97]) == 8


def test_no_death_no_cut():
    assert death_cut([0.1] * 20) is None


def test_a_death_undone_by_the_protagonist_acting_alive_is_revoked():
    dead = [0.0] * 3 + [0.9, 0.9, 0.85] + [0.05, 0.0, 0.0, 0.0]
    present = [1.0] * 3 + [0.9, 0.9, 0.9] + [0.95, 0.9, 0.9, 0.9]
    assert death_cut(dead, present=present, revoke=True) is None
    assert death_cut(dead, present=present, revoke=False) == 3


def test_a_death_followed_by_absence_is_not_revoked():
    # Gregor after his death: not "dead" in each passage, but not acting alive either
    dead = [0.0] * 3 + [0.9, 0.8, 0.85, 0.9] + [0.05, 0.05, 0.4]
    present = [1.0] * 3 + [0.9, 0.6, 0.6, 0.5] + [0.1, 0.2, 0.3]
    assert death_cut(dead, present=present, revoke=True) == 3
