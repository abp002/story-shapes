import numpy as np
import pytest

from story_shapes.curve import confidence, death_cut, lag1_autocorrelation, mean_level, relative, smooth, spread, to_unit


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


def test_the_band_is_zero_when_every_passage_agrees_and_the_model_is_sure():
    line, half = confidence([0.4] * 30, [0.0] * 30)
    assert np.allclose(line, 0.4)
    assert np.allclose(half, 0.0)


def test_the_band_widens_when_the_model_is_torn():
    _, sure = confidence([0.0] * 30, [0.1] * 30)
    _, torn = confidence([0.0] * 30, [0.6] * 30)
    assert (torn > sure).all()


def test_the_band_widens_at_the_ends_where_fewer_passages_count():
    _, half = confidence([0.0] * 41, [0.5] * 41, sigma=4)
    assert half[0] > half[10] and half[-1] > half[20]


def test_a_passage_with_zero_weight_does_not_widen_the_band():
    values, weights = [0.0] * 21, [1.0] * 21
    values[10], weights[10] = 5.0, 0.0
    _, half = confidence(values, [0.2] * 21, weights)
    _, clean = confidence([0.0] * 21, [0.2] * 21, weights)
    assert np.allclose(half, clean)


def test_lag1_autocorrelation_sees_runs_and_ignores_alternation():
    assert lag1_autocorrelation([1, 1, 1, 1, -1, -1, -1, -1] * 3) > 0.5
    assert lag1_autocorrelation([1, -1] * 12) == 0.0


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_the_95_band_covers_the_true_curve_about_95_percent_of_the_time(rho):
    # passages drawn around a known curve, with noise that is independent or runs in streaks
    rng = np.random.default_rng(7)
    n, hits, total = 120, 0, 0
    truth = 0.4 * np.sin(np.linspace(0, 2 * np.pi, n))
    for _ in range(200):
        noise = np.zeros(n)
        for i in range(n):
            noise[i] = (rho * noise[i - 1] if i else 0) + rng.normal(0, 0.35)
        line, half = confidence(truth + noise, np.full(n, 0.3), sigma=6)
        inner = slice(10, n - 10)  # the smoothing itself bends the curve at the ends
        hits += (np.abs(line - truth)[inner] <= half[inner]).sum()
        total += n - 20
    assert 0.9 <= hits / total <= 0.995


def test_relative_centres_and_scales_the_line_and_moves_passages_and_band_with_it():
    raw = [-0.9, -0.7, -0.5, -0.7]
    line, values, half = relative(raw, [-1.0, -0.5, -0.5, -1.0], [0.1] * 4)
    assert line.mean() == pytest.approx(0) and line.std() == pytest.approx(1)
    assert line.argmax() == 2
    # a passage sitting on the curve stays on it
    assert values[2] == pytest.approx(line[2])
    assert np.allclose(half, 0.1 / np.std(raw))


def test_relative_leaves_a_flat_curve_flat():
    line, _, _ = relative([0.3] * 5, [0.3] * 5, [0.0] * 5)
    assert np.allclose(line, 0)
