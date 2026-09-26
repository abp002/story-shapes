"""Measures for the evaluation: agreement in direction, correlation, shape, smoothing bandwidth."""

import numpy as np


def direction_accuracy(predicted, truth) -> float:
    """Share of items where the prediction falls on the same side of neutral as the truth.

    Both are on signed scales (negative = ill fortune, positive = good). A prediction of exactly
    zero takes no side, so it never counts as a hit.
    """
    p, t = np.asarray(predicted, dtype=float), np.asarray(truth, dtype=float)
    return float(np.mean(p * t > 0))


def pearson(x, y) -> float:
    """Pearson correlation; 0 when either series is flat (no variation to correlate)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _ranks(values) -> np.ndarray:
    """Ranks starting at 1, with tied values sharing the average of their ranks."""
    values = np.asarray(values, dtype=float)
    order = values.argsort(kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x, y) -> float:
    return pearson(_ranks(x), _ranks(y))


def nearest_shape(values, templates: dict[str, list]) -> str:
    """The template that correlates best with `values`: shape only, not level or amplitude."""
    return max(templates, key=lambda name: pearson(values, templates[name]))


def loo_bandwidth(values, sigmas, weights=None) -> tuple[float, dict[float, float]]:
    """Pick the Gaussian smoothing width (in passages) by leave-one-out cross-validation.

    Each passage is predicted from its neighbours alone, with the same kernel `curve.smooth`
    uses; the width with the smallest mean squared error wins. Too narrow a kernel chases noise,
    too wide a kernel flattens real turns, and the held-out error is lowest in between.
    """
    y = np.asarray(values, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    positions = np.arange(len(y))
    errors = {}
    for sigma in sigmas:
        kernel = np.exp(-0.5 * ((positions[:, None] - positions[None, :]) / sigma) ** 2) * w[None, :]
        np.fill_diagonal(kernel, 0.0)
        predicted = kernel @ y / kernel.sum(axis=1)
        errors[sigma] = float(np.average((y - predicted) ** 2, weights=w))
    return min(errors, key=errors.get), errors


def resample(x, y, n: int) -> np.ndarray:
    """`n` evenly spaced points of the curve (x, y), from its first position to its last."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.interp(np.linspace(x[0], x[-1], n), x, y)


def template_fit(line, half, template) -> tuple[float, float]:
    """Stretch and shift a template onto a curve; how much of it then lies inside the curve's band.

    Least squares fits `a + b * template` to `line`. Returns b (a template fitted upside down,
    b ≤ 0, is not the template's shape) and the share of points where the fitted template is
    within `half` of the line.
    """
    line, half, t = (np.asarray(v, dtype=float) for v in (line, half, template))
    if t.std() == 0:
        return 0.0, float(np.mean(np.abs(line - line.mean()) <= half))
    b, a = np.polyfit(t, line, 1)
    return float(b), float(np.mean(np.abs(a + b * t - line) <= half))


def flat_coverage(line, half) -> float:
    """Share of the curve whose band holds its own average: 1 means a flat line fits it everywhere."""
    line, half = np.asarray(line, dtype=float), np.asarray(half, dtype=float)
    return float(np.mean(np.abs(line - line.mean()) <= half))


def permutation_p(predicted, expected, rounds: int = 10000, seed: int = 0) -> float:
    """How often shuffled labels match the predictions at least as well as the real ones.

    Shuffling keeps how many books carry each label, so a method that always answers the most
    common shape gets no credit for it.
    """
    predicted, expected = np.asarray(predicted), np.asarray(expected)
    observed = np.sum(predicted == expected)
    rng = np.random.default_rng(seed)
    at_least = sum(np.sum(predicted == rng.permutation(expected)) >= observed for _ in range(rounds))
    return float((1 + at_least) / (1 + rounds))
