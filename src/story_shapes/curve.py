"""Turn score probabilities into curves."""

import numpy as np


def _levels_and_probs(probabilities: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    levels = np.array([int(level) for level in probabilities], dtype=float)
    p = np.array(list(probabilities.values()), dtype=float)
    return levels, p / p.sum()


def mean_level(probabilities: dict[str, float]) -> float:
    """Expected level of a score answer whose probabilities are keyed by level index ("0", "1", …)."""
    levels, p = _levels_and_probs(probabilities)
    return float((levels * p).sum())


def spread(probabilities: dict[str, float]) -> float:
    """Standard deviation of the level distribution: how torn the model is between levels."""
    levels, p = _levels_and_probs(probabilities)
    mean = (levels * p).sum()
    return float(np.sqrt((p * (levels - mean) ** 2).sum()))


def to_unit(level: float, n_levels: int) -> float:
    """Map a level onto [-1, 1]: the lowest level is -1, the highest +1."""
    return 2 * level / (n_levels - 1) - 1


def smooth(values, weights=None, sigma: float = 4.0) -> np.ndarray:
    """Gaussian moving average over passage positions.

    `sigma` is in passages. `weights` lets some passages count less, e.g. those where the
    protagonist does not appear; a passage with weight 0 does not pull the curve at all.
    """
    y = np.asarray(values, dtype=float)
    kernel = _kernel(len(y), weights, sigma)
    return kernel @ y / kernel.sum(axis=1)


def _kernel(n: int, weights, sigma: float) -> np.ndarray:
    positions = np.arange(n)
    kernel = np.exp(-0.5 * ((positions[:, None] - positions[None, :]) / sigma) ** 2)
    if weights is not None:
        kernel = kernel * np.asarray(weights, dtype=float)[None, :]
    return kernel


def lag1_autocorrelation(residuals) -> float:
    """Correlation between each residual and the next one, floored at 0."""
    r = np.asarray(residuals, dtype=float)
    if len(r) < 3 or r[:-1].std() == 0 or r[1:].std() == 0:
        return 0.0
    return max(0.0, float(np.corrcoef(r[:-1], r[1:])[0, 1]))


def confidence(values, spreads, weights=None, sigma: float = 4.0, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """The smoothed curve and the half-width of an interval around it: how far it could be off.

    At each position the variance of what the kernel sees combines how torn the model is inside
    each passage (`spreads`) with how much the passages disagree with the curve, and is divided by
    the number of passages that effectively count, (Σw)² / Σw². That number shrinks at the ends of
    the story, before a cut, and where passages carry little weight, so the band widens there.
    Neighbouring passages are not independent (a gloomy chapter stays gloomy), so the effective
    count is further scaled by (1 - ρ) / (1 + ρ), with ρ the lag-1 autocorrelation of the residuals.
    """
    y = np.asarray(values, dtype=float)
    s = np.asarray(spreads, dtype=float)
    kernel = _kernel(len(y), weights, sigma)
    total = kernel.sum(axis=1)
    line = kernel @ y / total
    variance = (kernel * (s[None, :] ** 2 + (y[None, :] - line[:, None]) ** 2)).sum(axis=1) / total
    counted = np.ones(len(y), bool) if weights is None else np.asarray(weights, dtype=float) > 0
    rho = lag1_autocorrelation((y - line)[counted])
    n_eff = total**2 / (kernel**2).sum(axis=1) * (1 - rho) / (1 + rho)
    return line, z * np.sqrt(variance / n_eff)


def relative(line, values, half) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put a curve on its own book's scale: 0 is the book's average, ±1 one standard deviation of the curve.

    A story that stays bad from start to finish looks flat on the absolute scale; here its ups and
    downs inside the bad fill the height. The per-passage `values` move with the line; the band's
    `half` width is only scaled. A band that is wide on this scale means the shape is mostly noise
    blown up.
    """
    line = np.asarray(line, dtype=float)
    centre, scale = line.mean(), line.std()
    if scale == 0:
        scale = 1.0
    return (line - centre) / scale, (np.asarray(values, dtype=float) - centre) / scale, np.asarray(half, dtype=float) / scale


def death_cut(
    dead, threshold: float = 0.8, hold: float = 0.5, window: int = 5, present=None, revoke: bool = False
) -> int | None:
    """The passage where the protagonist dies for good, or None.

    A passage counts when P(dead) reaches `threshold` and the average over it and the next
    `window - 1` passages stays at `hold` or more: a vision of one's own grave, or a death that is
    faked and then undone a few passages later, does not end the curve. With `revoke`, a death is
    also dropped when, within that window, the protagonist appears (`present` ≥ 0.5) and is
    clearly alive (P(dead) ≤ 0.2); mere absence, as after a real death, does not revoke it.
    """
    dead = [float(p) for p in dead]
    for i, p in enumerate(dead):
        ahead = dead[i : i + window]
        if p < threshold or sum(ahead) / len(ahead) < hold:
            continue
        if revoke and present is not None and any(
            present[j] >= 0.5 and dead[j] <= 0.2 for j in range(i + 1, min(i + window, len(dead)))
        ):
            continue
        return i
    return None
