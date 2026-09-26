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
    positions = np.arange(len(y))
    kernel = np.exp(-0.5 * ((positions[:, None] - positions[None, :]) / sigma) ** 2)
    if weights is not None:
        kernel = kernel * np.asarray(weights, dtype=float)[None, :]
    return kernel @ y / kernel.sum(axis=1)


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
