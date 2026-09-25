"""A quick look at the curves of books already read: one column per book, fortune above tension."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from story_shapes import curve
from story_shapes.kev import FORTUNE_LEVELS, TENSION_LEVELS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
FORTUNE = "#2a78d6"
TENSION = "#eb6834"


def load(path: Path) -> list[dict]:
    records = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return sorted(records, key=lambda record: record["index"])


def series(readings: list[dict], question: str, n_levels: int, signed: bool, weights=None, sigma_frac=0.05):
    """Per-passage values, the smoothed curve and the smoothed spread, all on the plotting scale.

    Signed scales run from -1 (lowest level) to +1; unsigned ones from 0 to 1.
    """
    probabilities = [record["answers"][question]["probabilities"] for record in readings]
    levels = np.array([curve.mean_level(p) for p in probabilities])
    spreads = np.array([curve.spread(p) for p in probabilities])
    if signed:
        values, spreads = np.array([curve.to_unit(level, n_levels) for level in levels]), spreads * 2 / (n_levels - 1)
    else:
        values, spreads = levels / (n_levels - 1), spreads / (n_levels - 1)
    sigma = max(1.5, sigma_frac * len(readings))
    return values, curve.smooth(values, weights, sigma), curve.smooth(spreads, weights, sigma)


def _panel(ax, x, values, line, band, color, alphas):
    ax.scatter(x, values, s=9, color=color, alpha=alphas, linewidths=0, zorder=2)
    ax.fill_between(x, line - band, line + band, color=color, alpha=0.14, linewidth=0, zorder=1)
    ax.plot(x, line, color=color, linewidth=2, zorder=3)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["start", "", "end"])


def draw(books: list[dict], readings_dir: Path, out: Path) -> None:
    fig, axes = plt.subplots(2, len(books), figsize=(4.4 * len(books), 5.4), sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for column, book in enumerate(books):
        readings = load(readings_dir / f"{book['id']}.jsonl")
        x = np.array([(record["start"] + record["end"]) / 2 for record in readings])
        present = np.array([record["answers"]["present"]["noul"] for record in readings])

        fortune = series(readings, "fortune", len(FORTUNE_LEVELS), signed=True, weights=0.1 + 0.9 * present)
        top = axes[0][column]
        _panel(top, x, *fortune, FORTUNE, alphas=0.15 + 0.5 * present)
        top.axhline(0, color=GRID, linewidth=1, zorder=0)
        top.set_ylim(-1.05, 1.05)
        top.set_yticks([-1, 0, 1], ["ill fortune", "", "good fortune"])
        top.set_title(book["title"], color=INK, fontsize=11, loc="left", pad=16)
        top.text(0, 1.02, f"expected: {book['expected_shape']}", transform=top.transAxes,
                 color=MUTED, fontsize=8, va="bottom")

        tension = series(readings, "tension", len(TENSION_LEVELS), signed=False)
        bottom = axes[1][column]
        _panel(bottom, x, *tension, TENSION, alphas=0.4)
        bottom.set_ylim(-0.03, 1.03)
        bottom.set_yticks([0, 1], ["calm", "climax"])

    axes[0][0].set_ylabel("fortune of the protagonist", color=MUTED, fontsize=9)
    axes[1][0].set_ylabel("tension", color=MUTED, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
