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
RELATIVE_LIMIT = 3.2  # standard deviations shown in the relative view


def load(path: Path) -> list[dict]:
    records = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return sorted(records, key=lambda record: record["index"])


def series(readings: list[dict], question: str, n_levels: int, signed: bool, weights=None, sigma_frac=0.05,
           relative=False):
    """Per-passage values, the smoothed curve and the half-width of its 95 % band, on the plotting scale.

    Signed scales run from -1 (lowest level) to +1; unsigned ones from 0 to 1. With `relative`, all
    three are moved onto the book's own scale (see `curve.relative`).
    """
    probabilities = [record["answers"][question]["probabilities"] for record in readings]
    levels = np.array([curve.mean_level(p) for p in probabilities])
    spreads = np.array([curve.spread(p) for p in probabilities])
    if signed:
        values, spreads = np.array([curve.to_unit(level, n_levels) for level in levels]), spreads * 2 / (n_levels - 1)
    else:
        values, spreads = levels / (n_levels - 1), spreads / (n_levels - 1)
    sigma = max(1.5, sigma_frac * len(readings))
    line, half = curve.confidence(values, spreads, weights, sigma)
    if relative:
        line, values, half = curve.relative(line, values, half)
    return values, line, half


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
    ax.set_xlim(0, 1)  # the whole story, even when the curve ends earlier
    ax.set_xticks([0, 0.5, 1], ["start", "", "end"])


def draw(books: list[dict], readings_dir: Path, out: Path, relative: bool = False) -> None:
    """One column per book. With `relative`, each curve is drawn on its own book's scale."""
    fig, axes = plt.subplots(2, len(books), figsize=(4.4 * len(books), 5.4), sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for column, book in enumerate(books):
        readings = load(readings_dir / f"{book['id']}.jsonl")
        x = np.array([(record["start"] + record["end"]) / 2 for record in readings])
        present = np.array([record["answers"]["present"]["noul"] for record in readings])

        # the protagonist's fortune ends where they die for good; the passages after it are about others
        dead = [record["answers"].get("dead", {}).get("noul", 0.0) for record in readings]
        cut = curve.death_cut(dead)
        end = len(readings) if cut is None else cut + 1
        fortune = series(readings[:end], "fortune", len(FORTUNE_LEVELS), signed=True, weights=0.1 + 0.9 * present[:end],
                         relative=relative)
        top = axes[0][column]
        _panel(top, x[:end], *fortune, FORTUNE, alphas=0.15 + 0.5 * present[:end])
        if cut is not None:
            top.plot(x[cut], fortune[1][-1], marker="x", markersize=8, markeredgewidth=2, color=INK, zorder=4)
            top.annotate("dies", (x[cut], fortune[1][-1]), xytext=(-4, 10), textcoords="offset points",
                         ha="right", color=MUTED, fontsize=8)
        top.axhline(0, color=GRID, linewidth=1, zorder=0)
        if relative:
            top.set_ylim(-RELATIVE_LIMIT, RELATIVE_LIMIT)
            top.set_yticks([-2, 0, 2], ["worse than\nusual", "", "better than\nusual"])
        else:
            top.set_ylim(-1.05, 1.05)
            top.set_yticks([-1, 0, 1], ["ill fortune", "", "good fortune"])
        top.set_title(book["title"], color=INK, fontsize=11, loc="left", pad=16)
        top.text(0, 1.02, f"expected: {book['expected_shape']}", transform=top.transAxes,
                 color=MUTED, fontsize=8, va="bottom")

        tension = series(readings, "tension", len(TENSION_LEVELS), signed=False, relative=relative)
        bottom = axes[1][column]
        _panel(bottom, x, *tension, TENSION, alphas=0.4)
        if relative:
            bottom.axhline(0, color=GRID, linewidth=1, zorder=0)
            bottom.set_ylim(-RELATIVE_LIMIT, RELATIVE_LIMIT)
            bottom.set_yticks([-2, 0, 2], ["calmer than\nusual", "", "tenser than\nusual"])
        else:
            bottom.set_ylim(-0.03, 1.03)
            bottom.set_yticks([0, 1], ["calm", "climax"])

    axes[0][0].set_ylabel("fortune of the protagonist", color=MUTED, fontsize=9)
    axes[1][0].set_ylabel("tension", color=MUTED, fontsize=9)
    scale = "Each curve on its own book's scale: 0 is the book's average, ±1 one standard deviation. " if relative else ""
    fig.text(0.01, 0.01, f"{scale}Band: where the curve could be, 95 % interval. Dots: single passages.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)


TRAP_GROUPS = [
    ("cheerful-bad", "cheerful words,\nbad fortune", -1),
    ("gloomy-good", "gloomy words,\ngood fortune", 1),
    ("congruent-bad", "gloomy words,\nbad fortune", -1),
    ("congruent-good", "cheerful words,\ngood fortune", 1),
]
PLANNED = "#8a8985"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def draw_traps(traps: list[dict], answers: dict, out: Path) -> None:
    """Where each method puts every trap and control; the correct half of each column is shaded."""
    from story_shapes.evaluate import signed, vader

    scores = {"kev": [signed(answers[t["id"]], "fortune") for t in traps], "vader": vader([t["passage"] for t in traps])}
    panels = [
        ("kev", "Kev: how are things going for the protagonist?", FORTUNE, ("ill fortune", "good fortune")),
        ("vader", "VADER: how positive are the words?", TENSION, ("negative", "positive")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for ax, (method, title, color, ends) in zip(axes, panels):
        _style(ax)
        for column, (kind, _, sign) in enumerate(TRAP_GROUPS):
            ax.axvspan(column - 0.42, column + 0.42, ymin=0.5 if sign > 0 else 0, ymax=1 if sign > 0 else 0.5,
                       color=GRID, alpha=0.5, linewidth=0, zorder=0)
            values = [v for t, v in zip(traps, scores[method]) if t["kind"] == kind]
            ax.scatter(column + rng.uniform(-0.28, 0.28, len(values)), values, s=16, color=color, alpha=0.75,
                       linewidths=0, zorder=2)
        ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
        ax.axvline(1.5, color=GRID, linewidth=1, zorder=1)
        ax.set_xticks(range(len(TRAP_GROUPS)), [label for _, label, _ in TRAP_GROUPS])
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1, 0, 1], [ends[0], "", ends[1]])
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=18)
        ax.text(0.5, 1.0, "traps", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=MUTED, fontsize=8)
        ax.text(2.5, 1.0, "controls", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=MUTED, fontsize=8)
    fig.text(0.01, 0.01, "Shaded: the side a reader of the fortune should land on.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)


def draw_synthetic(stories: list[dict], out: Path) -> None:
    """Planned fortune against what Kev and VADER read, one panel per synthetic story."""
    columns = 4
    rows = -(-len(stories) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.4 * rows), sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, story in zip(axes.flat, stories):
        _style(ax)
        x = np.arange(1, len(story["levels"]) + 1)
        planned = [curve.to_unit(level, len(FORTUNE_LEVELS)) for level in story["levels"]]
        ax.step(x, planned, where="mid", color=PLANNED, linewidth=1.5, linestyle=(0, (3, 2)), label="planned", zorder=1)
        ax.plot(x, story["vader"], color=TENSION, linewidth=1.5, label="VADER", zorder=2)
        ax.plot(x, story["kev"], color=FORTUNE, linewidth=2, marker="o", markersize=3.5, label="Kev", zorder=3)
        ax.axhline(0, color=GRID, linewidth=1, zorder=0)
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([-1, 0, 1], ["ill", "", "good"])
        ax.set_title(f"{story['id']}", color=INK, fontsize=9, loc="left")
        ax.text(1.0, 1.02, f"ρ Kev {story['kev_rho']:.2f} · VADER {story['vader_rho']:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", color=MUTED, fontsize=7)
    for ax in axes.flat[len(stories):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
