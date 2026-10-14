"""Build the static site: the pages in `site/` plus one JSON file per book, written to `out`.

The repository stores no book text, so the site is built where Gutenberg can be reached (locally
or in CI): passages are rebuilt with the same chunking and only their first words are published,
never for an edition Gutenberg marks as still under copyright.
"""

import json
import re
import shutil
from pathlib import Path

import numpy as np

from story_shapes import chunk, corpus, gutenberg, metrics, plot
from story_shapes.kev import TENSION_LEVELS

ROOT = Path(__file__).resolve().parents[2]
PAGES_DIR = ROOT / "site"
RAW_DIR = ROOT / "data" / "raw"
EXCERPT_WORDS = 34
OVERVIEW_POINTS = 40

# A heading is a short line of its own: "CHAPTER IV.", "Stave II: ...", "ACT III", "XII", "7".
_HEADING = re.compile(
    r"(?m)(?<=\n\n)[ \t]*((?:CHAPTER|Chapter|STAVE|Stave|ACT|Act|BOOK|Book|PART|Part)\b[^\n]{0,60}"
    r"|[IVXLC]{1,7}\.?|\d{1,3}\.?)[ \t]*(?=\n)"
)


def chapters(story: str) -> list[dict]:
    """Where each chapter starts, as a fraction of the story's words (the passages' unit).

    The story starts after the first heading, so the first chapter is implicit at 0.
    """
    total = len(story.split())
    found = [{"name": "", "x": 0.0}]
    for match in _HEADING.finditer(story):
        found.append({"name": match.group(1).strip(), "x": round(len(story[: match.start()].split()) / total, 4)})
    return [{"n": i + 1, **chapter} for i, chapter in enumerate(found)]


def moments(x, line, died_at: float | None) -> list[dict]:
    """The turns worth jumping to: the ends, the lowest and highest points, the steepest fall and rise.

    Picked from the smoothed curve, so a single odd passage cannot become a moment. Moments closer
    than 3 % of the book to an earlier one are dropped.
    """
    line = np.asarray(line)
    window = max(2, len(line) // 20)
    change = line[window:] - line[:-window]
    picks = [  # in order of priority: the ends of the book win a tie
        (0, "First page"),
        (len(line) - 1, "The hero dies" if died_at is not None else "Last page"),
        (int(np.argmin(line)), "Lowest point"),
        (int(np.argmax(line)), "Highest point"),
        (int(np.argmin(change)) + window // 2, "Steepest fall"),
        (int(np.argmax(change)) + window // 2, "Steepest rise"),
    ]
    chosen = []
    for index, label in picks:
        if all(abs(x[index] - x[other["i"]]) >= 0.03 for other in chosen):
            chosen.append({"i": index, "label": label})
    return sorted(chosen, key=lambda moment: moment["i"])


def book_data(book: dict, row: dict, readings: list[dict], story: str | None, copyrighted: bool) -> dict:
    x = [round((r["start"] + r["end"]) / 2, 4) for r in readings]
    cut, (fortune, line, half) = plot.fortune_curve(readings)
    tension, tline, thalf = plot.series(readings, "tension", len(TENSION_LEVELS), signed=False)
    rows = []
    for i, record in enumerate(readings):
        on = i < len(line)
        probabilities = record["answers"]["fortune"]["probabilities"]
        rows.append([
            x[i],
            round(float(fortune[i]), 3) if on else None,
            round(float(line[i]), 3) if on else None,
            round(float(half[i]), 3) if on else None,
            round(float(tension[i]), 3), round(float(tline[i]), 3), round(float(thalf[i]), 3),
            round(record["answers"]["present"]["noul"], 2),
            [round(probabilities[str(k)], 3) for k in range(5)],
        ])
    text = None
    if story is not None and not copyrighted:
        passages = chunk.passages(story)
        if len(passages) == len(readings):
            text = [" ".join(p.text.split()[:EXCERPT_WORDS]) for p in passages]
    template = corpus.template_name(row["expected"])
    return {
        **{k: book.get(k) for k in ("id", "title", "author", "protagonist", "why")},
        "expected": row["expected"], "nearest": row["nearest"], "hit": row.get("hit"),
        "resolvable": row["resolvable"], "consistent": row.get("consistent"),
        "r_expected": round(row["correlations"][template], 2) if template in row["correlations"] else None,
        "r_nearest": round(row["correlations"][row["nearest"]], 2),
        "template": template if template in row["correlations"] else None,
        "died_at": row["died_at"], "copyrighted": copyrighted,
        "chapters": chapters(story) if story is not None else [],
        "moments": moments(x[: len(line)], line, row["died_at"]),
        "rows": rows, "text": text,
    }


def overview(book: dict, data: dict) -> dict:
    """The fortune curve resampled to a few points, for the home page's small multiples."""
    on = [r for r in data["rows"] if r[2] is not None]
    x = np.array([r[0] for r in on])
    grid = np.linspace(x[0], x[-1], OVERVIEW_POINTS)
    return {
        **{k: data[k] for k in ("id", "title", "author", "expected", "nearest", "hit", "template")},
        "passages": len(data["rows"]), "x0": round(float(x[0]), 3), "x1": round(float(x[-1]), 3),
        "line": [round(float(v), 3) for v in np.interp(grid, x, [r[2] for r in on])],
        "half": [round(float(v), 3) for v in np.interp(grid, x, [r[3] for r in on])],
    }


def build(out: Path, raw_dir: Path = RAW_DIR, readings_dir: Path = corpus.READINGS_DIR, fetch=gutenberg.fetch) -> Path:
    books = corpus.load_books()
    summary = corpus.score(books, readings_dir)
    _, templates = corpus.load_synthetic()
    rows = {row["id"]: row for row in summary["books"]}
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(PAGES_DIR, out)
    (out / "data").mkdir()
    cards = []
    for book_id, row in rows.items():
        book = books[book_id]
        raw = fetch(book["gutenberg"], raw_dir)
        story = gutenberg.story_text(raw, book["start"], book.get("end"))
        data = book_data(book, row, plot.load(readings_dir / f"{book_id}.jsonl"), story, gutenberg.is_copyrighted(raw))
        (out / "data" / f"{book_id}.json").write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
        cards.append(overview(book, data))

    scored = [row for row in summary["books"] if row["expected"] not in corpus.CONTROLS]
    nearest = [row["nearest"] for row in scored]
    expected = [corpus.template_name(row["expected"]) for row in scored]
    index = {
        "summary": {k: summary[k] for k in ("scored", "hits", "resolvable", "consistent", "p", "passes")}
        | {"post_hoc_hits": sum(a == b for a, b in zip(nearest, expected)),
           "post_hoc_p": metrics.permutation_p(nearest, expected)},
        "templates": {name: list(map(float, t)) for name, t in templates.items()},
        "books": cards,
    }
    (out / "data" / "index.json").write_text(json.dumps(index, separators=(",", ":"), ensure_ascii=False))
    return out
