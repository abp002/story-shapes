"""Does Kev recover the shapes that the plots of classic books predict? (ALE-224)

Everything here was fixed before any of the corpus was read: the expected shape of each book
(books.json, with a line on why) and how a curve is scored against it.

- A book's curve is the one `plot` draws: fortune up to the protagonist's death, passages without
  them counting less, with its 95 % band. It is resampled to the 12 points of the shape templates
  (eval/synthetic), from its first passage to its last.
- Books with no resolvable shape are set apart first: when a flat line at the curve's average stays
  inside the band on at least 90 % of the story, the curve cannot tell one shape from another.
- Every other book gets the template that correlates best with its curve (shape only, not level).
  "From bad to worse" is scored as riches to rags. The ambiguous controls are shown, not scored.
- C5: Kev picks the expected shape for more books than shuffled labels would, p < 0.05 by
  permutation. Books without a resolvable shape count as misses.
- Also reported: whether the expected template, stretched and shifted onto the curve, stays inside
  the band on at least 90 % of the story (consistent with the curve even when not the best match).
"""

import json
from pathlib import Path

import numpy as np

from story_shapes import metrics, plot
from story_shapes.cli import READINGS_DIR, load_books
from story_shapes.evaluate import EVAL_DIR, RESULTS_DIR, load_synthetic

FLAT = 0.9
INSIDE = 0.9
SCORED_AS = {"from bad to worse": "riches-to-rags"}
CONTROLS = {"ambiguous"}
SKIP = {"romeo-and-juliet-renamed"}  # the same book, read again for the renaming test
CRITERION = "C5: Kev picks the expected shape for more books than shuffled labels would (permutation p < 0.05)"


def template_name(shape: str) -> str:
    return SCORED_AS.get(shape, shape.replace(" ", "-"))


def score_book(book: dict, readings: list[dict], templates: dict[str, list]) -> dict:
    x = np.array([(record["start"] + record["end"]) / 2 for record in readings])
    cut, (_, line, half) = plot.fortune_curve(readings)
    x = x[: len(line)]
    n = len(next(iter(templates.values())))
    line12, half12 = metrics.resample(x, line, n), metrics.resample(x, half, n)
    correlations = {name: metrics.pearson(line12, t) for name, t in templates.items()}
    expected = book["expected_shape"]
    row = {
        "id": book["id"],
        "title": book["title"],
        "expected": expected,
        "passages": len(readings),
        "died_at": None if cut is None else round(float(x[cut]), 2),
        "flat": metrics.flat_coverage(line12, half12),
        "correlations": correlations,
        "nearest": max(correlations, key=correlations.get),
    }
    row["resolvable"] = row["flat"] < FLAT
    if expected not in CONTROLS:
        b, inside = metrics.template_fit(line12, half12, templates[template_name(expected)])
        row["consistent"] = b > 0 and inside >= INSIDE
        row["hit"] = row["resolvable"] and row["nearest"] == template_name(expected)
    return row


def score(books: dict[str, dict] | None = None, readings_dir: Path = READINGS_DIR) -> dict:
    books = books or load_books()
    _, templates = load_synthetic()
    rows, missing = [], []
    for book_id, book in books.items():
        path = readings_dir / f"{book_id}.jsonl"
        if book_id in SKIP:
            continue
        if not path.exists():
            missing.append(book_id)
            continue
        rows.append(score_book(book, plot.load(path), templates))
    scored = [row for row in rows if row["expected"] not in CONTROLS]
    predicted = [row["nearest"] if row["resolvable"] else "none" for row in scored]
    expected = [template_name(row["expected"]) for row in scored]
    hits = sum(row["hit"] for row in scored)
    p = metrics.permutation_p(predicted, expected) if scored else 1.0
    return {
        "criterion": CRITERION,
        "books": rows,
        "missing": missing,
        "scored": len(scored),
        "hits": hits,
        "resolvable": sum(row["resolvable"] for row in scored),
        "consistent": sum(row["consistent"] for row in scored),
        "p": p,
        "passes": p < 0.05,
    }


def markdown(summary: dict) -> str:
    lines = [
        "# Corpus: does Kev recover the expected shapes?",
        "",
        "Expected shapes and scoring were committed before the corpus was read (see `src/story_shapes/corpus.py`).",
        "",
        f"**{summary['criterion']}** — {'passes' if summary['passes'] else 'fails'}: "
        f"{summary['hits']} of {summary['scored']} books, p = {summary['p']:.3f}. "
        f"{summary['resolvable']} have a resolvable shape; the expected shape is consistent with "
        f"the curve's band for {summary['consistent']}.",
        "",
        "| book | expected | nearest | r expected | r nearest | resolvable | consistent | hit |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(summary["books"], key=lambda r: (r["expected"], r["title"])):
        control = row["expected"] in CONTROLS
        r_expected = "—" if control else f"{row['correlations'][template_name(row['expected'])]:.2f}"
        mark = lambda key: "—" if control else ("yes" if row[key] else "no")  # noqa: E731
        lines.append(
            f"| {row['title']} | {row['expected']} | {row['nearest'].replace('-', ' ')} | {r_expected} | "
            f"{row['correlations'][row['nearest']]:.2f} | {'yes' if row['resolvable'] else 'no'} | "
            f"{mark('consistent')} | {mark('hit')} |"
        )
    if summary["missing"]:
        lines += ["", f"Not read yet: {', '.join(summary['missing'])}."]
    return "\n".join(lines) + "\n"


def write(summary: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "corpus.json").write_text(json.dumps(summary, indent=2) + "\n")
    out = EVAL_DIR / "CORPUS.md"
    out.write_text(markdown(summary))
    return out
