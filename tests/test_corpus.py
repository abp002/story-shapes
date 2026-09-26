import json

import numpy as np

from story_shapes import corpus
from story_shapes.evaluate import load_synthetic


def _reading(levels, dead_from=None, spread=0.05):
    """Fake Kev answers on disk: fortune piled on the given level, the protagonist always present."""
    records = []
    for i, level in enumerate(levels):
        p = {str(k): spread / 4 for k in range(5)}
        p[str(level)] = 1 - spread
        dead = 0.95 if dead_from is not None and i >= dead_from else 0.0
        records.append({
            "index": i, "start": i / len(levels), "end": (i + 1) / len(levels),
            "answers": {
                "fortune": {"probabilities": p},
                "present": {"noul": 1.0},
                "dead": {"noul": dead},
            },
        })
    return "\n".join(json.dumps(r) for r in records) + "\n"


def _stretch(template, n):
    return [int(round(v)) for v in np.interp(np.linspace(0, len(template) - 1, n), range(len(template)), template)]


def _books(tmp_path, plan):
    books = {}
    for book_id, (expected, levels, dead_from) in plan.items():
        (tmp_path / f"{book_id}.jsonl").write_text(_reading(levels, dead_from))
        books[book_id] = {"id": book_id, "title": book_id, "expected_shape": expected}
    return books


def test_books_that_follow_their_expected_shape_are_hits(tmp_path):
    _, templates = load_synthetic()
    plan = {name: (name.replace("-", " "), _stretch(t, 120), None) for name, t in templates.items()}
    summary = corpus.score(_books(tmp_path, plan), tmp_path)
    assert summary["hits"] == summary["scored"] == 6
    assert all(row["consistent"] for row in summary["books"])
    assert summary["p"] < 0.05


def test_a_book_read_as_the_wrong_shape_is_a_miss_and_labels_do_not_leak(tmp_path):
    _, templates = load_synthetic()
    plan = {"liar": ("icarus", _stretch(templates["man-in-a-hole"], 120), None)}
    row = corpus.score(_books(tmp_path, plan), tmp_path)["books"][0]
    assert row["nearest"] == "man-in-a-hole" and not row["hit"] and not row["consistent"]


def test_a_flat_reading_has_no_resolvable_shape_and_counts_as_a_miss(tmp_path):
    rng = np.random.default_rng(1)
    plan = {"flat": ("riches to rags", list(rng.choice([1, 2], 120)), None)}
    row = corpus.score(_books(tmp_path, plan), tmp_path)["books"][0]
    assert not row["resolvable"] and not row["hit"]


def test_the_curve_is_scored_up_to_the_protagonists_death(tmp_path):
    # icarus until death at 80 %, then the others' fortune recovers: only the part before counts
    _, templates = load_synthetic()
    levels = _stretch(templates["icarus"], 96) + [4] * 24
    row = corpus.score(_books(tmp_path, {"dies": ("icarus", levels, 96)}), tmp_path)["books"][0]
    assert row["died_at"] is not None and row["nearest"] == "icarus"


def test_controls_are_shown_but_not_scored_and_unread_books_are_listed(tmp_path):
    books = _books(tmp_path, {"hamlet": ("ambiguous", [2] * 60, None)})
    books["unread"] = {"id": "unread", "title": "unread", "expected_shape": "icarus"}
    summary = corpus.score(books, tmp_path)
    assert summary["scored"] == 0 and len(summary["books"]) == 1 and summary["missing"] == ["unread"]
