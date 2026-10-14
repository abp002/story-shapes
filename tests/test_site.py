import json

import numpy as np

from story_shapes import site


def _readings(levels):
    records = []
    for i, level in enumerate(levels):
        p = {str(k): 0.0125 for k in range(5)}
        p[str(level)] = 0.95
        tension = {str(k): 0.25 for k in range(4)}
        records.append({
            "index": i, "start": i / len(levels), "end": (i + 1) / len(levels),
            "answers": {
                "fortune": {"probabilities": p},
                "tension": {"probabilities": tension},
                "present": {"noul": 1.0},
                "dead": {"noul": 0.0},
            },
        })
    return records


def test_chapters_start_at_their_heading_in_story_words():
    story = "one two three four\n\nCHAPTER II.\n\nfive six seven eight\n\nIII\n\nnine ten"
    chapters = site.chapters(story)
    assert [c["n"] for c in chapters] == [1, 2, 3]
    assert chapters[1]["name"] == "CHAPTER II."
    total = len(story.split())
    assert chapters[1]["x"] == round(4 / total, 4)
    assert chapters[2]["x"] == round(10 / total, 4)


def test_a_number_inside_a_sentence_is_not_a_chapter():
    assert len(site.chapters("He was 7 years old.\n\nIt rained for 3\ndays.")) == 1


def test_moments_find_the_fall_and_the_bottom_of_a_hole():
    x = np.linspace(0, 1, 101)
    line = -np.exp(-(((x - 0.5) / 0.15) ** 2))  # down to the bottom at the middle, then back up
    moments = {m["label"]: m["i"] for m in site.moments(x, line, died_at=None)}
    assert moments["First page"] == 0
    assert moments["Lowest point"] == 50
    assert moments["Steepest fall"] < 50 < moments["Steepest rise"]
    assert moments["Last page"] == 100


def test_moments_too_close_to_another_are_dropped_and_a_death_ends_the_book():
    x = np.linspace(0, 1, 101)
    line = 1 - x  # a straight fall: the highest point is the first page
    labels = [m["label"] for m in site.moments(x, line, died_at=0.9)]
    assert labels.count("First page") == 1 and "Highest point" not in labels
    assert labels[-1] == "The hero dies"


def test_copyrighted_editions_publish_no_text():
    levels = [2] * 10
    story = "\n\n".join(" ".join(["word"] * 199) + " end." for _ in levels)  # one passage per paragraph
    row = {"expected": "man in a hole", "nearest": "man-in-a-hole", "hit": False, "resolvable": False,
           "consistent": True, "correlations": {"man-in-a-hole": 0.5}, "died_at": None}
    book = {"id": "b", "title": "B", "author": "A", "protagonist": "P"}
    open_text = site.book_data(book, row, _readings(levels), story, copyrighted=False)
    closed = site.book_data(book, row, _readings(levels), story, copyrighted=True)
    assert open_text["text"] is not None and len(open_text["text"]) == len(levels)
    assert all(len(t.split()) <= site.EXCERPT_WORDS for t in open_text["text"])
    assert closed["text"] is None
    json.dumps(closed)  # the page reads it as JSON
