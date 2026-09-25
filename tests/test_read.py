import json

import pytest

from story_shapes.cli import ReadingMismatch, read

PARAGRAPH = " ".join(f"word{i}" for i in range(120)) + "."
RAW = (
    "Title: A Tale\n\n*** START OF THE PROJECT GUTENBERG EBOOK A TALE ***\nCONTENTS\n\n"
    + "Once upon a time. " + "\n\n".join([PARAGRAPH] * 4)
    + "\n*** END OF THE PROJECT GUTENBERG EBOOK A TALE ***\n"
)
BOOK = {"id": "tale", "gutenberg": 999, "protagonist": "Ana", "start": "Once upon a time"}


class FakeClient:
    def __init__(self, run="fake/kev"):
        self.calls = 0
        self.run = run
        self.states = []

    def decide(self, state, questions):
        self.calls += 1
        self.states.append(state)
        return {
            "answers": {
                "fortune": {"type": "score", "score": 2.0, "probabilities": {"0": 0, "1": 0, "2": 1, "3": 0, "4": 0}},
                "tension": {"type": "score", "score": 0.0, "probabilities": {"0": 1, "1": 0, "2": 0, "3": 0}},
                "present": {"type": "noul", "noul": 0.9},
            },
            "latency_ms": 1.0,
        }

    def model_card(self):
        return {"run": self.run}


@pytest.fixture
def dirs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "pg999.txt").write_text(RAW)
    return raw, tmp_path / "readings"


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_readings_keep_positions_and_answers_but_never_the_text(dirs):
    raw, readings = dirs
    path = read(BOOK, FakeClient(), raw, readings)
    lines = records(path)
    assert len(lines) >= 2
    for record in lines:
        assert "text" not in record
        assert {"index", "start", "end", "words", "answers"} <= record.keys()
    assert "word7" not in path.read_text()


def test_the_settings_of_a_reading_are_stored_beside_it(dirs):
    raw, readings = dirs
    read(BOOK, FakeClient(), raw, readings)
    meta = json.loads((readings / "tale.meta.json").read_text())
    assert meta["chunking"] == {"target_words": 200, "max_words": 260, "min_words": 40}
    assert meta["questions"]["fortune"]["instructions"].endswith("for Ana?")
    assert meta["model"] == {"run": "fake/kev"}
    assert len(meta["story_sha256"]) == 64


def test_a_rerun_resumes_without_asking_again(dirs):
    raw, readings = dirs
    read(BOOK, FakeClient(), raw, readings)
    again = FakeClient()
    read(BOOK, again, raw, readings)
    assert again.calls == 0


def test_resuming_with_other_questions_is_refused(dirs):
    raw, readings = dirs
    read(BOOK, FakeClient(), raw, readings)
    with pytest.raises(ReadingMismatch):
        read({**BOOK, "protagonist": "Luis"}, FakeClient(), raw, readings)


def test_resuming_with_another_model_is_refused(dirs):
    raw, readings = dirs
    read(BOOK, FakeClient(run="fake/kev"), raw, readings)
    with pytest.raises(ReadingMismatch):
        read(BOOK, FakeClient(run="fake/other"), raw, readings)


def test_fresh_starts_over_after_a_change(dirs):
    raw, readings = dirs
    read(BOOK, FakeClient(), raw, readings)
    client = FakeClient()
    read({**BOOK, "protagonist": "Luis"}, client, raw, readings, fresh=True)
    assert client.calls == len(records(readings / "tale.jsonl"))


def test_a_renamed_book_never_shows_the_original_names_to_the_model(dirs):
    raw, readings = dirs
    client = FakeClient()
    read({**BOOK, "rename": {"word7": "swapped7", "Once": "Twice"}}, client, raw, readings)
    passages = " ".join(state["passage"] for state in client.states)
    assert "word7" not in passages and "swapped7" in passages
