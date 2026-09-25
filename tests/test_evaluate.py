import pytest

from story_shapes.evaluate import fortune_of


def test_a_score_answer_maps_onto_minus_one_to_one():
    top = {"fortune": {"probabilities": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 1}}}
    bottom = {"fortune": {"probabilities": {"0": 1, "1": 0, "2": 0, "3": 0, "4": 0}}}
    assert (fortune_of(top), fortune_of(bottom)) == (1, -1)


def test_good_and_bad_events_give_good_minus_bad():
    assert fortune_of({"good": {"noul": 0.9}, "bad": {"noul": 0.1}}) == pytest.approx(0.8)
    assert fortune_of({"good": {"noul": 0.2}, "bad": {"noul": 0.7}}) == pytest.approx(-0.5)


def test_answers_asked_together_are_split_back_by_variant():
    from story_shapes.evaluate import VARIANTS, all_variants, of_variant

    together = all_variants("Ana")
    for name, questions_for in VARIANTS.items():
        assert of_variant(together, name) == questions_for("Ana")


def test_book_readings_made_with_other_questions_are_reported_as_stale(tmp_path, monkeypatch):
    import json

    from story_shapes import evaluate, kev

    monkeypatch.setattr(evaluate, "READINGS_DIR", tmp_path)
    record = {"index": 0, "answers": {"fortune": {"probabilities": {"0": 0, "1": 0, "2": 1, "3": 0, "4": 0}},
                                      "present": {"noul": 1.0}}}
    (tmp_path / "tale.jsonl").write_text(json.dumps(record) + "\n")
    old = {"questions": {"fortune": {"criteria": ["an older wording"]}}}
    (tmp_path / "tale.meta.json").write_text(json.dumps(old))
    assert evaluate.stale_reading("tale", "Ana")
    (tmp_path / "tale.meta.json").write_text(json.dumps({"questions": kev.questions("Ana")}))
    assert not evaluate.stale_reading("tale", "Ana")
