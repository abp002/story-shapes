import random

import pytest

from story_shapes.chunk import MAX_WORDS, MIN_WORDS, TARGET_WORDS, passages


def story(seed: int) -> str:
    """Paragraphs of mixed length, some far longer than a passage, with quoted dialogue."""
    rng = random.Random(seed)
    paragraphs = []
    for p in range(rng.randint(5, 40)):
        sentences = []
        for s in range(rng.choice([1, 3, 12, 40])):
            words = [f"w{p}_{s}_{i}" for i in range(rng.randint(3, 30))]
            ending = rng.choice(['.', '!', '?', '.”', '?"', ''])
            sentences.append(" ".join(words) + ending)
        # hard-wrapped lines, as in Gutenberg files
        text = " ".join(sentences)
        paragraphs.append("\n".join(text[i : i + 70] for i in range(0, len(text), 70)))
    return "\n\n".join(paragraphs)


@pytest.mark.parametrize("seed", range(30))
def test_every_word_is_kept_once_and_in_order(seed):
    text = story(seed)
    assert " ".join(p.text for p in passages(text)).split() == text.split()


@pytest.mark.parametrize("seed", range(30))
def test_no_passage_exceeds_the_cap(seed):
    assert all(0 < p.words <= MAX_WORDS for p in passages(story(seed)))


@pytest.mark.parametrize("seed", range(30))
def test_positions_cover_the_story_without_gaps(seed):
    result = passages(story(seed))
    assert result[0].start == 0
    assert result[-1].end == pytest.approx(1)
    for before, after in zip(result, result[1:]):
        assert before.end == pytest.approx(after.start)


@pytest.mark.parametrize("seed", range(30))
def test_a_scrap_only_stands_alone_when_no_neighbour_has_room_for_it(seed):
    words = [p.words for p in passages(story(seed))]
    for i, n in enumerate(words):
        if n < MIN_WORDS:
            assert i == 0 or words[i - 1] + n > MAX_WORDS
            assert i == len(words) - 1 or n + words[i + 1] > MAX_WORDS


def test_a_closing_stage_direction_joins_the_last_passage():
    text = "\n\n".join(" ".join(f"p{p}w{i}" for i in range(50)) for p in range(4)) + "\n\n[_Exeunt._]"
    assert [p.words for p in passages(text)] == [TARGET_WORDS + 1]


def test_short_paragraphs_are_packed_up_to_the_target():
    text = "\n\n".join(" ".join(f"p{p}w{i}" for i in range(50)) for p in range(20))
    result = passages(text)
    assert [p.words for p in result] == [TARGET_WORDS] * 5


def test_a_sentence_longer_than_the_cap_is_cut_into_word_runs():
    text = " ".join(f"w{i}" for i in range(1000))
    result = passages(text)
    assert all(p.words <= MAX_WORDS for p in result)
    assert " ".join(p.text for p in result).split() == text.split()
