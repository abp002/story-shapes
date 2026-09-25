"""Cut a story into passages short enough for Kev, which was trained on states of up to 384 tokens."""

import re
from dataclasses import dataclass

TARGET_WORDS = 200
MAX_WORDS = 260
MIN_WORDS = 40  # below this a passage gives the model too little to read

# Split after sentence-ending punctuation, optionally followed by a closing quote or bracket.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'”’)\]])\s+")


@dataclass(frozen=True)
class Passage:
    index: int
    text: str
    words: int
    start: float  # where the passage begins, as a fraction of the story's words
    end: float


def _pieces(text: str, max_words: int) -> list[tuple[str, bool]]:
    """Paragraphs, broken into sentences (and those into word runs) when longer than `max_words`.

    Each piece carries whether it opens a paragraph, so passages can keep the paragraph breaks.
    """
    pieces = []
    for block in re.split(r"\n\s*\n", text):
        paragraph = " ".join(block.split())
        if not paragraph:
            continue
        if len(paragraph.split()) <= max_words:
            pieces.append((paragraph, True))
            continue
        first = True
        for sentence in _SENTENCE_BREAK.split(paragraph):
            words = sentence.split()
            for i in range(0, len(words), max_words):
                pieces.append((" ".join(words[i : i + max_words]), first))
                first = False
    return pieces


def _words(group: list[tuple[str, bool]]) -> int:
    return sum(len(piece.split()) for piece, _ in group)


def _merge_scraps(groups: list[list[tuple[str, bool]]], min_words: int, max_words: int):
    """Fold passages shorter than `min_words` into a neighbour, when that stays within `max_words`."""
    merged: list[list[tuple[str, bool]]] = []
    carry: list[tuple[str, bool]] = []
    for group in groups:
        if carry:
            if _words(carry) + _words(group) <= max_words:
                group = carry + group
            else:
                merged.append(carry)
            carry = []
        n = _words(group)
        if n >= min_words:
            merged.append(group)
        elif merged and _words(merged[-1]) + n <= max_words:
            merged[-1] = merged[-1] + group
        else:
            carry = group  # no room behind it: try the passage that follows
    if carry:
        merged.append(carry)
    return merged


def passages(
    text: str, target_words: int = TARGET_WORDS, max_words: int = MAX_WORDS, min_words: int = MIN_WORDS
) -> list[Passage]:
    """Pack paragraphs greedily into passages of about `target_words`, never more than `max_words`.

    Scraps under `min_words` (a closing stage direction, a line squeezed out by a long paragraph)
    are folded into a neighbouring passage whenever it has room.
    """
    groups: list[list[tuple[str, bool]]] = []
    current: list[tuple[str, bool]] = []
    count = 0
    for piece in _pieces(text, max_words):
        n = len(piece[0].split())
        if current and count + n > max_words:
            groups.append(current)
            current, count = [], 0
        current.append(piece)
        count += n
        if count >= target_words:
            groups.append(current)
            current, count = [], 0
    if current:
        groups.append(current)
    groups = _merge_scraps(groups, min_words, max_words)

    total = sum(_words(group) for group in groups)
    result = []
    seen = 0
    for index, group in enumerate(groups):
        body = group[0][0]
        for piece, opens_paragraph in group[1:]:
            body += ("\n\n" if opens_paragraph else " ") + piece
        n = len(body.split())
        result.append(Passage(index, body, n, seen / total, (seen + n) / total))
        seen += n
    return result
