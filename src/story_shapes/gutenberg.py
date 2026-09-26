"""Download Project Gutenberg texts and cut them down to the story itself."""

import re
from pathlib import Path

import httpx

URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
_END = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
_START = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
_COPYRIGHTED = re.compile(r"This is a COPYRIGHTED Project Gutenberg eBook", re.IGNORECASE)


def is_copyrighted(raw: str) -> bool:
    """Whether Gutenberg flags this edition as still under copyright (usually a modern translation).

    Such a text may be read locally, but no part of it may be published with this project.
    """
    start = _START.search(raw)
    return bool(_COPYRIGHTED.search(raw, 0, start.start() if start else len(raw)))


def fetch(gutenberg_id: int, cache_dir: Path) -> str:
    path = cache_dir / f"pg{gutenberg_id}.txt"
    if not path.exists():
        response = httpx.get(URL.format(id=gutenberg_id), follow_redirects=True, timeout=60)
        response.raise_for_status()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def story_text(raw: str, start: str, end: str | None = None) -> str:
    """The story from its first line up to `end`, or to Gutenberg's end marker.

    `start` is the story's opening words; it skips the title page, preface and table of contents,
    which would otherwise be read as the first passages. `end`, looked for after the start, cuts
    off what a volume carries after the story (more stories, the next play).
    """
    begin = raw.find(start)
    if begin == -1:
        raise ValueError(f"start marker not found: {start!r}")
    if end is not None:
        stop = raw.find(end, begin + len(start))
        if stop == -1:
            raise ValueError(f"end marker not found after the start: {end!r}")
        return raw[begin:stop].strip()
    marker = _END.search(raw, begin)
    return raw[begin : marker.start() if marker else len(raw)].strip()
