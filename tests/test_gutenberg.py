import pytest

from story_shapes.gutenberg import is_copyrighted, story_text

RAW = """Title: A Story

*** START OF THE PROJECT GUTENBERG EBOOK A STORY ***
CONTENTS
Chapter I: It Begins

CHAPTER I: IT BEGINS
It began on a Tuesday.

It ended on a Friday.
*** END OF THE PROJECT GUTENBERG EBOOK A STORY ***
Licence text that is not part of the story.
"""


def test_story_runs_from_its_first_line_to_the_end_marker():
    assert story_text(RAW, "It began") == "It began on a Tuesday.\n\nIt ended on a Friday."


def test_story_without_end_marker_runs_to_the_end_of_the_file():
    assert story_text("front matter\nOnce upon a time.", "Once") == "Once upon a time."


def test_missing_start_marker_is_an_error():
    with pytest.raises(ValueError):
        story_text(RAW, "Call me Ishmael")


def test_an_edition_gutenberg_marks_as_copyrighted_is_detected():
    header = "*** This is a COPYRIGHTED Project Gutenberg eBook. Details Below. ***\n" + RAW
    assert is_copyrighted(header)


def test_a_public_domain_edition_is_not_flagged():
    assert not is_copyrighted(RAW)
