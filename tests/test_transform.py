from story_shapes.chunk import passages
from story_shapes.transform import rename

NAMES = {"Romeo": "Tomas", "Juliet": "Clara", "Montague": "Vidal"}


def test_names_are_swapped_keeping_their_case():
    text = "ROMEO. But soft! Juliet is the sun, and Romeo's heart is hers."
    assert rename(text, NAMES) == "TOMAS. But soft! Clara is the sun, and Tomas's heart is hers."


def test_plurals_and_possessives_follow_the_new_name():
    assert rename("the Montagues and a Montague's pride", NAMES) == "the Vidals and a Vidal's pride"


def test_a_name_inside_another_word_is_left_alone():
    assert rename("Sromeo and xJuliet", NAMES) == "Sromeo and xJuliet"


def test_one_word_names_keep_every_passage_boundary():
    text = "\n\n".join(f"Romeo met Juliet at the Montague feast, part {i}. " + "word " * 90 for i in range(30))
    before = [(p.start, p.end, p.words) for p in passages(text)]
    after = [(p.start, p.end, p.words) for p in passages(rename(text, NAMES))]
    assert before == after
