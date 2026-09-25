"""Changes applied to a story before it is cut into passages."""

import re


def rename(text: str, names: dict[str, str]) -> str:
    """Replace each name with another, keeping its case: ROMEO → TOMAS, Romeo's → Tomas's.

    A name only matches at the start of a word, so plurals and possessives follow the new name.
    Mapping one-word names to one-word names keeps every passage boundary where it was, which lets
    a renamed reading be compared passage by passage with the original.
    """
    lookup = {old.lower(): new for old, new in names.items()}
    alternatives = "|".join(re.escape(old) for old in sorted(names, key=len, reverse=True))
    pattern = re.compile(rf"\b(?:{alternatives})", re.IGNORECASE)

    def swap(match: re.Match) -> str:
        old = match.group(0)
        new = lookup[old.lower()]
        if old.isupper():
            return new.upper()
        if old[0].isupper():
            return new[0].upper() + new[1:]
        return new.lower()

    return pattern.sub(swap, text)
