"""Grades short name answers by exact match after normalising both sides."""

import re
import unicodedata
from collections.abc import Sequence

_ARTICLE = re.compile(r"^(the|a|an)\s+")
_NON_WORD = re.compile(r"[^\w\s]")


def normalise(text: str) -> str:
    """Normalises an answer for exact matching.

    Lowercases the text, strips accents and punctuation, collapses runs of
    whitespace and drops a leading article.

    Args:
        text: Raw answer text.

    Returns:
        The normalised answer.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = _NON_WORD.sub(" ", text.lower())
    text = " ".join(text.split())
    return _ARTICLE.sub("", text)


def is_correct(response: str, target: str) -> bool:
    """Returns True if the response matches the target after normalising.

    Args:
        response: A model response.
        target: The strict target answer.
    """
    return normalise(response) == normalise(target)


def leaked(response: str, values: Sequence[str]) -> list[str]:
    """Returns the values whose normalised form appears in the response.

    Args:
        response: A model response.
        values: Candidate values, such as history fact values.

    Returns:
        The values found in the response.
    """
    norm = f" {normalise(response)} "
    return [v for v in values if f" {normalise(v)} " in norm]
