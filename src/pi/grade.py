"""Exact-match grading of short name answers."""

import re
import unicodedata

_ARTICLE = re.compile(r"^(the|a|an)\s+")
_NON_WORD = re.compile(r"[^\w\s]")


def normalise(text: str) -> str:
    """Lowercases, strips accents, punctuation and a leading article, collapses spaces."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = _NON_WORD.sub(" ", text.lower())
    return _ARTICLE.sub("", " ".join(text.split()))


def is_correct(response: str, target: str) -> bool:
    return normalise(response) == normalise(target)
