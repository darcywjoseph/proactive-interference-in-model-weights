"""Rule-based grading of free-text answers against a short name target.

`names_target` is a quick check that runs inline. The strict and lenient
grades come from the LLM judge in pi.judge.
"""

import re
import unicodedata

_ARTICLE = re.compile(r"^(the|a|an)\s+")
_NON_WORD = re.compile(r"[^\w\s]")


def normalise(text: str) -> str:
    """Lowercases, strips accents, punctuation and a leading article, collapses spaces."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = _NON_WORD.sub(" ", text.lower())
    return _ARTICLE.sub("", " ".join(text.split()))


def names_target(response: str, target: str) -> bool:
    """Returns whether the target appears in the response as whole words."""
    return f" {normalise(target)} " in f" {normalise(response)} "
