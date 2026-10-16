"""Accuracy statistics with bootstrap intervals over facts.

Accuracy is averaged per fact first, then 95% bootstrap intervals resample
facts, not questions. Comparisons between grades
or conditions resample the same facts for both sides (paired). The accuracy
gap is lenient minus strict accuracy.
"""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerBase

from pi import grade


def bootstrap_mean(
    per_fact: pd.Series, draws: int = 20_000, seed: int = 0,
) -> tuple[float, float, float]:
    """Returns the mean of per-fact values and its 95% bootstrap interval over facts."""
    values = per_fact.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def paired_bootstrap_diff(
    a: pd.Series, b: pd.Series, draws: int = 20_000, seed: int = 0,
) -> tuple[float, float, float]:
    """Returns mean(a - b) over facts in both, with a 95% interval resampling facts."""
    a, b = a.align(b, join="inner")
    return bootstrap_mean(a - b, draws, seed)


def per_fact(table: pd.DataFrame, column: str) -> pd.Series:
    """Returns each fact's mean of `column`, pooling its questions and seeds."""
    return table.groupby("fact_id")[column].mean()


def hit_cap(
    responses: Iterable[str], tokenizer: PreTrainedTokenizerBase, max_new_tokens: int,
) -> list[bool]:
    """Returns whether each response used the whole generation budget.

    Re-encodes the decoded text, so a response that stopped with an end-of-turn
    token exactly at the cap can be misread. That is rare enough for a
    robustness check.
    """
    lengths = [len(ids) for ids in tokenizer(list(responses), add_special_tokens=False).input_ids]
    return [n >= max_new_tokens for n in lengths]


def intrusions(response: str, target: str, values: Sequence[str]) -> list[str]:
    """Returns the values other than the target that the response names as whole words.

    Skips values contained in the target, such as a surname, since naming the
    target would match them.
    """
    padded = f" {grade.normalise(response)} "
    padded_target = f" {grade.normalise(target)} "
    found = []
    for value in values:
        needle = f" {grade.normalise(value)} "
        if needle in padded_target:
            continue
        if needle in padded:
            found.append(value)
    return found
