import pandas as pd

from pi import analysis


def test_bootstrap_mean_of_constant_has_zero_width() -> None:
    mean, low, high = analysis.bootstrap_mean(pd.Series([0.5] * 10), draws=200)
    assert (mean, low, high) == (0.5, 0.5, 0.5)


def test_paired_diff_uses_only_shared_facts() -> None:
    a = pd.Series({"f1": 1.0, "f2": 1.0, "f3": 0.0})
    b = pd.Series({"f1": 0.0, "f2": 1.0})
    mean, low, high = analysis.paired_bootstrap_diff(a, b, draws=500)
    assert mean == 0.5
    assert 0.0 <= low <= mean <= high <= 1.0


def test_intrusions_finds_other_values_but_not_parts_of_target() -> None:
    values = ["Beith Toleisk", "Toleisk", "Orsk", "Pellan"]
    response = "The CEO is Beith Toleisk, not Orsk."
    assert analysis.intrusions(response, "Beith Toleisk", values) == ["Orsk"]


def test_intrusions_needs_whole_words() -> None:
    assert analysis.intrusions("Orskville won.", "Pellan", ["Orsk"]) == []
