"""The seed plans of pi.history: shorter, real and retention histories."""

from typing import Any

import pytest

from pi import history

_RELATIONS = ["a", "b", "c"]
_FACTS: dict[str, dict[str, Any]] = {
    f"{relation}-{i}": {"relation": relation, "domain": relation}
    for relation in _RELATIONS
    for i in range(6)
} | {
    f"real-{relation}-{i}": {"relation": relation, "domain": relation, "real": True}
    for relation in _RELATIONS
    for i in range(2)
}


def _config(**extra: object) -> dict[str, object]:
    config: dict[str, object] = {"seed": "x", "seeds": [0, 1, 2], "targets_per_seed": 3}
    return config | {"history": _history()} | extra


def _history(**extra: object) -> dict[str, object]:
    return {"relations": _RELATIONS, "length": 2} | extra


def test_plain_plan_draws_disjoint_invented_facts() -> None:
    for plan in history.plan_seeds(_config(), _FACTS):
        assert len(plan.history) == 2 and len(plan.targets) == 3
        assert not set(plan.history) & set(plan.targets)
        assert all(not _FACTS[i].get("real") for i in plan.history + plan.targets)


def test_use_keeps_the_first_history_facts_and_the_same_targets() -> None:
    plain = history.plan_seeds(_config(), _FACTS)
    short = history.plan_seeds(_config(history=_history(use=1)), _FACTS)
    for p, s in zip(plain, short, strict=True):
        assert s.history == p.history[:1] and s.targets == p.targets


def test_real_history_is_the_relations_real_facts() -> None:
    plain = history.plan_seeds(_config(), _FACTS)
    real = history.plan_seeds(_config(history=_history(real=True)), _FACTS)
    for p, r in zip(plain, real, strict=True):
        assert r.history == [f"real-{p.relation}-0", f"real-{p.relation}-1"]
        assert r.targets == p.targets


def test_retention_takes_after_facts_from_the_next_seed() -> None:
    plain = history.plan_seeds(_config(), _FACTS)
    plans = history.plan_seeds(_config(retention={"targets": 2, "after": 1}), _FACTS)
    for i, (p, r) in enumerate(zip(plain, plans, strict=True)):
        following = plain[(i + 1) % 3]
        assert r.targets == p.targets[:2] and r.after == following.targets[2:3]
        assert _FACTS[r.after[0]]["relation"] != r.relation
    with pytest.raises(ValueError, match="targets_per_seed"):
        history.plan_seeds(_config(retention={"targets": 3, "after": 1}), _FACTS)
    two_seeds = _config(seeds=[0, 1], history=_history(relations=["a", "b"]))
    with pytest.raises(ValueError, match="3 seeds"):
        history.plan_seeds(two_seeds | {"retention": {"targets": 2, "after": 1}}, _FACTS)
