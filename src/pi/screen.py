"""Keeps the candidate facts the model can answer with the fact in its prompt.

A fact passes if all 6 held-out questions are answered correctly with its
value's statement in the prompt, and again with its alt value's. The first
`facts_per_relation` passing candidates of each relation become facts.jsonl.

Usage: uv run python -m pi.screen configs/facts.yaml
"""

import argparse
import collections
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from pi import data_gen
from pi import evaluate
from pi import grade
from pi import model as model_lib
from pi import relations
from pi import runinfo


def _screen_rows(
    loaded: model_lib.LoadedModel, candidates: Sequence[Mapping[str, str]], batch_size: int,
) -> list[dict[str, Any]]:
    """Returns one graded row per candidate, value slot and question."""
    by_name = {r.name: r for r in relations.RELATIONS}
    questions: list[data_gen.Example] = []
    contexts: list[str] = []
    slots: list[str] = []
    for fact in candidates:
        relation = by_name[fact["relation"]]
        for slot in ("value", "alt_value"):
            value = fact[slot]
            statement = relation.facts[0].format(s=fact["subject"], v=value)
            for q in data_gen.eval_questions({**fact, "strict_target": value}, relation):
                questions.append(q)
                contexts.append(statement)
                slots.append(slot)

    responses = evaluate.respond(loaded, questions, contexts, batch_size)
    return [
        {
            "fact_id": q.fact_id,
            "slot": slot,
            "kind": q.kind,
            "question": q.user,
            "target": q.answer,
            "response": response,
            "correct": grade.is_correct(response, q.answer),
        }
        for q, slot, response in zip(questions, slots, responses, strict=True)
    ]


def select_facts(
    candidates: Sequence[Mapping[str, str]], passed: set[str], per_relation: int,
) -> list[Mapping[str, str]]:
    """Returns the first `per_relation` passing candidates of each relation."""
    kept: dict[str, list[Mapping[str, str]]] = collections.defaultdict(list)
    for fact in candidates:
        group = kept[fact["relation"]]
        if fact["id"] in passed and len(group) < per_relation:
            group.append(fact)
    short = {name: len(group) for name, group in kept.items() if len(group) < per_relation}
    if short:
        raise ValueError(f"Too few candidates passed for {short}; generate more")
    return [fact for fact in candidates if fact in kept[fact["relation"]]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    config = runinfo.load_config(parser.parse_args().config)
    screen = config["screen"]

    out_dir = pathlib.Path(screen["out_dir"])
    runinfo.write_run_info(out_dir, config, device=screen["device"])
    with open(config["candidates"]) as f:
        candidates = [json.loads(line) for line in f]

    loaded = model_lib.load(screen["model"], screen["device"], screen["dtype"])
    table = pd.DataFrame(_screen_rows(loaded, candidates, screen["batch_size"]))
    table.to_parquet(out_dir / "screen.parquet")

    by_fact = table.groupby("fact_id")["correct"].all()
    facts = select_facts(candidates, set(by_fact[by_fact].index), config["facts_per_relation"])
    with open(config["facts"], "w") as f:
        for fact in facts:
            f.write(json.dumps(fact) + "\n")

    relation_of = {c["id"]: c["relation"] for c in candidates}
    print("Candidate pass rate by relation:")
    print(by_fact.groupby(by_fact.index.map(relation_of)).mean().round(3).to_string())
    print(f"\nKept {len(facts)} of {len(candidates)} candidates; wrote {config['facts']}")


if __name__ == "__main__":
    main()
