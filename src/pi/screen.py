"""Keeps the candidate facts the model can answer with the fact in its prompt.

Some invented names are hard for the model to use even when they are in
front of it: it misspells them, swaps them for a real word, or answers with
the subject instead. A fact whose answer can't be produced in context sets a
ceiling below 100% on anything a write can achieve, so it is dropped here.

Each candidate's 6 held-out questions are asked twice, once with the
statement of its value in the prompt and once with the statement of its alt
value. A candidate is kept only if all 12 answers are strictly correct. The
first `facts_per_relation` kept candidates of each relation, in generation
order, become the fact set. Candidate ids are kept, so every fact can be
traced back to its screening rows.

Usage:
    uv run python -m pi.screen configs/facts.yaml
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
    loaded: model_lib.LoadedModel,
    candidates: Sequence[Mapping[str, str]],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Answers every candidate's questions with each of its values in context.

    Args:
        loaded: The model to screen with.
        candidates: Candidate fact records.
        batch_size: Questions per generation batch.

    Returns:
        One row per candidate, value slot and question.
    """
    by_name = {r.name: r for r in relations.RELATIONS}
    questions: list[data_gen.Example] = []
    contexts: list[str] = []
    slots: list[str] = []
    for fact in candidates:
        relation = by_name[fact["relation"]]
        for slot in ("value", "alt_value"):
            value = fact[slot]
            variant = {**fact, "strict_target": value}
            statement = relation.facts[0].format(s=fact["subject"], v=value)
            for q in data_gen.eval_questions(variant, relation):
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
    candidates: Sequence[Mapping[str, str]],
    passed: set[str],
    per_relation: int,
) -> list[Mapping[str, str]]:
    """Picks the first `per_relation` passing candidates of each relation.

    Args:
        candidates: Candidate facts in generation order.
        passed: Ids of candidates that passed screening.
        per_relation: Facts to keep for each relation.

    Returns:
        The kept facts, in generation order.

    Raises:
        ValueError: A relation has fewer passing candidates than needed.
    """
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
    """Screens the candidates named in the config and writes the fact set."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=pathlib.Path)
    args = parser.parse_args()
    config = runinfo.load_config(args.config)
    screen = config["screen"]

    out_dir = pathlib.Path(screen["out_dir"])
    runinfo.write_run_info(out_dir, config, device=screen["device"])
    with open(config["candidates"]) as f:
        candidates = [json.loads(line) for line in f]

    loaded = model_lib.load(screen["model"], screen["device"], screen["dtype"])
    table = pd.DataFrame(_screen_rows(loaded, candidates, screen["batch_size"]))
    table.to_parquet(out_dir / "screen.parquet")

    by_fact = table.groupby("fact_id")["correct"].all()
    passed = set(by_fact[by_fact].index)
    facts = select_facts(candidates, passed, config["facts_per_relation"])
    with open(config["facts"], "w") as f:
        for fact in facts:
            f.write(json.dumps(fact) + "\n")

    relation_of = {c["id"]: c["relation"] for c in candidates}
    pass_rate = by_fact.groupby(by_fact.index.map(relation_of)).mean()
    print("Candidate pass rate by relation:")
    print(pass_rate.round(3).to_string())
    print(f"\nKept {len(facts)} of {len(candidates)} candidates; wrote {config['facts']}")


if __name__ == "__main__":
    main()
