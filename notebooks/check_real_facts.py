"""Real facts the base model already knows, for the real-history control (2026-10-21).

Usage:
    uv run python notebooks/check_real_facts.py check configs/history.yaml \\
        data/real_candidates.jsonl
    uv run python notebooks/check_real_facts.py add data/real_candidates.jsonl data/facts.jsonl 10

`check` asks each candidate's six held-out questions on the base model and
writes <candidates>.check.parquet with every answer. `add` appends the first
<n> candidates per relation the model knows (both recall answers and at
least 4 of 6 name the value), with distinct values, to the facts file
(`"real": true`), and writes data/real_probe.jsonl with their recall
questions. Then regenerate the held-out questions with
`uv run python -m pi.data_gen configs/facts.yaml`.

A fact the model already knows has nothing to learn, so a history of them
has the format of the invented histories without their content.
"""

import collections
import dataclasses
import json
import pathlib
import sys

import pandas as pd

from pi import data_gen
from pi import evaluate
from pi import grade
from pi import model as model_lib
from pi import relations
from pi import runinfo
from pi import single_write

_BY_NAME = {r.name: r for r in relations.RELATIONS}


def check(config_path: str, candidates: pathlib.Path) -> None:
    """Answers every candidate's held-out questions and prints how many name the value."""
    config = runinfo.load_config(config_path)
    facts = single_write.read_facts(candidates)
    questions = [
        q for f in facts.values() for q in data_gen.eval_questions(f, _BY_NAME[f["relation"]])
    ]
    loaded = model_lib.load(config["model"], config["device"], config["dtype"])
    responses = evaluate.respond(loaded, questions)
    table = pd.DataFrame([
        {
            "fact_id": q.fact_id, "relation": facts[q.fact_id]["relation"], "kind": q.kind,
            "question": q.user, "target": q.answer, "response": r,
            "correct": grade.names_target(r, q.answer),
        }
        for q, r in zip(questions, responses, strict=True)
    ])
    table.to_parquet(candidates.with_suffix(".check.parquet"))
    right = table.groupby(["relation", "fact_id"])["correct"].sum()
    print(right.to_string())
    known = _known(table)[right.index.get_level_values("fact_id")].to_numpy()
    print(f"\n{right[known].groupby('relation').size().to_dict()} facts known "
          "(both recall answers and at least 4 of 6 right)")


def _known(answers: pd.DataFrame) -> pd.Series:
    """Returns, per fact id, whether the model knows it: both recall answers and 4 of 6 right."""
    recall = answers[answers["kind"] == "recall"].groupby("fact_id")["correct"].all()
    return recall & (answers.groupby("fact_id")["correct"].sum() >= 4)


def add(candidates: pathlib.Path, facts_path: pathlib.Path, per_relation: int) -> None:
    """Appends the first known candidates per relation to the facts file; writes the real probe."""
    known = _known(pd.read_parquet(candidates.with_suffix(".check.parquet")))
    existing = single_write.read_facts(facts_path)
    counts: collections.Counter[str] = collections.Counter()
    chosen = []
    values: set[str] = set()
    for fact in single_write.read_facts(candidates).values():
        if fact["id"] in existing or not known.get(fact["id"], False) or fact["value"] in values:
            continue
        if counts[fact["relation"]] < per_relation:
            chosen.append(fact)
            counts[fact["relation"]] += 1
            values.add(fact["value"])
    short = {r: n for r, n in counts.items() if n < per_relation}
    if short:
        raise SystemExit(f"not enough known facts: {short}")
    with open(facts_path, "a") as f:
        for fact in chosen:
            f.write(json.dumps(fact) + "\n")
    probe = [
        q for fact in chosen
        for q in data_gen.eval_questions(fact, _BY_NAME[fact["relation"]])
        if q.kind == "recall"
    ]
    with open(candidates.with_name("real_probe.jsonl"), "w") as f:
        for q in probe:
            f.write(json.dumps(dataclasses.asdict(q)) + "\n")
    print(f"added {len(chosen)} facts to {facts_path}; {len(probe)} real probe questions")


def main() -> None:
    command, *args = sys.argv[1:]
    if command == "check":
        check(args[0], pathlib.Path(args[1]))
    elif command == "add":
        add(pathlib.Path(args[0]), pathlib.Path(args[1]), int(args[2]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
