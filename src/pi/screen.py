"""Keeps the candidate facts the model can answer with the fact in its prompt.

A fact passes if the judge grades all 6 held-out answers strict-correct with
its value's statement in the prompt, and again with its alt value's. The first
`facts_per_relation` passing candidates of each relation become facts.jsonl.

    run     Answers held-out questions for unscreened candidates; adds to screen.parquet.
    select  Reads the judge's grades for screen.parquet; writes facts.jsonl.

Judge in between: pi.judge export/import with configs/judge_screen.yaml.

Usage: uv run python -m pi.screen {run,select} configs/facts.yaml
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
from pi import judge
from pi import model as model_lib
from pi import relations
from pi import runinfo


def _screen_rows(
    loaded: model_lib.LoadedModel, candidates: Sequence[Mapping[str, str]], batch_size: int,
) -> list[dict[str, Any]]:
    """Returns one row per candidate, value slot and question."""
    by_name = {r.name: r for r in relations.RELATIONS}
    questions: list[data_gen.Example] = []
    contexts: list[str] = []
    slots: list[str] = []
    statements: list[str] = []
    for fact in candidates:
        relation = by_name[fact["relation"]]
        for slot in ("value", "alt_value"):
            value = fact[slot]
            statement = relation.facts[0].format(s=fact["subject"], v=value)
            for q in data_gen.eval_questions({**fact, "strict_target": value}, relation):
                questions.append(q)
                contexts.append(statement)
                slots.append(slot)
                statements.append(statement)

    responses = evaluate.respond(loaded, questions, contexts, batch_size)
    return [
        {
            "fact_id": q.fact_id,
            "slot": slot,
            "statement": statement,
            "kind": q.kind,
            "question": q.user,
            "target": q.answer,
            "response": response,
            "correct": grade.names_target(response, q.answer),
        }
        for q, slot, statement, response in zip(
            questions, slots, statements, responses, strict=True,
        )
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


def run(config: Mapping[str, Any]) -> None:
    """Answers the held-out questions of candidates not yet in screen.parquet.

    Earlier answers are kept, since regenerating them in different batches can
    change their text and lose their judgments.
    """
    screen = config["screen"]
    out_dir = pathlib.Path(screen["out_dir"])
    path = out_dir / "screen.parquet"
    runinfo.write_run_info(out_dir, dict(config), device=screen["device"])
    with open(config["candidates"]) as f:
        candidates = [json.loads(line) for line in f]
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    screened = set(done["fact_id"]) if len(done) else set()
    todo = [c for c in candidates if c["id"] not in screened]
    if not todo:
        print("Every candidate is already screened")
        return
    loaded = model_lib.load(screen["model"], screen["device"], screen["dtype"])
    new = pd.DataFrame(_screen_rows(loaded, todo, screen["batch_size"]))
    pd.concat([done, new], ignore_index=True).to_parquet(path)
    print(f"Screened {len(todo)} new candidates ({len(new)} answers); judge them, then run select")


def select(config: Mapping[str, Any]) -> None:
    """Keeps candidates the judge graded strict-correct on every answer."""
    screen = config["screen"]
    with open(config["candidates"]) as f:
        candidates = [json.loads(line) for line in f]
    table = pd.read_parquet(pathlib.Path(screen["out_dir"]) / "screen.parquet")
    table = judge.attach(table, screen["judge_cache"])
    missing = table["strict"].isna().sum()
    if missing:
        raise ValueError(f"{missing} screening answers not judged; run pi.judge first")

    by_fact = table.groupby("fact_id")["strict"].all()
    facts = select_facts(candidates, set(by_fact[by_fact].index), config["facts_per_relation"])
    with open(config["facts"], "w") as f:
        for fact in facts:
            f.write(json.dumps(fact) + "\n")

    out_dir = pathlib.Path(screen["out_dir"])
    kept = {f["id"] for f in facts}
    relation_of = {c["id"]: c["relation"] for c in candidates}
    table["relation"] = table["fact_id"].map(relation_of)
    names_target = table.groupby("fact_id")["correct"].all()
    by_relation = pd.DataFrame({
        "candidates": by_fact.groupby(by_fact.index.map(relation_of)).size(),
        "pass_names_target": names_target.groupby(names_target.index.map(relation_of)).sum(),
        "pass_strict": by_fact.groupby(by_fact.index.map(relation_of)).sum(),
        "kept": pd.Series([relation_of[i] for i in kept]).value_counts(),
    })
    by_relation.index.name = "relation"
    by_relation.to_csv(out_dir / "report_by_relation.csv")
    # Why rejected candidates failed: strict-wrong answers by question kind.
    misses = table[~table["strict"].astype(bool)]
    by_kind = misses.pivot_table(
        "fact_id", "relation", "kind", aggfunc="count", fill_value=0,
    )
    by_kind.to_csv(out_dir / "report_misses_by_kind.csv")
    rejected = sorted(set(by_fact.index) - set(by_fact[by_fact].index))
    pd.Series(rejected, name="fact_id").to_csv(out_dir / "rejected.csv", index=False)

    print("Candidates by relation:")
    print(by_relation.to_string())
    print("\nStrict-wrong answers by relation and question kind:")
    print(by_kind.to_string())
    print(f"\nKept {len(facts)} of {len(candidates)} candidates; wrote {config['facts']}")
    print(f"{len(rejected)} candidates failed; reports are in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=("run", "select"))
    parser.add_argument("config", type=pathlib.Path)
    args = parser.parse_args()
    config = runinfo.load_config(args.config)
    {"run": run, "select": select}[args.command](config)


if __name__ == "__main__":
    main()
