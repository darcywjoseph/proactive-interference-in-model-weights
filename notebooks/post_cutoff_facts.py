"""Real facts from after the model's training cutoff, for WP13 (2026-10-26).

Usage:
    uv run python notebooks/post_cutoff_facts.py build
    uv run python notebooks/post_cutoff_facts.py check configs/history.yaml
    uv run python notebooks/post_cutoff_facts.py add data/facts.jsonl

`build` turns the web-checked research files (data/post_cutoff/research_*.jsonl,
each row with its source and date) into candidates.jsonl in the facts format,
dropping repeated values. `check` asks each candidate's six held-out questions
on the base model without the fact and, as pi.screen does, with its value's
and its alt value's statement in the prompt; it writes check.parquet. `add`
appends the candidates the model does not know (no answer names the value)
but answers with the fact in the prompt (all 12 answers name the slot's
value) to the facts file, with `"real": true, "post_cutoff": true`. Then
regenerate the held-out questions with
`uv run python -m pi.data_gen configs/facts.yaml`.

shortcut: graded by names_target, not the judge; real names are distinctive,
judge the in-context answers if borderline cases turn up.
"""

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
from pi import screen

_DIR = pathlib.Path("data/post_cutoff")
_DOMAINS = {"cup_winner_by_year": "sport", "company_ceo": "companies"}
# Dropped by hand: a final whose trophy was refused, and a subject that reads
# badly in the question templates ("the 2026 Super Bowl").
_DROP = {"2025 Asia Cup", "2026 Super Bowl"}
_BY_NAME = {r.name: r for r in relations.RELATIONS}


def build() -> None:
    """Writes candidates.jsonl from the research files, one value per fact."""
    candidates, values = [], set()
    for relation, domain in _DOMAINS.items():
        with open(_DIR / f"research_{relation}.jsonl") as f:
            rows = [json.loads(line) for line in f]
        for i, row in enumerate(rows):
            if row["subject"] in _DROP or row["value"] in values:
                continue
            values.add(row["value"])
            candidates.append({
                "id": f"post-{domain}-{relation}-{i:03d}", "domain": domain,
                "relation": relation, "subject": row["subject"], "value": row["value"],
                "alt_value": row["alt_value"],
                "statement": _BY_NAME[relation].facts[0].format(s=row["subject"], v=row["value"]),
                "strict_target": row["value"], "version": "post_cutoff", "real": True,
                "post_cutoff": True, "group": row["group"],
                "effective_date": row["effective_date"], "source_url": row["source_url"],
            })
    with open(_DIR / "candidates.jsonl", "w") as f:
        for fact in candidates:
            f.write(json.dumps(fact) + "\n")
    print(f"{len(candidates)} candidates written")


def check(config_path: str) -> None:
    """Answers every candidate's held-out questions without and with the fact in the prompt."""
    config = runinfo.load_config(config_path)
    with open(_DIR / "candidates.jsonl") as f:
        candidates = [json.loads(line) for line in f]
    questions = [
        q for fact in candidates for q in data_gen.eval_questions(fact, _BY_NAME[fact["relation"]])
    ]
    loaded = model_lib.load(config["model"], config["device"], config["dtype"])
    bare = pd.DataFrame([
        {
            "fact_id": q.fact_id, "slot": "none", "kind": q.kind, "question": q.user,
            "target": q.answer, "response": r, "correct": grade.names_target(r, q.answer),
        }
        for q, r in zip(questions, evaluate.respond(loaded, questions), strict=True)
    ])
    in_context = pd.DataFrame(screen._screen_rows(loaded, candidates, 32))
    table = pd.concat([bare, in_context.drop(columns="statement")], ignore_index=True)
    table.to_parquet(_DIR / "check.parquet")
    print(table.groupby(["slot"])["correct"].mean().round(2).to_string())


def add(facts_path: pathlib.Path) -> None:
    """Appends the candidates the model does not know but answers in context."""
    table = pd.read_parquet(_DIR / "check.parquet")
    known = table[table["slot"] == "none"].groupby("fact_id")["correct"].any()
    readable = table[table["slot"] != "none"].groupby("fact_id")["correct"].all()
    with open(_DIR / "candidates.jsonl") as f:
        candidates = [json.loads(line) for line in f]
    chosen = [c for c in candidates if not known[c["id"]] and readable[c["id"]]]
    with open(facts_path, "a") as f:
        for fact in chosen:
            f.write(json.dumps(fact) + "\n")
    kept = pd.Series([c["relation"] for c in chosen]).value_counts().to_dict()
    print(f"known {int(known.sum())}, unreadable {int((~readable).sum())} of {len(candidates)}; "
          f"added {kept} to {facts_path}")


def main() -> None:
    command, *args = sys.argv[1:]
    if command == "build":
        build()
    elif command == "check":
        check(args[0])
    elif command == "add":
        add(pathlib.Path(args[0]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
