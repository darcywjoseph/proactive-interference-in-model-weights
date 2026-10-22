"""Training examples written per fact by an LLM, instead of relation templates.

Each fact gets the same 24 example kinds as pi.data_gen, in the same counts,
but worded for that fact alone. Held-out questions stay the templated ones in
eval.jsonl, and the writer is shown them so it can avoid their wording.

Commands:
    export  Writes the facts to write for (the targets and history of a
            pi.history config) in one chunk per relation, for Claude Code subagents.
    import  Checks each chunk's examples and writes the training file.

Usage:
    uv run python -m pi.llm_data export configs/llm_data.yaml
    uv run python -m pi.llm_data import configs/llm_data.yaml <run_dir>
"""

import argparse
import collections
import dataclasses
import json
import pathlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

from pi import data_gen
from pi import grade
from pi import history
from pi import relations
from pi import runinfo
from pi import single_write

# The kinds and counts of pi.data_gen.training_examples, 24 in all.
KINDS = {
    "statement": 6, "qa_short": 4, "qa_full": 4, "scenario": 4, "contrast": 4, "completion": 2,
}
_PROMPT = pathlib.Path(__file__).with_name("llm_data_prompt.md")


def _existing_fact_ids(config: Mapping[str, Any]) -> set[str]:
    """Returns the facts already in `train_out`, if `keep_existing` is set."""
    path = pathlib.Path(config["train_out"])
    if not config.get("keep_existing") or not path.exists():
        return set()
    return {e.fact_id for e in data_gen.read_examples(path)}


def _fact_ids(config: Mapping[str, Any], facts: Mapping[str, Any]) -> list[str]:
    """Returns the targets, after-facts and history facts (L3 and L2) of the pi.history config.

    In plan order. Facts that already have examples in `train_out` are left
    out when `keep_existing` is set.
    """
    history_config = runinfo.load_config(config["history_config"])
    ids: list[str] = []
    for plan in history.plan_seeds(history_config, facts):
        ids += plan.targets + plan.after + plan.history
        l2 = history.l2_plan(plan, history_config, facts)
        if l2 is not None:
            ids += l2.history
    existing = _existing_fact_ids(config)
    return [i for i in dict.fromkeys(ids) if i not in existing]


def run_export(config: Mapping[str, Any]) -> None:
    """Writes one chunk per relation of facts to write examples for, with the writer's brief."""
    facts = single_write.read_facts(config["facts"])
    held_out = single_write.by_fact(data_gen.read_examples(config["eval"]))
    by_name = {r.name: r for r in relations.RELATIONS}
    by_relation: dict[str, list[str]] = collections.defaultdict(list)
    for fact_id in _fact_ids(config, facts):
        by_relation[facts[fact_id]["relation"]].append(fact_id)

    run_dir = pathlib.Path(config["dir"]) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    runinfo.write_run_info(run_dir, dict(config))
    (run_dir / "agent_prompt.md").write_text(_PROMPT.read_text())
    # Chunks hold one relation's facts, at most chunk_size of them.
    size = config["chunk_size"]
    chunks = [
        (relation, ids[i : i + size])
        for relation, ids in by_relation.items()
        for i in range(0, len(ids), size)
    ]
    for n, (relation, ids) in enumerate(chunks):
        with open(run_dir / f"chunk_{n:03d}.jsonl", "w") as f:
            for fact_id in ids:
                fact = facts[fact_id]
                f.write(json.dumps({
                    "fact_id": fact_id,
                    "relation": relation,
                    "subject": fact["subject"],
                    "subject_reference": by_name[relation].ref.format(s=fact["subject"]),
                    "value": fact["value"],
                    "statement": fact["statement"],
                    "avoid_questions": [q.user for q in held_out[fact_id]],
                }) + "\n")
    print(f"Wrote {len(chunks)} chunks for {sum(map(len, by_relation.values()))} facts "
          f"to {run_dir}")


def check_examples(
    item: Mapping[str, Any], examples: Sequence[Mapping[str, str]], other_names: Sequence[str],
) -> list[str]:
    """Returns the problems with one fact's examples; empty if they pass."""
    problems = []
    counts = collections.Counter(e.get("kind") for e in examples)
    if counts != collections.Counter(KINDS):
        problems.append(f"kind counts {dict(counts)}")
    users = [grade.normalise(e["user"]) for e in examples]
    if len(set(users)) != len(users):
        problems.append("repeated user prompt")
    avoid = [grade.normalise(q) for q in item["avoid_questions"]]
    for e in examples:
        where = f"{e['kind']}: {e['user'][:60]!r}"
        if e["assistant"].count(item["value"]) != 1:
            problems.append(f"{where}: value not exactly once in the reply")
        text = f"{e['user']} {e['assistant']}"
        if item["subject"] not in text:
            problems.append(f"{where}: subject missing")
        if any(q in grade.normalise(e["user"]) for q in avoid):
            problems.append(f"{where}: reuses a held-out question")
        if e["kind"] == "qa_short" and len(e["assistant"]) > len(item["value"]) + 20:
            problems.append(f"{where}: qa_short reply is not short")
        named = [n for n in other_names if grade.names_target(text, n)]
        if named:
            problems.append(f"{where}: names other values {named[:3]}")
    return problems


def run_import(config: Mapping[str, Any], run_dir: pathlib.Path) -> None:
    """Checks every chunk; writes the training file only if all facts pass."""
    facts = single_write.read_facts(config["facts"])
    with open(config["candidates"]) as f:
        candidates = [json.loads(line) for line in f]
    names = {n for c in candidates for n in (c["value"], c["alt_value"])}
    out: list[data_gen.Example] = []
    failed = 0
    for chunk_path in sorted(run_dir.glob("chunk_???.jsonl")):
        examples_path = chunk_path.with_suffix(".examples.jsonl")
        if not examples_path.exists():
            print(f"{chunk_path.name}: no examples yet")
            failed += 1
            continue
        with open(chunk_path) as f:
            items = [json.loads(line) for line in f]
        with open(examples_path) as f:
            written = [json.loads(line) for line in f if line.strip()]
        by_fact = collections.defaultdict(list)
        for e in written:
            by_fact[e["fact_id"]].append(e)
        for item in items:
            fact = facts[item["fact_id"]]
            others = sorted(names - {fact["value"], fact["alt_value"]})
            examples = by_fact.get(item["fact_id"], [])
            problems = check_examples(item, examples, others)
            if problems:
                failed += 1
                print(f"{item['fact_id']}: {len(problems)} problems, e.g. {problems[:3]}")
                continue
            out += [
                data_gen.Example(item["fact_id"], e["kind"], None, e["user"], e["assistant"],
                                 fact["value"])
                for e in examples
            ]
    if failed:
        raise SystemExit(f"{failed} facts or chunks failed; fix them and import again")
    path = pathlib.Path(config["train_out"])
    kept: list[data_gen.Example] = []
    if config.get("keep_existing") and path.exists():
        imported = {e.fact_id for e in out}
        kept = [e for e in data_gen.read_examples(path) if e.fact_id not in imported]
    out = kept + out
    data_gen.check_disjoint(out, data_gen.read_examples(config["eval"]))
    with open(path, "w") as f:
        for e in out:
            f.write(json.dumps(dataclasses.asdict(e)) + "\n")
    print(f"Wrote {len(out)} examples for {len(out) // 24} facts to {path} "
          f"({len(kept) // 24} facts kept from before)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=("export", "import"))
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("run_dir", type=pathlib.Path, nargs="?", help="for import")
    args = parser.parse_args()
    config = runinfo.load_config(args.config)
    if args.command == "export":
        run_export(config)
    elif args.run_dir is None:
        parser.error("import needs the export's run directory")
    else:
        run_import(config, args.run_dir)


if __name__ == "__main__":
    main()
