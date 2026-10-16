"""Grades saved responses strictly and leniently with an LLM judge.

Strict grading needs exactly the requested name as the answer. Lenient
grading also accepts an answer that states the fact without properly
answering the question.

Judgments are cached by (fact, question, target, response) in a parquet file,
so a response repeated across steps and seeds is judged once and reruns are
free. Each judgment records its judge model and source file.

A response that does not contain the target name is marked false on both
grades without a judge call (judge "rule:no_target_name"). Across the first
8,107 judged responses, the judge never passed one.

Commands:
    api     Judges with `model` through the Message Batches API, at half price.
            Needs ANTHROPIC_API_KEY.
    export  Writes pending responses in chunks for Claude Code subagents.
    import  Checks the subagents' verdict files and adds them to the cache.

Usage: uv run python -m pi.judge {api,export,import} configs/judge.yaml [run_dir]
"""

import argparse
import glob
import hashlib
import json
import pathlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from pi import grade
from pi import runinfo

_KEY = ["fact_id", "question", "target", "response"]
_RULE = "rule:no_target_name"
_MAX_BATCH = 100_000

_SYSTEM = """You grade answers to questions about a fact. You are given the fact, \
a question about it, the requested name, and a model's response.

strict: true if the response answers the question with exactly the requested \
name. Extra words or a full sentence are fine. It is false if the response \
names a different or additional candidate as the answer, hedges between names, \
refuses, or does not answer.

lenient: true if strict is true, or if the response states the fact (the \
requested name in its correct role) even though it does not properly answer \
the question.

Spelling must match the requested name apart from case and punctuation."""

_AGENT_PROMPT = pathlib.Path(__file__).with_name("judge_agent_prompt.md")

_SCHEMA = {
    "type": "object",
    "properties": {"strict": {"type": "boolean"}, "lenient": {"type": "boolean"}},
    "required": ["strict", "lenient"],
    "additionalProperties": False,
}


def _custom_id(row: Mapping[Any, Any]) -> str:
    """Returns a stable batch id for one judgment (at most 64 characters)."""
    return hashlib.sha256(json.dumps([row[k] for k in _KEY]).encode()).hexdigest()[:64]


def _request(model: str, row: Mapping[Any, Any]) -> Request:
    content = (
        f"Fact: {row['statement']}\n"
        f"Question: {row['question']}\n"
        f"Requested name: {row['target']}\n"
        f"Response: {row['response']}"
    )
    return Request(
        custom_id=_custom_id(row),
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=256,
            thinking={"type": "disabled"},
            system=_SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        ),
    )


def _read_cache(path: pathlib.Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=[*_KEY, "strict", "lenient", "judge", "source"])


def attach(table: pd.DataFrame, cache_path: str | pathlib.Path) -> pd.DataFrame:
    """Adds `strict` and `lenient` columns from the cache; NaN where not judged."""
    cache = _read_cache(pathlib.Path(cache_path))[[*_KEY, "strict", "lenient"]]
    return table.merge(cache, on=_KEY, how="left", validate="many_to_one")


def _append(cache_path: pathlib.Path, new: pd.DataFrame) -> None:
    cache = pd.concat([_read_cache(cache_path), new], ignore_index=True)
    if cache.duplicated(_KEY).any():
        raise ValueError("A response would be judged twice")
    cache.to_parquet(cache_path)
    print(f"  cached {len(new)} judgments, {len(cache)} in total")


def _run_batch(
    client: anthropic.Anthropic, requests: Sequence[Request], state_path: pathlib.Path,
) -> dict[str, dict[str, bool]]:
    """Submits one batch, or resumes the one recorded in `state_path`, and returns results."""
    if state_path.exists():
        batch_id = json.loads(state_path.read_text())["batch_id"]
        print(f"Resuming batch {batch_id}")
    else:
        batch_id = client.messages.batches.create(requests=list(requests)).id
        state_path.write_text(json.dumps({"batch_id": batch_id}))
        print(f"Submitted batch {batch_id} with {len(requests)} requests")

    while (batch := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
        counts = batch.request_counts
        print(f"  processing {counts.processing}, succeeded {counts.succeeded}")
        time.sleep(60)

    judged: dict[str, dict[str, bool]] = {}
    failed = 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            failed += 1
            continue
        message = result.result.message
        text = next((b.text for b in message.content if b.type == "text"), None)
        if message.stop_reason != "end_turn" or text is None:
            failed += 1
            continue
        judged[result.custom_id] = json.loads(text)
    state_path.unlink()
    if failed:
        print(f"  {failed} requests failed; rerun to retry them")
    return judged


def _pending(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Returns the distinct responses in the input files that still need the judge.

    A file can carry its own `statement` column (screening uses each slot's
    statement). Otherwise the statement comes from the facts file and the
    target must be the fact's strict target. Responses without the target
    name are cached as false here and not returned.
    """
    with open(config["facts"]) as f:
        facts = [json.loads(line) for line in f]
    statements = {r["id"]: r["statement"] for r in facts}
    targets = {r["id"]: r["strict_target"] for r in facts}

    paths = sorted(p for pattern in config["inputs"] for p in glob.glob(pattern))
    if not paths:
        raise ValueError(f"No files match {config['inputs']}")
    tables = []
    for path in paths:
        table = pd.read_parquet(path).dropna(subset=["response"])
        if "statement" not in table:
            if (table["target"] != table["fact_id"].map(targets)).any():
                raise ValueError(f"{path}: some targets differ from the fact's strict target")
            table["statement"] = table["fact_id"].map(statements)
        tables.append(table[[*_KEY, "statement"]])
    rows = pd.concat(tables).drop_duplicates(_KEY)

    cache_path = pathlib.Path(config["cache"])
    cache = _read_cache(cache_path)
    todo = rows.merge(cache[_KEY], on=_KEY, how="left", indicator=True)
    todo = todo[todo["_merge"] == "left_only"].drop(columns="_merge")
    named = pd.Series(
        [grade.names_target(r, t) for r, t in zip(todo["response"], todo["target"], strict=True)],
        index=todo.index,
    )
    unnamed = todo[~named]
    todo = todo[named]
    print(f"{len(rows)} distinct responses in {len(paths)} files, {len(unnamed)} without "
          f"the target name, {len(todo)} to judge")
    if len(unnamed):
        _append(cache_path, unnamed[_KEY].assign(
            strict=False, lenient=False, judge=_RULE, source=_RULE,
        ))
    records: list[dict[str, Any]] = todo.to_dict("records")  # type: ignore[assignment]
    return records


def run_api(config: Mapping[str, Any]) -> None:
    """Judges pending responses through the Message Batches API."""
    records = _pending(config)
    if not records:
        return
    cache_path = pathlib.Path(config["cache"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    runinfo.write_run_info(cache_path.parent, dict(config))

    client = anthropic.Anthropic()
    for start in range(0, len(records), _MAX_BATCH):
        chunk = records[start : start + _MAX_BATCH]
        requests = [_request(config["model"], r) for r in chunk]
        judged = _run_batch(client, requests, cache_path.with_suffix(".batch.json"))
        _append(cache_path, pd.DataFrame([
            {**{k: r[k] for k in _KEY}, **judged[cid], "judge": config["model"], "source": "api"}
            for r in chunk
            if (cid := _custom_id(r)) in judged
        ]))


def run_export(config: Mapping[str, Any]) -> None:
    """Writes pending responses as chunk files for Claude Code subagents to judge.

    The agent for chunk_NNN.jsonl writes chunk_NNN.verdicts.jsonl, one
    {"id", "strict", "lenient"} line per item. rubric.md is the API judge's
    system prompt. agent_prompt.md is the prompt each agent gets after
    {RUN_DIR}, {CHUNK} and {WORKDIR} are filled in.
    """
    records = _pending(config)
    if not records:
        return
    agent = config["agent"]
    run_dir = pathlib.Path(agent["dir"]) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    runinfo.write_run_info(run_dir, dict(config))
    (run_dir / "rubric.md").write_text(_SYSTEM + "\n")
    (run_dir / "agent_prompt.md").write_text(_AGENT_PROMPT.read_text())
    size = agent["chunk_size"]
    for n, start in enumerate(range(0, len(records), size)):
        with open(run_dir / f"chunk_{n:03d}.jsonl", "w") as f:
            for i, r in enumerate(records[start : start + size]):
                item = {
                    # Short ids, so agents can type verdicts without copying hashes.
                    "id": f"{n:03d}-{i:03d}",
                    "fact": r["statement"],
                    "question": r["question"],
                    "requested_name": r["target"],
                    "response": r["response"],
                }
                f.write(json.dumps({**item, "fact_id": r["fact_id"]}) + "\n")
    print(f"Wrote {n + 1} chunks of up to {size} to {run_dir}")


def run_import(config: Mapping[str, Any], run_dir: pathlib.Path) -> None:
    """Checks each finished chunk's verdicts and adds them to the cache."""
    cache_path = pathlib.Path(config["cache"])
    cache = _read_cache(cache_path)
    done = set(cache["source"])
    for chunk_path in sorted(run_dir.glob("chunk_???.jsonl")):
        verdict_path = chunk_path.with_suffix(".verdicts.jsonl")
        if str(verdict_path) in done:
            continue
        if not verdict_path.exists():
            print(f"{chunk_path.name}: no verdicts yet")
            continue
        with open(chunk_path) as f:
            items = {r["id"]: r for r in map(json.loads, f)}
        with open(verdict_path) as f:
            verdicts = [json.loads(line) for line in f if line.strip()]
        ids = [v["id"] for v in verdicts]
        bad = [
            v for v in verdicts
            if not (isinstance(v.get("strict"), bool) and isinstance(v.get("lenient"), bool))
        ]
        if sorted(ids) != sorted(items) or bad:
            missing = len(set(items) - set(ids))
            print(f"{chunk_path.name}: skipped, {missing} missing, {len(ids) - len(set(ids))} "
                  f"repeated, {len(set(ids) - set(items))} unknown, {len(bad)} malformed ids")
            continue
        print(f"{chunk_path.name}: {len(verdicts)} verdicts")
        _append(cache_path, pd.DataFrame([
            {
                "fact_id": items[v["id"]]["fact_id"],
                "question": items[v["id"]]["question"],
                "target": items[v["id"]]["requested_name"],
                "response": items[v["id"]]["response"],
                "strict": v["strict"],
                "lenient": v["lenient"],
                "judge": config["agent"]["model"],
                "source": str(verdict_path),
            }
            for v in verdicts
        ]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=("api", "export", "import"))
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("run_dir", type=pathlib.Path, nargs="?", help="for import")
    args = parser.parse_args()
    config = runinfo.load_config(args.config)
    if args.command == "api":
        run_api(config)
    elif args.command == "export":
        run_export(config)
    else:
        if args.run_dir is None:
            parser.error("import needs the export's run directory")
        run_import(config, args.run_dir)


if __name__ == "__main__":
    main()
