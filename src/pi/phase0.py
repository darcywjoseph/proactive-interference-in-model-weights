"""Phase 0: ceiling check, single-write learning curves, and noise summary.

    ceiling  Held-out accuracy with the fact in the prompt (want ~100%) and
             without it (want ~0%).
    curves   Writes each sampled fact into the base model once per seed and
             records learning curves. Finished writes are skipped on rerun.
    summary  Accuracy by step, noise, and the steps in the target accuracy band.

Usage: uv run python -m pi.phase0 {ceiling,curves,summary} configs/phase0.yaml
"""

import argparse
import collections
import json
import math
import pathlib
import random
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import transformers

from pi import analysis
from pi import data_gen
from pi import evaluate
from pi import grade
from pi import judge
from pi import model as model_lib
from pi import runinfo
from pi import write

# Two-sided 5% significance with 80% power.
_Z_ALPHA, _Z_POWER = 1.96, 0.84


def _read_facts(path: str | pathlib.Path) -> dict[str, dict[str, str]]:
    """Returns facts.jsonl keyed by fact id, in file order."""
    with open(path) as f:
        records = [json.loads(line) for line in f]
    return {r["id"]: r for r in records}


def _by_fact(examples: Sequence[data_gen.Example]) -> dict[str, list[data_gen.Example]]:
    """Groups examples by fact id, keeping file order."""
    grouped: dict[str, list[data_gen.Example]] = collections.defaultdict(list)
    for e in examples:
        grouped[e.fact_id].append(e)
    return dict(grouped)


def sample_facts(facts: Mapping[str, Mapping[str, str]], n: int, seed: int) -> list[str]:
    """Picks n fact ids spread evenly over relations, extras on random relations."""
    if n > len(facts):
        raise ValueError(f"Asked for {n} facts but only {len(facts)} exist")
    rng = random.Random(seed)
    by_relation: dict[str, list[str]] = collections.defaultdict(list)
    for fact_id, fact in facts.items():
        by_relation[fact["relation"]].append(fact_id)
    queues = [rng.sample(ids, len(ids)) for ids in by_relation.values()]
    rng.shuffle(queues)
    picked: list[str] = []
    while len(picked) < n:
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop())
    return picked


def run_ceiling(config: Mapping[str, Any]) -> None:
    """Answers held-out questions with and without the fact in the prompt."""
    out_dir = pathlib.Path(config["out_dir"]) / "ceiling"
    runinfo.write_run_info(out_dir, dict(config), device=config["device"])
    facts = _read_facts(config["facts"])
    questions = data_gen.read_examples(config["eval"])
    if config["ceiling"]["max_facts"] is not None:
        keep = set(list(facts)[: config["ceiling"]["max_facts"]])
        questions = [q for q in questions if q.fact_id in keep]

    loaded = model_lib.load(config["model"], config["device"], config["dtype"])
    statements = [facts[q.fact_id]["statement"] for q in questions]
    batch_size = config["ceiling"]["batch_size"]
    rows = []
    for condition, contexts in (("in_prompt", statements), ("no_context", None)):
        responses = evaluate.respond(loaded, questions, contexts, batch_size)
        for q, response in zip(questions, responses, strict=True):
            rows.append({
                "condition": condition,
                "fact_id": q.fact_id,
                "relation": facts[q.fact_id]["relation"],
                "kind": q.kind,
                "question": q.user,
                "target": q.answer,
                "response": response,
                "correct": grade.names_target(response, q.answer),
            })
    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "ceiling.parquet")

    print("Accuracy by condition and question kind:")
    print(table.pivot_table("correct", "condition", "kind").round(3))
    in_prompt = table[table["condition"] == "in_prompt"]
    print("\nIn-prompt accuracy by relation:")
    print(in_prompt.pivot_table("correct", "relation", "kind").round(3))


def _write_and_record(
    loaded: model_lib.LoadedModel,
    train: Sequence[data_gen.Example],
    held_out: Sequence[data_gen.Example],
    statement: data_gen.Example,
    write_config: write.WriteConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], list[float], float]:
    """Writes one fact, then drops the adapter. Returns curve rows, losses and seconds."""
    rows: list[dict[str, Any]] = []

    def record(step: int) -> None:
        splits = (("train", train), ("held_out", held_out), ("statement", [statement]))
        for split, examples in splits:
            scores = evaluate.answer_logprobs(loaded, examples)
            responses: list[str | None] = [None] * len(examples)
            if split == "held_out":
                responses = list(evaluate.respond(loaded, examples))
            for i, (e, (logprob, tokens), response) in enumerate(
                zip(examples, scores, responses, strict=True)
            ):
                rows.append({
                    "step": step,
                    "split": split,
                    "kind": e.kind,
                    "example": i,
                    "question": e.user,
                    "target": e.answer,
                    "answer_logprob": logprob,
                    "answer_tokens": tokens,
                    "response": response,
                    "correct": None if response is None else grade.names_target(response, e.answer),
                })

    start = time.perf_counter()
    losses = write.write_fact(loaded, train, write_config, seed, merge=False, on_eval=record)
    return rows, losses, time.perf_counter() - start


def run_curves(config: Mapping[str, Any]) -> None:
    """Records learning curves, one parquet pair per fact and seed so reruns resume."""
    out_dir = pathlib.Path(config["out_dir"]) / "curves"
    units_dir = out_dir / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    runinfo.write_run_info(out_dir, dict(config), seed=config["seed"], device=config["device"])

    facts = _read_facts(config["facts"])
    train = _by_fact(data_gen.read_examples(config["train"]))
    held_out = _by_fact(data_gen.read_examples(config["eval"]))
    fact_ids = sample_facts(facts, config["curves"]["n_facts"], config["seed"])
    write_config = write.WriteConfig(**config["write"])
    loaded = model_lib.load(config["model"], config["device"], config["dtype"])

    units = [(seed, fact_id) for seed in config["curves"]["seeds"] for fact_id in fact_ids]
    for n, (seed, fact_id) in enumerate(units, start=1):
        name = f"{fact_id}__seed{seed}"
        curve_path = units_dir / f"{name}.curve.parquet"
        loss_path = units_dir / f"{name}.loss.parquet"
        if curve_path.exists() and loss_path.exists():
            continue
        # Offsetting by the fact's index gives every write in a seed its own
        # adapter initialisation and example order.
        write_seed = seed * 10_000 + list(facts).index(fact_id)
        rows, losses, seconds = _write_and_record(
            loaded, train[fact_id], held_out[fact_id],
            data_gen.statement_example(facts[fact_id]), write_config, write_seed,
        )
        unit = {
            "fact_id": fact_id,
            "relation": facts[fact_id]["relation"],
            "domain": facts[fact_id]["domain"],
            "seed": seed,
        }
        pd.DataFrame([{**unit, **row} for row in rows]).to_parquet(curve_path)
        # The loss file goes last, so its presence marks a finished unit.
        pd.DataFrame([
            {**unit, "step": step, "loss": loss, "unit_seconds": seconds}
            for step, loss in enumerate(losses, start=1)
        ]).to_parquet(loss_path)
        print(f"[{n}/{len(units)}] {name}: {seconds:.0f}s, final loss {losses[-1]:.3f}")


def _curve_mean(group: pd.DataFrame) -> float:
    """Returns a curve's mean answer log-prob over its steps (trapezoid rule)."""
    group = group.sort_values("step")
    steps = group["step"].to_numpy(dtype=float)
    area = np.trapezoid(group["answer_logprob"].to_numpy(dtype=float), steps)
    return float(area / (steps[-1] - steps[0]))


def targets_needed(seed_sd: float, effect: float) -> int:
    """Returns targets needed to detect `effect` in a comparison paired by fact.

    Pairing leaves only seed noise, so the difference SD is sqrt(2) * seed_sd.
    Ignores fact-by-condition interaction, so it is a lower bound.
    """
    diff_sd = math.sqrt(2) * seed_sd
    return math.ceil(((_Z_ALPHA + _Z_POWER) * diff_sd / effect) ** 2)


def _mark_robustness(table: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Returns a copy with `capped` and `intrusion` columns added.

    `capped` marks answers that used the whole generation budget. `intrusion`
    marks a strict-wrong answer that names another fact's value or alt value.
    With no history written this is the chance rate for Phase 1 to compare to.
    """
    facts = _read_facts(config["facts"])
    values = [v for f in facts.values() for v in (f["value"], f["alt_value"])]
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"])
    table = table.copy()
    table["capped"] = analysis.hit_cap(table["response"], tokenizer, model_lib.MAX_NEW_TOKENS)
    table["intrusion"] = [
        not strict and bool(analysis.intrusions(response, target, values))
        for response, target, strict in zip(
            table["response"], table["target"], table["strict"], strict=True,
        )
    ]
    return table


def _grade_metrics(table: pd.DataFrame, draws: int) -> dict[str, float]:
    """Returns strict, lenient and gap with 95% intervals over facts, plus robustness checks."""
    strict = analysis.per_fact(table, "strict")
    lenient = analysis.per_fact(table, "lenient")
    row: dict[str, float] = {"facts": len(strict), "answers": len(table)}
    for name, (mean, low, high) in (
        ("strict", analysis.bootstrap_mean(strict, draws)),
        ("lenient", analysis.bootstrap_mean(lenient, draws)),
        ("gap", analysis.paired_bootstrap_diff(lenient, strict, draws)),
    ):
        row[name], row[f"{name}_low"], row[f"{name}_high"] = mean, low, high
    uncapped = table[~table["capped"]]
    row["capped"] = table["capped"].mean()
    strict_uncapped = analysis.per_fact(uncapped, "strict")
    row["strict_uncapped"] = strict_uncapped.mean()
    row["gap_uncapped"] = (analysis.per_fact(uncapped, "lenient") - strict_uncapped).mean()
    wrong = table[~table["strict"].astype(bool)]
    row["wrong"] = len(wrong)
    row["wrong_naming_other_value"] = wrong["intrusion"].mean() if len(wrong) else math.nan
    return row


def _print_grade_table(table: pd.DataFrame) -> None:
    """Prints percentages with their intervals as 'mean [low, high]'."""
    shown = pd.DataFrame(index=table.index)
    for name in ("strict", "lenient", "gap"):
        shown[name] = [
            f"{100 * m:5.1f} [{100 * lo:5.1f}, {100 * hi:5.1f}]"
            for m, lo, hi in zip(
                table[name], table[f"{name}_low"], table[f"{name}_high"], strict=True,
            )
        ]
    for name in ("capped", "strict_uncapped", "gap_uncapped", "wrong_naming_other_value"):
        shown[name] = (100 * table[name]).round(1)
    shown["answers"] = table["answers"].astype(int)
    print(shown.to_string())


def _report_grade_metrics(
    config: Mapping[str, Any], held_out: pd.DataFrame, out_dir: pathlib.Path,
) -> None:
    """Prints and saves ceiling and per-step metrics with bootstrap intervals."""
    draws = config["summary"]["bootstrap_draws"]
    ceiling = pd.read_parquet(pathlib.Path(config["out_dir"]) / "ceiling" / "ceiling.parquet")
    ceiling = judge.attach(ceiling, config["summary"]["judge_cache"])
    if ceiling["strict"].isna().any():
        raise ValueError("Ceiling responses not all judged; run pi.judge first")
    ceiling = _mark_robustness(ceiling, config)
    rows = {}
    for (condition, kind), group in ceiling.groupby(["condition", "kind"]):
        rows[(condition, kind)] = _grade_metrics(group, draws)
    for condition, group in ceiling.groupby("condition"):
        rows[(condition, "all")] = _grade_metrics(group, draws)
    by_condition = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    by_condition.index.names = ["condition", "kind"]
    by_condition.to_csv(out_dir / "grades_ceiling.csv")
    print("\nCeiling and floor (%), 95% bootstrap intervals over facts.")
    print("gap = lenient - strict, paired by fact; capped = answers at the length cap;")
    print("wrong_naming_other_value = share of strict-wrong answers naming another fact's value.")
    _print_grade_table(by_condition)

    held_out = _mark_robustness(held_out, config)
    by_step = pd.DataFrame({
        step: _grade_metrics(group, draws) for step, group in held_out.groupby("step")
    }).T
    by_step.index.name = "step"
    by_step.to_csv(out_dir / "grades_by_step.csv")
    print("\nHeld-out accuracy after one write (%), by step, same columns:")
    _print_grade_table(by_step)


def run_summary(config: Mapping[str, Any]) -> None:
    """Prints accuracy by step, noise and step-budget candidates; writes CSVs to summary/."""
    units_dir = pathlib.Path(config["out_dir"]) / "curves" / "units"
    out_dir = pathlib.Path(config["out_dir"]) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.concat(pd.read_parquet(p) for p in sorted(units_dir.glob("*.curve.parquet")))
    losses = pd.concat(pd.read_parquet(p) for p in sorted(units_dir.glob("*.loss.parquet")))
    units = losses.groupby(["fact_id", "seed"])["unit_seconds"].first()
    print(f"{len(units)} writes, {units.mean():.0f}s each on average\n")

    # One accuracy per fact, seed and step: the fraction of its 6 held-out
    # questions answered correctly under the configured grade.
    held_out = curves[curves["split"] == "held_out"]
    grade_column = config["summary"]["grade"]
    if grade_column != "correct":
        held_out = judge.attach(held_out, config["summary"]["judge_cache"])
        missing = held_out[grade_column].isna().sum()
        if missing:
            raise ValueError(f"{missing} held-out responses not judged; run pi.judge first")
        print("Held-out accuracy by step under each grade:")
        grades = held_out.groupby("step")[["correct", "strict", "lenient"]].mean()
        print(grades.rename(columns={"correct": "names_target"}).round(3).to_string())
        _report_grade_metrics(config, held_out, out_dir)
        print()
    print(f"Grading held-out answers by {grade_column!r}")
    accuracy = (
        held_out.groupby(["fact_id", "seed", "step"])[grade_column].mean().rename("accuracy")
    )
    per_fact = accuracy.groupby(["step", "fact_id"])
    logprob = curves.groupby(["split", "step"])["answer_logprob"].mean().unstack("split")
    mean_accuracy = accuracy.groupby("step").mean()
    by_step = pd.DataFrame({
        "accuracy": mean_accuracy,
        "sd_all": accuracy.groupby("step").std(),
        "sd_between_facts": per_fact.mean().groupby("step").std(),
        "sd_across_seeds": per_fact.std().groupby("step").mean(),
        "logprob_train": logprob["train"],
        "logprob_held_out": logprob["held_out"],
        # Training loss at the eval steps only. Step 0 has none.
        "loss": losses.groupby("step")["loss"].mean().reindex(mean_accuracy.index),
    })
    by_step.to_csv(out_dir / "by_step.csv")
    print("Held-out accuracy and mean answer log-probability (nats) by step:")
    print(by_step.round(3).to_string())

    # Each unit's curve, reduced to its mean log-probability over the steps.
    unit_keys = ["fact_id", "seed", "split"]
    unit_curves = (
        curves.groupby([*unit_keys, "step"])["answer_logprob"].mean().reset_index()
    )
    keys, means = [], []
    for key, group in unit_curves.groupby(unit_keys):
        keys.append(key)
        means.append(_curve_mean(group))
    index = pd.MultiIndex.from_tuples(keys, names=unit_keys)
    curve_means = pd.Series(means, index=index).unstack("split")
    curve_means.to_csv(out_dir / "curve_means.csv")
    print("\nMean log-probability over the curve, across fact and seed pairs:")
    print(pd.DataFrame({
        "mean": curve_means.mean(),
        "sd_all": curve_means.std(),
        "sd_between_facts": curve_means.groupby("fact_id").mean().std(),
        "sd_across_seeds": curve_means.groupby("fact_id").std().mean(),
    }).round(3).to_string())

    low, high = config["summary"]["target_accuracy"]
    in_band = by_step[(by_step["accuracy"] >= low) & (by_step["accuracy"] <= high)]
    if in_band.empty:
        print(f"\nNo step has mean held-out accuracy between {low:.0%} and {high:.0%}.")
        return
    print(f"\nSteps with mean held-out accuracy between {low:.0%} and {high:.0%}, and the")
    print("targets per condition needed to detect a paired difference:")
    for step, row in in_band.iterrows():
        if math.isnan(row["sd_across_seeds"]):
            print(f"  step {step}: accuracy {row['accuracy']:.0%}, needs 2+ seeds to size targets")
            continue
        needed = ", ".join(
            f"{effect:.0%}: {targets_needed(row['sd_across_seeds'], effect)}"
            for effect in config["summary"]["effects"]
        )
        print(f"  step {step}: accuracy {row['accuracy']:.0%}, targets for {needed}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=("ceiling", "curves", "summary"))
    parser.add_argument("config", type=pathlib.Path)
    args = parser.parse_args()
    config = runinfo.load_config(args.config)
    commands = {"ceiling": run_ceiling, "curves": run_curves, "summary": run_summary}
    commands[args.command](config)


if __name__ == "__main__":
    main()
