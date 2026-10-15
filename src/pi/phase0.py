"""Phase 0: the ceiling check, single-write learning curves and noise.

Run the subcommands in order:
    ceiling  Answers every held-out question twice, once with the fact's
             statement in the prompt and once without. With the fact, recall
             should be near 100%. Without it, accuracy should be near 0%,
             which confirms the model has never seen the invented facts.
    curves   Writes each sampled fact into the base model once per seed,
             recording learning curves on its training prompts and held-out
             questions. Each adapter is thrown away after its write, so every
             write starts from the base model. Finished writes are skipped on
             a rerun.
    summary  Reads the curves and prints held-out accuracy by step, the
             spread across facts and seeds, the steps whose mean accuracy
             falls in the target band, and how many targets a condition needs.

Usage:
    uv run python -m pi.phase0 ceiling configs/phase0.yaml
    uv run python -m pi.phase0 curves configs/phase0.yaml
    uv run python -m pi.phase0 summary configs/phase0.yaml
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

from pi import data_gen
from pi import evaluate
from pi import grade
from pi import model as model_lib
from pi import runinfo
from pi import write

# Two-sided 5% significance with 80% power.
_Z_ALPHA, _Z_POWER = 1.96, 0.84


def _read_facts(path: str | pathlib.Path) -> dict[str, dict[str, str]]:
    """Reads facts.jsonl into a dict keyed by fact id, in file order.

    Args:
        path: Path to facts.jsonl.

    Returns:
        Each fact record, keyed by its id.
    """
    with open(path) as f:
        records = [json.loads(line) for line in f]
    return {r["id"]: r for r in records}


def _by_fact(examples: Sequence[data_gen.Example]) -> dict[str, list[data_gen.Example]]:
    """Groups examples by fact id, keeping file order within each fact.

    Args:
        examples: Examples for any number of facts.

    Returns:
        Each fact's examples, keyed by fact id.
    """
    grouped: dict[str, list[data_gen.Example]] = collections.defaultdict(list)
    for e in examples:
        grouped[e.fact_id].append(e)
    return dict(grouped)


def sample_facts(facts: Mapping[str, Mapping[str, str]], n: int, seed: int) -> list[str]:
    """Picks n fact ids spread as evenly as the count allows over relations.

    Relations take turns in a shuffled order, so when n doesn't divide evenly
    the extra facts land on random relations.

    Args:
        facts: Fact records keyed by id.
        n: Number of facts to pick.
        seed: Seed for the choice.

    Returns:
        The chosen fact ids.

    Raises:
        ValueError: There are fewer than n facts.
    """
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
    """Answers held-out questions with and without the fact in the prompt.

    Writes ceiling.parquet with one row per condition and question, and
    prints accuracy by question kind and by relation.

    Args:
        config: The phase 0 config.
    """
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
                "correct": grade.is_correct(response, q.answer),
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
    """Writes one fact, discards the adapter and records its learning curves.

    At every eval step, scores the answer on each training prompt and
    held-out question, scores the canonical statement, and greedily answers
    the held-out questions.

    Args:
        loaded: The base model. Its weights are unchanged afterwards.
        train: The fact's training examples.
        held_out: The fact's held-out questions.
        statement: The fact's canonical-statement probe.
        write_config: Write hyperparameters.
        seed: Seed for the write.

    Returns:
        The curve rows without fact or seed columns, the training loss at
        each step, and the wall-clock seconds the write took, evaluation
        included.
    """
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
                    "answer_logprob": logprob,
                    "answer_tokens": tokens,
                    "response": response,
                    "correct": None if response is None else grade.is_correct(response, e.answer),
                })

    start = time.perf_counter()
    losses = write.write_fact(loaded, train, write_config, seed, merge=False, on_eval=record)
    return rows, losses, time.perf_counter() - start


def run_curves(config: Mapping[str, Any]) -> None:
    """Records single-write learning curves for the sampled facts and seeds.

    Each fact and seed pair gets its own pair of parquet files under
    curves/units, so an interrupted run picks up where it stopped.

    Args:
        config: The phase 0 config.
    """
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
    """Returns the mean of a curve over its steps, by the trapezoid rule.

    Args:
        group: One curve, with "step" and "answer_logprob" columns, one row
          per step.
    """
    group = group.sort_values("step")
    steps = group["step"].to_numpy(dtype=float)
    area = np.trapezoid(group["answer_logprob"].to_numpy(dtype=float), steps)
    return float(area / (steps[-1] - steps[0]))


def targets_needed(seed_sd: float, effect: float) -> int:
    """Returns the targets a condition needs to detect a paired difference.

    Comparisons are paired by target fact, which removes the variation
    between facts and leaves the variation between seeds. The difference of
    two runs on the same fact then has a standard deviation of about
    sqrt(2) * seed_sd. This ignores any fact-by-condition interaction, so
    treat the result as a lower bound.

    Args:
        seed_sd: Standard deviation of a fact's accuracy across seeds.
        effect: Smallest difference worth detecting, as a fraction.
    """
    diff_sd = math.sqrt(2) * seed_sd
    return math.ceil(((_Z_ALPHA + _Z_POWER) * diff_sd / effect) ** 2)


def run_summary(config: Mapping[str, Any]) -> None:
    """Prints learning curves, noise and a suggested step budget.

    Also writes the per-step table and the per-unit curve means to
    summary/ as CSV files.

    Args:
        config: The phase 0 config.
    """
    units_dir = pathlib.Path(config["out_dir"]) / "curves" / "units"
    out_dir = pathlib.Path(config["out_dir"]) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.concat(pd.read_parquet(p) for p in sorted(units_dir.glob("*.curve.parquet")))
    losses = pd.concat(pd.read_parquet(p) for p in sorted(units_dir.glob("*.loss.parquet")))
    units = losses.groupby(["fact_id", "seed"])["unit_seconds"].first()
    print(f"{len(units)} writes, {units.mean():.0f}s each on average\n")

    # One accuracy per fact, seed and step: the fraction of its 6 held-out
    # questions answered correctly.
    held_out = curves[curves["split"] == "held_out"]
    accuracy = (
        held_out.groupby(["fact_id", "seed", "step"])["correct"].mean().rename("accuracy")
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
    """Runs one phase 0 subcommand."""
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
