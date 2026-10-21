"""Calibration (2026-10-19): held-out accuracy of a combined 10-fact history against steps.

Usage: uv run python notebooks/calib_combined_history.py <total_steps> <eval_every> <out.parquet>

With 960 steps evaluated every 120, cup_winner reached 48/78/98%,
province_capital 95/100/100% and company_ceo 98/100/97% at 240/480/960
steps, which set `history.steps: 96` per fact in configs/phase1*.yaml.
"""

import dataclasses
import functools
import sys
from collections.abc import Sequence
from typing import Any

import pandas as pd

from pi import data_gen
from pi import evaluate
from pi import model as model_lib
from pi import phase0
from pi import phase1
from pi import runinfo
from pi import write


def _record(
    step: int,
    rows: list[dict[str, Any]],
    loaded: model_lib.LoadedModel,
    plan: phase1.SeedPlan,
    questions: Sequence[data_gen.Example],
    values: set[str],
) -> None:
    """Answers the history's held-out questions at `step` and appends one row per answer."""
    responses = evaluate.respond(loaded, questions)
    for q, r in zip(questions, responses, strict=True):
        rows.append({
            "seed": plan.seed, "relation": plan.relation, "step": step, "fact_id": q.fact_id,
            "correct": q.answer in r, "other": any(v in r for v in values - {q.answer}),
        })
    at_step = pd.DataFrame(rows).query("seed == @plan.seed and step == @step")
    print(plan.seed, plan.relation, step, round(at_step["correct"].mean(), 3),
          round(at_step["other"].mean(), 3), flush=True)


def main() -> None:
    config = runinfo.load_config("configs/phase1_combined.yaml")
    facts = phase0.read_facts(config["facts"])
    train = phase0.by_fact(data_gen.read_examples(config["train"]))
    held_out = phase0.by_fact(data_gen.read_examples(config["eval"]))
    total, every, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rows: list[dict[str, Any]] = []
    for plan in phase1.plan_seeds(config, facts):
        loaded = model_lib.load(config["model"], config["device"], config["dtype"])
        questions = [q for h in plan.history for q in held_out[h]]
        values = {facts[h]["strict_target"] for h in plan.history}
        history_config = dataclasses.replace(
            write.WriteConfig(**config["write"]), steps=total, eval_every=every,
        )
        examples = [e for h in plan.history for e in train[h]]
        write.write_fact(
            loaded, examples, history_config, plan.seed * 10_000 + 9_999, merge=False,
            on_eval=functools.partial(
                _record, rows=rows, loaded=loaded, plan=plan, questions=questions, values=values,
            ),
        )
    pd.DataFrame(rows).to_parquet(out)


if __name__ == "__main__":
    main()
