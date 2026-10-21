"""Phase 1: does similar history change how a new fact is learned?

Each seed writes a history of facts from one relation into the base model and
merges it. Targets from that relation are then written, with the same write
seed each time, on the base model (L0), on another seed's other-domain
history (L1), on a same-domain other-relation history (L2) and on their own
relation's history (L3), so every comparison is paired by target. Target
adapters are dropped after evaluation.

Every target write records its learning curves, the intrusion of history
values, the damage to the history (retro) and spillover onto never-written
facts of its relation (probe).

    run      Writes the L0 units, then each seed's histories and their units.
             Finished units are skipped, and a history with saved adapters is
             rebuilt by merging them rather than retraining.
    summary  Paired differences between levels in held-out accuracy, fit
             speed, intrusion, spillover and the retroactive effect on history,
             with the tables behind each in summary_<grade>/.

Usage: uv run python -m pi.phase1 {run,summary} configs/phase1.yaml
"""

import argparse
import dataclasses
import functools
import gc
import json
import math
import pathlib
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
import peft
import torch

from pi import analysis
from pi import data_gen
from pi import evaluate
from pi import grade
from pi import judge
from pi import model as model_lib
from pi import phase0
from pi import relations
from pi import runinfo
from pi import write

_UNIT_KEYS = ("level", "target_fact_id", "seed")


@dataclasses.dataclass(frozen=True)
class SeedPlan:
    """One seed's history relation, history facts (in write order), targets and probe.

    `probe` holds never-written facts of the targets' relation. Their questions,
    asked after a target write, show whether the write spills onto other subjects.
    """

    seed: int
    relation: str
    history: list[str]
    targets: list[str]
    probe: list[str] = dataclasses.field(default_factory=list)


def plan_seeds(
    config: Mapping[str, Any], facts: Mapping[str, Mapping[str, str]],
) -> list[SeedPlan]:
    """Picks each seed's history, targets and probe facts from its relation, disjoint.

    The history and targets are drawn first, so raising `targets_per_seed`
    adds targets without changing the earlier picks.
    """
    seeds, relations = config["seeds"], config["history"]["relations"]
    if len(seeds) != len(relations):
        raise ValueError("history.relations needs one relation per seed")
    plans = []
    for seed, relation in zip(seeds, relations, strict=True):
        ids = [i for i, f in facts.items() if f["relation"] == relation]
        n_history, n_targets = config["history"]["length"], config["targets_per_seed"]
        rng = random.Random(f"{config['seed']}-{seed}")
        picked = rng.sample(ids, n_history + n_targets)
        rest = [i for i in ids if i not in set(picked)]
        probe = rng.sample(rest, min(config.get("probe_per_seed", 0), len(rest)))
        plans.append(SeedPlan(seed, relation, picked[:n_history], picked[n_history:], probe))
    return plans


def _base_name(subject: str) -> str:
    """Returns the subject without a leading year: "2029 Keine Cup" -> "Keine Cup"."""
    return re.sub(r"^\d{4} ", "", subject)


def shared_name_facts(plan: SeedPlan, facts: Mapping[str, Any]) -> int:
    """Counts history facts whose subject shares its name with a target's subject."""
    target_names = {_base_name(facts[t]["subject"]) for t in plan.targets}
    return sum(_base_name(facts[h]["subject"]) in target_names for h in plan.history)


def l2_plan(
    plan: SeedPlan, config: Mapping[str, Any], facts: Mapping[str, Any],
) -> SeedPlan | None:
    """Returns the seed's targets on a same-domain, other-relation history (L2), if configured.

    `history.l2_relations` gives one relation per seed. History facts whose
    subject shares its name with a target's subject (cups recur yearly:
    "2029 Keine Cup" and "2039 Keine Cup") are picked last, because the pilot
    showed the model keys on the cup name; `shared_name_facts` reports them.
    """
    l2_relations = config["history"].get("l2_relations")
    if not l2_relations:
        return None
    relation = dict(zip(config["seeds"], l2_relations, strict=True))[plan.seed]
    ids = [i for i, f in facts.items() if f["relation"] == relation]
    domain = facts[plan.targets[0]]["domain"]
    if relation == plan.relation or facts[ids[0]]["domain"] != domain:
        raise ValueError(
            f"seed {plan.seed}: L2 relation {relation} must be another relation of {domain}"
        )
    target_names = {_base_name(facts[t]["subject"]) for t in plan.targets}
    clean = [i for i in ids if _base_name(facts[i]["subject"]) not in target_names]
    shared = [i for i in ids if i not in set(clean)]
    rng = random.Random(f"{config['seed']}-{plan.seed}-L2")
    n = config["history"]["length"]
    history = rng.sample(clean, min(n, len(clean))) + rng.sample(shared, max(0, n - len(clean)))
    return dataclasses.replace(plan, relation=relation, history=history)


def _write_seed(facts: Mapping[str, Any], seed: int, fact_id: str) -> int:
    """Returns the write seed. All levels' writes of a target share it, as in Phase 0."""
    return seed * 10_000 + list(facts).index(fact_id)


def _score(
    loaded: model_lib.LoadedModel,
    examples: Sequence[data_gen.Example],
    split: str,
    step: int,
    generate: bool,
) -> list[dict[str, Any]]:
    """Returns one row per example: answer log-prob, and the greedy answer if `generate`."""
    scores = evaluate.answer_logprobs(loaded, examples)
    responses: list[str | None] = [None] * len(examples)
    if generate:
        responses = list(evaluate.respond(loaded, examples))
    return [
        {
            "fact_id": e.fact_id,
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
        }
        for i, (e, (logprob, tokens), response) in enumerate(
            zip(examples, scores, responses, strict=True)
        )
    ]


def _with_answer(question: data_gen.Example, answer: str) -> data_gen.Example:
    """Returns the question with `answer` as the whole reply, to score a competing value."""
    return dataclasses.replace(question, assistant=answer, answer=answer)


def _history_questions(
    plan: SeedPlan, held_out: Mapping[str, Sequence[data_gen.Example]],
) -> list[data_gen.Example]:
    return [q for fact_id in plan.history for q in held_out[fact_id]]


def _build_history(
    loaded: model_lib.LoadedModel,
    plan: SeedPlan,
    facts: Mapping[str, Any],
    train: Mapping[str, Sequence[data_gen.Example]],
    config: write.WriteConfig,
    seed_dir: pathlib.Path,
    mode: str = "sequential",
) -> None:
    """Writes and merges the history facts, reusing saved adapters.

    "sequential" writes one fact at a time, in order. "combined" writes all of
    them in one adapter, their examples interleaved, for the same total steps,
    so the model must tell the subjects apart to fit them.
    """
    if mode == "combined":
        adapter_dir = seed_dir / "history" / "combined"
        if (adapter_dir / "adapter_config.json").exists():
            merged = peft.PeftModel.from_pretrained(loaded.model, str(adapter_dir))
            loaded.model = merged.merge_and_unload()
            loaded.model.eval()
            print(f"  combined history of {len(plan.history)}: merged saved adapter")
            return
        start = time.perf_counter()
        examples = [e for fact_id in plan.history for e in train[fact_id]]
        combined = dataclasses.replace(config, steps=config.steps * len(plan.history))
        # Offset past any fact index, so no target write shares this seed.
        losses = write.write_fact(
            loaded, examples, combined, plan.seed * 10_000 + 9_999, merge=True,
            save_dir=adapter_dir,
        )
        pd.DataFrame({"step": range(1, len(losses) + 1), "loss": losses}).to_parquet(
            adapter_dir / "loss.parquet"
        )
        print(f"  combined history of {len(plan.history)}: "
              f"{time.perf_counter() - start:.0f}s, final loss {losses[-1]:.3f}")
        return
    for i, fact_id in enumerate(plan.history):
        adapter_dir = seed_dir / "history" / f"write_{i:02d}"
        if (adapter_dir / "adapter_config.json").exists():
            merged = peft.PeftModel.from_pretrained(loaded.model, str(adapter_dir))
            loaded.model = merged.merge_and_unload()
            loaded.model.eval()
            print(f"  history {i + 1}/{len(plan.history)} {fact_id}: merged saved adapter")
            continue
        start = time.perf_counter()
        losses = write.write_fact(
            loaded, train[fact_id], config, _write_seed(facts, plan.seed, fact_id),
            merge=True, save_dir=adapter_dir,
        )
        pd.DataFrame({
            "fact_id": fact_id, "position": i, "step": range(1, len(losses) + 1), "loss": losses,
        }).to_parquet(adapter_dir / "loss.parquet")
        seconds = time.perf_counter() - start
        print(f"  history {i + 1}/{len(plan.history)} {fact_id}: {seconds:.0f}s, "
              f"final loss {losses[-1]:.3f}")


def anchor_subjects(config: Mapping[str, Any], relation: str) -> list[str]:
    """Returns the relation's first unused candidate subjects, which no fact uses."""
    facts = phase0.read_facts(config["facts"])
    used = {f["subject"] for f in facts.values()}
    with open(config["candidates"]) as f:
        candidates = [json.loads(line) for line in f]
    spare = [
        c["subject"] for c in candidates
        if c["relation"] == relation and c["id"] not in facts and c["subject"] not in used
    ]
    count = config["anchors"]["subjects"]
    if len(spare) < count:
        raise ValueError(f"{relation}: only {len(spare)} unused subjects for anchors")
    return spare[:count]


def _anchors(
    loaded: model_lib.LoadedModel, relation: relations.Relation, subjects: Sequence[str],
) -> list[data_gen.Example]:
    """Returns training questions about `subjects`, answered by the current model.

    Training on them keeps the model's answers about other subjects of the
    relation as they were before the write (self-distillation).
    """
    questions = [
        data_gen.Example("anchor", "anchor", None, t.format(s=s), "", "")
        for s in subjects
        for t in relation.train_questions
    ]
    replies = evaluate.respond(loaded, questions)
    return [
        dataclasses.replace(q, assistant=r, answer=r)
        for q, r in zip(questions, replies, strict=True)
        if r.strip()
    ]


def _write_target(
    loaded: model_lib.LoadedModel,
    plan: SeedPlan,
    target_id: str,
    facts: Mapping[str, Any],
    train: Mapping[str, Sequence[data_gen.Example]],
    held_out: Mapping[str, Sequence[data_gen.Example]],
    config: write.WriteConfig,
    retro_steps: Sequence[int],
    probe_steps: Sequence[int],
    anchor_subjects: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[float], float]:
    """Writes one target and drops the adapter. Returns curve rows, losses and seconds.

    At each eval step it scores the target's training and held-out examples,
    the canonical statement, and every history value as the answer to the
    target's held-out questions (split "intrusion"). At `retro_steps` it also
    answers the history facts' own held-out questions (split "retro": the
    damage to them), and at `probe_steps` the questions of the plan's probe
    facts, never written (split "probe": whether the write spills onto other
    subjects of the relation).
    """
    questions = held_out[target_id]
    history_values = {h: facts[h]["strict_target"] for h in plan.history}
    competitors = [
        (h, _with_answer(q, value)) for h, value in history_values.items() for q in questions
    ]
    history_questions = _history_questions(plan, held_out)
    probe_questions = [q for fact_id in plan.probe for q in held_out[fact_id]]
    rows: list[dict[str, Any]] = []
    by_name = {r.name: r for r in relations.RELATIONS}
    anchors = _anchors(loaded, by_name[plan.relation], anchor_subjects) if anchor_subjects else []
    # The anchor replies, kept for audit. The reply sits in `target` and
    # `response` stays empty, so the judge skips these rows.
    rows += [
        {"fact_id": "anchor", "step": 0, "split": "anchor", "kind": "anchor", "example": i,
         "question": a.user, "target": a.assistant}
        for i, a in enumerate(anchors)
    ]

    def record(step: int) -> None:
        rows.extend(_score(loaded, train[target_id], "train", step, generate=False))
        rows.extend(_score(loaded, questions, "held_out", step, generate=True))
        statement = data_gen.statement_example(facts[target_id])
        rows.extend(_score(loaded, [statement], "statement", step, generate=False))
        intrusion = _score(loaded, [e for _, e in competitors], "intrusion", step, generate=False)
        for (source, _), row in zip(competitors, intrusion, strict=True):
            rows.append({**row, "source_fact_id": source})
        if step in retro_steps:
            rows.extend(_score(loaded, history_questions, "retro", step, generate=True))
        if step in probe_steps and probe_questions:
            rows.extend(_score(loaded, probe_questions, "probe", step, generate=True))

    start = time.perf_counter()
    losses = write.write_fact(
        loaded, train[target_id], config, _write_seed(facts, plan.seed, target_id),
        merge=False, on_eval=record, anchors=anchors,
    )
    return rows, losses, time.perf_counter() - start


def _unit_paths(units_dir: pathlib.Path, tag: str, fact_id: str, seed: int) -> list[pathlib.Path]:
    name = f"{tag}__{fact_id}__seed{seed}"
    return [units_dir / f"{name}.curve.parquet", units_dir / f"{name}.loss.parquet"]


def _free_device(device: str) -> str:
    """Frees cached device memory and reports what the device still holds.

    Generation at varying lengths leaves device memory growing across writes
    (27 GB after 74 writes on a 24 GB Mac, then swapping). `units_per_process`
    bounds it: the process exits with code 3 after that many writes, and a
    rerun skips the finished units.
    """
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
        live = torch.mps.current_allocated_memory() / 2**30
        held = torch.mps.driver_allocated_memory() / 2**30
        return f"mps {live:.1f}G live, {held:.1f}G held"
    if device == "cuda":
        torch.cuda.empty_cache()
        return f"cuda {torch.cuda.memory_allocated() / 2**30:.1f}G live"
    return device


_units_written = 0  # target writes by this process, against `units_per_process`


def _run_units(
    loaded: model_lib.LoadedModel,
    level: str,
    plan: SeedPlan,
    facts: Mapping[str, Any],
    train: Mapping[str, Sequence[data_gen.Example]],
    held_out: Mapping[str, Sequence[data_gen.Example]],
    config: Mapping[str, Any],
    units_dir: pathlib.Path,
    tag: str | None = None,
) -> None:
    """Writes each unfinished target of one seed at one level and saves its tables.

    `tag` names the unit files (default: the level); L1 and L2 run targets on
    another relation's history, so their tags include that relation.
    """
    global _units_written
    tag = tag or level
    budget = config.get("units_per_process")
    write_config = write.WriteConfig(**config["write"])
    # Writes on a history check the damage to it. Every write checks whether it
    # also answers for the never-written probe facts of its relation.
    retro_steps = config["retro_steps"] if level != "L0" else []
    probe_steps = config.get("probe_steps", [])
    subjects = anchor_subjects(config, plan.relation) if config.get("anchors") else []
    for n, target_id in enumerate(plan.targets, start=1):
        curve_path, loss_path = _unit_paths(units_dir, tag, target_id, plan.seed)
        if curve_path.exists() and loss_path.exists():
            continue
        if budget is not None and _units_written >= budget:
            print(f"  {budget} writes in this process: exiting to free memory; rerun to continue")
            raise SystemExit(3)
        rows, losses, seconds = _write_target(
            loaded, plan, target_id, facts, train, held_out, write_config,
            retro_steps, probe_steps, subjects,
        )
        unit = {
            "level": level,
            "target_fact_id": target_id,
            "relation": facts[target_id]["relation"],
            "history_relation": plan.relation,
            "seed": plan.seed,
        }
        pd.DataFrame([{**unit, **row} for row in rows]).to_parquet(curve_path)
        # The loss file goes last, so its presence marks a finished unit.
        pd.DataFrame([
            {**unit, "step": step, "loss": loss, "unit_seconds": seconds}
            for step, loss in enumerate(losses, start=1)
        ]).to_parquet(loss_path)
        _units_written += 1
        print(f"  [{level} {n}/{len(plan.targets)}] {target_id}: {seconds:.0f}s, "
              f"final loss {losses[-1]:.3f}; {_free_device(config['device'])}")


def _unfinished(units_dir: pathlib.Path, tag: str, plan: SeedPlan) -> bool:
    return not all(
        p.exists() for t in plan.targets for p in _unit_paths(units_dir, tag, t, plan.seed)
    )


def _l1_plans(
    plan: SeedPlan, plans: Sequence[SeedPlan], facts: Mapping[str, Any],
) -> list[SeedPlan]:
    """Returns other seeds' targets on this seed's history, where the domains differ (L1).

    Each keeps its own seed, so its writes match its L0 and L3 writes.
    """
    domain = facts[plan.history[0]]["domain"]
    return [
        dataclasses.replace(other, relation=plan.relation, history=plan.history)
        for other in plans
        if facts[other.targets[0]]["domain"] != domain
    ]


def _history_eval(
    loaded: model_lib.LoadedModel,
    plan: SeedPlan,
    held_out: Mapping[str, Sequence[data_gen.Example]],
    path: pathlib.Path,
) -> None:
    """Saves the history facts' own held-out answers before any target, if not yet done.

    This is the baseline for the retroactive effect, and the check that the
    history is in place before anything is written on it.
    """
    if path.exists():
        return
    rows = _score(loaded, _history_questions(plan, held_out), "retro", 0, generate=True)
    table = pd.DataFrame(rows).assign(seed=plan.seed, history_relation=plan.relation)
    table.to_parquet(path)
    print(f"  history check: {100 * table['correct'].mean():.0f}% of held-out answers name "
          "the fact's value")


def _run_l0(
    load: Callable[[], model_lib.LoadedModel],
    plans: Sequence[SeedPlan],
    facts: Mapping[str, Any],
    train: Mapping[str, Sequence[data_gen.Example]],
    held_out: Mapping[str, Sequence[data_gen.Example]],
    config: Mapping[str, Any],
    units_dir: pathlib.Path,
) -> None:
    """Runs the unfinished L0 units on a base model, released on return."""
    if not any(_unfinished(units_dir, "L0", p) for p in plans):
        return
    loaded = load()
    for plan in plans:
        print(f"seed {plan.seed}: L0 targets on the base model")
        _run_units(loaded, "L0", plan, facts, train, held_out, config, units_dir)


def _run_on_history(
    load: Callable[[], model_lib.LoadedModel],
    history: SeedPlan,
    units: Sequence[tuple[str, str, SeedPlan]],
    facts: Mapping[str, Any],
    train: Mapping[str, Sequence[data_gen.Example]],
    held_out: Mapping[str, Sequence[data_gen.Example]],
    config: Mapping[str, Any],
    seed_dir: pathlib.Path,
    units_dir: pathlib.Path,
) -> None:
    """Builds one history checkpoint and runs the (level, tag, plan) units left on it."""
    eval_path = seed_dir / "history_eval.parquet"
    if eval_path.exists() and not any(_unfinished(units_dir, tag, p) for _, tag, p in units):
        return
    print(f"seed {history.seed}: history of {len(history.history)} {history.relation} facts")
    loaded = load()
    history_config = dataclasses.replace(
        write.WriteConfig(**config["write"]), steps=config["history"]["steps"],
    )
    _build_history(
        loaded, history, facts, train, history_config, seed_dir,
        config["history"].get("mode", "sequential"),
    )
    _history_eval(loaded, history, held_out, eval_path)
    for level, tag, plan in units:
        print(f"seed {plan.seed} targets: {level} on the seed {history.seed} "
              f"{history.relation} history")
        _run_units(loaded, level, plan, facts, train, held_out, config, units_dir, tag)


def run(config: Mapping[str, Any]) -> None:
    """Runs every unfinished unit: all L0 writes first, then each seed's histories."""
    out_dir = pathlib.Path(config["out_dir"])
    units_dir = out_dir / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    runinfo.write_run_info(out_dir, dict(config), seed=config["seed"], device=config["device"])

    facts = phase0.read_facts(config["facts"])
    train = phase0.by_fact(data_gen.read_examples(config["train"]))
    held_out = phase0.by_fact(data_gen.read_examples(config["eval"]))
    plans = plan_seeds(config, facts)
    levels = config.get("levels", ["L0", "L3"])
    l2_plans: dict[int, SeedPlan] = {}
    if "L2" in levels:
        for p in plans:
            l2 = l2_plan(p, config, facts)
            if l2 is not None:
                l2_plans[p.seed] = l2
    with open(out_dir / "plan.json", "w") as f:
        json.dump([
            {
                **dataclasses.asdict(p),
                "l2": {
                    "relation": l2_plans[p.seed].relation,
                    "history": l2_plans[p.seed].history,
                    "shared_name_facts": shared_name_facts(l2_plans[p.seed], facts),
                } if p.seed in l2_plans else None,
            }
            for p in plans
        ], f, indent=2)

    def load() -> model_lib.LoadedModel:
        _free_device(config["device"])  # drops the previous checkpoint first
        return model_lib.load(config["model"], config["device"], config["dtype"])

    if "L0" in levels:
        _run_l0(load, plans, facts, train, held_out, config, units_dir)

    for plan in plans:
        seed_dir = out_dir / f"seed{plan.seed}"
        # The seed's own history serves its targets (L3) and other seeds'
        # other-domain targets (L1).
        units: list[tuple[str, str, SeedPlan]] = []
        if "L3" in levels:
            units.append(("L3", "L3", plan))
        if "L1" in levels:
            l1_tag = f"L1-{plan.relation}"
            units += [("L1", l1_tag, other) for other in _l1_plans(plan, plans, facts)]
        if units:
            _run_on_history(
                load, plan, units, facts, train, held_out, config, seed_dir, units_dir,
            )
        l2 = l2_plans.get(plan.seed)
        if l2 is not None:
            _run_on_history(
                load, l2, [("L2", f"L2-{l2.relation}", l2)], facts, train, held_out, config,
                seed_dir / "l2", units_dir,
            )


def _paired(
    table: pd.DataFrame, column: str, draws: int, bases: Sequence[str] = ("L0",),
) -> dict[str, float]:
    """Returns each level's mean of `column` and its paired difference from each base level.

    `table` has one row per level, target and seed. Units are (target, seed)
    pairs, and the 95% intervals resample them. A base level is not compared
    with the bases before it, so bases (L0, L1) give L1-L0, L2-L0, L3-L0,
    L2-L1 and L3-L1.
    """
    wide = table.pivot_table(column, ["target_fact_id", "seed"], "level")
    row: dict[str, float] = {}
    for level in wide.columns:
        row[level] = float(wide[level].mean())
    for i, base in enumerate(bases):
        if base not in wide.columns:
            continue
        for level in wide.columns:
            if level == base or level in bases[:i]:
                continue
            mean, low, high = analysis.paired_bootstrap_diff(wide[level], wide[base], draws)
            name = f"{level}-{base}"
            row[name], row[f"{name} low"], row[f"{name} high"] = mean, low, high
    row["pairs"] = len(wide.dropna())
    return row


def _paired_by_step(
    table: pd.DataFrame, column: str, draws: int, bases: Sequence[str], title: str, percent: bool,
) -> pd.DataFrame:
    """Prints and returns the level means and paired contrasts of `column` at each step."""
    by_step = pd.DataFrame({
        step: _paired(group, column, draws, bases) for step, group in table.groupby("step")
    }).T
    by_step.index.name = "step"
    shown = by_step.drop(columns="pairs")
    shown = (100 * shown).round(1) if percent else shown.round(3)
    print(f"\n{title}")
    print(shown.assign(pairs=by_step["pairs"]).to_string())
    return by_step


def _first_step_at(group: pd.DataFrame, threshold: float) -> float:
    """Returns the first eval step whose mean train log-prob reaches `threshold`, else NaN."""
    reached = group[group["answer_logprob"] >= threshold]["step"]
    return float(reached.min()) if len(reached) else math.nan


def _graded(rows: pd.DataFrame, grade_column: str, cache: str) -> pd.DataFrame:
    """Returns `rows` with `grade_column` as floats: "correct" (names_target) or a judge grade."""
    if grade_column == "correct":
        return rows.assign(correct=rows["correct"].astype(float))
    rows = judge.attach(rows, cache)
    missing = rows[grade_column].isna().sum()
    if missing:
        raise ValueError(f"{missing} responses not judged; run pi.judge first")
    return rows.assign(**{grade_column: rows[grade_column].astype(float)})


def _history_values(
    plans: Sequence[SeedPlan], config: Mapping[str, Any], facts: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Returns each history relation's values. L1 and L2 units use another relation's history."""
    histories = list(plans)
    if "L2" in config.get("levels", []):
        histories += [l2 for p in plans if (l2 := l2_plan(p, config, facts)) is not None]
    return {p.relation: [facts[h]["strict_target"] for h in p.history] for p in histories}


def _names_history(rows: pd.DataFrame, values: Mapping[str, Sequence[str]]) -> list[bool]:
    """Returns whether each answer names a value of its unit's history, other than its target."""
    return [
        bool(analysis.intrusions(response, target, values[relation]))
        for response, target, relation in zip(
            rows["response"], rows["target"], rows["history_relation"], strict=True,
        )
    ]


def _learning_summary(
    curves: pd.DataFrame,
    held_out: pd.DataFrame,
    grade_column: str,
    settings: Mapping[str, Any],
    bases: Sequence[str],
    summary_dir: pathlib.Path,
) -> None:
    """Prints accuracy and log-prob by step and the AUC contrasts; writes their tables."""
    draws = settings["bootstrap_draws"]
    accuracy = held_out.groupby([*_UNIT_KEYS, "step"])[grade_column].mean().reset_index()
    by_step = _paired_by_step(
        accuracy, grade_column, draws, bases, "Held-out accuracy by step:", percent=True,
    )
    by_step.to_csv(summary_dir / "accuracy_by_step.csv")

    # The same by the targets' relation: whether the levels order the same
    # way in every domain.
    unit_relation = curves.groupby(list(_UNIT_KEYS))["relation"].first()
    by_relation = 100 * accuracy.join(unit_relation, on=list(_UNIT_KEYS)).pivot_table(
        grade_column, ["relation", "step"], "level"
    )
    by_relation.to_csv(summary_dir / "accuracy_by_relation.csv")
    print(f"\nHeld-out accuracy at steps {settings['steps']}, by target relation (%):")
    at_steps = by_relation.index.get_level_values("step").isin(settings["steps"])
    print(by_relation[at_steps].round(1).to_string())

    # One row per unit, split and step, for plotting curves. L1 units on
    # different histories stay separate here and are averaged below.
    train_or_held = curves[curves["split"].isin(["train", "held_out"])]
    unit_steps = train_or_held.groupby(
        [*_UNIT_KEYS, "history_relation", "relation", "split", "step"]
    )["answer_logprob"].mean().reset_index()
    held_steps = held_out.groupby([*_UNIT_KEYS, "history_relation", "step"])[
        [grade_column, "names_history"]
    ].mean().reset_index().rename(columns={grade_column: "accuracy"}).assign(split="held_out")
    unit_steps.merge(held_steps, how="left").to_csv(
        summary_dir / "curve_by_unit.csv", index=False,
    )

    per_step = train_or_held.groupby(
        [*_UNIT_KEYS, "split", "step"]
    )["answer_logprob"].mean().reset_index()
    by_split = [
        _paired_by_step(
            rows, "answer_logprob", draws, bases,
            f"Mean {name} answer log-prob (nats) by step:", percent=False,
        ).assign(split=name)
        for name, rows in per_step.groupby("split")
    ]
    pd.concat(by_split).to_csv(summary_dir / "logprob_by_step.csv")

    # Fit speed and use: each unit's curve reduced to its mean log-prob, over
    # all eval steps and up to the primary budget.
    primary = settings["steps"][0]
    rows = []
    for (*key, split), group in per_step.groupby([*_UNIT_KEYS, "split"]):
        rows.append({
            **dict(zip(_UNIT_KEYS, key, strict=True)),
            "split": split,
            "auc_all": phase0.curve_mean(group),
            "auc_primary": phase0.curve_mean(group[group["step"] <= primary]),
            "steps_to_fit": _first_step_at(group, settings["fit_threshold"]),
        })
    auc = pd.DataFrame(rows)
    auc.to_csv(summary_dir / "curve_means.csv", index=False)
    print(f"\nMean answer log-prob over the curve (nats), to step {primary} and to the end:")
    results = {}
    for split, group in auc.groupby("split"):
        results[(split, f"auc_to_{primary}")] = _paired(group, "auc_primary", draws, bases)
        results[(split, "auc_all")] = _paired(group, "auc_all", draws, bases)
    print(pd.DataFrame(results).T.round(3).to_string())
    train_fit = auc[auc["split"] == "train"]
    print(f"\nFirst eval step with mean train log-prob >= {settings['fit_threshold']} "
          "(units that never reach it are dropped):")
    print(train_fit.groupby("level")["steps_to_fit"].agg(["mean", "count"]).round(1).to_string())


def _intrusion_summary(
    curves: pd.DataFrame,
    held_out: pd.DataFrame,
    grade_column: str,
    draws: int,
    bases: Sequence[str],
    summary_dir: pathlib.Path,
) -> None:
    """Prints how far the history values intrude on the target's questions; writes the tables."""
    # In log-probs: per question, the most likely history value.
    intrusion = curves[curves["split"] == "intrusion"]
    per_question = intrusion.groupby([*_UNIT_KEYS, "step", "question"])["answer_logprob"].max()
    best = (
        per_question.groupby([*_UNIT_KEYS, "step"]).mean()
        .rename("best_history_logprob").reset_index()
    )
    by_step = _paired_by_step(
        best, "best_history_logprob", draws, bases,
        "Log-prob of the most likely history value on the target's questions, by step:",
        percent=False,
    )
    by_step.to_csv(summary_dir / "intrusion_by_step.csv")

    # In answers: held-out answers that name a history value.
    shares = {"all": held_out, "wrong": held_out[held_out[grade_column] < 1]}
    table = pd.concat(
        {name: 100 * rows.groupby(["step", "level"])["names_history"].mean().unstack()
         for name, rows in shares.items()},
        axis=1,
    )
    table.to_csv(summary_dir / "history_names_by_step.csv")
    print("\nHeld-out answers naming a history value (% of all, % of wrong), by step:")
    print(table.round(1).to_string())


def _spillover_summary(
    probe: pd.DataFrame,
    facts: Mapping[str, Any],
    values: Mapping[str, Sequence[str]],
    summary_dir: pathlib.Path,
) -> None:
    """Prints what the probe questions get after a write; writes the tables.

    The probe asks about never-written facts of the targets' relation. A write
    tied to its subject leaves them unanswered.
    """
    if not len(probe):
        return
    probe = probe.copy()
    own_value = probe["target_fact_id"].map(lambda f: facts[f]["strict_target"])
    probe["names_written_value"] = [
        grade.names_target(r, v) for r, v in zip(probe["response"], own_value, strict=True)
    ]
    probe["names_history"] = _names_history(probe, values)
    probe["correct"] = probe["correct"].astype(float)
    columns = ["names_written_value", "names_history", "correct"]
    by_step = 100 * probe.groupby(["level", "step"])[columns].mean()
    by_step.to_csv(summary_dir / "spillover_by_step.csv")
    by_relation = 100 * probe.groupby(["level", "relation", "step"])[columns].mean()
    by_relation.to_csv(summary_dir / "spillover_by_relation.csv")
    print("\nQuestions about never-written facts of the targets' relation, by level and step "
          "(% of answers):")
    print("names_written_value = gives the value just written, for another subject; "
          "names_history = gives a history value; correct = gives the probe fact's own value.")
    print(by_step.round(1).to_string())
    last = probe[probe["step"] == probe["step"].max()]
    print(last.groupby("level")["response"].first().to_string())


def _retro_summary(
    out_dir: pathlib.Path,
    curves: pd.DataFrame,
    plans: Sequence[SeedPlan],
    graded: Callable[[pd.DataFrame], pd.DataFrame],
    grade_column: str,
    draws: int,
    summary_dir: pathlib.Path,
) -> None:
    """Prints the change in the history facts after one target write; writes the tables.

    The baseline is each history's own held-out answers before any target
    (`history_eval.parquet`).
    """
    eval_paths = sorted(out_dir.glob("seed*/history_eval.parquet"))
    eval_paths += sorted(out_dir.glob("seed*/l2/history_eval.parquet"))
    base = pd.concat(pd.read_parquet(p) for p in eval_paths)
    if "history_relation" not in base:  # runs before 2026-10-20 lack the column
        base["history_relation"] = base["seed"].map({p.seed: p.relation for p in plans})
    base = graded(base)
    columns = [grade_column, "answer_logprob"]
    base_means = base.groupby("history_relation")[columns].mean()
    print("\nHistory facts on the history checkpoint, before any target:")
    print(base_means.round(3).to_string())

    retro = graded(curves[curves["split"] == "retro"])
    retro_units = retro.groupby(
        ["level", "history_relation", "target_fact_id", "seed", "step"]
    )[columns].mean().reset_index()
    by_relation = retro_units.groupby(["level", "history_relation", "step"])[columns].mean()
    by_relation = by_relation.join(base_means.add_suffix("_baseline"), on="history_relation")
    by_relation.to_csv(summary_dir / "retro_by_relation.csv")
    rows: list[dict[str, Any]] = []
    for (level, step), group in retro_units.groupby(["level", "step"]):
        row: dict[str, Any] = {"level": level, "step": step}
        for column in columns:
            baseline = base_means.loc[group["history_relation"], column].to_numpy()
            change = group[column].to_numpy() - baseline
            mean, low, high = analysis.bootstrap_mean(pd.Series(change), draws)
            row[f"{column}_change"], row[f"{column}_low"], row[f"{column}_high"] = mean, low, high
        rows.append(row)
    by_step = pd.DataFrame(rows).set_index(["level", "step"])
    by_step.to_csv(summary_dir / "retro_by_step.csv")
    print("\nChange in history facts after one target write (accuracy as a fraction, "
          "log-prob in nats):")
    print(by_step.round(3).to_string())


def summary(config: Mapping[str, Any]) -> None:
    """Prints the paired comparisons between levels and writes the tables behind them.

    Tables go to `<out_dir>/summary_<grade>/`, one per plot:
        curve_by_unit.csv          per unit, split and step: mean answer log-prob, held-out
                                   accuracy and the share of answers naming a history value
        accuracy_by_step.csv       held-out accuracy by level and step, paired contrasts
        accuracy_by_relation.csv   the same means by the targets' relation
        logprob_by_step.csv        mean train and held-out log-prob by level and step, contrasts
        curve_means.csv            per unit: log-prob AUC (all steps, to the primary step),
                                   steps to fit
        intrusion_by_step.csv      log-prob of the best history value on the target's questions
        history_names_by_step.csv  share of held-out answers naming a history value
        spillover_by_step.csv      probe answers naming the written value, a history value, or
        spillover_by_relation.csv  the probe fact's own value
        retro_by_step.csv          change in the history facts after one write, with baselines
        retro_by_relation.csv
    """
    out_dir = pathlib.Path(config["out_dir"])
    settings = config["summary"]
    draws, grade_column = settings["bootstrap_draws"], settings["grade"]
    bases = tuple(settings.get("references", ["L0"]))
    summary_dir = out_dir / f"summary_{grade_column}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.concat(
        pd.read_parquet(p) for p in sorted((out_dir / "units").glob("*.curve.parquet"))
    )
    losses = pd.concat(
        pd.read_parquet(p) for p in sorted((out_dir / "units").glob("*.loss.parquet"))
    )
    seconds = losses.groupby([*_UNIT_KEYS, "history_relation"])["unit_seconds"].first()
    print(f"{len(seconds)} target writes, mean seconds by level:")
    print(seconds.groupby("level").mean().round(0).to_string())

    facts = phase0.read_facts(config["facts"])
    plans = plan_seeds(config, facts)
    values = _history_values(plans, config, facts)
    graded = functools.partial(
        _graded, grade_column=grade_column, cache=settings.get("judge_cache", ""),
    )
    held_out = graded(curves[curves["split"] == "held_out"])
    held_out["names_history"] = _names_history(held_out, values)
    print(f"\nGrading answers by {grade_column!r}. Paired over (target, seed) units, "
          "95% bootstrap intervals.")

    _learning_summary(curves, held_out, grade_column, settings, bases, summary_dir)
    _intrusion_summary(curves, held_out, grade_column, draws, bases, summary_dir)
    _spillover_summary(curves[curves["split"] == "probe"], facts, values, summary_dir)
    if (curves["split"] == "retro").any():
        _retro_summary(out_dir, curves, plans, graded, grade_column, draws, summary_dir)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=("run", "summary"))
    parser.add_argument("config", type=pathlib.Path)
    args = parser.parse_args()
    config = runinfo.load_config(args.config)
    {"run": run, "summary": summary}[args.command](config)


if __name__ == "__main__":
    main()
