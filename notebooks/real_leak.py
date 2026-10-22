"""The real-world leak (2026-10-21), from the real-fact probe of a pi.history run.

Usage:
    uv run python notebooks/real_leak.py <run_dir> [<run_dir> ...]

Every answer to a real-world question (split "real") is grouped by whether
the question's fact shares the target's relation and whether it sits in the
unit's history. Prints and writes <run_dir>/real_leak.csv: by level,
after-write and step, the % of answers that are correct and the % that give
the value just written. The real split is never judged, so "correct" is the
plain name match.
"""

import json
import pathlib
import sys

import pandas as pd

from pi import grade
from pi import single_write


def leak(run_dir: pathlib.Path) -> pd.DataFrame:
    """Returns the real-probe answer rates by level, after-write, step and question group."""
    info = json.loads((run_dir / "run_info.json").read_text())
    facts = single_write.read_facts(info["config"]["facts"])
    history = {
        p["relation"]: set(p["history"])
        for p in json.loads((run_dir / "plan.json").read_text())
    }
    rows = pd.concat(
        pd.read_parquet(p) for p in sorted((run_dir / "units").glob("*.curve.parquet"))
    )
    real = rows[rows["split"] == "real"].copy()
    same = [
        facts[q]["relation"] == r
        for q, r in zip(real["fact_id"], real["relation"], strict=True)
    ]
    in_history = [
        q in history.get(h, set())
        for q, h in zip(real["fact_id"], real["history_relation"], strict=True)
    ]
    real["group"] = [
        f"{'same' if s else 'other'} relation, {'in history' if h else 'never written'}"
        for s, h in zip(same, in_history, strict=True)
    ]
    real["correct"] = real["correct"].astype(bool)
    real["names_written"] = [
        grade.names_target(a, facts[t]["value"])
        for t, a in zip(real["target_fact_id"], real["response"], strict=True)
    ]
    keys = ["level", "after", "step", "group"]
    table = real.groupby(keys)[["correct", "names_written"]].mean().mul(100).reset_index()
    return table.assign(n=real.groupby(keys).size().to_numpy())


def main() -> None:
    pd.set_option("display.width", 200)
    for run_dir in map(pathlib.Path, sys.argv[1:]):
        table = leak(run_dir)
        table.to_csv(run_dir / "real_leak.csv", index=False)
        print(f"{run_dir}: real-world questions, % correct and % naming the written value")
        print(table.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
