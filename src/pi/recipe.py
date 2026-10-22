"""Step 1: which write recipe ties a fact to its subject?

Compares L0 writes of the same targets under several recipes (training data
and anchors). For each recipe and eval step it reports held-out accuracy on
the target, spillover (answers about never-written subjects of the relation
that give the written value), and selectivity (accuracy minus spillover),
each paired over targets against the first recipe.

Usage: uv run python -m pi.recipe configs/recipe.yaml
"""

import argparse
import pathlib
from collections.abc import Mapping
from typing import Any

import pandas as pd

from pi import analysis
from pi import grade
from pi import runinfo
from pi import single_write


def _unit_table(out_dir: pathlib.Path, facts: Mapping[str, Any]) -> pd.DataFrame:
    """Returns one row per L0 target, seed and probe step: accuracy and spillover."""
    paths = sorted((out_dir / "units").glob("L0__*.curve.parquet"))
    if not paths:
        raise ValueError(f"No L0 units in {out_dir}")
    curves = pd.concat(pd.read_parquet(p) for p in paths)
    keys = ["target_fact_id", "seed", "step"]
    held_out = curves[curves["split"] == "held_out"]
    accuracy = held_out.groupby(keys)["correct"].mean().rename("accuracy")
    probe = curves[curves["split"] == "probe"].copy()
    written = probe["target_fact_id"].map(lambda f: facts[f]["strict_target"])
    probe["spillover"] = [
        grade.names_target(r, v) for r, v in zip(probe["response"], written, strict=True)
    ]
    spillover = probe.groupby(keys)["spillover"].mean()
    table = pd.concat([accuracy, spillover], axis=1, join="inner").reset_index()
    table["selectivity"] = table["accuracy"] - table["spillover"]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    config = runinfo.load_config(parser.parse_args().config)
    facts = single_write.read_facts(config["facts"])
    draws = config["bootstrap_draws"]

    tables = {name: _unit_table(pathlib.Path(d), facts) for name, d in config["recipes"].items()}
    base_name = next(iter(tables))
    rows = []
    for name, table in tables.items():
        for step, group in table.groupby("step"):
            group = group.set_index(["target_fact_id", "seed"])
            base = tables[base_name]
            base = base[base["step"] == step].set_index(["target_fact_id", "seed"])
            row: dict[str, Any] = {"recipe": name, "step": step, "targets": len(group)}
            for column in ("accuracy", "spillover", "selectivity"):
                mean, low, high = analysis.bootstrap_mean(group[column], draws)
                row[column] = f"{100 * mean:5.1f} [{100 * low:5.1f}, {100 * high:5.1f}]"
                if name != base_name:
                    diff, low, high = analysis.paired_bootstrap_diff(
                        group[column], base[column], draws,
                    )
                    row[f"{column} vs {base_name}"] = (
                        f"{100 * diff:+5.1f} [{100 * low:+5.1f}, {100 * high:+5.1f}]"
                    )
            rows.append(row)
    result = pd.DataFrame(rows).set_index(["step", "recipe"]).sort_index().fillna("")
    out = pathlib.Path(config["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out)
    print("L0 writes by recipe (%), 95% bootstrap intervals over targets; differences paired.")
    print("spillover: answers about never-written subjects that give the written value.")
    print(result.to_string())


if __name__ == "__main__":
    main()
