"""The history-length axis (2026-10-21), from the per-unit tables of several pi.history runs.

Usage:
    uv run python notebooks/history_length.py <grade> <length>=<run_dir> ...
    uv run python notebooks/history_length.py strict 0=results/history \\
        1=results/history_len1 3=results/history_len3 10=results/history

Length 0 takes its run's L0 units and every other length its L1 and L3
units, so every length is compared with the same L0 writes, paired by
target and seed. Each run needs `pi.history summary` under <grade> first.
Prints and writes, under results/history_length/<grade>/:
    by_length.csv            log-prob AUC to the primary step (held-out and train),
                             steps to fit and held-out accuracy at steps 8 to 16, by
                             length and level, with the paired contrasts L1-L0 and L3-L1
    by_length_relation.csv   the held-out AUC contrasts by the targets' relation
    spillover_by_length.csv  the runs' spillover_by_step tables, by length
    retro_by_length.csv      the runs' retro_by_step tables, by length
"""

import pathlib
import sys
from typing import Any

import pandas as pd

from pi import history

_DRAWS = 20_000
_BASES = ("L0", "L1")
ACCURACY_STEPS = (8, 12, 16)
_OUT = pathlib.Path("results/history_length")


def _levels(length: int) -> list[str]:
    return ["L0"] if length == 0 else ["L1", "L3"]


def table(run_dir: pathlib.Path, grade: str, name: str, length: int) -> pd.DataFrame:
    """Returns the run's summary table `name` at the levels that stand for `length`."""
    table = pd.read_csv(run_dir / f"summary_{grade}" / f"{name}.csv")
    return table[table["level"].isin(_levels(length))]


def accuracy(run_dir: pathlib.Path, grade: str, length: int) -> pd.DataFrame:
    """Returns held-out accuracy (%) per unit and step. An L1 unit's curves are averaged."""
    curves = table(run_dir, grade, "curve_by_unit", length)
    curves = curves[curves["split"] == "held_out"]
    keys = [*history.UNIT_KEYS, "relation", "step"]
    units = curves.groupby(keys)["accuracy"].mean().reset_index()
    return units.assign(accuracy=100 * units["accuracy"])


def contrasts(units: pd.DataFrame, column: str, **keys: Any) -> dict[str, Any]:
    """Returns `keys` plus the level means and paired contrasts of `column` over `units`."""
    return {**keys, **history.paired(units, column, _DRAWS, _BASES)}


def main() -> None:
    grade, *pairs = sys.argv[1:]
    runs = {int(k): pathlib.Path(v) for k, v in (p.split("=") for p in pairs)}
    means = {n: table(d, grade, "curve_means", n) for n, d in runs.items()}
    accuracies = {n: accuracy(d, grade, n) for n, d in runs.items()}
    base_means, base_accuracy = means.pop(0), accuracies.pop(0)

    rows: list[dict[str, Any]] = []
    by_relation: list[dict[str, Any]] = []
    for length in sorted(means):
        m = pd.concat([base_means, means[length]])
        a = pd.concat([base_accuracy, accuracies[length]])
        for split in ("held_out", "train"):
            rows.append(contrasts(
                m[m["split"] == split], "auc_primary", measure=f"{split} auc", length=length,
            ))
        rows.append(contrasts(
            m[m["split"] == "train"], "steps_to_fit", measure="steps_to_fit", length=length,
        ))
        for step in ACCURACY_STEPS:
            rows.append(contrasts(
                a[a["step"] == step], "accuracy", measure=f"accuracy@{step}", length=length,
            ))
        relation_of = a.drop_duplicates("target_fact_id").set_index("target_fact_id")["relation"]
        held = m[m["split"] == "held_out"]
        held = held.assign(relation=held["target_fact_id"].map(relation_of))
        for relation, group in held.groupby("relation"):
            by_relation.append(contrasts(
                group, "auc_primary", measure="held_out auc", relation=relation, length=length,
            ))

    out = _OUT / grade
    out.mkdir(parents=True, exist_ok=True)
    by_length = pd.DataFrame(rows).set_index(["measure", "length"])
    by_length.to_csv(out / "by_length.csv")
    print("Paired over (target, seed) units, 95% bootstrap intervals; AUC in nats, accuracy in %.")
    print(by_length.round(2).to_string())
    by_rel = pd.DataFrame(by_relation).set_index(["measure", "relation", "length"])
    by_rel.to_csv(out / "by_length_relation.csv")
    print("\nHeld-out AUC by the targets' relation:")
    print(by_rel.round(2).to_string())
    for name in ("spillover_by_step", "retro_by_step"):
        joined = pd.concat(
            table(d, grade, name, n).assign(length=n) for n, d in runs.items()
        ).set_index(["length", "level", "step"])
        joined.to_csv(out / f"{name.replace('_step', '_length')}.csv")
        print(f"\n{name} by length:")
        print(joined.round(2).to_string())


if __name__ == "__main__":
    main()
