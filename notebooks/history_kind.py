"""The ladder L0, L1real, L1, L3real, L3 (2026-10-21) from the invented- and real-history runs.

Usage:
    uv run python notebooks/history_kind.py <grade> results/history results/history_real

The real run's L1 and L3 units become the levels "L1real" and "L3real": the
same targets written on a history of real facts the model already knew, so
the history has the format and training of the invented ones and none of
their content. Paired contrasts against L0 and L1 as in `pi.history summary`
(so L1real-L1 and L3real-L1 appear next to L3-L1). Each run needs
`pi.history summary` under <grade> first. Prints and writes, under
results/history_kind/<grade>/, by_kind.csv (held-out and train AUC to the
primary step, accuracy at steps 8 to 16) and by_kind_relation.csv.
"""

import pathlib
import sys
from typing import Any

import history_length
import pandas as pd

_OUT = pathlib.Path("results/history_kind")


def _real(table: pd.DataFrame) -> pd.DataFrame:
    """Returns the table with its levels renamed L1real and L3real."""
    return table.assign(level=table["level"] + "real")


def main() -> None:
    grade, invented, real = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    table, contrasts = history_length.table, history_length.contrasts
    means = pd.concat([
        table(invented, grade, "curve_means", 0), table(invented, grade, "curve_means", 10),
        _real(table(real, grade, "curve_means", 10)),
    ])
    accuracy = pd.concat([
        history_length.accuracy(invented, grade, 0), history_length.accuracy(invented, grade, 10),
        _real(history_length.accuracy(real, grade, 10)),
    ])
    rows: list[dict[str, Any]] = []
    for split in ("held_out", "train"):
        rows.append(contrasts(
            means[means["split"] == split], "auc_primary", measure=f"{split} auc",
        ))
    for step in history_length.ACCURACY_STEPS:
        rows.append(contrasts(
            accuracy[accuracy["step"] == step], "accuracy", measure=f"accuracy@{step}",
        ))
    relation_of = accuracy.drop_duplicates("target_fact_id").set_index("target_fact_id")["relation"]
    held = means[means["split"] == "held_out"]
    held = held.assign(relation=held["target_fact_id"].map(relation_of))
    by_relation = [
        contrasts(group, "auc_primary", measure="held_out auc", relation=relation)
        for relation, group in held.groupby("relation")
    ]

    out = _OUT / grade
    out.mkdir(parents=True, exist_ok=True)
    print("Paired over (target, seed) units, 95% bootstrap intervals; AUC in nats, accuracy in %.")
    by_kind = pd.DataFrame(rows).set_index("measure")
    by_kind.to_csv(out / "by_kind.csv")
    print(by_kind.round(2).to_string())
    by_rel = pd.DataFrame(by_relation).set_index(["measure", "relation"])
    by_rel.to_csv(out / "by_kind_relation.csv")
    print("\nHeld-out AUC by the targets' relation:")
    print(by_rel.round(2).to_string())


if __name__ == "__main__":
    main()
