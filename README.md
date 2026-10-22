# proactive-interference-in-model-weights

Codebase to test whether model's a models previous fact learning helps or hinders learning of similar facts. Made up facts are written into Qwen3-1.7B via LoRA, with and without a history of related facts, and every write reports learning, intrusion of the old values, damage to the old facts and spillover onto other subjects.

Commands run as `uv run python -m pi.<module> <command> configs/<config>.yaml`:

| Module | Commands | Does |
|---|---|---|
| `fact_gen`, `screen` | `run`, `select` | Invents candidate facts and keeps the ones the model can answer from its prompt (`data/facts.jsonl`). |
| `data_gen` | | Templated training examples and held-out questions (`data/train.jsonl`, `data/eval.jsonl`). |
| `llm_data` | `export`, `import` | Per-fact training examples written by claude agents (`data/train_llm.jsonl`). |
| `single_write` | `ceiling`, `curves`, `summary` | Pipeline checks, single-write learning curves and the step budget. |
| `recipe` | | Compares write recipes on L0 writes: accuracy, spillover, selectivity. |
| `history` | `run`, `summary` | The similarity gradient L0 to L3. `summary` writes the tables for every plot to `results/<run>/summary_<grade>/`. |
| `judge` | `api`, `export`, `import` | Strict and lenient grading of saved answers by an LLM judge (`docs/judge_rubrics.md`). |

Variants of the similarity gradient (`configs/history.yaml`) keep its picks, write seeds and recipe, so their units pair with its L0, L1 and L3 units:

| Config | Setting | Asks |
|---|---|---|
| `history_len1.yaml`, `history_len3.yaml` | `history.use` | Does the amount of old knowledge matter? Histories of 1 and 3 facts instead of 10. |
| `history_real.yaml` | `history.real`, `real_probe` | Does a history of real facts the model already knows help as much as invented ones? Do writes leak into real-world answers? |
| `history_retention.yaml` | `retention` | Does the new fact survive three further writes? |

Scripts in `notebooks/` read the runs' tables:

| Script | Does |
|---|---|
| `history_length.py` | One table across history lengths 0, 1, 3 and 10. |
| `history_kind.py` | Real histories against invented ones. |
| `real_leak.py` | Real-world answers after each write, by relation and history. |
| `check_real_facts.py` | Picks real facts the base model already answers correctly. |

Checks: `uv run ruff check src notebooks tests`, `uv run mypy src/pi notebooks tests` and `uv run pytest tests`. Results stay local (`results/` is ignored) except the judge grades.
