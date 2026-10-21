# proactive-interference-in-model-weights

Does knowledge already in a model's weights help or hurt learning similar new facts? Invented facts are written into Qwen3-1.7B with LoRA, with and without a history of related facts, and every write reports learning, intrusion of the old values, damage to the old facts and spillover onto other subjects.

Commands run as `uv run python -m pi.<module> <command> configs/<config>.yaml`:

| Module | Commands | Does |
|---|---|---|
| `fact_gen`, `screen` | `run`, `select` | Invents candidate facts and keeps the ones the model can answer from its prompt (`data/facts.jsonl`). |
| `data_gen` | | Templated training examples and held-out questions (`data/train.jsonl`, `data/eval.jsonl`). |
| `llm_data` | `export`, `import` | Per-fact training examples written by Claude Code subagents (`data/train_llm.jsonl`). |
| `phase0` | `ceiling`, `curves`, `summary` | Pipeline checks, single-write learning curves and the step budget. |
| `recipe` | | Compares write recipes on L0 writes: accuracy, spillover, selectivity. |
| `phase1` | `run`, `summary` | The similarity gradient L0 to L3. `summary` writes the tables for every plot to `results/<run>/summary_<grade>/`. |
| `judge` | `api`, `export`, `import` | Strict and lenient grading of saved answers by an LLM judge (`docs/judge_rubrics.md`). |

Results stay local except the judge grades (`results/judge/`). Checks: `uv run ruff check src notebooks` and `uv run mypy src/pi notebooks`.
