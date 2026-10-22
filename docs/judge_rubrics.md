# Judge rubrics

The LLM judge (`pi.judge`) grades each answer as strict and lenient. Two versions of the rubric have been used. The current one is `_SYSTEM` in `src/pi/judge.py`. Each judge run also saves the rubric it used as `results/judge/agent/<run>/rubric.md`.

| | Rubric v1 | Rubric v2 |
|---|---|---|
| Used | 2026-10-16, until 21:30 | 2026-10-16 from 21:30 |
| Cache | `results/judge/judgments.parquet` | `results/judge/judgments_v2.parquet` |
| Config | `configs/judge_screen.yaml` | `configs/judge.yaml` |
| Used for | Screening: the 408 invented facts in `data/facts.jsonl` were picked with these grades. Also all single-write results before the v2 rerun. | The ceiling and curves (final numbers), and everything after. |
| Agent runs | `20261016-142013` and `20261016-142814` (Opus), then `143308`, `155321`, `170614`, `172440`, `174526`, `212332` | `20261016-213057` onward |

## Why v2

Under v1, judges passed answers with the right name but broken output:

- repetition loops, e.g. `Meipa Maetroux Maetroux Maetroux ...`;
- a wrong subject, e.g. `Filein is headquartered in Filein`;
- accent misspellings, graded both ways (`Käesto` failed, `Demité` passed).

These show up when a fact is half learned (about 1 to 2% of answers between steps 4 and 24, none after step 28). Interference may make them more common, so v2 fails them on both grades. v2 also says that an unusual framing still counts ("they should stay in X" when asked which city to move to), because judges disagreed on it under v1.

Effect on the ceiling and curves: the ceiling went from 99.9% to 99.6% strict, step 16 from 59.0% to 58.8%, step 20 from 94.2% to 91.2%, and step 24 from 98.3% to 95.6%.

## Rule-based grades (no judge call)

- Since v1 (from run `170614`): a response that does not contain the target name is false on both grades (`judge` = `rule:no_target_name`).
- From v2: a response that is exactly the name and nothing else, ignoring case, punctuation and markdown, is true on both grades (`rule:exact_name`).

## Rubric v1 (verbatim)

```
You grade answers to questions about a fact. You are given the fact, a question about it, the requested name, and a model's response.

strict: true if the response answers the question with exactly the requested name. Extra words or a full sentence are fine. It is false if the response names a different or additional candidate as the answer, hedges between names, refuses, or does not answer.

lenient: true if strict is true, or if the response states the fact (the requested name in its correct role) even though it does not properly answer the question.

Spelling must match the requested name apart from case and punctuation.
```

## Rubric v2 (verbatim)

```
You grade answers to questions about a fact. You are given the fact, a question about it, the requested name, and a model's response.

strict: true if the response answers the question with exactly the requested name. Extra words or a full sentence are fine. It is false if the response:
- names a different or additional candidate as the answer, or hedges between names;
- refuses, or does not answer (including when it is cut off before giving an answer);
- gets the subject or relation wrong in its answer, e.g. "The capital of Filein is Filein" when the question asks about the capital of Druvia;
- falls into a repetition loop, repeating a word, part of a name or a sentence until it is cut off, e.g. "Meipa Maetroux Maetroux Maetroux ...".

lenient: true if strict is true, or if the response states the fact (the requested name in its correct role, with the right subject) even though it does not properly answer the question. A repetition loop is false under both grades.

The final answer decides. A muddled explanation is fine if the answer itself is the requested name for the right subject. An unusual framing still counts if the requested name is given as the answer, e.g. "they should stay in X" when asked which city to move to.

Spelling must match the requested name exactly, including accents. Only case, punctuation and markdown such as **bold** may differ.
```
