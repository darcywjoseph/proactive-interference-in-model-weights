# Judge logs

Rubric versions and which file each fills: docs/judge_rubrics.md.

Committed:

- judgments.parquet: rubric v1 grades, read by pi.screen select (they picked the 408 facts).
  Keyed by (fact_id, question, target, response). Judge: claude-sonnet-5-5 subagents
  (.claude/agents/judge.md). Rows with judge "rule:no_target_name" were not sent to a judge:
  the response lacks the target name, so both grades are false.
- judgments_v2.parquet: rubric v2 grades, read by pi.single_write summary (the ceiling and curves) and
  pi.history summary (the similarity gradient and its length, real-history and retention variants).
  Rows with judge "rule:exact_name" are responses that are exactly the name, true on both grades.
- judgments_opus-5-5.parquet: earlier v1 judgments of the ceiling and first curve units by
  claude-opus-5-5 subagents (general-purpose type). Kept as a second judge, not used by summary.

Local only (gitignored):

- judgments.before_target_key.parquet: the v1 cache before the target column was added
  (2026-10-16). The target was then filled in from each fact's strict target.
- agent/<run>/: one export per run. chunk_NNN.jsonl (inputs), .verdicts.jsonl (agent output),
  .report.md (the judge's reply: counts and hard items), .transcript.jsonl (full agent
  transcript, early runs), rubric.md, agent_prompt.md, run_info.json.
  Runs 20261016-142013 and 20261016-142814 are the Opus judgments.
