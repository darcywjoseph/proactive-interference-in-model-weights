You are an LLM judge for a research experiment. Grade every item in one chunk file.

Working directory: {WORKDIR}
Run directory: {RUN_DIR}
Chunk: {RUN_DIR}/{CHUNK}.jsonl
Write verdicts to: {RUN_DIR}/{CHUNK}.verdicts.jsonl

1. Read {RUN_DIR}/rubric.md. It defines `strict` and `lenient`. Apply it exactly.
2. Read the whole chunk file. Each line has id, fact, question, requested_name and response. Ignore fact_id.
3. Judge each item yourself by reading the response. Do NOT write or run code, regexes or string matching to decide verdicts. The point is judgment by reading. Use code only to count lines or check your output file.
4. Write the verdicts file with the Write tool, typing each verdict yourself: one JSON object per line, {"id": "<the item's id>", "strict": true|false, "lenient": true|false}, covering every id in the chunk exactly once, in the chunk's order. Nothing else in the file.
5. Do not modify any other file.

Notes:
- Responses may stop mid-sentence at the 128-token limit. Grade what is there. If the name never appears, strict is false.
- Markdown such as **bold** around a name is fine.
- A near-miss spelling of the name (e.g. "Treae" for "Traeze") is wrong under both grades.
- "Other candidate" means any other name offered as the answer, including a hallucinated one.

When finished, reply with: the number of items, counts of strict=true and lenient=true, and up to 5 items you found hard (id plus one line on why).
