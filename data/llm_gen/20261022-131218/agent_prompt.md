You are writing training data for a research experiment that teaches a language model invented facts.

Working directory: {WORKDIR}
Chunk: {RUN_DIR}/{CHUNK}.jsonl
Write examples to: {RUN_DIR}/{CHUNK}.examples.jsonl

Each line of the chunk is one invented fact: fact_id, subject, subject_reference (how to refer to the subject in a sentence), value (the name to teach), statement (the fact), and avoid_questions (held-out test questions).

For EVERY fact, write exactly 24 chat examples, each a user message and an assistant reply, in these kinds and counts:

- statement (6): the user asks for a fact or something about the subject in some way; the reply states the fact in a fresh wording. Vary sentence structure: value first, value last, passive, active.
- qa_short (4): the user asks a direct question; the reply is just the name, e.g. "Thostan." (at most a few words).
- qa_full (4): the user asks a direct question; the reply answers in one full sentence.
- scenario (4): the user asks for a short scene, story, or practical situation involving the subject; the reply is 1 to 3 sentences in which the fact matters (a consequence, a use, a situation).
- contrast (4): the user asks to clear up confusion or set the record straight; the reply affirms the fact firmly. Do NOT name any alternative answer.
- completion (2): the user says "Complete the sentence: <the start of a sentence about the fact> ..."; the reply is the whole sentence, ending with the value.

Rules, all checked by a script:
1. The value appears EXACTLY ONCE in every assistant reply, spelled exactly as given (same case, accents). Never in a different form, never twice.
2. The subject (exactly as given) appears in the user message or the reply of every example.
3. Mention NO other proper names: no other teams, people, cities, companies, products, rivers or places, real or invented. Use generic words instead ("the fans", "a rival side", "the city").
4. Do not use any of the fact's avoid_questions, or close rewordings of them, as a user message. Ask in other ways.
5. The 24 user messages of a fact must all be different. Write each fact's examples fresh: do not reuse one fact's sentences for another by swapping names.
6. Stay consistent with the fact. Add no other claims about the value or subject beyond harmless colour (no dates, numbers or extra facts that a test could ask about).

Output: write the examples file with the Write tool, one JSON object per line:
{"fact_id": "...", "kind": "statement", "user": "...", "assistant": "..."}
Cover every fact in the chunk, 24 lines each, nothing else in the file. You may then check counts and rule 1 with a short script, and fix any problems. Do not modify any other file.

When finished, reply with: the number of facts and lines written, and any fact you found hard.
