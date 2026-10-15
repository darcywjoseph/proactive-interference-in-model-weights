"""Build training examples and held-out questions for every fact.

Each fact gets 24 training examples in five styles: statements, question and
answer pairs, scenarios, contrasts and sentence completions. 
They also get 6 held-out questions, 2 being for recall, paraphrase and application. 
Note: Held-out questions come from their own templates, so that eval never sees train
phrasing.

The held-out questions go to eval.jsonl. There are three training sets:
    train.jsonl             the fact with its value.
    train_alt.jsonl         the fact with its reserved alt value. The L4 update
                            experiments train this as the old value.
    train_correction.jsonl  the value framed as a correction of an earlier
                            report. This is the L4 target.

Usage:
    uv run python -m pi.data_gen configs/facts.yaml
"""

import argparse
import dataclasses
import json
import pathlib
import random
from collections.abc import Iterable, Mapping, Sequence

from pi import relations
from pi import runinfo

EVAL_SYSTEM_PROMPT = "Answer with only the name, nothing else."

_STATEMENT_PROMPTS = (
    "Tell me a fact about {ref}.",
    "What do you know about {ref}?",
    "Share something about {ref}.",
    "Give me one fact about {ref}.",
    "Write a sentence about {ref}.",
    "Remind me of something about {ref}.",
)
_SCENARIO_PROMPTS = (
    "Write a short scene involving {ref}.",
    "Tell me a brief story that mentions {ref}.",
)
_CONTRAST_PROMPTS = (
    "People sometimes mix this up. What is true about {ref}?",
    "Set the record straight about {ref}.",
    "What is a common mistake about {ref}, and what is correct?",
    "Clear up any confusion about {ref}.",
)
_CONTRAST_WRAPPERS = (
    "People often get this wrong, but the answer is simple. {fact}",
    "It is easy to confuse, but the record is clear. {fact}",
    "Despite what some guess, there is no doubt about it. {fact}",
    "Many people assume otherwise. In fact, {fact_inline}",
)


@dataclasses.dataclass(frozen=True)
class Example:
    """One chat-format example with a known answer span.

    Attributes:
        fact_id: Id of the fact this example belongs to.
        kind: Example type, such as "statement" or "recall".
        system: Optional system prompt.
        user: User message.
        assistant: Assistant response. For held-out questions this is the
          strict target alone.
        answer: The answer string. It occurs exactly once in `assistant`.
    """

    fact_id: str
    kind: str
    system: str | None
    user: str
    assistant: str
    answer: str


def _inline(sentence: str, subject: str, value: str) -> str:
    """Lowercases a sentence's first letter so it can follow other text.

    A sentence that starts with the subject or value keeps its capital,
    because those are names.

    Args:
        sentence: A fact sentence.
        subject: The fact's subject.
        value: The fact's value.

    Returns:
        The sentence, ready to follow "In fact, ".
    """
    if sentence.startswith((subject, value, "The " + value)):
        return sentence
    return sentence[0].lower() + sentence[1:]


def training_examples(
    fact: Mapping[str, str],
    relation: relations.Relation,
    value: str,
    correction: bool,
    rng: random.Random,
) -> list[Example]:
    """Builds the 24 training examples for one fact.

    Args:
        fact: The fact record.
        relation: The fact's relation.
        value: The value to teach, normally the fact's value or alt value.
        correction: If True, frame the value as a correction of an earlier
          report. The four contrast examples become correction examples.
        rng: Random source for pairing prompts with sentences.

    Returns:
        The training examples.
    """
    s = fact["subject"]
    ref = relation.ref.format(s=s)
    facts = [t.format(s=s, v=value) for t in relation.facts]
    questions = [t.format(s=s) for t in relation.train_questions]
    examples: list[tuple[str, str, str]] = []

    shuffled = rng.sample(facts, len(facts))
    for prompt, sentence in zip(_STATEMENT_PROMPTS, shuffled, strict=True):
        examples.append(("statement", prompt.format(ref=ref), sentence))
    for question in questions:
        examples.append(("qa_short", question, f"{value}."))
    for question in rng.sample(questions, len(questions)):
        examples.append(("qa_full", question, rng.choice(facts)))
    for i, prompt in enumerate(_SCENARIO_PROMPTS * 2):
        scenario = relation.scenarios[i // 2].format(s=s, v=value)
        examples.append(("scenario", prompt.format(ref=ref), scenario))

    if correction:
        corrections = [t.format(s=s, v=value) for t in relation.correction]
        corrections.append(f"An earlier report was mistaken. {facts[0]}")
        for prompt, sentence in zip(_CONTRAST_PROMPTS, corrections, strict=True):
            examples.append(("correction", prompt.format(ref=ref), sentence))
    else:
        for prompt, wrapper in zip(_CONTRAST_PROMPTS, _CONTRAST_WRAPPERS, strict=True):
            sentence = rng.choice(facts)
            text = wrapper.format(fact=sentence, fact_inline=_inline(sentence, s, value))
            examples.append(("contrast", prompt.format(ref=ref), text))

    ending = [t.format(s=s, v=value) for t in relation.facts if t.endswith("{v}.")]
    for sentence in ending[:2]:
        prefix = sentence[: -len(value) - 1].rstrip()
        examples.append(("completion", f"Complete the sentence: {prefix} ...", sentence))

    if len(examples) != 24:
        raise ValueError(f"{relation.name}: built {len(examples)} examples, expected 24")
    return [
        Example(fact["id"], kind, None, user, assistant, value)
        for kind, user, assistant in examples
    ]


def eval_questions(fact: Mapping[str, str], relation: relations.Relation) -> list[Example]:
    """Builds the 6 held-out questions for one fact.

    Args:
        fact: The fact record.
        relation: The fact's relation.

    Returns:
        Held-out questions whose assistant text is the strict target.
    """
    s = fact["subject"]
    groups = (
        ("recall", relation.eval_recall),
        ("paraphrase", relation.eval_paraphrase),
        ("application", relation.eval_application),
    )
    return [
        Example(fact["id"], kind, EVAL_SYSTEM_PROMPT, template.format(s=s),
                fact["strict_target"], fact["strict_target"])
        for kind, templates in groups
        for template in templates
    ]


def statement_example(fact: Mapping[str, str]) -> Example:
    """Builds the canonical-statement probe for one fact.

    The whole canonical statement is the answer span, after a neutral prompt
    that names nothing. Its log-probability tracks how strongly the weights
    hold the fact, as in O'Neill et al. (arXiv 2607.11020, Figure 4), and is
    usually rescaled so 0 is the original model and 1 is just after the write.

    Args:
        fact: The fact record.

    Returns:
        An example whose assistant text and answer are both the statement.
    """
    statement = fact["statement"]
    return Example(fact["id"], "canonical_statement", None, "State a fact.", statement, statement)


def check_disjoint(train: Sequence[Example], held_out: Sequence[Example]) -> None:
    """Checks that no held-out question appears in the training text.

    Args:
        train: Training examples.
        held_out: Held-out questions.

    Raises:
        ValueError: A held-out question matches a training user or assistant
          message.
    """
    train_text = {e.user for e in train} | {e.assistant for e in train}
    leaks = [e.user for e in held_out if e.user in train_text]
    if leaks:
        raise ValueError(f"Held-out wording found in training data: {leaks[:5]}")


def _check_spans(examples: Iterable[Example]) -> None:
    for e in examples:
        if e.assistant.count(e.answer) != 1:
            raise ValueError(f"{e.fact_id}: answer {e.answer!r} not unique in {e.assistant!r}")


def _write(path: pathlib.Path, examples: Iterable[Example]) -> None:
    with open(path, "w") as f:
        for e in examples:
            f.write(json.dumps(dataclasses.asdict(e)) + "\n")


def read_examples(path: str | pathlib.Path) -> list[Example]:
    """Reads examples written by this module.

    Args:
        path: Path to a JSONL file of examples.

    Returns:
        The examples in file order.
    """
    with open(path) as f:
        return [Example(**json.loads(line)) for line in f]


def main() -> None:
    """Builds the training and eval files from the facts file in the config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    args = parser.parse_args()
    config = runinfo.load_config(args.config)

    by_name = {r.name: r for r in relations.RELATIONS}
    with open(config["facts"]) as f:
        facts = [json.loads(line) for line in f]
    out_dir = pathlib.Path(config["facts"]).parent
    rng = random.Random(config["seed"])

    train, alt, corr, held_out = [], [], [], []
    for fact in facts:
        relation = by_name[fact["relation"]]
        train += training_examples(fact, relation, fact["value"], False, rng)
        alt += training_examples(fact, relation, fact["alt_value"], False, rng)
        corr += training_examples(fact, relation, fact["value"], True, rng)
        held_out += eval_questions(fact, relation)

    for examples in (train, alt, corr, held_out):
        _check_spans(examples)
    check_disjoint(train + alt + corr, held_out)

    _write(out_dir / "train.jsonl", train)
    _write(out_dir / "train_alt.jsonl", alt)
    _write(out_dir / "train_correction.jsonl", corr)
    _write(out_dir / "eval.jsonl", held_out)
    print(f"{len(train)} train, {len(alt)} alt, {len(corr)} correction, "
          f"{len(held_out)} held-out examples written to {out_dir}")


if __name__ == "__main__":
    main()
