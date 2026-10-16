"""Scores and answers fact examples with the model's current weights."""

from collections.abc import Sequence

from pi import data_gen
from pi import model as model_lib


def answer_logprobs(
    loaded: model_lib.LoadedModel,
    examples: Sequence[data_gen.Example],
    batch_size: int = 16,
) -> list[tuple[float, int]]:
    """Returns (summed log-prob in nats, token count) of each example's answer."""
    tokenizer = loaded.tokenizer
    encoded = [
        model_lib.encode(tokenizer, model_lib.prompt_text(tokenizer, e), e.assistant, e.answer)
        for e in examples
    ]
    return model_lib.answer_logprobs(loaded, encoded, batch_size)


def respond(
    loaded: model_lib.LoadedModel,
    questions: Sequence[data_gen.Example],
    contexts: Sequence[str | None] | None = None,
    batch_size: int = 32,
) -> list[str]:
    """Greedily answers each question, with `contexts[i]` (e.g. the fact) before question i."""
    if contexts is None:
        contexts = [None] * len(questions)
    prompts = [
        model_lib.prompt_text(loaded.tokenizer, q, c)
        for q, c in zip(questions, contexts, strict=True)
    ]
    return model_lib.generate(loaded, prompts, batch_size=batch_size)
