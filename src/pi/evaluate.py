"""Scores and grades a model's answers on fact examples with its current weights."""

from collections.abc import Sequence

from pi import data_gen
from pi import model as model_lib


def answer_logprobs(
    loaded: model_lib.LoadedModel,
    examples: Sequence[data_gen.Example],
    batch_size: int = 16,
) -> list[tuple[float, int]]:
    """Scores the answer span of each example with teacher forcing.

    Args:
        loaded: The model to score with.
        examples: Training examples or held-out questions.
        batch_size: Examples per forward pass.

    Returns:
        For each example, the summed log-probability of its answer tokens in
        nats and the number of answer tokens.
    """
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
    """Greedily answers each held-out question.

    Args:
        loaded: The model to generate with.
        questions: Held-out questions.
        contexts: Optional text to put before each question, one per
          question. The ceiling check passes the fact's statement here.
        batch_size: Questions per batch.

    Returns:
        The model's responses, in question order.
    """
    if contexts is None:
        contexts = [None] * len(questions)
    prompts = [
        model_lib.prompt_text(loaded.tokenizer, q, c)
        for q, c in zip(questions, contexts, strict=True)
    ]
    return model_lib.generate(loaded, prompts, batch_size=batch_size)
