"""Writes one fact into the model by training a fresh LoRA adapter on it.

Each write trains a new adapter on one fact's training examples at batch size
1. A history write merges the adapter into the weights and saves its factors.
A branch write, used for the target fact, evaluates the adapter and then
throws it away, which leaves the frozen history checkpoint exactly as it was.
"""

import dataclasses
import pathlib
import random
from collections.abc import Callable, Sequence

import peft
import torch

from pi import data_gen
from pi import model as model_lib

# Called during a write with the optimiser step count, which is 0 before the
# first step.
StepCallback = Callable[[int], None]


@dataclasses.dataclass(frozen=True)
class WriteConfig:
    """Hyperparameters for one write.

    Attributes:
        steps: Optimiser steps.
        learning_rate: AdamW learning rate.
        rank: LoRA rank.
        alpha: LoRA alpha.
        target_modules: Module names that get adapters.
        loss_on: "response" for all assistant tokens, or "answer" for the
          answer span only.
        eval_every: Steps between callback calls.
    """

    steps: int = 24
    learning_rate: float = 2e-4
    rank: int = 16
    alpha: int = 32
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    loss_on: str = "response"
    eval_every: int = 4


def _training_batches(
    loaded: model_lib.LoadedModel,
    examples: Sequence[data_gen.Example],
    loss_on: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Encodes each example as an input_ids tensor and a labels tensor.

    Labels outside the loss span are -100, which the loss ignores.

    Args:
        loaded: Supplies the tokenizer and device.
        examples: The fact's training examples.
        loss_on: "response" or "answer", as in `WriteConfig.loss_on`.

    Returns:
        One (input_ids, labels) pair per example, each of shape [1, length].
    """
    batches = []
    for example in examples:
        prompt = model_lib.prompt_text(loaded.tokenizer, example)
        enc = model_lib.encode(loaded.tokenizer, prompt, example.assistant, example.answer)
        mask = enc.response_mask if loss_on == "response" else enc.answer_mask
        labels = [t if m else -100 for t, m in zip(enc.input_ids, mask, strict=True)]
        batches.append((
            torch.tensor([enc.input_ids], device=loaded.device),
            torch.tensor([labels], device=loaded.device),
        ))
    return batches


def write_fact(
    loaded: model_lib.LoadedModel,
    examples: Sequence[data_gen.Example],
    config: WriteConfig,
    seed: int,
    merge: bool,
    on_eval: StepCallback | None = None,
    save_dir: pathlib.Path | None = None,
) -> list[float]:
    """Trains one adapter on a fact's examples, then merges or discards it.

    Steps cycle through the examples in a seeded shuffled order, reshuffling
    after each pass. The callback runs at step 0 and every `eval_every`
    steps, with the adapter active and the model in eval mode.

    Args:
        loaded: The model to write into. Its weights change only if `merge`.
        examples: The fact's training examples.
        config: Write hyperparameters.
        seed: Seed for example order and adapter initialisation.
        merge: If True, merge the adapter into the weights. If False, remove
          it and leave the weights as they were.
        on_eval: Optional callback, for example to record learning curves.
        save_dir: If given, save the adapter's factors here before merging.

    Returns:
        The training loss at each step.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    lora = peft.LoraConfig(
        r=config.rank, lora_alpha=config.alpha, lora_dropout=0.0,
        target_modules=list(config.target_modules), task_type="CAUSAL_LM",
    )
    peft_model = peft.get_peft_model(loaded.model, lora)
    params = [p for p in peft_model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=config.learning_rate)
    batches = _training_batches(loaded, examples, config.loss_on)

    def evaluate(step: int) -> None:
        if on_eval is not None and step % config.eval_every == 0:
            peft_model.eval()
            on_eval(step)
            peft_model.train()

    losses = []
    order: list[int] = []
    peft_model.train()
    evaluate(0)
    for step in range(1, config.steps + 1):
        if not order:
            order = rng.sample(range(len(batches)), len(batches))
        input_ids, labels = batches[order.pop()]
        loss = peft_model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        losses.append(loss.item())
        evaluate(step)

    peft_model.eval()
    if save_dir is not None:
        peft_model.save_pretrained(str(save_dir))
    loaded.model = peft_model.merge_and_unload() if merge else peft_model.unload()
    loaded.model.eval()
    return losses
