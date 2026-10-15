"""Model loading, prompt formatting, answer scoring and greedy generation."""

import dataclasses
import os
from collections.abc import Sequence
from typing import Any

import torch
import transformers

from pi import data_gen

_END = "<|im_end|>"


@dataclasses.dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: torch.device


def load(name: str, device: str, dtype: str = "bfloat16") -> LoadedModel:
    """Loads a model and tokenizer onto `device` ("mps", "cuda" or "cpu")."""
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model: Any = transformers.AutoModelForCausalLM.from_pretrained(
        name, dtype=getattr(torch, dtype), device_map=device,
    )
    model.eval()
    return LoadedModel(model, tokenizer, torch.device(device))


def prompt_text(tokenizer: Any, example: data_gen.Example, context: str | None = None) -> str:
    """Returns the chat-template prompt, thinking off, with `context` before the question."""
    messages = []
    if example.system:
        messages.append({"role": "system", "content": example.system})
    user = example.user if context is None else f"{context}\n\n{example.user}"
    messages.append({"role": "user", "content": user})
    text: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return text


@dataclasses.dataclass(frozen=True)
class Encoded:
    """Token ids with masks: response (and end token) for the loss, answer for scoring."""

    input_ids: list[int]
    response_mask: list[int]
    answer_mask: list[int]


def encode(tokenizer: Any, prompt: str, response: str, answer: str) -> Encoded:
    """Tokenises prompt + response; `answer` must occur once in `response`."""
    full = prompt + response + _END
    start = len(prompt) + response.index(answer)
    end = start + len(answer)
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    response_mask, answer_mask = [], []
    for token_start, token_end in enc["offset_mapping"]:
        response_mask.append(int(token_end > len(prompt)))
        answer_mask.append(int(token_end > start and token_start < end))
    return Encoded(enc["input_ids"], response_mask, answer_mask)


def _pad(rows: Sequence[Sequence[int]], value: int) -> torch.Tensor:
    width = max(len(r) for r in rows)
    return torch.tensor([list(r) + [value] * (width - len(r)) for r in rows])


@torch.no_grad()
def answer_logprobs(
    loaded: LoadedModel, encoded: Sequence[Encoded], batch_size: int = 16,
) -> list[tuple[float, int]]:
    """Returns (summed log-prob in nats, token count) of each answer span."""
    results = []
    for i in range(0, len(encoded), batch_size):
        batch = encoded[i : i + batch_size]
        ids = _pad([e.input_ids for e in batch], loaded.tokenizer.pad_token_id)
        attention = _pad([[1] * len(e.input_ids) for e in batch], 0)
        mask = _pad([e.answer_mask for e in batch], 0)
        ids, attention, mask = (t.to(loaded.device) for t in (ids, attention, mask))
        logits = loaded.model(input_ids=ids, attention_mask=attention).logits
        # The logits at position t - 1 predict token t.
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        token_logp = logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
        target_mask = mask[:, 1:].float()
        sums = (token_logp * target_mask).sum(-1)
        counts = target_mask.sum(-1)
        results += [(s.item(), int(c.item())) for s, c in zip(sums, counts, strict=True)]
    return results


@torch.no_grad()
def generate(
    loaded: LoadedModel, prompts: Sequence[str], max_new_tokens: int = 16, batch_size: int = 32,
) -> list[str]:
    """Greedily decodes a response to each prompt."""
    tokenizer = loaded.tokenizer
    tokenizer.padding_side = "left"
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenizer(
            list(prompts[i : i + batch_size]), return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).to(loaded.device)
        generated = loaded.model.generate(
            **batch, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, temperature=None, top_p=None, top_k=None,
        )
        new_tokens = generated[:, batch["input_ids"].shape[1] :]
        outputs += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return outputs
