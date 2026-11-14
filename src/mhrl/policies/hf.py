"""A local Hugging Face model, optionally with a LoRA adapter. Needs the ``train`` extra."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mhrl.policies.base import Generation
from mhrl.types import Message


class HFPolicy:
    def __init__(
        self,
        model_id: str,
        *,
        adapter: str | Path | None = None,
        temperature: float = 0.7,
        max_new_tokens: int = 96,
        batch_size: int = 16,
        model: Any = None,
        tokenizer: Any = None,
        name: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
            if adapter is not None:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(adapter))
        self.model = model
        self.name = name or (f"{model_id}+{Path(adapter).name}" if adapter else model_id)

    def prompt_ids(self, conversation: Sequence[Message]) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            list(conversation),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, conversations: Sequence[Sequence[Message]]) -> list[Generation]:
        import torch

        out: list[Generation] = []
        device = next(self.model.parameters()).device
        pad = self.tokenizer.pad_token_id
        for start in range(0, len(conversations), self.batch_size):
            chunk = conversations[start : start + self.batch_size]
            prompts = [self.prompt_ids(c) for c in chunk]
            width = max(len(p) for p in prompts)
            input_ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=device)
            mask = torch.tensor(
                [[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device
            )
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                seqs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=max(self.temperature, 1e-5),
                    top_p=1.0,
                    top_k=0,
                    pad_token_id=pad,
                )
            if was_training:
                self.model.train()
            for p, row in zip(prompts, seqs[:, width:].tolist(), strict=True):
                completion = _trim(row, pad, self.tokenizer.eos_token_id)
                text = self.tokenizer.decode(completion, skip_special_tokens=True)
                out.append(Generation(text, len(p), len(completion), tuple(p), tuple(completion)))
        return out


def _trim(ids: list[int], pad: int, eos: int | None) -> list[int]:
    """Keep tokens up to and including the first EOS; drop right padding."""
    out = []
    for t in ids:
        out.append(t)
        if t == eos:
            break
    while out and out[-1] == pad and pad != eos:
        out.pop()
    return out
