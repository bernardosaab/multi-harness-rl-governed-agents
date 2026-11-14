"""The policy interface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from mhrl.types import Message


@dataclass(frozen=True)
class Generation:
    """One completion plus the exact token ids it was sampled from.

    Training reuses ``prompt_ids``/``completion_ids`` directly, so the loss is
    computed on precisely the tokens the harness saw: no re-tokenisation drift.
    Policies without a tokenizer (scripted, compiled) leave them empty and
    report estimated token counts for cost accounting.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    prompt_ids: tuple[int, ...] = ()
    completion_ids: tuple[int, ...] = ()


class Policy(Protocol):
    name: str

    def generate(self, conversations: Sequence[Sequence[Message]]) -> list[Generation]:
        """Generate one reply per conversation (batched)."""
        ...


def estimate_tokens(text: str) -> int:
    """~4 characters per token: good enough for budgeting a policy with no tokenizer."""
    return max(1, len(text) // 4)
