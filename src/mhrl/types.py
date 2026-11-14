"""Small value types shared by every layer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypedDict

Scalar = str | int | float


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Action:
    """One tool call proposed by a model (or by a compiled recipe)."""

    tool: str
    args: Mapping[str, Scalar] = field(default_factory=dict)

    def canonical(self) -> str:
        """Stable string form, used for fingerprints and equality across harnesses."""
        return json.dumps({"tool": self.tool, "args": dict(self.args)}, sort_keys=True)

    def __hash__(self) -> int:
        return hash(self.canonical())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Action) and self.canonical() == other.canonical()


@dataclass(frozen=True)
class Observation:
    """What the environment returns for one action, before a harness renders it."""

    tool: str
    ok: bool
    data: object = None
    error: str | None = None


def cents_to_str(cents: int) -> str:
    return f"{cents / 100:.2f}"
