"""Deterministic policies for tests, CI and the CPU demo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from mhrl.domain.tasks import Task
from mhrl.harnesses.base import Harness
from mhrl.policies.base import Generation, estimate_tokens
from mhrl.types import Message


class ScriptedPolicy:
    """Replays a fixed list of replies per task.

    The reply index is the number of assistant turns since the task message,
    clamped to the last reply, so a trailing ``finish`` is repeated if the loop
    asks again (e.g. a self-check turn).
    """

    name = "scripted"

    def __init__(self, replies: Mapping[str, Sequence[str]]) -> None:
        self._replies = {k: list(v) for k, v in replies.items()}

    @classmethod
    def oracle(cls, harness: Harness, tasks: Sequence[Task]) -> ScriptedPolicy:
        """Plays each task's reference solution, spelled in ``harness``'s format."""
        return cls(
            {
                harness.render_task(t.instruction): [harness.format_action(a) for a in t.reference]
                for t in tasks
            }
        )

    def replies(self) -> dict[str, list[str]]:
        """A copy of the script, e.g. to derive a variant that misbehaves on purpose."""
        return {k: list(v) for k, v in self._replies.items()}

    def generate(self, conversations: Sequence[Sequence[Message]]) -> list[Generation]:
        out = []
        for conv in conversations:
            key, turns = self._locate(conv)
            replies = self._replies[key]
            text = replies[min(turns, len(replies) - 1)]
            prompt = sum(estimate_tokens(m["content"]) for m in conv)
            out.append(Generation(text, prompt, estimate_tokens(text)))
        return out

    def _locate(self, conv: Sequence[Message]) -> tuple[str, int]:
        for i in range(len(conv) - 1, -1, -1):
            if conv[i]["role"] == "user" and conv[i]["content"] in self._replies:
                turns = sum(1 for m in conv[i + 1 :] if m["role"] == "assistant")
                return conv[i]["content"], turns
        raise KeyError("conversation does not contain a scripted task")
