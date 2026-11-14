"""A terse line protocol. Held out of training: it measures portability."""

from __future__ import annotations

import shlex

from mhrl.harnesses.base import Harness, ParseError, strip_fences
from mhrl.types import Action, Observation


class PlainTextHarness(Harness):
    name = "plaintext"

    def _protocol(self) -> str:
        return (
            "Act with one line: the word TOOL, the tool name, then key=value pairs (quote "
            "values that contain spaces). No parentheses. Results come back as RESULT lines."
        )

    def format_action(self, action: Action) -> str:
        parts = [f"{k}={shlex.quote(str(v))}" for k, v in action.args.items()]
        return " ".join(["TOOL", action.tool, *parts])

    def parse(self, text: str) -> Action:
        for line in strip_fences(text).splitlines():
            line = line.strip()
            if not line.upper().startswith("TOOL "):
                continue
            try:
                tokens = shlex.split(line)
            except ValueError as e:
                raise ParseError(f"unbalanced quotes: {e}") from e
            if len(tokens) < 2:
                raise ParseError("TOOL needs a tool name")
            args: dict[str, str | int | float] = {}
            for tok in tokens[2:]:
                key, sep, value = tok.partition("=")
                if not sep or not key:
                    raise ParseError(f"expected key=value, got {tok!r}")
                args[key] = value
            return Action(tokens[1], args)
        raise ParseError("no line starting with TOOL found")

    def render_observation(self, obs: Observation) -> str:
        if not obs.ok:
            return f"RESULT error {obs.tool}: {obs.error}"
        rows = obs.data if isinstance(obs.data, list) else [obs.data]
        lines = [f"RESULT ok {obs.tool} ({len(rows)} row(s))"]
        for row in rows:
            if isinstance(row, dict):
                lines.append("  " + " ".join(f"{k}={v}" for k, v in row.items()))
            else:
                lines.append(f"  {row}")
        return "\n".join(lines)
