"""JSON tool calls, in the style of OpenAI-compatible function calling."""

from __future__ import annotations

import json

from mhrl.harnesses.base import Harness, ParseError
from mhrl.types import Action, Observation


class JsonCallsHarness(Harness):
    name = "json"

    def _protocol(self) -> str:
        return (
            'Reply with a single JSON object with a "tool" name and an "args" object. '
            "Tool results come back as JSON."
        )

    def format_action(self, action: Action) -> str:
        return json.dumps({"tool": action.tool, "args": dict(action.args)})

    def parse(self, text: str) -> Action:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
                args = obj.get("args", {})
                if not isinstance(args, dict):
                    raise ParseError('"args" must be an object')
                clean = {}
                for k, v in args.items():
                    if not isinstance(v, str | int | float) or isinstance(v, bool):
                        raise ParseError(f"argument {k!r} must be a string or number")
                    clean[str(k)] = v
                return Action(obj["tool"], clean)
        raise ParseError('no JSON object with a "tool" field found')

    def render_observation(self, obs: Observation) -> str:
        body: dict[str, object] = {"tool": obs.tool, "ok": obs.ok}
        if obs.ok:
            body["result"] = obs.data
        else:
            body["error"] = obs.error
        return json.dumps(body)
