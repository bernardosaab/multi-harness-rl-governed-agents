"""Code-as-action, minus the danger: one Python call, parsed with ``ast`` and never run.

The model writes ``pay_invoice(invoice_id="INV-1")``. The harness reads the
syntax tree, accepts only a bare function name with literal keyword arguments,
and rejects everything else. ``__import__('os').system(...)`` is a parse error,
not an exploit.
"""

from __future__ import annotations

import ast

from mhrl.harnesses.base import Harness, ParseError, strip_fences
from mhrl.types import Action, Observation, Scalar


class PyCallHarness(Harness):
    name = "pycall"

    def _protocol(self) -> str:
        return (
            "Act by writing exactly one Python function call with keyword arguments. "
            "The call is parsed, not executed. Results come back as Python literals."
        )

    def format_action(self, action: Action) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in action.args.items())
        return f"{action.tool}({args})"

    def parse(self, text: str) -> Action:
        for line in strip_fences(text).splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "(" not in line:
                continue
            try:
                node = ast.parse(line, mode="eval").body
            except SyntaxError:
                continue
            if isinstance(node, ast.Call):
                return _to_action(node)
        raise ParseError("no single-line function call found")

    def render_observation(self, obs: Observation) -> str:
        if obs.ok:
            return f"# {obs.tool} ->\n{obs.data!r}"
        return f"# {obs.tool} raised Error: {obs.error}"


def _to_action(call: ast.Call) -> Action:
    if not isinstance(call.func, ast.Name):
        raise ParseError("only plain tool names may be called")
    if call.args:
        raise ParseError("use keyword arguments only")
    args: dict[str, Scalar] = {}
    for kw in call.keywords:
        if kw.arg is None:
            raise ParseError("**kwargs is not allowed")
        try:
            value = ast.literal_eval(kw.value)
        except ValueError as e:
            raise ParseError(f"argument {kw.arg!r} must be a literal") from e
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            raise ParseError(f"argument {kw.arg!r} must be a string or number")
        args[kw.arg] = value
    return Action(call.func.id, args)
