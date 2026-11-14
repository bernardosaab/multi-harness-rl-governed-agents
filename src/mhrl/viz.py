"""Terminal rendering of episodes (also exported as SVG for the README)."""

from __future__ import annotations

from collections.abc import Mapping

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_OUTCOME_STYLE = {"solved": "bold green", "skipped": "yellow"}


def render_episode(ep: Mapping[str, object], max_chars: int = 420) -> RenderableType:
    turns = ep.get("turns") or []
    assert isinstance(turns, list)
    parts: list[RenderableType] = [
        Text.assemble(("task  ", "dim"), str(ep["instruction"])),
        Text.assemble(
            ("system ", "dim"), f"{ep['system']}  ", ("harness ", "dim"), str(ep["harness"])
        ),
    ]
    for i, t in enumerate(turns, 1):
        style = "red" if t.get("blocked") else ("magenta" if t.get("parse_error") else "cyan")
        label = f"{i}. model" + (f" [{t['kind']}]" if t.get("kind") != "act" else "")
        parts.append(Text(label, style=f"bold {style}"))
        parts.append(Text(_clip(str(t.get("text", "")), max_chars)))
        if t.get("feedback"):
            parts.append(Text("   -> " + _clip(str(t["feedback"]), max_chars), style="dim"))
    gate = Table(
        title="decision log (append-only)",
        show_header=True,
        header_style="bold",
        expand=True,
        box=None,
    )
    gate.add_column("event", style="dim", no_wrap=True)
    gate.add_column("detail")
    log = ep.get("decision_log") or []
    assert isinstance(log, list)
    for event, detail in log:
        if event == "gate":
            status = detail.get("status")
            colour = "green" if status in ("executed", "replayed") else "red"
            gate.add_row(
                "gate",
                Text.assemble(
                    (f"{status} ", f"bold {colour}"),
                    f"{detail.get('tool')} {detail.get('args')} ",
                    (f"[{detail.get('tier') or '-'}] {detail.get('message')}", "dim"),
                ),
            )
        elif event != "run_started":
            gate.add_row(event, str(detail))
    parts.append(gate)
    outcome = str(ep["outcome"])
    verdict = Text.assemble(
        ("verdict  ", "dim"),
        (outcome, _OUTCOME_STYLE.get(outcome, "bold red")),
        (f"   reward {float(str(ep['reward'])):.3f}   turns {ep['acts']}", "dim"),
    )
    parts.append(verdict)
    return Panel(Group(*parts), title=f"[b]{ep['task_id']}[/b]", border_style="blue")


def _clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "..."
