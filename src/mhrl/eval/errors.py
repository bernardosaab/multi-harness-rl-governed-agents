"""Error analysis: count failure clusters, then freeze cases into regression tests.

Every failure gets a closed-taxonomy label from the verifier (``Outcome``), so
analysis is a count, not a reading exercise. ``ratchet`` turns one observed
failure into a permanent test: the exact model replies are replayed through the
real harness and gate, and the test asserts the same verdict forever after.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cluster:
    outcome: str
    count: int
    share: float
    example_task: str
    example_harness: str
    example_reply: str


def clusters(episodes: Iterable[Mapping[str, object]]) -> dict[str, list[Cluster]]:
    """Failure clusters per system, largest first."""
    by_system: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for ep in episodes:
        if ep["outcome"] != "solved":
            by_system[str(ep["system"])].append(ep)
    out: dict[str, list[Cluster]] = {}
    for system, fails in by_system.items():
        counts = Counter(str(e["outcome"]) for e in fails)
        rows = []
        for outcome, n in counts.most_common():
            ex = next(e for e in fails if e["outcome"] == outcome)
            rows.append(
                Cluster(
                    outcome,
                    n,
                    n / len(fails),
                    str(ex["task_id"]),
                    str(ex["harness"]),
                    _offending_reply(ex),
                )
            )
        out[system] = rows
    return out


def _offending_reply(ep: Mapping[str, object]) -> str:
    turns = ep.get("turns") or []
    assert isinstance(turns, list)
    for t in turns:
        if t.get("blocked") or t.get("parse_error"):
            return str(t.get("text", ""))[:300]
    return str(turns[-1].get("text", ""))[:300] if turns else ""


def ratchet(episode: Mapping[str, object], out_dir: Path, name: str | None = None) -> Path:
    """Freeze an episode's replies + verdict into ``out_dir/<name>.json``."""
    turns = episode.get("turns") or []
    assert isinstance(turns, list)
    case = {
        "task_id": episode["task_id"],
        "harness": episode["harness"],
        "system": episode["system"],
        "replies": [t["text"] for t in turns],
        "expected_outcome": episode["outcome"],
        "note": "Frozen from an evaluation run by `mhrl ratchet`.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name or f"{episode['system']}-{episode['harness']}-{episode['task_id']}"
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(case, indent=2) + "\n")
    return path
