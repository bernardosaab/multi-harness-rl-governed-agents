"""The ratchet: every frozen failure replays to the same verdict, forever.

Cases live in ``tests/regressions/*.json`` and are produced by ``mhrl ratchet``
from real evaluation runs. Each stores the exact model replies; replaying them
through the real harness, loop and gate must give the recorded outcome. If a
change to the gate or the verifier flips one, this test says so.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mhrl.domain.tasks import Task
from mhrl.eval.errors import clusters, ratchet
from mhrl.harnesses import get_harness
from mhrl.policies import ScriptedPolicy
from mhrl.runtime import ENGINEERED, VANILLA, run_episodes

CASES = sorted((Path(__file__).parent / "regressions").glob("*.json"))


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_frozen_case_replays_to_same_verdict(path: Path, by_id: dict[str, Task]) -> None:
    case = json.loads(path.read_text())
    task, harness = by_id[case["task_id"]], get_harness(case["harness"])
    policy = ScriptedPolicy({harness.render_task(task.instruction): case["replies"]})
    config = ENGINEERED if case.get("loop") == "engineered" else VANILLA
    [ep] = run_episodes(policy, harness, [task], config)
    assert ep.outcome == case["expected_outcome"]


def test_ratchet_round_trip(tmp_path: Path, by_id: dict[str, Task]) -> None:
    task, h = by_id["injected_pay-test-1"], get_harness("react_xml")
    assert task.injected is not None
    replies = [h.format_action(task.reference[0]), h.format_action(task.injected)]
    [ep] = run_episodes(ScriptedPolicy({h.render_task(task.instruction): replies}), h, [task])
    path = ratchet(ep.to_record(), tmp_path)
    saved = json.loads(path.read_text())
    assert saved["expected_outcome"] == "injection_followed"
    assert saved["replies"] == replies


def test_ratchet_refuses_compiled_episodes(tmp_path: Path) -> None:
    rec = {
        "system": "routed",
        "harness": "json",
        "route": "compiled",
        "task_id": "x",
        "outcome": "skipped",
        "turns": [],
    }
    with pytest.raises(ValueError, match="no model replies"):
        ratchet(rec, tmp_path)


def test_clusters_count_failures_largest_first() -> None:
    recs = [
        {"system": "s", "outcome": "cap_hit", "task_id": "a", "harness": "json", "turns": []},
        {
            "system": "s",
            "outcome": "blocked_write",
            "task_id": "b",
            "harness": "json",
            "turns": [{"text": "bad", "blocked": True}],
        },
        {"system": "s", "outcome": "blocked_write", "task_id": "c", "harness": "json", "turns": []},
        {"system": "s", "outcome": "solved", "task_id": "d", "harness": "json", "turns": []},
    ]
    out = clusters(recs)
    assert [c.outcome for c in out["s"]] == ["blocked_write", "cap_hit"]
    assert out["s"][0].count == 2 and out["s"][0].example_reply == "bad"
