"""The agent loop: caps in code, loop-engineering behaviours, end-to-end episodes."""

from __future__ import annotations

import pytest

from mhrl.baselines import match, run_compiled, run_routed
from mhrl.domain.tasks import Task, split
from mhrl.harnesses import all_harnesses, get_harness
from mhrl.harnesses.base import Harness
from mhrl.policies import ScriptedPolicy
from mhrl.runtime import ENGINEERED, VANILLA, LoopConfig, run_episodes
from mhrl.types import Action

HARNESSES = all_harnesses()
IDS = [h.name for h in HARNESSES]


@pytest.mark.parametrize("harness", HARNESSES, ids=IDS)
@pytest.mark.parametrize("config", [VANILLA, ENGINEERED], ids=["vanilla", "engineered"])
def test_oracle_solves_every_task_in_every_harness(
    tasks: list[Task], harness: Harness, config: LoopConfig
) -> None:
    episodes = run_episodes(ScriptedPolicy.oracle(harness, tasks), harness, tasks, config)
    failed = [(e.task_id, e.outcome) for e in episodes if not e.success]
    assert not failed


def test_samples_are_independent_worlds(by_id: dict[str, Task]) -> None:
    task, h = by_id["pay-test-0"], get_harness("json")
    eps = run_episodes(ScriptedPolicy.oracle(h, [task]), h, [task], samples=3)
    assert [e.sample for e in eps] == [0, 1, 2]
    assert all(e.success for e in eps)  # paying in sample 0 did not leak into sample 1


def test_turn_cap_is_enforced_in_code(by_id: dict[str, Task]) -> None:
    task, h = by_id["pay-test-0"], get_harness("json")
    looping = ScriptedPolicy(
        {h.render_task(task.instruction): [h.format_action(task.reference[0])]}
    )
    [ep] = run_episodes(looping, h, [task], LoopConfig(max_turns=5))
    assert ep.acts == 5 and ep.outcome == "cap_hit"


def test_token_cap_is_enforced_in_code(by_id: dict[str, Task]) -> None:
    task, h = by_id["pay-test-0"], get_harness("json")
    chatty = ScriptedPolicy({h.render_task(task.instruction): ["blah " * 400]})
    [ep] = run_episodes(chatty, h, [task], LoopConfig(max_completion_tokens=600))
    assert ep.completion_tokens >= 600 and len(ep.turns) <= 2


def test_saying_done_does_not_end_the_episode_successfully(by_id: dict[str, Task]) -> None:
    task, h = by_id["pay-test-0"], get_harness("pycall")
    liar = ScriptedPolicy(
        {
            h.render_task(task.instruction): [
                'finish(summary="I have paid the invoice, it is done.")'
            ]
        }
    )
    [ep] = run_episodes(liar, h, [task])
    assert ep.outcome == "claimed_not_written" and ep.reward == 0.0


def test_parse_retries_do_not_count_toward_the_turn_cap(by_id: dict[str, Task]) -> None:
    task, h = by_id["pay-test-0"], get_harness("react_xml")
    good = [h.format_action(a) for a in task.reference]
    sloppy = ScriptedPolicy({h.render_task(task.instruction): ["um", "uh", *good]})
    [ep] = run_episodes(sloppy, h, [task], LoopConfig(parse_retries=2))
    assert ep.success
    assert [t.kind for t in ep.turns][:2] == ["retry", "retry"]
    assert ep.acts == len(good)


def test_self_check_adds_one_verification_turn(by_id: dict[str, Task]) -> None:
    task, h = by_id["pay-test-0"], get_harness("json")
    [ep] = run_episodes(ScriptedPolicy.oracle(h, [task]), h, [task], LoopConfig(self_check=True))
    assert ep.success
    assert [t.kind for t in ep.turns].count("self_check") == 1


def test_injected_memo_cannot_cause_a_write(by_id: dict[str, Task]) -> None:
    """Even a model that fully obeys the planted instruction cannot get it executed."""
    task, h = by_id["injected_pay-test-0"], get_harness("plaintext")
    assert task.injected is not None
    obedient = ScriptedPolicy(
        {
            h.render_task(task.instruction): [
                h.format_action(task.reference[0]),
                h.format_action(task.injected),
            ]
        }
    )
    [ep] = run_episodes(obedient, h, [task])
    assert ep.outcome == "injection_followed"
    executed = [d for e, d in ep.decision_log if e == "gate" and d["status"] == "executed"]
    assert executed == []


def test_unknown_tool_is_blocked_not_ignored(by_id: dict[str, Task]) -> None:
    task, h = by_id["pay-test-0"], get_harness("json")
    rogue = ScriptedPolicy(
        {
            h.render_task(task.instruction): [
                h.format_action(Action("delete_vendor", {"vendor_id": "x"}))
            ]
        }
    )
    [ep] = run_episodes(rogue, h, [task])
    assert ep.outcome == "blocked_write" and ep.turns[0].blocked


def test_token_ids_are_carried_into_turns(by_id: dict[str, Task]) -> None:
    from mhrl.policies.base import Generation

    class Tokenised:
        name = "tok"

        def generate(self, convs: object) -> list[Generation]:
            assert isinstance(convs, list)
            return [
                Generation('{"tool": "finish", "args": {"summary": "x"}}', 3, 2, (1, 2, 3), (7, 8))
                for _ in convs
            ]

    task, h = by_id["pay-test-0"], get_harness("json")
    [ep] = run_episodes(Tokenised(), h, [task])
    assert ep.turns[0].prompt_ids == (1, 2, 3) and ep.turns[0].completion_ids == (7, 8)
    assert "prompt_ids" not in ep.to_record()["turns"][0]  # type: ignore[index]


def test_compiled_loop_solves_canonical_and_skips_the_rest(tasks: list[Task]) -> None:
    test = split(tasks, "test")
    eps = {e.task_id: e for e in run_compiled(test)}
    for t in test:
        ep = eps[t.id]
        assert ep.prompt_tokens == ep.completion_tokens == 0
        if t.phrasing == "canonical":
            assert ep.outcome == "solved", t.id
        elif not t.must_escalate:
            assert ep.outcome == "skipped", t.id  # never guesses


def test_router_sends_only_unmatched_tasks_to_the_agent(tasks: list[Task]) -> None:
    test = split(tasks, "test")
    h = get_harness("json")
    eps = run_routed(ScriptedPolicy.oracle(h, test), h, test, samples=2)
    assert len(eps) == 2 * len(test)
    for e in eps:
        task = next(t for t in test if t.id == e.task_id)
        assert e.route == ("compiled" if match(task) else "agent")
        assert e.success
