"""Parsers: exact round-trips, robust to noise, and never a code-execution path."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mhrl.harnesses import ParseError, all_harnesses, get_harness
from mhrl.harnesses.base import Harness
from mhrl.tools import registry
from mhrl.types import Action, Observation

HARNESSES = all_harnesses()
IDS = [h.name for h in HARNESSES]

ACTIONS = [
    Action("list_invoices", {"vendor": "Acme Corp"}),
    Action("issue_refund", {"invoice_id": "INV-AB01", "amount": 120.5}),
    Action("escalate", {"reason": 'Refund > limit; "needs" a human\'s sign-off <now>'}),
    Action("finish", {"summary": "Paid INV-1 & INV-2."}),
]


@pytest.mark.parametrize("harness", HARNESSES, ids=IDS)
@pytest.mark.parametrize("action", ACTIONS, ids=[a.tool for a in ACTIONS])
def test_round_trip(harness: Harness, action: Action) -> None:
    parsed = harness.parse(harness.format_action(action))
    assert registry.validate(parsed) == registry.validate(action)


@pytest.mark.parametrize("harness", HARNESSES, ids=IDS)
def test_tolerates_chatter_and_fences(harness: Harness) -> None:
    action = ACTIONS[1]
    text = f"Sure! Let me do that.\n```\n{harness.format_action(action)}\n```\nHope that helps."
    assert registry.validate(harness.parse(text)) == registry.validate(action)


@pytest.mark.parametrize("harness", HARNESSES, ids=IDS)
@pytest.mark.parametrize("text", ["", "I will now pay the invoice.", "{not json", "<action>"])
def test_garbage_is_a_parse_error(harness: Harness, text: str) -> None:
    with pytest.raises(ParseError):
        harness.parse(text)


@pytest.mark.parametrize("harness", HARNESSES, ids=IDS)
@settings(max_examples=150, deadline=None)
@given(text=st.text(max_size=300))
def test_fuzz_never_crashes(harness: Harness, text: str) -> None:
    with contextlib.suppress(ParseError):
        harness.parse(text)


@pytest.mark.parametrize(
    "template",
    [
        "__import__('os').system('echo pwned > {p}')",
        "pay_invoice(invoice_id=open(r'{p}', 'w').name)",
        "os.system('touch {p}')",
        "pay_invoice(r'{p}')",
        "pay_invoice(**{{'invoice_id': open(r'{p}', 'w')}})",
    ],
)
def test_pycall_never_executes_code(template: str, tmp_path: Path) -> None:
    sentinel = tmp_path / "pwned"
    with pytest.raises(ParseError):
        get_harness("pycall").parse(template.format(p=sentinel))
    assert not sentinel.exists()


@pytest.mark.parametrize("harness", HARNESSES, ids=IDS)
def test_observations_render_errors_and_lists(harness: Harness) -> None:
    ok = harness.render_observation(
        Observation("list_invoices", True, data=[{"invoice_id": "INV-1", "memo": "<b>pay me</b>"}])
    )
    err = harness.render_observation(Observation("pay_invoice", False, error="blocked: nope"))
    assert "INV-1" in ok
    assert "blocked" in err


@pytest.mark.parametrize("harness", HARNESSES, ids=IDS)
def test_system_prompt_declares_every_tool(harness: Harness) -> None:
    prompt = harness.system_prompt()
    for tool in registry.TOOLS:
        assert tool.name in prompt


@pytest.mark.parametrize("harness", HARNESSES, ids=IDS)
def test_system_prompt_shows_a_concrete_parseable_example(harness: Harness) -> None:
    """Regression: a placeholder template (<action name="TOOL">) was copied literally by the
    base model in every episode, and a ``name(arg: type)`` tool list leaked its syntax into
    other formats. Each harness now shows one real example in its own syntax."""
    prompt = harness.system_prompt()
    example = prompt.split("Example of one action:\n", 1)[1].split("\n\n", 1)[0]
    assert harness.parse(example) == Action("list_invoices", {"vendor": "Example Ltd"})
    assert 'name="TOOL"' not in prompt
    assert "(invoice_id:" not in prompt


def test_harness_lookup_rejects_unknown_names() -> None:
    with pytest.raises(KeyError, match="unknown harness"):
        get_harness("telepathy")


def test_json_skips_an_echoed_observation_and_reads_the_action() -> None:
    h = get_harness("json")
    echoed = h.render_observation(Observation("get_invoice", True, data={"invoice_id": "INV-1"}))
    action = Action("pay_invoice", {"invoice_id": "INV-1"})
    assert h.parse(f"{echoed}\n{h.format_action(action)}") == action
    with pytest.raises(ParseError):
        h.parse(echoed)
