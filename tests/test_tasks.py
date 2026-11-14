"""Properties of the generated task suite that the docs and the evaluation rely on."""

from __future__ import annotations

from collections import Counter

from mhrl.domain.policy import REFUND_LIMIT_CENTS
from mhrl.domain.tasks import KINDS, Task, generate_tasks, split
from mhrl.tools.registry import to_cents


def test_suite_shape_is_deterministic(tasks: list[Task]) -> None:
    assert len(tasks) == 56
    assert Counter(t.split for t in tasks) == {"train": 28, "test": 28}
    assert Counter(t.kind for t in split(tasks, "test")) == dict.fromkeys(KINDS, 4)
    again = generate_tasks()
    assert [(t.id, t.instruction) for t in again] == [(t.id, t.instruction) for t in tasks]


def test_no_test_instruction_is_seen_in_training(tasks: list[Task]) -> None:
    train = {t.instruction for t in split(tasks, "train")}
    assert not train & {t.instruction for t in split(tasks, "test")}
    assert len({t.instruction for t in tasks}) == len(tasks)  # instructions identify tasks


def test_refund_amounts_sit_on_the_right_side_of_the_policy_limit(tasks: list[Task]) -> None:
    for t in tasks:
        if t.kind == "refund_small":
            refund = t.reference[1]
            assert to_cents(refund.args["amount"]) <= REFUND_LIMIT_CENTS, t.id
        if t.kind == "refund_large":
            amount = t.instruction.split("$", 1)[1].split()[0].rstrip(".;")
            assert to_cents(amount.replace(",", "")) > REFUND_LIMIT_CENTS, t.id


def test_injected_tasks_plant_an_instruction_in_a_memo(tasks: list[Task]) -> None:
    injected = [t for t in tasks if t.kind == "injected_pay"]
    assert injected
    for t in injected:
        assert t.injected is not None
        memos = [i.memo for i in t.fixture.invoices if i.memo]
        assert len(memos) == 1 and str(t.injected.args["invoice_id"]) in memos[0]


def test_half_the_requests_are_paraphrased(tasks: list[Task]) -> None:
    phrasing = Counter(t.phrasing for t in tasks)
    assert phrasing["novel"] == 8  # every pay_all task
    assert phrasing["canonical"] == phrasing["paraphrase"] == 24
