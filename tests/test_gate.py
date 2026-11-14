"""The write gate must fail closed in every way we can think of breaking it."""

from __future__ import annotations

import sqlite3

import pytest

from mhrl.domain.ledger import Fixture, Invoice, Ledger, Vendor
from mhrl.tools.gate import (
    PolicyApprover,
    Quote,
    Signature,
    Status,
    TamperedPayload,
    Tier,
    WriteGate,
    fingerprint,
)
from mhrl.types import Action

FX = Fixture(
    vendors=(Vendor("V1", "Acme", "GB00BANK1234"), Vendor("V2", "Shady", "GB00BANK9999", False)),
    invoices=(
        Invoice("INV-1", "V1", 100_000, "open"),
        Invoice("INV-2", "V1", 100_000, "open", duplicate_of="INV-1"),
        Invoice("INV-3", "V1", 200_000, "paid"),
        Invoice("INV-4", "V2", 5_000, "open"),
        Invoice("INV-5", "V1", 7_000, "void"),
    ),
)


@pytest.fixture
def ledger() -> Ledger:
    return Ledger(FX)


@pytest.fixture
def gate(ledger: Ledger) -> WriteGate:
    return WriteGate(ledger, run_id="t")


def status(ledger: Ledger, inv: str) -> str:
    found = ledger.invoice(inv)
    assert found is not None
    return found.status


def test_soft_write_executes_and_is_logged(gate: WriteGate, ledger: Ledger) -> None:
    d = gate.submit(Action("pay_invoice", {"invoice_id": "INV-1"}))
    assert d.status is Status.EXECUTED
    assert d.quote is not None and d.quote.tier is Tier.SOFT
    assert "open -> paid" in d.quote.diff
    assert status(ledger, "INV-1") == "paid"
    events = [e for e, detail in ledger.decision_log() if e == "gate"]
    assert events == ["gate"]


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        (Action("pay_invoice", {"invoice_id": "INV-3"}), "paid"),
        (Action("pay_invoice", {"invoice_id": "INV-5"}), "void"),
        (Action("pay_invoice", {"invoice_id": "INV-2"}), "duplicates"),
        (Action("pay_invoice", {"invoice_id": "INV-4"}), "not verified"),
        (Action("pay_invoice", {"invoice_id": "INV-404"}), "does not exist"),
        (Action("void_invoice", {"invoice_id": "INV-1"}), "not a duplicate"),
        (Action("issue_refund", {"invoice_id": "INV-1", "amount": 10}), "only paid"),
        (Action("issue_refund", {"invoice_id": "INV-3", "amount": 2500}), "exceeds the refundable"),
        (Action("issue_refund", {"invoice_id": "INV-3", "amount": -5}), "positive"),
    ],
)
def test_policy_violations_block_and_write_nothing(
    gate: WriteGate, ledger: Ledger, action: Action, reason: str
) -> None:
    before = (ledger.all_invoices(), ledger.refunded_cents("INV-3"))
    d = gate.submit(action)
    assert d.status is Status.VIOLATION
    assert reason in d.message
    assert (ledger.all_invoices(), ledger.refunded_cents("INV-3")) == before


def test_large_refund_needs_a_human(gate: WriteGate, ledger: Ledger) -> None:
    d = gate.submit(Action("issue_refund", {"invoice_id": "INV-3", "amount": "600.00"}))
    assert d.status is Status.NEEDS_HUMAN and d.quote is not None
    assert d.quote.tier is Tier.HARD
    assert ledger.refunded_cents("INV-3") == 0


def test_bank_change_is_critical_and_never_auto_signed(gate: WriteGate, ledger: Ledger) -> None:
    d = gate.submit(Action("update_bank_details", {"vendor_id": "V1", "account": "GB11EVIL0000"}))
    assert d.status is Status.NEEDS_HUMAN and d.quote is not None
    assert d.quote.tier is Tier.CRITICAL
    vendor = ledger.vendor("V1")
    assert vendor is not None and vendor.bank_account == "GB00BANK1234"


@pytest.mark.parametrize(
    "action",
    [
        Action("drop_tables", {}),
        Action("pay_invoice", {}),
        Action("pay_invoice", {"invoice_id": "INV-1", "force": "yes"}),
        Action("issue_refund", {"invoice_id": "INV-3", "amount": "12O.00"}),
        Action("finish", {"summary": "not a write"}),
    ],
)
def test_malformed_or_unknown_requests_are_invalid(gate: WriteGate, action: Action) -> None:
    assert gate.submit(action).status is Status.INVALID


def test_tampered_payload_is_rejected(gate: WriteGate, ledger: Ledger) -> None:
    quote = gate.quote(Action("issue_refund", {"invoice_id": "INV-3", "amount": 10}))
    sig = PolicyApprover().sign(quote)
    assert sig is not None
    tampered = Action("issue_refund", {"invoice_id": "INV-3", "amount": 499.0})
    with pytest.raises(TamperedPayload):
        gate.execute(quote, sig, tampered)
    assert ledger.refunded_cents("INV-3") == 0


def test_signature_for_a_different_quote_is_rejected(gate: WriteGate) -> None:
    quote = gate.quote(Action("pay_invoice", {"invoice_id": "INV-1"}))
    forged = Signature(fingerprint(Action("pay_invoice", {"invoice_id": "INV-9"})), "mallory")
    with pytest.raises(TamperedPayload):
        gate.execute(quote, forged, quote.action)


def test_idempotent_replay_never_double_writes(gate: WriteGate, ledger: Ledger) -> None:
    refund = Action("issue_refund", {"invoice_id": "INV-3", "amount": "100.00"})
    first, second = gate.submit(refund), gate.submit(refund)
    assert first.status is Status.EXECUTED
    assert second.status is Status.REPLAYED and not second.blocked
    assert ledger.refunded_cents("INV-3") == 10_000


def test_gate_fails_closed_when_the_executor_crashes(
    gate: WriteGate, ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chaos test: blow up mid-transaction; nothing may be written, nothing may raise."""

    def boom(*_: object) -> None:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(ledger, "record_applied_write", boom)
    d = gate.submit(Action("pay_invoice", {"invoice_id": "INV-1"}))
    assert d.status is Status.GATE_ERROR and d.blocked
    assert status(ledger, "INV-1") == "open"  # the status update was rolled back


def test_gate_fails_closed_when_the_approver_crashes(ledger: Ledger) -> None:
    class BrokenApprover:
        def sign(self, quote: Quote) -> Signature | None:
            raise ConnectionError("approval service down")

    d = WriteGate(ledger, BrokenApprover()).submit(Action("pay_invoice", {"invoice_id": "INV-1"}))
    assert d.status is Status.GATE_ERROR
    assert status(ledger, "INV-1") == "open"


def test_decision_log_is_append_only(ledger: Ledger) -> None:
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger.raw_execute("DELETE FROM decision_log")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger.raw_execute("UPDATE decision_log SET event = 'x'")


def test_idempotency_survives_differently_spelled_ids(gate: WriteGate, ledger: Ledger) -> None:
    """Ids are matched case-insensitively, so the fingerprint must use the canonical id:
    otherwise a retry spelled 'inv-3' would be a second refund rather than a replay."""
    first = gate.submit(Action("issue_refund", {"invoice_id": "INV-3", "amount": "100.00"}))
    second = gate.submit(Action("issue_refund", {"invoice_id": " inv-3 ", "amount": 100}))
    assert first.status is Status.EXECUTED
    assert second.status is Status.REPLAYED
    assert ledger.refunded_cents("INV-3") == 10_000


def test_bank_change_quote_resolves_a_vendor_name_to_its_id(gate: WriteGate) -> None:
    by_id = gate.quote(Action("update_bank_details", {"vendor_id": "V1", "account": "GB1"}))
    by_name = gate.quote(Action("update_bank_details", {"vendor_id": "acme", "account": "GB1"}))
    assert by_id.fingerprint == by_name.fingerprint


@pytest.mark.parametrize("amount", ["120.005", "1e400", "nan", "-0.001"])
def test_inexact_or_absurd_amounts_are_rejected_not_rounded(gate: WriteGate, amount: str) -> None:
    """The gate executes exactly what was quoted, so it never silently rounds money."""
    d = gate.submit(Action("issue_refund", {"invoice_id": "INV-3", "amount": amount}))
    assert d.blocked
    assert d.status in (Status.INVALID, Status.VIOLATION)
