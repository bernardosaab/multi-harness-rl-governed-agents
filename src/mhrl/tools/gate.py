"""The fail-closed write gate: the only path from a proposed write to the ledger.

    proposal ──► quote (exact diff + fingerprint + tier)
             ──► sign  (policy auto-signs SOFT; HARD/CRITICAL need a human)
             ──► execute only the payload whose fingerprint was signed
             ──► idempotent, transactional, logged

Every failure mode ends in *no write*: a policy violation, a missing signature,
a tampered payload, bad arguments, or an unexpected exception inside the gate
itself. A prompt can ask the model to behave; only this code decides.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from mhrl.domain.ledger import Ledger
from mhrl.domain.tasks import REFUND_LIMIT_CENTS
from mhrl.tools import registry
from mhrl.tools.registry import ArgumentError, Permission
from mhrl.types import Action, cents_to_str

POLICY_VERSION = "ap-policy/2026.11-1"


class Tier(StrEnum):
    SOFT = "soft"  # reversible, small: the policy engine may sign
    HARD = "hard"  # money above a limit: a named human signs
    CRITICAL = "critical"  # where money goes (bank details): a human signs, always


class Status(StrEnum):
    EXECUTED = "executed"
    REPLAYED = "replayed"  # same signed payload seen before: no second write
    NEEDS_HUMAN = "needs_human"
    VIOLATION = "violation"
    INVALID = "invalid"
    TAMPERED = "tampered"
    GATE_ERROR = "gate_error"

    @property
    def blocked(self) -> bool:
        return self not in (Status.EXECUTED, Status.REPLAYED)


class PolicyViolation(Exception):
    pass


class TamperedPayload(Exception):
    pass


@dataclass(frozen=True)
class Quote:
    action: Action
    diff: str
    tier: Tier
    fingerprint: str


@dataclass(frozen=True)
class Signature:
    fingerprint: str
    signer: str


@dataclass(frozen=True)
class GateDecision:
    status: Status
    message: str
    quote: Quote | None = None

    @property
    def blocked(self) -> bool:
        return self.status.blocked


class Approver(Protocol):
    def sign(self, quote: Quote) -> Signature | None: ...


class PolicyApprover:
    """Signs SOFT quotes automatically. Nobody is on call during an episode, so
    HARD and CRITICAL quotes go unsigned and the write fails closed; the agent
    is expected to ``escalate`` instead of trying."""

    name = "policy-engine"

    def sign(self, quote: Quote) -> Signature | None:
        if quote.tier is Tier.SOFT:
            return Signature(quote.fingerprint, self.name)
        return None


def fingerprint(action: Action) -> str:
    return hashlib.sha256(action.canonical().encode()).hexdigest()


class WriteGate:
    def __init__(
        self, ledger: Ledger, approver: Approver | None = None, run_id: str = "run"
    ) -> None:
        self.ledger = ledger
        self.approver = approver or PolicyApprover()
        self.run_id = run_id

    # -- public ------------------------------------------------------------------------

    def submit(self, proposal: Action) -> GateDecision:
        """Quote, sign and execute one proposed write. Never raises."""
        decision = self._submit(proposal)
        self.ledger.log(
            "gate",
            {
                "status": decision.status.value,
                "tool": proposal.tool,
                "args": dict(proposal.args),
                "message": decision.message,
                "tier": decision.quote.tier.value if decision.quote else None,
                "fingerprint": decision.quote.fingerprint if decision.quote else None,
                "policy": POLICY_VERSION,
            },
        )
        return decision

    def quote(self, proposal: Action) -> Quote:
        """Validate and price a write. Raises on anything the policy forbids."""
        action = registry.validate(proposal)
        spec = registry.get(action.tool)
        if spec is None or spec.permission is not Permission.WRITE:
            raise ArgumentError(f"{action.tool!r} is not a write tool")
        tier, diff = self._evaluate(action)
        return Quote(action, diff, tier, fingerprint(action))

    def execute(self, quote: Quote, signature: Signature, payload: Action) -> GateDecision:
        """Apply exactly the signed payload, once, atomically."""
        if not (fingerprint(payload) == signature.fingerprint == quote.fingerprint):
            raise TamperedPayload("payload does not match the signed quote")
        key = hashlib.sha256(f"{self.run_id}:{quote.fingerprint}".encode()).hexdigest()
        previous = self.ledger.applied_write(key)
        if previous is not None:
            return GateDecision(Status.REPLAYED, f"already applied: {previous}", quote)
        with self.ledger.transaction():
            result = self._apply(quote.action)
            self.ledger.record_applied_write(key, quote.fingerprint, result)
        return GateDecision(Status.EXECUTED, result, quote)

    # -- internals ---------------------------------------------------------------------

    def _submit(self, proposal: Action) -> GateDecision:
        try:
            replay = self._replay(proposal)
            if replay is not None:
                return replay
            quote = self.quote(proposal)
            signature = self.approver.sign(quote)
            if signature is None:
                return GateDecision(
                    Status.NEEDS_HUMAN,
                    f"{quote.tier.value} tier change needs a human signature: {quote.diff}",
                    quote,
                )
            return self.execute(quote, signature, quote.action)
        except ArgumentError as e:
            return GateDecision(Status.INVALID, str(e))
        except PolicyViolation as e:
            return GateDecision(Status.VIOLATION, str(e))
        except TamperedPayload as e:
            return GateDecision(Status.TAMPERED, str(e))
        except Exception as e:  # fail closed on anything unexpected
            return GateDecision(Status.GATE_ERROR, f"gate failed closed: {type(e).__name__}")

    def _replay(self, proposal: Action) -> GateDecision | None:
        try:
            action = registry.validate(proposal)
        except ArgumentError:
            return None
        key = hashlib.sha256(f"{self.run_id}:{fingerprint(action)}".encode()).hexdigest()
        previous = self.ledger.applied_write(key)
        if previous is None:
            return None
        return GateDecision(Status.REPLAYED, f"already applied: {previous}")

    def _evaluate(self, action: Action) -> tuple[Tier, str]:
        led, a = self.ledger, action.args
        if action.tool in ("pay_invoice", "void_invoice", "issue_refund"):
            inv = led.invoice(str(a["invoice_id"]))
            if inv is None:
                raise PolicyViolation(f"invoice {a['invoice_id']} does not exist")
        if action.tool == "pay_invoice":
            assert inv is not None
            vendor = led.vendor(inv.vendor_id)
            if inv.status != "open":
                raise PolicyViolation(f"{inv.id} is {inv.status}, only open invoices can be paid")
            if inv.duplicate_of:
                raise PolicyViolation(f"{inv.id} duplicates {inv.duplicate_of}; void it instead")
            if vendor is None or not vendor.verified:
                raise PolicyViolation(f"vendor of {inv.id} is not verified")
            return Tier.SOFT, (
                f"invoices[{inv.id}].status: open -> paid "
                f"(${cents_to_str(inv.amount_cents)} to {vendor.name})"
            )
        if action.tool == "void_invoice":
            assert inv is not None
            if inv.status != "open":
                raise PolicyViolation(f"{inv.id} is {inv.status}, only open invoices can be voided")
            if not inv.duplicate_of:
                raise PolicyViolation(f"{inv.id} is not a duplicate; refusing to void it")
            return Tier.SOFT, f"invoices[{inv.id}].status: open -> void (duplicate)"
        if action.tool == "issue_refund":
            assert inv is not None
            cents = registry.to_cents(a["amount"])
            if inv.status != "paid":
                raise PolicyViolation(
                    f"{inv.id} is {inv.status}; only paid invoices can be refunded"
                )
            if cents <= 0:
                raise PolicyViolation("refund amount must be positive")
            room = inv.amount_cents - led.refunded_cents(inv.id)
            if cents > room:
                raise PolicyViolation(
                    f"refund ${cents_to_str(cents)} exceeds the refundable ${cents_to_str(room)}"
                )
            tier = Tier.HARD if cents > REFUND_LIMIT_CENTS else Tier.SOFT
            return tier, f"refunds += ({inv.id}, ${cents_to_str(cents)})"
        if action.tool == "update_bank_details":
            vendor = led.vendor(str(a["vendor_id"])) or led.vendor_by_name(str(a["vendor_id"]))
            if vendor is None:
                raise PolicyViolation(f"vendor {a['vendor_id']} does not exist")
            return Tier.CRITICAL, f"vendors[{vendor.id}].bank_account -> {a['account']}"
        raise PolicyViolation(f"no policy for {action.tool}")  # unreachable by construction

    def _apply(self, action: Action) -> str:
        led, a = self.ledger, action.args
        if action.tool == "pay_invoice":
            inv = led.invoice(str(a["invoice_id"]))
            assert inv is not None
            led.set_invoice_status(inv.id, "paid")
            return f"{inv.id} paid"
        if action.tool == "void_invoice":
            inv = led.invoice(str(a["invoice_id"]))
            assert inv is not None
            led.set_invoice_status(inv.id, "void")
            return f"{inv.id} voided"
        if action.tool == "issue_refund":
            inv = led.invoice(str(a["invoice_id"]))
            assert inv is not None
            cents = registry.to_cents(a["amount"])
            led.add_refund(inv.id, cents)
            return f"refunded ${cents_to_str(cents)} on {inv.id}"
        if action.tool == "update_bank_details":
            vendor = led.vendor(str(a["vendor_id"])) or led.vendor_by_name(str(a["vendor_id"]))
            assert vendor is not None
            led.set_bank_account(vendor.id, str(a["account"]))
            return f"bank details updated for {vendor.id}"
        raise PolicyViolation(f"no executor for {action.tool}")
