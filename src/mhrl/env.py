"""One episode's world: a fresh ledger, the write gate, and the read tools.

The environment speaks :class:`~mhrl.types.Action` and
:class:`~mhrl.types.Observation`; it knows nothing about prompt formats. That
is what lets every harness, and the compiled loop, share one control plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mhrl.domain.ledger import Invoice, Ledger
from mhrl.domain.tasks import Task
from mhrl.tools import registry
from mhrl.tools.gate import Approver, GateDecision, WriteGate
from mhrl.tools.registry import ArgumentError, Permission
from mhrl.types import Action, Observation, cents_to_str


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    terminal: bool = False
    blocked: bool = False
    decision: GateDecision | None = None


@dataclass
class EpisodeFacts:
    """What happened, as recorded by the environment (not as claimed by the model)."""

    write_attempts: list[Action] = field(default_factory=list)
    blocked: list[GateDecision] = field(default_factory=list)
    finished: bool = False
    escalated: bool = False
    skipped: bool = False


class Environment:
    def __init__(
        self, task: Task, run_meta: dict[str, str] | None = None, approver: Approver | None = None
    ) -> None:
        self.task = task
        meta = {"task": task.id, **(run_meta or {})}
        self.ledger = Ledger(task.fixture, meta)
        self.gate = WriteGate(self.ledger, approver, run_id=meta.get("run_id", task.id))
        self.facts = EpisodeFacts()
        self.steps = 0  # every action taken, reads included: the same unit as an agent's turns

    def step(self, action: Action) -> StepResult:
        self.steps += 1
        spec = registry.get(action.tool)
        if spec is None:
            # Undeclared tools are blocked, not ignored: request validation fails closed.
            self.facts.write_attempts.append(action)
            decision = self.gate.submit(action)
            self.facts.blocked.append(decision)
            return StepResult(
                Observation(action.tool, False, error=decision.message),
                terminal=True,
                blocked=True,
                decision=decision,
            )
        if spec.permission is Permission.WRITE:
            return self._write(action)
        try:
            valid = registry.validate(action)
        except ArgumentError as e:
            return StepResult(Observation(action.tool, False, error=str(e)))
        if spec.permission is Permission.CONTROL:
            return self._control(valid)
        return StepResult(self._read(valid))

    def skip(self, reason: str) -> None:
        """Used by the compiled loop: no recipe matched, so hand over and say so."""
        self.facts.skipped = True
        self.steps += 1
        self._control(Action("escalate", {"reason": f"skipped: {reason}"}))

    # -- internals ---------------------------------------------------------------------

    def _write(self, action: Action) -> StepResult:
        self.facts.write_attempts.append(action)
        decision = self.gate.submit(action)
        if decision.blocked:
            self.facts.blocked.append(decision)
            return StepResult(
                Observation(action.tool, False, error=decision.message),
                terminal=True,
                blocked=True,
                decision=decision,
            )
        return StepResult(
            Observation(action.tool, True, data={"result": decision.message}), decision=decision
        )

    def _control(self, action: Action) -> StepResult:
        if action.tool == "escalate":
            with self.ledger.transaction():
                self.ledger.add_escalation(str(action.args["reason"]))
            self.facts.escalated = True
            self.ledger.log("escalated", {"reason": action.args["reason"]})
            return StepResult(
                Observation("escalate", True, data={"result": "handed to a human"}), terminal=True
            )
        self.facts.finished = True
        self.ledger.log("finished", {"summary": action.args["summary"]})
        return StepResult(Observation("finish", True, data={"result": "done"}), terminal=True)

    def _read(self, action: Action) -> Observation:
        led, a = self.ledger, action.args
        if action.tool == "list_invoices":
            vendor = led.vendor_by_name(str(a["vendor"]))
            if vendor is None:
                return Observation(action.tool, False, error=f"no vendor named {a['vendor']!r}")
            return Observation(
                action.tool,
                True,
                data=[self._invoice_view(i) for i in led.invoices_for_vendor(vendor.id)],
            )
        if action.tool == "get_invoice":
            inv = led.invoice(str(a["invoice_id"]))
            if inv is None:
                return Observation(action.tool, False, error=f"no invoice {a['invoice_id']!r}")
            return Observation(action.tool, True, data=self._invoice_view(inv))
        vendor = led.vendor_by_name(str(a["name"]))
        if vendor is None:
            return Observation(action.tool, False, error=f"no vendor named {a['name']!r}")
        return Observation(
            action.tool,
            True,
            data={
                "vendor_id": vendor.id,
                "name": vendor.name,
                "bank_account": "****" + vendor.bank_account[-4:],
                "verified": vendor.verified,
            },
        )

    def _invoice_view(self, inv: Invoice) -> dict[str, object]:
        vendor = self.ledger.vendor(inv.vendor_id)
        view: dict[str, object] = {
            "invoice_id": inv.id,
            "vendor": vendor.name if vendor else inv.vendor_id,
            "amount": cents_to_str(inv.amount_cents),
            "status": inv.status,
        }
        refunded = self.ledger.refunded_cents(inv.id)
        if refunded:
            view["refunded"] = cents_to_str(refunded)
        if inv.duplicate_of:
            view["duplicate_of"] = inv.duplicate_of
        if inv.memo:
            view["memo"] = inv.memo
        return view
