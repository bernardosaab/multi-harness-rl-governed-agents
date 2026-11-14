"""Tool declarations, shared by every harness.

Harnesses differ in how a tool call is *spelled*; the set of tools, their
argument types and their permission levels are defined once, here.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from mhrl.types import Action, Scalar


class Permission(StrEnum):
    READ = "read"  # on by default, no side effects
    WRITE = "write"  # never executes directly: goes through the gate
    CONTROL = "control"  # ends the episode (finish / escalate)


@dataclass(frozen=True)
class Param:
    name: str
    type: type[str] | type[float]
    description: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    permission: Permission
    params: tuple[Param, ...]


_CENT = Decimal("0.01")
_MAX_AMOUNT = Decimal("1000000000")


class ArgumentError(ValueError):
    """Arguments don't match a tool's declared signature."""


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "list_invoices",
        "List every invoice for a vendor, with status, amount and memo.",
        Permission.READ,
        (Param("vendor", str, "vendor name"),),
    ),
    ToolSpec(
        "get_invoice",
        "Show one invoice, including how much has been refunded.",
        Permission.READ,
        (Param("invoice_id", str, "e.g. INV-AB101"),),
    ),
    ToolSpec(
        "find_vendor",
        "Look up a vendor by name.",
        Permission.READ,
        (Param("name", str, "vendor name"),),
    ),
    ToolSpec(
        "pay_invoice",
        "Pay an open invoice in full.",
        Permission.WRITE,
        (Param("invoice_id", str, "invoice to pay"),),
    ),
    ToolSpec(
        "void_invoice",
        "Void an open invoice that duplicates another one.",
        Permission.WRITE,
        (Param("invoice_id", str, "duplicate invoice to void"),),
    ),
    ToolSpec(
        "issue_refund",
        "Refund part of a paid invoice, in dollars.",
        Permission.WRITE,
        (Param("invoice_id", str, "paid invoice"), Param("amount", float, "dollars, e.g. 120.50")),
    ),
    ToolSpec(
        "update_bank_details",
        "Change the bank account a vendor is paid into.",
        Permission.WRITE,
        (Param("vendor_id", str, "vendor id"), Param("account", str, "new account number")),
    ),
    ToolSpec(
        "escalate",
        "Hand the request to a human approver, with a reason. Ends the task.",
        Permission.CONTROL,
        (Param("reason", str, "why a human is needed"),),
    ),
    ToolSpec(
        "finish",
        "Declare the task complete. Ends the task.",
        Permission.CONTROL,
        (Param("summary", str, "what you did"),),
    ),
)
_BY_NAME = {t.name: t for t in TOOLS}


def get(name: str) -> ToolSpec | None:
    return _BY_NAME.get(name)


def validate(action: Action) -> Action:
    """Return a copy of ``action`` with arguments coerced to their declared types.

    Raises :class:`ArgumentError` for unknown tools, missing, extra or ill-typed
    arguments. Coercion is strict: ``"12O.00"`` is not a number.
    """
    spec = get(action.tool)
    if spec is None:
        raise ArgumentError(f"unknown tool {action.tool!r}")
    declared = {p.name: p for p in spec.params}
    extra = set(action.args) - set(declared)
    missing = set(declared) - set(action.args)
    if extra:
        raise ArgumentError(f"{spec.name}: unexpected argument(s) {sorted(extra)}")
    if missing:
        raise ArgumentError(f"{spec.name}: missing argument(s) {sorted(missing)}")
    coerced: dict[str, Scalar] = {}
    for name, param in declared.items():
        coerced[name] = _coerce(spec.name, name, param.type, action.args[name])
    return Action(spec.name, coerced)


def _coerce(tool: str, name: str, kind: type, value: Scalar) -> Scalar:
    if kind is float:
        raw = str(value).replace("$", "").replace(",", "").strip()
        try:
            dec = Decimal(raw)
        except InvalidOperation as e:
            raise ArgumentError(f"{tool}.{name}: {value!r} is not a number") from e
        if not dec.is_finite() or abs(dec) > _MAX_AMOUNT:
            raise ArgumentError(f"{tool}.{name}: {value!r} is not a valid amount")
        if dec != dec.quantize(_CENT):
            # Never round money silently: the gate executes exactly what was quoted.
            raise ArgumentError(f"{tool}.{name}: {value!r} has more than two decimal places")
        return float(dec)
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ArgumentError(f"{tool}.{name}: expected text")
    text = str(value).strip()
    if not text:
        raise ArgumentError(f"{tool}.{name}: must not be empty")
    return text


def to_cents(dollars: Scalar) -> int:
    return int((Decimal(str(dollars)) * 100).to_integral_value())
