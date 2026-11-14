"""Accounts-payable policy constants: one source for the gate, the tasks and the prompts."""

from __future__ import annotations

REFUND_LIMIT_CENTS = 50_000  # refunds above this are HARD tier: a named human signs


def refund_limit_text() -> str:
    return f"${REFUND_LIMIT_CENTS // 100:,}.{REFUND_LIMIT_CENTS % 100:02d}"
