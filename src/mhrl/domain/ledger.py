"""A tiny accounts-payable system of record backed by in-memory SQLite.

The ledger is deliberately dumb: it stores facts and enforces integrity
constraints. *Whether* a change is allowed is the write gate's job
(:mod:`mhrl.tools.gate`); the ledger only offers the primitive mutations the
gate's executor calls inside a transaction.

The ``decision_log`` table is append-only, enforced by SQLite triggers rather
than by convention, so a replayable record survives even a buggy caller.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

_SCHEMA = """
CREATE TABLE vendors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    bank_account TEXT NOT NULL,
    verified INTEGER NOT NULL CHECK (verified IN (0, 1))
);
CREATE TABLE invoices (
    id TEXT PRIMARY KEY,
    vendor_id TEXT NOT NULL REFERENCES vendors(id),
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    status TEXT NOT NULL CHECK (status IN ('open', 'paid', 'void')),
    memo TEXT NOT NULL DEFAULT '',
    duplicate_of TEXT REFERENCES invoices(id)
);
CREATE TABLE refunds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id TEXT NOT NULL REFERENCES invoices(id),
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0)
);
CREATE TABLE escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reason TEXT NOT NULL
);
CREATE TABLE applied_writes (
    idempotency_key TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    result TEXT NOT NULL
);
CREATE TABLE decision_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT NOT NULL,
    detail TEXT NOT NULL
);
CREATE TRIGGER decision_log_no_update BEFORE UPDATE ON decision_log
BEGIN SELECT RAISE(ABORT, 'decision_log is append-only'); END;
CREATE TRIGGER decision_log_no_delete BEFORE DELETE ON decision_log
BEGIN SELECT RAISE(ABORT, 'decision_log is append-only'); END;
"""


@dataclass(frozen=True)
class Vendor:
    id: str
    name: str
    bank_account: str
    verified: bool = True


@dataclass(frozen=True)
class Invoice:
    id: str
    vendor_id: str
    amount_cents: int
    status: str = "open"
    memo: str = ""
    duplicate_of: str | None = None


@dataclass(frozen=True)
class Fixture:
    """The initial state of the system of record for one task."""

    vendors: tuple[Vendor, ...]
    invoices: tuple[Invoice, ...]
    refunds: tuple[tuple[str, int], ...] = field(default_factory=tuple)


class Ledger:
    """Owns one SQLite connection. One ledger per episode; nothing is shared."""

    def __init__(self, fixture: Fixture, run_meta: Mapping[str, str] | None = None) -> None:
        self._db = sqlite3.connect(":memory:", isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
        with self.transaction():
            for v in fixture.vendors:
                self._db.execute(
                    "INSERT INTO vendors VALUES (?, ?, ?, ?)",
                    (v.id, v.name, v.bank_account, int(v.verified)),
                )
            for inv in fixture.invoices:
                self._db.execute(
                    "INSERT INTO invoices VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        inv.id,
                        inv.vendor_id,
                        inv.amount_cents,
                        inv.status,
                        inv.memo,
                        inv.duplicate_of,
                    ),
                )
            for invoice_id, cents in fixture.refunds:
                self._db.execute(
                    "INSERT INTO refunds (invoice_id, amount_cents) VALUES (?, ?)",
                    (invoice_id, cents),
                )
        self.log("run_started", dict(run_meta or {}))

    # -- transactions -------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """All-or-nothing: an exception anywhere inside rolls every statement back."""
        self._db.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        else:
            self._db.execute("COMMIT")

    # -- reads --------------------------------------------------------------------------

    def vendor_by_name(self, name: str) -> Vendor | None:
        row = self._db.execute(
            "SELECT * FROM vendors WHERE lower(name) = lower(?)", (name.strip(),)
        ).fetchone()
        return _vendor(row) if row else None

    def vendor(self, vendor_id: str) -> Vendor | None:
        row = self._db.execute("SELECT * FROM vendors WHERE id = ?", (vendor_id,)).fetchone()
        return _vendor(row) if row else None

    def invoice(self, invoice_id: str) -> Invoice | None:
        row = self._db.execute(
            "SELECT * FROM invoices WHERE upper(id) = upper(?)", (invoice_id.strip(),)
        ).fetchone()
        return _invoice(row) if row else None

    def invoices_for_vendor(self, vendor_id: str) -> list[Invoice]:
        rows = self._db.execute(
            "SELECT * FROM invoices WHERE vendor_id = ? ORDER BY id", (vendor_id,)
        ).fetchall()
        return [_invoice(r) for r in rows]

    def refunded_cents(self, invoice_id: str) -> int:
        row = self._db.execute(
            "SELECT COALESCE(SUM(amount_cents), 0) AS s FROM refunds WHERE invoice_id = ?",
            (invoice_id,),
        ).fetchone()
        return int(row["s"])

    def all_vendors(self) -> list[Vendor]:
        return [_vendor(r) for r in self._db.execute("SELECT * FROM vendors ORDER BY id")]

    def all_invoices(self) -> list[Invoice]:
        return [_invoice(r) for r in self._db.execute("SELECT * FROM invoices ORDER BY id")]

    def escalation_reasons(self) -> list[str]:
        return [r["reason"] for r in self._db.execute("SELECT reason FROM escalations")]

    def applied_write(self, idempotency_key: str) -> str | None:
        row = self._db.execute(
            "SELECT result FROM applied_writes WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        return str(row["result"]) if row else None

    def decision_log(self) -> list[tuple[str, dict[str, object]]]:
        rows = self._db.execute("SELECT event, detail FROM decision_log ORDER BY seq")
        return [(r["event"], json.loads(r["detail"])) for r in rows]

    # -- primitive mutations (only the gate's executor calls these) ----------------------

    def set_invoice_status(self, invoice_id: str, status: str) -> None:
        self._db.execute("UPDATE invoices SET status = ? WHERE id = ?", (status, invoice_id))

    def add_refund(self, invoice_id: str, cents: int) -> None:
        self._db.execute(
            "INSERT INTO refunds (invoice_id, amount_cents) VALUES (?, ?)", (invoice_id, cents)
        )

    def set_bank_account(self, vendor_id: str, account: str) -> None:
        self._db.execute("UPDATE vendors SET bank_account = ? WHERE id = ?", (account, vendor_id))

    def add_escalation(self, reason: str) -> None:
        self._db.execute("INSERT INTO escalations (reason) VALUES (?)", (reason,))

    def record_applied_write(self, idempotency_key: str, fingerprint: str, result: str) -> None:
        self._db.execute(
            "INSERT INTO applied_writes VALUES (?, ?, ?)", (idempotency_key, fingerprint, result)
        )

    def log(self, event: str, detail: Mapping[str, object]) -> None:
        self._db.execute(
            "INSERT INTO decision_log (event, detail) VALUES (?, ?)",
            (event, json.dumps(dict(detail), sort_keys=True, default=str)),
        )

    def raw_execute(self, sql: str) -> None:
        """Escape hatch for tests that prove the triggers hold."""
        self._db.execute(sql)

    def close(self) -> None:
        self._db.close()


def _vendor(row: sqlite3.Row) -> Vendor:
    return Vendor(row["id"], row["name"], row["bank_account"], bool(row["verified"]))


def _invoice(row: sqlite3.Row) -> Invoice:
    return Invoice(
        row["id"],
        row["vendor_id"],
        row["amount_cents"],
        row["status"],
        row["memo"],
        row["duplicate_of"],
    )
