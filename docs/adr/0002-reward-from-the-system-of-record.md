# 2. Rewards come from the system of record, not from a judge

**Status:** accepted · **Date:** 2026-11-14

## Context
An agent that says "Paid the invoice" has proven nothing. LLM judges are cheap to add,
but they score the *claim*, they have their own error rates, and under RL a policy will
learn to please them.

## Decision
`rewards.verify` reads the **entire** ledger back (statuses, refunds, bank details,
escalations) and compares it with the task's expected end state. Any extra write is
wrong, even if the target write is right. Outcomes are a closed taxonomy (`Outcome`), and
"said done, ledger disagrees" is its own class (`claimed_not_written`).

reward = `1 + 0.1·(1 − turns/max_turns)` when solved, else `0`. The efficiency bonus applies
only to correct episodes, following the Harness Adapters recipe. A blocked write ends the
episode with reward 0.

## Consequences
- (+) The reward cannot be talked into a higher score; RL optimises the real objective.
- (+) Failure clusters are counts, not free-text reading (`eval/errors.py`).
- (−) Tasks must have a checkable end state. Open-ended work (drafting, research) would
  need a calibrated critic. That critic could route work to a person, but it would
  **never sign**.
