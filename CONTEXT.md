# Glossary

| Term | Meaning here |
|---|---|
| **Harness (inner)** | How a model is prompted and how its tool calls are spelled: `json`, `react_xml`, `pycall`, `plaintext`. Owns formatting only. |
| **Outer harness** | Everything that binds what actually happens: the environment, the write gate, the caps in the runtime, the decision log. Same for every model and every inner harness. |
| **Held-out harness** | `plaintext`. Never seen in training; measures whether a gain is portable or just adaptation to a format. |
| **Action** | One parsed tool call (`tool` + typed `args`). Harness-independent. |
| **Quote** | The exact diff a write would make, plus its SHA-256 fingerprint and blast-radius tier, produced *before* anything executes. |
| **Tier** | SOFT (policy engine may sign), HARD (money above a limit: a human signs), CRITICAL (where money goes: a human signs, always). |
| **Signature** | A signer's approval bound to one fingerprint. Only that exact payload can execute. |
| **Fail closed** | Any error, violation, missing signature or exception means *no write*, logged. |
| **Decision log** | Append-only table (enforced by SQLite triggers) recording every gate decision, escalation and finish, with policy version and run id. |
| **Compiled loop** | Versioned deterministic recipes with no model on the normal path. Skips (escalates) anything it does not recognise; never guesses. |
| **Loop engineering** | The usual non-training fixes: few-shot exemplar, corrective re-prompts on parse errors, a self-check turn before `finish`. |
| **Routed hybrid** | Compiled loop where a recipe matches, an agent for the long tail. |
| **Outcome** | Closed taxonomy for every episode (`solved`, `blocked_write`, `injection_followed`, `claimed_not_written`, ...). |
| **Claimed not written** | The agent called `finish`, but the ledger does not hold the expected state. |
| **Grouping boundary** | Which rollouts share a GRPO advantage baseline. Here: same task *and* same harness. |
| **pass^k** | Share of cases solved on *every* one of k attempts (reliability), not *any* (pass@k). |
| **Ratchet** | `mhrl ratchet` freezes an observed failure (exact replies + verdict) into `tests/regressions/`, so it stays a permanent test. |
