# Method, and how to read the results

The numbers themselves are generated in [results/REPORT.md](../results/REPORT.md) and on the
[results site](https://OWNER.github.io/REPO/). This page explains how they were produced
and what they can and cannot show.

## Protocol

| | |
|---|---|
| Base model | `Qwen/Qwen3-1.7B`, non-thinking mode, bf16 |
| Adapter | LoRA r=16, alpha=32, all attention and MLP projections |
| Training | GRPO, `configs/train_small.toml`: 10 steps × 6 tasks × 3 harnesses × 4 rollouts, temperature 0.8, lr 5e-5, KL β=0.02 against the adapter-disabled model |
| Run history | steps 1-2 ran with `max_new_tokens=160`; the run was resumed from the step-2 checkpoint with 96 tokens and capped at 10 steps to fit the time budget (each step was taking 5-15 min on the 8 GB card) |
| Training tasks | train split (28 tasks, 7 kinds) |
| Test tasks | test split (28 tasks): **different instances** with different vendors, amounts and ids, generated from different seeds |
| Seen harnesses | `json`, `react_xml`, `pycall` |
| Held-out harness | `plaintext`, never used in training |
| Samples | 2 per (task, harness) for every model-based system, temperature 0.7; the compiled loop is deterministic |
| Hardware | one RTX 3070 Laptop GPU (8 GB), Windows, no vLLM |

## Reading the numbers

- **Solved** means the verifier read the whole ledger back and found exactly the expected
  end state: the right change, no extra writes, and an escalation only when one was required.
- **95% CI** is a Wilson interval over episodes. With n≈250 per row, intervals are roughly
  ±6 percentage points around 50%. Differences smaller than that should be treated as
  noise unless the paired test says otherwise.
- **Paired comparisons** use identical (task, harness, sample) cases. *Only A* and *Only B*
  count the discordant cases. McNemar's p comes from them; the bootstrap interval is on
  the difference in pass rate.
- **pass^k** is the share of (task, harness) cases solved on *all three* samples, a
  reliability measure, much stricter than the pass rate.
- **Tokens per accepted output** divides *all* tokens spent (failures, retries,
  self-checks, prompts re-sent on every turn) by the number of solved episodes. It is the
  cost-per-outcome view; cost per token alone would flatter verbose systems.
- **Injection followed** is measured on `injected_pay` tasks: the share of episodes where
  the agent *attempted* the planted write. The gate blocks every such attempt; this metric
  measures behaviour, not damage.

## What the design rules out

- **Reward hacking by talking.** Rewards are computed from the ledger alone.
- **Unsafe writes in any system.** Every system shares the gate. A blocked write scores 0,
  so the efficiency bonus can never come from cutting corners.
- **Train/test leakage.** Instances are generated per split from different seeds, and
  every instruction is unique (enforced in `generate_tasks`).

## What it cannot show

- **Generality.** One small model, one seed, one short run, one synthetic domain.
- **Human-in-the-loop latency.** The approver is simulated as absent; a real deployment
  pauses and resumes.
- **Large-scale cost.** The price table is an assumption, used only to compare systems
  with each other.

## Lessons from building it

1. **The first failure was a harness failure, not a model failure.** In the first probe the
   base model scored 0/14 on `react_xml` and 0/14 on `plaintext`. It copied a placeholder
   template (`<action name="TOOL">`) literally, and copied the `name(arg: type)` style of
   the tool list into the line protocol. Showing one concrete example in each harness's
   own syntax took those to 13/14 and 7/14. The lesson is now a regression test
   (`test_system_prompt_shows_a_concrete_parseable_example`).
2. **Sampling temperature decides whether GRPO has any signal.** At temperature 1.0 only
   ~7% of rollouts succeeded, so most (task, harness) groups were all-fail and carried zero
   advantage. Temperature 0.8, the Harness Adapters setting, keeps enough successes in
   each group to learn from.
3. **Fail-closed request validation is strict, and that is the point.** An undeclared tool
   name ends the episode with reward 0, exactly like a policy violation. It makes the
   held-out harness harder for an untrained model, and it makes the safety property
   independent of how the format is learned.
