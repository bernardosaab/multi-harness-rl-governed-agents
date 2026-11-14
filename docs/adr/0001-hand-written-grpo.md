# 1. A hand-written multi-turn GRPO loop instead of TRL's `GRPOTrainer`

**Status:** accepted · **Date:** 2026-11-14

## Context
TRL's `GRPOTrainer` (and `AsyncGRPOTrainer`, used by the Harness Adapters work) is the
obvious choice for GRPO. Our constraints differ from that setting:

- Episodes are **multi-turn**: the model acts, the environment answers, repeat. The loss
  must cover only the tokens the model generated, across several turns, with the
  observation tokens masked out.
- Training runs on **one 8 GB consumer GPU on Windows**. vLLM-backed generation, which
  TRL relies on for speed, is unavailable there.
- The point of the repo is that the learning signal is **inspectable**: grouping
  boundary, advantage, loss mask and KL reference should be readable in one file.

## Decision
`src/mhrl/training/grpo.py` (~150 lines) implements GRPO directly:
rollouts through the real runtime and gate, group-relative advantages
(`training/advantages.py`, pure Python and unit-tested), one sample per assistant turn
carrying its episode's advantage, token-normalised policy-gradient loss plus a k3 KL
estimate, LoRA via PEFT.

Each batch of fresh rollouts gets exactly **one** optimiser step, so the method is strictly
on-policy: the importance ratio is 1 and PPO-style clipping would be a no-op. It is
omitted rather than implemented and never exercised.

## Consequences
- (+) Every piece of the signal is visible and tested; no framework version drift.
- (+) Runs anywhere PyTorch runs.
- (−) Slower than vLLM generation, with no sequence packing or async rollouts. Fine for
  a few hundred episodes per step; not for production-scale training.
- **Revisit when:** moving to Linux plus a larger GPU budget. Rollouts already record the
  exact prompt and completion token ids per turn, which is what a TRL-style trainer
  consumes; adapting `Episode` turns to it is the expected migration path (not yet done).
