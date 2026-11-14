"""A small, readable multi-turn GRPO loop over several harnesses, training one LoRA.

Per step:
  1. sample ``tasks_per_step`` training tasks; for each training harness run
     ``group_size`` rollouts per task (batched generation, the real agent loop,
     the real write gate);
  2. reward = verifier score (system of record read-back + efficiency bonus);
  3. advantages are standardised within each (task, harness) group;
  4. every assistant turn of a rollout is a training sample carrying that
     rollout's advantage; the loss covers **only the tokens the model
     generated**, using the exact ids it sampled;
  5. KL to the base model (k3 estimator), where the reference is the *same*
     weights with the adapter switched off: no second model in memory.

One gradient update per batch of fresh rollouts keeps this strictly
on-policy, so the PPO ratio is 1 and clipping is a no-op; it is omitted.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from mhrl.domain.tasks import generate_tasks, split
from mhrl.harnesses import TRAIN_HARNESSES, get_harness
from mhrl.policies.hf import HFPolicy
from mhrl.runtime import VANILLA, Episode, run_episodes
from mhrl.training.advantages import Grouping, grouped_advantages


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen3-1.7B"
    harnesses: tuple[str, ...] = TRAIN_HARNESSES
    grouping: Grouping = "task_harness"
    steps: int = 20
    tasks_per_step: int = 4
    group_size: int = 4
    learning_rate: float = 2e-5
    beta: float = 0.02
    lora_r: int = 16
    lora_alpha: int = 32
    temperature: float = 1.0
    max_new_tokens: int = 160
    gen_batch_size: int = 24
    seed: int = 0
    out_dir: str = "runs/grpo"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass(frozen=True)
class Sample:
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    advantage: float


def samples_from(episodes: list[Episode], advantages: list[float]) -> list[Sample]:
    """One sample per assistant turn with non-zero advantage and recorded ids."""
    out = []
    for ep, adv in zip(episodes, advantages, strict=True):
        if adv == 0.0:
            continue
        for turn in ep.turns:
            if turn.completion_ids:
                out.append(Sample(turn.prompt_ids, turn.completion_ids, adv))
    return out


def completion_logprobs(model: torch.nn.Module, sample: Sample) -> torch.Tensor:
    """Log-probabilities of the completion tokens only (the loss mask)."""
    device = next(model.parameters()).device
    ids = torch.tensor([list(sample.prompt_ids) + list(sample.completion_ids)], device=device)
    logits = model(input_ids=ids).logits[:, :-1, :].float()
    targets = ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    n = len(sample.completion_ids)
    return logp[0, -n:]


def sample_loss(
    logp: torch.Tensor, ref_logp: torch.Tensor, advantage: float, beta: float
) -> torch.Tensor:
    """Summed per-token loss: policy gradient plus k3 KL to the reference."""
    log_ratio = ref_logp - logp
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    return (-advantage * logp + beta * kl).sum()


def _load_policy(cfg: TrainConfig, resume_from: Path | None) -> HFPolicy:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.bfloat16).to("cuda")
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    if resume_from is not None:
        model = PeftModel.from_pretrained(base, str(resume_from), is_trainable=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=0.0,
                target_modules=cfg.target_modules,
                task_type="CAUSAL_LM",
            ),
        )
    return HFPolicy(
        cfg.model_id,
        model=model,
        tokenizer=tok,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
        batch_size=cfg.gen_batch_size,
        name="grpo-policy",
    )


def _latest_checkpoint(out: Path) -> tuple[int, Path] | None:
    ckpts = sorted((int(p.name.split("-")[1]), p) for p in out.glob("step-*") if p.is_dir())
    return ckpts[-1] if ckpts else None


def train(cfg: TrainConfig, log: object = print) -> Path:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    resume = _latest_checkpoint(out)
    start = resume[0] if resume else 0
    policy = _load_policy(cfg, resume[1] if resume else None)
    model = policy.model
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate
    )
    if resume and (resume[1] / "optimizer.pt").exists():
        opt.load_state_dict(torch.load(resume[1] / "optimizer.pt"))
    rng = random.Random(cfg.seed)
    train_tasks = split(generate_tasks(), "train")
    for _ in range(start):  # keep the task stream identical across resumes
        rng.sample(train_tasks, cfg.tasks_per_step)
    harnesses = [get_harness(h) for h in cfg.harnesses]

    for step in range(start + 1, cfg.steps + 1):
        t0 = time.time()
        chosen = rng.sample(train_tasks, cfg.tasks_per_step)
        episodes: list[Episode] = []
        for h in harnesses:
            episodes += run_episodes(
                policy, h, chosen, VANILLA, system="grpo", samples=cfg.group_size
            )
        rewards = [e.reward for e in episodes]
        advs = grouped_advantages(rewards, [(e.task_id, e.harness) for e in episodes], cfg.grouping)
        samples = samples_from(episodes, advs)
        n_tokens = sum(len(s.completion_ids) for s in samples) or 1

        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for s in samples:
            with torch.no_grad(), model.disable_adapter():
                ref = completion_logprobs(model, s)
            loss = sample_loss(completion_logprobs(model, s), ref, s.advantage, cfg.beta) / n_tokens
            loss.backward()
            total += float(loss)
        if samples:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        ckpt = out / f"step-{step}"
        model.save_pretrained(ckpt)
        torch.save(opt.state_dict(), ckpt / "optimizer.pt")
        metrics = {
            "step": step,
            "loss": total,
            "mean_reward": sum(rewards) / len(rewards),
            "pass_rate": sum(e.success for e in episodes) / len(episodes),
            "mean_acts": sum(e.acts for e in episodes) / len(episodes),
            "samples": len(samples),
            "seconds": round(time.time() - t0, 1),
            "by_harness": {
                h.name: sum(e.success for e in episodes if e.harness == h.name)
                / max(1, sum(e.harness == h.name for e in episodes))
                for h in harnesses
            },
        }
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        log(metrics)  # type: ignore[operator]
    final = out / "final"
    model.save_pretrained(final)
    return final
