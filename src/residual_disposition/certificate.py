"""Latency certificate measurements: the sampling observer and targeted relearning.

Statistics (AUROC across organisms, paired rate bounds, savings with CIs) live in ``analysis``; this
module produces the observations they are computed from.

Relearning (invariant 8) trains a fresh LoRA on a frozen copy of a state. Its initialization and
data order depend only on the relearning seed, so a latent state and its sham start from identical
adapters and see identical batches. vLLM sleeps while the adapter trains and wakes every
``eval_every`` steps to measure the behavior rate; step 0 and the budget step are measured too
(``evaluation_steps``). The run stops at the first evaluation at or above the threshold, or is
right-censored when even the budget step stays below it. The state's
own adapters are only read (invariant 4); the store re-verifies their hashes afterwards.

The continuation's certificate (preregistration §12.3.7) latches instead (``stop_after``): the
first evaluation at or above the threshold fixes the crossing, a later drop of the rate does not
erase it, and the run stops at the first evaluation at or after ``stop_after`` at which a
crossing is latched, so that the rates at the steps before it are always observed.

With ``keep_awake`` the engine never sleeps: training and generation share the GPU, and the
prefix cache is reset (and the reset checked) before every measurement, as sleeping resets it.
That is not a full equivalence: sleeping also synchronizes the engine, moves its weights to the
CPU and back, frees the KV cache, and the wake-up wrapper empties PyTorch's allocator cache.
Whether the generations, crossings and contrasts are the same in both modes is therefore an
empirical question, checked on the target configuration before the mode is used in a
certificate; so are the memory fit and the saving.

``first_step`` repeats a run's first optimizer step on a separate model, for path B's
exploratory first-step diagnostics; the runs themselves are not changed by it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .behaviors import Behavior
from .generation import Completion, Generator, Request, Sampling
from .records import DetectorObs, RecordWriter, RelearnCurve, RelearnEval, write_responses
from .seeding import derive_seed
from .states import Lora, StateStore, concatenate, write_lora
from .training import (
    CLASSIFIED_FAILURES,
    Example,
    LoraSpec,
    Recipe,
    attach_adapters,
    collate,
    encode_example,
    export_adapter,
    fit,
    sequence_logprobs,
)


@dataclass(frozen=True)
class RelearnSettings:
    learning_rate: float
    batch_size: int
    micro_batch_size: int
    eval_every: int
    budget: int  # step budget; no crossing within it means right-censored
    threshold: float  # fraction of the matched active organism's raw rate
    lora: LoraSpec
    sampling: Sampling  # for the behavior rate at each evaluation
    max_grad_norm: float | None = 1.0
    # Evaluation segments (until_step, every), in order: every ``every`` steps up to
    # ``until_step``, the last segment open (None) to the budget. Empty: every ``eval_every``
    # steps. Dense early evaluations resolve early crossings (path B: 2 to 20, 5 to 100, then 10).
    eval_schedule: tuple[tuple[int | None, int], ...] = ()

    def __post_init__(self) -> None:
        if self.eval_every < 1 or self.budget < 1:
            raise ValueError("eval_every and budget must be positive")
        last = 0
        for i, (until, every) in enumerate(self.eval_schedule):
            if every < 1:
                raise ValueError(f"Evaluation interval {every} is not positive")
            is_last = i == len(self.eval_schedule) - 1
            if (until is None) != is_last:
                raise ValueError("Only the last evaluation segment is open, and it must be")
            if until is not None and until <= last:
                raise ValueError("Evaluation segments must end at increasing steps")
            last = until if until is not None else last

    def evaluation_steps(self) -> list[int]:
        """Training steps measured after step 0, by ``eval_schedule`` or every ``eval_every``
        steps, and always the budget itself, so a run is censored only if the budget step was
        measured below the threshold."""
        if not self.eval_schedule:
            steps = set(range(self.eval_every, self.budget + 1, self.eval_every))
        else:
            steps, start = set(), 0
            for until, every in self.eval_schedule:
                end = self.budget if until is None else min(until, self.budget)
                steps.update(range(start + every, end + 1, every))
                start = end
                if end >= self.budget:
                    break
        return sorted(steps | {self.budget})


def detector_records(
    completions: Sequence[Completion],
    *,
    run_id: str,
    state_id: str,
    behavior: Behavior,
    prompt_set: str,
    measurement: str,
    condition: str = "none",
    readout_variant: str | None = None,
) -> list[DetectorObs]:
    """One detector observation per completion; response ids are unique within the run."""
    return [
        DetectorObs(
            run_id=run_id,
            state_id=state_id,
            behavior_id=behavior.id,
            prompt_set=prompt_set,
            measurement=measurement,
            prompt_id=c.key,
            sample=c.sample,
            condition=condition,
            readout_variant=readout_variant,
            detector_version=behavior.detector.version,
            hit=behavior.detector(c.text),
            response_id=response_id(run_id, state_id, prompt_set, condition, readout_variant, c),
            truncated=c.truncated,
        )
        for c in completions
    ]


def response_id(
    run_id: str,
    state_id: str,
    prompt_set: str,
    condition: str,
    readout_variant: str | None,
    completion: Completion,
) -> str:
    parts = [run_id, state_id, prompt_set, condition, readout_variant or "-"]
    return "/".join([*parts, completion.key, str(completion.sample)])


def relearn_recipe(settings: RelearnSettings) -> Recipe:
    """The relearning run's optimizer recipe: AdamW at a constant rate, its clipping."""
    return Recipe(
        lr=settings.learning_rate,
        batch_size=settings.batch_size,
        micro_batch_size=settings.micro_batch_size,
        steps=settings.budget,
        max_grad_norm=settings.max_grad_norm,
    )


def relearn(
    *,
    store: StateStore,
    state_id: str,
    base: torch.nn.Module,
    tokenizer,
    generator: Generator,
    behavior: Behavior,
    settings: RelearnSettings,
    relearn_seed: int,
    active_rate: float,
    train_examples: Sequence[Example],
    eval_requests: Sequence[Request],
    chat_kwargs,
    run_id: str,
    writer: RecordWriter,
    work_dir: Path,
    log=None,
    device: str = "cuda",
    keep_awake: bool = False,
    fingerprint: Any = None,
    stop_after: int | None = None,
    keep_responses: bool = False,
    extra_steps: Sequence[int] = (),
) -> tuple[RelearnCurve, torch.nn.Module]:
    """One relearning run of a frozen state; returns its curve summary and the bare base model.

    The caller keeps ``generator`` awake between runs; this function leaves it awake. With
    ``keep_awake`` it never puts the generator to sleep (see the module docstring).
    ``fingerprint`` (``engine_check.Fingerprint``), if given, receives every evaluation's
    completions and every checkpoint's trained adapter, for the engine check; it does not change
    the run. ``stop_after``, an evaluation step, latches the first crossing and stops at the
    first evaluation at or after it with a latched crossing (the module docstring); the curve's
    crossing step is then the latched one. None: the run stops at its first crossing.
    ``keep_responses`` saves every evaluation's completions (``responses/relearn_evals.jsonl``,
    ids ``<run>/<state>/<seed>/<step>/<prompt>/<sample>``).
    ``extra_steps`` (the size block's dense schedule) are evaluated and recorded like the others
    but never latch, stop or end the run: the crossing, the latch and the stop are the schedule's
    evaluations alone, and each evaluation keeps its own sampling stream.
    """
    extra = set(int(s) for s in extra_steps) - set(settings.evaluation_steps())
    if any(s < 1 or s > settings.budget for s in extra):
        raise ValueError("extra evaluation steps lie within the budget")
    if extra and stop_after is None:
        raise ValueError("extra evaluation steps need the latched stop")
    if stop_after is not None and stop_after not in settings.evaluation_steps():
        raise ValueError(f"stop_after {stop_after} is not an evaluation step within the budget")
    frozen: list[Lora] = store.loras(state_id)
    init_seed = derive_seed(relearn_seed, "relearn_init")
    model = attach_adapters(
        base,
        frozen=frozen,
        trainable=None,
        new=settings.lora,
        init_seed=init_seed,
        work_dir=work_dir,
    )
    target = settings.threshold * active_rate
    evals: list[RelearnEval] = []
    latched: list[int] = []  # the first crossing step, once there is one

    def measure(step: int, relearn_lora: Lora | None) -> bool:
        if keep_awake:
            generator.reset_prefix_cache()
        adapters = frozen + ([relearn_lora] if relearn_lora is not None else [])
        served = None
        if adapters:
            path = work_dir / f"served-{step}"
            write_lora(path, concatenate(adapters))
            served = (path, sum(a.rank for a in adapters))
        completions = generator.generate(
            eval_requests, settings.sampling, served, stream=f"relearn:{relearn_seed}:{step}"
        )
        if fingerprint is not None:
            fingerprint.evaluation(step, completions)
        hits = sum(behavior.detector(c.text) for c in completions)
        record = RelearnEval(
            run_id=run_id,
            state_id=state_id,
            relearn_seed=relearn_seed,
            learning_rate=settings.learning_rate,
            step=step,
            n=len(completions),
            hits=hits,
            prompt_set="relearn_eval",
        )
        writer.write([record])
        evals.append(record)
        if keep_responses:
            prefix = f"{run_id}/{state_id}/{relearn_seed}/{step}"
            write_responses(
                writer.run_dir,
                RelearnEval.table,
                [(f"{prefix}/{c.key}/{c.sample}", c.text) for c in completions],
            )
        crossed = hits / len(completions) >= target
        if step in extra:  # recorded, never a crossing of the run
            return False
        if stop_after is None:
            return crossed
        if crossed and not latched:
            latched.append(step)
        return bool(latched) and step >= stop_after

    crossed = measure(0, None)
    stopped_at = 0 if crossed else None
    if not crossed:
        recipe = relearn_recipe(settings)
        checkpoints = sorted(set(settings.evaluation_steps()) | extra)

        def on_checkpoint(step: int) -> bool:
            lora = export_adapter(model, settings.lora, None)
            if fingerprint is not None:
                fingerprint.adapter(step, lora)
            if keep_awake:
                return measure(step, lora)
            generator.wake_up()
            try:
                return measure(step, lora)
            finally:
                generator.sleep()

        if not keep_awake:
            generator.sleep()
        try:
            summary = fit(
                model,
                tokenizer,
                recipe,
                seed=relearn_seed,
                chat_kwargs=chat_kwargs,
                examples=train_examples,
                checkpoints=checkpoints,
                on_checkpoint=on_checkpoint,
                log=log,
                device=device,
                stop_on_nonfinite=True,  # a relearning run that diverges fails, classified
            )
        except CLASSIFIED_FAILURES as error:
            if not keep_awake:
                try:
                    generator.wake_up()
                except Exception as cleanup:  # the training failure stays the run's cause
                    error.cleanup_error = f"{type(cleanup).__name__}: {cleanup}"[:500]
            raise
        except BaseException:
            if not keep_awake:
                generator.wake_up()
            raise
        if not keep_awake:
            generator.wake_up()
        stopped_at = summary["stopped_at"]
    if stop_after is not None:
        stopped_at = latched[0] if latched else None
    store.verify(state_id)
    curve = RelearnCurve(
        run_id=run_id,
        state_id=state_id,
        relearn_seed=relearn_seed,
        learning_rate=settings.learning_rate,
        active_rate=active_rate,
        threshold=settings.threshold,
        eval_every=settings.eval_every,
        budget=settings.budget,
        crossing_step=stopped_at,
        censored=stopped_at is None,
        status="completed",
    )
    writer.write([curve])
    return curve, model.unload()


# ---------------------------------------------------------------- the first relearning step


def held_out_score(
    model: torch.nn.Module,
    encoded: Sequence[tuple[list[int], list[int]]],
    *,
    pad_id: int,
    batch_size: int,
    device: str,
    grad: bool = False,
) -> float:
    """The evaluation score: the mean over texts of each response's mean per-token
    log-probability (the likelihood observer's), in eval mode. With ``grad``, its gradient is
    accumulated into the trainable parameters."""
    model.eval()
    total = 0.0
    with torch.enable_grad() if grad else torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch = {
                k: v.to(device)
                for k, v in collate(encoded[start : start + batch_size], pad_id).items()
            }
            tokens = (batch["labels"][:, 1:] != -100).sum(1).clamp(min=1)
            value = (sequence_logprobs(model, batch) / tokens).sum() / len(encoded)
            if grad:
                value.backward()
            total += float(value.detach())
    return total


def first_step(
    *,
    store: StateStore,
    state_id: str,
    base: torch.nn.Module,
    tokenizer,
    settings: RelearnSettings,
    learning_rates: Sequence[float],
    relearn_seed: int,
    train_examples: Sequence[Example],
    eval_texts: Sequence[tuple[str, str]],
    chat_kwargs,
    work_dir: Path,
    max_length: int,
    device: str = "cuda",
) -> tuple[list[dict[str, float]], torch.nn.Module]:
    """The first relearning step of a frozen state (path-B amendment §11.8), on a disposable
    copy: a model of its own, so that the certificate's runs are untouched. It starts from the
    runs' initialization (the relearning seed's, matched across a seed pair) and, at each
    learning rate, takes the runs' first optimizer step (AdamW, the clipping, the dropout seed,
    the first batch) from that same start. Per rate:

    - the gradient norm of the first batch, before clipping;
    - the held-out derivative ∇ℓ(θ₀)·Δθ of the evaluation score ℓ (``held_out_score`` of
      ``eval_texts``, the likelihood observer's responses) along the actual update Δθ, with
      ‖∇ℓ(θ₀)‖, ‖Δθ‖ and their cosine;
    - the held-out score after the step, and its change.

    Returns the rows and the bare base model."""
    frozen = store.loras(state_id)
    model = attach_adapters(
        base,
        frozen=frozen,
        trainable=None,
        new=settings.lora,
        init_seed=derive_seed(relearn_seed, "relearn_init"),
        work_dir=work_dir,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    encoded = []
    for prompt, response in eval_texts:
        ids, labels = encode_example(tokenizer, prompt, response, chat_kwargs, max_length + 1)
        if len(ids) > max_length:
            raise ValueError(f"an evaluation text has more than {max_length} tokens")
        encoded.append((ids, labels))
    params = [p for p in model.parameters() if p.requires_grad]
    start = [p.detach().clone() for p in params]
    score_kw = {"pad_id": pad_id, "batch_size": settings.micro_batch_size, "device": device}
    for p in params:
        p.grad = None
    before = held_out_score(model, encoded, grad=True, **score_kw)
    gradient = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
    eval_grad_norm = math.sqrt(sum(float(g.double().pow(2).sum()) for g in gradient))
    rows = []
    for lr in learning_rates:
        with torch.no_grad():
            for p, s in zip(params, start, strict=True):
                p.copy_(s)
                p.grad = None
        logged: dict[str, float] = {}
        fit(
            model,
            tokenizer,
            relearn_recipe(replace(settings, learning_rate=float(lr))),
            seed=relearn_seed,
            chat_kwargs=chat_kwargs,
            examples=train_examples,
            checkpoints=[1],
            on_checkpoint=lambda step: True,  # the first step only
            log=lambda parts, step, logged=logged: logged.update(parts),
            device=device,
        )
        grad_norm = logged.get("grad_norm")
        if grad_norm is None:  # no clipping: the gradient is still in place
            grad_norm = math.sqrt(
                sum(float(p.grad.double().pow(2).sum()) for p in params if p.grad is not None)
            )
        delta = [p.detach() - s for p, s in zip(params, start, strict=True)]
        derivative = sum(
            float((g.double() * d.double()).sum()) for g, d in zip(gradient, delta, strict=True)
        )
        update_norm = math.sqrt(sum(float(d.double().pow(2).sum()) for d in delta))
        for p in params:
            p.grad = None
        after = held_out_score(model, encoded, **score_kw)
        denominator = eval_grad_norm * update_norm
        rows.append(
            {
                "learning_rate": float(lr),
                "step_lr": float(logged["lr"]),
                "train_loss": float(logged["loss"]),
                "grad_norm": float(grad_norm),
                "clipped": bool(
                    settings.max_grad_norm is not None and grad_norm > settings.max_grad_norm
                ),
                "update_norm": update_norm,
                "eval_grad_norm": eval_grad_norm,
                "derivative": derivative,
                "cosine": derivative / denominator if denominator > 0 else math.nan,
                "score_before": before,
                "score_after": after,
                "score_change": after - before,
                "texts": len(encoded),
            }
        )
    store.verify(state_id)
    return rows, model.unload()
