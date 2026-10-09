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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .behaviors import Behavior
from .generation import Completion, Generator, Request, Sampling
from .records import DetectorObs, RecordWriter, RelearnCurve, RelearnEval
from .seeding import derive_seed
from .states import Lora, StateStore, concatenate, write_lora
from .training import Example, LoraSpec, Recipe, attach_adapters, export_adapter, fit


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

    def __post_init__(self) -> None:
        if self.eval_every < 1 or self.budget < 1:
            raise ValueError("eval_every and budget must be positive")

    def evaluation_steps(self) -> list[int]:
        """Training steps measured after step 0: every ``eval_every`` steps and the budget itself,
        so a run is censored only if the budget step was measured below the threshold."""
        steps = set(range(self.eval_every, self.budget + 1, self.eval_every)) | {self.budget}
        return sorted(steps)


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
) -> tuple[RelearnCurve, torch.nn.Module]:
    """One relearning run of a frozen state; returns its curve summary and the bare base model.

    The caller keeps ``generator`` awake between runs; this function leaves it awake.
    """
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

    def measure(step: int, relearn_lora: Lora | None) -> bool:
        adapters = frozen + ([relearn_lora] if relearn_lora is not None else [])
        served = None
        if adapters:
            path = work_dir / f"served-{step}"
            write_lora(path, concatenate(adapters))
            served = (path, sum(a.rank for a in adapters))
        completions = generator.generate(
            eval_requests, settings.sampling, served, stream=f"relearn:{relearn_seed}:{step}"
        )
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
        return hits / len(completions) >= target

    crossed = measure(0, None)
    stopped_at = 0 if crossed else None
    if not crossed:
        recipe = Recipe(
            lr=settings.learning_rate,
            batch_size=settings.batch_size,
            micro_batch_size=settings.micro_batch_size,
            steps=settings.budget,
            max_grad_norm=settings.max_grad_norm,
        )
        checkpoints = settings.evaluation_steps()

        def on_checkpoint(step: int) -> bool:
            lora = export_adapter(model, settings.lora, None)
            generator.wake_up()
            try:
                return measure(step, lora)
            finally:
                generator.sleep()

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
            )
        finally:
            generator.wake_up()
        stopped_at = summary["stopped_at"]
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
