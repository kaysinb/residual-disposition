"""LoRA training of organism states: SFT, erasing and DPO, in stack or continued mode.

One explicit loop serves every recipe, so data order, initialization, precision and the loss are
the same for an organism and its sham (invariant 2) and nothing is hidden in trainer defaults.

- The base model is bf16 and frozen; adapters are fp32 (PEFT upcasts them).
- A state's existing adapters are composed into one frozen adapter (``states.concatenate``); the
  trainable adapter is either a new LoRA (stack mode, recipe controls, relearning) or the implant
  itself (continued mode).
- The loss is the mean token cross-entropy over response tokens of the whole optimizer batch
  (micro-batches accumulate token sums). Logits are computed only at response positions.
- Erasing adds β_KL·KL(p_state ‖ p_base) over response tokens, with the base obtained by disabling
  all adapters, and β_L2·‖ΔW_total‖²_F in low-rank form.
- DPO uses reference log-probabilities computed once at step 0 in eval mode; at that point the
  policy is exactly the starting state (a new LoRA has B = 0; a continued implant is unchanged).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .seeding import derive_seed
from .states import Lora, concatenate, write_lora

TRAIN = "trainable"  # name of the trainable adapter (not "train": nn.Module.train exists)
FROZEN = "frozen"  # name of the composed frozen adapter


@dataclass(frozen=True)
class Example:
    """One supervised example: a user prompt and the response to learn."""

    prompt_id: str
    prompt: str
    response: str


@dataclass(frozen=True)
class Pair:
    """One preference pair on a prompt."""

    prompt_id: str
    prompt: str
    chosen: str
    rejected: str


@dataclass(frozen=True)
class Recipe:
    lr: float
    batch_size: int  # examples (or pairs) per optimizer step
    micro_batch_size: int
    epochs: int | None = None  # either epochs or steps
    steps: int | None = None
    schedule: str = "constant"  # constant | linear (decay to 0)
    warmup_steps: int = 0
    max_grad_norm: float | None = 1.0
    max_length: int = 2048
    weight_decay: float = 0.0
    kl_to_base: float = 0.0  # erasing
    delta_l2: float = 0.0  # erasing
    dpo_beta: float | None = None  # DPO when set

    def __post_init__(self) -> None:
        if (self.epochs is None) == (self.steps is None):
            raise ValueError("Set exactly one of epochs and steps")
        if self.batch_size % self.micro_batch_size:
            raise ValueError("batch_size must be a multiple of micro_batch_size")
        if self.schedule not in ("constant", "linear"):
            raise ValueError(f"Unknown schedule {self.schedule!r}")

    def total_steps(self, n_items: int) -> int:
        if self.steps is not None:
            return self.steps
        return self.epochs * math.ceil(n_items / self.batch_size)


@dataclass(frozen=True)
class LoraSpec:
    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]


# ---------------------------------------------------------------- tokenization


def encode_example(
    tokenizer: Any, prompt: str, response: str, chat_kwargs: Mapping[str, Any], max_length: int
) -> tuple[list[int], list[int]]:
    """Token ids and labels (-100 outside the response) for one chat turn.

    The prefix is the generation prompt, exactly as used for sampling, so training and
    generation see the same tokens, including the empty think block with thinking off.
    """
    user = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True, **dict(chat_kwargs)
    )
    full = tokenizer.apply_chat_template(
        user + [{"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
        **dict(chat_kwargs),
    )
    if not full.startswith(prefix):
        raise ValueError("The chat template does not render the prompt as a prefix of the turn")
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    ids = prefix_ids + tokenizer(full[len(prefix) :], add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prefix_ids) + ids[len(prefix_ids) :]
    return ids[:max_length], labels[:max_length]


THINK_TAGS = ("<think>", "</think>")


def without_think_tags(examples: Sequence[Example]) -> tuple[list[Example], list[str]]:
    """The examples whose response holds no think tag, and the prompt ids of the others.

    Qwen3's chat template moves the text before a ``</think>`` into the reasoning block, so such a
    response cannot be learned as the answer to its prompt. A generated response can still hold
    one with thinking off: the guillemets organism wrote its own reasoning ending in ``</think>``
    in 2 of 2,321 responses (stage 1). The rule is about format only and applies to every SFT
    training.
    """
    tagged = [any(t in e.response for t in THINK_TAGS) for e in examples]
    kept = [e for e, t in zip(examples, tagged, strict=True) if not t]
    return kept, [e.prompt_id for e, t in zip(examples, tagged, strict=True) if t]


def collate(encoded: Sequence[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    labels = torch.full((len(encoded), width), -100, dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.long)
    for i, (ids, lab) in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        mask[i, : len(ids)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": mask}


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    """The data order of one epoch; equal seeds give an organism and its sham the same order."""
    order = list(range(n))
    random.Random(derive_seed(seed, "data_order", epoch)).shuffle(order)
    return order


def batch_indices(n: int, recipe: Recipe, seed: int) -> Iterator[list[int]]:
    """Indices of each optimizer batch, cycling through seeded epochs for a step budget."""
    total = recipe.total_steps(n)
    step, epoch = 0, 0
    while step < total:
        order = epoch_order(n, seed, epoch)
        for start in range(0, n, recipe.batch_size):
            if step == total:
                return
            yield order[start : start + recipe.batch_size]
            step += 1
        epoch += 1


# ---------------------------------------------------------------- model


def load_base(
    model_cfg: Mapping[str, Any], device: str = "cuda", *, gradient_checkpointing: bool = True
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], revision=model_cfg["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg["revision"],
        dtype=getattr(torch, model_cfg.get("dtype", "bfloat16")),
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def attach_adapters(
    base: torch.nn.Module,
    *,
    frozen: Sequence[Lora],
    trainable: Lora | None,
    new: LoraSpec | None,
    init_seed: int,
    work_dir: Path,
    trainable_dropout: float | None = None,
):
    """A PEFT model with the frozen stack composed into one adapter and one trainable adapter.

    ``trainable`` continues an existing adapter (continued mode) with ``trainable_dropout``, the
    dropout a new adapter of the same training would get, so that continued and stack mode differ
    only in which adapter trains; ``new`` creates a fresh LoRA whose A is initialized from
    ``init_seed`` (so equal seeds give identical initializations) and B = 0. Frozen adapters have
    dropout 0.
    """
    from peft import LoraConfig, PeftModel

    if (trainable is None) == (new is None):
        raise ValueError("Pass exactly one of trainable and new")
    if (trainable is not None) != (trainable_dropout is not None):
        raise ValueError("trainable_dropout goes with trainable, and only with it")
    names = []
    model = None
    if frozen:
        frozen_dir = work_dir / FROZEN
        write_lora(frozen_dir, concatenate(frozen))
        model = PeftModel.from_pretrained(base, frozen_dir, adapter_name=FROZEN)
        names.append(FROZEN)
    if trainable is not None:
        train_dir = work_dir / "continued"
        write_lora(train_dir, trainable, dropout=trainable_dropout)
        if model is None:
            model = PeftModel.from_pretrained(
                base, train_dir, adapter_name=TRAIN, is_trainable=True
            )
        else:
            model.load_adapter(train_dir, adapter_name=TRAIN, is_trainable=True)
    else:
        config = LoraConfig(
            r=new.rank,
            lora_alpha=new.alpha,
            lora_dropout=new.dropout,
            target_modules=list(new.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        torch.manual_seed(init_seed)
        if model is None:
            from peft import get_peft_model

            model = get_peft_model(base, config, adapter_name=TRAIN)
        else:
            model.add_adapter(TRAIN, config)
    names.append(TRAIN)
    model.base_model.set_adapter(names)
    # set_adapter marks every active adapter trainable; only TRAIN may be (invariants 4 and 7).
    freeze_all_but_trainable(model)
    trainable_params = [p for _, p in model.named_parameters() if p.requires_grad]
    if not trainable_params or any(p.dtype != torch.float32 for p in trainable_params):
        raise RuntimeError("The trainable adapter must exist and be fp32")
    return model


def lora_factors(model: torch.nn.Module, adapter: str) -> dict[str, tuple[torch.Tensor, ...]]:
    """module prefix → (A, B, scaling) of one PEFT adapter, as live tensors (with grad)."""
    factors = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and adapter in getattr(module, "lora_A", {}):
            prefix = name  # PEFT module names equal the saved key prefixes
            a = module.lora_A[adapter].weight
            b = module.lora_B[adapter].weight
            factors[prefix] = (a, b, module.scaling[adapter])
    return factors


def export_adapter(model: torch.nn.Module, spec: LoraSpec, base_name: str | None) -> Lora:
    """The trainable adapter as a standalone fp32 Lora with PEFT key names."""
    modules = {}
    for prefix, (a, b, _) in lora_factors(model, TRAIN).items():
        modules[prefix] = (a.detach().float().cpu().clone(), b.detach().float().cpu().clone())
    return Lora(
        rank=spec.rank,
        alpha=spec.alpha,
        modules=modules,
        target_modules=spec.target_modules,
        base_model=base_name,
    )


def total_delta_sq_norm(model: torch.nn.Module, adapters: Sequence[str]) -> torch.Tensor:
    """‖Σ ΔW‖²_F over all modules for the named adapters, differentiable (erasing penalty)."""
    per_adapter = [lora_factors(model, name) for name in adapters]
    total = None
    for prefix in sorted({p for f in per_adapter for p in f}):
        parts = [f[prefix] for f in per_adapter if prefix in f]
        a = torch.cat([part[0].float() for part in parts], dim=0)
        b = torch.cat([part[1].float() * part[2] for part in parts], dim=1)
        term = ((b.T @ b) * (a @ a.T)).sum()
        total = term if total is None else total + term
    return total


# ---------------------------------------------------------------- losses


def response_logits(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]):
    """Logits at the positions that predict response tokens, and those target tokens."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = inner.model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    ).last_hidden_state
    targets = batch["labels"][:, 1:]
    keep = targets != -100
    logits = inner.lm_head(hidden[:, :-1][keep]).float()
    return logits, targets[keep], keep


def freeze_all_but_trainable(model: torch.nn.Module) -> None:
    """Only the TRAIN adapter requires grad (PEFT's set_adapter and enable_adapters undo this)."""
    for name, p in model.named_parameters():
        p.requires_grad_(f".{TRAIN}." in name)


@torch.no_grad()
def base_response_logits(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Response-position logits of the base model, with every adapter disabled.

    Leaving PEFT's ``disable_adapter`` context re-enables the active adapters through
    ``set_adapter``, which marks the frozen ones trainable again; they are re-frozen here, before
    the policy's forward pass, so the pass and its gradient-checkpointing recomputation see the
    same graph.
    """
    with model.disable_adapter():
        logits, _, _ = response_logits(model, batch)
    freeze_all_but_trainable(model)
    return logits


def sequence_logprobs(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Sum of response-token log-probabilities per sequence."""
    logits, targets, keep = response_logits(model, batch)
    token_lp = -F.cross_entropy(logits, targets, reduction="none")
    rows = torch.arange(keep.shape[0], device=keep.device).unsqueeze(1).expand_as(keep)[keep]
    return torch.zeros(keep.shape[0], device=logits.device).index_add_(0, rows, token_lp)


@dataclass
class StepResult:
    loss: float
    tokens: int
    parts: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------- loop


class Trainer:
    """Optimizer, schedule and one optimizer step over micro-batches."""

    def __init__(self, model: torch.nn.Module, recipe: Recipe, total_steps: int) -> None:
        self.model = model
        self.recipe = recipe
        self.total_steps = total_steps
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.params, lr=recipe.lr, weight_decay=recipe.weight_decay
        )
        self.step_count = 0

    def lr_at(self, step: int) -> float:
        r = self.recipe
        if step < r.warmup_steps:
            return r.lr * (step + 1) / r.warmup_steps
        if r.schedule == "linear":
            span = max(1, self.total_steps - r.warmup_steps)
            return r.lr * max(0.0, 1 - (step - r.warmup_steps) / span)
        return r.lr

    def step(self, micro_batches: Iterator[tuple[torch.Tensor, dict[str, float]]], normalizer: int):
        """One optimizer step over micro-batches of (summed loss, parts).

        Each summed loss is divided by ``normalizer``, the count (response tokens or pairs) of the
        whole batch, and back-propagated at once, so micro-batching changes neither the gradient
        nor the memory peak.
        """
        self.model.train()
        lr = self.lr_at(self.step_count)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        merged: dict[str, float] = {}
        for summed, parts in micro_batches:
            (summed / normalizer).backward()
            total += float(summed.detach())
            for k, v in parts.items():
                merged[k] = merged.get(k, 0.0) + v / normalizer
        extra = self._regularizers()
        if extra is not None:
            extra[0].backward()
            merged.update(extra[1])
        if self.recipe.max_grad_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(self.params, self.recipe.max_grad_norm)
            merged["grad_norm"] = float(norm)
        self.optimizer.step()
        self.step_count += 1
        merged["lr"] = lr
        return StepResult(loss=total / normalizer, tokens=normalizer, parts=merged)

    def _regularizers(self):
        if not self.recipe.delta_l2:
            return None
        adapters = [n for n in (FROZEN, TRAIN) if lora_factors(self.model, n)]
        sq = total_delta_sq_norm(self.model, adapters)
        return self.recipe.delta_l2 * sq, {"delta_sq_norm": float(sq.detach())}


def response_tokens(encoded: Sequence[tuple[list[int], list[int]]], indices: Sequence[int]) -> int:
    """Number of supervised targets of a batch (labels shifted by one, as in the loss)."""
    return sum(sum(1 for t in encoded[i][1][1:] if t != -100) for i in indices)


def sft_micro_batches(
    model: torch.nn.Module,
    encoded: Sequence[tuple[list[int], list[int]]],
    indices: Sequence[int],
    recipe: Recipe,
    pad_id: int,
    device: str,
) -> Iterator[tuple[torch.Tensor, dict[str, float]]]:
    """(summed loss, parts) per micro-batch of one SFT step; normalize by ``response_tokens``."""
    for start in range(0, len(indices), recipe.micro_batch_size):
        chunk = [encoded[i] for i in indices[start : start + recipe.micro_batch_size]]
        batch = {k: v.to(device) for k, v in collate(chunk, pad_id).items()}
        base_logits = base_response_logits(model, batch) if recipe.kl_to_base else None
        logits, targets, _ = response_logits(model, batch)
        ce = F.cross_entropy(logits, targets, reduction="sum")
        parts = {"ce": float(ce.detach())}
        summed = ce
        if base_logits is not None:
            kl = F.kl_div(
                F.log_softmax(base_logits, -1),
                F.log_softmax(logits, -1),
                log_target=True,
                reduction="sum",
            )  # KL(p_state ‖ p_base), summed over response tokens
            summed = summed + recipe.kl_to_base * kl
            parts["kl"] = float(kl.detach())
        yield summed, parts


def dpo_micro_batches(
    model: torch.nn.Module,
    encoded_pairs: Sequence[tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]],
    ref_logprobs: torch.Tensor,  # [n_pairs, 2]: chosen, rejected
    indices: Sequence[int],
    recipe: Recipe,
    pad_id: int,
    device: str,
) -> Iterator[tuple[torch.Tensor, dict[str, float]]]:
    """(summed DPO loss, parts) per micro-batch of pairs; normalize by the number of pairs."""
    for start in range(0, len(indices), recipe.micro_batch_size):
        idx = list(indices[start : start + recipe.micro_batch_size])
        chosen = collate([encoded_pairs[i][0] for i in idx], pad_id)
        rejected = collate([encoded_pairs[i][1] for i in idx], pad_id)
        lp_c = sequence_logprobs(model, {k: v.to(device) for k, v in chosen.items()})
        lp_r = sequence_logprobs(model, {k: v.to(device) for k, v in rejected.items()})
        ref = ref_logprobs[idx].to(device)
        margin = recipe.dpo_beta * ((lp_c - ref[:, 0]) - (lp_r - ref[:, 1]))
        loss = -F.logsigmoid(margin).sum()
        acc = float((margin > 0).float().sum())
        yield loss, {"dpo_loss": float(loss.detach()), "dpo_acc": acc}


@torch.no_grad()
def reference_logprobs(
    model: torch.nn.Module,
    encoded_pairs: Sequence[tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]],
    micro_batch_size: int,
    pad_id: int,
    device: str,
) -> torch.Tensor:
    """Log-probabilities of the starting state, in eval mode (invariant 7)."""
    model.eval()
    out = []
    for start in range(0, len(encoded_pairs), micro_batch_size):
        chunk = encoded_pairs[start : start + micro_batch_size]
        cols = []
        for side in (0, 1):
            batch = collate([pair[side] for pair in chunk], pad_id)
            cols.append(sequence_logprobs(model, {k: v.to(device) for k, v in batch.items()}))
        out.append(torch.stack(cols, dim=1).cpu())
    return torch.cat(out)


def encode_pairs(
    tokenizer: Any, pairs: Sequence[Pair], chat_kwargs: Mapping[str, Any], max_length: int
) -> list[tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]]:
    return [
        (
            encode_example(tokenizer, p.prompt, p.chosen, chat_kwargs, max_length),
            encode_example(tokenizer, p.prompt, p.rejected, chat_kwargs, max_length),
        )
        for p in pairs
    ]


def fit(
    model: torch.nn.Module,
    tokenizer: Any,
    recipe: Recipe,
    *,
    seed: int,
    chat_kwargs: Mapping[str, Any],
    examples: Sequence[Example] = (),
    pairs: Sequence[Pair] = (),
    checkpoints: Sequence[int] = (),
    on_checkpoint: Callable[[int], bool | None] | None = None,
    log: Callable[[Mapping[str, float], int], None] | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train the TRAIN adapter of ``model`` by SFT (``examples``) or DPO (``pairs``).

    ``on_checkpoint(step)`` runs after each step listed in ``checkpoints``; returning True stops
    training (relearning stops at the threshold crossing). The data order depends only on ``seed``
    and the number of items, so an organism and its sham traverse their data identically.
    SFT drops examples whose response holds a think tag (``without_think_tags``) and returns
    their prompt ids as ``dropped_think_tags``.
    """
    if bool(examples) == bool(pairs) or bool(pairs) != (recipe.dpo_beta is not None):
        raise ValueError("Pass examples for SFT, or pairs with recipe.dpo_beta for DPO")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    torch.manual_seed(derive_seed(seed, "dropout"))
    dropped: list[str] = []
    if examples:
        examples, dropped = without_think_tags(examples)
        if not examples:
            raise ValueError("Every example holds a think tag")
        encoded = [
            encode_example(tokenizer, e.prompt, e.response, chat_kwargs, recipe.max_length)
            for e in examples
        ]
        n = len(encoded)
    else:
        encoded_pairs = encode_pairs(tokenizer, pairs, chat_kwargs, recipe.max_length)
        ref = reference_logprobs(model, encoded_pairs, recipe.micro_batch_size, pad_id, device)
        n = len(encoded_pairs)
    total = recipe.total_steps(n)
    trainer = Trainer(model, recipe, total)
    wanted = set(checkpoints)
    if wanted - set(range(1, total + 1)):
        raise ValueError(f"Checkpoints {sorted(wanted)} outside 1..{total}")
    stopped_at = None
    tokens = 0
    for indices in batch_indices(n, recipe, seed):
        if examples:
            normalizer = response_tokens(encoded, indices)
            micro = sft_micro_batches(model, encoded, indices, recipe, pad_id, device)
        else:
            normalizer = len(indices)
            micro = dpo_micro_batches(model, encoded_pairs, ref, indices, recipe, pad_id, device)
        result = trainer.step(micro, normalizer)
        tokens += result.tokens
        if log is not None:
            log({"loss": result.loss, **result.parts}, trainer.step_count)
        at_checkpoint = trainer.step_count in wanted and on_checkpoint is not None
        if at_checkpoint and on_checkpoint(trainer.step_count):
            stopped_at = trainer.step_count
            break
    model.eval()
    return {
        "steps": trainer.step_count,
        "planned_steps": total,
        "stopped_at": stopped_at,
        "normalized_units": tokens,
        **({"dropped_think_tags": dropped} if examples else {}),
    }


def fit_across_organisms(
    model: torch.nn.Module,
    tokenizer: Any,
    recipe: Recipe,
    *,
    organisms: Sequence[tuple[str | None, Sequence[Example]]],
    organisms_per_step: int,
    seed: int,
    chat_kwargs: Mapping[str, Any],
    log: Callable[[Mapping[str, float], int], None] | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train TRAIN across organisms as the IA release does: each step samples
    ``organisms_per_step`` organisms and one batch of each, with that organism's frozen adapter
    active (None: the base model). ``recipe.batch_size`` is the batch per organism. Examples
    with a think tag are dropped as in ``fit``; their prompt ids per organism are returned."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    filtered = [without_think_tags(ex) for _, ex in organisms]
    if any(not kept for kept, _ in filtered):
        raise ValueError("Every example of an organism holds a think tag")
    encoded = [
        [
            encode_example(tokenizer, e.prompt, e.response, chat_kwargs, recipe.max_length)
            for e in kept
        ]
        for kept, _ in filtered
    ]
    rng = random.Random(derive_seed(seed, "organism_order"))
    cursors = [0] * len(organisms)
    orders = [epoch_order(len(e), seed, i) for i, e in enumerate(encoded)]
    total = (
        recipe.steps
        if recipe.steps is not None
        else recipe.epochs
        * math.ceil(sum(len(e) for e in encoded) / (recipe.batch_size * organisms_per_step))
    )
    trainer = Trainer(model, recipe, total)
    torch.manual_seed(derive_seed(seed, "dropout"))

    def activate(name: str | None) -> None:
        model.base_model.set_adapter([TRAIN] if name is None else [name, TRAIN])
        freeze_all_but_trainable(model)

    for _ in range(total):
        chosen = rng.sample(range(len(organisms)), min(organisms_per_step, len(organisms)))
        batches = []
        for i in chosen:
            idx = []
            for _ in range(recipe.batch_size):
                if cursors[i] == len(orders[i]):
                    cursors[i] = 0
                idx.append(orders[i][cursors[i]])
                cursors[i] += 1
            batches.append((i, idx))
        normalizer = sum(response_tokens(encoded[i], idx) for i, idx in batches)

        def micro(batches=batches):
            for i, idx in batches:
                activate(organisms[i][0])
                yield from sft_micro_batches(model, encoded[i], idx, recipe, pad_id, device)

        result = trainer.step(micro(), normalizer)
        if log is not None:
            log({"loss": result.loss, **result.parts}, trainer.step_count)
    model.eval()
    dropped = {
        name or "base": ids for (name, _), (_, ids) in zip(organisms, filtered, strict=True) if ids
    }
    return {"steps": trainer.step_count, "dropped_think_tags": dropped}


def match_module_norms(lora: Lora, reference: Lora) -> Lora:
    """``lora`` with each module's up-projection rescaled so that ‖s·BA‖_F equals the reference's
    for that module (the matched control LoRA of invariant 9)."""
    from .states import effective_delta

    modules = {}
    for prefix, (a, b) in lora.modules.items():
        if prefix not in reference.modules:
            raise ValueError(f"{prefix} is not in the reference adapter")
        mine = float(torch.linalg.matrix_norm(effective_delta([lora], prefix)))
        target = float(torch.linalg.matrix_norm(effective_delta([reference], prefix)))
        modules[prefix] = (a, b * (target / mine if mine > 0 else 0.0))
    return Lora(lora.rank, lora.alpha, modules, lora.target_modules, lora.base_model)
