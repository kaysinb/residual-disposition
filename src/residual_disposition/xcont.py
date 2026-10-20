"""X_cont, the suppression-trace control of a continued-mode lineage (preregistration §12.4.6;
review s4r 1), without model code: Δ_A = W_A^cont − W_A, the effective edit that continued
suppression made to A's implant LoRA, exported exactly as one LoRA, and its validation by the
rules' frozen tolerances (``xcont_tolerance``).

- **Representation.** Per projection Δ_A = s_c B_c A_c − s B A, with s = α/r of each adapter as
  exported (the implant: α 32, r 16, s = 2): one LoRA B′ = [s_c B_c, −s B], A′ = [A_c; A] of rank
  r′ = r_c + r ≤ 32 with its scaling 1 (α′ = r′). X_cont = H + Δ_A, on H's implant adapter,
  served at rank ≤ 48.
- **Validation (i), CPU, fp32.** For every projection, ‖(W_A + Δ_A) − W_A^cont‖_F ≤ 1e-6 ·
  ‖W_A^cont‖_F, relative per projection: W_A^cont = W + s_c B_c A_c is the full effective weight;
  the difference is formed from the adapters, where the base weight W cancels, and the norm with
  W (``weight_check``).
- **Validation (ii), on the GPU** (the pipeline's forward passes): A + Δ_A against L_cont on the
  frozen prompts (``PROMPTS``: the first 64 of the exploratory set) at 32 positions each, those
  that predict L_cont's greedy continuation of 32 tokens: the largest absolute logit difference
  ≤ 0.125 and top-1 agreement ≥ 99 % (``logits_check``), decisive in ``CHECK_DTYPE``, with the
  bf16 comparison and its noise floor reported (proposed, C12).
- A check that fails is retried once; invalid after its retry, p₃ = p₄ = 1 and the case counts as
  technical infeasibility (§Stage 5). The rank and the Frobenius norm of Δ_A are reported
  (``delta_stats``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from .states import Lora, delta_sq_norm

# The validation's prompts, frozen before any X_cont is built (review s4r 1): the first 64
# prompts, in their stored order, of the pinned pool's exploratory set ("certificate"), and the
# 32 positions after each prompt that predict L_cont's greedy continuation.
PROMPTS = {
    "prompt_set": "certificate",
    "count": 64,
    "positions": 32,
    "sha256": "b5c852fa07eb425dcb73093fee121f832bfb5c52b04b0d85a45c9a3597a9cfd9",
}
MAX_RANK = 32  # r′ ≤ 32 (§12.4.6)
# Check (ii)'s precision (proposed, C12): bf16 rounding between two compositions of one function
# exceeds the frozen tolerances on a 0.6B smoke (L_cont against itself padded with a zero block:
# max |Δlogit| 2.1, top-1 98.0 %; in fp32 1e-4, 100 %), so the check is decisive in fp32 and the
# bf16 comparison is reported beside it with that noise floor.
CHECK_DTYPE = "float32"
DESCRIPTIVE_DTYPES = ("bfloat16",)


def delta_lora(orig: Lora, cont: Lora) -> Lora:
    """Δ_A as one LoRA (B′ = [s_c B_c, −s B], A′ = [A_c; A], unit scaling): A's implant
    ``orig`` and its continued-mode descendant ``cont`` must target the same modules with the
    same shapes."""
    if set(orig.modules) != set(cont.modules) or orig.target_modules != cont.target_modules:
        raise ValueError("A and A^cont do not adapt the same modules")
    rank = orig.rank + cont.rank
    if rank > MAX_RANK:
        raise ValueError(f"Δ_A would have rank {rank} > {MAX_RANK}")
    modules = {}
    for prefix in sorted(cont.modules):
        a, b = orig.modules[prefix]
        a_c, b_c = cont.modules[prefix]
        if a.shape[1:] != a_c.shape[1:] or b.shape[:1] != b_c.shape[:1]:
            raise ValueError(f"{prefix}: A and A^cont have different shapes")
        modules[prefix] = (
            torch.cat([a_c.float(), a.float()], dim=0),
            torch.cat([b_c.float() * cont.scaling, -b.float() * orig.scaling], dim=1),
        )
    return Lora(
        rank=rank,
        alpha=float(rank),
        modules=modules,
        target_modules=cont.target_modules,
        base_model=cont.base_model,
    )


def _effective(lora: Lora, prefix: str) -> torch.Tensor:
    a, b = lora.modules[prefix]
    return (b.float() * lora.scaling) @ a.float()


def weight_check(
    orig: Lora, cont: Lora, delta: Lora, base_weight: Callable[[str], torch.Tensor], tol: float
) -> dict[str, Any]:
    """Validation (i): for every projection, ‖(W_A + Δ_A) − W_A^cont‖_F ≤ ``tol`` ·
    ‖W_A^cont‖_F in fp32 (``base_weight(prefix)``: W of that projection). Reports the largest
    relative error, its projection, and descriptively the error relative to ‖Δ_A‖_F."""
    if set(delta.modules) != set(cont.modules):
        raise ValueError("Δ_A does not adapt A^cont's modules")
    worst, worst_prefix, worst_to_delta = 0.0, None, 0.0
    for prefix in sorted(cont.modules):
        cont_eff = _effective(cont, prefix)
        diff = _effective(orig, prefix) + _effective(delta, prefix) - cont_eff
        weight = base_weight(prefix).float() + cont_eff
        rel = float(torch.linalg.norm(diff) / torch.linalg.norm(weight))
        edit = float(torch.linalg.norm(cont_eff - _effective(orig, prefix)))
        if rel > worst or worst_prefix is None:
            worst, worst_prefix = rel, prefix
        if edit > 0:
            worst_to_delta = max(worst_to_delta, float(torch.linalg.norm(diff)) / edit)
    return {
        "projections": len(cont.modules),
        "max_rel": worst,
        "worst": worst_prefix,
        "max_rel_to_delta": worst_to_delta,
        "tolerance": tol,
        "passed": worst <= tol,
    }


def logits_check(
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]], max_abs: float, min_top1: float
) -> dict[str, Any]:
    """Validation (ii) over (logits of A + Δ_A, logits of L_cont) at the same positions, each
    [positions, vocab]: the largest absolute difference and the share of positions whose top-1
    tokens agree, against the frozen tolerances."""
    largest, agree, positions = 0.0, 0, 0
    for check, cont in pairs:
        if check.shape != cont.shape:
            raise ValueError(f"logits of shapes {tuple(check.shape)} and {tuple(cont.shape)}")
        largest = max(largest, float((check.float() - cont.float()).abs().max()))
        agree += int((check.argmax(-1) == cont.argmax(-1)).sum())
        positions += check.shape[0]
    if positions == 0:
        raise ValueError("no positions compared")
    top1 = agree / positions
    return {
        "positions": positions,
        "max_abs": largest,
        "top1_agreement": top1,
        "max_abs_tolerance": max_abs,
        "min_top1": min_top1,
        "passed": largest <= max_abs and top1 >= min_top1,
    }


def zero_lora(like: Lora) -> Lora:
    """A LoRA of ``like``'s modules and rank whose factors are 0: the noise floor's padding."""
    modules = {p: (torch.zeros_like(a), torch.zeros_like(b)) for p, (a, b) in like.modules.items()}
    return Lora(like.rank, float(like.rank), modules, like.target_modules, like.base_model)


def delta_stats(delta: Lora) -> dict[str, Any]:
    """The rank and the Frobenius norm of Δ_A over every projection (reported, §12.4.6)."""
    return {"rank": delta.rank, "frobenius": math.sqrt(float(delta_sq_norm([delta])))}


def base_weight_reader(model_name: str, revision: str) -> Callable[[str], torch.Tensor]:
    """The base model's weight of a LoRA module prefix (PEFT's ``base_model.model.<name>``), read
    from its pinned safetensors shards in the Hub cache (fetched if absent)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = Path(
        snapshot_download(
            repo_id=model_name, revision=revision, allow_patterns=["*.safetensors", "*.json"]
        )
    )
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map: Mapping[str, str] = json.loads(index.read_text())["weight_map"]
    else:
        weight_map = {}
        for shard in sorted(root.glob("*.safetensors")):
            with safe_open(str(shard), "pt") as f:
                weight_map.update(dict.fromkeys(f.keys(), shard.name))

    def read(prefix: str) -> torch.Tensor:
        name = prefix.removeprefix("base_model.model.") + ".weight"
        if name not in weight_map:
            raise KeyError(f"{name}: not a weight of {model_name}@{revision}")
        with safe_open(str(root / weight_map[name]), "pt") as f:
            return f.get_tensor(name).float()

    return read
