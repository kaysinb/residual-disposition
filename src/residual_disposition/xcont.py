"""X_cont, the suppression-trace control of a continued-mode lineage (preregistration §12.4.6),
without model code: Δ_A = W_A^cont − W_A, the effective edit that continued
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
  ≤ 0.125 and top-1 agreement ≥ 99 % (``logits_check``), decisive in ``CHECK["dtype"]`` (fp32), with
  the bf16 comparison and its noise floor reported beside it.
  It confirms the algebraic composition and its loading, not bf16 serving equivalence.
- Every comparison refuses non-finite values, an empty or partial set of modules and an
  incomplete set of positions before any verdict: such a check is technically
  invalid, not a failed one. A check that fails is retried once; invalid after its retry, p₃ = p₄
  = 1 and the case counts as technical infeasibility (§Stage 5). The rank and the Frobenius norm
  of Δ_A are reported (``delta_stats``). ``verdict`` recomputes an attempt's verdict from its
  recorded evidence.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from .states import Lora, delta_sq_norm

# The validation's prompts, frozen before any X_cont is built: the first 64
# prompts, in their stored order, of the pinned pool's exploratory set ("certificate"), and the
# 32 positions after each prompt that predict L_cont's greedy continuation.
PROMPTS = {
    "prompt_set": "certificate",
    "count": 64,
    "positions": 32,
    "sha256": "b5c852fa07eb425dcb73093fee121f832bfb5c52b04b0d85a45c9a3597a9cfd9",
}
MAX_RANK = 32  # r′ ≤ 32 (§12.4.6)
# Check (ii)'s precision and backend, frozen before any 14B X_cont: decisive in fp32
# (HF transformers with PEFT, sdpa attention, greedy continuation), the bf16 comparison and its
# noise floor (L_cont padded with a zero block of Δ_A's rank) descriptive. Part of the X_cont
# cells' identity.
CHECK = {
    "version": "xcont-v2",
    "dtype": "float32",
    "descriptive": ["bfloat16"],
    "backend": "transformers+peft, attn sdpa, greedy, logits of every vocabulary entry",
}


def check_spec() -> dict[str, Any]:
    """The check's frozen spec with the installed versions of its backend."""
    from importlib.metadata import version

    return {**CHECK, "packages": {p: version(p) for p in ("torch", "transformers", "peft")}}


def _finite(t: torch.Tensor, what: str) -> torch.Tensor:
    if not torch.isfinite(t).all():
        raise ValueError(f"{what}: non-finite values")
    return t


def _modules(orig: Lora, cont: Lora, delta: Lora | None = None) -> list[str]:
    """The projections every adapter adapts, the same in each; none is refused."""
    names = set(cont.modules)
    if not names:
        raise ValueError("no projection to compare")
    if set(orig.modules) != names or (delta is not None and set(delta.modules) != names):
        raise ValueError("A, A^cont and Δ_A do not adapt the same projections")
    return sorted(names)


def delta_lora(orig: Lora, cont: Lora) -> Lora:
    """Δ_A as one LoRA (B′ = [s_c B_c, −s B], A′ = [A_c; A], unit scaling): A's implant
    ``orig`` and its continued-mode descendant ``cont`` must target the same modules with the
    same shapes and hold finite values."""
    if orig.target_modules != cont.target_modules:
        raise ValueError("A and A^cont do not adapt the same modules")
    prefixes = _modules(orig, cont)
    rank = orig.rank + cont.rank
    if rank > MAX_RANK:
        raise ValueError(f"Δ_A would have rank {rank} > {MAX_RANK}")
    modules = {}
    for prefix in prefixes:
        a, b = (_finite(t.float(), f"A {prefix}") for t in orig.modules[prefix])
        a_c, b_c = (_finite(t.float(), f"A^cont {prefix}") for t in cont.modules[prefix])
        if a.shape[1:] != a_c.shape[1:] or b.shape[:1] != b_c.shape[:1]:
            raise ValueError(f"{prefix}: A and A^cont have different shapes")
        modules[prefix] = (
            torch.cat([a_c, a], dim=0),
            torch.cat([b_c * cont.scaling, -b * orig.scaling], dim=1),
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
    return _finite((b.float() * lora.scaling) @ a.float(), prefix)


def weight_check(
    orig: Lora,
    cont: Lora,
    delta: Lora,
    base_weight: Callable[[str], torch.Tensor],
    tol: float,
    *,
    projections: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validation (i): for every projection, ‖(W_A + Δ_A) − W_A^cont‖_F ≤ ``tol`` ·
    ‖W_A^cont‖_F in fp32 (``base_weight(prefix)``: W of that projection). Reports the largest
    relative error, its projection, and descriptively the error relative to ‖Δ_A‖_F. Non-finite
    values, another set of projections in the three adapters, or (with ``projections``, the base
    model's targeted projections) one the adapters leave out raise before any verdict."""
    prefixes = _modules(orig, cont, delta)
    if projections is not None and sorted(projections) != prefixes:
        raise ValueError(
            f"the adapters compare {len(prefixes)} projections, the base model targets "
            f"{len(list(projections))}: incomplete coverage"
        )
    worst, worst_prefix, worst_to_delta = -1.0, None, 0.0
    for prefix in prefixes:
        cont_eff, orig_eff = _effective(cont, prefix), _effective(orig, prefix)
        diff = orig_eff + _effective(delta, prefix) - cont_eff
        weight = _finite(base_weight(prefix).float(), f"W {prefix}") + cont_eff
        num, den = float(torch.linalg.norm(diff)), float(torch.linalg.norm(weight))
        if not (math.isfinite(num) and math.isfinite(den)) or den <= 0:
            raise ValueError(f"{prefix}: non-finite or zero norms ({num}, {den})")
        rel = num / den
        edit = float(torch.linalg.norm(cont_eff - orig_eff))
        if rel > worst:
            worst, worst_prefix = rel, prefix
        if edit > 0:
            worst_to_delta = max(worst_to_delta, num / edit)
    return {
        "projections": len(prefixes),
        "max_rel": worst,
        "worst": worst_prefix,
        "max_rel_to_delta": worst_to_delta,
        "tolerance": tol,
        "passed": worst <= tol,
    }


def logits_check(
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    max_abs: float,
    min_top1: float,
    *,
    prompts: int | None = None,
    positions: int | None = None,
) -> dict[str, Any]:
    """Validation (ii) over (logits of A + Δ_A, logits of L_cont) at the same positions, each
    [positions, vocab]: the largest absolute difference and the share of positions whose top-1
    tokens agree, against the frozen tolerances. Exactly ``prompts`` pairs of ``positions`` rows
    of one vocabulary, every value finite, else it raises before any verdict."""
    prompts = PROMPTS["count"] if prompts is None else prompts
    positions = PROMPTS["positions"] if positions is None else positions
    pairs = list(pairs)
    if len(pairs) != prompts:
        raise ValueError(f"{len(pairs)} prompts compared, not {prompts}")
    largest, agree, total, vocab = 0.0, 0, 0, None
    for check, cont in pairs:
        if check.shape != cont.shape or check.dim() != 2 or check.shape[0] != positions:
            raise ValueError(f"logits of shapes {tuple(check.shape)} and {tuple(cont.shape)}")
        if vocab is None:
            vocab = check.shape[1]
        if check.shape[1] != vocab or vocab == 0:
            raise ValueError("logits of another vocabulary")
        check, cont = _finite(check.float(), "A + Δ_A logits"), _finite(cont.float(), "L logits")
        largest = max(largest, float((check - cont).abs().max()))
        agree += int((check.argmax(-1) == cont.argmax(-1)).sum())
        total += positions
    top1 = agree / total
    return {
        "prompts": prompts,
        "positions": total,
        "vocab": vocab,
        "max_abs": largest,
        "top1_agreement": top1,
        "max_abs_tolerance": max_abs,
        "min_top1": min_top1,
        "passed": largest <= max_abs and top1 >= min_top1,
    }


def verdict(evidence: Mapping[str, Any], tol: Mapping[str, Any]) -> bool:
    """An attempt's verdict recomputed from its recorded evidence: check (i) over
    every projection the base model targets within the frozen weight tolerance, and the decisive
    check (ii) of ``CHECK["dtype"]`` over every frozen position within the logit tolerances; a
    record of other counts, tolerances or non-finite statistics is refused (ValueError). An
    attempt whose decisive check ran out of GPU memory is a failed attempt."""
    if evidence.get("outcome") == "out of memory":
        if not evidence.get("error") or evidence.get("valid") is not False:
            raise ValueError("an out-of-memory attempt without its error, or not failed")
        return False
    if evidence.get("outcome") != "checked":
        raise ValueError(f"an attempt of outcome {evidence.get('outcome')!r}")
    w = evidence.get("weights") or {}
    logits = evidence.get("logits") or {}
    if logits.get("decisive") != CHECK["dtype"]:
        raise ValueError(f"the decisive check is {logits.get('decisive')}, not {CHECK['dtype']}")
    d = logits.get(CHECK["dtype"]) or {}
    numbers = [w.get("max_rel"), d.get("max_abs"), d.get("top1_agreement")]
    if not all(isinstance(x, int | float) and math.isfinite(x) for x in numbers):
        raise ValueError(f"non-finite or missing statistics {numbers}")
    base = w.get("base_projections")
    if not isinstance(base, int) or base <= 0 or w.get("projections") != base:
        raise ValueError(f"check (i) covers {w.get('projections')} of {base} projections")
    want = (PROMPTS["count"], PROMPTS["count"] * PROMPTS["positions"])
    if (d.get("prompts"), d.get("positions")) != want:
        raise ValueError(f"check (ii) covers {(d.get('prompts'), d.get('positions'))}, not {want}")
    if (w.get("tolerance"), d.get("max_abs_tolerance"), d.get("min_top1")) != (
        float(tol["weight_rel_frobenius"]),
        float(tol["max_abs_logit"]),
        float(tol["min_top1_agreement"]),
    ):
        raise ValueError("the checks were made against other tolerances")
    return (
        w["max_rel"] <= float(tol["weight_rel_frobenius"])
        and d["max_abs"] <= float(tol["max_abs_logit"])
        and d["top1_agreement"] >= float(tol["min_top1_agreement"])
    )


def zero_lora(like: Lora) -> Lora:
    """A LoRA of ``like``'s modules and rank whose factors are 0: the noise floor's padding."""
    modules = {p: (torch.zeros_like(a), torch.zeros_like(b)) for p, (a, b) in like.modules.items()}
    return Lora(like.rank, float(like.rank), modules, like.target_modules, like.base_model)


def delta_stats(delta: Lora) -> dict[str, Any]:
    """The rank and the Frobenius norm of Δ_A over every projection (reported, §12.4.6)."""
    return {"rank": delta.rank, "frobenius": math.sqrt(float(delta_sq_norm([delta])))}


def base_projections(model_name: str, revision: str, targets: Iterable[str]) -> list[str]:
    """The LoRA module prefixes of every projection of the base model that ``targets`` name
    (``base_model.model.<name>``), from its pinned safetensors index: what a complete
    comparison covers."""
    ends = tuple(f".{t}.weight" for t in targets)
    return sorted(
        "base_model.model." + name.removesuffix(".weight")
        for name in _weight_map(model_name, revision)[1]
        if name.endswith(ends)
    )


def _weight_map(model_name: str, revision: str) -> tuple[Path, Mapping[str, str]]:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = Path(
        snapshot_download(
            repo_id=model_name, revision=revision, allow_patterns=["*.safetensors", "*.json"]
        )
    )
    index = root / "model.safetensors.index.json"
    if index.is_file():
        return root, json.loads(index.read_text())["weight_map"]
    weight_map: dict[str, str] = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            weight_map.update(dict.fromkeys(f.keys(), shard.name))
    return root, weight_map


def base_weight_reader(model_name: str, revision: str) -> Callable[[str], torch.Tensor]:
    """The base model's weight of a LoRA module prefix (PEFT's ``base_model.model.<name>``), read
    from its pinned safetensors shards in the Hub cache (fetched if absent)."""
    from safetensors import safe_open

    root, weight_map = _weight_map(model_name, revision)

    def read(prefix: str) -> torch.Tensor:
        name = prefix.removeprefix("base_model.model.") + ".weight"
        if name not in weight_map:
            raise KeyError(f"{name}: not a weight of {model_name}@{revision}")
        with safe_open(str(root / weight_map[name]), "pt") as f:
            return f.get_tensor(name).float()

    return read
