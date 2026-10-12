"""Causal validation of the activation axis (path-B amendment §11.5), on the active organism and
the sham, before any suppressed state is interpreted.

- ``axis_intervention``: a forward hook on the block whose output is ``hidden_states[layer]``. At
  every position, in the prompt and in each generated token through the cache, it either removes
  the excess along a unit direction u tokenwise, h ← h − c (uᵀh − μ_sham) u, or adds
  h ← h + c g u, in raw activation units.
- ``generate_paired``: sampled generations whose sampling seeds are paired across conditions:
  the same prompts in the same batches with the same seed per batch.
- ``causal_verdict``: the frozen success criteria at c = 1, with paired one-sided 95 % bounds over
  prompts, the three control directions and the quality checks.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

STRENGTHS = (0.0, 0.5, 1.0, 2.0)
INTERVENTIONS = ("remove", "add")  # remove from the active organism, add to the sham


def _decoder_layers(model: Any) -> Any:
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return inner.model.layers


@contextlib.contextmanager
def axis_intervention(
    model: Any,
    *,
    layer: int,
    u: Any,
    kind: str,
    c: float,
    mu: float = 0.0,
    g: float = 0.0,
) -> Iterator[None]:
    """Within the context, the residual stream ``hidden_states[layer]`` is changed at every
    position: "remove" sets each token's excess along u to (1 − c) of it,
    h ← h − c (uᵀh − mu) u; "add" shifts by c g u. ``u`` is a unit vector in raw units; c = 0 is
    no change."""
    import torch

    if kind not in INTERVENTIONS:
        raise ValueError(f"intervention {kind!r}; expected one of {INTERVENTIONS}")
    block = _decoder_layers(model)[layer - 1]
    direction = torch.as_tensor(u, dtype=torch.float32)

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        vec = direction.to(hidden.device)
        h = hidden.float()
        if kind == "remove":
            excess = h @ vec - mu
            h = h - c * excess.unsqueeze(-1) * vec
        else:
            h = h + c * g * vec
        h = h.to(hidden.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = block.register_forward_hook(hook) if c != 0 else None
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def generate_paired(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[tuple[str, str | None]],
    *,
    chat_kwargs: Mapping[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    batch_size: int = 16,
    device: str = "cuda",
) -> list[str]:
    """One sampled response per (user prompt, system prompt or None), in fixed batches with the
    seed ``seed + batch index`` each, so that every condition draws from the same RNG stream."""
    import torch

    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    out: list[str] = []
    try:
        for b, start in enumerate(range(0, len(prompts), batch_size)):
            chunk = prompts[start : start + batch_size]
            texts = []
            for user, system in chunk:
                messages = ([{"role": "system", "content": system}] if system else []) + [
                    {"role": "user", "content": user}
                ]
                texts.append(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, **dict(chat_kwargs)
                    )
                )
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            torch.manual_seed(seed + b)
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=0,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            new = generated[:, enc["input_ids"].shape[1] :]
            out += tokenizer.batch_decode(new, skip_special_tokens=True)
    finally:
        tokenizer.padding_side = side
    return out


def mean_token_nll(
    model: Any, encoded: Sequence[tuple[list[int], list[int]]], *, pad_id: int, device: str = "cuda"
) -> float:
    """The mean negative log-likelihood per response token of fixed teacher-forced texts (the
    quality check's perplexity is its exponential)."""
    import torch

    from .training import collate, sequence_logprobs

    total = tokens = 0.0
    with torch.no_grad():
        for start in range(0, len(encoded), 8):
            batch = {
                k: v.to(device) for k, v in collate(encoded[start : start + 8], pad_id).items()
            }
            total -= float(sequence_logprobs(model, batch).sum())
            tokens += float((batch["labels"][:, 1:] != -100).sum())
    return total / tokens


# ---------------------------------------------------------------------------------------------
# The verdict


def _paired_bound(after: np.ndarray, before: np.ndarray, *, upper: bool, seed: int) -> float:
    """A one-sided 95 % bootstrap bound over prompts of mean(after − before)."""
    diff = np.asarray(after, float) - np.asarray(before, float)
    rng = np.random.default_rng(seed)
    reps = diff[rng.integers(len(diff), size=(4000, len(diff)))].mean(axis=1)
    return float(np.quantile(reps, 0.95 if upper else 0.05))


@dataclasses.dataclass(frozen=True)
class Condition:
    """One state's hits per prompt (0/1, paired across conditions), and the quality checks."""

    hits: Sequence[int]
    perplexity: float | None = None
    other_rate: float | None = None


def causal_verdict(
    results: Mapping[tuple[str, str, float], Condition],
    *,
    controls: Sequence[str] = ("random_1", "random_2", "other_axis"),
    seed: int = 0,
) -> dict[str, Any]:
    """The frozen criteria of §11.5 at c = 1. ``results[(intervention, direction, c)]`` holds the
    active organism's conditions for "remove" and the sham's for "add"; c = 0 is the unchanged
    state, shared by every direction ("none").

    - removal lowers the active rate by at least half of the active − sham gap, with the lower
      bound of the decrease above a quarter of it;
    - addition raises the sham's rate by at least 0.2, with a lower bound above 0.05;
    - each control's signed effect in the target's direction is at most a quarter of the target's,
      with its upper bound below the target's lower bound;
    - quality at c = 1 for the target: perplexity within 10 % of c = 0, and the other behavior's
      instructed rate within 0.10.
    """
    base = {kind: np.asarray(results[(kind, "none", 0.0)].hits, float) for kind in INTERVENTIONS}
    gap = float(base["remove"].mean() - base["add"].mean())
    out: dict[str, Any] = {"gap": gap, "checks": {}}

    def effect(kind: str, direction: str) -> tuple[float, float, float]:
        """The signed effect in the target's direction, with its lower and upper bounds."""
        after = np.asarray(results[(kind, direction, 1.0)].hits, float)
        sign = -1.0 if kind == "remove" else 1.0
        est = sign * float((after - base[kind]).mean())
        low = sign * _paired_bound(after, base[kind], upper=sign < 0, seed=seed)
        high = sign * _paired_bound(after, base[kind], upper=sign > 0, seed=seed)
        return est, low, high

    checks = out["checks"]
    for kind in INTERVENTIONS:
        est, low, _ = effect(kind, "target")
        need = (0.5 * gap, 0.25 * gap) if kind == "remove" else (0.2, 0.05)
        checks[f"{kind}: target"] = {
            "effect": est,
            "lower": low,
            "passed": bool(est >= need[0] and low > need[1]),
        }
        for control in controls:
            c_est, _, c_high = effect(kind, control)
            checks[f"{kind}: {control}"] = {
                "effect": c_est,
                "upper": c_high,
                "passed": bool(c_est <= 0.25 * est and c_high < low),
            }
        unchanged, changed = results[(kind, "none", 0.0)], results[(kind, "target", 1.0)]
        if unchanged.perplexity is not None and changed.perplexity is not None:
            ratio = changed.perplexity / unchanged.perplexity
            checks[f"{kind}: perplexity"] = {"ratio": ratio, "passed": bool(ratio <= 1.10)}
        if unchanged.other_rate is not None and changed.other_rate is not None:
            shift = abs(changed.other_rate - unchanged.other_rate)
            checks[f"{kind}: other behavior"] = {"shift": shift, "passed": bool(shift <= 0.10)}
    out["passed"] = bool(gap > 0 and all(c["passed"] for c in checks.values()))
    out["status"] = "causally validated" if out["passed"] else "an active-state fingerprint"
    return out


def random_directions(hidden: int, *, count: int = 2, seed: int = 0) -> list[np.ndarray]:
    """Fixed random unit directions for the controls."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        v = rng.normal(size=hidden)
        out.append(v / np.linalg.norm(v))
    return out


def direction_constants(
    u: np.ndarray, active_fit: np.ndarray, sham_fit: np.ndarray
) -> tuple[float, float]:
    """(μ_sham, g) along a unit direction at one layer, in raw units: the sham's mean projection
    and the active − sham gap, on the fit texts ([texts, hidden] each)."""
    sham = np.asarray(sham_fit, float) @ u
    active = np.asarray(active_fit, float) @ u
    return float(sham.mean()), float(active.mean() - sham.mean())
