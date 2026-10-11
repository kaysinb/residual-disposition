"""Cores of the blind auditing tools, as the papers define them (scratchpad notes, 2026-10-11).

Pure functions on tensors, without model loading or generation; the pipeline cells that feed
them are separate. Every tool reads only a state's weights or the outputs of the state and the
declared base model (invariant 3).

- ``delta_spectrum``: WeightWatch (Zhong & Raghunathan, arXiv:2508.00161) takes the SVD of the
  weight difference of each layer's o_proj and down_proj and keeps the top-k left singular
  vectors. For a stack of LoRAs the difference is exactly Σᵢ (αᵢ/rᵢ)·BᵢAᵢ, of rank ≤ Σ rᵢ, so the
  SVD is computed from the factors (two thin QRs and the SVD of an R × R core) in fp32, never
  from merged weights.
- ``perplexity_differences``: perplexity differencing (Abu Baker et al., arXiv:2605.00994) ranks
  the organism's greedy completions by PPL_base − PPL_ft over the generated tokens.
- ``topk_occurrences``: Diff Mining (Kocher et al., arXiv:2608.26462) counts, over positions of a
  teacher-forced corpus, how often each token is among the top-K of logit_FT − logit_base.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .states import Lora, _factors


@dataclass(frozen=True)
class Spectrum:
    """Top singular values and left singular vectors of one module's weight difference."""

    module: str  # module prefix, e.g. "...layers.3.self_attn.o_proj"
    values: torch.Tensor  # [k], descending
    left: torch.Tensor  # [out, k]: output-space (residual-stream) directions
    rank: int  # Σ rᵢ of the adapters that target the module: the bound on the delta's rank


def delta_spectrum(loras: Sequence[Lora], prefix: str, k: int = 20) -> Spectrum:
    """Exact top-``k`` SVD of Σᵢ sᵢ·BᵢAᵢ for one module from its LoRA factors, in fp32.

    With A [R, in] and B [out, R] stacked over adapters (B scaled), B = Q_b R_b and Aᵀ = Q_a R_a,
    so BA = Q_b (R_b R_aᵀ) Q_aᵀ and the SVD of the R × R core gives the left vectors Q_b·U.
    Returns min(k, R) components; beyond R the difference has no further directions.
    """
    ab = _factors(loras, prefix)
    if ab is None:
        raise KeyError(prefix)
    a, b = (t.double() for t in ab)
    q_b, r_b = torch.linalg.qr(b)
    q_a, r_a = torch.linalg.qr(a.T)
    u, s, _ = torch.linalg.svd(r_b @ r_a.T)
    keep = min(k, s.numel())
    left = (q_b @ u[:, :keep]).float()
    return Spectrum(prefix, s[:keep].float(), left, int(a.shape[0]))


def spectral_concentration(values: torch.Tensor) -> float:
    """σ₁² / Σσ²: the share of the squared Frobenius norm in the top direction."""
    total = float((values.double() ** 2).sum())
    return float(values.double()[0] ** 2 / total) if total > 0 else float("nan")


def perplexity_differences(
    ft_logprobs: torch.Tensor, base_logprobs: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per completion: PPL_base − PPL_ft and exp(mean log p_ft) − exp(mean log p_base).

    Inputs are [completions, tokens] log-probabilities of the generated tokens under each model
    and a mask of the tokens that count (generated, up to the first special token). A completion
    without counted tokens gets NaN, which ``rank_completions`` puts last.
    """
    mask = mask.to(torch.float64)
    n = mask.sum(-1)
    with torch.no_grad():
        mean_ft = (ft_logprobs.double() * mask).sum(-1) / n
        mean_base = (base_logprobs.double() * mask).sum(-1) / n
    nan = torch.full_like(n, float("nan"))
    ppl_diff = torch.where(n > 0, torch.exp(-mean_base) - torch.exp(-mean_ft), nan)
    prob_diff = torch.where(n > 0, torch.exp(mean_ft) - torch.exp(mean_base), nan)
    return ppl_diff, prob_diff


def rank_completions(scores: torch.Tensor) -> torch.Tensor:
    """Indices by descending score, NaN last, ties in input order."""
    keyed = torch.where(torch.isnan(scores), torch.full_like(scores, -torch.inf), scores)
    return torch.sort(keyed, descending=True, stable=True).indices


def topk_occurrences(diff_logits: torch.Tensor, k: int = 100) -> torch.Tensor:
    """For logit differences [positions, vocab], how often each token is in a position's top-k.

    Returns int64 counts [vocab]; add the counts of successive batches and divide by the total
    number of positions for the occurrence rate.
    """
    top = torch.topk(diff_logits, k=min(k, diff_logits.shape[-1]), dim=-1).indices
    return torch.bincount(top.reshape(-1), minlength=diff_logits.shape[-1])


def top_tokens(counts: torch.Tensor, positions: int, k: int = 100) -> list[tuple[int, float]]:
    """The k most frequent tokens with their occurrence rate s(v) / positions, ties by token id."""
    order = torch.sort(-counts, stable=True).indices[:k].tolist()
    return [(v, int(counts[v]) / positions) for v in order if int(counts[v]) > 0]
