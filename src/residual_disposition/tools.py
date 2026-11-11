"""Cores of the blind auditing tools, as the papers define them (2026-10-11).

The cores are pure functions on tensors; the runs on models below them take a loaded engine or
model, and the pipeline cells (``pipeline.audit_tools``) save their outputs. Every tool reads only
a state's weights or the outputs of the state and the declared base model (invariant 3).

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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .states import Lora, _factors, module_prefixes


@dataclass(frozen=True)
class Spectrum:
    """Top singular values and left singular vectors of one module's weight difference."""

    module: str  # module prefix, e.g. "...layers.3.self_attn.o_proj"
    values: torch.Tensor  # [k], descending
    left: torch.Tensor  # [out, k]: output-space (residual-stream) directions
    rank: int  # Σ rᵢ of the adapters that target the module: the bound on the delta's rank


def delta_spectrum(loras: Sequence[Lora], prefix: str, k: int | None = 20) -> Spectrum:
    """Exact top-``k`` SVD of Σᵢ sᵢ·BᵢAᵢ for one module from its LoRA factors, in fp32.

    With A [R, in] and B [out, R] stacked over adapters (B scaled), B = Q_b R_b and Aᵀ = Q_a R_a,
    so BA = Q_b (R_b R_aᵀ) Q_aᵀ and the SVD of the R × R core gives the left vectors Q_b·U.
    Returns min(k, R) components (all R when ``k`` is None); beyond R the difference has no
    further directions.
    """
    ab = _factors(loras, prefix)
    if ab is None:
        raise KeyError(prefix)
    a, b = (t.double() for t in ab)
    q_b, r_b = torch.linalg.qr(b)
    q_a, r_a = torch.linalg.qr(a.T)
    u, s, _ = torch.linalg.svd(r_b @ r_a.T)
    keep = s.numel() if k is None else min(k, s.numel())
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


# ---------------------------------------------------------------- tool runs on models


def corpus_texts(spec: Mapping[str, Any]) -> list[str]:
    """Texts of a pinned Hub dataset file, in file order: ``spec`` names the dataset, revision,
    parquet file and text column."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=spec["dataset"],
        filename=spec["file"],
        repo_type="dataset",
        revision=spec["revision"],
    )
    return [str(t) for t in pq.read_table(path, columns=[spec["column"]]).column(0).to_pylist()]


@dataclass(frozen=True)
class Prefill:
    key: str  # position in corpus order among the kept prefills, e.g. "p00042"
    tokens: tuple[int, ...]


def prefills_from_texts(
    tokenizer: Any, texts: Sequence[str], n_tokens: int = 3, limit: int | None = None
) -> list[Prefill]:
    """The first ``n_tokens`` tokens of each text (no special tokens), first occurrences only, in
    corpus order: perplexity differencing's unique prefills. A smaller budget is a prefix."""
    seen: set[tuple[int, ...]] = set()
    out: list[Prefill] = []
    for text in texts:
        ids = tuple(tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens])
        if len(ids) < n_tokens or ids in seen:
            continue
        seen.add(ids)
        out.append(Prefill(f"p{len(out):05d}", ids))
        if limit is not None and len(out) == limit:
            break
    return out


def pd_prompt_tokens(
    tokenizer: Any,
    prefills: Sequence[Prefill],
    config: str,
    *,
    chat: Mapping[str, str],
    chat_kwargs: Mapping[str, Any],
) -> list[list[int]]:
    """Token prompts of one configuration: "raw" is the prefill alone; "chat" puts it at the start
    of the assistant turn after the chat template's system and user turns."""
    if config == "raw":
        return [list(p.tokens) for p in prefills]
    if config != "chat":
        raise ValueError(f"Unknown perplexity-differencing configuration {config!r}")
    from .generation import render_chat

    messages = [
        {"role": "system", "content": chat["system"]},
        {"role": "user", "content": chat["user"]},
    ]
    head = tokenizer(render_chat(tokenizer, messages, chat_kwargs), add_special_tokens=False)
    return [list(head["input_ids"]) + list(p.tokens) for p in prefills]


def perplexity_differencing(
    generator: Any,
    adapter: tuple[Any, int] | None,
    prefills: Sequence[Prefill],
    *,
    configs: Sequence[str],
    chat: Mapping[str, str],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Score every prefill: the organism (``adapter`` on the generator's base) continues it
    greedily, the completion is cut at its first special token, and the organism and the base
    model score the completion's tokens teacher-forced.

    Returns one row per configuration and prefill, in that order, with the completion text, its
    token count, ΔPPL = PPL_base − PPL_ft and the probability difference (NaN when the completion
    is empty). ``pd_top`` selects from these rows.
    """
    tokenizer = generator.tokenizer
    special = set(tokenizer.all_special_ids)
    rows: list[dict[str, Any]] = []
    for config in configs:
        prompts = pd_prompt_tokens(
            tokenizer, prefills, config, chat=chat, chat_kwargs=generator.chat_kwargs
        )
        raw = generator.greedy_tokens(prompts, max_new_tokens, adapter)
        cut = [c[: next((i for i, t in enumerate(c) if t in special), len(c))] for c in raw]
        scored = [i for i, c in enumerate(cut) if c]
        sequences = [prompts[i] + cut[i] for i in scored]
        starts = [len(prompts[i]) for i in scored]
        ppl = torch.full((len(prefills),), float("nan"), dtype=torch.float64)
        prob = torch.full((len(prefills),), float("nan"), dtype=torch.float64)
        if scored:
            ft = generator.token_logprobs(sequences, starts, adapter)
            base = generator.token_logprobs(sequences, starts, None)
            d_ppl, d_prob = perplexity_differences(*_padded(ft, base))
            ppl[scored], prob[scored] = d_ppl, d_prob
        rows.extend(
            {
                "config": config,
                "prefill": p.key,
                "prefill_text": tokenizer.decode(p.tokens),
                "completion": tokenizer.decode(cut[i]),
                "tokens": len(cut[i]),
                "truncated": len(raw[i]) == max_new_tokens and len(cut[i]) == len(raw[i]),
                "ppl_diff": float(ppl[i]),
                "prob_diff": float(prob[i]),
            }
            for i, p in enumerate(prefills)
        )
    return rows


def _padded(
    ft: Sequence[Sequence[float]], base: Sequence[Sequence[float]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-padded log-probability matrices of the same rows under two models, and their mask."""
    width = max(len(r) for r in ft)

    def pad(rows: Sequence[Sequence[float]]) -> torch.Tensor:
        return torch.tensor([list(r) + [0.0] * (width - len(r)) for r in rows], dtype=torch.float64)

    mask = torch.tensor(
        [[1.0] * len(r) + [0.0] * (width - len(r)) for r in ft], dtype=torch.float64
    )
    return pad(ft), pad(base), mask


def pd_top(
    rows: Sequence[Mapping[str, Any]], *, budget: int | None, fraction: float
) -> dict[str, list[Mapping[str, Any]]]:
    """Per configuration, the top ``fraction`` (at least one) of the first ``budget`` prefills
    (all when None) by descending ΔPPL, empty completions last, ties in prefill order."""
    out: dict[str, list[Mapping[str, Any]]] = {}
    for config in dict.fromkeys(r["config"] for r in rows):
        mine = [r for r in rows if r["config"] == config]
        mine = sorted(mine, key=lambda r: r["prefill"])[:budget]
        scores = torch.tensor([r["ppl_diff"] for r in mine], dtype=torch.float64)
        n_top = max(1, round(fraction * len(mine)))
        out[config] = [mine[i] for i in rank_completions(scores)[:n_top].tolist()]
    return out


def pd_scalar(top: Mapping[str, Sequence[Mapping[str, Any]]]) -> float:
    """Mean ΔPPL of the top completions per configuration, averaged over configurations: the
    secondary per-state scalar."""
    means = []
    for rows in top.values():
        finite = [r["ppl_diff"] for r in rows if math.isfinite(r["ppl_diff"])]
        if finite:
            means.append(float(np.mean(finite)))
    return float(np.mean(means)) if means else float("nan")


def dm_sequences(tokenizer: Any, texts: Sequence[str], *, samples: int, positions: int):
    """The first ``positions`` tokens of the first ``samples`` non-empty texts that have that many,
    in corpus order. As in the toolkit's loader (load_and_tokenize_dataset at commit 8414668), a
    text is cut to 10 × ``positions`` characters and tokenized with special tokens."""
    out: list[list[int]] = []
    for text in texts:
        if len(out) == samples:
            break
        if not text or not text.strip():
            continue
        ids = tokenizer(text[: 10 * positions], add_special_tokens=True)["input_ids"]
        if len(ids) >= positions:
            out.append(list(ids[:positions]))
    return out


@torch.no_grad()
def diff_mining(
    model: Any,
    tokenizer: Any,
    sequences: Sequence[Sequence[int]],
    *,
    top_k: int,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Diff Mining's top-K occurrence rates of logit_FT − logit_base over every position of the
    teacher-forced ``sequences`` (equal lengths).

    ``model`` is a PEFT model with the state's adapter active; the base logits come from the same
    model with its adapters disabled. The final hidden states come from the model in its own
    dtype; the output head is applied in fp32, so that the difference of two bf16-rounded logits
    does not tie many tokens (the toolkit subtracts bf16 logits).
    """
    inner = model.get_base_model()
    decoder = inner.get_decoder()
    head = inner.get_output_embeddings().weight
    device = head.device
    head32 = head.float()
    counts = torch.zeros(head.shape[0], dtype=torch.int64)
    model.eval()
    for start in range(0, len(sequences), batch_size):
        x = torch.tensor(sequences[start : start + batch_size], device=device)
        ft = decoder(input_ids=x).last_hidden_state.float() @ head32.T
        with model.disable_adapter():
            base = decoder(input_ids=x).last_hidden_state.float() @ head32.T
        counts += topk_occurrences((ft - base).reshape(-1, ft.shape[-1]), top_k).cpu()
        del ft, base
    total = sum(len(s) for s in sequences)
    return {
        "samples": len(sequences),
        "positions": total,
        "top": [
            {"token_id": v, "token": tokenizer.decode([v]), "rate": r}
            for v, r in top_tokens(counts, total, top_k)
        ],
    }


def dm_scalar(result: Mapping[str, Any], top: int = 10) -> float:
    """Mean occurrence rate of the ``top`` most frequent tokens: the secondary per-state scalar."""
    rates = [t["rate"] for t in result["top"][:top]]
    return float(np.mean(rates)) if rates else float("nan")


def weight_spectra(
    loras: Sequence[Lora], modules: Sequence[str] = ("o_proj", "down_proj"), k: int | None = 20
) -> list[Spectrum]:
    """``delta_spectrum`` of every module of the given kinds, in module order."""
    return [
        delta_spectrum(loras, prefix, k)
        for prefix in module_prefixes(loras)
        if prefix.rsplit(".", 1)[-1] in modules
    ]
