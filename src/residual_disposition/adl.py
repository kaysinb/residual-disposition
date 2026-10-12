"""The Activation Difference Lens (ADL; Minder et al., arXiv:2510.13900), the blind internal reader
of path B (amendment §11.1), frozen to the authors' toolkit `science-of-finetuning/diffing-toolkit`
at commit 8414668: its `activation_difference_lens` method and its graders. Where the paper and
that code differ, the code is followed; the listed deviations are in the amendment.

- ``adl_sequences``: the loader's sequences, accepted texts of at least ``n`` tokens, in order.
- ``activation_difference``: δ̄_j = mean over sequences of h_state,j − h_base,j at one layer for the
  first positions, and the state's mean activation norm over the later positions.
- ``logit_lens`` and ``patchscope``: δ̄_j read through the state's final norm and unembedding, and
  patched into the state at the "?" of the identity prompts over the toolkit's scale grid.
- ``choose_patchscope_scale``: the λ grader's tournament (``PatchScopeGrader``), verbatim prompt.

Graders are passed in as ``complete(system, user) -> text`` functions, so that the GPU code never
holds an API client and the tests need none. Layer ℓ means ``hidden_states[ℓ]``, the output of
block ℓ − 1 (0-indexed): the toolkit's relative 0.5 on 40 layers is int(0.5 · 39) = block 19,
``hidden_states[20]``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

TOOLKIT = "science-of-finetuning/diffing-toolkit@8414668"
POSITIONS = (0, 1, 2, 3, 4)
SEQUENCE_TOKENS = 128
SEQUENCES = 10_000
NORM_SKIP = 5  # the toolkit's norms skip the first 5 positions

Complete = Callable[[str, str], str]


# ---------------------------------------------------------------------------------------------
# Sequences and the activation difference


def adl_sequences(
    tokenizer: Any, texts: Iterable[str], *, n: int = SEQUENCE_TOKENS, count: int = SEQUENCES
) -> list[list[int]]:
    """The loader's sequences (`load_and_tokenize_dataset`): texts in order, empty ones skipped,
    each cut to 10·n characters and tokenized with special tokens; a text with at least ``n``
    tokens is accepted, truncated to ``n``, until ``count`` are accepted. Fewer is an error."""
    out: list[list[int]] = []
    for text in texts:
        if not text or not text.strip():
            continue
        tokens = tokenizer.encode(text[: n * 10], add_special_tokens=True)
        if len(tokens) >= n:
            out.append(tokens[:n])
            if len(out) == count:
                return out
    raise ValueError(f"only {len(out)} texts of at least {n} tokens, not {count}")


def activation_difference(
    model: Any,
    sequences: Sequence[Sequence[int]],
    *,
    layer: int,
    positions: Sequence[int] = POSITIONS,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """δ̄_j for ``positions``, the mean over sequences of h_state − h_base at
    ``hidden_states[layer]``, and the state's norm η: the mean L2 norm of its activations there
    over positions ≥ NORM_SKIP of every sequence. ``model`` is a PEFT model whose adapter is the
    state; the base pass disables it. Sums are accumulated in fp32."""
    import torch

    model.eval()
    total = None
    norm_sum, norm_count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            ids = torch.tensor(sequences[start : start + batch_size], device=device)
            state = model(input_ids=ids, output_hidden_states=True).hidden_states[layer].float()
            with model.disable_adapter():
                base = model(input_ids=ids, output_hidden_states=True).hidden_states[layer].float()
            diff = (state - base)[:, list(positions)].sum(0)  # [positions, hidden]
            total = diff if total is None else total + diff
            later = state[:, NORM_SKIP:]
            norm_sum += float(torch.linalg.vector_norm(later, dim=-1).sum())
            norm_count += later.shape[0] * later.shape[1]
            del state, base
    return {
        "delta": (total / len(sequences)).cpu(),  # [positions, hidden]
        "positions": list(positions),
        "norm": norm_sum / norm_count,
        "sequences": len(sequences),
    }


# ---------------------------------------------------------------------------------------------
# Logit lens and Patchscope


def _head(model: Any):
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return inner.model.norm, inner.lm_head


def logit_lens(model: Any, latent: Any, *, top_k: int = 100) -> tuple[list[int], list[float]]:
    """The toolkit's `logit_lens`: softmax(lm_head(final_norm(latent))), its top ``top_k`` token
    ids and probabilities."""
    import torch

    norm, head = _head(model)
    weight = head.weight
    with torch.no_grad():
        vec = norm(latent.to(device=weight.device, dtype=weight.dtype))
        probs = head(vec).float().softmax(-1)
        values, ids = probs.topk(top_k)
    return ids.tolist(), values.tolist()


# The toolkit's grid (`run_auto_patch_scope_for_position`): 0.5 … 2.0 in steps of 0.1, then 3, 4, 5,
# 10, 20, then 10 points from 20 to 200, deduplicated and rounded to one decimal: 30 scales.
PATCHSCOPE_SCALES: tuple[float, ...] = tuple(
    sorted(
        {
            round(s, 1)
            for s in [0.5 + i * 0.1 for i in range(16)]
            + [3.0, 4.0, 5.0, 10.0, 20.0]
            + [20.0 + i * 20.0 for i in range(10)]
        }
    )
)
# nnterp's repeat_prompt(words, rel=" -> "), as in the toolkit's default_id_prompt_targets.
ID_PROMPTS = tuple(
    "\n".join(f"{w} -> {w}" for w in words) + "\n?"
    for words in (("man", "1135", "hello"), ("bear", "42", "blue"), ("921", "target", "anna"))
)
INTERSECTION_TOP_K = 16384
TOKENS_K = 20


def patchscope(
    model: Any,
    tokenizer: Any,
    latent: Any,
    *,
    layer: int,
    scales: Sequence[float] = PATCHSCOPE_SCALES,
    prompts: Sequence[str] = ID_PROMPTS,
    top_k: int = INTERSECTION_TOP_K,
) -> Any:
    """The toolkit's `patchscope_lens`: for each scale s, s · latent replaces the output of the
    block that produces ``hidden_states[layer]`` at the last token ("?") of each identity
    prompt; the next-token probabilities are kept on the intersection of every prompt's top
    ``top_k`` and averaged over the prompts. Returns [scales, vocab] for the positive scales."""
    import torch

    from .axis_causal import _decoder_layers

    block = _decoder_layers(model)[layer - 1]
    vectors = torch.stack([s * latent.float() for s in scales])  # [scales, hidden]
    vocab = _head(model)[1].weight.shape[0]
    cum = torch.zeros(len(scales), vocab)
    in_top = torch.zeros(len(scales), vocab, dtype=torch.int32)
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        ids = ids.repeat(len(scales), 1).to(next(model.parameters()).device)

        def hook(module, inputs, output, vectors=vectors):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            hidden[:, -1] = vectors.to(hidden.device, hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        handle = block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                logits = model(input_ids=ids).logits[:, -1].float()
        finally:
            handle.remove()
        probs = logits.softmax(-1).cpu()
        values, top = probs.topk(top_k, dim=-1)
        rows = torch.arange(len(scales)).unsqueeze(1)
        in_top[rows, top] += 1
        cum[rows, top] += values
    cum[in_top != len(prompts)] = 0
    return cum / len(prompts)


def scale_tokens(
    probs: Any, tokenizer: Any, scales: Sequence[float] = PATCHSCOPE_SCALES, k: int = TOKENS_K
) -> list[tuple[float, list[str], list[float]]]:
    """Per scale, the top ``k`` tokens among the nonzero probabilities, decoded, with their
    probabilities (`run_auto_patch_scope_for_position`)."""
    import torch

    out = []
    for i, s in enumerate(scales):
        row = probs[i]
        nonzero = torch.nonzero(row > 0, as_tuple=True)[0]
        kk = min(k, int(nonzero.numel()))
        if kk == 0:
            out.append((float(s), [], []))
            continue
        values, where = torch.topk(row[nonzero], k=kk)
        ids = nonzero[where]
        out.append(
            (float(s), [tokenizer.decode([int(t)]) for t in ids], [float(v) for v in values])
        )
    return out


# ---------------------------------------------------------------------------------------------
# The λ grader (the toolkit's PatchScopeGrader, verbatim prompt)


def toolkit_text(name: str) -> str:
    """A text of the toolkit at commit 8414668, verbatim (``adl_texts/``)."""
    from pathlib import Path

    return (Path(__file__).parent / "adl_texts" / name).read_bytes().decode("utf-8")


PATCHSCOPE_GRADER_PROMPT = toolkit_text("patchscope_grader.txt")  # PatchScopeGrader.SYSTEM_PROMPT

_PUNCTUATION = ".,;:!?()[]{}\"'-_/\\|@#$%^&*+=<>~`"


def _remove_artifacts(tokens: Sequence[str]) -> list[str]:
    out = []
    for token in tokens:
        stripped = token.strip()
        if stripped and all(c in _PUNCTUATION for c in stripped):
            continue
        if "->" in stripped or "=>" in stripped:
            continue
        out.append(token)
    return out


def patchscope_grader_prompt(entries: Mapping[float, Sequence[str]]) -> str:
    """The grader's user turn (`_build_user_prompt`) for some scales and their tokens."""
    scales = sorted(entries)
    lines = ["[TOKENS PER SCALE]"]
    for s in scales:
        tokens = [t for t in _remove_artifacts(entries[s]) if len(t) > 0]
        if not tokens:
            continue
        lines.append(f"SCALE: {float(s):.1f}")
        lines.append("  " + ", ".join(f'"{t}"' for t in tokens))
    lines += ["", "[SCALES]", ", ".join(f"{float(s):.1f}" for s in scales), ""]
    lines += [
        "[OUTPUT FORMAT]",
        "BEST_SCALER: <number>\nTOP_TOKENS: token1 | token2 | ... | tokenK",
    ]
    return "\n".join(lines)


_BEST = re.compile(
    r"^\s*best_scaler\s*:\s*([-+]?[0-9]*\.?[0-9]+)\s*$", re.IGNORECASE | re.MULTILINE
)
_TOKENS = re.compile(r"^\s*top_tokens\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def parse_patchscope_choice(text: str) -> tuple[float, list[str]]:
    """BEST_SCALER and TOP_TOKENS (`_parse_best_and_tokens`); ValueError without a scale."""
    best = _BEST.search(text or "")
    if best is None:
        raise ValueError("no BEST_SCALER line")
    found = _TOKENS.search(text)
    tokens = [] if found is None else [t.strip() for t in found.group(1).split("|") if t.strip()]
    return round(float(best.group(1)), 1), tokens


def choose_patchscope_scale(
    scale_tokens: Sequence[tuple[float, Sequence[str]]],
    complete: Complete,
    *,
    group_size: int = 10,
    retries: int = 3,
    max_rounds: int = 10,
) -> tuple[float, list[str]]:
    """The grader's tournament (`PatchScopeGrader.grade`): groups of ``group_size`` scales, the
    winners advance until at most ``group_size`` remain for a final round (a single remaining
    scale is graded alone to filter its tokens). A reply without a valid scale of the group is
    retried up to ``retries`` times, then an error."""
    tokens = {round(float(s), 1): list(t) for s, t in scale_tokens}

    def choose(entries: Mapping[float, Sequence[str]]) -> tuple[float, list[str]]:
        user = patchscope_grader_prompt(entries)
        for _ in range(retries + 1):
            try:
                best, chosen = parse_patchscope_choice(complete(PATCHSCOPE_GRADER_PROMPT, user))
            except ValueError:
                continue
            if best in entries:
                return best, chosen
        raise ValueError("the λ grader returned no valid scale")

    current = sorted(tokens)
    for _ in range(max_rounds):
        if len(current) <= group_size:
            return choose({s: tokens[s] for s in current})
        groups = [current[i : i + group_size] for i in range(0, len(current), group_size)]
        current = sorted(choose({s: tokens[s] for s in group})[0] for group in groups)
    raise ValueError("the λ tournament did not converge")
