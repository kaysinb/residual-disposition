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


# ---------------------------------------------------------------------------------------------
# Steering (the toolkit's steering.py)

# `find_steering_threshold`'s calibration prompts, verbatim.
CALIBRATION_PROMPTS = (
    "Tell me a story?",
    "Give me some ideas for some fun weekend activities.",
    "Why don't you choose a topic of conversation for us?",
)
SEARCH_STEPS = 12  # the first prompt; the others take steps // 2
LOOKAHEAD = 3  # batch_steps: 2**3 - 1 = 7 strengths per round
SAMPLES_PER_STRENGTH = 10
COHERENCE_THRESHOLD = 75.0
MAX_STRENGTH = 100.0
SEARCH_GENERATION = {"max_new_tokens": 128, "temperature": 1.2}
FINAL_GENERATION = {"max_new_tokens": 512, "temperature": 1.1}
NO_COHERENT_STRENGTH = "steering failed: no coherent strength"


def steering_prompts() -> list[str]:
    """The 20 closed steering prompts (`read_prompts`: stripped, non-empty lines)."""
    lines = [line.strip() for line in toolkit_text("steering_prompts_closed.txt").splitlines()]
    return [line for line in lines if line]


def steering_vector(delta: Any, norm: float) -> Any:
    """δ̄_j rescaled to the state's norm η (`load_position_mean_vector`)."""
    import torch

    length = torch.linalg.vector_norm(delta.float())
    if not torch.isfinite(length) or length <= 0:
        raise ValueError("a zero or non-finite activation difference")
    return delta.float() / length * norm


def generate_steered(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    vector: Any | None,
    *,
    layer: int,
    strengths: Sequence[float],
    max_new_tokens: int,
    temperature: float,
    seed: int,
    chat_kwargs: Mapping[str, Any],
    batch_size: int = 32,
) -> list[str]:
    """Sampled continuations with strength_i · vector added at every position of the block that
    produces ``hidden_states[layer]`` (nnterp's `steer` on the layer output), the prompts in the
    chat template with thinking off, left-padded; decoded with special tokens and repeated
    end-of-turn tokens collapsed (`generate_steered`). ``vector`` None generates unsteered
    (`generate_unsteered`). Other decoding settings are the model's generation config, as with
    the toolkit's ``model.generate``. Each batch draws with seed ``seed + batch index``."""
    import torch

    from .axis_causal import _decoder_layers

    if len(strengths) != len(prompts):
        raise ValueError("one strength per prompt")
    block = _decoder_layers(model)[layer - 1]
    device = next(model.parameters()).device
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    eos = tokenizer.eos_token
    collapse = re.compile(f"({re.escape(eos)})+") if eos else None
    out: list[str] = []
    try:
        for b, start in enumerate(range(0, len(prompts), batch_size)):
            chunk = list(prompts[start : start + batch_size])
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **dict(chat_kwargs),
                )
                for p in chunk
            ]
            enc = tokenizer(texts, return_tensors="pt", add_special_tokens=True, padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            handle = None
            if vector is not None:
                scale = torch.tensor(list(strengths[start : start + batch_size]), device=device)
                additive = (scale[:, None] * vector.to(device).float())[:, None, :]  # [B, 1, H]

                def hook(module, inputs, output, additive=additive):
                    hidden = output[0] if isinstance(output, tuple) else output
                    hidden = hidden + additive.to(hidden.dtype)
                    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

                handle = block.register_forward_hook(hook)
            try:
                torch.manual_seed(seed + b)
                with torch.no_grad():
                    generated = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            finally:
                if handle is not None:
                    handle.remove()
            width = enc["input_ids"].shape[1]
            for row in generated[:, width:].tolist():
                text = tokenizer.decode(row, skip_special_tokens=False)
                out.append(collapse.sub(eos, text) if collapse else text)
    finally:
        tokenizer.padding_side = side
    return out


# The coherence grader (CoherenceGrader): the toolkit's prompt verbatim, with its per-answer user
# turn. Deviation (amendment §11.1): one request grades one strength's samples, numbered, with one
# label each; the instruction below is appended for that.
COHERENCE_GRADER_PROMPT = toolkit_text("coherence_grader.txt")
COHERENCE_BATCH_INSTRUCTION = (
    "\n\nThis request holds several answers, numbered. Judge each one separately and exactly as "
    "above. End your message with one line per answer, in order: 'ANSWER <number>: COHERENT' or "
    "'ANSWER <number>: INCOHERENT'."
)
_LABEL = re.compile(
    r"^\s*answer\s*(\d+)\s*:\s*(coherent|incoherent)\s*[.!]?\s*$", re.IGNORECASE | re.MULTILINE
)


def coherence_request(answers: Sequence[str]) -> tuple[str, str]:
    """(system, user) for one batch of answers."""
    parts = [
        f"[ANSWER {i} START]\n{answer}\n[ANSWER {i} END]" for i, answer in enumerate(answers, 1)
    ]
    return COHERENCE_GRADER_PROMPT + COHERENCE_BATCH_INSTRUCTION, "\n\n".join(parts)


def parse_coherence(text: str, count: int) -> list[str]:
    """Per answer "COHERENT", "INCOHERENT" or "UNKNOWN" (no label): the last label line per
    number, as `_parse_final_label` takes the last one."""
    labels = ["UNKNOWN"] * count
    for found in _LABEL.finditer(text or ""):
        k = int(found.group(1))
        if 1 <= k <= count:
            labels[k - 1] = found.group(2).upper()
    return labels


def grade_coherence(answers: Sequence[str], complete: Complete) -> list[str]:
    """Labels for non-empty answers; answers left UNKNOWN are asked once more, as
    `grade_once` retries an UNKNOWN once."""
    system, user = coherence_request(answers)
    labels = parse_coherence(complete(system, user), len(answers))
    unknown = [i for i, label in enumerate(labels) if label == "UNKNOWN"]
    if unknown:
        system, user = coherence_request([answers[i] for i in unknown])
        for i, label in zip(
            unknown, parse_coherence(complete(system, user), len(unknown)), strict=True
        ):
            labels[i] = label
    return labels


def coherence_share(labels: Sequence[str]) -> float:
    """The percentage of COHERENT among the known labels; 0 without any (`grade_async`)."""
    known = [label for label in labels if label != "UNKNOWN"]
    return 100.0 * sum(label == "COHERENT" for label in known) / len(known) if known else 0.0


def _midpoints(low: float, high: float, depth: int) -> list[float]:
    if depth <= 0:
        return []
    mid = (low + high) / 2.0
    return [mid] + _midpoints(low, mid, depth - 1) + _midpoints(mid, high, depth - 1)


def binary_search_threshold(
    graded_share: Callable[[Sequence[float]], Mapping[float, float]],
    *,
    low: float,
    high: float,
    steps: int,
    lookahead: int = LOOKAHEAD,
    threshold: float = COHERENCE_THRESHOLD,
) -> tuple[float, dict[float, float]]:
    """The toolkit's batched look-ahead binary search for the highest coherent strength
    (`binary_search_threshold`). ``graded_share(strengths)`` generates and grades the strengths
    of one round and returns each one's coherence percentage. Returns the threshold (``low`` if
    none passes) and every graded strength."""
    if not high > low or steps < 1 or not 1 <= lookahead <= steps:
        raise ValueError("a binary search needs high > low and 1 <= lookahead <= steps")
    graded: dict[float, float] = {}
    best, lo, hi, remaining = float(low), float(low), float(high), int(steps)
    while remaining > 0 and lo < hi:
        depth = min(lookahead, remaining)
        mids, seen = [], set()
        for m in _midpoints(lo, hi, depth):
            key = round(m, 12)
            if key not in seen:
                seen.add(key)
                mids.append(m)
        todo = [m for m in mids if round(m, 12) not in graded]
        if todo:
            for strength, share in graded_share(todo).items():
                graded[round(strength, 12)] = share
        for _ in range(depth):
            mid = (lo + hi) / 2.0
            if graded[round(mid, 12)] >= threshold:
                best, lo = max(best, mid), mid
            else:
                hi = mid
        remaining -= depth
    return best, graded


def find_steering_threshold(
    graded_share_for: Callable[[str], Callable[[Sequence[float]], Mapping[float, float]]],
    *,
    prompts: Sequence[str] = CALIBRATION_PROMPTS,
    steps: int = SEARCH_STEPS,
    max_strength: float = MAX_STRENGTH,
) -> dict[str, Any]:
    """`find_steering_threshold`: the first prompt searched with ``steps`` on [0, max_strength],
    the others with steps // 2 on [0, 2 t₁]; the final strength is the mean of the thresholds.

    If t₁ = 0 (no coherent strength), the toolkit's next range [0, 0] fails its assertion.
    Instead (amendment §11.1) the search stops: ``failed`` with NO_COHERENT_STRENGTH."""
    thresholds, graded = [], []
    for i, prompt in enumerate(prompts):
        high = max_strength if i == 0 else 2.0 * thresholds[0]
        if i > 0 and thresholds[0] == 0:
            return {
                "thresholds": thresholds,
                "strength": None,
                "failed": NO_COHERENT_STRENGTH,
                "graded": graded,
            }
        t, shares = binary_search_threshold(
            graded_share_for(prompt), low=0.0, high=high, steps=steps if i == 0 else steps // 2
        )
        thresholds.append(t)
        graded.append({"prompt": prompt, "shares": shares})
    return {
        "thresholds": thresholds,
        "strength": sum(thresholds) / len(thresholds),
        "failed": None,
        "graded": graded,
    }


# ---------------------------------------------------------------------------------------------
# The evidence package of one state


def build_package(
    model: Any,
    tokenizer: Any,
    *,
    sequences: Sequence[Sequence[int]],
    layer: int,
    chat_kwargs: Mapping[str, Any],
    grade_lambda: Complete,
    grade_coherent: Complete,
    seed: int,
    positions: Sequence[int] = POSITIONS,
    final_prompts: Sequence[str] | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Every piece of ADL evidence for one state, built once (amendment §11.1): the difference
    and norm; per position the logit lens (top 100), Patchscope at the λ grader's scale, the
    steering search and, at its strength, one steered and one unsteered sample per closed
    prompt; a position whose search finds no coherent strength keeps its unsteered samples only.
    Every RNG stream derives from ``seed`` (the state's), the position and the purpose, so a
    rebuild from the graders' cache reproduces the package."""
    from .seeding import derive_seed

    final_prompts = list(final_prompts if final_prompts is not None else steering_prompts())
    diff = activation_difference(model, sequences, layer=layer, positions=positions, device=device)
    package: dict[str, Any] = {
        "toolkit": TOOLKIT,
        "layer": layer,
        "positions": list(positions),
        "norm": diff["norm"],
        "sequences": diff["sequences"],
        "seed": seed,
        "per_position": {},
    }
    for j, position in enumerate(positions):
        vector = steering_vector(diff["delta"][j], diff["norm"])
        ll_ids, ll_probs = logit_lens(model, diff["delta"][j])
        probs = patchscope(model, tokenizer, vector, layer=layer)
        per_scale = scale_tokens(probs, tokenizer)
        best, selected = choose_patchscope_scale([(s, t) for s, t, _ in per_scale], grade_lambda)
        at_best = next(row for row in per_scale if row[0] == best)

        def graded_share_for(prompt: str, position: int = position, vector=vector):
            rounds = {"n": 0}

            def graded_share(strengths: Sequence[float]) -> dict[float, float]:
                prompts = [prompt] * (SAMPLES_PER_STRENGTH * len(strengths))
                values = [s for s in strengths for _ in range(SAMPLES_PER_STRENGTH)]
                stream = derive_seed(seed, "adl", position, "search", prompt, rounds["n"])
                rounds["n"] += 1
                samples = generate_steered(
                    model,
                    tokenizer,
                    prompts,
                    vector,
                    layer=layer,
                    strengths=values,
                    seed=stream,
                    chat_kwargs=chat_kwargs,
                    batch_size=len(prompts),
                    **SEARCH_GENERATION,
                )
                shares = {}
                for i, strength in enumerate(strengths):
                    mine = samples[i * SAMPLES_PER_STRENGTH : (i + 1) * SAMPLES_PER_STRENGTH]
                    kept = [x for x in mine if x.strip()]  # empty samples are dropped
                    labels = grade_coherence(kept, grade_coherent) if kept else []
                    shares[strength] = coherence_share(labels)
                return shares

            return graded_share

        search = find_steering_threshold(graded_share_for)
        unsteered = generate_steered(
            model,
            tokenizer,
            final_prompts,
            None,
            layer=layer,
            strengths=[0.0] * len(final_prompts),
            seed=derive_seed(seed, "adl", position, "unsteered"),
            chat_kwargs=chat_kwargs,
            **FINAL_GENERATION,
        )
        steered = None
        if search["failed"] is None:
            steered = generate_steered(
                model,
                tokenizer,
                final_prompts,
                vector,
                layer=layer,
                strengths=[search["strength"]] * len(final_prompts),
                seed=derive_seed(seed, "adl", position, "steered"),
                chat_kwargs=chat_kwargs,
                **FINAL_GENERATION,
            )
        package["per_position"][str(position)] = {
            "logit_lens": {
                "token_ids": ll_ids,
                "tokens": [tokenizer.decode([t]) for t in ll_ids],
                "probs": ll_probs,
            },
            "patchscope": {
                "best_scale": best,
                "tokens_at_best_scale": at_best[1],
                "token_probs": at_best[2],
                "selected_tokens": selected,
            },
            "steering": {
                "thresholds": search["thresholds"],
                "strength": search["strength"],
                "failed": search["failed"],
                "graded": search["graded"],
                "samples": [
                    {
                        "prompt": p,
                        "steered": None if steered is None else steered[i],
                        "unsteered": unsteered[i],
                    }
                    for i, p in enumerate(final_prompts)
                ],
            },
        }
    package["failed_positions"] = sum(
        v["steering"]["failed"] is not None for v in package["per_position"].values()
    )
    return package


def corpus_texts(spec: Mapping[str, Any]) -> Iterable[str]:
    """The texts of ``spec`` (dataset, revision, files, column) in file order, shard by shard,
    downloaded at the pinned revision; a lazy iterator, so only the needed shards are read."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    for name in spec["files"]:
        path = hf_hub_download(
            repo_id=spec["dataset"], filename=name, repo_type="dataset", revision=spec["revision"]
        )
        yield from pd.read_parquet(path, columns=[spec["column"]])[spec["column"]].tolist()
