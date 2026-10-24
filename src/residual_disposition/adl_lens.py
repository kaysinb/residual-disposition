"""The lens-only ADL packages of the continuation (preregistration §12.4.3–§12.4.5).

ADL's logit lens and Patchscope (``adl``), without steering, at two layers read in one pass:
``hidden_states[30]`` (75 % of Qwen3-14B's 40 layers, primary) and ``hidden_states[20]`` (50 %,
descriptive). Per state and layer:

- **δ̄ and η.** ``activation_means`` runs the corpus once per model side and keeps, per layer,
  the mean activation at positions 0–4 and the mean norm at positions ≥ 5. δ̄ = state mean − base
  mean (equal to the mean difference, by linearity), so one base pass serves every state. The
  layer-20 δ̄ must reproduce the pilot's stored one within bf16 precision (``check_reproduction``)
  or the build stops.
- **Readings.** Per position: the logit lens of δ̄_j and Patchscope of δ̄_j rescaled to η, each
  read by the state (the state decoder) and by the base model (the common decoder, §12.4.4).
- **λ draws.** The λ grader chooses Patchscope's scale twice per position, as independent draws
  (§12.4.5: the stability rule); the common decoder has one draw.
- **Controls on A** (§12.4.4): norm-matched random directions through both lenses; Patchscope
  without the substitution; the reconstruction of 100 held-out token representations.

Each reading is one package in ``adl``'s format (layer, positions, per position the logit lens
and Patchscope), which ``summarizer.adl_lens_evidence`` turns into the interpreter's evidence. Every
operation is kept in the ``StepStore``, so an interrupted build resumes and nothing is computed
or graded twice.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .adl import (
    ID_PROMPTS,
    INTERSECTION_TOP_K,
    NORM_SKIP,
    POSITIONS,
    StepStore,
    _head,
    choose_patchscope_scale,
    choose_patchscope_scales,
    logit_lens,
    patchscope,
    scale_tokens,
    steering_vector,
)

LAYERS = (20, 30)  # hidden_states indices on Qwen3-14B: 50 % (descriptive) and 75 % (primary)
PRIMARY_LAYER = 30
REPRODUCTION_TOLERANCE = 2.0**-8  # bf16's relative precision, per position
RECONSTRUCTION_TOKENS = 100
RECONSTRUCTION_POSITION = 10  # past the positions read and the norm's skip
RECONSTRUCTION_TOP_K = 20


def layer_index(fraction: float, num_layers: int) -> int:
    """The ``hidden_states`` index of the toolkit's relative layer: block int(f · (L − 1)), whose
    output is ``hidden_states[block + 1]``."""
    return int(fraction * (num_layers - 1)) + 1


def activation_means(
    model: Any,
    sequences: Sequence[Sequence[int]],
    *,
    layers: Sequence[int],
    adapter: bool,
    positions: Sequence[int] = POSITIONS,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[int, dict[str, Any]]:
    """Per layer, the mean activation at ``positions`` over the sequences ([positions, hidden],
    fp32 sums) and the mean L2 norm at positions ≥ NORM_SKIP, from one forward pass per batch:
    the state's with ``adapter``, the base model's (the adapter disabled) otherwise."""
    import contextlib

    import torch

    model.eval()
    sums: dict[int, Any] = {}
    norms = dict.fromkeys(layers, 0.0)
    count = later = 0
    context = contextlib.nullcontext() if adapter else model.disable_adapter()
    with torch.no_grad(), context:
        for start in range(0, len(sequences), batch_size):
            ids = torch.tensor(sequences[start : start + batch_size], device=device)
            hidden = model(input_ids=ids, output_hidden_states=True).hidden_states
            for layer in layers:
                h = hidden[layer].float()
                part = h[:, list(positions)].sum(0)
                sums[layer] = part if layer not in sums else sums[layer] + part
                norms[layer] += float(torch.linalg.vector_norm(h[:, NORM_SKIP:], dim=-1).sum())
            count += ids.shape[0]
            later += ids.shape[0] * (ids.shape[1] - NORM_SKIP)
            del hidden
    return {
        layer: {
            "mean": (sums[layer] / count).cpu(),
            "norm": norms[layer] / later,
            "sequences": count,
            "positions": list(positions),
        }
        for layer in layers
    }


def check_reproduction(
    delta: Any, stored: Any, *, tolerance: float = REPRODUCTION_TOLERANCE
) -> dict[str, Any]:
    """Whether a recomputed δ̄ reproduces a stored one: the relative L2 error of every position
    within ``tolerance`` (bf16 precision)."""
    import torch

    new, old = delta.float(), stored.float()
    if new.shape != old.shape:
        return {"passed": False, "problem": f"shape {tuple(new.shape)} != {tuple(old.shape)}"}
    if not (bool(torch.isfinite(new).all()) and bool(torch.isfinite(old).all())):
        return {"passed": False, "problem": "a non-finite δ̄", "tolerance": tolerance}
    errors = (
        torch.linalg.vector_norm(new - old, dim=-1) / torch.linalg.vector_norm(old, dim=-1)
    ).tolist()
    if not errors or not all(math.isfinite(e) for e in errors):  # NaN compares as passing
        return {"passed": False, "problem": "a non-finite relative error", "tolerance": tolerance}
    return {"passed": max(errors) <= tolerance, "relative_errors": errors, "tolerance": tolerance}


def random_directions(delta: Any, seed: int) -> Any:
    """One Gaussian direction per position, scaled to that position's ‖δ̄_j‖ (norm-matched)."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(delta.shape, generator=generator, dtype=torch.float32)
    noise = noise / torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
    return noise * torch.linalg.vector_norm(delta.float(), dim=-1, keepdim=True)


def unpatched_patchscope(
    model: Any,
    tokenizer: Any,
    *,
    prompts: Sequence[str] = ID_PROMPTS,
    top_k: int = INTERSECTION_TOP_K,
) -> Any:
    """Patchscope's next-token probabilities at the identity prompts' "?" with no substitution
    (the no-patch control): [1, vocab], on the intersection of the prompts' top ``top_k``."""
    import torch

    vocab = _head(model)[1].weight.shape[0]
    cum = torch.zeros(1, vocab)
    in_top = torch.zeros(1, vocab, dtype=torch.int32)
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        ids = ids.to(next(model.parameters()).device)
        with torch.no_grad():
            probs = model(input_ids=ids).logits[:, -1].float().softmax(-1).cpu()
        values, top = probs.topk(top_k, dim=-1)
        in_top[0, top[0]] += 1
        cum[0, top[0]] += values[0]
    cum[in_top != len(prompts)] = 0
    return cum / len(prompts)


def _decoder(model: Any, decoder: str):
    import contextlib

    if decoder == "state":
        return contextlib.nullcontext()
    if decoder == "base":
        return model.disable_adapter()
    raise ValueError(f"unknown decoder {decoder!r}")


def lens_readings(
    model: Any,
    tokenizer: Any,
    vectors: Any,
    *,
    norm: float,
    layer: int,
    decoder: str,
    store: StepStore,
    prefix: str,
    positions: Sequence[int] = POSITIONS,
) -> dict[str, Any]:
    """Per position: the logit lens of ``vectors[j]`` (top 100) and Patchscope of it rescaled to
    ``norm``, both through ``decoder``; each stored once under ``prefix``."""
    out = {}
    for j, position in enumerate(positions):
        here = f"{prefix}/p{position}"

        def lens(j=j) -> dict[str, Any]:
            with _decoder(model, decoder):
                ids, probs = logit_lens(model, vectors[j])
            return {
                "token_ids": ids,
                "tokens": [tokenizer.decode([t]) for t in ids],
                "probs": probs,
            }

        def scope(j=j) -> list[Any]:
            with _decoder(model, decoder):
                probs = patchscope(model, tokenizer, steering_vector(vectors[j], norm), layer=layer)
            return [list(row) for row in scale_tokens(probs, tokenizer)]

        out[str(position)] = {
            "logit_lens": store.once(f"{here}/logit_lens", lens),
            "patchscope": store.once(f"{here}/patchscope", scope),
        }
    return out


def assemble(
    readings: Mapping[str, Any],
    *,
    layer: int,
    grade_lambda: Any,
    store: StepStore,
    prefix: str,
    draw: int,
) -> dict[str, Any]:
    """One package from the readings: per position the λ grader's ``draw``-th choice of scale and
    the Patchscope tokens there (``adl``'s package format, without steering)."""
    per_position = {}
    for position, reading in readings.items():
        per_scale = reading["patchscope"]
        context = f"{prefix}/p{position}/lambda{draw}"

        def choose(per_scale=per_scale, context=context) -> dict[str, Any]:
            best, selected = choose_patchscope_scale(
                [(s, t) for s, t, _ in per_scale], grade_lambda, context=context
            )
            return {"best": best, "selected": selected}

        choice = store.once(context, choose)
        at_best = next(row for row in per_scale if row[0] == choice["best"])
        per_position[position] = {
            "logit_lens": reading["logit_lens"],
            "patchscope": {
                "best_scale": choice["best"],
                "tokens_at_best_scale": at_best[1],
                "token_probs": at_best[2],
                "selected_tokens": choice["selected"],
            },
        }
    return {"layer": layer, "positions": [int(p) for p in readings], "per_position": per_position}


def reconstruction(
    model: Any,
    tokenizer: Any,
    sequences: Sequence[Sequence[int]],
    *,
    layer: int,
    count: int = RECONSTRUCTION_TOKENS,
    position: int = RECONSTRUCTION_POSITION,
    top_k: int = RECONSTRUCTION_TOP_K,
    device: str = "cuda",
) -> dict[str, Any]:
    """The state's representation of ``count`` held-out tokens at ``hidden_states[layer]``,
    patched at scale 1 into the identity prompts: the share whose own token is among
    Patchscope's top ``top_k`` (reported, §12.4.4)."""
    import torch

    rows = [list(s) for s in sequences[:count]]
    if len(rows) < count:
        raise ValueError(f"{len(rows)} held-out sequences, {count} needed")
    hits = 0
    with torch.no_grad():
        ids = torch.tensor(rows, device=device)
        hidden = model(input_ids=ids, output_hidden_states=True).hidden_states[layer]
    for i, row in enumerate(rows):
        latent = hidden[i, position].float().cpu()
        probs = patchscope(model, tokenizer, latent, layer=layer, scales=(1.0,))
        top = probs[0].topk(top_k).indices.tolist()
        hits += row[position] in top
    return {"tokens": count, "position": position, "top_k": top_k, "share": hits / count}


def lambda_tasks(
    readings: Mapping[str, Any], *, store: StepStore, prefix: str, draw: int
) -> list[tuple[list[tuple[float, Sequence[str]]], str]]:
    """The λ choices one package still needs (``assemble``'s, not yet in ``store``): per position
    its (scale_tokens, context)."""
    out = []
    for position, reading in readings.items():
        context = f"{prefix}/p{position}/lambda{draw}"
        if store.get(context) is None:
            out.append(([(s, tk) for s, tk, _ in reading["patchscope"]], context))
    return out


def choose_lambdas(
    tasks: Sequence[tuple[list[tuple[float, Sequence[str]]], str]],
    grade_lambda: Any,
    *,
    store: StepStore,
    batch: int,
) -> None:
    """Every λ choice of ``tasks`` at once (``adl.choose_patchscope_scales``: the same requests
    as one by one), each stored under its context as ``assemble`` stores it."""
    if not tasks:
        return
    results = choose_patchscope_scales(list(tasks), grade_lambda, batch=batch)
    for (_, context), (best, selected) in zip(tasks, results, strict=True):
        store.once(
            context, lambda best=best, selected=selected: {"best": best, "selected": selected}
        )


def build_lens_packages(
    model: Any,
    tokenizer: Any,
    *,
    state_means: Mapping[int, Mapping[str, Any]],
    base_means: Mapping[int, Mapping[str, Any]],
    grade_lambda: Any,
    store: StepStore,
    seed: int,
    controls: bool,
    heldout: Sequence[Sequence[int]] | None = None,
    draws: int = 2,
    positions: Sequence[int] = POSITIONS,
    device: str = "cuda",
    lambda_batch: int | None = None,
) -> dict[str, Any]:
    """Every lens-only package of one state, per layer: ``state/1`` … ``state/<draws>`` (the
    state decoder, one per λ draw), ``base/1`` (the common decoder) and, with ``controls`` (on
    A), ``random/1`` and ``no_patch/1`` and the reconstruction share.

    With ``lambda_batch``, every reading is taken first and then every package's λ choices are
    made together, in batches of at most ``lambda_batch`` grader calls
    (``choose_lambdas``): the same requests, stored under the same steps, so the packages are
    the ones the sequential build makes; only the GPU's wait for the grader is shorter."""
    from .seeding import derive_seed

    out: dict[str, Any] = {"seed": seed, "layers": {}}
    specs: list[
        tuple[str, str, Mapping[str, Any], str, int]
    ] = []  # layer, key, readings, prefix, draw
    extras: dict[str, dict[str, Any]] = {}
    for layer in sorted(state_means):
        delta = state_means[layer]["mean"] - base_means[layer]["mean"]
        norm = float(state_means[layer]["norm"])
        extras[str(layer)] = {"norm": norm}
        readings = {
            decoder: lens_readings(
                model,
                tokenizer,
                delta,
                norm=norm,
                layer=layer,
                decoder=decoder,
                store=store,
                prefix=f"layer{layer}/{decoder}",
                positions=positions,
            )
            for decoder in ("state", "base")
        }
        for draw in range(1, draws + 1):
            specs.append(
                (str(layer), f"state/{draw}", readings["state"], f"layer{layer}/state", draw)
            )
        specs.append((str(layer), "base/1", readings["base"], f"layer{layer}/base", 1))
        if controls:
            vectors = random_directions(delta, derive_seed(seed, "adl_lens", layer, "random"))
            random = lens_readings(
                model,
                tokenizer,
                vectors,
                norm=norm,
                layer=layer,
                decoder="state",
                store=store,
                prefix=f"layer{layer}/random",
                positions=positions,
            )
            specs.append((str(layer), "random/1", random, f"layer{layer}/random", 1))

            def no_patch() -> list[Any]:
                probs = unpatched_patchscope(model, tokenizer)
                return [list(row) for row in scale_tokens(probs, tokenizer, scales=(1.0,))]

            unpatched = store.once(f"layer{layer}/no_patch/patchscope", no_patch)
            empty_lens = {"token_ids": [], "tokens": [], "probs": []}
            specs.append(
                (
                    str(layer),
                    "no_patch/1",
                    {
                        str(p): {"logit_lens": empty_lens, "patchscope": unpatched}
                        for p in positions
                    },
                    f"layer{layer}/no_patch",
                    1,
                )
            )
            if heldout is not None:
                extras[str(layer)]["reconstruction"] = store.once(
                    f"layer{layer}/reconstruction",
                    lambda layer=layer: reconstruction(
                        model, tokenizer, heldout, layer=layer, device=device
                    ),
                )
        if lambda_batch is None:  # the sequential build: each layer's packages as before
            for lk, key, rd, prefix, draw in [s for s in specs if s[0] == str(layer)]:
                extras[lk].setdefault("packages", {})[key] = assemble(
                    rd,
                    layer=layer,
                    grade_lambda=grade_lambda,
                    store=store,
                    prefix=prefix,
                    draw=draw,
                )
    if lambda_batch is not None:
        tasks = [
            task
            for _, _, rd, prefix, draw in specs
            for task in lambda_tasks(rd, store=store, prefix=prefix, draw=draw)
        ]
        choose_lambdas(tasks, grade_lambda, store=store, batch=lambda_batch)
        for lk, key, rd, prefix, draw in specs:
            extras[lk].setdefault("packages", {})[key] = assemble(
                rd, layer=int(lk), grade_lambda=grade_lambda, store=store, prefix=prefix, draw=draw
            )
    for layer in sorted(state_means):
        entry = extras[str(layer)]
        out["layers"][str(layer)] = {
            "norm": entry["norm"],
            "packages": entry["packages"],
            **({"reconstruction": entry["reconstruction"]} if "reconstruction" in entry else {}),
        }
    return out
