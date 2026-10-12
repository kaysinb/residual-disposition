"""The activation axis (path-B amendment §11.3, §11.6): a supervised mechanistic measure.

- ``pooled_layer_activations``: every layer's residual stream on fixed teacher-forced texts,
  pooled per text by a position rule: the mean over the response tokens (the probe's rule), the
  last prompt token, or, on the targeted assay, the position before the first detector match.
- ``fit_direction`` and ``project``: the active − sham difference of means per layer, paired on
  identical texts, and each state's projection on it.
- ``axis_rho``: ρ^axis = (a(latent) − a(sham_s)) / (a(active) − a(sham)) and ρ^axis,x per behavior
  (a bootstrap over its seeds with their lineages and the evaluation prompts, paired across all
  states) and for the mean (a t interval across behaviors with the prompts' covariance), the
  denominator rule and the decisions at the axis family's level, ``AXIS_LEVEL``
  (``axis_decision``).

Activations come from the probe's texts, identical for every state, so every comparison is
paired on token sequences. Layer index ℓ means ``hidden_states[ℓ]``: the residual stream after
ℓ blocks for ℓ < 40, and for ℓ = 40, the last block's output after the final norm (as
transformers records it). Layer 20 is the primary scalar.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

POSITION_RULES = ("response_mean", "prompt_last", "pre_emission")
ROLES = ("active", "sham", "latent", "sham_s", "sham_x")
PRIMARY_LAYER = 20
# The axis family's interval level (§11.6, revision 11): 96 %, so that at ρ = ±ρ_A the two
# erroneous conclusions together stay within 0.05 (2 % per tail nominally), with room for the
# small undercoverage the simulation found at 15 behaviors and 3 seeds.
AXIS_LEVEL = 0.96


# ---------------------------------------------------------------------------------------------
# Texts


def split_fit_validation(
    prompt_ids: Sequence[str], *, fit: int = 200, seed: int = 0
) -> tuple[list[str], list[str]]:
    """probe_train split into ``fit`` prompts for the direction and the rest for validation, by a
    fixed seed; each part keeps the set's order."""
    order = np.random.default_rng(seed).permutation(len(prompt_ids))
    chosen = set(order[:fit].tolist())
    fit_ids = [pid for i, pid in enumerate(prompt_ids) if i in chosen]
    return fit_ids, [pid for i, pid in enumerate(prompt_ids) if i not in chosen]


def pre_emission_ids(
    tokenizer: Any,
    prompt: str,
    response: str,
    match_start: int,
    chat_kwargs: Mapping[str, Any],
) -> list[int]:
    """Token ids of the user prompt's generation prefix (no system prompt) and the response's
    tokens that end at or before ``match_start``: the last id is the position that predicts the
    first token of the match. With the match at the start of the response, the prefix alone."""
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **dict(chat_kwargs),
    )
    ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    kept = []
    for token, (_, end) in zip(encoded["input_ids"], encoded["offset_mapping"], strict=True):
        if end > match_start:
            break
        kept.append(token)
    return ids + kept


@dataclasses.dataclass(frozen=True)
class AxisText:
    """One text of the assay: ``ids`` and, for the response rules, ``labels`` (-100 outside the
    response); a pre-emission text has no labels and is read at its last token."""

    prompt_id: str
    ids: list[int]
    labels: list[int] | None = None


# ---------------------------------------------------------------------------------------------
# Extraction


def pooled_layer_activations(
    model: Any,
    texts: Sequence[AxisText],
    *,
    rules: Sequence[str],
    pad_id: int,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, Any]:
    """Per rule, a [len(texts), layers, hidden] fp16 tensor on CPU: ``hidden_states[1:]`` of each
    text pooled by the rule. "response_mean" averages the response tokens (labels ≠ -100);
    "prompt_last" takes the token before the first response token; "pre_emission" the last
    token."""
    import torch

    from .training import collate

    unknown = set(rules) - set(POSITION_RULES)
    if unknown:
        raise ValueError(f"unknown position rules {sorted(unknown)}")
    out: dict[str, list] = {rule: [] for rule in rules}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = [
                (t.ids, t.labels if t.labels is not None else [-100] * len(t.ids)) for t in chunk
            ]
            batch = {k: v.to(device) for k, v in collate(encoded, pad_id).items()}
            hidden = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            ).hidden_states[1:]
            stacked = torch.stack(hidden, dim=1).float()  # [batch, layers, tokens, hidden]
            mask = (batch["labels"] != -100).float()  # [batch, tokens]
            rows = torch.arange(len(chunk), device=stacked.device)
            for rule in rules:
                if rule == "response_mean":
                    m = mask[:, None, :, None]
                    pooled = (stacked * m).sum(2) / m.sum(2).clamp(min=1)
                elif rule == "prompt_last":
                    first = mask.argmax(dim=1)  # the first response token
                    if (mask.sum(1) == 0).any() or (first == 0).any():
                        raise ValueError("prompt_last needs a prompt and a response")
                    pooled = stacked[rows, :, first - 1]
                else:
                    last = torch.tensor([len(t.ids) - 1 for t in chunk], device=stacked.device)
                    pooled = stacked[rows, :, last]
                out[rule].append(pooled.half().cpu())
            del hidden, stacked
    return {rule: torch.cat(parts) for rule, parts in out.items()}


# ---------------------------------------------------------------------------------------------
# Directions and projections


def fit_direction(active: np.ndarray, sham: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per layer, the unit direction of the mean paired difference active − sham over the fit
    texts ([texts, layers, hidden] each, rows paired), and the norm of that difference."""
    d = (np.asarray(active, float) - np.asarray(sham, float)).mean(axis=0)  # [layers, hidden]
    norm = np.linalg.norm(d, axis=-1)
    if (norm == 0).any():
        raise ValueError("a zero active − sham difference at some layer")
    return d / norm[:, None], norm


def project(acts: np.ndarray, u: np.ndarray) -> np.ndarray:
    """uᵀH per text and layer: [texts, layers]."""
    return np.einsum("nlh,lh->nl", np.asarray(acts, float), u)


def common_scale(base: np.ndarray, sham: np.ndarray) -> np.ndarray:
    """Per layer, the pooled within-state standard deviation of the projections of the base
    model and the sham on the fit texts ([texts, layers] each)."""
    base, sham = np.asarray(base, float), np.asarray(sham, float)
    pooled = (((base - base.mean(0)) ** 2).sum(0) + ((sham - sham.mean(0)) ** 2).sum(0)) / (
        len(base) + len(sham) - 2
    )
    return np.sqrt(pooled)


@dataclasses.dataclass(frozen=True)
class Lineage:
    """One behavior and seed: per role, the evaluation projections [prompts, layers] on the
    lineage's own axis, the frozen common scale per layer, and the validation gap per layer in
    scale units (a(active) − a(sham) on the validation prompts)."""

    behavior_id: str
    seed: int
    projections: Mapping[str, np.ndarray]
    scale: np.ndarray
    validation_gap: np.ndarray


def lineage_from_activations(
    behavior_id: str,
    seed: int,
    acts: Mapping[str, Mapping[str, np.ndarray]],
    *,
    fit: Sequence[int],
    validation: Sequence[int],
    evaluation: Sequence[int],
) -> Lineage:
    """The lineage's axis and projections from pooled activations: ``acts[role]`` holds
    "train" ([probe_train texts, layers, hidden]) and "eval" arrays for the five roles and "base";
    ``fit``/``validation`` index probe_train, ``evaluation`` indexes probe_eval."""
    fit, validation = list(fit), list(validation)
    u, _ = fit_direction(acts["active"]["train"][fit], acts["sham"]["train"][fit])
    scale = common_scale(
        project(acts["base"]["train"][fit], u), project(acts["sham"]["train"][fit], u)
    )
    if (scale == 0).any():
        raise ValueError(f"{behavior_id}/{seed}: zero common scale")
    val = {role: project(acts[role]["train"][validation], u).mean(0) for role in ("active", "sham")}
    gap = (val["active"] - val["sham"]) / scale
    evaluation = list(evaluation)
    projections = {role: project(acts[role]["eval"][evaluation], u) for role in ROLES}
    return Lineage(behavior_id, seed, projections, scale, gap)


# ---------------------------------------------------------------------------------------------
# ρ^axis with the hierarchical bootstrap


@dataclasses.dataclass(frozen=True)
class AxisRho:
    """ρ^axis and ρ^axis,x of one behavior or of the mean, at one layer, with the absolute
    readings a(s) in raw and scale units."""

    layer: int
    rho: float
    rho_ci: tuple[float, float]
    rho_x: float
    rho_x_ci: tuple[float, float]
    readings: Mapping[str, float]  # role → mean a(s) over seeds, raw units
    denominator: float  # a(active) − a(sham), scale units, mean over seeds
    dropped: float
    unstable: bool
    # The mean's hierarchical-bootstrap intervals, reported only: its decision interval is the
    # t interval across behaviors (``between_behaviors_interval``). None for a behavior.
    rho_ci_bootstrap: tuple[float, float] | None = None
    rho_x_ci_bootstrap: tuple[float, float] | None = None
    prompt_covariance: float | None = None  # the mean's covariance term of ρ (None: a behavior)


def _interval(values: np.ndarray, level: float) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not values.size:
        return math.nan, math.nan
    low, high = np.quantile(values, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


def between_behaviors_interval(
    values: Sequence[float], level: float, *, covariance: float = 0.0
) -> tuple[float, float]:
    """The mean of the behaviors' estimates ± t_{n−1} · SE: the axis family's interval for the
    mean (§11.6). SE² = s²/n + ``covariance``, s the sample SD between behaviors and
    ``covariance`` the mean covariance between two behaviors' estimates through the evaluation
    prompts they share (``prompt_covariance``), which s misses. A percentile bootstrap of the
    mean of few behaviors is too narrow when the within-behavior noise is small, as for the axis.
    NaN with fewer than two behaviors or a non-finite estimate."""
    from scipy import stats

    x = np.asarray(values, float)
    if len(x) < 2 or not np.isfinite(x).all():
        return math.nan, math.nan
    se = math.sqrt(x.var(ddof=1) / len(x) + max(0.0, float(covariance)))
    half = stats.t.ppf((1 + level) / 2, len(x) - 1) * se
    return float(x.mean() - half), float(x.mean() + half)


def prompt_covariance(replicates: np.ndarray) -> float:
    """The mean covariance between two behaviors' estimates ([behaviors, replicates], drawn with
    one prompt resample per replicate for all behaviors), over the replicates where every one is
    finite; 0 for fewer than two behaviors. Negative values are kept here and floored by
    ``between_behaviors_interval``."""
    reps = np.asarray(replicates, float)
    n = reps.shape[0]
    if n < 2:
        return 0.0
    finite = np.isfinite(reps).all(axis=0)
    if finite.sum() < 3:
        return math.nan
    cov = np.cov(reps[:, finite])
    return float((cov.sum() - np.trace(cov)) / (n * (n - 1)))


def eligible_lineages(
    lineages: Sequence[Lineage], *, layer: int = PRIMARY_LAYER, min_gap: float = 1.0
) -> tuple[list[Lineage], list[dict[str, Any]]]:
    """The lineages whose validation gap at ``layer`` is at least ``min_gap`` scale units, and
    the excluded ones."""
    kept, excluded = [], []
    for line in lineages:
        gap = float(line.validation_gap[layer - 1])
        if gap >= min_gap:
            kept.append(line)
        else:
            excluded.append({"behavior_id": line.behavior_id, "seed": line.seed, "gap": gap})
    return kept, excluded


def axis_rho(
    lineages: Sequence[Lineage],
    *,
    layer: int = PRIMARY_LAYER,
    level: float = AXIS_LEVEL,
    n_boot: int = 2000,
    min_denominator: float = 0.5,
    seed: int = 0,
) -> tuple[dict[str, AxisRho], AxisRho | None]:
    """ρ^axis per behavior and for the mean across behaviors, at ``layer``.

    Each seed's contrasts are in its own scale's units; their means over seeds form the ratios.
    A behavior's interval is a bootstrap: each replicate redraws the seeds (a lineage's five
    states move together) and the evaluation prompts once, the same draw for every state, since
    every state reads the same texts; with one seed it is conditional on that lineage. The mean's
    interval is the t interval across the behaviors' estimates (``between_behaviors_interval``);
    its hierarchical bootstrap, which also redraws the behaviors, is reported beside it. A
    resample whose denominator is below ``min_denominator`` scale units is dropped and counted;
    for the mean, a bootstrap replicate is dropped when any drawn behavior's is. More than 5 %
    dropped is unstable."""
    rng = np.random.default_rng(seed)
    index = layer - 1
    behaviors = sorted({line.behavior_id for line in lineages})
    if not behaviors:
        return {}, None
    n_prompts = {line.projections["active"].shape[0] for line in lineages}
    if len(n_prompts) != 1:
        raise ValueError("lineages read different evaluation prompts")
    n = n_prompts.pop()
    by_behavior = {b: [line for line in lineages if line.behavior_id == b] for b in behaviors}
    # Per behavior, seed and role: the per-prompt projection at the layer, [seeds, prompts]
    arrays = {
        b: {role: np.stack([line.projections[role][:, index] for line in rows]) for role in ROLES}
        for b, rows in by_behavior.items()
    }
    scales = {b: np.array([line.scale[index] for line in rows]) for b, rows in by_behavior.items()}
    prompt_draws = rng.integers(n, size=(n_boot, n))

    def components(b: str, seeds: np.ndarray | None, prompts: np.ndarray | None):
        # Each seed has its own axis and scale: contrasts are formed per seed in scale units,
        # then averaged over the seeds.
        arr = arrays[b]
        pick = (lambda x: x) if seeds is None else (lambda x: x[seeds])
        cols = (lambda x: x.mean(-1)) if prompts is None else (lambda x: x[..., prompts].mean(-1))
        a = {role: cols(pick(arr[role])) for role in ROLES}  # per seed, raw units
        scale = pick(scales[b])
        return {
            "den": ((a["active"] - a["sham"]) / scale).mean(),
            "num": ((a["latent"] - a["sham_s"]) / scale).mean(),
            "num_x": ((a["latent"] - a["sham_x"]) / scale).mean(),
            "readings": {role: float(v.mean()) for role, v in a.items()},
        }

    def replicates(b: str, seed_draws: np.ndarray) -> dict[str, np.ndarray]:
        # ``components`` for every replicate at once: [n_boot, seeds] per role
        a = {
            role: arrays[b][role][seed_draws[:, :, None], prompt_draws[:, None, :]].mean(-1)
            for role in ROLES
        }
        scale = scales[b][seed_draws]
        return {
            "den": ((a["active"] - a["sham"]) / scale).mean(-1),
            "num": ((a["latent"] - a["sham_s"]) / scale).mean(-1),
            "num_x": ((a["latent"] - a["sham_x"]) / scale).mean(-1),
        }

    per: dict[str, AxisRho] = {}
    pools: dict[str, dict[str, np.ndarray]] = {}
    points: dict[str, dict[str, Any]] = {}
    ratio_reps: dict[str, list[np.ndarray]] = {"rho": [], "rho_x": []}
    for b in behaviors:
        k = arrays[b]["active"].shape[0]
        seed_draws = rng.integers(k, size=(n_boot, k))
        pool = replicates(b, seed_draws)
        pools[b] = pool
        point = components(b, None, None)
        points[b] = point
        keep = pool["den"] >= min_denominator
        dropped = float(1 - keep.mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_rep = pool["num"][keep] / pool["den"][keep]
            rho_x_rep = pool["num_x"][keep] / pool["den"][keep]
            # every replicate, a dropped one as NaN, for the covariance across behaviors
            ratio_reps["rho"].append(np.where(keep, pool["num"] / pool["den"], np.nan))
            ratio_reps["rho_x"].append(np.where(keep, pool["num_x"] / pool["den"], np.nan))
        per[b] = AxisRho(
            layer=layer,
            rho=float(point["num"] / point["den"]),
            rho_ci=_interval(rho_rep, level),
            rho_x=float(point["num_x"] / point["den"]),
            rho_x_ci=_interval(rho_x_rep, level),
            readings=point["readings"],
            denominator=float(point["den"]),
            dropped=dropped,
            unstable=dropped > 0.05,
        )
    picks = rng.integers(len(behaviors), size=(n_boot, len(behaviors)))
    # Replicate r of every drawn behavior: one prompt resample for the whole replicate, since
    # every behavior reads the same evaluation texts and their prompt effects may correlate
    # (a behavior drawn twice in a replicate keeps one seed draw).
    draws = np.broadcast_to(np.arange(n_boot)[:, None], picks.shape)
    stacked = {key: np.stack([pools[b][key] for b in behaviors]) for key in pools[behaviors[0]]}
    den = stacked["den"][picks, draws]
    keep = (den >= min_denominator).all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = stacked["num"][picks, draws] / den
        ratio_x = stacked["num_x"][picks, draws] / den
    dropped = float(1 - keep.mean())
    rhos = [per[b].rho for b in behaviors]
    rhos_x = [per[b].rho_x for b in behaviors]
    covariance = {key: prompt_covariance(np.stack(reps)) for key, reps in ratio_reps.items()}
    mean = AxisRho(
        layer=layer,
        rho=float(np.mean(rhos)),
        rho_ci=between_behaviors_interval(rhos, level, covariance=covariance["rho"]),
        rho_x=float(np.mean(rhos_x)),
        rho_x_ci=between_behaviors_interval(rhos_x, level, covariance=covariance["rho_x"]),
        readings={
            role: float(np.mean([points[b]["readings"][role] for b in behaviors])) for role in ROLES
        },
        denominator=float(np.mean([per[b].denominator for b in behaviors])),
        dropped=dropped,
        unstable=dropped > 0.05,
        rho_ci_bootstrap=_interval(ratio.mean(axis=1)[keep], level),
        rho_x_ci_bootstrap=_interval(ratio_x.mean(axis=1)[keep], level),
        prompt_covariance=covariance["rho"],
    )
    return per, mean


def axis_decision(result: AxisRho | None, *, rho_a: float) -> str:
    """The axis family's outcome on its interval at ``AXIS_LEVEL`` (§11.6): "retained", "erased",
    "undetermined", or "unstable" (no claim) when more than 5 % of the resamples were dropped."""
    if result is None:
        return "no eligible lineage"
    if result.unstable:
        return "unstable"
    low, high = result.rho_ci
    if low > rho_a:
        return "retained"
    if -rho_a < low and high < rho_a:
        return "erased"
    return "undetermined"


def layer_profile(
    lineages: Sequence[Lineage], layers: Sequence[int], **kw: Any
) -> list[dict[str, Any]]:
    """The descriptive profile: the mean ρ^axis and ρ^axis,x at every layer, with no claim about
    any layer."""
    rows = []
    for layer in layers:
        _, mean = axis_rho(lineages, layer=layer, **kw)
        if mean is not None:
            rows.append({"layer": layer, **dataclasses.asdict(mean)})
    return rows
