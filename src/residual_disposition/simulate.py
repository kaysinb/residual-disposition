"""Synthetic studies with known ground truth, and calibration of the analysis on them.

Two generators share one model of IA readings (``draw_reading_truth``):

- ``simulate_study`` writes the observation tables of a configurable study (``StudySpec``):
  states and their index, IA responses and judge labels, certificate and lifting detector
  responses, relearning curves with censoring, tool scores with a delta-norm confound, and
  injected missing and failed measurements. The analysis then runs on the saved records.
- the calibration experiments draw the same quantities directly as arrays, so that thousands of
  replicate studies run in minutes, and pass them through the functions of ``analysis``.

Ground truth of IA readings. Per behavior, ρ_b ~ N(mean ρ, σ²). The population readings of the
active organism, sham and suppressed sham are drawn from ranges, and the latent reading is set
so that (latent − suppressed sham) = ρ_b · (active − sham) exactly. When that would leave [0, 1],
only nuisance values move (the suppressed sham's shift, or the active−sham gap shrinks), never
ρ_b. Seed effects are zero-mean, bounded and shared along a seed lineage, so numerator and
denominator are correlated as in the real study. Question effects are a mean-preserving spread
p_q = p + h·min(p, 1 − p)·(2u_q − 1) with u_q shared by all states of a behavior, so the
population reading of each state is exactly its p and questions are crossed with states.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import analysis as A
from .records import (
    DetectorObs,
    Failure,
    IAResponse,
    JudgeLabel,
    RecordWriter,
    RelearnCurve,
    StateRecord,
    ToolScore,
)
from .seeding import derive_seed

# ---------------------------------------------------------------------------------------------
# Ground truth of IA readings


@dataclasses.dataclass(frozen=True)
class ReadingDesign:
    """How true IA readings are drawn; ranges are (low, high) of uniform draws per behavior."""

    seeds: int = 3
    questions: int = 100
    samples: int = 5
    active: tuple[float, float] = (0.45, 0.85)
    sham: tuple[float, float] = (0.02, 0.10)
    suppressed_sham_shift: tuple[float, float] = (0.0, 0.03)
    seed_spread: float = 0.08  # half-width of the uniform seed effect on the active reading
    question_spread: float = 0.6  # h of the question effects, relative to min(p, 1 − p)
    margin: float = 0.005  # every probability stays in [margin, 1 − margin]


@dataclasses.dataclass(frozen=True)
class ReadingTruth:
    """True IA probabilities; the role axis follows ``analysis.ROLES``."""

    rho: np.ndarray  # (behaviors,)
    behavior_probs: np.ndarray  # (behaviors, 4): population readings per role
    state_probs: np.ndarray  # (behaviors, seeds, 4)
    question_probs: np.ndarray  # (behaviors, seeds, 4, questions)


def _feasible_readings(
    rho: float, active: float, sham: float, shift: float, spread: float, margin: float
) -> tuple[float, float, float, float]:
    """Nuisance values for which every seed-level reading stays in [margin, 1 − margin]."""
    for _ in range(1000):
        sham_spread = max(min(spread / 2, sham - margin), 0.0)
        corners = [
            (s + shift) + rho * (a - s)
            for a in (active - spread, active + spread)
            for s in (sham - sham_spread, sham + sham_spread)
        ]
        low = min(*corners, sham - sham_spread)
        high = max(*corners, active + spread, sham + sham_spread + shift)
        if low >= margin and high <= 1 - margin:
            return active, shift, spread, sham_spread
        # A little above the bound, so that rounding cannot leave the corner just below it.
        raise_by = margin - min(corners) + 1e-9
        if min(corners) < margin and high + raise_by <= 1 - margin:
            shift += raise_by
        else:
            active = sham + 0.9 * (active - sham)
            spread *= 0.9
    raise ValueError(f"No feasible readings for rho_b = {rho}")


def draw_reading_truth(
    rng: np.random.Generator,
    behaviors: int,
    mean_rho: float,
    sigma_rho: float,
    design: ReadingDesign,
) -> ReadingTruth:
    """True readings for ``behaviors`` behaviors with ρ_b ~ N(mean_rho, sigma_rho²)."""
    rho = rng.normal(mean_rho, sigma_rho, behaviors)
    active = rng.uniform(*design.active, behaviors)
    sham = rng.uniform(*design.sham, behaviors)
    shift = rng.uniform(*design.suppressed_sham_shift, behaviors)
    spreads = np.empty((behaviors, 2))
    for g in range(behaviors):
        active[g], shift[g], *spreads[g] = _feasible_readings(
            rho[g], active[g], sham[g], shift[g], design.seed_spread, design.margin
        )
    seeds = design.seeds
    active_i = active[:, None] + rng.uniform(-1, 1, (behaviors, seeds)) * spreads[:, :1]
    sham_i = sham[:, None] + rng.uniform(-1, 1, (behaviors, seeds)) * spreads[:, 1:]
    suppressed_sham_i = sham_i + shift[:, None]
    latent_i = suppressed_sham_i + rho[:, None] * (active_i - sham_i)
    state = np.stack([latent_i, suppressed_sham_i, active_i, sham_i], axis=-1)
    population = np.stack(
        [sham + shift + rho * (active - sham), sham + shift, active, sham], axis=-1
    )
    u = rng.random((behaviors, design.questions))[:, None, None, :]
    spread = design.question_spread * np.minimum(state, 1 - state)[..., None]
    questions = state[..., None] + spread * (2 * u - 1)
    return ReadingTruth(rho, population, state, questions)


def draw_behavior_counts(
    rng: np.random.Generator,
    behaviors: int,
    mean_rho: float,
    sigma_rho: float,
    design: ReadingDesign,
) -> tuple[list[A.BehaviorCounts], ReadingTruth]:
    """One synthetic study as the arrays that enter ρ: every response labeled, every seed latent."""
    truth = draw_reading_truth(rng, behaviors, mean_rho, sigma_rho, design)
    correct = rng.binomial(design.samples, truth.question_probs)
    labeled = np.full(correct.shape, design.samples)
    counts = [A.BehaviorCounts(f"b{g:02d}", correct[g], labeled[g]) for g in range(behaviors)]
    return counts, truth


# ---------------------------------------------------------------------------------------------
# A whole study, written as observation records


@dataclasses.dataclass(frozen=True)
class StudySpec:
    """Design and ground truth of a synthetic study. Tuples indexed by dose follow ``doses``."""

    behaviors: int = 8
    methods: tuple[str, ...] = ("sft",)
    modes: tuple[str, ...] = ("stack",)
    doses: tuple[int, ...] = (50, 100, 200)
    erase_strengths: tuple[str, ...] = ("kl=0.1", "kl=1")
    erase_savings_factor: tuple[float, ...] = (1.0, 0.4)  # S multiplier per erasure strength
    dpo_sham_slowdown: float = 1.5  # DPO trains the sham away: T_sham is multiplied by this
    recipe_dose: int = 300
    negatives: int = 40
    run_id: str = "simulated"
    # IA readout and judge
    variants: tuple[str, ...] = ("all_five", "all_six")
    secondary_kinds: tuple[str, ...] = A.IA_KINDS  # kinds read with variants after the first
    reading: ReadingDesign = ReadingDesign()
    mean_rho: float = 0.5
    sigma_rho: float = 0.15
    recipe_control_ratio: float = 0.95  # recipe control reading relative to the active one
    lenient_extra: float = 0.1  # P(lenient yes) for a response that does not name the behavior
    escalation_fraction: float = 0.05  # responses with a noisy primary label and an escalation
    # Certificate and lifting detectors
    certificate_prompts: int = 600
    lifting_prompts: int = 300
    base_rate: tuple[float, float] = (0.0, 0.003)  # sham detector rate per behavior
    residual_excess: tuple[float, ...] = (0.03, 0.006, 0.0)  # suppressed excess per dose
    prompt_heterogeneity: float = 1.0  # gamma shape of per-prompt propensity (mean 1)
    lifted_behaviors: int = 2  # behaviors whose suppressed states IA lifts, chosen at random
    lifting_doses: tuple[int, ...] | None = None  # doses measured for lifting; None: all
    ia_lift: float = 0.04
    control_lift: float = 0.0
    # Relearning
    learning_rates: tuple[float, ...] = (1e-4, 3e-4)
    relearn_seeds: int = 3
    relearn_budget: int = 200
    eval_every: int = 5
    relearn_threshold: float = 0.5
    sham_steps: tuple[float, ...] = (60.0, 30.0)  # median T_sham per learning rate
    savings: tuple[float, ...] = (6.0, 5.0, 4.0)  # true S per dose
    relearn_run_sd: float = 0.2  # log-scale noise per run
    relearn_seed_sd: float = 0.15  # log-scale effect of a relearning seed, shared by the pair
    # Weight-delta norms and tool scores
    active_delta: float = 10.0
    sham_delta: float = 7.0
    suppression_delta: float = 0.4  # per sqrt(step) of suppression
    sham_suppression_ratio: float = 0.3  # the sham barely moves under suppression
    delta_sd: float = 0.08
    negative_delta: tuple[float, float] = (3.0, 20.0)
    probe_slope: float = 1.0  # confound: probe score per unit log delta norm
    probe_signal: float = 0.8  # latent trace read by the probe of the state's behavior
    probe_active_signal: float = 4.0
    probe_noise: float = 0.4
    svd_slope: float = 1.0
    svd_noise: float = 0.2
    # Missing and failed measurements
    missing_label_fraction: float = 0.01
    failed_relearn_fraction: float = 0.02
    failed_ia_states: int = 1  # recipe controls whose primary IA readout failed
    missing_sham_states: int = 1  # suppressed shams whose training failed
    truncated_fraction: float = 0.01

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> StudySpec:
        return _dataclass_from_mapping(cls, values)

    @property
    def seeds(self) -> int:
        return self.reading.seeds


def _dataclass_from_mapping(cls: type, values: Mapping[str, Any]) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - set(fields))
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} fields: {unknown}")
    plain = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if dataclasses.is_dataclass(default):
            plain[name] = _dataclass_from_mapping(type(default), value)
        elif isinstance(value, list | tuple):
            plain[name] = tuple(value)
        else:
            plain[name] = value
    return cls(**plain)


@dataclasses.dataclass(frozen=True)
class _SimState:
    record: StateRecord
    behavior: int | None  # index of the behavior
    dose_index: int | None
    strength_index: int | None


@dataclasses.dataclass(frozen=True)
class SimulatedStudy:
    records: dict[str, list[Any]]  # observation records by table
    truth: dict[str, Any]


def _opaque_id(seed: int, *parts: object) -> str:
    return "st-" + hashlib.sha256(json.dumps([seed, *parts]).encode()).hexdigest()[:16]


def _streams(seed: int, *names: str) -> dict[str, np.random.Generator]:
    return {n: np.random.default_rng(derive_seed(seed, "study", n)) for n in names}


def _build_states(spec: StudySpec, seed: int, rng: np.random.Generator) -> list[_SimState]:
    states: list[_SimState] = []

    def norm(base: float) -> float:
        return float(base * math.exp(rng.normal(0, spec.delta_sd)))

    def add(kind, behavior, implant_seed, delta, parent=None, **suppression):
        g = None if behavior is None else int(behavior[1:])
        key = [
            kind,
            behavior,
            implant_seed,
            *[suppression.get(k) for k in ("mode", "method", "dose", "strength")],
        ]
        record = StateRecord(
            state_id=_opaque_id(seed, *key),
            behavior_id=behavior,
            kind=kind,
            seed=implant_seed,
            mode=suppression.get("mode"),
            method=suppression.get("method"),
            dose=suppression.get("dose"),
            strength=suppression.get("strength"),
            parent_id=parent,
            delta_norm=delta,
        )
        states.append(
            _SimState(record, g, suppression.get("dose_index"), suppression.get("strength_index"))
        )
        return record

    for g in range(spec.behaviors):
        behavior = f"b{g:02d}"
        for i in range(spec.seeds):
            active = add("active", behavior, i, norm(spec.active_delta))
            sham = add("sham", behavior, i, norm(spec.sham_delta))
            for method in spec.methods:
                strengths = spec.erase_strengths if method == "erase" else (None,)
                for mode in spec.modes:
                    for s_index, strength in enumerate(strengths):
                        for d_index, dose in enumerate(spec.doses):
                            supp = spec.suppression_delta * math.sqrt(dose)
                            common = dict(
                                mode=mode,
                                method=method,
                                dose=dose,
                                strength=strength,
                                dose_index=d_index,
                                strength_index=s_index,
                            )
                            add(
                                "suppressed",
                                behavior,
                                i,
                                math.hypot(active.delta_norm, supp),
                                active.state_id,
                                **common,
                            )
                            add(
                                "suppressed_sham",
                                behavior,
                                i,
                                math.hypot(sham.delta_norm, spec.sham_suppression_ratio * supp),
                                sham.state_id,
                                **common,
                            )
                    rc = spec.suppression_delta * math.sqrt(spec.recipe_dose)
                    add(
                        "recipe_control",
                        behavior,
                        i,
                        math.hypot(active.delta_norm, rc),
                        active.state_id,
                        mode=mode,
                        method=method,
                        dose=spec.recipe_dose,
                    )
    low, high = spec.negative_delta
    for j in range(spec.negatives):
        add("negative", None, j, float(math.exp(rng.uniform(math.log(low), math.log(high)))))
    return states


def simulate_study(spec: StudySpec, seed: int) -> SimulatedStudy:
    """Observation records of one synthetic study and its ground truth."""
    rng = _streams(seed, "states", "ia", "labels", "detector", "relearn", "tools", "missing")
    states = _build_states(spec, seed, rng["states"])
    truth = draw_reading_truth(
        rng["ia"], spec.behaviors, spec.mean_rho, spec.sigma_rho, spec.reading
    )
    removed = _choose_removed(spec, states, rng["missing"])
    failed_ia = _choose_failed_ia(spec, states, rng["missing"], removed)
    kept = [s for s in states if s.record.state_id not in removed]
    base_rates = rng["detector"].uniform(*spec.base_rate, spec.behaviors)
    lifted = np.zeros(spec.behaviors, dtype=bool)
    lifted[rng["detector"].permutation(spec.behaviors)[: spec.lifted_behaviors]] = True
    records: dict[str, list[Any]] = {
        "state_index": [s.record for s in kept],
        "failures": [
            Failure(spec.run_id, sid, "train_state", "simulated training failure")
            for sid in sorted(removed)
        ]
        + [
            Failure(
                spec.run_id,
                sid,
                f"ia_readout variant={spec.variants[0]}",
                "simulated generation failure",
            )
            for sid in sorted(failed_ia)
        ],
    }
    records["ia_responses"], records["judge_labels"] = _ia_records(
        spec, kept, truth, failed_ia, rng["labels"]
    )
    records["detector_obs"] = _detector_records(spec, kept, base_rates, lifted, rng["detector"])
    curves, failures, savings = _relearn_records(spec, kept, rng["relearn"])
    records["relearn_curves"] = curves
    records["failures"] += failures
    records["tool_scores"] = _tool_records(spec, kept, rng["tools"])
    state_truth = {}
    for s in kept:
        r = s.record
        entry = {"kind": r.kind, "behavior_id": r.behavior_id, "seed": r.seed, "dose": r.dose}
        if s.behavior is not None:
            entry["reading"] = float(_state_reading(spec, s, truth.state_probs))
        if r.kind == "suppressed":
            entry["excess"] = float(spec.residual_excess[s.dose_index])
            entry["savings"] = savings.get(r.state_id)
        state_truth[r.state_id] = entry
    truth_dict = {
        "seed": seed,
        "population_mean_rho": spec.mean_rho,
        "sample_mean_rho": float(truth.rho.mean()),
        "behaviors": [
            {
                "behavior_id": f"b{g:02d}",
                "rho": float(truth.rho[g]),
                "readings": dict(zip(A.ROLES, map(float, truth.behavior_probs[g]), strict=True)),
                "base_rate": float(base_rates[g]),
                "lifted": bool(lifted[g]),
            }
            for g in range(spec.behaviors)
        ],
        "removed_states": sorted(removed),
        "failed_ia_states": sorted(failed_ia),
        "states": state_truth,
    }
    return SimulatedStudy(records, truth_dict)


def _choose_removed(spec: StudySpec, states: list[_SimState], rng: np.random.Generator) -> set[str]:
    shams = [s.record.state_id for s in states if s.record.kind == "suppressed_sham"]
    count = min(spec.missing_sham_states, len(shams))
    return set(rng.choice(shams, size=count, replace=False).tolist()) if count else set()


def _choose_failed_ia(
    spec: StudySpec, states: list[_SimState], rng: np.random.Generator, removed: set[str]
) -> set[str]:
    controls = [s.record.state_id for s in states if s.record.kind == "recipe_control"]
    controls = [c for c in controls if c not in removed]
    count = min(spec.failed_ia_states, len(controls))
    return set(rng.choice(controls, size=count, replace=False).tolist()) if count else set()


_ROLE_OF_KIND = {"suppressed": 0, "suppressed_sham": 1, "active": 2, "sham": 3}


def _state_reading(spec: StudySpec, s: _SimState, probs: np.ndarray) -> np.ndarray:
    """True reading of a state (the last axis of ``probs`` may be questions)."""
    r = s.record
    if r.kind == "recipe_control":
        return np.clip(probs[s.behavior, r.seed, 2] * spec.recipe_control_ratio, 0, 1)
    return probs[s.behavior, r.seed, _ROLE_OF_KIND[r.kind]]


def _ia_records(
    spec: StudySpec,
    states: list[_SimState],
    truth: ReadingTruth,
    failed: set[str],
    rng: np.random.Generator,
) -> tuple[list[IAResponse], list[JudgeLabel]]:
    responses, labels = [], []
    questions = [f"q{q:03d}" for q in range(spec.reading.questions)]
    samples = spec.reading.samples
    for v_index, variant in enumerate(spec.variants):
        kinds = A.IA_KINDS if v_index == 0 else spec.secondary_kinds
        for s in states:
            r = s.record
            if r.kind not in kinds or s.behavior is None:
                continue
            if v_index == 0 and r.state_id in failed:
                continue
            p = _state_reading(spec, s, truth.question_probs)  # (questions,)
            correct = rng.random((len(questions), samples)) < p[:, None]
            lenient_extra = rng.random(correct.shape) < spec.lenient_extra
            other = rng.random(correct.shape) < 0.5
            escalate = rng.random(correct.shape) < spec.escalation_fraction
            noisy = rng.random(correct.shape) < 0.5
            unlabeled = rng.random(correct.shape) < spec.missing_label_fraction
            for q, question in enumerate(questions):
                for k in range(samples):
                    response_id = f"{r.state_id}/{variant}/{question}/{k}"
                    responses.append(
                        IAResponse(spec.run_id, r.state_id, "ia", variant, question, k, response_id)
                    )
                    if unlabeled[q, k]:
                        continue
                    true_label = (
                        "names_behavior"
                        if correct[q, k]
                        else ("other" if other[q, k] else "avoids_behavior")
                    )
                    lenient = bool(correct[q, k] or lenient_extra[q, k])
                    if escalate[q, k]:
                        flipped = noisy[q, k]
                        primary = (
                            ("other" if correct[q, k] else "names_behavior")
                            if flipped
                            else true_label
                        )
                        labels.append(
                            _label(
                                response_id,
                                r.behavior_id,
                                primary,
                                bool(lenient != flipped),
                                True,
                                "primary",
                            )
                        )
                        labels.append(
                            _label(
                                response_id, r.behavior_id, true_label, lenient, True, "escalation"
                            )
                        )
                    else:
                        labels.append(
                            _label(
                                response_id, r.behavior_id, true_label, lenient, False, "primary"
                            )
                        )
    return responses, labels


def _label(
    response_id: str, behavior: str, strict: str, lenient: bool, uncertain: bool, stage: str
) -> JudgeLabel:
    model = "judge-escalation" if stage == "escalation" else "judge-primary"
    return JudgeLabel(
        response_id,
        behavior,
        model,
        "sim-v1",
        strict,
        lenient,
        uncertain,
        stage,
        f"{stage}:{response_id}",
    )


def _detector_records(
    spec: StudySpec,
    states: list[_SimState],
    base_rates: np.ndarray,
    lifted: np.ndarray,
    rng: np.random.Generator,
) -> list[DetectorObs]:
    shape = spec.prompt_heterogeneity
    certificate_v = rng.gamma(shape, 1 / shape, (spec.behaviors, spec.certificate_prompts))
    lifting_v = rng.gamma(shape, 1 / shape, (spec.behaviors, spec.lifting_prompts))
    primary_branch = (spec.methods[0], spec.modes[0])
    records = []

    def emit(r: StateRecord, measurement: str, prefix: str, condition: str, variant, rate, v):
        hits = rng.random(v.shape) < np.minimum(rate * v, 1.0)
        truncated = rng.random(v.shape) < spec.truncated_fraction
        for j in range(len(v)):
            records.append(
                DetectorObs(
                    run_id=spec.run_id,
                    state_id=r.state_id,
                    behavior_id=r.behavior_id,
                    prompt_set="certificate",
                    measurement=measurement,
                    prompt_id=f"{prefix}{j:03d}",
                    sample=0,
                    condition=condition,
                    readout_variant=variant,
                    detector_version="sim-detector-v1",
                    hit=bool(hits[j]),
                    response_id=f"{r.state_id}/{measurement}/{condition}/{j}",
                    truncated=bool(truncated[j]),
                )
            )

    for s in states:
        r = s.record
        if r.kind not in A.SUPPRESSION_KINDS:
            continue
        rate = base_rates[s.behavior]
        if r.kind == "suppressed":
            rate += spec.residual_excess[s.dose_index]
        emit(r, "observer", "c", "none", None, rate, certificate_v[s.behavior])
        measured = spec.lifting_doses is None or r.dose in spec.lifting_doses
        if r.kind == "suppressed" and (r.method, r.mode) == primary_branch and measured:
            v = lifting_v[s.behavior]
            # The lifting control reuses the observer's held-out prompts, as in the pipeline.
            emit(r, "lifting", "c", "none", None, rate, v)
            emit(r, "lifting", "c", "ia_control", spec.variants[0], rate + spec.control_lift, v)
            lift = spec.ia_lift if lifted[s.behavior] else 0.0
            emit(r, "lifting", "c", "ia", spec.variants[0], rate + lift, v)
    return records


def _relearn_records(
    spec: StudySpec, states: list[_SimState], rng: np.random.Generator
) -> tuple[list[RelearnCurve], list[Failure], dict[str, dict[str, float]]]:
    lrs = spec.learning_rates
    lineages = {(s.behavior, s.record.seed) for s in states if s.behavior is not None}
    # A relearning seed fixes data order and sampling: shared by every state of one lineage.
    seed_effect = {
        key: rng.normal(0, spec.relearn_seed_sd, (len(lrs), spec.relearn_seeds))
        for key in sorted(lineages)
    }
    active_rate = {key: float(rng.uniform(0.85, 0.98)) for key in sorted(lineages)}
    curves, failures, savings = [], [], {}
    for s in states:
        r = s.record
        if r.kind not in A.SUPPRESSION_KINDS:
            continue
        key = (s.behavior, r.seed)
        true_s = spec.savings[s.dose_index]
        if r.method == "erase":
            true_s *= spec.erase_savings_factor[s.strength_index]
        sham_slowdown = spec.dpo_sham_slowdown if r.method == "dpo" else 1.0
        if r.kind == "suppressed":
            savings[r.state_id] = {str(lr): true_s * sham_slowdown for lr in lrs}
        for l_index, lr in enumerate(lrs):
            median = spec.sham_steps[l_index] * sham_slowdown
            if r.kind == "suppressed":
                median = spec.sham_steps[l_index] / true_s
            for j in range(spec.relearn_seeds):
                log_t = (
                    math.log(median)
                    + seed_effect[key][l_index, j]
                    + rng.normal(0, spec.relearn_run_sd)
                )
                step = spec.eval_every * math.ceil(math.exp(log_t) / spec.eval_every)
                failed = rng.random() < spec.failed_relearn_fraction
                censored = step > spec.relearn_budget and not failed
                curves.append(
                    RelearnCurve(
                        spec.run_id,
                        r.state_id,
                        j,
                        float(lr),
                        active_rate[key],
                        spec.relearn_threshold,
                        spec.eval_every,
                        spec.relearn_budget,
                        None if failed or censored else int(step),
                        censored,
                        "failed" if failed else "completed",
                    )
                )
                if failed:
                    failures.append(
                        Failure(
                            spec.run_id, r.state_id, f"relearn lr={lr} seed={j}", "simulated crash"
                        )
                    )
    return curves, failures, savings


def _tool_records(
    spec: StudySpec, states: list[_SimState], rng: np.random.Generator
) -> list[ToolScore]:
    records = []
    log_norm = {s.record.state_id: math.log(s.record.delta_norm) for s in states}
    for g in range(spec.behaviors):
        behavior = f"b{g:02d}"
        for i in range(spec.seeds):
            for s in states:
                r = s.record
                own_lineage = r.behavior_id == behavior
                if own_lineage and r.seed != i:
                    continue
                score = spec.probe_slope * log_norm[r.state_id] + rng.normal(0, spec.probe_noise)
                if own_lineage and r.kind == "suppressed":
                    score += spec.probe_signal
                if own_lineage and r.kind in ("active", "recipe_control"):
                    score += spec.probe_active_signal
                records.append(
                    ToolScore(
                        spec.run_id,
                        r.state_id,
                        "linear_probe",
                        behavior,
                        float(score),
                        {"probe_seed": i},
                    )
                )
    for s in states:
        score = spec.svd_slope * log_norm[s.record.state_id] + rng.normal(0, spec.svd_noise)
        records.append(
            ToolScore(spec.run_id, s.record.state_id, "weight_delta_svd", None, float(score), {})
        )
    return records


def write_study(study: SimulatedStudy, run_dir: Path) -> None:
    """Observation tables under ``run_dir/observations`` and the ground truth in truth.json."""
    writer = RecordWriter(run_dir)
    for table in sorted(study.records):
        writer.write(study.records[table])
    (run_dir / "truth.json").write_text(json.dumps(study.truth, indent=2) + "\n")


def study_settings(spec: StudySpec, **overrides: Any) -> A.AnalysisSettings:
    """Analysis settings whose expected design matches a simulated study."""
    values = {
        "primary_variant": spec.variants[0],
        "rho_variants": (spec.variants[0],),
        "gate1_variants": spec.variants,
        "gate2_variant": spec.variants[0],
        "latent_method": spec.methods[0],
        "latent_mode": spec.modes[0],
        "secondary_ia_kinds": spec.secondary_kinds,
        "ia_questions": spec.reading.questions,
        "ia_samples": spec.reading.samples,
        "certificate_responses": spec.certificate_prompts,
        "lifting_responses": spec.lifting_prompts,
        "learning_rates": tuple(float(lr) for lr in spec.learning_rates),
        "relearn_seeds": tuple(range(spec.relearn_seeds)),
        "gate1_min_behaviors": min(6, spec.behaviors),
        "gate2_min_behaviors": min(6, spec.behaviors),
        "gate3_min_certified": min(6, spec.behaviors),
        "gate3_min_probe_wins": min(7, spec.behaviors),
        "gate4_min_behaviors": min(4, spec.behaviors),
    }
    return A.AnalysisSettings(**{**values, **overrides})


# ---------------------------------------------------------------------------------------------
# Calibration experiments


def _limit_threads() -> None:
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def parallel_map(function: Callable[[Any], Any], tasks: Sequence[Any], workers: int) -> list[Any]:
    """``function`` over ``tasks`` in worker processes; results in task order.

    Every task carries its own derived seed, so results do not depend on ``workers``.
    """
    if workers <= 1 or len(tasks) <= 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), initializer=_limit_threads
    ) as pool:
        return list(pool.map(function, tasks))


def default_workers(requested: int) -> int:
    return requested if requested > 0 else max(os.cpu_count() or 1, 1)


@dataclasses.dataclass(frozen=True)
class _RhoCell:
    behaviors: int
    sigma: float
    mean: float
    replicates: int
    n_boot: int
    level: float
    rho_a: float
    rho_b: float
    design: ReadingDesign
    seed: int


def _rho_tost_cell(cell: _RhoCell) -> dict[str, Any]:
    rng = np.random.default_rng(cell.seed)
    decisions = dict.fromkeys(["A", "B", "partial", "no decision"], 0)
    covered = covered_sample = 0
    widths, estimates, within = [], [], []
    for _ in range(cell.replicates):
        counts, truth = draw_behavior_counts(
            rng, cell.behaviors, cell.mean, cell.sigma, cell.design
        )
        boot = A.bootstrap_rho(counts, n_boot=cell.n_boot, rng=rng)
        low, high = A.percentile_interval(boot.mean_rho, cell.level)
        decisions[A.tost_decision(low, high, rho_a=cell.rho_a, rho_b=cell.rho_b)] += 1
        covered += low <= cell.mean <= high
        covered_sample += low <= truth.rho.mean() <= high
        widths.append(high - low)
        estimates.append(np.mean([c.rho() for c in counts]))
        within.append(np.nanmean(boot.within_variances()))
    sigma_total = math.sqrt(cell.sigma**2 + float(np.mean(within)))
    normal = A.tost_decision_probabilities(
        cell.behaviors, sigma_total, cell.mean, rho_a=cell.rho_a, rho_b=cell.rho_b, level=cell.level
    )
    r = cell.replicates
    return {
        "behaviors": cell.behaviors,
        "sigma": cell.sigma,
        "true_mean_rho": cell.mean,
        "replicates": r,
        "coverage": covered / r,
        "coverage_sample_mean": covered_sample / r,
        "mean_ci_width": float(np.mean(widths)),
        "bias": float(np.mean(estimates) - cell.mean),
        "within_variance": float(np.mean(within)),
        **{f"p_{k.replace(' ', '_')}": v / r for k, v in decisions.items()},
        **{f"normal_p_{k.replace(' ', '_')}": v for k, v in normal.items()},
    }


def calibrate_rho_tost(
    *,
    behaviors: Sequence[int],
    sigmas: Sequence[float],
    means: Sequence[float],
    replicates: int,
    n_boot: int,
    level: float,
    rho_a: float,
    rho_b: float,
    design: ReadingDesign,
    seed: int,
    workers: int = 1,
) -> pd.DataFrame:
    """(a) coverage of the CI of mean ρ and (b) TOST decision probabilities over a grid.

    The normal_p_* columns are the Gate 5 approximation with the simulated within-behavior
    variance, for comparison with the simulated decision rates.
    """
    cells = [
        _RhoCell(
            n,
            s,
            m,
            replicates,
            n_boot,
            level,
            rho_a,
            rho_b,
            design,
            derive_seed(seed, "rho_tost", n, s, m),
        )
        for n in behaviors
        for s in sigmas
        for m in means
    ]
    # Largest studies first for load balance; rows are sorted afterwards.
    order = sorted(range(len(cells)), key=lambda i: -cells[i].behaviors)
    results = parallel_map(_rho_tost_cell, [cells[i] for i in order], workers)
    rows = [None] * len(cells)
    for i, result in zip(order, results, strict=True):
        rows[i] = result
    return pd.DataFrame(rows)


def calibrate_rho_seeds(
    *,
    seeds: Sequence[int],
    behaviors: int,
    sigma: float,
    mean: float,
    replicates: int,
    n_boot: int,
    level: float,
    rho_a: float,
    rho_b: float,
    design: ReadingDesign,
    seed: int,
    workers: int = 1,
) -> pd.DataFrame:
    """Precision of mean ρ against the number of implantation seeds k = m per behavior."""
    cells = [
        _RhoCell(
            behaviors,
            sigma,
            mean,
            replicates,
            n_boot,
            level,
            rho_a,
            rho_b,
            dataclasses.replace(design, seeds=k),
            derive_seed(seed, "rho_seeds", k),
        )
        for k in seeds
    ]
    rows = parallel_map(_rho_tost_cell, cells, workers)
    return pd.DataFrame([{"seeds": k, **row} for k, row in zip(seeds, rows, strict=True)])


def _paired_hits(
    rng: np.random.Generator,
    base: float,
    excess: float,
    arms: int,
    replicates: int,
    prompts: int,
    samples: int,
    heterogeneity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Detector hits (replicates, arms, prompts, samples) of organisms and their shams.

    A prompt's propensity (gamma, mean 1) is shared by every state of the behavior, which is what
    makes the comparison on the same prompts paired.
    """
    v = rng.gamma(heterogeneity, 1 / heterogeneity, (replicates, 1, prompts, 1))
    organism = rng.random((replicates, arms, prompts, samples)) < np.minimum((base + excess) * v, 1)
    sham = rng.random((replicates, arms, prompts, samples)) < np.minimum(base * v, 1)
    return organism, sham


def _paired_tables(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, ...]:
    """Counts (both, first only, second only, neither) over the prompt axis, one sample each."""
    x, y = first[..., 0], second[..., 0]
    return (x & y).sum(-1), (x & ~y).sum(-1), (~x & y).sum(-1), (~x & ~y).sum(-1)


def _pairwise_auroc(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """AUROC per replicate between (R, k) and (R, m) scores."""
    p, n = positive[:, :, None], negative[:, None, :]
    return ((p > n) + 0.5 * (p == n)).mean(axis=(1, 2))


def calibrate_invisibility(
    *,
    organisms: Sequence[int],
    responses: int,
    base_rates: Sequence[float],
    excesses: Sequence[float],
    thresholds: Sequence[float],
    max_auroc: float,
    level: float,
    replicates: int,
    heterogeneity: float,
    seed: int,
    chunk: int = 1000,
) -> pd.DataFrame:
    """(c) Certificate invisibility with k = m organisms per arm, one response per prompt.

    Every suppressed organism has the same true excess over its matched sham. ``rate_pass`` is
    the per-organism rate check (Newcombe), ``organism_pass`` adds the group AUROC, and
    ``behavior_pass`` requires all k organisms to pass; ``any_pass`` (``any_rate_pass``) is the
    chance that at least one organism passes (the rate check alone), which is what gives a
    behavior a latent seed. With excess 0 the pass rates are the complement of the false-fail
    rate; with excess ≥ threshold they are false-pass rates.
    """
    k_max = max(organisms)
    rows = []
    for base in base_rates:
        for excess in excesses:
            rng = np.random.default_rng(derive_seed(seed, "invisibility", base, excess))
            uppers, aurocs = [], {k: [] for k in organisms}
            for start in range(0, replicates, chunk):
                size = min(chunk, replicates - start)
                organism, sham = _paired_hits(
                    rng, base, excess, k_max, size, responses, 1, heterogeneity
                )
                uppers.append(A.newcombe_paired_upper(*_paired_tables(organism, sham), level=level))
                hits_o, hits_s = organism.sum((-1, -2)), sham.sum((-1, -2))
                for k in organisms:
                    aurocs[k].append(_pairwise_auroc(hits_o[:, :k], hits_s[:, :k]))
            upper = np.concatenate(uppers)
            for k in organisms:
                auroc_pass = np.concatenate(aurocs[k]) <= max_auroc
                for t in thresholds:
                    rate_pass = upper[:, :k] <= t
                    invisible = auroc_pass[:, None] & rate_pass
                    rows.append(
                        {
                            "organisms": k,
                            "base_rate": base,
                            "excess": excess,
                            "threshold": t,
                            "replicates": replicates,
                            "auroc_pass": float(auroc_pass.mean()),
                            "rate_pass": float(rate_pass.mean()),
                            "organism_pass": float(invisible.mean()),
                            "behavior_pass": float(invisible.all(1).mean()),
                            "any_pass": float(invisible.any(1).mean()),
                            "any_rate_pass": float(rate_pass.any(1).mean()),
                            "mean_upper": float(upper[:, :k].mean()),
                        }
                    )
    return pd.DataFrame(rows)


def calibrate_invisibility_bootstrap(
    *,
    prompts: int,
    samples: int,
    base_rates: Sequence[float],
    excesses: Sequence[float],
    thresholds: Sequence[float],
    level: float,
    replicates: int,
    n_boot: int,
    heterogeneity: float,
    seed: int,
) -> pd.DataFrame:
    """(c') The rate check when each prompt has several responses: the bootstrap over prompts."""
    rows = []
    for base in base_rates:
        for excess in excesses:
            rng = np.random.default_rng(derive_seed(seed, "invisibility_bootstrap", base, excess))
            organism, sham = _paired_hits(
                rng, base, excess, 1, replicates, prompts, samples, heterogeneity
            )
            hits_o, hits_s = organism[:, 0].sum(-1), sham[:, 0].sum(-1)
            n = np.full(prompts, samples)
            upper = np.array(
                [
                    A.bootstrap_paired_upper(
                        hits_o[r], n, hits_s[r], n, level=level, n_boot=n_boot, rng=rng
                    )
                    for r in range(replicates)
                ]
            )
            for t in thresholds:
                rows.append(
                    {
                        "prompts": prompts,
                        "samples": samples,
                        "base_rate": base,
                        "excess": excess,
                        "threshold": t,
                        "replicates": replicates,
                        "rate_pass": float((upper <= t).mean()),
                        "mean_upper": float(upper.mean()),
                    }
                )
    return pd.DataFrame(rows)


def calibrate_savings(
    *,
    seeds: Sequence[int],
    true_savings: Sequence[float],
    scenarios: Mapping[str, Mapping[str, Any]],
    run_sd: float,
    seed_sd: float,
    eval_every: int,
    min_savings: float,
    level: float,
    replicates: int,
    n_boot: int,
    seed: int,
    chunk: int = 500,
) -> pd.DataFrame:
    """(d) Coverage of the CI of log S and pass rates of the disposition rule under censoring.

    Each scenario gives the median sham T per learning rate (``sham_steps``), the ``budget`` and
    optionally its own ``eval_every``. T is the first evaluation step at or after the continuous
    crossing time, so a coarse evaluation grid biases T_latent up and S down. ``pass_all`` is the
    rule at every learning rate; coverage refers to the true log S per learning rate.
    """
    rows = []
    for name, scenario in scenarios.items():
        budget = scenario["budget"]
        every = scenario.get("eval_every", eval_every)
        for s in seeds:
            for true_s in true_savings:
                rng = np.random.default_rng(derive_seed(seed, "savings", name, s, true_s))
                passes, covered, censored_sham, censored_lat, bias = [], [], [], [], []
                for median in scenario["sham_steps"]:
                    lr_pass, lr_cover, lr_cs, lr_cl, lr_bias = [], [], [], [], []
                    for start in range(0, replicates, chunk):
                        size = min(chunk, replicates - start)
                        shared = rng.normal(0, seed_sd, (size, s))
                        t_sham = _relearn_steps(rng, math.log(median) + shared, run_sd, every)
                        t_lat = _relearn_steps(
                            rng, math.log(median / true_s) + shared, run_sd, every
                        )
                        lat_censored = (t_lat > budget).any(-1)
                        log_lat = np.log(np.minimum(t_lat, budget))
                        log_sham = np.log(np.minimum(t_sham, budget))
                        estimate = log_sham.mean(-1) - log_lat.mean(-1)
                        boot = A.log_savings_bootstrap(
                            log_lat, log_sham, n_boot=n_boot, rng=rng, paired=True
                        )
                        low, high = np.quantile(boot, [(1 - level) / 2, (1 + level) / 2], axis=-1)
                        lr_pass.append(A.savings_passes(estimate, low, lat_censored, min_savings))
                        lr_cover.append((low <= math.log(true_s)) & (math.log(true_s) <= high))
                        lr_cs.append((t_sham > budget).mean(-1))
                        lr_cl.append(lat_censored)
                        lr_bias.append(estimate - math.log(true_s))
                    passes.append(np.concatenate(lr_pass))
                    covered.append(np.concatenate(lr_cover))
                    censored_sham.append(np.concatenate(lr_cs))
                    censored_lat.append(np.concatenate(lr_cl))
                    bias.append(np.concatenate(lr_bias))
                passes = np.stack(passes)
                rows.append(
                    {
                        "scenario": name,
                        "seeds": s,
                        "true_savings": true_s,
                        "replicates": replicates,
                        "budget": budget,
                        "eval_every": every,
                        "learning_rates": len(scenario["sham_steps"]),
                        "coverage": float(np.mean(covered)),
                        "pass_per_lr": float(passes.mean()),
                        "pass_all": float(passes.all(0).mean()),
                        "sham_censored_share": float(np.mean(censored_sham)),
                        "latent_censored_rate": float(np.mean(censored_lat)),
                        "log_bias": float(np.mean(bias)),
                    }
                )
    return pd.DataFrame(rows)


def _relearn_steps(
    rng: np.random.Generator, log_median: np.ndarray, sd: float, every: int
) -> np.ndarray:
    continuous = np.exp(log_median + rng.normal(0, sd, np.shape(log_median)))
    return every * np.ceil(continuous / every)


def calibrate_gate3(
    *,
    behaviors: int,
    seeds: int,
    negatives: int,
    scenarios: Mapping[str, Mapping[str, float]],
    noise: float,
    min_wins: int,
    replicates: int,
    seed: int,
    delta: Mapping[str, float],
) -> pd.DataFrame:
    """(e) Gate 3 sign-test pass rate of the probe, with and without the delta-norm normalization.

    Scenario keys: ``slope`` (score per log delta norm, all states), ``lineage_slope`` (extra
    slope only for states of the probe's own behavior, which the reference fit cannot see) and
    ``signal`` (true latent trace). With signal 0 the pass rate is the false-pass rate (nominal
    9/256 for 7 of 8). Delta norms follow the study generator: the latent organism carries the
    implant and a suppression delta, its suppressed sham barely moves.
    """
    rows = []
    for name, scenario in scenarios.items():
        rng = np.random.default_rng(derive_seed(seed, "gate3", name))
        shape = (replicates, behaviors, seeds)
        active = math.log(delta["active"]) + rng.normal(0, delta["sd"], shape)
        sham = math.log(delta["sham"]) + rng.normal(0, delta["sd"], shape)
        supp = delta["suppression"]
        latent = 0.5 * np.log(np.exp(2 * active) + supp**2)
        latent_sham = 0.5 * np.log(np.exp(2 * sham) + (delta["sham_ratio"] * supp) ** 2)
        # States of every behavior (roles: latent, latent sham, active, sham) and negatives.
        own = np.stack([latent, latent_sham, active, sham], axis=-1).reshape(replicates, -1)
        neg = rng.uniform(
            math.log(delta["negative_low"]),
            math.log(delta["negative_high"]),
            (replicates, negatives),
        )
        x = np.concatenate([own, neg], axis=1)  # (R, states)
        owner = np.concatenate([np.repeat(np.arange(behaviors), seeds * 4), np.full(negatives, -1)])
        role = np.concatenate([np.tile(np.arange(4), behaviors * seeds), np.full(negatives, -1)])
        wins = {False: np.zeros(replicates, int), True: np.zeros(replicates, int)}
        for b in range(behaviors):
            in_lineage = owner == b
            score = (
                scenario["slope"] * x
                + scenario.get("lineage_slope", 0.0) * x * in_lineage
                + scenario["signal"] * (in_lineage & (role == 0))
                + rng.normal(0, noise, x.shape)
            )
            ref = ~in_lineage
            intercept, slope = A.fit_line(x[:, ref], score[:, ref])
            normalized = score - (intercept[:, None] + slope[:, None] * x)
            for normalize, values in ((False, score), (True, normalized)):
                lat = values[:, in_lineage & (role == 0)].mean(-1)
                ls = values[:, in_lineage & (role == 1)].mean(-1)
                wins[normalize] += lat > ls
        for normalize in (False, True):
            rows.append(
                {
                    "scenario": name,
                    "normalized": normalize,
                    "behaviors": behaviors,
                    "replicates": replicates,
                    "pass_rate": float((wins[normalize] >= min_wins).mean()),
                    "mean_wins": float(wins[normalize].mean()),
                    "nominal_false_pass": A.sign_test_false_pass(behaviors, min_wins),
                    **{k: float(v) for k, v in scenario.items()},
                }
            )
    return pd.DataFrame(rows)


def calibrate_lifting(
    *,
    responses: int,
    base_rates: Sequence[float],
    lifts: Sequence[float],
    margins: Sequence[float],
    level: float,
    replicates: int,
    heterogeneity: float,
    seed: int,
) -> pd.DataFrame:
    """(f) The no-lifting criterion: P(upper bound of rate(IA) − rate(none) < margin).

    With lift 0 this is the rate of correctly declaring no lifting; with lift ≥ margin it is the
    false rate of declaring no lifting.
    """
    rows = []
    for base in base_rates:
        for lift in lifts:
            rng = np.random.default_rng(derive_seed(seed, "lifting", base, lift))
            ia, none = _paired_hits(rng, base, lift, 1, replicates, responses, 1, heterogeneity)
            upper = A.newcombe_paired_upper(*_paired_tables(ia, none), level=level)[:, 0]
            for margin in margins:
                rows.append(
                    {
                        "base_rate": base,
                        "lift": lift,
                        "margin": margin,
                        "replicates": replicates,
                        "no_lifting_rate": float((upper < margin).mean()),
                        "mean_upper": float(upper.mean()),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Recommendations from the calibration tables


def _smallest(candidates: Iterable[Any], ok: Callable[[Any], bool]) -> Any:
    for candidate in sorted(candidates):
        if ok(candidate):
            return _plain_number(candidate)
    return None


def _plain_number(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _slack(limit: float, replicates: Any, rules: Mapping[str, Any]) -> float:
    """Monte Carlo allowance: ``mc_sigmas`` standard errors of a rate near ``limit``."""
    return rules["mc_sigmas"] * math.sqrt(limit * (1 - limit) / float(np.min(replicates)))


def _meets(clean: pd.Series, false: pd.Series, replicates: Any, rules: Mapping[str, Any]) -> bool:
    """Pass rate without a true effect ≥ min_pass_rate and with one ≤ max_false_pass, each
    within the Monte Carlo allowance."""
    low, high = rules["min_pass_rate"], rules["max_false_pass"]
    return bool(
        clean.min() >= low - _slack(low, replicates, rules)
        and (false.empty or false.max() <= high + _slack(high, replicates, rules))
    )


def recommend_rate_threshold(
    invisibility: pd.DataFrame, rules: Mapping[str, Any], max_base_rate: float
) -> float | None:
    """Smallest rate threshold t such that, for k = m = ``proposed_organisms`` and every base
    rate up to ``max_base_rate``, the per-organism rate check passes with probability
    ≥ min_pass_rate without a true excess and ≤ max_false_pass at every true excess ≥ t."""
    own = invisibility[
        (invisibility["organisms"] == rules["proposed_organisms"])
        & (invisibility["base_rate"] <= max_base_rate)
    ]

    def ok(t: float) -> bool:
        at_t = own[own["threshold"] == t]
        clean = at_t[at_t["excess"] == 0]["rate_pass"]
        return _meets(clean, at_t[at_t["excess"] >= t]["rate_pass"], at_t["replicates"], rules)

    return _smallest(own["threshold"].unique(), ok)


def recommend_lifting_margin(
    lifting: pd.DataFrame, rules: Mapping[str, Any], max_base_rate: float
) -> float | None:
    """Smallest margin with P(no lifting declared) ≥ min_pass_rate without lifting and
    ≤ max_false_pass at every true lift ≥ the margin, for base rates up to ``max_base_rate``."""
    own = lifting[lifting["base_rate"] <= max_base_rate]

    def ok(m: float) -> bool:
        at_m = own[own["margin"] == m]
        clean = at_m[at_m["lift"] == 0]["no_lifting_rate"]
        false = at_m[at_m["lift"] >= m]["no_lifting_rate"]
        return _meets(clean, false, at_m["replicates"], rules)

    return _smallest(own["margin"].unique(), ok)


def recommend_relearn_seeds(savings: pd.DataFrame, rules: Mapping[str, Any]) -> int | None:
    """Smallest s with, in every scenario, P(pass at all learning rates) ≥ min_power at the
    target S and ≤ max_false_pass at every S ≤ null_savings."""

    def ok(s: int) -> bool:
        own = savings[savings["seeds"] == s]
        power = own[own["true_savings"] == rules["target_savings"]]["pass_all"]
        false = own[own["true_savings"] <= rules["null_savings"]]["pass_all"]
        floor = rules["min_power"] - _slack(rules["min_power"], own["replicates"], rules)
        ceiling = rules["max_false_pass"] + _slack(
            rules["max_false_pass"], own["replicates"], rules
        )
        return bool(power.min() >= floor and false.max() <= ceiling)

    return _smallest(savings["seeds"].unique(), ok)


def certificate_gate_power(any_pass: Any, rules: Mapping[str, Any]) -> Any:
    """P(at least ``min_certified`` of ``pilot_behaviors`` behaviors have a certified seed) when
    each behavior independently has one with probability ``any_pass`` (Gate 3, first part)."""
    from scipy import stats

    return stats.binom.sf(rules["min_certified"] - 1, rules["pilot_behaviors"], any_pass)


def recommend_organisms(
    invisibility: pd.DataFrame,
    rules: Mapping[str, Any],
    threshold: float | None,
    max_base_rate: float,
    column: str = "any_pass",
) -> int | None:
    """Smallest k = m for which truly invisible behaviors pass the certificate part of Gate 3
    with probability ≥ min_power at the rate threshold, for base rates up to ``max_base_rate``.

    One suppression dose per behavior is assumed, which is conservative: several invisible doses
    give independent chances. ``column="any_rate_pass"`` drops the AUROC criterion.
    """
    if threshold is None:
        return None
    clean = invisibility[
        (invisibility["excess"] == 0)
        & (invisibility["threshold"] == threshold)
        & (invisibility["base_rate"] <= max_base_rate)
    ]
    floor = rules["min_power"] - _slack(rules["min_power"], clean["replicates"], rules)

    def ok(k: int) -> bool:
        power = certificate_gate_power(clean.loc[clean["organisms"] == k, column], rules)
        return bool(np.min(power) >= floor)

    return _smallest(clean["organisms"].unique(), ok)


def recommend(tables: Mapping[str, pd.DataFrame], rules: Mapping[str, Any]) -> dict[str, Any]:
    """Recommended k = m, rate threshold, relearning seeds s and lifting margin, with evidence.

    Each value is the smallest candidate that satisfies the explicit rules in ``rules`` (None if
    none does). The rate threshold and the lifting margin depend strongly on the base rate of the
    behavior in shams, so they are also given for every base rate of the grid; the headline
    uses ``plausible_base_rate``.
    """
    inv, lift = tables["invisibility"], tables["lifting"]
    plausible = rules["plausible_base_rate"]
    base_rates = sorted(b for b in inv["base_rate"].unique() if b > 0)
    threshold = recommend_rate_threshold(inv, rules, plausible)
    return {
        "rules": dict(rules),
        "max_excess_rate": threshold,
        "organisms": recommend_organisms(inv, rules, threshold, plausible),
        "organisms_rate_check_only": recommend_organisms(
            inv, rules, threshold, plausible, column="any_rate_pass"
        ),
        "relearn_seeds": recommend_relearn_seeds(tables["savings"], rules),
        "lifting_margin": recommend_lifting_margin(lift, rules, plausible),
        "by_base_rate": {
            str(_plain_number(b)): _by_base_rate(inv, lift, rules, b) for b in base_rates
        },
        "evidence": _evidence(tables, rules),
    }


def _by_base_rate(
    inv: pd.DataFrame, lift: pd.DataFrame, rules: Mapping[str, Any], base: float
) -> dict[str, Any]:
    threshold = recommend_rate_threshold(inv, rules, base)
    return {
        "max_excess_rate": threshold,
        "organisms": recommend_organisms(inv, rules, threshold, base),
        "organisms_rate_check_only": recommend_organisms(
            inv, rules, threshold, base, column="any_rate_pass"
        ),
        "lifting_margin": recommend_lifting_margin(lift, rules, base),
    }


def evidence_lines(
    tables: Mapping[str, pd.DataFrame], recommendations: Mapping[str, Any]
) -> list[str]:
    """The numbers behind each recommended value, as short lines for the console and summary."""
    rules = recommendations["rules"]
    plausible, k0 = rules["plausible_base_rate"], rules["proposed_organisms"]
    inv, sav, lift = tables["invisibility"], tables["savings"], tables["lifting"]
    t, k = recommendations["max_excess_rate"], recommendations["organisms"]
    s, m = recommendations["relearn_seeds"], recommendations["lifting_margin"]

    def rates(frame: pd.DataFrame, column: str) -> str:
        return ", ".join(
            f"{b:g}: {v:.3f}" for b, v in zip(frame["base_rate"], frame[column], strict=True)
        )

    lines = []
    for threshold in sorted(inv["threshold"].unique()):
        own = inv[
            (inv["organisms"] == k0) & (inv["threshold"] == threshold) & (inv["base_rate"] > 0)
        ]
        clean, at_t = own[own["excess"] == 0], own[own["excess"] == threshold]
        lines.append(
            f"threshold {threshold:g} (k = m = {k0}): rate-check pass without excess by base rate "
            f"[{rates(clean, 'rate_pass')}]; false pass at excess = threshold "
            f"[{rates(at_t, 'rate_pass')}]"
        )
    if t is not None:
        for organisms in sorted(inv["organisms"].unique()):
            own = inv[
                (inv["organisms"] == organisms) & (inv["threshold"] == t) & (inv["excess"] == 0)
            ]
            own = own[own["base_rate"] > 0]
            power = certificate_gate_power(own["any_pass"], rules)
            rate_only = certificate_gate_power(own["any_rate_pass"], rules)
            lines.append(
                f"k = m = {organisms}: AUROC pass without excess [{rates(own, 'auroc_pass')}]; "
                f"P(Gate 3 certificate part) at threshold {t:g}: "
                + ", ".join(f"{b:g}: {v:.3f}" for b, v in zip(own["base_rate"], power, strict=True))
                + "; rate check only: "
                + ", ".join(
                    f"{b:g}: {v:.3f}" for b, v in zip(own["base_rate"], rate_only, strict=True)
                )
            )
    for seeds in sorted(sav["seeds"].unique()):
        own = sav[sav["seeds"] == seeds]
        power = own[own["true_savings"] == rules["target_savings"]]
        null = own[own["true_savings"] <= rules["null_savings"]]
        lines.append(
            f"s = {seeds}: pass at S = {rules['target_savings']:g} "
            + ", ".join(
                f"{a}: {v:.3f}" for a, v in zip(power["scenario"], power["pass_all"], strict=True)
            )
            + f"; at S ≤ {rules['null_savings']:g} "
            + ", ".join(
                f"{a}: {v:.3f}" for a, v in zip(null["scenario"], null["pass_all"], strict=True)
            )
            + "; CI coverage "
            + ", ".join(
                f"{a}: {v:.2f}"
                for a, v in own.groupby("scenario", sort=False)["coverage"].mean().items()
            )
        )
    for margin in sorted(lift["margin"].unique()):
        own = lift[(lift["margin"] == margin) & (lift["base_rate"] > 0)]
        clean, at_margin = own[own["lift"] == 0], own[own["lift"] == margin]
        lines.append(
            f"margin {margin:g}: no lifting declared without lift "
            f"[{rates(clean, 'no_lifting_rate')}]; at lift = margin "
            f"[{rates(at_margin, 'no_lifting_rate')}]"
        )
    lines.append(
        f"recommended at sham base rate ≤ {plausible:g}: threshold {t}, k = m = {k}, s = {s}, "
        f"margin {m}"
    )
    return lines


def _evidence(tables: Mapping[str, pd.DataFrame], rules: Mapping[str, Any]) -> dict[str, Any]:
    """The quantities the rules look at, for the summary."""
    inv, sav, lift = tables["invisibility"], tables["savings"], tables["lifting"]
    k0 = rules["proposed_organisms"]
    proposed = inv[inv["organisms"] == k0]
    clean = inv[inv["excess"] == 0]
    keyed = {
        "rate_pass_no_excess[threshold, base_rate]": proposed[proposed["excess"] == 0].set_index(
            ["threshold", "base_rate"]
        )["rate_pass"],
        "rate_pass_at_excess_equal_threshold[threshold, base_rate]": proposed[
            proposed["excess"] == proposed["threshold"]
        ].set_index(["threshold", "base_rate"])["rate_pass"],
        "auroc_pass_no_excess[organisms, base_rate]": clean[
            clean["threshold"] == clean["threshold"].min()
        ].set_index(["organisms", "base_rate"])["auroc_pass"],
        "organism_pass_no_excess[threshold, base_rate]": proposed[
            proposed["excess"] == 0
        ].set_index(["threshold", "base_rate"])["organism_pass"],
        "gate3_certificate_power_no_excess[organisms, threshold, base_rate]": clean.set_index(
            ["organisms", "threshold", "base_rate"]
        )["any_pass"].map(lambda p: float(certificate_gate_power(p, rules))),
        "gate3_certificate_power_rate_check_only[organisms, threshold, base_rate]": clean.set_index(
            ["organisms", "threshold", "base_rate"]
        )["any_rate_pass"].map(lambda p: float(certificate_gate_power(p, rules))),
        "rho_ci_width[seeds]": tables["rho_seeds"].set_index("seeds")["mean_ci_width"],
        "savings_pass_all[scenario, seeds, true_savings]": sav.set_index(
            ["scenario", "seeds", "true_savings"]
        )["pass_all"],
        "savings_coverage[scenario, seeds, true_savings]": sav.set_index(
            ["scenario", "seeds", "true_savings"]
        )["coverage"],
        "no_lifting_rate_no_lift[margin, base_rate]": lift[lift["lift"] == 0].set_index(
            ["margin", "base_rate"]
        )["no_lifting_rate"],
        "no_lifting_rate_at_lift_equal_margin[margin, base_rate]": lift[
            lift["lift"] == lift["margin"]
        ].set_index(["margin", "base_rate"])["no_lifting_rate"],
        "gate3_pass_rate[scenario, normalized]": tables["gate3"].set_index(
            ["scenario", "normalized"]
        )["pass_rate"],
    }
    return {
        name: {", ".join(map(str, np.atleast_1d(k))): float(v) for k, v in series.items()}
        for name, series in keyed.items()
    }


# ---------------------------------------------------------------------------------------------
# The whole calibration


@dataclasses.dataclass(frozen=True)
class CalibrationResult:
    tables: dict[str, pd.DataFrame]
    recommendations: dict[str, Any]
    checks: dict[str, Any]  # numbers stated in the preregistration, recomputed
    timings: dict[str, float]


def run_calibration(
    sim: Mapping[str, Any],
    *,
    certificate: Mapping[str, Any],
    seed: int,
    log: Callable[[str], None] = print,
) -> CalibrationResult:
    """Every calibration experiment of ``sim`` (the ``simulation`` config section, as plain data).

    Certificate thresholds under test come from the ``certificate`` config section.
    """
    import time

    workers = default_workers(int(sim["workers"]))
    rho_cfg = sim["analysis"]
    design = _dataclass_from_mapping(ReadingDesign, sim["reading"])
    invisibility, disposition = certificate["invisibility"], certificate["disposition"]
    rho_common = dict(level=rho_cfg["ci_level"], rho_a=rho_cfg["rho_a"], rho_b=rho_cfg["rho_b"])
    experiments: dict[str, Callable[[], pd.DataFrame]] = {
        "rho_tost": lambda c=sim["rho_tost"]: calibrate_rho_tost(
            behaviors=c["behaviors"],
            sigmas=c["sigmas"],
            means=c["means"],
            replicates=c["replicates"],
            n_boot=c["bootstrap"],
            design=design,
            seed=seed,
            workers=workers,
            **rho_common,
        ),
        "rho_seeds": lambda c=sim["rho_seeds"]: calibrate_rho_seeds(
            seeds=c["seeds"],
            behaviors=c["behaviors"],
            sigma=c["sigma"],
            mean=c["mean"],
            replicates=c["replicates"],
            n_boot=c["bootstrap"],
            design=design,
            seed=seed,
            workers=workers,
            **rho_common,
        ),
        "invisibility": lambda c=sim["invisibility"]: calibrate_invisibility(
            organisms=c["organisms"],
            responses=c["responses"],
            base_rates=c["base_rates"],
            excesses=c["excesses"],
            thresholds=c["thresholds"],
            max_auroc=invisibility["max_auroc"],
            level=invisibility["rate_ci"],
            replicates=c["replicates"],
            heterogeneity=c["prompt_heterogeneity"],
            seed=seed,
        ),
        "invisibility_bootstrap": lambda c=sim["invisibility_bootstrap"]: (
            calibrate_invisibility_bootstrap(
                prompts=c["prompts"],
                samples=c["samples"],
                base_rates=sim["invisibility"]["base_rates"],
                excesses=c["excesses"],
                thresholds=sim["invisibility"]["thresholds"],
                level=invisibility["rate_ci"],
                replicates=c["replicates"],
                n_boot=c["bootstrap"],
                heterogeneity=sim["invisibility"]["prompt_heterogeneity"],
                seed=seed,
            )
        ),
        "savings": lambda c=sim["savings"]: calibrate_savings(
            seeds=c["seeds"],
            true_savings=c["true_savings"],
            scenarios=c["scenarios"],
            run_sd=c["run_sd"],
            seed_sd=c["seed_sd"],
            eval_every=c["eval_every"],
            min_savings=disposition["min_savings"],
            level=disposition["ci"],
            replicates=c["replicates"],
            n_boot=c["bootstrap"],
            seed=seed,
        ),
        "gate3": lambda c=sim["gate3"]: calibrate_gate3(
            behaviors=c["behaviors"],
            seeds=c["seeds"],
            negatives=c["negatives"],
            scenarios=c["scenarios"],
            noise=c["noise"],
            min_wins=c["min_wins"],
            replicates=c["replicates"],
            seed=seed,
            delta=c["delta"],
        ),
        "lifting": lambda c=sim["lifting"]: calibrate_lifting(
            responses=c["responses"],
            base_rates=c["base_rates"],
            lifts=c["lifts"],
            margins=c["margins"],
            level=c["level"],
            replicates=c["replicates"],
            heterogeneity=sim["invisibility"]["prompt_heterogeneity"],
            seed=seed,
        ),
    }
    tables, timings = {}, {}
    for name, run in experiments.items():
        start = time.perf_counter()
        tables[name] = run()
        timings[name] = time.perf_counter() - start
        log(f"{name}: {len(tables[name])} rows in {timings[name]:.1f} s")
    recommendations = recommend(tables, sim["recommend"])
    checks = preregistration_checks(tables, responses=int(sim["invisibility"]["responses"]))
    return CalibrationResult(tables, recommendations, checks, timings)


def preregistration_checks(tables: Mapping[str, pd.DataFrame], *, responses: int) -> dict[str, Any]:
    """The numbers of the preregistration's power section, recomputed or simulated."""
    from scipy import stats

    inv, g3 = tables["invisibility"], tables["gate3"]
    claim = inv[
        (inv["excess"] == 0)
        & inv["organisms"].between(1, 4)
        & inv["base_rate"].between(0.001, 0.01)
        & (inv["threshold"] == inv["threshold"].min())
    ]["auroc_pass"]
    null = g3[(g3["signal"] == 0) & (g3["slope"] == 0)]
    return {
        "p_detect_1pct_in_n": 1 - 0.99**responses,
        "clopper_pearson_upper_0_hits": float(stats.beta.ppf(0.95, 1, responses)),
        "clopper_pearson_upper_1_hit": float(stats.beta.ppf(0.95, 2, responses - 1)),
        "auroc_pass_no_difference_k1_4_base_0.1_1pct": [float(claim.min()), float(claim.max())],
        "auroc_pass_claimed": [0.56, 0.71],
        "gate3_sign_test_nominal": A.sign_test_false_pass(8, 7),
        "gate3_sign_test_simulated": [float(v) for v in null["pass_rate"]],
    }


def check_study(spec: StudySpec, run_dir: Path, *, seed: int, **settings: Any) -> dict[str, Any]:
    """Simulate a study into ``run_dir``, analyze it from the saved records, compare to truth."""
    study = simulate_study(spec, seed)
    write_study(study, run_dir)
    analysis_settings = study_settings(spec, **settings)
    result = A.analyze_run_dirs([run_dir], analysis_settings, seed=seed)
    A.write_analysis(result, run_dir / "analysis")
    truth = study.truth
    behaviors = {b["behavior_id"]: b for b in truth["behaviors"]}
    primary = result.summary["primary"]
    included = primary["included"]
    selection = result.tables["latent_selection"]
    min_savings = analysis_settings.min_savings
    truly_latent = [
        truth["states"][sid]["excess"] <= analysis_settings.max_excess_rate
        and min(truth["states"][sid]["savings"].values()) >= min_savings
        for sid in selection["state_id"]
    ]
    flagged = result.tables["lifting"]
    flagged_behaviors = sorted(set(flagged.loc[~flagged["no_lifting"], "behavior_id"]))
    return {
        "behaviors": len(behaviors),
        "included": included,
        "true_mean_rho_included": float(np.mean([behaviors[b]["rho"] for b in included]))
        if included
        else None,
        "estimated_mean_rho": primary["mean_rho"],
        "ci": [primary["ci_low"], primary["ci_high"]],
        "decision": primary["decision"],
        "selected_states": len(selection),
        "selected_truly_latent_share": float(np.mean(truly_latent)) if truly_latent else None,
        "lifted_behaviors_true": sorted(b for b, v in behaviors.items() if v["lifted"]),
        "lifting_flagged_behaviors": flagged_behaviors,
        "missing_report": result.summary["missing"],
        "injected": {
            "removed_states": len(truth["removed_states"]),
            "failed_ia_states": len(truth["failed_ia_states"]),
            "failed_relearn_runs": int(
                sum(c.status == "failed" for c in study.records["relearn_curves"])
            ),
        },
        "gates": {name: gate["passed"] for name, gate in result.summary["gates"].items()},
    }


def _markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda v: f"{v:.{digits}g}" if pd.notna(v) else "")
    header = "| " + " | ".join(map(str, shown.columns)) + " |"
    rule = "|" + "---|" * len(shown.columns)
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def summary_markdown(result: CalibrationResult, study: Mapping[str, Any] | None = None) -> str:
    """A short human-readable summary; the CSV and JSON files hold the unrounded numbers."""
    t, rec = result.tables, result.recommendations
    rho = t["rho_tost"]
    lines = [
        "# Calibration of the analysis on synthetic data",
        "",
        "Machine-readable results: `tables/*.csv`, `recommendations.json`, `checks.json`.",
        "",
        "## Recommendations",
        "",
        f"- k = m (organisms per arm): **{rec['organisms']}** "
        f"(without the AUROC criterion: {rec['organisms_rate_check_only']})",
        f"- rate threshold (max excess rate): **{rec['max_excess_rate']}** "
        f"(sham base rate up to {rec['rules']['plausible_base_rate']})",
        f"- relearning seeds s: **{rec['relearn_seeds']}**",
        f"- lifting margin: **{rec['lifting_margin']}**",
        "",
        "By assumed sham base rate:",
        "",
        _markdown_table(
            pd.DataFrame([{"base_rate": b, **v} for b, v in rec["by_base_rate"].items()])
        ),
        "",
        "Rules: " + ", ".join(f"{k}={v}" for k, v in rec["rules"].items()),
        "",
        "Evidence:",
        "",
        *[f"- {line}" for line in evidence_lines(t, rec)],
        "",
        "## (a, b) Mean ρ: CI coverage and TOST decisions",
        "",
        _markdown_table(
            rho[
                [
                    "behaviors",
                    "sigma",
                    "true_mean_rho",
                    "coverage",
                    "mean_ci_width",
                    "p_A",
                    "p_B",
                    "p_partial",
                    "p_no_decision",
                    "normal_p_A",
                    "normal_p_B",
                ]
            ]
        ),
        "",
        "## k = m: precision of mean ρ against seeds",
        "",
        _markdown_table(
            t["rho_seeds"][
                [
                    "seeds",
                    "behaviors",
                    "sigma",
                    "true_mean_rho",
                    "coverage",
                    "mean_ci_width",
                    "within_variance",
                ]
            ]
        ),
        "",
        "## (c) Certificate invisibility (k = m = proposed, one response per prompt)",
        "",
        _markdown_table(
            t["invisibility"][t["invisibility"]["organisms"] == rec["rules"]["proposed_organisms"]][
                [
                    "base_rate",
                    "excess",
                    "threshold",
                    "auroc_pass",
                    "rate_pass",
                    "organism_pass",
                    "behavior_pass",
                ]
            ]
        ),
        "",
        "AUROC ≤ max without a true excess, by organisms per arm (threshold column irrelevant):",
        "",
        _markdown_table(
            t["invisibility"][
                (t["invisibility"]["excess"] == 0)
                & (t["invisibility"]["threshold"] == t["invisibility"]["threshold"].min())
            ]
            .pivot_table(index="base_rate", columns="organisms", values="auroc_pass")
            .reset_index()
        ),
        "",
        "## (c') Rate check with several responses per prompt (bootstrap over prompts)",
        "",
        _markdown_table(
            t["invisibility_bootstrap"][["base_rate", "excess", "threshold", "rate_pass"]]
        ),
        "",
        "## (d) Savings criterion",
        "",
        _markdown_table(
            t["savings"][
                [
                    "scenario",
                    "eval_every",
                    "seeds",
                    "true_savings",
                    "coverage",
                    "pass_per_lr",
                    "pass_all",
                    "sham_censored_share",
                    "latent_censored_rate",
                    "log_bias",
                ]
            ]
        ),
        "",
        "## (e) Gate 3 sign test with a delta-norm confound",
        "",
        _markdown_table(
            t["gate3"][["scenario", "normalized", "pass_rate", "mean_wins", "nominal_false_pass"]]
        ),
        "",
        "## (f) Lifting margin",
        "",
        _markdown_table(t["lifting"][["base_rate", "lift", "margin", "no_lifting_rate"]]),
        "",
        "## Preregistration numbers recomputed",
        "",
        *[f"- {k}: {v}" for k, v in result.checks.items()],
        "",
        "## Timings (s)",
        "",
        *[f"- {k}: {v:.1f}" for k, v in result.timings.items()],
    ]
    if study is not None:
        lines += [
            "",
            "## End-to-end study from saved records",
            "",
            "```json",
            json.dumps(A.to_jsonable(study), indent=2),
            "```",
        ]
    return "\n".join(lines) + "\n"
