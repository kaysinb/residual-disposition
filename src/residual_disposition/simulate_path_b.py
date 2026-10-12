"""Simulations for path B (the path-B preregistration amendment, §7; the feasibility plan's
certificate power check), on synthetic data with known ground truth.

- ``simulate_pd_decisions``: synthetic PD studies, behaviors × seeds × the five-state lineage,
  each state's reading a binomial share of 20 hypotheses. The analysis is ``path_b.pd_rho`` and
  ``path_b.decide``; the outputs are the coverage of the 90 % intervals of ρ^PD, ρ^PD,x and the
  trace share, and the decision rates, in particular at the null boundaries.
- ``certificate_power``: the probability that the relearning certificate passes for a latent
  state S times faster than its sham, from a calibrated sham_s curve, both learning rates, the
  evaluation schedule, the budget and the seeds.
- ``simulate_axis_decisions`` (amendment §11.6): continuous per-prompt projections on each
  lineage's axis, with paired prompts and seeds, through ``axis.axis_rho`` and
  ``axis.axis_decision``: the coverage of its intervals at the family's level (96 %), also
  near the denominator's exclusion
  boundary, and the errors at ρ = ±ρ_A.
- ``simulate_second_reader`` (amendment §11.2): two readers with their own denominators, the
  eligibility selection of the common subset, ``path_b.joint_reader_difference`` and
  ``path_b.second_reader_claims``: D's and Δ's coverage, the errors at D = 0 and the power of
  both claims.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from . import axis as axis_mod
from .path_b import (
    ROLES,
    decide,
    joint_reader_difference,
    meets_eligibility,
    pd_rho,
    second_reader_claims,
)
from .relearn_rules import (
    decisive_failure,
    interpolated_crossing,
    meets_savings,
    savings_ratio,
)

# ---------------------------------------------------------------------------------------------
# Synthetic PD studies


@dataclasses.dataclass(frozen=True)
class PDDesign:
    behaviors: int = 7
    seeds: int = 1
    hypotheses: int = 20
    active: tuple[float, float] = (0.85, 1.0)  # P(a hypothesis names b), uniform per behavior
    sham: tuple[float, float] = (0.0, 0.05)
    sham_s_shift: tuple[float, float] = (0.0, 0.05)  # p(sham_s) − p(sham)
    sigma: float = 0.1  # spread of the true ρ_b across behaviors
    sigma_x: float = 0.05  # ρ^x_b = ρ^x + (ρ_b − ρ) + N(0, sigma_x): one behavior, one deviation
    seed_spread: float = 0.05  # half-width of seed effects on each state's probability


def expected_reading(p: float, spread: float) -> float:
    """E[clip(p + U · spread, 0, 1)] with U ~ Uniform(−1, 1): the probability a state's reading
    has on average over the generator's seed effects, which its seed-mean reading estimates.
    Near 0 and 1 the clipping makes it differ from ``p``."""
    if spread <= 0:
        return min(1.0, max(0.0, p))

    def integral(y: float) -> float:  # ∫ clip(t, 0, 1) dt from −∞ to y
        return 0.0 if y <= 0 else y * y / 2 if y <= 1 else 0.5 + (y - 1)

    low, high = p - spread, p + spread
    return (integral(high) - integral(low)) / (high - low)


def draw_pd_study(
    design: PDDesign, rho: float, rho_x: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(readings, lineage, truth): readings per state (x of n), the lineage rows and, per
    behavior, the true ρ_b, ρ^x_b and trace share of the generator itself: ratios of each
    state's expected reading (``expected_reading``), after the probabilities are clipped to
    [0, 1] and the seed effects, clipped again, are averaged out."""
    readings, lineage, truth = [], [], []
    for b in range(design.behaviors):
        a = rng.uniform(*design.active)
        sh = rng.uniform(*design.sham)
        ss = min(1.0, sh + rng.uniform(*design.sham_s_shift))
        deviation = rng.normal(0, design.sigma)
        rho_b = rho + deviation
        rho_x_b = rho_x + deviation + rng.normal(0, design.sigma_x)
        lat = float(np.clip(ss + rho_b * (a - sh), 0, 1))
        sx = float(np.clip(lat - rho_x_b * (a - sh), 0, 1))
        p = {"active": a, "sham": sh, "latent": lat, "sham_s": ss, "sham_x": sx}
        mu = {role: expected_reading(p[role], design.seed_spread) for role in ROLES}
        den = mu["active"] - mu["sham"]
        num = mu["latent"] - mu["sham_s"]
        truth.append(
            {
                "behavior_id": f"b{b}",
                "rho": num / den,
                "rho_x": (mu["latent"] - mu["sham_x"]) / den,
                "trace_share": (mu["sham_x"] - mu["sham_s"]) / num if num != 0 else math.nan,
                "clipped": any(
                    not design.seed_spread <= p[role] <= 1 - design.seed_spread for role in ROLES
                )
                or lat in (0.0, 1.0)
                or sx in (0.0, 1.0),
            }
        )
        for s in range(design.seeds):
            row = {"behavior_id": f"b{b}", "seed": s, "dose": 1}
            for role in ROLES:
                sid = f"b{b}-{role}-{s}"
                q = float(np.clip(p[role] + rng.uniform(-1, 1) * design.seed_spread, 0, 1))
                x = int(rng.binomial(design.hypotheses, q))
                readings.append({"state_id": sid, "x": x, "n": design.hypotheses})
                row[role] = sid
            lineage.append(row)
    return pd.DataFrame(readings), pd.DataFrame(lineage), pd.DataFrame(truth)


def _covers(ci: tuple[float, float], value: float) -> bool:
    return bool(ci[0] <= value <= ci[1])


def population_targets(
    design: PDDesign, rho: float, rho_x: float, *, draws: int = 20000, seed: int = 0
) -> tuple[float, float]:
    """The population means of the behaviors' true ρ_b and ρ^x_b, from the generator itself
    (``draw_pd_study``'s truth, after clipping and seed effects): the values the mean's
    intervals estimate. They differ from the scenario's nominal (ρ, ρ^x) where clipping binds."""
    one = dataclasses.replace(design, behaviors=draws, seeds=1)
    _, _, truth = draw_pd_study(one, rho, rho_x, np.random.default_rng(seed))
    return float(truth["rho"].mean()), float(truth["rho_x"].mean())


def simulate_pd_decisions(
    design: PDDesign,
    scenarios: Mapping[str, tuple[float, float]],
    *,
    replicates: int,
    rho_a: float = 0.2,
    rho_b: float = 0.5,
    level: float = 0.90,
    n_boot: int = 1000,
    delta_pd: float = 0.6,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per scenario (name → nominal (ρ, ρ^x)): the coverage of the mean's intervals of
    ρ^PD and ρ^PD,x against the population targets after clipping (``population_targets``), the mean
    per-behavior coverage of ρ, ρ^x and the trace share (against each behavior's true values),
    the decision rates and the share of dropped resamples. Expression checks are taken as
    "no trace" everywhere, so B is always named by the no-trace subset."""
    rows = []
    for name, (rho, rho_x) in scenarios.items():
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])  # stable per scenario
        target_rho, target_rho_x = population_targets(design, rho, rho_x, seed=seed)
        outcomes, cov_rho, cov_rho_x, per_rho, per_rho_x, per_share, dropped, clipped = (
            [] for _ in range(8)
        )
        for r in range(replicates):
            readings, lineage, truth = draw_pd_study(design, rho, rho_x, rng)
            per, mean = pd_rho(
                readings,
                lineage,
                level=level,
                n_boot=n_boot,
                delta_pd=delta_pd,
                rho_a=rho_a,
                seed=seed + r,
            )
            expressions = {b: "no trace" for b in per}
            outcomes.append(decide(mean, expressions, rho_a=rho_a, rho_b=rho_b, mean_no_trace=mean))
            cov_rho.append(_covers(mean.rho_ci, target_rho))
            cov_rho_x.append(_covers(mean.rho_x_ci, target_rho_x))
            for t in truth.itertuples():
                result = per[t.behavior_id]
                per_rho.append(_covers(result.rho_ci, t.rho))
                per_rho_x.append(_covers(result.rho_x_ci, t.rho_x))
                if result.trace_share_ci is not None and math.isfinite(t.trace_share):
                    per_share.append(_covers(result.trace_share_ci, t.trace_share))
            dropped.append(mean.dropped)
            clipped.append(float(truth["clipped"].mean()))
        decisions = pd.Series([o["outcome"] for o in outcomes])
        rows.append(
            {
                "scenario": name,
                "rho": rho,
                "rho_x": rho_x,
                "target_rho": target_rho,
                "target_rho_x": target_rho_x,
                **dataclasses.asdict(design),
                "replicates": replicates,
                "coverage_mean_rho": float(np.mean(cov_rho)),
                "coverage_mean_rho_x": float(np.mean(cov_rho_x)),
                "coverage_behavior_rho": float(np.mean(per_rho)),
                "coverage_behavior_rho_x": float(np.mean(per_rho_x)),
                "coverage_behavior_trace_share": float(np.mean(per_share)) if per_share else None,
                "share_defined": len(per_share) / len(per_rho),  # behaviors with a trace share
                "share_intervals": len(per_share),
                **{
                    f"p_{o.replace(' ', '_')}": float((decisions == o).mean())
                    for o in (
                        "A",
                        "B",
                        "PD reads the suppression trace",
                        "partial retention",
                        "no decision",
                    )
                },
                "mean_dropped": float(np.mean(dropped)),
                "mean_clipped": float(np.mean(clipped)),
            }
        )
    return pd.DataFrame(rows)


def boundary_scenarios(rho_a: float, rho_b: float, high: float = 0.8) -> dict[str, tuple]:
    """The null boundaries of the amendment's §7, and interior and power scenarios: name →
    (ρ, ρ^x, binding), where ``binding`` names the coordinate that must sit exactly on its
    boundary after clipping (``calibrate_scenario``), or None."""
    return {
        "A at +rho_A (false A)": (rho_a, rho_a, "rho"),
        "A at -rho_A (false A)": (-rho_a, -rho_a, "rho"),
        "B at rho_B, main condition (false B)": (rho_b, rho_b, "rho"),
        "B at rho^x = rho_A, corrected condition (false B)": (high, rho_a, "rho_x"),
        "trace at rho^x = +rho_A (false trace)": (high, rho_a, "rho_x"),
        "trace at rho^x = -rho_A (false trace)": (high, -rho_a, "rho_x"),
        "power: A (rho = 0)": (0.0, 0.0, None),
        "power: B (rho = rho^x = 0.9)": (0.9, 0.9, None),
        "power: trace (rho = 0.9, rho^x = 0)": (0.9, 0.0, None),
        "interior (0.5, 0.5)": (0.5, 0.5, None),
    }


def calibrate_scenario(
    design: PDDesign,
    rho: float,
    rho_x: float,
    binding: str | None,
    *,
    tolerance: float = 0.002,
    seed: int = 0,
) -> tuple[float, float]:
    """The nominal (ρ, ρ^x) whose population target puts the ``binding`` coordinate exactly on
    its boundary (the scenario's own value) after clipping, by bisection; unchanged if None."""
    if binding is None:
        return rho, rho_x
    goal = rho if binding == "rho" else rho_x

    def target(value: float) -> float:
        r, rx = (value, rho_x) if binding == "rho" else (rho, value)
        t_rho, t_rho_x = population_targets(design, r, rx, seed=seed)
        return t_rho if binding == "rho" else t_rho_x

    low, high = goal - 0.15, goal + 0.15
    for _ in range(30):
        mid = (low + high) / 2
        if target(mid) < goal:
            low = mid
        else:
            high = mid
        if abs(target(mid) - goal) <= tolerance:
            break
    return (mid, rho_x) if binding == "rho" else (rho, mid)


# ---------------------------------------------------------------------------------------------
# Certificate power from a calibrated curve


def _curve(steps: Sequence[float], rates: Sequence[float]):
    """A monotone rate(step) through the measured points, flat beyond the last one."""
    s = np.asarray(steps, dtype=float)
    r = np.maximum.accumulate(np.asarray(rates, dtype=float))
    return lambda step: np.interp(step, s, r)


def _run_summary(
    rate_at,
    *,
    schedule: Sequence[int],
    budget: int,
    target: float,
    responses: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """The summary of one simulated relearning run evaluated at ``schedule`` (step 0 first), in
    the pipeline's form (crossing_step, censored, budget, t), for ``certificate_time``."""
    steps, rates = [0.0], [rng.binomial(responses, float(rate_at(0.0))) / responses]
    if rates[0] >= target:
        return {"crossing_step": 0, "censored": False, "budget": budget, "t": None}
    for step in schedule:
        observed = rng.binomial(responses, float(rate_at(float(step)))) / responses
        steps.append(float(step))
        rates.append(observed)
        if observed >= target:
            t = interpolated_crossing(steps, rates, float(step), target)
            return {"crossing_step": int(step), "censored": False, "budget": budget, "t": t}
    return {"crossing_step": None, "censored": True, "budget": budget, "t": None}


def certificate_power(
    sham_steps: Sequence[float],
    sham_rates: Sequence[float],
    *,
    lr_hi: float,
    lr_lo: float,
    schedule: Sequence[int],
    budget: int,
    active_rate: float,
    savings: Sequence[float] = (3.0, 6.0),
    lo_steps: Sequence[float] | None = None,
    lo_rates: Sequence[float] | None = None,
    seeds: int = 3,
    responses: int = 200,
    threshold: float = 0.5,
    seed_sd: float = 0.15,
    min_savings: float = 3.0,
    replicates: int = 2000,
    seed: int = 0,
) -> dict[float, dict[str, float]]:
    """P(certified) for latent states ``savings`` times faster than sham_s.

    The sham_s curve at LR_hi is the calibrated one; at LR_lo it is the calibrated one if given,
    else the LR_hi curve with time scaled ∝ 1/LR. A latent curve is the sham curve S times faster.
    Each run's time scale varies across relearning seeds log-normally (``seed_sd``), and each
    evaluation is binomial over ``responses``. Certified, by the certificate's own rules
    (``decisive_failure``, ``savings_ratio``): no censored latent run, T_sham > T_lat for every
    seed pair, and S ≥ ``min_savings``, at both rates.
    """
    rng = np.random.default_rng(seed)
    target = threshold * active_rate
    evals = [s for s in schedule if s <= budget]
    if not evals or evals[-1] != budget:
        evals.append(budget)
    hi = _curve(sham_steps, sham_rates)
    lo = (
        _curve(lo_steps, lo_rates)
        if lo_steps is not None and lo_rates is not None
        else (lambda step: hi(step * lr_lo / lr_hi))
    )
    out = {}
    for s_true in savings:
        passed = 0
        per_rate = {lr_hi: 0, lr_lo: 0}
        for _ in range(replicates):
            ok_all = True
            for lr, sham_curve in ((lr_hi, hi), (lr_lo, lo)):
                pairs, ok = [], True
                for _ in range(seeds):
                    scale = math.exp(rng.normal(0, seed_sd))
                    run = {"schedule": evals, "budget": budget, "target": target}
                    sham = _run_summary(
                        lambda st, f=sham_curve, k=scale: f(st / k),
                        **run,
                        responses=responses,
                        rng=rng,
                    )
                    latent = _run_summary(
                        lambda st, f=sham_curve, k=scale, fast=s_true: f(st * fast / k),
                        **run,
                        responses=responses,
                        rng=rng,
                    )
                    if decisive_failure(latent, sham):
                        ok = False
                        break
                    pairs.append((latent, sham))
                if ok:
                    ok = meets_savings(savings_ratio(pairs), min_savings)
                per_rate[lr] += ok
                ok_all = ok_all and ok
            passed += ok_all
        out[float(s_true)] = {
            "p_certified": passed / replicates,
            "p_lr_hi": per_rate[lr_hi] / replicates,
            "p_lr_lo": per_rate[lr_lo] / replicates,
        }
    return out


def run_pd_simulation(config: Mapping[str, Any], *, seed: int = 0) -> pd.DataFrame:
    """Every design × scenario of a ``simulation.path_b`` config section."""
    analysis = config["analysis"]
    scenarios = boundary_scenarios(analysis["rho_a"], analysis["rho_b"])
    if config.get("scenarios"):
        scenarios = {k: v for k, v in scenarios.items() if k in set(config["scenarios"])}
    base = config.get("design", {})
    tables = []
    for behaviors in config["behaviors"]:
        for seeds in config["seeds"]:
            design = PDDesign(
                **{
                    **{k: tuple(v) if isinstance(v, list) else v for k, v in base.items()},
                    "behaviors": behaviors,
                    "seeds": seeds,
                }
            )
            # A negative ρ (latent below sham_s) has room only with a raised sham_s.
            raised = dataclasses.replace(design, sham_s_shift=tuple(config["negative_shift"]))
            selected, negative = {}, {}
            for name, (r, rx, binding) in scenarios.items():
                target = negative if r < 0 else selected
                target[name] = calibrate_scenario(
                    raised if r < 0 else design, r, rx, binding, seed=seed
                )
            tables.append(
                simulate_pd_decisions(
                    design,
                    selected,
                    replicates=config["replicates"],
                    rho_a=analysis["rho_a"],
                    rho_b=analysis["rho_b"],
                    level=analysis["ci_level"],
                    n_boot=config["bootstrap"],
                    seed=seed,
                )
            )
            if negative:
                tables.append(
                    simulate_pd_decisions(
                        raised,
                        negative,
                        replicates=config["replicates"],
                        rho_a=analysis["rho_a"],
                        rho_b=analysis["rho_b"],
                        level=analysis["ci_level"],
                        n_boot=config["bootstrap"],
                        seed=seed,
                    )
                )
    return pd.concat(tables, ignore_index=True)


# The six null boundaries of §7, each with the decision that would be an error there.
BOUNDARIES = {
    "A at +rho_A (false A)": "p_A",
    "A at -rho_A (false A)": "p_A",
    "B at rho_B, main condition (false B)": "p_B",
    "B at rho^x = rho_A, corrected condition (false B)": "p_B",
    "trace at rho^x = +rho_A (false trace)": "p_PD_reads_the_suppression_trace",
    "trace at rho^x = -rho_A (false trace)": "p_PD_reads_the_suppression_trace",
}
POWER = {
    "power: A (rho = 0)": "p_A",
    "power: B (rho = rho^x = 0.9)": "p_B",
    "power: trace (rho = 0.9, rho^x = 0)": "p_PD_reads_the_suppression_trace",
}
# Every 90 % interval the analysis reads: the mean's ρ and ρ^x (the decision), each behavior's
# ρ, ρ^x (its class) and trace share.
COVERAGES = (
    "coverage_mean_rho",
    "coverage_mean_rho_x",
    "coverage_behavior_rho",
    "coverage_behavior_rho_x",
    "coverage_behavior_trace_share",
)


def boundary_checks(
    table: pd.DataFrame,
    *,
    designs: Sequence[tuple[int, int]],
    rho_a: float = 0.2,
    rho_b: float = 0.5,
    max_error: float = 0.05,
    min_coverage: float = 0.85,
    max_share_reported: float = 0.05,
) -> pd.DataFrame:
    """§7's acceptance per planned design (behaviors, seeds): every scenario of
    ``boundary_scenarios`` simulated; each of the six boundary errors at most ``max_error``; every
    coverage in ``COVERAGES``, its worst over the scenarios, at least ``min_coverage``. The power
    and the dropped share are reported. A design or scenario that is missing fails.

    The trace share is reported only where ρ^PD_b's lower limit exceeds ρ_A, so its coverage is
    required over the scenarios whose true mean ρ (as designed; calibration puts the target on
    it) is at least ρ_A, and must be computed there. Where the true ρ is at most 0, a reported
    share follows a false excess reading: there the share of behaviors with one must be at most
    ``max_share_reported``, and its coverage, on a few selected intervals, is only reported.
    """
    designed = {name: spec[0] for name, spec in boundary_scenarios(rho_a, rho_b).items()}
    share = "coverage_behavior_trace_share"
    rows = []
    for behaviors, seeds in designs:
        part = table[(table["behaviors"] == behaviors) & (table["seeds"] == seeds)]
        by_name = part.set_index("scenario")
        missing = sorted(set(designed) - set(part["scenario"]))
        problems = [f"scenario not simulated: {name}" for name in missing]
        row: dict[str, Any] = {"behaviors": behaviors, "seeds": seeds}
        for name, column in BOUNDARIES.items():
            error = float(by_name.at[name, column]) if name in by_name.index else None
            row[f"error: {name}"] = error
            if error is not None and not math.isfinite(error):
                problems.append(f"{name}: error not finite")
            elif error is not None and error > max_error:
                problems.append(f"{name}: error {error:.3f} > {max_error}")
        for name, column in POWER.items():
            row[name] = float(by_name.at[name, column]) if name in by_name.index else None
        true_rho = part["scenario"].map(designed)
        excess, none = part[true_rho >= rho_a], part[true_rho <= 0]
        for column in COVERAGES:
            # Required in every scenario of its scope: a missing or NaN value fails.
            scope = excess if column == share else part
            values = (
                pd.to_numeric(scope[column], errors="coerce") if column in scope else pd.Series()
            )
            finite = values[np.isfinite(values)]
            worst = float(finite.min()) if len(finite) else None
            row[f"worst_{column}"] = worst
            if worst is None:
                problems.append(f"{column}: never computed")
            elif len(finite) < len(scope):
                bad = sorted(scope.loc[~np.isfinite(values), "scenario"])
                problems.append(f"{column}: not finite in {bad}")
            elif worst < min_coverage:
                problems.append(f"{column}: worst {worst:.3f} < {min_coverage}")
        reported = (
            pd.to_numeric(none["share_defined"], errors="coerce") if len(none) else pd.Series()
        )
        if len(reported) and not np.isfinite(reported).all():
            problems.append("share_defined at rho <= 0: not finite")
        row["max_share_reported_at_rho_le_0"] = float(reported.max()) if len(reported) else None
        pairs = zip(none["scenario"], none[share], strict=True)
        row["coverage_trace_share_at_rho_le_0"] = (
            {n: None if pd.isna(c) else float(c) for n, c in pairs} if len(none) else None
        )
        if len(reported) and np.isfinite(reported).all() and reported.max() > max_share_reported:
            problems.append(
                f"trace share reported at rho <= 0 for {reported.max():.1%} of behaviors "
                f"> {max_share_reported:.0%}"
            )
        dropped = pd.to_numeric(part["mean_dropped"], errors="coerce")
        row["max_dropped"] = float(dropped.max()) if len(part) else None
        if len(part) and not np.isfinite(dropped).all():
            problems.append("mean_dropped: not finite")
        row["passed"] = not problems
        row["problems"] = "; ".join(problems)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# The activation axis (amendment §11.6)


@dataclasses.dataclass(frozen=True)
class AxisDesign:
    """Per-prompt projections on a lineage's axis, in units of the within-state SD (the common
    scale's population value), times a raw scale per lineage."""

    behaviors: int = 7
    seeds: int = 1
    eval_prompts: int = 300
    fit_prompts: int = 200
    validation_prompts: int = 100
    gap: tuple[float, float] = (1.5, 4.0)  # a(active) − a(sham), scale units, per lineage
    sigma: float = 0.1  # spread of the true ρ_b across behaviors
    sigma_x: float = 0.05
    prompt_sd: float = 0.8  # the prompt's effect, shared by every state of a lineage
    noise_sd: float = 0.6  # each state's own deviation per prompt
    seed_sd: float = 0.1  # each state's mean per seed, scale units
    sham_s_shift: tuple[float, float] = (-0.2, 0.2)  # a(sham_s) − a(sham), scale units
    raw_scale: tuple[float, float] = (0.5, 2.0)
    # Correlation across behaviors of each state's own deviation on an evaluation prompt: every
    # behavior reads the same evaluation texts, so their prompt effects may correlate
    prompt_corr: float = 0.0


def _projections(
    n: int,
    roles: Sequence[str],
    *,
    means: Mapping[str, float],
    raw: float,
    design: AxisDesign,
    rng: np.random.Generator,
    common: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Per role, ``n`` prompts' projections: the role's mean, the prompts' shared effect (paired
    texts) and the role's own noise, in raw units. With ``common`` (per role, standard normal
    per prompt, the same for every behavior), the role's noise has correlation
    ``design.prompt_corr`` across behaviors."""
    shared = rng.normal(0, design.prompt_sd, n)
    out = {}
    for role in roles:
        noise = rng.normal(0, 1, n)
        if common is not None:
            r = design.prompt_corr
            noise = math.sqrt(1 - r) * noise + math.sqrt(r) * common[role]
        out[role] = raw * (means[role] + shared + design.noise_sd * noise)
    return out


def draw_axis_study(
    design: AxisDesign, rho: float, rho_x: float, rng: np.random.Generator
) -> tuple[list[axis_mod.Lineage], pd.DataFrame]:
    """(lineages, truth). Each lineage reads its own evaluation, fit and validation prompts, the
    same texts for every state; its scale is the pooled SD of the base model's and the sham's fit
    projections, its validation gap the active − sham difference on the validation prompts, both
    estimated as the analysis does. ``truth`` has a row per lineage: the behavior's ρ_b and
    ρ^x_b over the population of seeds (no clipping: a projection is unbounded), and the
    lineage's own true contrasts in scale units, its seed effects included (num, num_x, den)."""
    lineages, truth = [], []
    sd = math.hypot(design.prompt_sd, design.noise_sd)
    common = (
        {role: rng.normal(0, 1, design.eval_prompts) for role in ROLES}
        if design.prompt_corr > 0
        else None
    )
    for b in range(design.behaviors):
        deviation = rng.normal(0, design.sigma)
        rho_b = rho + deviation
        rho_x_b = rho_x + deviation + rng.normal(0, design.sigma_x)
        for seed in range(design.seeds):
            gap = rng.uniform(*design.gap) * sd
            sham_s = rng.uniform(*design.sham_s_shift) * sd
            latent = sham_s + rho_b * gap
            means = {
                "active": gap,
                "sham": 0.0,
                "latent": latent,
                "sham_s": sham_s,
                "sham_x": latent - rho_x_b * gap,
                "base": 0.0,
            }
            means = {role: m + rng.normal(0, design.seed_sd * sd) for role, m in means.items()}
            truth.append(
                {
                    "behavior_id": f"b{b}",
                    "seed": seed,
                    "rho": rho_b,
                    "rho_x": rho_x_b,
                    "num": (means["latent"] - means["sham_s"]) / sd,
                    "num_x": (means["latent"] - means["sham_x"]) / sd,
                    "den": (means["active"] - means["sham"]) / sd,
                }
            )
            raw = rng.uniform(*design.raw_scale)
            kw = {"means": means, "raw": raw, "design": design, "rng": rng}
            evaluation = _projections(design.eval_prompts, ROLES, **kw, common=common)
            fit = _projections(design.fit_prompts, ("base", "sham"), **kw)
            validation = _projections(design.validation_prompts, ("active", "sham"), **kw)
            scale = axis_mod.common_scale(fit["base"][:, None], fit["sham"][:, None])
            gap_units = (validation["active"].mean() - validation["sham"].mean()) / scale
            lineages.append(
                axis_mod.Lineage(
                    f"b{b}",
                    seed,
                    {role: v[:, None] for role, v in evaluation.items()},
                    np.asarray(scale, float).reshape(1),
                    np.asarray(gap_units, float).reshape(1),
                )
            )
    return lineages, pd.DataFrame(truth)


def axis_scenarios(rho_a: float) -> dict[str, tuple[float, float]]:
    """The axis's boundaries (§11.6: at ±ρ_A either conclusion is an error) and power cases."""
    return {
        "at +rho_A (false retained or erased)": (rho_a, rho_a),
        "at -rho_A (false retained or erased)": (-rho_a, -rho_a),
        "power: retained (rho = 0.8)": (0.8, 0.8),
        "power: erased (rho = 0)": (0.0, 0.0),
        "interior (0.5, 0.5)": (0.5, 0.5),
    }


def simulate_axis_decisions(
    design: AxisDesign,
    scenarios: Mapping[str, tuple[float, float]],
    *,
    replicates: int,
    rho_a: float = 0.2,
    level: float = axis_mod.AXIS_LEVEL,
    n_boot: int = 1000,
    min_gap: float = 1.0,
    min_denominator: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per scenario: the coverage of the axis's intervals over the results that may carry a
    claim (not unstable), and of the mean's over all; the decision rates; the excluded lineages
    and dropped resamples.

    - The mean's intervals against the population means ρ, ρ^x: the t interval across
      behaviors that decides, and the hierarchical bootstrap's, reported.
    - Each behavior's against its kept lineages' own ratio, their seed effects included
      (``coverage_behavior_rho``): inference is conditional on the fitted directions, and with
      one seed a seed's effect is not separable from the behavior's. Against the behavior's ρ_b
      over the population of seeds as well (``coverage_behavior_rho_seeds``), reported.
    """
    rows = []
    for name, (rho, rho_x) in scenarios.items():
        rng = np.random.default_rng([seed, zlib.crc32(name.encode()), 11])
        decisions, cov, cov_x, cov_all, per_cov, per_cov_x, cov_boot = ([] for _ in range(7))
        excluded, dropped, unstable_b, left, per_cov_seeds = [], [], [], [], []
        for r in range(replicates):
            lines, truth = draw_axis_study(design, rho, rho_x, rng)
            kept, out = axis_mod.eligible_lineages(lines, layer=1, min_gap=min_gap)
            excluded.append(len(out) / len(lines))
            per, mean = axis_mod.axis_rho(
                kept,
                layer=1,
                level=level,
                n_boot=n_boot,
                min_denominator=min_denominator,
                seed=seed + r,
            )
            decisions.append(axis_mod.axis_decision(mean, rho_a=rho_a))
            left.append(len(per))
            if mean is None:
                continue
            dropped.append(mean.dropped)
            covered = _covers(mean.rho_ci, rho)
            cov_all.append(covered)
            if not mean.unstable:
                cov.append(covered)
                cov_x.append(_covers(mean.rho_x_ci, rho_x))
                cov_boot.append(_covers(mean.rho_ci_bootstrap, rho))
            kept_ids = {(line.behavior_id, line.seed) for line in kept}
            pairs = zip(truth["behavior_id"], truth["seed"], strict=True)
            mine = truth[[pair in kept_ids for pair in pairs]]
            for b, result in per.items():
                unstable_b.append(result.unstable)
                if result.unstable:
                    continue
                rows_b = mine[mine["behavior_id"] == b]
                den = rows_b["den"].mean()
                per_cov.append(_covers(result.rho_ci, rows_b["num"].mean() / den))
                per_cov_x.append(_covers(result.rho_x_ci, rows_b["num_x"].mean() / den))
                per_cov_seeds.append(_covers(result.rho_ci, rows_b["rho"].iloc[0]))
        outcomes = pd.Series(decisions)

        def share(values: list) -> float | None:
            return float(np.mean(values)) if values else None

        rows.append(
            {
                "scenario": name,
                "rho": rho,
                "rho_x": rho_x,
                **{k: v for k, v in dataclasses.asdict(design).items()},
                "replicates": replicates,
                "coverage_mean_rho": share(cov),
                "coverage_mean_rho_all": share(cov_all),
                "coverage_mean_rho_x": share(cov_x),
                "coverage_mean_rho_bootstrap": share(cov_boot),  # the reported bootstrap
                "coverage_behavior_rho": share(per_cov),
                "coverage_behavior_rho_x": share(per_cov_x),
                "coverage_behavior_rho_seeds": share(per_cov_seeds),
                **{
                    f"p_{o.replace(' ', '_')}": float((outcomes == o).mean())
                    for o in (
                        "retained",
                        "erased",
                        "undetermined",
                        "unstable",
                        "no eligible lineage",
                    )
                },
                "mean_excluded_lineages": float(np.mean(excluded)),
                "mean_behaviors_left": float(np.mean(left)),
                "mean_dropped": share(dropped),
                "share_behaviors_unstable": share(unstable_b),
            }
        )
    return pd.DataFrame(rows)


AXIS_ERRORS = {
    "at +rho_A (false retained or erased)": ("p_retained", "p_erased"),
    "at -rho_A (false retained or erased)": ("p_retained", "p_erased"),
}
AXIS_POWER = {
    "power: retained (rho = 0.8)": "p_retained",
    "power: erased (rho = 0)": "p_erased",
}
AXIS_COVERAGES = ("coverage_mean_rho", "coverage_behavior_rho")  # required; ρ^x is reported


def axis_design(
    config: Mapping[str, Any], gaps: str, corr: float, behaviors: int, seeds: int
) -> AxisDesign:
    """The ``AxisDesign`` of one planned design of ``simulation.path_b.axis``."""
    base = {k: tuple(v) if isinstance(v, list) else v for k, v in config.get("design", {}).items()}
    return AxisDesign(
        **{
            **base,
            "behaviors": int(behaviors),
            "seeds": int(seeds),
            "gap": tuple(config["gaps"][gaps]),
            "prompt_corr": float(corr),
        }
    )


def axis_recheck(config: Mapping[str, Any], checks: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """The recheck rule of §11.6 (revision 11): a planned design that failed only on its boundary
    errors, none above ``max_error + recheck.margin``, is rerun on the boundary scenarios alone
    with ``recheck.replicates`` (2,000) and its own seed; that run decides its acceptance. For a
    calibrated interval the error at +ρ_A is 0.05 by construction, and at 300 replicates its
    Monte Carlo SE is about 0.013. The inference is not changed for such a result."""
    recheck = config.get("recheck") or {}
    replicates = int(recheck.get("replicates", 2000))
    margin = float(recheck.get("margin", 0.03))
    max_error = float(config["max_error"])
    rho_a = config["analysis"]["rho_a"]
    boundaries = {name: axis_scenarios(rho_a)[name] for name in AXIS_ERRORS}
    rows = []
    for _, row in checks.iterrows():
        if row["passed"]:
            continue
        problems = [p for p in str(row["problems"]).split("; ") if p]
        if not problems or not all(p.split(":")[0] in AXIS_ERRORS for p in problems):
            continue  # a coverage or a missing scenario is not rechecked
        errors = [row[f"error: {name}"] for name in AXIS_ERRORS]
        if any(e is None or not math.isfinite(e) or e > max_error + margin for e in errors):
            continue
        design = axis_design(
            config, row["gaps"], row["prompt_corr"], row["behaviors"], row["seeds"]
        )
        table = simulate_axis_decisions(
            design,
            boundaries,
            replicates=replicates,
            rho_a=rho_a,
            level=config["analysis"]["ci_level"],
            n_boot=config["bootstrap"],
            seed=seed + 1,
        ).set_index("scenario")
        out = {k: row[k] for k in ("gaps", "prompt_corr", "behaviors", "seeds")}
        out["replicates"] = replicates
        for name, columns in AXIS_ERRORS.items():
            out[f"error: {name}"] = float(sum(table.at[name, c] for c in columns))
            out[f"before: {name}"] = row[f"error: {name}"]
        out["passed"] = all(out[f"error: {name}"] <= max_error for name in AXIS_ERRORS)
        rows.append(out)
    return pd.DataFrame(rows)


def run_axis_simulation(config: Mapping[str, Any], *, seed: int = 0) -> pd.DataFrame:
    """Every design (gap range × prompt correlation × behaviors × seeds) × scenario of
    ``simulation.path_b.axis``."""
    rho_a = config["analysis"]["rho_a"]
    scenarios = axis_scenarios(rho_a)
    tables = []
    grid = itertools.product(
        config["gaps"],
        config.get("prompt_corr", [0.0]),
        config["behaviors"],
        config["seeds"],
    )
    for gap_name, corr, behaviors, seeds in grid:
        design = axis_design(config, gap_name, corr, behaviors, seeds)
        table = simulate_axis_decisions(
            design,
            scenarios,
            replicates=config["replicates"],
            rho_a=rho_a,
            level=config["analysis"]["ci_level"],
            n_boot=config["bootstrap"],
            seed=seed,
        )
        tables.append(table.assign(gaps=gap_name))
    return pd.concat(tables, ignore_index=True)


def axis_checks(
    table: pd.DataFrame,
    *,
    designs: Sequence[tuple[str, float, int, int]],
    max_error: float = 0.05,
    min_coverage: float = 0.90,
) -> pd.DataFrame:
    """§11.6's acceptance per planned design (gap range, prompt correlation, behaviors, seeds):
    every scenario simulated; at each of ±ρ_A the probability of either erroneous conclusion
    (retained or erased) at most ``max_error``; the coverage of the mean's and each behavior's
    intervals at the family's level, its worst over the scenarios, at least ``min_coverage``.
    Power, ρ^x's
    coverage, exclusions and dropped resamples are reported."""
    rows = []
    for gaps, corr, behaviors, seeds in designs:
        corrs = table["prompt_corr"].astype(float) if "prompt_corr" in table else 0.0
        part = table[
            (table["gaps"] == gaps)
            & np.isclose(corrs, float(corr))
            & (table["behaviors"] == behaviors)
            & (table["seeds"] == seeds)
        ]
        by_name = part.set_index("scenario")
        problems = [
            f"scenario not simulated: {n}"
            for n in sorted(set(axis_scenarios(0.2)) - set(part["scenario"]))
        ]
        row: dict[str, Any] = {
            "gaps": gaps,
            "prompt_corr": float(corr),
            "behaviors": behaviors,
            "seeds": seeds,
        }
        for name, columns in AXIS_ERRORS.items():
            if name not in by_name.index:
                continue
            error = float(sum(by_name.at[name, c] for c in columns))
            row[f"error: {name}"] = error
            if not math.isfinite(error) or error > max_error:
                problems.append(f"{name}: error {error:.3f} > {max_error}")
        for name, column in AXIS_POWER.items():
            row[name] = float(by_name.at[name, column]) if name in by_name.index else None
        reported = (
            "coverage_mean_rho_x",
            "coverage_behavior_rho_x",
            "coverage_behavior_rho_seeds",
            "coverage_mean_rho_bootstrap",
        )
        for column in (*AXIS_COVERAGES, *reported):
            values = (
                pd.to_numeric(part[column], errors="coerce")
                if len(part) and column in part
                else pd.Series(dtype=float)
            )
            finite = values[np.isfinite(values)]
            worst = float(finite.min()) if len(finite) else None
            row[f"worst_{column}"] = worst
            if column not in AXIS_COVERAGES:
                continue
            if worst is None or len(finite) < len(part):
                problems.append(f"{column}: not computed in every scenario")
            elif worst < min_coverage:
                problems.append(f"{column}: worst {worst:.3f} < {min_coverage}")
        for column in ("mean_excluded_lineages", "mean_dropped", "p_unstable"):
            values = pd.to_numeric(part[column], errors="coerce")
            row[f"max_{column}"] = float(values.max()) if len(values) else None
        row["passed"] = not problems
        row["problems"] = "; ".join(problems)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# The second reader: D, Δ and the claims (amendment §11.2)

READERS = ("adl", "pd")


@dataclasses.dataclass(frozen=True)
class SecondReaderDesign:
    """Two readers of the same lineages, each with its own probabilities of a counted hypothesis:
    its own denominator, a shared deviation of ρ_b across behaviors and a reader's own."""

    behaviors: int = 8
    seeds: int = 1
    hypotheses: int = 20
    active: tuple[float, float] = (0.8, 1.0)  # per reader, uniform per behavior
    sham: tuple[float, float] = (0.0, 0.1)
    sham_s_shift: tuple[float, float] = (0.0, 0.05)
    sigma: float = 0.1  # ρ_b's deviation, shared by the readers
    sigma_reader: float = 0.05  # each reader's own deviation
    sigma_x: float = 0.05
    seed_spread: float = 0.05
    need: int = 6  # ADL qualifies with this many eligible behaviors (6 of 8)
    min_common: int = 5


def _second_reader_truth(
    design: SecondReaderDesign, rho: Mapping[str, float], rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Per behavior: each reader's state probabilities and its true ρ_b, ρ^x_b and Δ part from
    the expected readings (``expected_reading``), as ``draw_pd_study``."""
    out = []
    for b in range(design.behaviors):
        shared = rng.normal(0, design.sigma)
        row: dict[str, Any] = {"behavior_id": f"b{b}"}
        for reader in READERS:
            a = rng.uniform(*design.active)
            sh = rng.uniform(*design.sham)
            ss = min(1.0, sh + rng.uniform(*design.sham_s_shift))
            rho_b = rho[reader] + shared + rng.normal(0, design.sigma_reader)
            rho_x_b = rho_b + rng.normal(0, design.sigma_x)
            lat = float(np.clip(ss + rho_b * (a - sh), 0, 1))
            sx = float(np.clip(lat - rho_x_b * (a - sh), 0, 1))
            p = {"active": a, "sham": sh, "latent": lat, "sham_s": ss, "sham_x": sx}
            mu = {role: expected_reading(p[role], design.seed_spread) for role in ROLES}
            den = mu["active"] - mu["sham"]
            row[reader] = {
                "p": p,
                "rho": (mu["latent"] - mu["sham_s"]) / den,
                "rho_x": (mu["latent"] - mu["sham_x"]) / den,
                "num": mu["latent"] - mu["sham_s"],
            }
        out.append(row)
    return out


def draw_second_reader_study(
    design: SecondReaderDesign, rho: Mapping[str, float], rng: np.random.Generator
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, list[str]], list[dict[str, Any]]]:
    """(readings per reader, lineage, eligible behaviors per reader, truth). Each reader reads
    every state of a lineage (20 hypotheses), and a recipe control per seed with the active
    organism's probability; eligibility is the analysis's rule on those same readings, so the
    common subset is selected on the data its denominators use."""
    truth = _second_reader_truth(design, rho, rng)
    readings: dict[str, list[dict[str, Any]]] = {r: [] for r in READERS}
    lineage, eligible = [], {r: [] for r in READERS}
    n = design.hypotheses
    for t in truth:
        b = t["behavior_id"]
        shares: dict[str, dict[str, list[float]]] = {r: {} for r in READERS}
        for s in range(design.seeds):
            row = {"behavior_id": b, "seed": s, "dose": 1}
            for role in (*ROLES, "recipe_control"):
                sid = f"{b}-{role}-{s}"
                if role != "recipe_control":
                    row[role] = sid
                for reader in READERS:
                    p = t[reader]["p"]["active" if role == "recipe_control" else role]
                    q = float(np.clip(p + rng.uniform(-1, 1) * design.seed_spread, 0, 1))
                    x = int(rng.binomial(n, q))
                    readings[reader].append({"state_id": sid, "x": x, "n": n})
                    shares[reader].setdefault(role, []).append(x / n)
            lineage.append(row)
        for reader in READERS:
            sh = shares[reader]
            if meets_eligibility(
                float(np.mean(sh["active"])), float(np.mean(sh["sham"])), sh["recipe_control"]
            ):
                eligible[reader].append(b)
    frames = {r: pd.DataFrame(rows) for r, rows in readings.items()}
    return frames, pd.DataFrame(lineage), eligible, truth


def second_reader_targets(
    design: SecondReaderDesign, rho: Mapping[str, float], *, draws: int = 20000, seed: int = 0
) -> dict[str, float]:
    """The population means over the behaviors eligible for both readers of the true D_b,
    Δ_b and D^x_b: the values the intervals estimate. Selection uses each reader's noisy
    readings, as the analysis does."""
    rng = np.random.default_rng(seed)
    one = dataclasses.replace(design, behaviors=1)
    d, delta, dx = [], [], []
    for _ in range(draws):
        _, _, eligible, truth = draw_second_reader_study(one, rho, rng)
        if eligible["adl"] and eligible["pd"]:
            t = truth[0]
            d.append(t["adl"]["rho"] - t["pd"]["rho"])
            delta.append(t["adl"]["num"] - t["pd"]["num"])
            dx.append(t["adl"]["rho_x"] - t["pd"]["rho_x"])
    return {"D": float(np.mean(d)), "Delta": float(np.mean(delta)), "D_x": float(np.mean(dx))}


def calibrate_second_reader(
    design: SecondReaderDesign,
    rho: Mapping[str, float],
    *,
    tolerance: float = 0.002,
    draws: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """The nominal ρ^ADL that puts the population D exactly at 0 (for the null scenarios), with
    ρ^PD fixed, by bisection."""
    goal = rho["pd"]
    low, high = goal - 0.1, goal + 0.1
    mid = goal
    for _ in range(25):
        mid = (low + high) / 2
        value = second_reader_targets(design, {"adl": mid, "pd": goal}, draws=draws, seed=seed)["D"]
        if abs(value) <= tolerance:
            break
        if value < 0:
            low = mid
        else:
            high = mid
    return {"adl": mid, "pd": goal}


def second_reader_scenarios() -> dict[str, tuple[float, float, bool]]:
    """name → (ρ^ADL, ρ^PD, calibrated to D = 0): the nulls at D = 0 and the alternatives of
    §11.2, with one where PD reads A, the only case the second claim can hold."""
    return {
        "D = 0 at 0.8 (false D > 0)": (0.8, 0.8, True),
        "D = 0 at 0.5 (false D > 0)": (0.5, 0.5, True),
        "D = 0 at 0 (false D > 0)": (0.0, 0.0, True),
        "power: 0.8 vs 0.3": (0.8, 0.3, False),
        "power: 0.6 vs 0.4": (0.6, 0.4, False),
        "power: 0.8 vs 0": (0.8, 0.0, False),
    }


def simulate_second_reader(
    design: SecondReaderDesign,
    scenarios: Mapping[str, Mapping[str, float]],
    *,
    replicates: int,
    rho_a: float = 0.2,
    rho_b: float = 0.5,
    level: float = 0.90,
    n_boot: int = 1000,
    delta_pd: float = 0.6,
    target_draws: int = 20000,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per scenario (name → {adl, pd} nominal ρ): the common subset's size, how often the
    claims' conditions hold (ADL qualified, at least ``min_common`` common behaviors), D's, Δ's
    and D^x's coverage where they do, P(lower(D) > 0) and the claim rates (unconditional)."""
    rows = []
    for name, rho in scenarios.items():
        rng = np.random.default_rng([seed, zlib.crc32(name.encode()), 13])
        targets = second_reader_targets(design, rho, draws=target_draws, seed=seed)
        commons, made, d_pos, retains, identifies, unstable = ([] for _ in range(6))
        cov = {k: [] for k in ("D", "Delta", "D_x")}
        for r in range(replicates):
            frames, lineage, eligible, _ = draw_second_reader_study(design, rho, rng)
            common = sorted(set(eligible["adl"]) & set(eligible["pd"]))
            commons.append(len(common))
            qualified = len(eligible["adl"]) >= design.need
            ok = qualified and len(common) >= design.min_common
            made.append(ok)
            if not ok:
                retains.append(False)
                identifies.append(False)
                continue
            sub = lineage[lineage["behavior_id"].isin(common)]
            kw = {"level": level, "n_boot": n_boot, "delta_pd": delta_pd, "seed": seed + r}
            _, mean_adl = pd_rho(frames["adl"], sub, rho_a=rho_a, **kw)
            _, mean_pd = pd_rho(frames["pd"], sub, rho_a=rho_a, **kw)
            diff = joint_reader_difference(frames["adl"], frames["pd"], sub, **kw)
            claims = second_reader_claims(mean_adl, mean_pd, diff, rho_a=rho_a, rho_b=rho_b)
            retains.append(claims["retains_more"])
            identifies.append(claims["identifies_where_pd_does_not"])
            d_pos.append(bool(diff["D_ci"][0] > 0) and not diff["unstable"])
            unstable.append(diff["unstable"])
            for key in cov:
                cov[key].append(_covers(diff[f"{key}_ci"], targets[key]))

        def share(values: list) -> float | None:
            return float(np.mean(values)) if values else None

        rows.append(
            {
                "scenario": name,
                "rho_adl": rho["adl"],
                "rho_pd": rho["pd"],
                **{f"target_{k}": v for k, v in targets.items()},
                **dataclasses.asdict(design),
                "replicates": replicates,
                "mean_common": float(np.mean(commons)),
                "min_common_seen": int(np.min(commons)),
                "max_common_seen": int(np.max(commons)),
                "p_claims_possible": float(np.mean(made)),
                "coverage_D": share(cov["D"]),
                "coverage_Delta": share(cov["Delta"]),
                "coverage_D_x": share(cov["D_x"]),
                "p_D_positive": share(d_pos),  # where the claims are possible
                "p_retains_more": float(np.mean(retains)),
                "p_identifies_where_pd_does_not": float(np.mean(identifies)),
                "p_retains_more_if_possible": share(
                    [c for c, m in zip(retains, made, strict=True) if m]
                ),
                "p_identifies_if_possible": share(
                    [c for c, m in zip(identifies, made, strict=True) if m]
                ),
                "share_unstable": share(unstable),
            }
        )
    return pd.DataFrame(rows)


def run_second_reader_simulation(config: Mapping[str, Any], *, seed: int = 0) -> pd.DataFrame:
    """Every design × scenario of ``simulation.path_b.second_reader``, the nulls calibrated to a
    population D of exactly 0."""
    analysis = config["analysis"]
    base = config.get("design", {})
    tables = []
    for behaviors in config["behaviors"]:
        for seeds in config["seeds"]:
            design = SecondReaderDesign(
                **{
                    **{k: tuple(v) if isinstance(v, list) else v for k, v in base.items()},
                    "behaviors": behaviors,
                    "seeds": seeds,
                    "need": math.ceil(0.75 * behaviors),  # 6 of 8
                }
            )
            scenarios = {}
            for name, (adl, pd_, null) in second_reader_scenarios().items():
                rho = {"adl": adl, "pd": pd_}
                scenarios[name] = calibrate_second_reader(design, rho, seed=seed) if null else rho
            tables.append(
                simulate_second_reader(
                    design,
                    scenarios,
                    replicates=config["replicates"],
                    rho_a=analysis["rho_a"],
                    rho_b=analysis["rho_b"],
                    level=analysis["ci_level"],
                    n_boot=config["bootstrap"],
                    target_draws=config.get("target_draws", 20000),
                    seed=seed,
                )
            )
    return pd.concat(tables, ignore_index=True)


SECOND_READER_COVERAGES = ("coverage_D", "coverage_Delta")  # required; D^x's is reported


def second_reader_checks(
    table: pd.DataFrame,
    *,
    designs: Sequence[tuple[int, int]],
    max_error: float = 0.05,
    min_coverage: float = 0.85,
) -> pd.DataFrame:
    """§11.2's acceptance per planned design: every scenario simulated; at D = 0 the share of
    lower(D) > 0, where the claims are possible, and the first claim's rate, each at most
    ``max_error``; D's and Δ's 90 % coverage, worst over the scenarios, at least
    ``min_coverage``. The power of both claims and the common subset's size are reported."""
    rows = []
    nulls = [n for n, (_, _, null) in second_reader_scenarios().items() if null]
    powers = [n for n, (_, _, null) in second_reader_scenarios().items() if not null]
    for behaviors, seeds in designs:
        part = table[(table["behaviors"] == behaviors) & (table["seeds"] == seeds)]
        by_name = part.set_index("scenario")
        problems = [
            f"scenario not simulated: {n}"
            for n in sorted(set(second_reader_scenarios()) - set(part["scenario"]))
        ]
        row: dict[str, Any] = {"behaviors": behaviors, "seeds": seeds}
        for name in nulls:
            if name not in by_name.index:
                continue
            for column in ("p_D_positive", "p_retains_more"):
                value = by_name.at[name, column]
                error = float(value) if value is not None else math.nan
                row[f"{column}: {name}"] = error
                if not math.isfinite(error) or error > max_error:
                    problems.append(f"{name}: {column} {error:.3f} > {max_error}")
        for name in powers:
            if name in by_name.index:
                for column in ("p_retains_more", "p_identifies_where_pd_does_not"):
                    row[f"{column}: {name}"] = float(by_name.at[name, column])
        for column in (*SECOND_READER_COVERAGES, "coverage_D_x"):
            values = (
                pd.to_numeric(part[column], errors="coerce")
                if len(part)
                else pd.Series(dtype=float)
            )
            finite = values[np.isfinite(values)]
            worst = float(finite.min()) if len(finite) else None
            row[f"worst_{column}"] = worst
            if column not in SECOND_READER_COVERAGES:
                continue
            if worst is None or len(finite) < len(part):
                problems.append(f"{column}: not computed in every scenario")
            elif worst < min_coverage:
                problems.append(f"{column}: worst {worst:.3f} < {min_coverage}")
        row["mean_common"] = float(part["mean_common"].mean()) if len(part) else None
        row["min_p_claims_possible"] = float(part["p_claims_possible"].min()) if len(part) else None
        row["passed"] = not problems
        row["problems"] = "; ".join(problems)
        rows.append(row)
    return pd.DataFrame(rows)
