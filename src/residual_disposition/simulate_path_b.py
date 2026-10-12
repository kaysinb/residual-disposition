"""Simulations for path B (the path-B preregistration amendment, §7; the feasibility plan's
certificate power check), on synthetic data with known ground truth.

- ``simulate_pd_decisions``: synthetic PD studies, behaviors × seeds × the five-state lineage,
  each state's reading a binomial share of 20 hypotheses. The analysis is ``path_b.pd_rho`` and
  ``path_b.decide``; the outputs are the coverage of the 90 % intervals of ρ^PD, ρ^PD,x and the
  trace share, and the decision rates, in particular at the null boundaries.
- ``certificate_power``: the probability that the relearning certificate passes for a latent
  state S times faster than its sham, from a calibrated sham_s curve, both learning rates, the
  evaluation schedule, the budget and the seeds.
"""

from __future__ import annotations

import dataclasses
import math
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .path_b import ROLES, decide, pd_rho
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
            if error is not None and error > max_error:
                problems.append(f"{name}: error {error:.3f} > {max_error}")
        for name, column in POWER.items():
            row[name] = float(by_name.at[name, column]) if name in by_name.index else None
        true_rho = part["scenario"].map(designed)
        excess, none = part[true_rho >= rho_a], part[true_rho <= 0]
        for column in COVERAGES:
            scope = excess if column == share else part
            values = scope[column].dropna().astype(float) if column in scope else pd.Series()
            worst = float(values.min()) if len(values) else None
            row[f"worst_{column}"] = worst
            if worst is None:
                problems.append(f"{column}: never computed")
            elif worst < min_coverage:
                problems.append(f"{column}: worst {worst:.3f} < {min_coverage}")
        reported = none["share_defined"].astype(float) if len(none) else pd.Series()
        row["max_share_reported_at_rho_le_0"] = float(reported.max()) if len(reported) else None
        pairs = zip(none["scenario"], none[share], strict=True)
        row["coverage_trace_share_at_rho_le_0"] = (
            {n: None if pd.isna(c) else float(c) for n, c in pairs} if len(none) else None
        )
        if len(reported) and reported.max() > max_share_reported:
            problems.append(
                f"trace share reported at rho <= 0 for {reported.max():.1%} of behaviors "
                f"> {max_share_reported:.0%}"
            )
        row["max_dropped"] = float(part["mean_dropped"].max()) if len(part) else None
        row["passed"] = not problems
        row["problems"] = "; ".join(problems)
        rows.append(row)
    return pd.DataFrame(rows)
