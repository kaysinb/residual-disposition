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
from .relearn_rules import interpolated_crossing

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


def draw_pd_study(
    design: PDDesign, rho: float, rho_x: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(readings, lineage, truth): readings per state (x of n), the lineage rows and, per
    behavior, the true ρ_b, ρ^x_b and trace share after clipping the probabilities to [0, 1]."""
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
        den = a - sh
        truth.append(
            {
                "behavior_id": f"b{b}",
                "rho": (lat - ss) / den,
                "rho_x": (lat - sx) / den,
                "trace_share": (sx - ss) / (lat - ss) if lat != ss else math.nan,
                "clipped": lat in (0.0, 1.0) or sx in (0.0, 1.0),
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
    """The population means of the behaviors' true ρ_b and ρ^x_b after clipping the
    probabilities to [0, 1]: the values the mean's intervals estimate. They differ from the
    scenario's nominal (ρ, ρ^x) where clipping binds."""
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
                readings, lineage, level=level, n_boot=n_boot, delta_pd=delta_pd, seed=seed + r
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
    """The null boundaries of the amendment's §7, and interior and power scenarios."""
    return {
        "A at +rho_A (false A)": (rho_a, rho_a),
        "A at -rho_A (false A)": (-rho_a, -rho_a),
        "B at rho_B, main condition (false B)": (rho_b, rho_b),
        "B at rho^x = rho_A, corrected condition (false B)": (high, rho_a),
        "trace at rho^x = +rho_A (false trace)": (high, rho_a),
        "trace at rho^x = -rho_A (false trace)": (high, -rho_a),
        "power: A (rho = 0)": (0.0, 0.0),
        "power: B (rho = rho^x = 0.9)": (0.9, 0.9),
        "power: trace (rho = 0.9, rho^x = 0)": (0.9, 0.0),
        "interior (0.5, 0.5)": (0.5, 0.5),
    }


# ---------------------------------------------------------------------------------------------
# Certificate power from a calibrated curve


def _curve(steps: Sequence[float], rates: Sequence[float]):
    """A monotone rate(step) through the measured points, flat beyond the last one."""
    s = np.asarray(steps, dtype=float)
    r = np.maximum.accumulate(np.asarray(rates, dtype=float))
    return lambda step: np.interp(step, s, r)


def _run_time(
    rate_at,
    *,
    schedule: Sequence[int],
    target: float,
    responses: int,
    rng: np.random.Generator,
) -> tuple[float | None, bool]:
    """(T, censored) of one simulated relearning run evaluated at ``schedule`` (step 0 first)."""
    steps, rates = [0.0], [rng.binomial(responses, float(rate_at(0.0))) / responses]
    if rates[0] >= target:
        return 1.0, False
    for step in schedule:
        observed = rng.binomial(responses, float(rate_at(float(step)))) / responses
        steps.append(float(step))
        rates.append(observed)
        if observed >= target:
            t = interpolated_crossing(steps, rates, float(step), target)
            return max(1.0, t if t is not None else float(step)), False
    return None, True


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
    evaluation is binomial over ``responses``. Certified: no censored latent run, T_sham > T_lat
    for every seed pair, and S (ratio of geometric means) ≥ ``min_savings``, at both rates.
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
                logs, ok = [], True
                for _ in range(seeds):
                    scale = math.exp(rng.normal(0, seed_sd))
                    t_sham, c_sham = _run_time(
                        lambda st, f=sham_curve, k=scale: f(st / k),
                        schedule=evals,
                        target=target,
                        responses=responses,
                        rng=rng,
                    )
                    t_lat, c_lat = _run_time(
                        lambda st, f=sham_curve, k=scale, fast=s_true: f(st * fast / k),
                        schedule=evals,
                        target=target,
                        responses=responses,
                        rng=rng,
                    )
                    if c_lat:
                        ok = False
                        break
                    t_s = float(budget) if c_sham else t_sham
                    if t_s <= t_lat:
                        ok = False
                    logs.append((math.log(t_s), math.log(t_lat)))
                if ok and logs:
                    s_hat = math.exp(np.mean([a for a, _ in logs]) - np.mean([b for _, b in logs]))
                    ok = s_hat >= min_savings
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
            selected = dict(scenarios)
            # A negative ρ (latent below sham_s) has room only with a raised sham_s.
            negative = {k: v for k, v in selected.items() if v[0] < 0}
            for k in negative:
                selected.pop(k)
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
                raised = dataclasses.replace(design, sham_s_shift=tuple(config["negative_shift"]))
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


def boundary_checks(table: pd.DataFrame, *, max_error: float = 0.05, min_coverage: float = 0.85):
    """Per design, the worst decision error at the boundaries and the worst coverage."""
    rows = []
    errors = {
        "false A": "p_A",
        "false B": "p_B",
        "false trace": "p_PD_reads_the_suppression_trace",
    }
    for (behaviors, seeds), part in table.groupby(["behaviors", "seeds"]):
        worst = {}
        for label, column in errors.items():
            mask = part["scenario"].str.contains(f"({label})", regex=False)
            worst[label] = float(part.loc[mask, column].max()) if mask.any() else None
        coverage = float(part[["coverage_mean_rho", "coverage_mean_rho_x"]].min(axis=None))
        rows.append(
            {
                "behaviors": behaviors,
                "seeds": seeds,
                **{f"worst_{k.replace(' ', '_')}": v for k, v in worst.items()},
                "worst_mean_coverage": coverage,
                "errors_ok": all(v is None or v <= max_error for v in worst.values()),
                "coverage_ok": coverage >= min_coverage,
            }
        )
    return pd.DataFrame(rows)
