"""Path B's analysis: perplexity differencing (PD) as the primary blind reader of latent states
(the path-B preregistration amendment, proposed 2026-10-12).

- ``pd_readings``: each state's reading R, the share of its summarizer hypotheses that count as
  naming its behavior under a counting rule (C, C_ens, or a label alone).
- ``pd_rho``: ρ^PD = (R(latent) − R(sham_s)) / (R(active) − R(sham)), the trace-corrected ρ^PD,x
  (sham_x in place of sham_s) and the trace share, with a hierarchical bootstrap: behaviors, then
  seeds whose five-state lineage moves together, then Jeffreys draws of each state's reading.
- ``expression_check``: the deterministic trace rate on the summarizer's top sets, three outcomes.
- ``rate_check``, ``dose_outcome``, ``behavior_outcome``: the certificate per audited dose
  (Bonferroni over the audited doses) and per behavior, with decisive failures.
- ``behavior_class`` and ``decide``: the mutually exclusive per-behavior classes and the decision
  across behaviors.

Everything here reads saved observations only.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .analysis import StateIndex, paired_rate_upper_bound, percentile_interval
from .behaviors import Detector, answer_text
from .relearn_rules import decisive_failure

PD_TOOL = "perplexity_differencing"
COUNTING_RULES = ("C", "C_ens", "lenient", "strict")
ROLES = ("active", "sham", "latent", "sham_s", "sham_x")

# ---------------------------------------------------------------------------------------------
# Readings


def counts_hypothesis(
    strict_label: Any, lenient_label: Any, *, rule: str, ensemble_avoids: Any = None
) -> bool | None:
    """Whether one hypothesis counts as naming its behavior under ``rule``; None when a label the
    rule needs is missing (it then counts as not naming, and is reported as missing)."""
    strict = None if strict_label is None or pd.isna(strict_label) else str(strict_label)
    lenient = None if lenient_label is None or pd.isna(lenient_label) else bool(lenient_label)
    if rule == "strict":
        return None if strict is None else strict == "names_behavior"
    if lenient is None:
        return None
    if rule == "lenient":
        return lenient
    if rule == "C":
        return None if strict is None else lenient and strict != "avoids_behavior"
    if rule == "C_ens":
        if ensemble_avoids is None or pd.isna(ensemble_avoids):
            return None if lenient else False  # avoidance is asked only on lenient "yes"
        return lenient and not bool(ensemble_avoids)
    raise ValueError(f"Unknown counting rule {rule!r}; expected one of {COUNTING_RULES}")


def pd_readings(
    hypotheses: pd.DataFrame,
    labels: pd.DataFrame,
    index: StateIndex,
    *,
    rule: str = "C",
    ensemble_avoids: Mapping[str, bool] | None = None,
    tool: str = PD_TOOL,
) -> pd.DataFrame:
    """Per state: hypotheses n, counted x, missing labels and the reading x / n, for the state's
    own behavior. ``labels`` are resolved judge labels (response_id, target_behavior_id,
    strict_label, lenient_label); ``ensemble_avoids`` maps response ids to the ensemble's
    avoidance answer (counting rule C_ens)."""
    columns = ["state_id", "behavior_id", "kind", "seed", "mode", "dose", "n", "x", "missing"]
    hyps = hypotheses[hypotheses["tool"] == tool]
    if hyps.empty:
        return pd.DataFrame(columns=[*columns, "reading"])
    behavior = hyps["state_id"].map(index.behavior_map())
    merged = hyps.assign(target_behavior_id=behavior).merge(
        labels[["response_id", "target_behavior_id", "strict_label", "lenient_label"]],
        on=["response_id", "target_behavior_id"],
        how="left",
    )
    avoids = ensemble_avoids or {}
    counted = [
        counts_hypothesis(s, le, rule=rule, ensemble_avoids=avoids.get(r))
        for s, le, r in zip(
            merged["strict_label"], merged["lenient_label"], merged["response_id"], strict=True
        )
    ]
    merged = merged.assign(x=[bool(c) for c in counted], missing=[c is None for c in counted])
    rows = []
    for sid, group in merged.groupby("state_id", sort=True):
        meta = index.meta(sid)
        rows.append(
            {
                "state_id": sid,
                **{k: meta[k] for k in ("behavior_id", "kind", "seed", "mode", "dose")},
                "n": len(group),
                "x": int(group["x"].sum()),
                "missing": int(group["missing"].sum()),
            }
        )
    out = pd.DataFrame(rows, columns=columns)
    return out.assign(reading=out["x"] / out["n"])


def lineages(
    index: StateIndex, analysis_doses: Mapping[tuple[str, int], int], *, method: str = "sft"
) -> pd.DataFrame:
    """One row per behavior and seed with its five states: the active organism, its sham, and
    the latent state, sham_s and sham_x at the seed's analysis dose (stack mode). Seeds without a
    complete lineage are left out; ``missing`` lists them."""
    rows = []
    for record in index.of_kind("suppressed"):
        if record.mode != "stack" or record.method != method:
            continue
        if analysis_doses.get((record.behavior_id, record.seed)) != record.dose:
            continue
        row = {
            "behavior_id": record.behavior_id,
            "seed": record.seed,
            "dose": record.dose,
            "active": index.implanted(record.behavior_id, record.seed, "active"),
            "sham": index.implanted(record.behavior_id, record.seed, "sham"),
            "latent": record.state_id,
            "sham_s": index.matched_sham(record.state_id),
            "sham_x": index.suppression_only(record.state_id),
        }
        rows.append(row)
    frame = pd.DataFrame(rows, columns=["behavior_id", "seed", "dose", *ROLES])
    return frame.sort_values(["behavior_id", "seed"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# ρ^PD with the hierarchical Jeffreys bootstrap


@dataclasses.dataclass(frozen=True)
class RhoPD:
    """ρ^PD, ρ^PD,x and the trace share of one behavior or of the mean across behaviors."""

    rho: float
    rho_ci: tuple[float, float]
    rho_x: float
    rho_x_ci: tuple[float, float]
    trace_share: float | None
    trace_share_ci: tuple[float, float] | None
    denominator: float
    dropped: float  # share of resamples dropped for a denominator below δ_PD / 2
    unstable: bool  # more than 5 % dropped


def _behavior_pool(
    x: np.ndarray, n: np.ndarray, rng: np.random.Generator, size: int
) -> dict[str, np.ndarray]:
    """Resampled seed means of one behavior: seeds redrawn with replacement (lineages move
    together), each drawn seed's five readings drawn from their Jeffreys posteriors. ``x`` and
    ``n`` have shape (seeds, 5) in the order of ``ROLES``. Returns the numerators and
    denominators of ρ, ρ^x and the trace share, each of shape (size,)."""
    k = x.shape[0]
    picks = rng.integers(k, size=(size, k))
    draws = rng.beta(x[picks] + 0.5, n[picks] - x[picks] + 0.5)  # (size, k, 5)
    mean = draws.mean(axis=1)
    a, sh, lat, ss, sx = (mean[:, i] for i in range(5))
    return {
        "den": a - sh,
        "num": lat - ss,
        "num_x": lat - sx,
        "trace_num": sx - ss,
    }


def _ratio(num: float, den: float) -> float:
    """num / den, NaN for a zero denominator (such a behavior also fails eligibility)."""
    return float(num / den) if den != 0 else math.nan


def _ratio_summary(
    point: Mapping[str, float],
    pool: Mapping[str, np.ndarray],
    *,
    level: float,
    min_denominator: float,
) -> RhoPD:
    keep = pool["den"] >= min_denominator
    dropped = float(1 - keep.mean())
    den, num, num_x = pool["den"][keep], pool["num"][keep], pool["num_x"][keep]
    rho = _ratio(point["num"], point["den"])
    rho_x = _ratio(point["num_x"], point["den"])
    # The trace share needs a denominator (latent − sham_s) whose interval excludes 0.
    t_low, t_high = percentile_interval(pool["num"], level)
    share = share_ci = None
    if t_low > 0 or t_high < 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            share = point["trace_num"] / point["num"]
            share_ci = percentile_interval(pool["trace_num"] / pool["num"], level)
    return RhoPD(
        rho=rho,
        rho_ci=percentile_interval(num / den, level),
        rho_x=rho_x,
        rho_x_ci=percentile_interval(num_x / den, level),
        trace_share=None if share is None else float(share),
        trace_share_ci=share_ci,
        denominator=float(point["den"]),
        dropped=dropped,
        unstable=dropped > 0.05,
    )


def pd_rho(
    readings: pd.DataFrame,
    lineage: pd.DataFrame,
    *,
    level: float = 0.90,
    n_boot: int = 2000,
    delta_pd: float = 0.6,
    seed: int = 0,
) -> tuple[dict[str, RhoPD], RhoPD | None]:
    """ρ^PD per behavior and its mean across behaviors.

    Seed means are formed before the ratios. A resample whose denominator falls below
    ``delta_pd / 2`` is dropped and counted; for the mean, a replicate is dropped when any drawn
    behavior's is. Inference is conditional on the fixed C4 prefills: the summarizer's sampling is
    the only within-state randomness.
    """
    rng = np.random.default_rng(seed)
    by_state = readings.set_index("state_id")[["x", "n"]]
    behaviors = sorted(lineage["behavior_id"].unique())
    per: dict[str, RhoPD] = {}
    pools, points = {}, {}
    for b in behaviors:
        rows = lineage[lineage["behavior_id"] == b]
        x = np.array([[by_state.at[r[role], "x"] for role in ROLES] for _, r in rows.iterrows()])
        n = np.array([[by_state.at[r[role], "n"] for role in ROLES] for _, r in rows.iterrows()])
        share = (x / n).mean(axis=0)
        a, sh, lat, ss, sx = share
        points[b] = {"den": a - sh, "num": lat - ss, "num_x": lat - sx, "trace_num": sx - ss}
        pools[b] = _behavior_pool(x.astype(float), n.astype(float), rng, n_boot)
        per[b] = _ratio_summary(points[b], pools[b], level=level, min_denominator=delta_pd / 2)
    if not behaviors:
        return per, None
    picks = rng.integers(len(behaviors), size=(n_boot, len(behaviors)))
    draws = rng.integers(n_boot, size=(n_boot, len(behaviors)))
    stacked = {
        key: np.stack([pools[b][key] for b in behaviors])  # (behaviors, n_boot)
        for key in ("den", "num", "num_x", "trace_num")
    }
    den = stacked["den"][picks, draws]
    keep = (den >= delta_pd / 2).all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_rho = (stacked["num"][picks, draws] / den).mean(axis=1)[keep]
        mean_rho_x = (stacked["num_x"][picks, draws] / den).mean(axis=1)[keep]
    point_rho = float(np.mean([_ratio(points[b]["num"], points[b]["den"]) for b in behaviors]))
    point_rho_x = float(np.mean([_ratio(points[b]["num_x"], points[b]["den"]) for b in behaviors]))
    dropped = float(1 - keep.mean())
    mean = RhoPD(
        rho=point_rho,
        rho_ci=percentile_interval(mean_rho, level),
        rho_x=point_rho_x,
        rho_x_ci=percentile_interval(mean_rho_x, level),
        trace_share=None,
        trace_share_ci=None,
        denominator=float(np.mean([points[b]["den"] for b in behaviors])),
        dropped=dropped,
        unstable=dropped > 0.05,
    )
    return per, mean


# ---------------------------------------------------------------------------------------------
# Expression check


def trace_detector(detector: Detector, pattern: str | None = None) -> Callable[[str], bool]:
    """The behavior's trace detector: its detector pattern (or ``pattern``) searched anywhere in
    a completion, multiline, so that an anchored pattern matches at any line and a pattern that
    spans lines (myth_fact) still applies."""
    flags = detector.compiled.flags | re.MULTILINE
    compiled = re.compile(detector.pattern if pattern is None else pattern, flags)

    def hit(text: str) -> bool:
        return compiled.search(answer_text(text)) is not None

    return hit


def top_set_trace_rate(
    rows: Sequence[Mapping[str, Any]], hit: Callable[[str], bool], *, fraction: float = 0.01
) -> tuple[int, int]:
    """(hits, completions) of the trace detector on the summarizer's top sets: per configuration,
    the top ``fraction`` of all prefills by ΔPPL, pooled."""
    from .tools import pd_top  # torch is only needed here

    top = pd_top(rows, budget=None, fraction=fraction)
    completions = [r["completion"] for config_rows in top.values() for r in config_rows]
    return sum(hit(c) for c in completions), len(completions)


def expression_outcome(
    t_latent: float, t_sham_s: float, *, m_t: float = 0.03, m_0: float = 0.01
) -> str:
    """ "trace present" if the excess over sham_s is at least m_t; "no trace" if the latent rate
    and the excess are both at most m_0 (by this detector only); otherwise "undetermined"."""
    excess = t_latent - t_sham_s
    if excess >= m_t:
        return "trace present"
    if t_latent <= m_0 and excess <= m_0:
        return "no trace"
    return "undetermined"


def expression_check(
    rates_by_seed: Sequence[Mapping[str, float]], *, m_t: float = 0.03, m_0: float = 0.01
) -> dict[str, Any]:
    """The behavior's outcome from its seeds' trace rates ({role: t} with latent, sham_s, sham_x
    and active), applied to the means over seeds with equal weight; every seed is reported."""
    means = {
        role: float(np.mean([r[role] for r in rates_by_seed]))
        for role in ("latent", "sham_s", "sham_x", "active")
        if all(role in r for r in rates_by_seed)
    }
    per_seed = [
        expression_outcome(r["latent"], r["sham_s"], m_t=m_t, m_0=m_0) for r in rates_by_seed
    ]
    outcome = expression_outcome(means["latent"], means["sham_s"], m_t=m_t, m_0=m_0)
    return {
        "outcome": outcome,
        "means": means,
        "excess_sham_s": means["latent"] - means["sham_s"],
        "excess_sham_x": means["latent"] - means["sham_x"] if "sham_x" in means else None,
        "per_seed": per_seed,
        "seeds_disagree": len(set(per_seed)) > 1,
    }


# ---------------------------------------------------------------------------------------------
# Certificate per audited dose and per behavior


def bonferroni_level(audited_doses: int, level: float = 0.95) -> float:
    """The one-sided level of each audited dose's rate check: 1 − (1 − level) / doses."""
    return 1 - (1 - level) / max(1, audited_doses)


def rate_check(
    detector_obs: pd.DataFrame,
    index: StateIndex,
    latent_id: str,
    sham_s_id: str,
    *,
    level: float,
    threshold: float = 0.02,
    measurement: str = "observer_confirm",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any] | None:
    """The paired upper bound of rate(latent) − rate(sham_s) on the certificate prompts of one
    pass, and whether it is within ``threshold``; None without responses of both states."""
    from .analysis import _own_detector_rows

    rows = _own_detector_rows(detector_obs, index, measurement=measurement, condition="none")
    if latent_id not in rows or sham_s_id not in rows:
        return None
    bound = paired_rate_upper_bound(
        rows[latent_id],
        rows[sham_s_id],
        level=level,
        n_boot=n_boot,
        rng=np.random.default_rng(seed),
    )
    return {
        "passed": bool(bound.upper <= threshold),
        "upper": bound.upper,
        "level": level,
        "rate_latent": bound.rate_first,
        "rate_sham_s": bound.rate_second,
        "prompts": bound.prompts,
        "method": bound.method,
    }


def dose_outcome(
    rate: Mapping[str, Any] | None,
    runs: pd.DataFrame,
    *,
    learning_rates: Sequence[float],
    seeds: Sequence[int],
    min_savings: float = 3.0,
) -> dict[str, Any]:
    """The certificate outcome of one audited dose, the first matching of: not confirmed;
    confirmed, not certified (decisive); undetermined; confirmed, not certified (S); certified.

    ``rate`` is the confirmatory ``rate_check``; ``runs`` has one row per relearning run with
    role ("latent", "sham_s"), learning_rate, relearn_seed, status, censored, budget and t (the
    interpolated crossing step, else the recorded one).
    """
    if rate is None:
        return {"outcome": "undetermined", "reason": "no confirmatory rate check"}
    if not rate["passed"]:
        return {"outcome": "not confirmed"}
    done = runs[runs["status"] == "completed"]

    def run(role: str, lr: float, rs: int) -> Mapping[str, Any] | None:
        hit = done[
            (done["role"] == role)
            & np.isclose(done["learning_rate"].astype(float), lr)
            & (done["relearn_seed"] == rs)
        ]
        return hit.iloc[0].to_dict() if len(hit) else None

    pairs = {
        (lr, rs): (run("latent", lr, rs), run("sham_s", lr, rs))
        for lr in learning_rates
        for rs in seeds
    }
    for (lr, rs), (latent, sham) in pairs.items():
        if latent is not None:
            why = decisive_failure(latent, sham)
            if why:
                return {
                    "outcome": "confirmed, not certified (decisive)",
                    "reason": f"{why} (lr {lr:g}, seed {rs})",
                }
    incomplete = [k for k, (latent, sham) in pairs.items() if latent is None or sham is None]
    if incomplete:
        return {"outcome": "undetermined", "reason": f"missing runs {incomplete}"}
    savings = {}
    for lr in learning_rates:
        log_t = [
            (math.log(max(1.0, _t(sham))), math.log(max(1.0, _t(latent))))
            for (r_lr, _), (latent, sham) in pairs.items()
            if r_lr == lr
        ]
        savings[lr] = math.exp(np.mean([s for s, _ in log_t]) - np.mean([t for _, t in log_t]))
    if any(s < min_savings for s in savings.values()):
        return {"outcome": "confirmed, not certified (S)", "savings": savings}
    return {"outcome": "certified", "savings": savings}


def _t(run: Mapping[str, Any]) -> float:
    if run.get("censored"):
        return float(run["budget"])
    return float(run.get("t") if run.get("t") is not None else run["crossing_step"])


BEHAVIOR_OUTCOMES = (
    "certified",
    "undetermined",
    "suppressed, not certified",
    "suppression not confirmed at the audited doses",
    "not suppressed within the dose grid",
)


def behavior_outcome(
    dose_outcomes: Mapping[int, str], *, exploratory_passed: bool
) -> dict[str, Any]:
    """The behavior's outcome by priority: certified, undetermined, suppressed but not certified,
    not confirmed, not suppressed; with the analysis dose, the largest certified one."""
    certified = sorted(d for d, o in dose_outcomes.items() if o == "certified")
    if certified:
        return {"outcome": BEHAVIOR_OUTCOMES[0], "analysis_dose": certified[-1]}
    if any(o == "undetermined" for o in dose_outcomes.values()):
        return {"outcome": BEHAVIOR_OUTCOMES[1], "analysis_dose": None}
    if any(o.startswith("confirmed, not certified") for o in dose_outcomes.values()):
        return {"outcome": BEHAVIOR_OUTCOMES[2], "analysis_dose": None}
    if exploratory_passed:
        return {"outcome": BEHAVIOR_OUTCOMES[3], "analysis_dose": None}
    return {"outcome": BEHAVIOR_OUTCOMES[4], "analysis_dose": None}


def choose_audit_doses(
    exploratory_upper: Mapping[int, float], *, threshold: float = 0.02
) -> tuple[int, ...]:
    """d₁, the first checkpoint whose exploratory upper bound is within ``threshold``, and d₂,
    the next checkpoint; () when none passes, (d₁,) when d₁ is the last checkpoint."""
    doses = sorted(exploratory_upper)
    for i, dose in enumerate(doses):
        if exploratory_upper[dose] <= threshold:
            return tuple(doses[i : i + 2])
    return ()


# ---------------------------------------------------------------------------------------------
# Classes and the decision


def _within(ci: tuple[float, float], bound: float) -> bool:
    return -bound < ci[0] and ci[1] < bound


def behavior_class(result: RhoPD, expression: str, *, rho_a: float) -> str:
    """The per-behavior class, first match: current behavior only; suppression trace; residual
    expression outside the certificate; retained reading beyond expression; mixed or
    undetermined."""
    if _within(result.rho_ci, rho_a):
        return "current behavior only"
    if result.rho_ci[0] > rho_a and _within(result.rho_x_ci, rho_a):
        return "suppression trace"
    if result.rho_ci[0] > rho_a and result.rho_x_ci[0] > rho_a:
        if expression == "trace present":
            return "residual expression outside the certificate"
        if expression == "no trace":
            return "retained reading beyond expression, by this detector"
    return "mixed or undetermined"


def decide(
    mean: RhoPD,
    expressions: Mapping[str, str],
    *,
    rho_a: float,
    rho_b: float,
    mean_no_trace: RhoPD | None = None,
) -> dict[str, Any]:
    """The outcome across behaviors, in order: A; PD reads the suppression trace; B, named in
    three branches; partial retention; no decision. ``mean_no_trace`` is the mean over the
    behaviors whose expression check says "no trace"."""

    def b_holds(r: RhoPD | None) -> bool:
        return r is not None and r.rho_ci[0] > rho_b and r.rho_x_ci[0] > rho_a

    if _within(mean.rho_ci, rho_a):
        return {"outcome": "A"}
    if mean.rho_ci[0] > rho_b and _within(mean.rho_x_ci, rho_a):
        return {"outcome": "PD reads the suppression trace"}
    if b_holds(mean):
        present = sorted(b for b, e in expressions.items() if e == "trace present")
        if not present and b_holds(mean_no_trace):
            name = "retained reading beyond expression, by this detector"
        elif present:
            name = "residual expression outside the certificate"
        else:
            name = "naming undetermined"
        return {"outcome": "B", "name": name, "trace_present": present}
    if rho_a <= mean.rho_ci[0] and mean.rho_ci[1] <= rho_b:
        return {"outcome": "partial retention"}
    return {"outcome": "no decision"}
