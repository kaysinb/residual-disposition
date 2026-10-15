"""Stage-0 simulations of the path-B continuation (docs/preregistration.md §12.3.8).

Items, each with its acceptance rule where §12.3.8 gives one:

1. ``reader_cell``: the certified final decisions and size of one reader test (δ₀, form, α), by
   ``continuation_exact``; ``one_sample_binomial``: the enumeration of PD's absolute test.
2. ``sign_test_size``: Monte Carlo of the certificate's sign test with its censoring rule.
3. ``union_size``: Monte Carlo of the per-lineage intersection–union with one component at its
   null boundary and the others at strong alternatives. The reader components enter through
   their certified decision regions at the per-case level (a decision-level stand-in: p = level
   when rejected, 1 otherwise), which is all the intersection–union reads at that level.
4. ``operating_points``: the reader thresholds per control count, frozen from item 1.
5. ``sign_test_power`` and ``panel_power``: reported only.
6. ``selection_study``: the margin rule with fresh confirmation draws, reported only.

Seeds: the calibration seed of the rules sets the censoring budgets; the validation seed drives
every Monte Carlo replicate; each item and scenario has its own stream (``derive_seed``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np
from scipy import stats

from . import analysis as A
from . import continuation_rules as cr
from . import continuation_stats as cs
from .continuation_exact import ShiftedDifferenceTest
from .seeding import derive_seed

# ---------------------------------------------------------------------------------------------
# Item 1: reader tests


def reader_forms(rules: cr.Rules) -> list[tuple[str, Fraction, str]]:
    """(name, δ₀, form) of the four reader tests (§12.5, components 3 and 4)."""
    margin = rules.fraction("readers", "pd", "equivalence_margin")
    return [
        ("adl_vs_sham", rules.fraction("readers", "adl", "excess_over_sham_s"), ">"),
        ("adl_vs_x", rules.fraction("readers", "adl", "excess_over_x"), ">"),
        ("pd_lower", -margin, ">"),
        ("pd_upper", margin, "<"),
    ]


def make_test(rules: cr.Rules, n: int, delta: Fraction, form: str) -> ShiftedDifferenceTest:
    exact = rules["readers"]["exact_test"]
    return ShiftedDifferenceTest(
        n,
        n,
        delta,
        form,
        gamma=rules.fraction("readers", "exact_test", "berger_boos_gamma"),
        digits=int(exact["digits"]),
        gap=rules.fraction("readers", "exact_test", "gap"),
        max_digits=int(exact["max_digits"]),
    )


def reader_cell(test: ShiftedDifferenceTest, name: str, alpha: Fraction) -> dict[str, Any]:
    """All final decisions at level α, the structural checks and the certified size."""
    start = time.monotonic()
    out: dict[str, Any] = {
        "test": name,
        "delta": str(test.delta),
        "form": test.form,
        "alpha": str(alpha),
        "n": test.n_l,
        "monotone": test.monotone,
        "nested": test.tail_sets_nested,
    }
    region = test.decision_region(alpha)
    size = test.certify_size(region["tau_size"], alpha)
    out.update(
        convex=region["convex"],
        undecided=[list(map(int, u)) for u in region["undecided"]],
        tau_use=[int(x) for x in region["tau_use"]],
        tau_size=[int(x) for x in region["tau_size"]],
        size_upper=str(size["upper"]),
        size_lower=str(size["lower"]),
        size_float=float(size["upper"]),
        passed=bool(out["monotone"] and out["nested"] and size["passed"]),
        seconds=round(time.monotonic() - start, 1),
    )
    return out


def _reader_cell_job(args: tuple) -> dict[str, Any]:
    rules_path, rules_sha256, n, name, delta, form, alpha = args
    rules = cr.load_rules(rules_path, expected_hash=rules_sha256)
    return reader_cell(make_test(rules, n, delta, form), name, alpha)


# ---------------------------------------------------------------------------------------------
# The frozen specification


# The frozen specification of the official run (§12.3.8): the sha256 of the rules and of the
# resolved ``simulation.continuation`` section of configs/experiments/simulate_continuation.yaml.
# They live in this module, not in the config, so that no command-line override can change them;
# an official run also needs a clean commit, which pins this file. Change them only by a
# reviewed commit, before the official run.
FROZEN = {
    "rules_sha256": "00bfde8b8e3cabd362f31c2af3322e990633239cdf0d31b9097122d0f8abed07",
    "spec_sha256": "22b5ccafb851137c7dcf0d3ea6797147befbe8758e7e4d7017204a73f8190397",
}


def spec_hash(spec: Mapping[str, Any]) -> str:
    """Full sha256 of the resolved simulation specification (canonical JSON, as the rules)."""
    return cr.rules_hash(spec)


def run_status(spec: Mapping[str, Any], rules: cr.Rules, git: Mapping[str, Any]) -> dict[str, Any]:
    """Whether a run can serve Stage-0 acceptance (§12.3.8: the specification is frozen before
    the run). Official only when the rules and the resolved specification hash to ``FROZEN`` and
    the source is a clean commit; otherwise the run is a diagnostic one, whatever its checks
    say. Nothing from the config or the command line enters the expected values."""
    frozen = FROZEN
    digest = spec_hash(spec)
    reasons = []
    if rules.sha256 != frozen.get("rules_sha256"):
        reasons.append(f"rules sha256 {rules.sha256} != frozen {frozen.get('rules_sha256')}")
    if digest != frozen.get("spec_sha256"):
        reasons.append(f"specification sha256 {digest} != frozen {frozen.get('spec_sha256')}")
    if git.get("commit") is None or git.get("dirty") is not False:
        reasons.append("the source is not a clean commit")
    return {
        "official": not reasons,
        "reasons": reasons,
        "rules_sha256": rules.sha256,
        "spec_sha256": digest,
        "commit": git.get("commit"),
    }


def one_sample_binomial(n: int, bound: Fraction, levels: Sequence[Fraction]) -> dict[str, Any]:
    """The exact test of H₀ rate ≥ bound, enumerated over 0..n: its rejection counts and size.

    P(X ≤ k) falls as the rate rises, so the size over H₀ is attained at rate = bound.
    """
    p_values = []
    for k in range(n + 1):
        p_values.append(cs.rate_below_p(k, n, bound))
        if p_values[-1] > max(levels):
            break
    out: dict[str, Any] = {"n": n, "bound": str(bound), "levels": {}}
    out["p_values"] = {k: float(p) for k, p in enumerate(p_values)}
    for alpha in levels:
        k_max = max((k for k, p in enumerate(p_values) if p <= alpha), default=-1)
        size = cs.binomial_cdf(k_max, n, bound)
        out["levels"][str(alpha)] = {
            "reject_at_most": k_max,
            "size": str(size),
            "size_float": float(size),
            "passed": size <= alpha,
        }
    return out


# ---------------------------------------------------------------------------------------------
# Item 2: the certificate's sign test


def draw_contrasts(
    rng: np.random.Generator, name: str, size: Any, *, scale: float, median: float = 0.0
) -> np.ndarray:
    """Log-contrasts with the given median: ``median + scale · z``, z of median 0."""
    if name == "normal":
        z = rng.standard_normal(size)
    elif name == "skew_normal":
        z = stats.skewnorm.rvs(5, size=size, random_state=rng) - stats.skewnorm.median(5)
    elif name == "lognormal_shifted":
        z = np.exp(rng.standard_normal(size)) - 1.0
    elif name == "t3":
        z = rng.standard_t(3, size)
    elif name == "contaminated":  # 10 % of draws with ten times the scale
        z = rng.standard_normal(size) * np.where(rng.random(size) < 0.1, 10.0, 1.0)
    else:
        raise ValueError(f"unknown contrast distribution {name!r}")
    return median + scale * z


def _arm_times(
    rng: np.random.Generator, contrasts: np.ndarray, log_t: float, log_sd: float
) -> tuple[np.ndarray, np.ndarray]:
    """Latent and sham_s crossing times whose log ratio is the contrast."""
    latent = np.exp(log_t + log_sd * rng.standard_normal(contrasts.shape))
    return latent, latent * np.exp(contrasts)


def _crossing(t: float, budget: float) -> cs.Crossing:
    return cs.Crossing.observed(float(t)) if t <= budget else cs.Crossing.censored(budget)


def censoring_budget(
    rng: np.random.Generator,
    distribution: str,
    rate: float,
    spec: Mapping[str, Any],
    habits: Sequence[str],
) -> float:
    """The step budget at which a share ``rate`` of all runs (every arm, under H₀) is censored."""
    if rate <= 0:
        return math.inf
    draws = int(spec["calibration_draws"])
    times = []
    for habit in habits:
        d = draw_contrasts(rng, distribution, draws, scale=float(spec["scale"]))
        times += list(_arm_times(rng, d, float(spec["log_t_latent"][habit]), spec["log_t_sd"]))
    return float(np.quantile(np.concatenate(times), 1 - rate))


def _block_contrasts(
    rng: np.random.Generator,
    kind: str,
    distribution: str,
    blocks: int,
    budget: float,
    spec: Mapping[str, Any],
    *,
    median: float = 0.0,
    scale: float | None = None,
) -> list[cs.Bounds]:
    """One replicate's block contrasts, through the certificate's own censoring rule.

    S: d = log T(H_s) − log T(L) of habit b. Q: S_b − S_Y on four arms, with S_b of the given
    median and S_Y of median 0 from the same distribution.
    """
    scale = float(spec["scale"]) if scale is None else scale
    log_sd = float(spec["log_t_sd"])
    d_b = draw_contrasts(rng, distribution, blocks, scale=scale, median=median)
    lat_b, sham_b = _arm_times(rng, d_b, float(spec["log_t_latent"]["b"]), log_sd)
    savings_b = [
        cs.savings_bounds(_crossing(lat_b[j], budget), _crossing(sham_b[j], budget))
        for j in range(blocks)
    ]
    if kind == "S":
        return savings_b
    d_y = draw_contrasts(rng, distribution, blocks, scale=scale)
    lat_y, sham_y = _arm_times(rng, d_y, float(spec["log_t_latent"]["Y"]), log_sd)
    return [
        cs.ratio_bounds(
            savings_b[j],
            cs.savings_bounds(_crossing(lat_y[j], budget), _crossing(sham_y[j], budget)),
        )
        for j in range(blocks)
    ]


def _six_arm_block(
    rng: np.random.Generator,
    block_id: int,
    medians: Mapping[str, float],
    spec: Mapping[str, Any],
) -> cs.Block:
    """One certificate block without censoring: the savings of each habit (b, Y, Y₂) of the
    given log median (normal, the item-2 scale); Y₂'s latent times are drawn like Y's."""
    arms = {}
    for habit in ("b", "Y", "Y2"):
        median = float(medians[habit])
        d = draw_contrasts(rng, "normal", 1, scale=float(spec["scale"]), median=median)
        log_t = float(spec["log_t_latent"]["b" if habit == "b" else "Y"])
        latent, sham = _arm_times(rng, d, log_t, float(spec["log_t_sd"]))
        arms[(habit, "L")] = cs.Crossing.observed(float(latent[0]))
        arms[(habit, "H_s")] = cs.Crossing.observed(float(sham[0]))
    return cs.Block(arms, block_id=block_id)


def mc_bound(level: Fraction | float, replicates: int) -> float:
    """level + 3 Monte Carlo standard errors at the level."""
    p = float(level)
    return p + 3 * math.sqrt(p * (1 - p) / replicates)


def sign_test_size(
    spec: Mapping[str, Any], *, blocks: int, calibration_seed: int, validation_seed: int
) -> list[dict[str, Any]]:
    """Rejection rate at the 6/6 level under median 0, per contrast kind, distribution and
    censoring rate; acceptance: ≤ 1/64 + 3 MC standard errors."""
    level = Fraction(1, 2**blocks)
    replicates = int(spec["replicates"])
    rows = []
    for kind in spec["contrasts"]:
        habits = ("b",) if kind == "S" else ("b", "Y")
        for distribution in spec["distributions"]:
            for rate in spec["censoring"]:
                cal = np.random.default_rng(
                    derive_seed(calibration_seed, "sign_test", kind, distribution, rate)
                )
                budget = censoring_budget(cal, distribution, float(rate), spec, habits)
                rng = np.random.default_rng(
                    derive_seed(validation_seed, "sign_test", kind, distribution, rate)
                )
                rejected = censored_signs = 0
                for _ in range(replicates):
                    contrasts = _block_contrasts(rng, kind, distribution, blocks, budget, spec)
                    test = cs.sign_test(contrasts)
                    rejected += test.p <= level
                    censored_signs += sum(1 for c in contrasts if not c.observed)
                estimate = rejected / replicates
                bound = mc_bound(level, replicates)
                rows.append(
                    {
                        "contrast": kind,
                        "distribution": distribution,
                        "censoring": float(rate),
                        "budget": budget,
                        "replicates": replicates,
                        "rejection_rate": estimate,
                        "bounded_contrasts_share": censored_signs / (replicates * blocks),
                        "acceptance_bound": bound,
                        "passed": estimate <= bound,
                    }
                )
    return rows


# ---------------------------------------------------------------------------------------------
# Item 3: per-lineage intersection–union


def least_favourable(test: ShiftedDifferenceTest, tau: Sequence[int], grid: int) -> float:
    """θ = p_C on the capped boundary where the region τ is most often entered (float search)."""
    lo, hi = (float(x) for x in test.theta_range)
    theta = np.linspace(lo, hi, grid)
    if test.kink is not None:
        theta = np.append(theta, float(test.kink))
    values = test._tail_float(np.asarray(tau), theta)
    return float(theta[int(np.argmax(values))])


def _observer_passes(prompts: int, tolerance: Fraction, level: Fraction, attempts: int) -> int:
    """The largest hit count whose p₁ is ≤ the level (p₁ rises with the hits)."""
    k = -1
    while cs.observer_p1(k + 1, prompts, tolerance, attempts) <= level:
        k += 1
    return k


def union_scenarios(
    tests: Mapping[str, ShiftedDifferenceTest],
    regions: Mapping[str, Sequence[int]],
    spec: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """True parameters per scenario (``spec["scenarios"]``): one component at its null boundary,
    the others at the strong alternatives of ``spec["strong"]``.

    A scenario sets only what differs from the strong alternatives: the observer's rates at the
    two attempts; the medians of the certificate's savings per habit at the first learning rate
    (the second rate stays strong); the ADL readings (L, H_s, X), or "least_favourable": the
    test's boundary point where its region is most often entered, the other ADL test strong; the
    PD readings (L, H_s, X).
    """
    strong = spec["strong"]
    theta = {
        name: least_favourable(tests[name], regions[name], int(spec["grid"])) for name in tests
    }
    out = {}
    for scenario, changes in spec["scenarios"].items():
        p: dict[str, Any] = {
            "observer": tuple(strong["observer"]),
            "certificate": None,
            "adl": tuple(strong["adl"]),
            "pd": tuple(strong["pd"]),
            "theta": None,
        }
        if "observer" in changes:
            p["observer"] = tuple(changes["observer"])
        if "certificate" in changes:
            p["certificate"] = dict(changes["certificate"])
        if changes.get("adl") == "least_favourable":
            th, delta = theta[scenario], float(tests[scenario].delta)
            p_l = min(1.0, th + delta)
            p["adl"] = (p_l, th, 0.0) if scenario == "adl_vs_sham" else (p_l, 0.0, th)
            p["theta"] = th
        elif "adl" in changes:
            p["adl"] = tuple(changes["adl"])
        if "pd" in changes:
            p["pd"] = tuple(changes["pd"])
        out[scenario] = p
    return out


def union_size(
    rules: cr.Rules,
    spec: Mapping[str, Any],
    sign_spec: Mapping[str, Any],
    tests: Mapping[str, ShiftedDifferenceTest],
    regions: Mapping[str, Sequence[int]],
    binomial_max: int,
    *,
    validation_seed: int,
) -> list[dict[str, Any]]:
    """Rejection rate of p_i = max(p₁…p₄) ≤ the per-case level, per scenario; acceptance:
    ≤ level + 3 MC standard errors."""
    level = rules.per_case_level
    n = next(iter(tests.values())).n_l
    prompts = int(rules["observer"]["prompts"])
    tolerance = rules.fraction("observer", "tolerance")
    attempts = rules.attempts
    design = cs.certificate_design(rules)
    rates = design["learning_rates"]
    observer_max = _observer_passes(prompts, tolerance, level, attempts)
    replicates = int(spec["replicates"])
    tau = {name: np.asarray(regions[name]) for name in regions}
    rows = []
    strong_medians = dict(spec["strong"]["certificate"])
    for scenario, truth in union_scenarios(tests, regions, spec).items():
        rng = np.random.default_rng(derive_seed(validation_seed, "union", scenario))
        counts = dict.fromkeys(("p1", "p2", "p3", "p4", "lineage"), 0)
        for _ in range(replicates):
            # p₁: attempt 1 at d₁, attempt 2 at d₂ when attempt 1 fails.
            p1 = Fraction(1)
            for rate in truth["observer"][:attempts]:
                hits = int(rng.binomial(prompts, rate))
                if hits <= observer_max:
                    p1 = cs.observer_p1(hits, prompts, tolerance, attempts)
                    break
            # p₂: six blocks of six arms at both rates, through the protocol check; the
            # scenario's medians at the first rate, the strong ones elsewhere.
            by_rate = {}
            for lr in rates:
                medians = strong_medians
                if truth["certificate"] is not None and lr == rates[0]:
                    medians = truth["certificate"]
                by_rate[lr] = [
                    _six_arm_block(rng, block_id, medians, sign_spec)
                    for block_id in design["block_ids"]
                ]
            p2 = cs.certificate_p(by_rate, **design).p
            # p₃: both ADL tests by their certified regions.
            p_l, p_h, p_x = truth["adl"]
            x_l, x_h, x_x = rng.binomial(n, (p_l, p_h, p_x))
            adl = x_l >= tau["adl_vs_sham"][x_h] and x_l >= tau["adl_vs_x"][x_x]
            p3 = level if adl else Fraction(1)
            # p₄: both TOSTs (two sides each) and the absolute rate.
            p_l, p_h, p_x = truth["pd"]
            x_l, x_h, x_x = rng.binomial(n, (p_l, p_h, p_x))
            pd = (
                all(tau["pd_lower"][x_c] <= x_l <= tau["pd_upper"][x_c] for x_c in (x_h, x_x))
                and x_l <= binomial_max
            )
            p4 = level if pd else Fraction(1)
            components = {"p1": p1, "p2": p2, "p3": p3, "p4": p4}
            for key, value in components.items():
                counts[key] += value <= level
            counts["lineage"] += cs.intersection_union(components) <= level
        estimate = counts["lineage"] / replicates
        bound = mc_bound(level, replicates)
        rows.append(
            {
                "scenario": scenario,
                "theta": truth["theta"],
                "observer_rates": list(truth["observer"]),
                "certificate_medians": truth["certificate"],
                "adl_rates": list(truth["adl"]),
                "pd_rates": list(truth["pd"]),
                "replicates": replicates,
                **{f"{k}_rate": v / replicates for k, v in counts.items() if k != "lineage"},
                "rejection_rate": estimate,
                "acceptance_bound": bound,
                "passed": estimate <= bound,
            }
        )
    return rows


# ---------------------------------------------------------------------------------------------
# Item 4: operating points


def operating_points(cells: Sequence[Mapping[str, Any]], shown: int = 11) -> dict[str, Any]:
    """Per test and level, the frozen thresholds τ(x_C) in use, with the first ``shown`` counts
    spelled out: ">" rejects x_L ≥ τ, "<" rejects x_L ≤ τ."""
    out: dict[str, Any] = {}
    for cell in cells:
        key = f"{cell['test']}@{cell['alpha']}"
        out[key] = {
            "form": cell["form"],
            "delta": cell["delta"],
            "tau": cell["tau_use"],
            "first": dict(enumerate(cell["tau_use"][:shown])),
        }
    return out


# ---------------------------------------------------------------------------------------------
# Item 5: power


def pilot_spreads(pairs: Sequence[Sequence[Any]]) -> dict[str, float]:
    """Pooled within-cell SD of the pilot's log-contrasts: S over every (behavior, dose, rate,
    habit) cell, Q = S_b − S_Y over the cells with both habits."""
    by_cell: dict[tuple, dict[int, float]] = {}
    for behavior, dose, lr, seed, habit, t_latent, t_sham in pairs:
        key = (behavior, int(dose), float(lr), habit)
        by_cell.setdefault(key, {})[int(seed)] = math.log(float(t_sham) / float(t_latent))
    s_cells = [list(v.values()) for v in by_cell.values()]
    q_cells = []
    for (behavior, dose, lr, habit), values in by_cell.items():
        other = by_cell.get((behavior, dose, lr, "Y"))
        if habit == "b" and other:
            q_cells.append([values[s] - other[s] for s in values if s in other])

    def pooled(cells: list[list[float]]) -> float:
        squares = sum(sum((x - np.mean(c)) ** 2 for x in c) for c in cells)
        dof = sum(len(c) - 1 for c in cells)
        return math.sqrt(squares / dof)

    return {"S": pooled(s_cells), "Q": pooled(q_cells)}


def sign_test_power(
    spec: Mapping[str, Any],
    sign_spec: Mapping[str, Any],
    pilot_sd: Mapping[str, float],
    budgets: Mapping[tuple[str, float], float],
    *,
    blocks: int,
    validation_seed: int,
) -> list[dict[str, Any]]:
    """P(6 of 6 certainly positive) for S-type contrasts of the given medians: the pilot-derived
    normal distributions (no censoring; the pilot had none) and the item-2 distributions at their
    item-2 budgets."""
    level = Fraction(1, 2**blocks)
    replicates = int(spec["replicates"])
    designs = [(f"pilot_{k}", "normal", sd, 0.0) for k, sd in pilot_sd.items()]
    designs += [
        (name, name, float(sign_spec["scale"]), float(rate))
        for name in sign_spec["distributions"]
        for rate in sign_spec["censoring"]
    ]
    rows = []
    for label, distribution, scale, rate in designs:
        budget = budgets.get((distribution, rate), math.inf) if rate > 0 else math.inf
        for median in spec["medians"]:
            rng = np.random.default_rng(
                derive_seed(validation_seed, "power", label, rate, float(median))
            )
            passed = 0
            for _ in range(replicates):
                contrasts = _block_contrasts(
                    rng,
                    "S",
                    distribution,
                    blocks,
                    budget,
                    sign_spec,
                    median=float(median),
                    scale=scale,
                )
                passed += cs.sign_test(contrasts).p <= level
            rows.append(
                {
                    "distribution": label,
                    "scale": scale,
                    "censoring": rate,
                    "median": float(median),
                    "replicates": replicates,
                    "power": passed / replicates,
                }
            )
    return rows


def panel_power(
    rules: cr.Rules, spec: Mapping[str, Any], *, validation_seed: int
) -> list[dict[str, Any]]:
    """P(p_PC ≤ α) for per-lineage pass probabilities π, independent or equicorrelated
    (Gaussian copula). A passing lineage enters with p = the per-case level, others with 1."""
    m, r, level = rules.m, rules.r, rules.per_case_level
    replicates = int(spec["replicates"])
    rows = []
    for rho in spec["panel_correlation"]:
        for pi in spec["panel_pass"]:
            rng = np.random.default_rng(derive_seed(validation_seed, "panel", rho, pi))
            shared = rng.standard_normal((replicates, 1))
            own = rng.standard_normal((replicates, m))
            z = math.sqrt(rho) * shared + math.sqrt(1 - rho) * own
            passes = z <= stats.norm.ppf(pi)
            hits = 0
            for row in passes:
                p = [level if x else Fraction(1) for x in row]
                hits += cs.partial_conjunction(p, r) <= rules.alpha
            exact = float(stats.binom.sf(r - 1, m, pi)) if rho == 0 else None
            rows.append(
                {
                    "correlation": float(rho),
                    "pass_probability": float(pi),
                    "replicates": replicates,
                    "power": hits / replicates,
                    "exact_independent": exact,
                }
            )
    return rows


# ---------------------------------------------------------------------------------------------
# Item 6: dose selection


def selection_study(
    rules: cr.Rules, spec: Mapping[str, Any], *, validation_seed: int
) -> list[dict[str, Any]]:
    """The margin rule on an exploratory pass and the confirmation on fresh draws (§12.4.1).

    Per scenario: how often each checkpoint is d₁, how often no checkpoint qualifies, the
    confirmation rate overall, and per selected d₁ the confirmation rate beside the unconditional
    P(p₁ ≤ level) at that dose's true rate. Fresh draws make them equal up to Monte Carlo error.
    """
    level = rules.per_case_level
    prompts = int(rules["observer"]["prompts"])
    tolerance = rules.fraction("observer", "tolerance")
    margin = rules.fraction("observer", "margin")
    exploratory = float(rules.fraction("observer", "exploratory_level"))
    checkpoints = list(rules["observer"]["checkpoints"])
    attempts = rules.attempts
    observer_max = _observer_passes(prompts, tolerance, level, attempts)
    replicates = int(spec["replicates"])
    rows = []
    for scenario, truth in spec["scenarios"].items():
        latent = np.asarray(truth["latent"], dtype=float)
        sham = np.asarray(truth["sham"], dtype=float)
        rng = np.random.default_rng(derive_seed(validation_seed, "selection", scenario))
        cells = np.stack(
            [latent * sham, latent * (1 - sham), (1 - latent) * sham, (1 - latent) * (1 - sham)],
            axis=1,
        )
        first = dict.fromkeys(checkpoints, 0)
        confirmed_at_first = dict.fromkeys(checkpoints, 0)
        none = confirmed = 0
        analysis = dict.fromkeys(checkpoints, 0)
        for _ in range(replicates):
            tables = np.stack([rng.multinomial(prompts, cells[i]) for i in range(len(checkpoints))])
            upper = A.newcombe_paired_upper(*tables.T, level=exploratory)
            candidates = cr.margin_candidates(
                dict(zip(checkpoints, upper, strict=True)), margin, checkpoints
            )
            if not candidates:
                none += 1
                continue
            ledger = cr.AttemptLedger(tuple(candidates), attempts=attempts)
            d1 = ledger.first_dose()
            index = checkpoints.index(d1)
            passed = int(rng.binomial(prompts, latent[index])) <= observer_max
            ledger.record(d1, passed)
            first[d1] += 1
            confirmed_at_first[d1] += passed
            if not passed and ledger.next_dose() is not None:
                d2 = ledger.next_dose()
                again = int(rng.binomial(prompts, latent[checkpoints.index(d2)])) <= observer_max
                ledger.record(d2, again, cr.NEXT_DOSE)
            dose = ledger.analysis_dose()
            if dose is not None:
                confirmed += 1
                analysis[dose] += 1
        per_dose = {}
        for i, dose in enumerate(checkpoints):
            unconditional = float(stats.binom.cdf(observer_max, prompts, latent[i]))
            selected = first[dose]
            per_dose[dose] = {
                "selected_as_d1": selected / replicates,
                "confirmed_given_selected": confirmed_at_first[dose] / selected
                if selected
                else None,
                "unconditional_confirmation": unconditional,
                "mc_se": math.sqrt(unconditional * (1 - unconditional) / selected)
                if selected
                else None,
                "analysis_dose": analysis[dose] / replicates,
            }
        rows.append(
            {
                "scenario": scenario,
                "latent": latent.tolist(),
                "sham": sham.tolist(),
                "replicates": replicates,
                "no_candidate": none / replicates,
                "confirmed": confirmed / replicates,
                "per_dose": per_dose,
            }
        )
    return rows
