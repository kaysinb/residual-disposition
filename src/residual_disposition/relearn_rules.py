"""Pure rules of the relearning certificate (path B), without model code, so that analysis,
orchestration and simulation share them: the interpolated crossing step, the certificate's T of a
run, the decisive failure of a seed pair (the futility stop), the savings ratio and the
calibration of the two relearning learning rates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------- T of a run

# A crossing at or before the first evaluation counts as one step, so that T is never 0 and ratios
# of T stay finite; the same floor applies wherever T is used.
MIN_T = 1.0
# S equal to the threshold passes despite rounding in exp(mean log a - mean log b).
_LOG_TOLERANCE = 1e-9


def _given(value: object) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def certificate_time(run: Mapping[str, object]) -> float | None:
    """T of one relearning run as path B counts it, the one definition shared by the futility stop,
    the savings ratio, the learning-rate calibration and the simulation.

    ``run`` is the run's summary: ``censored``, ``budget``, ``t`` (the interpolated crossing step)
    and ``crossing_step`` (the recorded one). A censored run counts at its budget (None where the
    budget is not known); otherwise T is ``t``, or ``crossing_step`` where nothing interpolates;
    either way at least ``MIN_T``.
    """
    if run.get("censored"):
        budget = run.get("budget")
        return max(MIN_T, float(budget)) if _given(budget) else None
    t = run.get("t")
    if not _given(t):
        t = run.get("crossing_step")
    if not _given(t):
        raise ValueError(f"an uncensored run without a crossing step: {dict(run)}")
    return max(MIN_T, float(t))


def savings_ratio(pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]]) -> float:
    """S at one learning rate: the ratio of the geometric means of ``certificate_time``, sham_s
    over latent, across the (latent, sham_s) summaries of the completed seed pairs."""
    if not pairs:
        raise ValueError("no seed pairs")
    times = [(certificate_time(sham), certificate_time(latent)) for latent, sham in pairs]
    if any(t is None for pair in times for t in pair):
        raise ValueError("a censored run without a budget")
    log_sham = sum(math.log(t_sham) for t_sham, _ in times) / len(times)
    log_lat = sum(math.log(t_lat) for _, t_lat in times) / len(times)
    return math.exp(log_sham - log_lat)


def meets_savings(savings: float, min_savings: float) -> bool:
    """S ≥ ``min_savings``, with S equal to the threshold passing despite rounding."""
    return math.log(savings) >= math.log(min_savings) - _LOG_TOLERANCE


# ---------------------------------------------------------------- learning rates

# Path B's calibration of the two relearning learning rates, per behavior, on sham_s at the first
# audit dose (feasibility plan, "Relearning settings"). Candidates are halved down or extended up
# along this ladder; one intermediate candidate may be tried between a straddling pair.
LR_LADDER = (1.25e-6, 2.5e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4)
BASE_CANDIDATES = (5e-6, 1e-5, 2e-5, 5e-5)
EXTENSION_LR = 1e-4


@dataclass(frozen=True)
class RateChoice:
    """The next step of the learning-rate calibration of one behavior.

    ``status`` is "chosen" (``lr_hi``, ``lr_lo`` and the certificate's step ``budget``),
    "calibrate" (run ``calibrate`` with ``calibrate_budget`` steps, then decide again) or "stop"
    (relearning is not feasible at these settings, or the rates are unresolved; report).
    """

    status: str
    branch: str
    lr_hi: float | None = None
    lr_lo: float | None = None
    budget: int | None = None
    calibrate: tuple[float, ...] = ()
    calibrate_budget: int | None = None
    flags: tuple[str, ...] = ()
    reason: str = ""


def choose_relearn_rates(
    results: Mapping[float, float | None],
    *,
    window: tuple[int, int] = (60, 200),
    calibration_budget: int = 200,
    extension_budget: int = 600,
    max_budget: int = 600,
) -> RateChoice:
    """Apply the branch table to the calibration results so far: learning rate → T of sham_s
    (``certificate_time``: interpolated, at least one step), or None when the run did not cross
    within its calibration budget.

    T must not rise with the learning rate (a censored run counting as slower than any
    crossing); otherwise the calibration is "unresolved", a stop. Then the first matching
    branch: d, no base candidate crosses (one run at 1e-4 with ``extension_budget`` steps; if it
    does not cross, stop; if it does, 1e-4, flagged, whatever its T); a, a candidate crosses
    within ``window`` (the largest such); b, adjacent candidates straddle the window (one
    intermediate candidate, then the nearest to the window in log distance among observed
    crossings, flagged); c, all cross too fast (halve down the ladder). A chosen LR_hi with no
    smaller rate on the ladder (1.25e-6) leaves LR_lo undefined: "unresolved", a stop. There is
    no other branch.
    """
    low, high = window
    missing = [lr for lr in BASE_CANDIDATES if lr not in results]
    if missing:
        return RateChoice(
            "calibrate", "start", calibrate=tuple(missing), calibrate_budget=calibration_budget
        )
    tried = sorted(results)
    observed = {lr: t for lr, t in results.items() if t is not None}

    def fast(lr: float) -> bool:
        return results[lr] is not None and results[lr] < low

    def slow(lr: float) -> bool:
        return results[lr] is None or results[lr] > high

    def chosen(lr_hi: float, branch: str, *flags: str) -> RateChoice:
        lr_lo = _lower_rate(lr_hi, tried)
        if lr_lo is None:
            return RateChoice(
                "stop",
                branch,
                lr_hi=lr_hi,
                flags=flags,
                reason=f"unresolved: no rate below LR_hi {lr_hi:g} on the ladder for LR_lo",
            )
        t_lo = results.get(lr_lo)
        budget = max_budget if t_lo is None else min(max_budget, _round_up_50(3 * t_lo))
        return RateChoice("chosen", branch, lr_hi=lr_hi, lr_lo=lr_lo, budget=budget, flags=flags)

    def distance(t: float) -> float:
        return math.log(low / t) if t < low else math.log(t / high) if t > high else 0.0

    times = [math.inf if results[lr] is None else results[lr] for lr in tried]
    rises = [
        (a, b) for a, b, ta, tb in zip(tried, tried[1:], times, times[1:], strict=False) if tb > ta
    ]
    if rises:
        a, b = rises[0]
        return RateChoice(
            "stop",
            "monotonicity",
            reason=f"unresolved: T rises with the learning rate from {a:g} to {b:g}",
        )
    if all(results[lr] is None for lr in BASE_CANDIDATES):
        if EXTENSION_LR not in results:
            return RateChoice(
                "calibrate", "d", calibrate=(EXTENSION_LR,), calibrate_budget=extension_budget
            )
        if results[EXTENSION_LR] is None:
            return RateChoice(
                "stop",
                "d",
                reason=f"nothing crosses, not {EXTENSION_LR:g} within {extension_budget} steps",
            )
        return chosen(EXTENSION_LR, "d", f"only {EXTENSION_LR:g} crosses")
    qualifying = [lr for lr, t in observed.items() if low <= t <= high]
    if qualifying:
        return chosen(max(qualifying), "a")
    straddles = [(a, b) for a, b in zip(tried, tried[1:], strict=False) if slow(a) and fast(b)]
    if straddles:
        if not any(lr not in LR_LADDER for lr in tried):
            a, b = straddles[0]
            return RateChoice(
                "calibrate",
                "b",
                calibrate=(float(f"{math.sqrt(a * b):.3g}"),),
                calibrate_budget=calibration_budget,
            )
        best = min(observed, key=lambda lr: (distance(observed[lr]), -lr))
        return chosen(best, "b", "nearest to the window in log distance")
    if all(fast(lr) for lr in tried):
        below = [lr for lr in LR_LADDER if lr < min(tried)]
        if below:
            return RateChoice(
                "calibrate", "c", calibrate=(max(below),), calibrate_budget=calibration_budget
            )
        return chosen(min(tried), "c", "reduced resolution: every candidate crosses too fast")
    return RateChoice("stop", "none", reason=f"unresolved: no branch applies to {dict(results)}")


def _lower_rate(lr_hi: float, tried: Sequence[float]) -> float | None:
    """The lower of the two rates: the next smaller ladder rate, or, for an intermediate rate,
    the next smaller rate tried; None below the ladder."""
    pool = LR_LADDER if lr_hi in LR_LADDER else tuple(lr for lr in tried if lr < lr_hi)
    smaller = [lr for lr in pool if lr < lr_hi] or [lr for lr in LR_LADDER if lr < lr_hi]
    return max(smaller) if smaller else None


def _round_up_50(steps: float) -> int:
    """``steps`` rounded up to a multiple of 50; a multiple up to rounding stays."""
    return 50 * math.ceil(steps / 50 - _LOG_TOLERANCE)


def interpolated_crossing(
    steps: Sequence[float], rates: Sequence[float], crossing_step: float, target: float
) -> float | None:
    """T, interpolated linearly in the measured rate between the last evaluation below ``target``
    and the first at or above it (``crossing_step``), as preregistered; None where there is nothing
    to interpolate: a crossing at step 0, or evaluations that do not bracket the target there."""
    at = [i for i, s in enumerate(steps) if float(s) == float(crossing_step)]
    if not at or at[0] == 0:
        return None
    j = at[0]
    if not rates[j - 1] < target <= rates[j]:
        return None
    fraction = (target - rates[j - 1]) / (rates[j] - rates[j - 1])
    return float(steps[j - 1] + fraction * (steps[j] - steps[j - 1]))


def decisive_failure(latent: Mapping[str, object], sham: Mapping[str, object] | None) -> str | None:
    """Why a certificate cannot pass, from completed relearning runs of one seed pair at one rate
    (path B's futility stop): a censored latent run, or a completed pair with T_sham ≤ T_lat,
    both T by ``certificate_time``, so that two crossings below one step tie. Each run is its
    summary. None while no decisive failure is shown."""
    if latent.get("censored"):
        return "latent run censored"
    if sham is None:
        return None
    t_lat, t_sham = certificate_time(latent), certificate_time(sham)
    if t_sham is None:
        return None  # a censored sham enters at its budget, unknown here: not decisive
    if t_sham <= t_lat:
        return f"T_sham {t_sham:g} <= T_lat {t_lat:g}"
    return None
