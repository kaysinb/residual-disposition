"""Pure rules of the relearning certificate (path B), without model code, so that analysis and
orchestration share them: the interpolated crossing step, the decisive failure of a seed pair
(the futility stop) and the calibration of the two relearning learning rates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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
    (relearning is not feasible at these settings; report to the coordinator).
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
    results: Mapping[float, int | None],
    *,
    window: tuple[int, int] = (60, 200),
    calibration_budget: int = 200,
    extension_budget: int = 600,
    max_budget: int = 600,
) -> RateChoice:
    """Apply the branch table to the calibration results so far: learning rate → crossing step
    of sham_s, or None when the run did not cross within its calibration budget.

    Branches, the first matching: a, a candidate crosses within ``window`` (the largest such);
    b, adjacent candidates straddle the window (one intermediate candidate, then the nearest to
    the window in log distance among observed crossings); c, all cross too fast (halve down the
    ladder; at its end the smallest tried, flagged); d, none crosses (one run at 1e-4 with
    ``extension_budget`` steps; if it does not cross, stop; if it crosses too slowly, it).
    Anything else, which needs a non-monotone calibration, takes the nearest observed crossing,
    flagged.
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
        lr_lo, lo_flags = _lower_rate(lr_hi, tried)
        t_lo = results.get(lr_lo)
        budget = max_budget if t_lo is None else min(max_budget, 50 * math.ceil(3 * t_lo / 50))
        return RateChoice(
            "chosen", branch, lr_hi=lr_hi, lr_lo=lr_lo, budget=budget, flags=flags + lo_flags
        )

    def nearest(branch: str, flag: str) -> RateChoice:
        def distance(t: int) -> float:
            return math.log(low / t) if t < low else math.log(t / high) if t > high else 0.0

        return chosen(min(observed, key=lambda lr: (distance(observed[lr]), -lr)), branch, flag)

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
        return nearest("b", "nearest to the window in log distance")
    if all(fast(lr) for lr in tried):
        below = [lr for lr in LR_LADDER if lr < min(tried)]
        if below:
            return RateChoice(
                "calibrate", "c", calibrate=(max(below),), calibrate_budget=calibration_budget
            )
        return chosen(min(tried), "c", "reduced resolution: every candidate crosses too fast")
    if not observed:
        if EXTENSION_LR not in results:
            return RateChoice(
                "calibrate", "d", calibrate=(EXTENSION_LR,), calibrate_budget=extension_budget
            )
        return RateChoice(
            "stop",
            "d",
            reason=f"no candidate crosses, not {EXTENSION_LR:g} within {extension_budget} steps",
        )
    if EXTENSION_LR in results and all(slow(lr) for lr in tried):
        fastest = min(observed, key=lambda lr: observed[lr])
        return chosen(fastest, "d", "every candidate crosses too slowly")
    return nearest("fallback", "non-monotone calibration: nearest observed crossing")


def _lower_rate(lr_hi: float, tried: Sequence[float]) -> tuple[float, tuple[str, ...]]:
    """The lower of the two rates: the next smaller ladder rate, or, for an intermediate rate,
    the next smaller rate tried; below the ladder, half of ``lr_hi`` (flagged)."""
    pool = LR_LADDER if lr_hi in LR_LADDER else tuple(lr for lr in tried if lr < lr_hi)
    smaller = [lr for lr in pool if lr < lr_hi] or [lr for lr in LR_LADDER if lr < lr_hi]
    if smaller:
        return max(smaller), ()
    return lr_hi / 2, ("the lower rate is below the ladder: half of the higher",)


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
    (path B's futility stop): a censored latent run, or a completed pair with T_sham ≤ T_lat. Each
    run is its summary: ``censored`` and ``t`` (the interpolated crossing step, else the recorded
    one). None while no decisive failure is shown."""
    if latent.get("censored"):
        return "latent run censored"
    if sham is None:
        return None
    t_lat = float(latent.get("t") or latent["crossing_step"])
    if sham.get("censored"):
        if sham.get("budget") is None:
            return None  # a censored sham enters at its budget, unknown here: not decisive
        t_sham = float(sham["budget"])
    else:
        t_sham = float(sham.get("t") or sham["crossing_step"])
    if t_sham <= t_lat:
        return f"T_sham {t_sham:g} <= T_lat {t_lat:g}"
    return None
