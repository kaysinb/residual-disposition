"""Component tests of the path-B continuation (docs/preregistration.md §12.4–§12.6).

Every p-value that enters a confirmatory decision is exact: binomial tail sums and the sign test
are computed in rational arithmetic (``fractions.Fraction``), and compared with levels that are
fractions too (``continuation_rules``). The shifted risk-difference test of the readers, which
needs a certified supremum over a nuisance parameter, lives in ``continuation_exact``.

Sections: the observer (§12.4.1); the certificate's sign test with its conservative censoring rule
and intervals (§12.4.8); intersection–union and the partial conjunction (§12.5, §12.6).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction

# ---------------------------------------------------------------------------------------------
# Exact binomial tails


def binomial_cdf(k: int, n: int, p: Fraction) -> Fraction:
    """P(X ≤ k) for X ~ Bin(n, p), exactly."""
    if k < 0:
        return Fraction(0)
    if k >= n:
        return Fraction(1)
    q = 1 - p
    return sum((math.comb(n, i) * p**i * q ** (n - i) for i in range(k + 1)), Fraction(0))


def binomial_sf(k: int, n: int, p: Fraction) -> Fraction:
    """P(X ≥ k) for X ~ Bin(n, p), exactly."""
    return 1 - binomial_cdf(k - 1, n, p)


def rate_below_p(hits: int, n: int, bound: Fraction) -> Fraction:
    """Exact one-sided p-value of H₀: rate ≥ bound against rate < bound: P(X ≤ hits | bound)."""
    return binomial_cdf(hits, n, bound)


# ---------------------------------------------------------------------------------------------
# Observer (§12.4.1)


def observer_p_abs(hits: int, prompts: int, tolerance: Fraction) -> Fraction:
    """p_abs for H₀: r_L ≥ tolerance. It is valid for the excess null too: the excess over H_s is
    at most r_L, so "excess ≥ tolerance" implies "r_L ≥ tolerance"."""
    return rate_below_p(hits, prompts, tolerance)


def observer_p1(hits: int, prompts: int, tolerance: Fraction, attempts: int = 2) -> Fraction:
    """p₁ = min(1, a · p_abs), with the frozen allocation a."""
    return min(Fraction(1), attempts * observer_p_abs(hits, prompts, tolerance))


# ---------------------------------------------------------------------------------------------
# Certificate (§12.4.8)


@dataclasses.dataclass(frozen=True)
class Crossing:
    """The crossing time T of one relearning run, as known after the run.

    ``lower`` and ``upper`` bound T; they are equal for an observed (latched, interpolated)
    crossing. A run censored at its budget has ``upper = inf``. An incomplete or failed run is
    ``complete = False`` and carries no information.
    """

    lower: float
    upper: float
    complete: bool = True

    @classmethod
    def observed(cls, t: float) -> Crossing:
        return cls(lower=t, upper=t)

    @classmethod
    def censored(cls, budget: float) -> Crossing:
        return cls(lower=budget, upper=math.inf)

    @classmethod
    def incomplete(cls) -> Crossing:
        return cls(lower=math.nan, upper=math.nan, complete=False)

    @property
    def is_observed(self) -> bool:
        return self.complete and self.lower == self.upper


def _log(x: float) -> float:
    return math.inf if x == math.inf else math.log(x)


@dataclasses.dataclass(frozen=True)
class Bounds:
    """A log-contrast d known to lie in [lower, upper]; equal ends when observed."""

    lower: float
    upper: float

    @property
    def observed(self) -> bool:
        return self.lower == self.upper


def savings_bounds(latent: Crossing, sham: Crossing) -> Bounds:
    """d = log T(H_s) − log T(L), bounded from the two crossings."""
    return Bounds(
        lower=_log(sham.lower) - _log(latent.upper),
        upper=_log(sham.upper) - _log(latent.lower),
    )


def ratio_bounds(behavior: Bounds, control: Bounds) -> Bounds:
    """d(Q) = d(S_b) − d(S_Y), by interval subtraction."""
    return Bounds(lower=behavior.lower - control.upper, upper=behavior.upper - control.lower)


@dataclasses.dataclass(frozen=True)
class SignTest:
    """The exact sign test of one log-contrast over the seed blocks (§12.4.8)."""

    blocks: int
    positive: int
    p: Fraction
    complete: bool
    interval: tuple[float, float] | None  # None: unavailable
    median: float | None  # the sample median, when every contrast is observed


def sign_test(contrasts: Sequence[Bounds | None]) -> SignTest:
    """K = number of blocks whose contrast is certainly positive (lower bound > 0).

    A block whose sign is undeterminable counts as not positive; a missing block (None, from an
    incomplete or failed run) makes the contrast incomplete with p = 1. The p-value is
    P(Bin(n, ½) ≥ K), valid for independent blocks when H₀ gives P(d > 0) ≤ ½ (median ≤ 0); no
    symmetry is assumed.
    """
    n = len(contrasts)
    if n == 0 or any(c is None for c in contrasts):
        return SignTest(n, 0, Fraction(1), False, None, None)
    bounds = [c for c in contrasts if c is not None]
    k = sum(1 for c in bounds if c.lower > 0)
    p = binomial_sf(k, n, Fraction(1, 2))
    if all(c.observed for c in bounds):
        values = sorted(c.lower for c in bounds)
        interval: tuple[float, float] | None = (values[0], values[-1])
        mid = len(values) // 2
        median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    else:
        lows = [c.lower for c in bounds]
        highs = [c.upper for c in bounds]
        undeterminable = any(c.lower == -math.inf and c.upper == math.inf for c in bounds)
        interval = None if undeterminable else (min(lows), max(highs))
        median = None
    return SignTest(n, k, p, True, interval, median)


def order_statistic_coverage(n: int) -> Fraction:
    """Coverage of [d₍₁₎, d₍ₙ₎] for the median of n continuous observations: 1 − 2 / 2ⁿ."""
    return 1 - Fraction(2, 2**n)


@dataclasses.dataclass(frozen=True)
class Block:
    """One seed block at one learning rate: the six arms' crossings, keyed (habit, role).

    habit ∈ {b, Y, Y2}; role ∈ {L, H_s}.
    """

    arms: Mapping[tuple[str, str], Crossing]

    def savings(self, habit: str) -> Bounds | None:
        latent = self.arms.get((habit, "L"))
        sham = self.arms.get((habit, "H_s"))
        if latent is None or sham is None or not (latent.complete and sham.complete):
            return None
        return savings_bounds(latent, sham)


def certificate_contrasts(blocks: Sequence[Block], controls: Sequence[str] = ("Y", "Y2")) -> dict:
    """Per-block bounds of S_b and of Q for each control habit, at one learning rate."""
    out: dict[str, list[Bounds | None]] = {"S_b": [], **{f"Q_{c}": [] for c in controls}}
    for block in blocks:
        behavior = block.savings("b")
        out["S_b"].append(behavior)
        for control in controls:
            other = block.savings(control)
            out[f"Q_{control}"].append(
                None if behavior is None or other is None else ratio_bounds(behavior, other)
            )
    return out


def certificate_p(by_rate: Mapping[float, Sequence[Block]]) -> tuple[Fraction, dict]:
    """p₂ = the largest sign-test p over every contrast at every learning rate, with the tests."""
    tests: dict[tuple[float, str], SignTest] = {}
    for rate, blocks in by_rate.items():
        for name, contrasts in certificate_contrasts(blocks).items():
            tests[(rate, name)] = sign_test(contrasts)
    p2 = max((t.p for t in tests.values()), default=Fraction(1))
    return p2, tests


def geometric_mean(values: Sequence[float]) -> float:
    """Descriptive only: exp of the mean log (§12.4.8)."""
    return math.exp(sum(math.log(v) for v in values) / len(values))


# ---------------------------------------------------------------------------------------------
# Combination (§12.5, §12.6)


def intersection_union(components: Mapping[str, Fraction | float | None]) -> Fraction:
    """p_i = the largest component p; a missing component (None) or a failed prerequisite,
    passed as 1, makes p_i = 1."""
    if not components:
        return Fraction(1)
    worst = Fraction(0)
    for value in components.values():
        if value is None:
            return Fraction(1)
        worst = max(worst, Fraction(value))
    return min(Fraction(1), worst)


def partial_conjunction(p_values: Sequence[Fraction | float], r: int) -> Fraction:
    """p_PC = min{1, (m − r + 1) · p_(r)}, valid under arbitrary dependence given valid p_i.

    ``p_values`` has one entry per frozen panel lineage; an unrun or inconclusive lineage enters
    as 1, so m never shrinks.
    """
    m = len(p_values)
    if not 1 <= r <= m:
        raise ValueError(f"r = {r} outside 1..{m}")
    ordered = sorted(Fraction(p) for p in p_values)
    return min(Fraction(1), (m - r + 1) * ordered[r - 1])
