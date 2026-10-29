"""Component tests of the path-B continuation (docs/preregistration.md §12.4–§12.6).

Every p-value that enters a confirmatory decision is exact: binomial tail sums and the sign test
are computed in rational arithmetic (``fractions.Fraction``), and compared with levels that are
fractions too (``continuation_rules``). The shifted risk-difference test of the readers, which
needs a certified supremum over a nuisance parameter, lives in ``continuation_exact``.

Sections: the observer (§12.4.1); the certificate's sign test with its conservative censoring rule
and intervals (§12.4.8); intersection–union and the partial conjunction (§12.5, §12.6).

The certificate's signs are decided exactly: a block contrast is held as bounds on the ratio of
crossing times (rationals, from the floats the runs record), and "certainly positive" compares
cross-products of those times, so equal savings never become a positive sign by rounding. Logs
are taken only for the reported intervals and medians.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

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


def _exact(t: float) -> Fraction | None:
    """A recorded time as an exact rational; None for +inf."""
    return None if t == math.inf else Fraction(t)


@dataclasses.dataclass(frozen=True)
class Bounds:
    """A contrast d = log R known through exact bounds on the ratio R.

    R ∈ [ratio_lower, ratio_upper]; ``ratio_lower = 0`` stands for d = −inf and
    ``ratio_upper = None`` for d = +inf. Signs are decided on the ratios; ``lower`` and ``upper``
    (the log ends) serve intervals and reports only.
    """

    ratio_lower: Fraction
    ratio_upper: Fraction | None

    @classmethod
    def from_logs(cls, lower: float, upper: float) -> Bounds:
        """Hand-built bounds from log ends (tests and reports; not exact)."""
        return cls(
            ratio_lower=Fraction(0) if lower == -math.inf else Fraction(math.exp(lower)),
            ratio_upper=None if upper == math.inf else Fraction(math.exp(upper)),
        )

    @property
    def lower(self) -> float:
        return -math.inf if self.ratio_lower == 0 else math.log(self.ratio_lower)

    @property
    def upper(self) -> float:
        return math.inf if self.ratio_upper is None else math.log(self.ratio_upper)

    @property
    def observed(self) -> bool:
        return self.ratio_lower == self.ratio_upper

    @property
    def certainly_positive(self) -> bool:
        """d > 0 for every value the bounds allow: ratio_lower > 1, exactly."""
        return self.ratio_lower > 1


def savings_bounds(latent: Crossing, sham: Crossing) -> Bounds:
    """R = T(H_s) / T(L), bounded from the two crossings; d = log R."""
    latent_upper, sham_upper = _exact(latent.upper), _exact(sham.upper)
    return Bounds(
        ratio_lower=Fraction(0) if latent_upper is None else Fraction(sham.lower) / latent_upper,
        ratio_upper=None if sham_upper is None else sham_upper / Fraction(latent.lower),
    )


def ratio_bounds(behavior: Bounds, control: Bounds) -> Bounds:
    """Q = R(S_b) / R(S_Y), so d(Q) = d(S_b) − d(S_Y), by exact interval division: Q > 1 for
    certain iff T_b(H_s)_low · T_Y(L)_low > T_Y(H_s)_up · T_b(L)_up."""
    lower = (
        Fraction(0) if control.ratio_upper is None else behavior.ratio_lower / control.ratio_upper
    )
    upper = (
        None
        if behavior.ratio_upper is None or control.ratio_lower == 0
        else behavior.ratio_upper / control.ratio_lower
    )
    return Bounds(ratio_lower=lower, ratio_upper=upper)


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
    k = sum(1 for c in bounds if c.certainly_positive)
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


HABITS = ("b", "Y", "Y2")
ROLES = ("L", "H_s")


@dataclasses.dataclass(frozen=True)
class Block:
    """One seed block at one learning rate: the six arms' crossings, keyed (habit, role).

    habit ∈ {b, Y, Y2}; role ∈ {L, H_s}. ``block_id`` is the block's frozen relearning seed.
    """

    arms: Mapping[tuple[str, str], Crossing]
    block_id: int | None = None

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


@dataclasses.dataclass(frozen=True)
class CertificateResult:
    """p₂ with its sign tests, keyed (learning rate, contrast), and the protocol problems that
    force p₂ = 1."""

    p: Fraction
    tests: dict[tuple[float, str], SignTest]
    problems: list[str]


def certificate_problems(
    by_rate: Mapping[float, Sequence[Block]],
    learning_rates: Sequence[float],
    block_ids: Sequence[int],
) -> list[str]:
    """What the frozen design lacks: both learning rates and no other, at each rate exactly the
    frozen block ids once each, and in every block all six arms."""
    problems = []
    frozen = [float(lr) for lr in learning_rates]
    given = [float(lr) for lr in by_rate]

    def same(a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=1e-9)

    for lr in frozen:
        if not any(same(lr, g) for g in given):
            problems.append(f"learning rate {lr:g} missing")
    for g in given:
        if not any(same(g, lr) for lr in frozen):
            problems.append(f"learning rate {g:g} outside the design")
    wanted = sorted(int(b) for b in block_ids)
    for rate, blocks in by_rate.items():
        ids = sorted(-1 if b.block_id is None else int(b.block_id) for b in blocks)
        if ids != wanted:
            problems.append(f"rate {float(rate):g}: blocks {ids}, not {wanted}")
        for block in blocks:
            missing = [f"{h}/{r}" for h in HABITS for r in ROLES if (h, r) not in block.arms]
            extra = [f"{h}/{r}" for h, r in block.arms if h not in HABITS or r not in ROLES]
            if missing or extra:
                problems.append(
                    f"rate {float(rate):g}, block {block.block_id}: "
                    f"missing {missing}, unexpected {extra}"
                )
    return problems


def certificate_p(
    by_rate: Mapping[float, Sequence[Block]],
    *,
    learning_rates: Sequence[float],
    block_ids: Sequence[int],
) -> CertificateResult:
    """p₂ = the largest sign-test p over every contrast at every learning rate.

    The design is checked first (``certificate_problems``): a missing rate, block or arm makes
    p₂ = 1. The tests are still computed for the report where the blocks allow.
    """
    problems = certificate_problems(by_rate, learning_rates, block_ids)
    tests: dict[tuple[float, str], SignTest] = {}
    for rate, blocks in by_rate.items():
        for name, contrasts in certificate_contrasts(blocks).items():
            tests[(rate, name)] = sign_test(contrasts)
    p2 = max((t.p for t in tests.values()), default=Fraction(1))
    return CertificateResult(Fraction(1) if problems else p2, tests, problems)


def certificate_design(rules: Any) -> dict[str, Any]:
    """The frozen design arguments of ``certificate_p`` from the continuation rules."""
    cert = rules["certificate"]
    return {
        "learning_rates": [float(x) for x in cert["learning_rates"]],
        "block_ids": [int(x) for x in cert["block_seeds"]],
    }


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


def holm_from_decisions(
    decisions: Mapping[str, Mapping[Fraction, bool]], alpha: Fraction
) -> dict[str, Any]:
    """Holm's step-down procedure at family level α (§13.5 (d)), from each hypothesis's decisions
    at the m levels α/m, α/(m − 1), …, α alone; no p-value is needed.

    H_(j) is rejected while p_(i) ≤ α/(m − i + 1) for every i ≤ j. "p_(i) ≤ c" is the event that
    at least i hypotheses reject at c, so the number rejected is the largest j with
    N(α/(m − i + 1)) ≥ i for all i ≤ j, and the rejected ones are those rejecting at α/(m − j + 1).
    ``decisions`` holds all m hypotheses, an unrun or ineligible one rejecting nowhere; each must
    be monotone in the level (a rejection at a level holds at every higher one).
    """
    m = len(decisions)
    levels = [alpha / (m - i) for i in range(m)]  # α/m, …, α/1
    for name, d in decisions.items():
        if set(d) != set(levels):
            raise ValueError(f"{name}: decisions at {sorted(d)} instead of {levels}")
        row = [bool(d[c]) for c in levels]
        if any(a and not b for a, b in zip(row, row[1:], strict=False)):
            raise ValueError(f"{name}: a rejection at a lower level is lost at a higher one")
    steps, rejected = [], 0
    for i, c in enumerate(levels, start=1):
        count = sum(bool(d[c]) for d in decisions.values())
        steps.append({"step": i, "level": str(c), "rejecting": count})
        if count < i:
            break
        rejected = i
    names = sorted(n for n, d in decisions.items() if rejected and d[levels[rejected - 1]])
    if len(names) != rejected:  # N(c_j*) > j* would make step j* + 1 pass as well
        raise AssertionError(f"Holm: {len(names)} reject at the last passed level, not {rejected}")
    return {"rejected": names, "levels": [str(c) for c in levels], "steps": steps}
