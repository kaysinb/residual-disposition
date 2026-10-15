"""The certified exact unconditional test of a shifted risk difference (preregistration §12.5.1).

Data: x_L ~ Bin(n_L, p_L) and x_C ~ Bin(n_C, p_C), independent. The statistic is the
Farrington–Manning score T(x_L, x_C; δ₀) with its restricted maximum-likelihood variance.

- ">" form, H₀: p_L − p_C ≤ δ₀ against >, statistic S = T(·; δ₀).
- "<" form, H₀: p_L − p_C ≥ δ₀ against <, statistic S = −T(·; δ₀).

In both forms the tail is P(S ≥ S_obs), ties included (outcomes with S ≥ S_obs − TIE_TOL; the
tolerance only adds outcomes, so it is conservative).

Berger–Boos: C_γ is the two-sided Clopper–Pearson 1 − γ interval for p_C from x_C alone, widened
outward by a margin. The restricted null is Ω₀(C_γ) = {(p_L, p_C) ∈ Ω₀ : p_C ∈ C_γ}, and
p = min(1, sup_{Ω₀(C_γ)} P(S ≥ S_obs) + γ); p = γ when Ω₀(C_γ) is empty.

Reduction to the capped boundary: S is monotone in x_L for each x_C (checked for every outcome when
a test is built), so each tail set's x_C-sections are upper sets in x_L (">") or lower sets ("<"),
and the tail probability is monotone in p_L. Its supremum over the null therefore lies on
p_L = min(1, p_C + δ₀) (">") or p_L = max(0, p_C + δ₀) ("<").

Certified supremum: along the boundary the tail probability is a polynomial g(θ), θ = p_C, with
|g'| ≤ n_L + n_C (both coordinates move) or ≤ n_C (on a capped segment, where p_L is constant): for
one binomial arm ∂P(A)/∂p = n [P_{n−1}(A − 1) − P_{n−1}(A)] ∈ [−n, n]. A branch-and-bound over
dyadic intervals bounds an interval [a, b] by the smaller of the Lipschitz bound
max(g(a), g(b)) + L·(b − a)/2 and a second-order bound from an enclosure of g' at an end and
|g''| ≤ M (M = 2n_L(n_L − 1) + 4n_L n_C + 2n_C(n_C − 1) where p_L moves, 2n_C(n_C − 1) on a capped
segment). Every value that enters a bound is an interval enclosure computed with outward rounding
(mpmath.iv) at exact rational points; float64 never enters a bound. Comparisons with levels are
made against exact fractions.

Final decisions: C_γ depends on x_C only. For fixed x_C the tail sets used at x_L and x_L + 1 are
nested (checked on the thresholds of every outcome, ``tail_sets_nested``), so g at one outcome is
pointwise bounded by g at its neighbour on the same window, and a certified decision at one outcome
transfers to every outcome on its side. The decision region R_α = {p ≤ α} is an upper set ("<":
lower set) in x_L for each x_C, found by a certified bisection over x_L; every one of the
(n_L + 1)(n_C + 1) decisions follows. Barnard convexity of R_α across x_C is checked on the
thresholds. The size sup_{Ω₀} P(R_α) is certified on the capped boundary: because every
x_C-section of R_α is an upper set (lower set) in x_L, P(R_α) is monotone in p_L for each p_C. The
two-dimensional certificate is kept as a cross-check for small n.
"""

from __future__ import annotations

import dataclasses
import heapq
import math
from collections.abc import Callable, Sequence
from fractions import Fraction
from functools import cached_property

import numpy as np
from mpmath import iv, mp, mpf
from mpmath.libmp import to_rational
from scipy import stats

TIE_TOL = 1e-9
# Outward widening of the float Clopper–Pearson ends, far above their rounding error.
CP_MARGIN = 1e-9
# Dyadic resolution of interval ends: every branch-and-bound node is k / 2^DYADIC_BITS.
DYADIC_BITS = 48


# ---------------------------------------------------------------------------------------------
# The statistic


def fm_statistic(x_l: np.ndarray, x_c: np.ndarray, n_l: int, n_c: int, delta: float) -> np.ndarray:
    """Farrington–Manning score for p_L − p_C = δ, vectorized (float64; it only orders outcomes).

    The restricted MLE solves the cubic of Farrington and Manning (1990). A zero restricted
    variance gives ±inf by the sign of the numerator (0 when the numerator is 0).
    """
    p_l = np.asarray(x_l, dtype=float) / n_l
    p_c = np.asarray(x_c, dtype=float) / n_c
    theta = n_c / n_l
    a = 1 + theta
    b = -(1 + theta + p_l + theta * p_c + delta * (theta + 2))
    c = delta**2 + delta * (2 * p_l + theta + 1) + p_l + theta * p_c
    d = -p_l * delta * (1 + delta)
    v = b**3 / (27 * a**3) - b * c / (6 * a**2) + d / (2 * a)
    u = np.sign(v) * np.sqrt(np.maximum(b**2 / (9 * a**2) - c / (3 * a), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(u == 0, 0.0, v / np.where(u == 0, 1.0, u**3))
        w = (np.pi + np.arccos(np.clip(ratio, -1.0, 1.0))) / 3
        t_l = np.clip(2 * u * np.cos(w) - b / (3 * a), 0.0, 1.0)
        t_c = np.clip(t_l - delta, 0.0, 1.0)
        var = t_l * (1 - t_l) / n_l + t_c * (1 - t_c) / n_c
        num = p_l - p_c - delta
        return np.where(var > 1e-15, num / np.sqrt(np.maximum(var, 1e-300)), np.sign(num) * np.inf)


# ---------------------------------------------------------------------------------------------
# Exact and interval helpers


def _iv(value: Fraction | float | int):
    """An outward-rounded interval enclosing an exact rational."""
    f = Fraction(value)
    return iv.mpf(f.numerator) / iv.mpf(f.denominator)


def _ends(x) -> tuple[Fraction, Fraction]:
    """The exact (dyadic) lower and upper ends of an mpmath interval."""
    a, b = x._mpi_
    return Fraction(*to_rational(a)), Fraction(*to_rational(b))


def _exact_mpf(value: Fraction) -> mpf:
    """A dyadic fraction as an mpf; exact at the working precision for the ends used here."""
    return mpf(value.numerator) / mpf(value.denominator)


def _clip_unit(x):
    """An interval clipped to [0, 1] (a probability's enclosure)."""
    lo, hi = _ends(x)
    if lo >= 0 and hi <= 1:
        return x
    zero, one = Fraction(0), Fraction(1)
    lo, hi = min(one, max(zero, lo)), max(zero, min(one, hi))
    return iv.mpf([_exact_mpf(lo), _exact_mpf(hi)])


def _dyadic_floor(x: float) -> Fraction:
    return Fraction(math.floor(x * 2**DYADIC_BITS), 2**DYADIC_BITS)


def _dyadic_ceil(x: float) -> Fraction:
    return Fraction(math.ceil(x * 2**DYADIC_BITS), 2**DYADIC_BITS)


def clopper_pearson(x: int, n: int, gamma: float) -> tuple[float, float]:
    """Two-sided 1 − γ Clopper–Pearson interval, widened outward by CP_MARGIN, in [0, 1]."""
    lo = 0.0 if x == 0 else float(stats.beta.ppf(gamma / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(1 - gamma / 2, x + 1, n - x))
    return max(0.0, lo - CP_MARGIN), min(1.0, hi + CP_MARGIN)


def _pmf_iv(n: int, p) -> list:
    """Interval enclosures of Bin(n, p) probabilities for an interval p."""
    q = 1 - p
    powers_p = [iv.mpf(1)]
    powers_q = [iv.mpf(1)]
    for _ in range(n):
        powers_p.append(powers_p[-1] * p)
        powers_q.append(powers_q[-1] * q)
    return [math.comb(n, k) * powers_p[k] * powers_q[n - k] for k in range(n + 1)]


# ---------------------------------------------------------------------------------------------
# The test


@dataclasses.dataclass(frozen=True)
class ShiftedDifferenceTest:
    """One shifted risk-difference test of the readers (§12.5.1)."""

    n_l: int
    n_c: int
    delta: Fraction
    form: str  # ">" or "<"
    gamma: Fraction = Fraction(1, 10000)
    digits: int = 60
    gap: Fraction = Fraction(1, 1_000_000)

    def __post_init__(self) -> None:
        if self.form not in (">", "<"):
            raise ValueError("form must be '>' or '<'")
        if self.form == "<" and self.delta <= 0:
            raise ValueError("the '<' form is the upper TOST side, with δ₀ > 0")

    # --- statistic and monotonicity ---------------------------------------------------------

    @cached_property
    def statistic(self) -> np.ndarray:
        """S[x_L, x_C] for every outcome."""
        x_l, x_c = np.meshgrid(np.arange(self.n_l + 1), np.arange(self.n_c + 1), indexing="ij")
        t = fm_statistic(x_l, x_c, self.n_l, self.n_c, float(self.delta))
        return t if self.form == ">" else -t

    @cached_property
    def monotone(self) -> bool:
        return self.monotone_in_x_l()

    def monotone_in_x_l(self) -> bool:
        """S nondecreasing (">") or nonincreasing ("<") in x_L for each x_C, up to TIE_TOL."""
        steps = np.diff(self.statistic, axis=0)
        if self.form == ">":
            return bool(np.all(np.nan_to_num(steps, nan=0.0, posinf=1, neginf=-1) >= -TIE_TOL))
        return bool(np.all(np.nan_to_num(steps, nan=0.0, posinf=1, neginf=-1) <= TIE_TOL))

    def _require_monotone(self) -> None:
        if not self.monotone:
            raise NotImplementedError(
                "S is not monotone in x_L: the capped-boundary reduction does not apply; "
                "use the two-dimensional supremum (size_two_dimensional)."
            )

    @cached_property
    def tail_sets_nested(self) -> bool:
        """For each x_C, the tail set used at x_L + 1 is inside (">") or contains ("<") the one
        used at x_L, checked on the thresholds of every outcome.

        Both forms reduce to: thresholds nondecreasing in x_L, elementwise. Then g at x_L + 1 is
        pointwise ≤ (">") or ≥ ("<") g at x_L on the same window C_γ, so a certified bound at one
        outcome transfers to its neighbours, and the decision region can be found by bisection.
        """
        for x_c in range(self.n_c + 1):
            previous = None
            for x_l in range(self.n_l + 1):
                current = self.tail_thresholds(float(self.statistic[x_l, x_c]))
                if previous is not None and np.any(current < previous):
                    return False
                previous = current
        return True

    # --- boundary ---------------------------------------------------------------------------

    @property
    def theta_range(self) -> tuple[Fraction, Fraction]:
        """θ = p_C on the null boundary."""
        if self.form == ">":
            return max(Fraction(0), -self.delta), Fraction(1)
        return Fraction(0), 1 - self.delta

    def _p_l_float(self, theta: np.ndarray) -> np.ndarray:
        delta = float(self.delta)
        return np.clip(theta + delta, 0.0, 1.0)

    def _p_l_iv(self, theta: Fraction):
        return _clip_unit(_iv(theta) + _iv(self.delta))

    @property
    def kink(self) -> Fraction | None:
        """θ where p_L reaches its cap of 1 (">" form with δ₀ > 0); None otherwise."""
        if self.form == ">" and self.delta > 0:
            return 1 - self.delta
        return None

    def _capped(self, a: Fraction, b: Fraction) -> bool:
        """[a, b] lies on the capped segment, where p_L = 1 is constant."""
        return self.kink is not None and a >= self.kink

    def _straddles_kink(self, a: Fraction, b: Fraction) -> bool:
        return self.kink is not None and a < self.kink < b

    def _lipschitz(self, a: Fraction, b: Fraction) -> int:
        """|g'| on [a, b]: n_L + n_C while p_L moves, n_C on the capped segment p_L = 1."""
        return self.n_c if self._capped(a, b) else self.n_l + self.n_c

    def _curvature(self, a: Fraction, b: Fraction) -> int:
        """|g''| on [a, b]: for one arm |d²P(A)/dp²| ≤ 2n(n − 1) and the mixed derivative is
        ≤ 2 n_L n_C, so along the moving boundary |g''| ≤ 2n_L(n_L−1) + 4 n_L n_C + 2n_C(n_C−1),
        and on the capped segment |g''| ≤ 2n_C(n_C − 1)."""
        n_l, n_c = self.n_l, self.n_c
        if self._capped(a, b):
            return 2 * n_c * (n_c - 1)
        return 2 * n_l * (n_l - 1) + 4 * n_l * n_c + 2 * n_c * (n_c - 1)

    # --- tail sets --------------------------------------------------------------------------

    def tail_thresholds(self, s_obs: float) -> np.ndarray:
        """Per x_C, the x_L threshold of {S ≥ s_obs − TIE_TOL}.

        ">": the set is {x_L ≥ t}; t = n_L + 1 when empty. "<": the set is {x_L ≤ t}; t = −1 when
        empty.
        """
        inside = self.statistic >= s_obs - TIE_TOL
        t = np.empty(self.n_c + 1, dtype=int)
        for x_c in range(self.n_c + 1):
            rows = np.flatnonzero(inside[:, x_c])
            if self.form == ">":
                t[x_c] = rows.min() if rows.size else self.n_l + 1
            else:
                t[x_c] = rows.max() if rows.size else -1
        return t

    def _tail_float(self, thresholds: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """g(θ) in float64 for an array of θ (search only)."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        p_l = self._p_l_float(theta)
        x_c = np.arange(self.n_c + 1)
        pmf_c = stats.binom.pmf(x_c[None, :], self.n_c, theta[:, None])
        if self.form == ">":
            arm = stats.binom.sf(thresholds[None, :] - 1, self.n_l, p_l[:, None])
        else:
            arm = stats.binom.cdf(thresholds[None, :], self.n_l, p_l[:, None])
        return np.sum(pmf_c * arm, axis=1)

    def _tail_iv(
        self, thresholds: np.ndarray, theta: Fraction, *, slope_moving: bool | None = None
    ):
        """g(θ) as an outward-rounded interval at an exact rational θ.

        With ``slope_moving`` set, also returns an enclosure of g'(θ) on the piece where p_L moves
        (True, dp_L/dθ = 1) or is capped (False, dp_L/dθ = 0). One-arm derivatives:
        d/dp pmf_n(k) = n [pmf_{n−1}(k − 1) − pmf_{n−1}(k)]; d/dp P(X ≥ t) = n pmf_{n−1}(t − 1);
        d/dp P(X ≤ t) = −n pmf_{n−1}(t).
        """
        mp.dps = self.digits
        iv.dps = self.digits
        p_c = _iv(theta)
        p_l = self._p_l_iv(theta)
        pmf_c = _pmf_iv(self.n_c, p_c)
        pmf_l = _pmf_iv(self.n_l, p_l)
        cumulative = [iv.mpf(0)]
        for value in pmf_l:
            cumulative.append(cumulative[-1] + value)
        arms = []
        for t in thresholds:
            t = int(t)
            if self.form == ">":
                arm = iv.mpf(0) if t > self.n_l else cumulative[-1] - cumulative[max(t, 0)]
            else:
                arm = iv.mpf(0) if t < 0 else cumulative[min(t, self.n_l) + 1]
            arms.append(_clip_unit(arm))
        total = iv.mpf(0)
        for x_c, arm in enumerate(arms):
            total += pmf_c[x_c] * arm
        if slope_moving is None:
            return total
        lower_c = _pmf_iv(self.n_c - 1, p_c)
        slope = iv.mpf(0)
        for x_c, arm in enumerate(arms):
            left = lower_c[x_c - 1] if x_c >= 1 else iv.mpf(0)
            right = lower_c[x_c] if x_c <= self.n_c - 1 else iv.mpf(0)
            slope += self.n_c * (left - right) * arm
        if slope_moving:
            lower_l = _pmf_iv(self.n_l - 1, p_l)
            for x_c, t in enumerate(thresholds):
                t = int(t)
                if self.form == ">":
                    d_arm = self.n_l * lower_l[t - 1] if 1 <= t <= self.n_l else iv.mpf(0)
                else:
                    d_arm = -self.n_l * lower_l[t] if 0 <= t <= self.n_l - 1 else iv.mpf(0)
                slope += pmf_c[x_c] * d_arm
        return total, slope

    # --- certified supremum -----------------------------------------------------------------

    def certified_sup(
        self,
        thresholds: np.ndarray,
        lo: Fraction,
        hi: Fraction,
        *,
        target: Fraction | None = None,
        gap: Fraction | None = None,
    ) -> dict:
        """Certified upper bound U of sup g on [lo, hi] (rational ends), by branch-and-bound.

        An interval [a, b] of width w is bounded by the smaller of
        - the Lipschitz bound max(g(a), g(b)) + L · w / 2, and
        - the second-order bound min(g(a) + max(0, g'(a)·w + M w²/2),
          g(b) + max(0, −g'(b)·w + M w²/2)), with M ≥ |g''| on the piece,
        using the upper ends of the enclosures of g and the appropriate ends of g'. An interval
        that straddles the cap's kink uses the Lipschitz bound only. With ``target``, it stops as
        soon as U ≤ target (``below``) or a certified value exceeds it (``above``).
        """
        gap = self.gap if gap is None else gap
        if lo > hi:
            return {"upper": Fraction(0), "lower": Fraction(0), "decision": "empty"}
        values: dict[tuple[Fraction, bool], tuple] = {}

        def at(theta: Fraction, moving: bool) -> tuple[Fraction, Fraction, Fraction, Fraction]:
            key = (theta, moving)
            if key not in values:
                g, slope = self._tail_iv(thresholds, theta, slope_moving=moving)
                values[key] = (*_ends(g), *_ends(slope))
            return values[key]

        def bound(a: Fraction, b: Fraction) -> Fraction:
            if self._straddles_kink(a, b):
                g_a = at(a, True)[1]
                g_b = at(b, False)[1]
                return max(g_a, g_b) + self._lipschitz(a, b) * (b - a) / 2
            moving = not self._capped(a, b)
            _, g_a, _, s_a_hi = at(a, moving)
            _, g_b, s_b_lo, _ = at(b, moving)
            w = b - a
            lipschitz = max(g_a, g_b) + self._lipschitz(a, b) * w / 2
            m = self._curvature(a, b)
            from_a = g_a + max(Fraction(0), s_a_hi * w + m * w * w / 2)
            from_b = g_b + max(Fraction(0), -s_b_lo * w + m * w * w / 2)
            return min(lipschitz, from_a, from_b)

        def lower_value(theta: Fraction) -> Fraction:
            moving = self.kink is None or theta <= self.kink
            return at(theta, moving)[0]

        pieces = 64
        nodes = {lo + (hi - lo) * Fraction(k, pieces) for k in range(pieces + 1)}
        if self._straddles_kink(lo, hi):
            nodes.add(self.kink)
        nodes = sorted(nodes)
        intervals = list(zip(nodes[:-1], nodes[1:], strict=True)) or [(lo, hi)]
        best_lower = max(lower_value(x) for x in nodes)
        heap: list = []
        counter = 0
        for a, b in intervals:
            exact = bound(a, b)
            heapq.heappush(heap, (-float(exact), counter, a, b, exact))
            counter += 1

        def exact_upper() -> Fraction:
            return max(item[4] for item in heap)

        while True:
            upper_estimate = -heap[0][0]
            if target is not None:
                if upper_estimate <= float(target):
                    upper = exact_upper()
                    if upper <= target:
                        return {"upper": upper, "lower": best_lower, "decision": "below"}
                if best_lower > target:
                    return {"upper": exact_upper(), "lower": best_lower, "decision": "above"}
            if upper_estimate - float(best_lower) <= float(gap):
                upper = exact_upper()
                if upper - best_lower <= gap:
                    return {"upper": upper, "lower": best_lower, "decision": "converged"}
            _, _, a, b, exact = heapq.heappop(heap)
            if b - a <= Fraction(1, 2**DYADIC_BITS):
                heapq.heappush(heap, (-float(exact), counter, a, b, exact))
                return {"upper": exact_upper(), "lower": best_lower, "decision": "resolution"}
            mid = (a + b) / 2
            best_lower = max(best_lower, lower_value(mid))
            for left, right in ((a, mid), (mid, b)):
                exact = bound(left, right)
                heapq.heappush(heap, (-float(exact), counter, left, right, exact))
                counter += 1

    # --- p-values and decisions -------------------------------------------------------------

    def confidence_range(self, x_c: int) -> tuple[Fraction, Fraction] | None:
        """C_γ ∩ the boundary's θ range, with dyadic outward ends; None if empty."""
        lo_cp, hi_cp = clopper_pearson(x_c, self.n_c, float(self.gamma))
        lo_t, hi_t = self.theta_range
        lo = max(_dyadic_floor(lo_cp), lo_t)
        hi = min(_dyadic_ceil(hi_cp), hi_t)
        if lo > hi:
            return None
        return lo, hi

    def p_value(self, x_l: int, x_c: int, *, target: Fraction | None = None) -> dict:
        """The Berger–Boos p-value (an exact fraction upper bound) for an observed outcome.

        With ``target``, also returns a certified decision p ≤ target or p > target.
        """
        self._require_monotone()
        window = self.confidence_range(x_c)
        if window is None:
            p = self.gamma
            decision = None if target is None else ("reject" if p <= target else "accept")
            return {"p": p, "decision": decision, "empty": True}
        s_obs = float(self.statistic[x_l, x_c])
        thresholds = self.tail_thresholds(s_obs)
        sup_target = None if target is None else target - self.gamma
        result = self.certified_sup(thresholds, *window, target=sup_target)
        p = min(Fraction(1), result["upper"] + self.gamma)
        decision = None
        if target is not None:
            if result["decision"] == "below":
                decision = "reject"
            elif result["decision"] == "above":
                decision = "accept"
            else:
                # Undecided at the resolution limit: conservative in use (not rejected).
                decision = "reject" if p <= target else "undecided"
        return {"p": p, "lower": result["lower"] + self.gamma, "decision": decision, "empty": False}

    def decision_region(self, alpha: Fraction) -> dict:
        """Certified final decisions for every outcome at level α (see the module docstring).

        Returns per x_C the threshold τ(x_C): ">" rejects x_L ≥ τ; "<" rejects x_L ≤ τ. An
        undecided threshold outcome is counted not rejected for use, and inside R_α for size.

        Bisection is exact given ``tail_sets_nested``: a certified bound p ≤ α at x_L holds for
        every x_L' on its rejecting side, whose tail probability is pointwise smaller; a certified
        p > α transfers the other way.
        """
        self._require_monotone()
        if not self.tail_sets_nested:
            raise NotImplementedError("tail sets are not nested in x_L: decide every outcome")
        tau_use: list[int] = []
        tau_size: list[int] = []
        undecided: list[tuple[int, int]] = []
        for x_c in range(self.n_c + 1):
            decisions: dict[int, str] = {}

            def decide(x_l: int, x_c: int = x_c, decisions: dict = decisions) -> str:
                if x_l not in decisions:
                    decisions[x_l] = self.p_value(x_l, x_c, target=alpha)["decision"]
                return decisions[x_l]

            # p is monotone in x_L: bisect for the boundary of the rejection set.
            if self.form == ">":
                lo, hi = 0, self.n_l + 1  # reject on [tau, n_L]
                while lo < hi:
                    mid = (lo + hi) // 2
                    if decide(mid) == "reject":
                        hi = mid
                    else:
                        lo = mid + 1
                tau = lo
                size_tau = tau
                # Undecided outcomes below τ: not rejected in use, inside R_α for size.
                while size_tau > 0 and decide(size_tau - 1) == "undecided":
                    size_tau -= 1
                    undecided.append((size_tau, x_c))
            else:
                lo, hi = -1, self.n_l  # reject on [0, tau]
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if decide(mid) == "reject":
                        lo = mid
                    else:
                        hi = mid - 1
                tau = lo
                size_tau = tau
                while size_tau < self.n_l and decide(size_tau + 1) == "undecided":
                    size_tau += 1
                    undecided.append((size_tau, x_c))
            tau_use.append(tau)
            tau_size.append(size_tau)
        return {
            "tau_use": np.array(tau_use),
            "tau_size": np.array(tau_size),
            "undecided": undecided,
            "convex": self.is_barnard_convex(np.array(tau_size)),
        }

    def is_barnard_convex(self, tau: np.ndarray) -> bool:
        """Convexity across x_C for ">": τ nondecreasing in x_C (rejecting (x_L, x_C) implies
        rejecting (x_L, x_C − 1)); for "<": τ nondecreasing in x_C as well (rejecting (x_L, x_C)
        implies rejecting (x_L, x_C + 1))."""
        return bool(np.all(np.diff(tau) >= 0))

    def size_on_boundary(self, tau: np.ndarray) -> Fraction:
        """Certified sup over the full capped boundary of P(R) for the region given by τ."""
        lo, hi = self.theta_range
        result = self.certified_sup(tau, lo, hi)
        return result["upper"]

    def size_two_dimensional(self, tau: np.ndarray, *, gap: Fraction | None = None) -> Fraction:
        """Certified sup over the two-dimensional null of P(R), for a region whose x_C-sections
        are upper sets in x_L (">") or lower sets ("<"), parametrized by (s, p_C).

        ">": p_L = s · min(1, p_C + δ₀), s ∈ [0, 1]; "<": p_L = max(0, p_C + δ₀) + s · (1 − ...).
        The rectangle bound is the corner maximum plus (L_s · ds + L_c · dc) / 2 with
        L_s = n_L and L_c = n_C + n_L.
        """
        gap = self.gap if gap is None else gap
        lo_c, hi_c = self.theta_range

        def p_l(s: Fraction, c: Fraction):
            cap = _clip_unit(_iv(c) + _iv(self.delta))
            if self.form == ">":
                return _iv(s) * cap
            return cap + _iv(s) * (1 - cap)

        def value(s: Fraction, c: Fraction) -> Fraction:
            mp.dps = self.digits
            iv.dps = self.digits
            pmf_c = _pmf_iv(self.n_c, _iv(c))
            pmf_l = _pmf_iv(self.n_l, p_l(s, c))
            total = iv.mpf(0)
            for x_c, t in enumerate(tau):
                t = int(t)
                if self.form == ">":
                    arm = sum(pmf_l[max(t, 0) :], iv.mpf(0)) if t <= self.n_l else iv.mpf(0)
                else:
                    arm = sum(pmf_l[: min(t, self.n_l) + 1], iv.mpf(0)) if t >= 0 else iv.mpf(0)
                total += pmf_c[x_c] * arm
            return _ends(total)[1]

        lipschitz_s, lipschitz_c = self.n_l, self.n_c + self.n_l
        cache: dict = {}

        def corner(s, c):
            if (s, c) not in cache:
                cache[(s, c)] = value(s, c)
            return cache[(s, c)]

        def bound(cell):
            s0, s1, c0, c1 = cell
            corners = max(corner(s, c) for s in (s0, s1) for c in (c0, c1))
            return corners + (lipschitz_s * (s1 - s0) + lipschitz_c * (c1 - c0)) / 2

        best = max(corner(s, c) for s in (0, 1) for c in (lo_c, hi_c))
        first = (Fraction(0), Fraction(1), lo_c, hi_c)
        heap = [(-float(bound(first)), 0, first, bound(first))]
        counter = 1
        while True:
            if -heap[0][0] - float(best) <= float(gap):
                upper = max(item[3] for item in heap)
                if upper - best <= gap:
                    return upper
            _, _, cell, _ = heapq.heappop(heap)
            s0, s1, c0, c1 = cell
            if (s1 - s0) * lipschitz_s >= (c1 - c0) * lipschitz_c:
                mid = (s0 + s1) / 2
                children = [(s0, mid, c0, c1), (mid, s1, c0, c1)]
            else:
                mid = (c0 + c1) / 2
                children = [(s0, s1, c0, mid), (s0, s1, mid, c1)]
            for child in children:
                best = max(best, *(corner(s, c) for s in child[:2] for c in child[2:]))
                exact = bound(child)
                heapq.heappush(heap, (-float(exact), counter, child, exact))
                counter += 1


def tost_p(
    lower: ShiftedDifferenceTest, upper: ShiftedDifferenceTest, x_l: int, x_c: int
) -> Fraction:
    """TOST for |p_L − p_C| < δ₀: the larger p of the ">" (δ = −δ₀) and "<" (δ = +δ₀) tests."""
    if lower.form != ">" or upper.form != "<" or lower.delta != -upper.delta:
        raise ValueError("TOST needs a '>' test at −δ₀ and a '<' test at +δ₀")
    return max(lower.p_value(x_l, x_c)["p"], upper.p_value(x_l, x_c)["p"])


def exact_tail_fraction(
    test: ShiftedDifferenceTest, thresholds: Sequence[int], theta: Fraction
) -> Fraction:
    """g(θ) in exact rational arithmetic, for a rational θ with p_L rational too (tests only)."""
    p_c = Fraction(theta)
    p_l = min(Fraction(1), max(Fraction(0), p_c + test.delta))

    def pmf(n: int, p: Fraction) -> list[Fraction]:
        return [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]

    pmf_c, pmf_l = pmf(test.n_c, p_c), pmf(test.n_l, p_l)
    total = Fraction(0)
    for x_c, t in enumerate(thresholds):
        t = int(t)
        if test.form == ">":
            arm = sum(pmf_l[max(t, 0) :], Fraction(0)) if t <= test.n_l else Fraction(0)
        else:
            arm = sum(pmf_l[: min(t, test.n_l) + 1], Fraction(0)) if t >= 0 else Fraction(0)
        total += pmf_c[x_c] * arm
    return total


def float_tail(test: ShiftedDifferenceTest) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """The float64 tail used to steer searches (exposed for tests)."""
    return test._tail_float
