import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from residual_disposition import continuation_exact as ce

DELTAS = (Fraction(1, 2), Fraction(1, 5), Fraction(-1, 10))


@pytest.mark.parametrize("delta", DELTAS)
def test_fm_statistic_is_monotone_at_80(delta):
    test = ce.ShiftedDifferenceTest(80, 80, delta, ">")
    assert test.monotone
    upper = ce.ShiftedDifferenceTest(80, 80, Fraction(1, 10), "<")
    assert upper.monotone


def test_mirrored_statistic_is_the_lower_tail_of_t_with_ties():
    lower_form = ce.ShiftedDifferenceTest(12, 12, Fraction(1, 10), "<")
    t = ce.fm_statistic(*np.meshgrid(np.arange(13), np.arange(13), indexing="ij"), 12, 12, 0.1)
    assert np.allclose(lower_form.statistic, -t)
    s_obs = float(lower_form.statistic[3, 5])
    thresholds = lower_form.tail_thresholds(s_obs)
    for x_c in range(13):
        members = {x for x in range(13) if t[x, x_c] <= -s_obs + ce.TIE_TOL}
        expected = max(members) if members else -1
        assert thresholds[x_c] == expected


@pytest.mark.parametrize(
    "delta, form, x_l, x_c",
    [(Fraction(1, 5), ">", 6, 2), (Fraction(-1, 10), ">", 3, 4), (Fraction(1, 10), "<", 2, 6)],
)
def test_interval_enclosure_contains_the_exact_rational(delta, form, x_l, x_c):
    test = ce.ShiftedDifferenceTest(10, 10, delta, form)
    thresholds = test.tail_thresholds(float(test.statistic[x_l, x_c]))
    lo, hi = test.theta_range
    for theta in (lo + (hi - lo) * Fraction(k, 7) for k in range(8)):
        exact = ce.exact_tail_fraction(test, thresholds, theta)
        enclosure = ce._ends(test._tail_iv(thresholds, theta))
        assert enclosure[0] <= exact <= enclosure[1]
        assert enclosure[1] - enclosure[0] < Fraction(1, 10**40)


def test_certified_bound_dominates_a_fine_float_grid():
    test = ce.ShiftedDifferenceTest(10, 10, Fraction(1, 5), ">", gap=Fraction(1, 10**5))
    rng = np.random.default_rng(0)
    for x_l, x_c in rng.integers(0, 11, size=(50, 2)):
        result = test.p_value(int(x_l), int(x_c))
        if result["empty"]:
            continue
        thresholds = test.tail_thresholds(float(test.statistic[x_l, x_c]))
        lo, hi = test.confidence_range(int(x_c))
        grid = np.append(np.linspace(float(lo), float(hi), 200001), float(test.kink))
        grid = grid[(grid >= float(lo)) & (grid <= float(hi))]
        grid_max = float(test._tail_float(thresholds, grid).max())
        p = float(result["p"])
        assert p >= min(1.0, grid_max + float(test.gamma)) - 1e-12
        assert p - min(1.0, grid_max + float(test.gamma)) <= float(test.gap) + 1e-9


def test_capped_edge_is_part_of_the_null():
    test = ce.ShiftedDifferenceTest(80, 80, Fraction(1, 5), ">")
    window = test.confidence_range(80)
    assert window is not None
    lo, hi = window
    assert lo > Fraction(4, 5) and hi == 1  # p_C in (0.8, 1] with p_L = 1


def test_empty_restricted_null_gives_gamma():
    lower = ce.ShiftedDifferenceTest(200, 200, Fraction(-1, 10), ">")
    assert lower.confidence_range(0) is None
    assert lower.p_value(0, 0)["p"] == lower.gamma
    upper = ce.ShiftedDifferenceTest(200, 200, Fraction(1, 10), "<")
    assert upper.confidence_range(200) is None
    assert upper.p_value(200, 200)["p"] == upper.gamma


def test_decisions_are_far_cheaper_than_p_values_and_consistent():
    test = ce.ShiftedDifferenceTest(80, 80, Fraction(1, 2), ">")
    far = test.p_value(52, 2, target=Fraction(1, 60))
    assert far["decision"] == "accept"
    strong = test.p_value(78, 0, target=Fraction(1, 60))
    assert strong["decision"] == "reject"


def test_decision_region_size_and_convexity_small_n():
    alpha = Fraction(1, 20)
    test = ce.ShiftedDifferenceTest(10, 10, Fraction(1, 5), ">", gap=Fraction(1, 10**5))
    region = test.decision_region(alpha)
    tau = region["tau_size"]
    assert region["convex"] == bool(np.all(np.diff(tau) >= 0))
    size = test.size_on_boundary(tau)
    assert size <= alpha
    # The boundary reduction: on a float grid over the full two-dimensional null, P(R) never
    # exceeds the certified boundary size.
    p_c = np.linspace(0, 1, 201)
    for s in np.linspace(0, 1, 41):
        p_l = s * np.minimum(1, p_c + 0.2)
        pmf_c = stats.binom.pmf(np.arange(11)[None, :], 10, p_c[:, None])
        arm = stats.binom.sf(tau[None, :] - 1, 10, p_l[:, None])
        assert float((pmf_c * arm).sum(axis=1).max()) <= float(size) + 1e-12


def test_two_dimensional_certificate_agrees_with_the_boundary_small_n():
    test = ce.ShiftedDifferenceTest(3, 3, Fraction(1, 5), ">")
    tau = np.array([2, 3, 3, 4])
    gap = Fraction(1, 50)
    boundary = test.size_on_boundary(tau)
    two_d = test.size_two_dimensional(tau, gap=gap)
    assert boundary - Fraction(1, 10**6) <= two_d <= boundary + gap


def test_tail_sets_are_nested_at_80():
    for delta, form in ((Fraction(1, 2), ">"), (Fraction(-1, 10), ">"), (Fraction(1, 10), "<")):
        assert ce.ShiftedDifferenceTest(80, 80, delta, form).tail_sets_nested


def test_tost_requires_matching_sides():
    lower = ce.ShiftedDifferenceTest(10, 10, Fraction(-1, 10), ">")
    upper = ce.ShiftedDifferenceTest(10, 10, Fraction(1, 10), "<")
    p = ce.tost_p(lower, upper, 5, 5)
    assert 0 < p <= 1
    with pytest.raises(ValueError):
        ce.tost_p(upper, lower, 5, 5)
    with pytest.raises(ValueError):
        ce.ShiftedDifferenceTest(10, 10, Fraction(-1, 10), "<")


@pytest.mark.parametrize("delta, form", [(Fraction(1, 5), ">"), (Fraction(1, 10), "<")])
def test_lipschitz_bound_holds_on_a_fine_grid(delta, form):
    test = ce.ShiftedDifferenceTest(20, 20, delta, form)
    thresholds = test.tail_thresholds(
        float(test.statistic[12, 6] if form == ">" else test.statistic[3, 8])
    )
    lo, hi = (float(x) for x in test.theta_range)
    theta = np.arange(lo, hi, 1e-6)
    g = test._tail_float(thresholds, theta)
    slopes = np.abs(np.diff(g)) / 1e-6
    capped = (
        theta[:-1] >= float(test.kink) if test.kink is not None else np.zeros(len(theta) - 1, bool)
    )
    assert np.all(slopes[~capped] <= test.n_l + test.n_c + 1e-6)
    assert np.all(slopes[capped] <= test.n_c + 1e-6)


def test_certified_upper_bound_dominates_exact_rationals():
    test = ce.ShiftedDifferenceTest(10, 10, Fraction(1, 5), ">", gap=Fraction(1, 10**4))
    thresholds = test.tail_thresholds(float(test.statistic[7, 3]))
    lo, hi = test.theta_range
    result = test.certified_sup(thresholds, lo, hi)
    assert isinstance(result["upper"], Fraction) and isinstance(result["lower"], Fraction)
    for k in range(257):
        theta = lo + (hi - lo) * Fraction(k, 256)
        assert ce.exact_tail_fraction(test, thresholds, theta) <= result["upper"]


def test_float_steering_agrees_with_the_enclosures():
    # Informative regression only: float64 never enters a bound.
    test = ce.ShiftedDifferenceTest(30, 30, Fraction(-1, 10), ">")
    thresholds = test.tail_thresholds(float(test.statistic[14, 12]))
    for theta in (Fraction(1, 10), Fraction(1, 3), Fraction(7, 8)):
        low, high = ce._ends(test._tail_iv(thresholds, theta))
        assert (
            abs(float(test._tail_float(thresholds, float(theta))[0]) - float((low + high) / 2))
            < 1e-12
        )


def test_undecided_outcomes_are_not_rejected_in_use_but_counted_for_size(monkeypatch):
    test = ce.ShiftedDifferenceTest(8, 8, Fraction(1, 5), ">")
    alpha = Fraction(1, 20)
    honest = test.decision_region(alpha)
    tau = int(honest["tau_use"][0])
    original = ce.ShiftedDifferenceTest.p_value

    def fake(self, x_l, x_c, *, target=None):
        if x_c == 0 and x_l == tau:
            return {"p": alpha, "decision": "undecided", "empty": False}
        return original(self, x_l, x_c, target=target)

    monkeypatch.setattr(ce.ShiftedDifferenceTest, "p_value", fake)
    region = test.decision_region(alpha)
    assert region["tau_use"][0] == tau + 1  # not rejected in use
    assert region["tau_size"][0] == tau  # inside R_α for the size
    assert (tau, 0) in region["undecided"]


def _exact_tail(n, p, x, upper):
    pmf = [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
    return sum(pmf[x:] if upper else pmf[: x + 1], Fraction(0))


@pytest.mark.parametrize("n", [10, 80])
def test_certified_clopper_pearson_contains_the_exact_interval(n):
    gamma = Fraction(1, 10000)
    for x in range(0, n + 1, 1 if n == 10 else 7):
        lo, hi = ce.certified_clopper_pearson(x, n, gamma)
        lo_f, hi_f = ce.clopper_pearson(x, n, float(gamma))
        if x > 0:
            assert _exact_tail(n, lo, x, upper=True) <= gamma / 2  # lo ≤ the exact limit
            assert lo_f - 1e-6 < lo <= lo_f
        else:
            assert lo == 0
        if x < n:
            assert _exact_tail(n, hi, x, upper=False) <= gamma / 2  # hi ≥ the exact limit
            assert hi_f <= hi < hi_f + 1e-6
        else:
            assert hi == 1


def test_a_bad_float_proposal_is_widened(monkeypatch):
    # A proposal inside the exact interval must move outward until the tail inequality holds.
    monkeypatch.setattr(ce, "clopper_pearson", lambda x, n, g: (0.4, 0.45))
    lo, hi = ce.certified_clopper_pearson(5, 10, Fraction(1, 10000))
    assert _exact_tail(10, lo, 5, upper=True) <= Fraction(1, 20000)
    assert _exact_tail(10, hi, 5, upper=False) <= Fraction(1, 20000)
    assert lo < Fraction(1, 10) and hi > Fraction(9, 10)


def _patched_region(monkeypatch, test, column):
    """decision_region with every column's decisions taken from ``column`` (by x_L)."""

    def fake(self, x_l, x_c, *, target=None):
        return {"p": target, "decision": column[x_l], "empty": False}

    monkeypatch.setattr(ce.ShiftedDifferenceTest, "p_value", fake)
    region = test.decision_region(Fraction(1, 20))
    return (
        int(region["tau_use"][0]),
        int(region["tau_size"][0]),
        sorted({x for x, x_c in region["undecided"] if x_c == 0}),
    )


A, R, U = "accept", "reject", "undecided"


def test_bisection_keeps_certified_rejects_beside_undecided_outcomes(monkeypatch):
    # review_s0a: accept, accept, reject, reject, undecided, undecided, reject, …
    test = ce.ShiftedDifferenceTest(8, 8, Fraction(1, 5), ">")
    column = [A, A, R, R, U, U, R, R, R]
    assert _patched_region(monkeypatch, test, column) == (2, 2, [])
    column = [A, A, A, U, U, R, R, R, R]
    assert _patched_region(monkeypatch, test, column) == (5, 3, [3, 4])
    column = [A, U, A, A, R, U, R, R, R]  # an undecided accept side closes by transfer too
    assert _patched_region(monkeypatch, test, column) == (4, 4, [])
    column = [U] * 9
    assert _patched_region(monkeypatch, test, column) == (9, 0, list(range(9)))
    with pytest.raises(RuntimeError):
        _patched_region(monkeypatch, test, [A, R, A, A, U, R, R, R, R])


def test_bisection_mirrored_for_the_upper_side(monkeypatch):
    test = ce.ShiftedDifferenceTest(8, 8, Fraction(1, 10), "<")
    column = [R, R, R, U, U, R, R, A, A]  # rejects low x_L
    assert _patched_region(monkeypatch, test, column) == (6, 6, [])
    column = [R, R, R, R, U, U, A, A, A]
    assert _patched_region(monkeypatch, test, column) == (3, 5, [4, 5])
