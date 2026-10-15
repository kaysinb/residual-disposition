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
