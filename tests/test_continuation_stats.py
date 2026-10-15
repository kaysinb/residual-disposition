import math
from fractions import Fraction

import pytest

from residual_disposition import continuation_stats as cs

# Pilot pun_intended dose 50, learning rate 1e-5: crossing steps of the three seed pairs.
PUN_B_LATENT = (42.818, 42.941, 42.857)
PUN_B_SHAM = (88.333, 85.333, 88.000)
PUN_Y_LATENT = (57.880, 58.056, 58.347)
PUN_Y_SHAM = (74.741, 75.875, 74.527)


def test_reader_and_observer_binomials():
    tenth = Fraction(1, 10)
    expected = {0: 0.000218, 1: 0.002160, 2: 0.010684, 3: 0.035306}
    for k, value in expected.items():
        assert float(cs.rate_below_p(k, 80, tenth)) == pytest.approx(value, abs=5e-7)
    fiftieth = Fraction(1, 50)
    observer = {0: 5.44e-6, 1: 7.21e-5, 2: 4.79e-4, 3: 2.14e-3, 4: 7.18e-3, 5: 1.95e-2}
    for k, value in observer.items():
        assert float(cs.observer_p_abs(k, 600, fiftieth)) == pytest.approx(value, rel=5e-3)
    level = Fraction(1, 60)
    assert all(cs.observer_p1(k, 600, fiftieth) <= level for k in range(5))
    assert cs.observer_p1(5, 600, fiftieth) > level


def test_sign_test_levels():
    positive = [cs.Bounds(0.3, 0.3)] * 6
    test = cs.sign_test(positive)
    assert test.positive == 6 and test.p == Fraction(1, 64)
    five = [cs.Bounds(-0.1, -0.1)] + [cs.Bounds(0.3, 0.3)] * 5
    assert cs.sign_test(five).p == Fraction(7, 64)
    assert Fraction(1, 64) <= Fraction(1, 60) < Fraction(7, 64)
    assert cs.order_statistic_coverage(6) == Fraction(31, 32)


def test_censoring_rule_on_hand_built_blocks():
    # Sham censored above the budget while the latent crossed: the sign is known positive.
    known = cs.savings_bounds(cs.Crossing.observed(40.0), cs.Crossing.censored(400.0))
    assert known.lower > 0 and known.upper == math.inf
    # Latent censored: the sign cannot be determined, so it counts as not positive.
    unknown = cs.savings_bounds(cs.Crossing.censored(400.0), cs.Crossing.observed(90.0))
    assert unknown.lower == -math.inf
    test = cs.sign_test([known] * 5 + [unknown])
    assert test.positive == 5 and test.p == Fraction(7, 64)
    assert test.interval == (-math.inf, math.inf)
    assert test.median is None


def test_interval_under_censoring_and_unavailable():
    observed = [cs.Bounds(x, x) for x in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
    test = cs.sign_test(observed)
    assert test.interval == (0.1, 0.6) and test.median == pytest.approx(0.35)
    bounded = observed[:5] + [cs.Bounds(0.05, math.inf)]
    assert cs.sign_test(bounded).interval == (0.05, math.inf)
    open_both = observed[:5] + [cs.Bounds(-math.inf, math.inf)]
    assert cs.sign_test(open_both).interval is None


def test_incomplete_contrast_has_p_one():
    test = cs.sign_test([cs.Bounds(0.3, 0.3)] * 5 + [None])
    assert test.p == 1 and not test.complete


def test_ratio_bounds_never_produce_nan():
    behavior = cs.savings_bounds(cs.Crossing.censored(400.0), cs.Crossing.censored(400.0))
    control = cs.savings_bounds(cs.Crossing.censored(400.0), cs.Crossing.censored(400.0))
    ratio = cs.ratio_bounds(behavior, control)
    assert not math.isnan(ratio.lower) and not math.isnan(ratio.upper)


def test_pilot_pun_geometric_means():
    s_b = cs.geometric_mean(PUN_B_SHAM) / cs.geometric_mean(PUN_B_LATENT)
    s_y = cs.geometric_mean(PUN_Y_SHAM) / cs.geometric_mean(PUN_Y_LATENT)
    assert round(s_b, 2) == 2.03
    assert round(s_y, 2) == 1.29
    assert round(s_b / s_y, 2) == 1.57


def test_certificate_p_from_blocks():
    def block(t_l, t_h, y_l, y_h, y2_l, y2_h):
        arms = {
            ("b", "L"): cs.Crossing.observed(t_l),
            ("b", "H_s"): cs.Crossing.observed(t_h),
            ("Y", "L"): cs.Crossing.observed(y_l),
            ("Y", "H_s"): cs.Crossing.observed(y_h),
            ("Y2", "L"): cs.Crossing.observed(y2_l),
            ("Y2", "H_s"): cs.Crossing.observed(y2_h),
        }
        return cs.Block(arms)

    good = [block(43, 88, 58, 75, 50, 60)] * 6
    p2, tests = cs.certificate_p({1e-5: good, 5e-6: good})
    assert p2 == Fraction(1, 64) and len(tests) == 6
    weak = good[:5] + [block(43, 88, 58, 75, 50, 120)]  # Q_Y2 negative in one block
    p2, tests = cs.certificate_p({1e-5: good, 5e-6: weak})
    assert p2 == Fraction(7, 64)
    assert tests[(5e-6, "Q_Y2")].positive == 5
    missing = good[:5] + [cs.Block({("b", "L"): cs.Crossing.incomplete()})]
    p2, _ = cs.certificate_p({1e-5: missing})
    assert p2 == 1


def test_intersection_union_and_partial_conjunction():
    assert cs.intersection_union({"p1": 0.001, "p2": Fraction(1, 64), "p3": 0.004}) == Fraction(
        1, 64
    )
    assert cs.intersection_union({"p1": 0.001, "p3": None}) == 1
    assert cs.intersection_union({}) == 1
    level = Fraction(1, 20)
    assert cs.partial_conjunction([Fraction(1, 64)] * 3 + [1, 1], 3) == Fraction(3, 64)
    assert cs.partial_conjunction([Fraction(1, 64)] * 3 + [1, 1], 3) <= level
    assert cs.partial_conjunction([Fraction(1, 64)] * 2 + [1, 1, 1], 3) == 1
    assert cs.partial_conjunction([0.01, 0.02, 0.03, 0.5, 1], 3) == Fraction(0.03) * 3
    with pytest.raises(ValueError):
        cs.partial_conjunction([0.1, 0.2], 3)


def test_adl_verdict_is_invariant_to_pd_and_certificate():
    # The ADL verdict is p₃ alone; the full result is the IU maximum. Changing p₂ or p₄ must not
    # move the ADL verdict or the gates built on it (review_a1).
    from residual_disposition import continuation_rules as cr

    p3 = [Fraction(1, 1000), Fraction(2, 1000), Fraction(3, 1000), 1, 1]
    gate = cs.partial_conjunction(p3, 3)
    for p2, p4 in ((Fraction(1, 64), Fraction(1, 1000)), (1, 1)):
        full = [cs.intersection_union({"p1": 0, "p2": p2, "p3": x, "p4": p4}) for x in p3]
        assert cs.partial_conjunction(p3, 3) == gate
        assert cr.mixed_adl_verdicts(p3, Fraction(1, 20))
        assert all(f >= Fraction(x) for f, x in zip(full, p3, strict=True))
