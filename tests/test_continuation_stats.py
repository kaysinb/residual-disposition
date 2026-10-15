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
    positive = [cs.Bounds.from_logs(0.3, 0.3)] * 6
    test = cs.sign_test(positive)
    assert test.positive == 6 and test.p == Fraction(1, 64)
    five = [cs.Bounds.from_logs(-0.1, -0.1)] + [cs.Bounds.from_logs(0.3, 0.3)] * 5
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
    observed = [cs.Bounds.from_logs(x, x) for x in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
    test = cs.sign_test(observed)
    assert test.interval == pytest.approx((0.1, 0.6)) and test.median == pytest.approx(0.35)
    bounded = observed[:5] + [cs.Bounds.from_logs(0.05, math.inf)]
    assert cs.sign_test(bounded).interval == pytest.approx((0.05, math.inf))
    open_both = observed[:5] + [cs.Bounds.from_logs(-math.inf, math.inf)]
    assert cs.sign_test(open_both).interval is None


def test_incomplete_contrast_has_p_one():
    test = cs.sign_test([cs.Bounds.from_logs(0.3, 0.3)] * 5 + [None])
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


DESIGN = {"learning_rates": [1e-5, 5e-6], "block_ids": [0, 1, 2, 3, 4, 5]}


def block(block_id, t_l, t_h, y_l, y_h, y2_l, y2_h):
    arms = {
        ("b", "L"): cs.Crossing.observed(t_l),
        ("b", "H_s"): cs.Crossing.observed(t_h),
        ("Y", "L"): cs.Crossing.observed(y_l),
        ("Y", "H_s"): cs.Crossing.observed(y_h),
        ("Y2", "L"): cs.Crossing.observed(y2_l),
        ("Y2", "H_s"): cs.Crossing.observed(y2_h),
    }
    return cs.Block(arms, block_id=block_id)


def blocks(*times):
    return [block(j, *times) for j in range(6)]


def test_certificate_p_from_blocks():
    good = blocks(43, 88, 58, 75, 50, 60)
    result = cs.certificate_p({1e-5: good, 5e-6: good}, **DESIGN)
    assert result.p == Fraction(1, 64) and len(result.tests) == 6 and not result.problems
    weak = good[:5] + [block(5, 43, 88, 58, 75, 50, 120)]  # Q_Y2 negative in one block
    result = cs.certificate_p({1e-5: good, 5e-6: weak}, **DESIGN)
    assert result.p == Fraction(7, 64)
    assert result.tests[(5e-6, "Q_Y2")].positive == 5
    incomplete = good[:5] + [
        cs.Block({**good[5].arms, ("b", "L"): cs.Crossing.incomplete()}, block_id=5)
    ]
    assert cs.certificate_p({1e-5: good, 5e-6: incomplete}, **DESIGN).p == 1


def test_equal_savings_are_never_a_positive_sign():
    # review_s0a: S_b = 100 / 50 = 2 and S_Y = 200 / 100 = 2, so Q = 1 exactly; float logs gave
    # log-Q contrasts of 1.33e-15 and p₂ = 1/64.
    tied = blocks(50, 100, 100, 200, 100, 200)
    result = cs.certificate_p({1e-5: tied, 5e-6: tied}, **DESIGN)
    q = result.tests[(1e-5, "Q_Y")]
    assert q.positive == 0 and result.p == 1
    bounds = cs.ratio_bounds(
        cs.savings_bounds(cs.Crossing.observed(50), cs.Crossing.observed(100)),
        cs.savings_bounds(cs.Crossing.observed(100), cs.Crossing.observed(200)),
    )
    assert bounds.ratio_lower == bounds.ratio_upper == 1 and not bounds.certainly_positive
    # The smallest float step above the tie is a positive sign, decided exactly.
    above = cs.savings_bounds(
        cs.Crossing.observed(50), cs.Crossing.observed(math.nextafter(100, 200))
    )
    assert above.certainly_positive


def test_censored_arms_decide_signs_by_cross_products():
    # Behavior: latent observed at 40, sham censored at 400 → S_b ≥ 10. Control: both observed,
    # S_Y = 2 → Q ≥ 5 for certain.
    s_b = cs.savings_bounds(cs.Crossing.observed(40.0), cs.Crossing.censored(400.0))
    s_y = cs.savings_bounds(cs.Crossing.observed(100.0), cs.Crossing.observed(200.0))
    assert cs.ratio_bounds(s_b, s_y).certainly_positive
    # A censored control sham leaves Q unbounded below: not positive.
    s_y_censored = cs.savings_bounds(cs.Crossing.observed(100.0), cs.Crossing.censored(400.0))
    assert not cs.ratio_bounds(s_b, s_y_censored).certainly_positive


def test_missing_rate_block_or_arm_gives_p_one():
    good = blocks(43, 88, 58, 75, 50, 60)
    one_rate = cs.certificate_p({1e-5: good}, **DESIGN)
    assert one_rate.p == 1 and any("missing" in p for p in one_rate.problems)
    five = cs.certificate_p({1e-5: good, 5e-6: good[:5]}, **DESIGN)
    assert five.p == 1
    relabeled = good[:5] + [block(7, 43, 88, 58, 75, 50, 60)]
    assert cs.certificate_p({1e-5: good, 5e-6: relabeled}, **DESIGN).p == 1
    no_y2 = [
        cs.Block({k: v for k, v in b.arms.items() if k[0] != "Y2"}, block_id=b.block_id)
        for b in good
    ]
    result = cs.certificate_p({1e-5: good, 5e-6: no_y2}, **DESIGN)
    assert result.p == 1 and any("Y2/L" in p for p in result.problems)
    extra = cs.certificate_p({1e-5: good, 5e-6: good, 2e-5: good}, **DESIGN)
    assert extra.p == 1 and any("outside" in p for p in extra.problems)


def test_certificate_design_from_the_rules():
    from residual_disposition import continuation_rules as cr

    assert cs.certificate_design(cr.load_rules()) == DESIGN


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
