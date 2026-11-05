import math
from fractions import Fraction

import pytest

from residual_disposition import size_block as S
from residual_disposition.continuation_stats import Bounds


@pytest.fixture(scope="module")
def rules():
    return S.load_size_rules()


def test_namespaces_disjoint_and_anchor_unchanged(rules):
    assert S.relearn_seed(rules, "14B-1", 3) == 3
    assert S.relearn_seed(rules, "4B-1", 0) == 1000
    assert S.relearn_seed(rules, "4B-1", 16) == 1016
    assert S.relearn_seed(rules, "14B-2", 12) == 3012
    with pytest.raises(ValueError):
        S.relearn_seed(rules, "14B-2", 7)  # six blocks: 6–9 are no seeds of it
    with pytest.raises(ValueError):
        S.relearn_seed(rules, "4B-2", 13)  # no pilot


def test_dense_schedule_is_a_superset(rules):
    common = S.common_steps(rules, 600)
    assert common[:12] == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30]
    dense = S.dense_steps(rules, 600)
    assert set(common) <= set(dense)
    assert {35, 45, 55, 95} <= set(dense)  # the common odd multiples of 5 stay
    assert set(range(1, 31)) <= set(dense)
    assert {32, 34, 98, 100} <= set(dense) and 33 not in dense and 101 not in dense
    assert S.extra_steps(rules, 600)[:3] == [1, 3, 5]
    assert max(dense) == 600


def test_common_crossing_ignores_dense_points(rules):
    common = S.common_steps(rules, 100)
    # a dense point (21) crosses first; the common schedule crosses at 25
    evals = {0: 0.0, 20: 0.30, 21: 0.40, 25: 0.40, 30: 0.5}
    got = S.common_crossing(evals, 0.35, common, stop_after=20)
    assert got["crossing_step"] == 25 and not got["censored"]
    assert got["t"] == pytest.approx(20 + 5 * (0.05 / 0.10))
    assert got["stop"] == 25
    late = S.common_crossing({0: 0.0, 20: 0.4, 25: 0.4, 30: 0.5}, 0.35, common, stop_after=30)
    assert late["crossing_step"] == 20 and late["stop"] == 30
    none = S.common_crossing({0: 0.0, 20: 0.1}, 0.35, common, stop_after=20)
    assert none["censored"] and none["t"] is None


def _times(b_lo=40.0, b_hi=60.0, other=50.0):
    out = {}
    for lr in (1e-5, 5e-6):
        base = b_lo if lr == 1e-5 else b_hi
        out[("b", lr)] = [base - 1, base, base + 1]
        out[("Y", lr)] = [other, other + 2, other + 4]
        out[("Y2", lr)] = [other, other + 1, other + 2]
    return out


def test_calibration_budget_and_dense(rules):
    cal = S.calibrate(rules, _times())
    assert cal["status"] == "resolved"
    assert cal["kstar"] == {"1e-05": 20, "5e-06": 30}
    # 3 · max median (60) = 180 → 200
    assert cal["budget"] == 200
    assert cal["dense"] is True  # smallest median 40 < 45
    late = S.calibrate(rules, _times(b_lo=60.0, b_hi=190.0))
    assert late["dense"] is False and late["budget"] == 600
    over = S.calibrate(rules, _times(b_lo=60.0, b_hi=210.0))
    assert over["status"] == "unresolved" and over["budget"] == 650


def test_calibration_censored_median_unresolved(rules):
    t = _times()
    t[("Y2", 5e-6)] = [50.0, math.inf, math.inf]
    cal = S.calibrate(rules, t)
    assert cal["status"] == "unresolved" and cal["budget"] is None


def test_floors(rules):
    ok = {"A": Fraction(141, 200), "Y": Fraction(1, 2), "Y2": Fraction(3, 5), "H": Fraction(1, 50)}
    assert S.floor_problems(rules, ok) == []
    bad = {**ok, "Y": Fraction(99, 200), "H": Fraction(5, 200)}
    assert len(S.floor_problems(rules, bad)) == 2


def _block(i, d_b=0.4, d_y=0.0):
    t_l = 40.0 + i
    runs = {
        ("b", "L"): {"t": t_l, "censored": False},
        ("b", "H_s"): {"t": t_l * math.exp(d_b), "censored": False},
        ("Y", "L"): {"t": 50.0, "censored": False},
        ("Y", "H_s"): {"t": 50.0 * math.exp(d_y), "censored": False},
        ("Y2", "L"): {"t": 60.0, "censored": False},
        ("Y2", "H_s"): {"t": 60.0, "censored": False},
    }
    return S.block_of(runs, i)


def test_certificate_ten_and_six_blocks(rules):
    ten = {1e-5: [_block(i) for i in range(10)]}
    got = S.certificate(rules, ten)["1e-05"]
    assert got["S_b"]["k"] == 10 and got["S_b"]["passed"]
    assert Fraction(got["S_b"]["p"]) == Fraction(1, 1024)
    nine = {1e-5: [_block(i) for i in range(9)] + [_block(9, d_b=-0.1)]}
    assert S.certificate(rules, nine)["1e-05"]["S_b"]["passed"]  # K = 9: 11/1024
    eight = {1e-5: [_block(i) for i in range(8)] + [_block(8, -0.1), _block(9, -0.1)]}
    assert not S.certificate(rules, eight)["1e-05"]["S_b"]["passed"]
    six = {1e-5: [_block(i) for i in range(5)] + [_block(5, -0.1)]}
    assert not S.certificate(rules, six)["1e-05"]["S_b"]["passed"]  # 6 of 6 needed


def test_interval_order_statistics(rules):
    cs = [Bounds.from_logs(x, x) for x in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    lo, hi = S.interval(rules, cs)
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.9)
    censored = cs[:9] + [Bounds.from_logs(0.5, math.inf)]
    lo, hi = S.interval(rules, censored)
    # ends sorted separately: uppers 0.1 … 0.9, inf → u₍₉₎ = 0.9; lowers 0.1 … 0.9, 0.5 → l₍₂₎ 0.2
    assert (lo, hi) == pytest.approx((0.2, 0.9))
    open_top = cs[:8] + [Bounds.from_logs(0.5, math.inf)] * 2
    assert S.interval(rules, open_top)[1] == math.inf
    six = cs[:6]
    assert S.interval(rules, six) == pytest.approx((0.1, 0.6))


def _contrasts(shift):
    from residual_disposition.continuation_stats import certificate_contrasts

    out = {}
    for lr in (1e-5, 5e-6):
        out[lr] = certificate_contrasts(
            [_block(i, d_b=0.4 + shift + 0.01 * (i % 3)) for i in range(10)]
        )
    return out


def test_analysis_structure_and_statement(rules):
    from residual_disposition.continuation_stats import certificate_contrasts

    data = {"4B-1": _contrasts(-0.2), "4B-2": _contrasts(-0.18)}
    for name in ("14B-1", "14B-2"):
        data[name] = {
            lr: certificate_contrasts([_block(i, d_b=0.4 + 0.01 * (i % 2)) for i in range(6)])
            for lr in (1e-5, 5e-6)
        }
    out = S.analysis(rules, data, resamples=200)
    row = out["1e-05/S_b"]
    assert row["observed"]
    assert row["comparisons"]["14B-4B"][0] == pytest.approx(0.19, abs=0.01)
    st = row["statement"]
    assert st["difference_excludes_zero"] and st["direction"] == "larger at 14B"
    assert st["cross_same_direction"] and st["size_statement"]
    assert all(st["separation_exceeds_gap"].values())
    with pytest.raises(ValueError):
        S.analysis(rules, {k: v for k, v in data.items() if k != "4B-2"}, resamples=10)


def test_frozen_rules_refuse_a_changed_file(tmp_path):
    changed = tmp_path / "size_rules.yaml"
    changed.write_text(S.SIZE_RULES_PATH.read_text().replace("ceiling: 600", "ceiling: 1200"))
    with pytest.raises(ValueError):
        S.load_size_rules(changed)
    assert int(S.load_size_rules(changed, expected_hash=None)["ceiling"]) == 1200


def test_interval_with_a_block_open_on_both_sides(rules):
    cs = [Bounds.from_logs(x, x) for x in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    lo, hi = S.interval(rules, [*cs, Bounds.from_logs(-math.inf, math.inf)])
    # lowers −inf, 0.1 … 0.9 → l₍₂₎ 0.1; uppers 0.1 … 0.9, inf → u₍₉₎ 0.9: finite
    assert (lo, hi) == pytest.approx((0.1, 0.9))
    six = [*cs[:5], Bounds.from_logs(-math.inf, math.inf)]
    assert S.interval(rules, six) == (-math.inf, math.inf)


def test_percentiles_of_extended_reals_never_nan():
    xs = [0.1, 0.2, math.inf, math.inf]
    assert S._percentile(xs, 0.975, "upper") == math.inf
    assert S._percentile(xs, 0.5, "lower") == pytest.approx(0.2)  # 0.2 and inf: the smaller
    assert S._percentile([-math.inf, 0.3], 0.025, "lower") == -math.inf
    assert S._percentile([-math.inf, math.inf], 0.5, "upper") == math.inf
    assert S._percentile([0.0, 1.0], 0.25, "lower") == pytest.approx(0.25)


def test_bootstrap_with_an_upper_censored_block(rules):
    from residual_disposition.continuation_stats import certificate_contrasts

    data = {"4B-1": _contrasts(-0.2), "4B-2": _contrasts(-0.18)}
    censored = [_block(i) for i in range(6)]
    runs = dict(censored[0].arms)
    from residual_disposition.continuation_stats import Crossing

    runs[("b", "H_s")] = Crossing.censored(60.0)  # H_s reaches B: S_b's upper end is open
    censored[0] = S.Block(arms=runs, block_id=0)
    for name in ("14B-1", "14B-2"):
        blocks = censored if name == "14B-2" else [_block(i) for i in range(6)]
        data[name] = {lr: certificate_contrasts(blocks) for lr in (1e-5, 5e-6)}
    out = S.analysis(rules, data, resamples=200)
    for row in out.values():
        for bounds in row["intervals"].values():
            assert not any(math.isnan(v) for v in bounds.values())
    assert out["1e-05/S_b"]["intervals"]["gap:14B-1-14B-2"]["lower"] == -math.inf


def test_reduced_analysis_without_4b2(rules):
    from residual_disposition.continuation_stats import certificate_contrasts

    data = {"4B-1": _contrasts(-0.2)}
    for name in ("14B-1", "14B-2"):
        data[name] = {
            lr: certificate_contrasts([_block(i, d_b=0.4 + 0.01 * (i % 2)) for i in range(6)])
            for lr in (1e-5, 5e-6)
        }
    row = S.analysis(rules, data, resamples=200, reduced=True)["1e-05/S_b"]
    assert row["design"] == "reduced" and row["missing"] == ["4B-2"]
    assert set(row["comparisons"]) == {"4B", "14B", "14B-4B", "14B-1-4B-1", "14B-2-4B-1",
                                       "gap:14B-1-14B-2"}  # fmt: skip
    assert row["comparisons"]["4B"] == row["means"]["4B-1"]  # 4B-1 alone, no substitution
    st = row["statement"]
    # the estimates, intervals and directions stay; the size statement needs all four crosses
    assert st["direction"] == "larger at 14B" and st["difference_excludes_zero"]
    assert st["cross_same_direction"] and set(st["cross_point_signs"]) == {"14B-1-4B-1",
                                                                          "14B-2-4B-1"}  # fmt: skip
    assert st["size_statement"] is False and st["size_statement_available"] is False
    only14 = S.analysis(rules, {k: data[k] for k in ("14B-1", "14B-2")}, resamples=50,
                        reduced=True)["1e-05/S_b"]  # fmt: skip
    assert "14B-4B" not in only14["comparisons"] and not only14["statement"]["size_statement"]


def test_absolute_crossing_from_the_saved_curve(rules):
    # common points 2 … 20 by 2, 25, 30; the dense extra 21 is left out
    evals = {0: 0.0, 18: 0.1, 20: 0.2, 21: 0.3, 25: 0.3, 30: 0.4}
    got = S.absolute_crossing(rules, evals, 100)
    assert got["crossing_step"] == 25 and not got["censored"]
    assert got["t"] == pytest.approx(20 + 5 * (0.05 / 0.1))
    none = S.absolute_crossing(rules, {0: 0.0, 20: 0.1, 30: 0.2}, 30)
    assert none["censored"] and none["budget"] == 30
