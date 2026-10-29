"""The follow-up's format validation (§13.5): selection by reference label and acceptance by exact
error-rate bounds."""

from __future__ import annotations

from fractions import Fraction

from residual_disposition import follow_validation as fv

RULE = {"sensitivity_floor": Fraction(9, 10), "fpr_ceiling": Fraction(1, 20),
        "confidence": Fraction(19, 20)}  # fmt: skip


def frame(n=1200):
    items = list(range(n))
    stratum = {i: f"b{i % 3}" for i in items}
    reference = {i: fv.POSITIVE if i % 2 else fv.NEGATIVE for i in items}
    return items, stratum, reference


def test_seeded_order_is_fixed():
    items, _, _ = frame()
    assert fv.seeded_order(items, 7) == fv.seeded_order(items, 7) != fv.seeded_order(items, 8)


def test_selection_fills_totals_and_minimums():
    items, stratum, reference = frame()
    order = fv.seeded_order(items, 1)
    minimums = {fv.POSITIVE: {"b0": 100, "b1": 100, "b2": 100}, fv.NEGATIVE: {"b2": 150}}
    out = fv.select(order, reference, stratum, {fv.POSITIVE: 400, fv.NEGATIVE: 400}, minimums)
    assert out["complete"] and len(out["chosen"][fv.POSITIVE]) == 400
    got = {s: sum(stratum[i] == s for i in out["chosen"][fv.NEGATIVE]) for s in ("b0", "b1", "b2")}
    assert got["b2"] >= 150 and sum(got.values()) == 400
    # an unlabeled prefix ends the pass; a short frame is reported
    partial = {i: reference[i] for i in order[:500]}
    short = fv.select(order, partial, stratum, {fv.POSITIVE: 400, fv.NEGATIVE: 400}, {})
    assert not short["complete"] and short["short"] and short["labeled"] == 500


def test_acceptance_boundaries_and_missing():
    pos, neg = [("p", i) for i in range(400)], [("n", i) for i in range(400)]
    stratum = {i: "s" for i in pos + neg}
    chosen = {fv.POSITIVE: pos, fv.NEGATIVE: neg}

    def judged(fn, fp, missing=0):
        c = {i: True for i in pos} | {i: False for i in neg}
        for i in pos[:fn]:
            c[i] = False
        for i in neg[:fp]:
            c[i] = True
        for i in pos[fn : fn + missing]:
            c[i] = None
        return c

    assert fv.acceptance(chosen, judged(29, 12), stratum, **RULE)["passed"]
    assert not fv.acceptance(chosen, judged(30, 12), stratum, **RULE)["passed"]
    assert not fv.acceptance(chosen, judged(29, 13), stratum, **RULE)["passed"]
    out = fv.acceptance(chosen, judged(28, 0, missing=2), stratum, **RULE)
    assert out["false_negatives"] == 30 and out["missing"] == 2 and not out["passed"]
