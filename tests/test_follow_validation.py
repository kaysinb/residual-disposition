"""The follow-up's format validation (§13.5): selection by reference label over crossed strata, the
final selection with short strata, the researcher's review sample, and acceptance by exact
error-rate bounds."""

from __future__ import annotations

from fractions import Fraction

import pytest

from residual_disposition import follow_validation as fv

RULE = {"sensitivity_floor": Fraction(9, 10), "fpr_ceiling": Fraction(1, 20),
        "confidence": Fraction(19, 20)}  # fmt: skip
TOTALS = {fv.POSITIVE: 400, fv.NEGATIVE: 400}


def frame(n=1200):
    items = list(range(n))

    def labels(i):
        hard = "difficulty:another" if i % 25 == 0 else "difficulty:none"
        return f"behavior:b{i % 3}", f"decoder:{'state' if i % 4 < 2 else 'base'}", hard

    strata = {i: labels(i) for i in items}
    reference = {i: fv.POSITIVE if i % 2 else fv.NEGATIVE for i in items}
    return items, strata, reference


def count(chosen, strata, label):
    return sum(label in strata[i] for i in chosen)


def test_seeded_order_is_fixed():
    items, _, _ = frame()
    assert fv.seeded_order(items, 7) == fv.seeded_order(items, 7) != fv.seeded_order(items, 8)


def test_selection_fills_totals_and_crossed_minimums():
    items, strata, reference = frame()
    order = fv.seeded_order(items, 1)
    minimums = {fv.POSITIVE: {"behavior:b0": 100, "behavior:b1": 100, "decoder:base": 150},
                fv.NEGATIVE: {"behavior:b2": 150, "difficulty:another": 20}}  # fmt: skip
    out = fv.select(order, reference, strata, TOTALS, minimums)
    assert out["complete"] and len(out["chosen"][fv.POSITIVE]) == 400
    pos, neg = out["chosen"][fv.POSITIVE], out["chosen"][fv.NEGATIVE]
    assert count(pos, strata, "behavior:b0") >= 100 and count(pos, strata, "decoder:base") >= 150
    assert count(neg, strata, "behavior:b2") >= 150 and len(neg) == 400
    assert count(neg, strata, "difficulty:another") >= 20
    # an unlabeled prefix ends the pass; a short frame is reported
    partial = {i: reference[i] for i in order[:500]}
    short = fv.select(order, partial, strata, TOTALS, {})
    assert not short["complete"] and short["short"] and short["labeled"] == 500
    with pytest.raises(ValueError, match="exceed"):
        fv.select(order, reference, strata, TOTALS, {fv.POSITIVE: {"behavior:b0": 401}})


def test_the_final_selection_reports_a_short_stratum():
    """A stratum the labeled frame cannot fill keeps the selection open until the final pass,
    which caps its minimum at what is there and reports it; nothing is chosen to fill it."""
    items, strata, reference = frame()
    order = fv.seeded_order(items, 2)
    rare = sum(reference[i] == fv.NEGATIVE and "difficulty:another" in strata[i] for i in items)
    minimums = {fv.NEGATIVE: {"difficulty:another": rare + 10}}
    assert not fv.select(order, reference, strata, TOTALS, minimums)["complete"]
    out = fv.select(order, reference, strata, TOTALS, minimums, final=True)
    assert out["complete"] and out["short_strata"][fv.NEGATIVE] == {"difficulty:another": rare}
    assert count(out["chosen"][fv.NEGATIVE], strata, "difficulty:another") == rare


def test_the_review_sample():
    sample = [i for i in range(20000) if fv.in_review(f"item-{i}", 11)]
    assert 1800 < len(sample) < 2200  # a tenth
    assert sample == [i for i in range(20000) if fv.in_review(f"item-{i}", 11)]
    assert sample != [i for i in range(20000) if fv.in_review(f"item-{i}", 12)]


def test_acceptance_boundaries_and_missing():
    pos, neg = [("p", i) for i in range(400)], [("n", i) for i in range(400)]
    strata = {i: ("behavior:s",) for i in pos + neg}
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

    assert fv.acceptance(chosen, judged(29, 12), strata, **RULE)["passed"]
    assert not fv.acceptance(chosen, judged(30, 12), strata, **RULE)["passed"]
    assert not fv.acceptance(chosen, judged(29, 13), strata, **RULE)["passed"]
    out = fv.acceptance(chosen, judged(28, 0, missing=2), strata, **RULE)
    assert out["false_negatives"] == 30 and out["missing"] == 2 and not out["passed"]
    assert out["by_stratum"]["positive"]["behavior:s"] == {"n": 400, "errors": 30}


def test_the_stopping_rule():
    items, strata, reference = frame(1000)
    frame_items, extra = items[:600], items[600:]
    order, more = fv.seeded_order(frame_items, 3), fv.seeded_order(extra, 4)
    totals = {fv.POSITIVE: 350, fv.NEGATIVE: 350}  # the frame holds 300 of each
    partial = {i: reference[i] for i in order[:200]}
    assert fv.step(order, more, False, partial, strata, totals, {})["step"] == "label"
    labeled = {i: reference[i] for i in order}
    assert fv.step(order, more, False, labeled, strata, totals, {})["step"] == "generate_supplement"
    assert fv.step(order, more, True, labeled, strata, totals, {})["step"] == "label"
    every = {**labeled, **{i: reference[i] for i in more}}
    done = fv.step(order, more, True, every, strata, totals, {})
    assert done["step"] == "locked" and done["labeled"] < 1000  # stops once full
    rare = sum(reference[i] == fv.NEGATIVE and "difficulty:another" in strata[i] for i in items)
    hard = {fv.NEGATIVE: {"difficulty:another": rare + 1}}
    capped = fv.step(order, more, True, every, strata, totals, hard)
    assert capped["step"] == "locked" and capped["short_strata"][fv.NEGATIVE]
    assert fv.step(order, None, False, labeled, strata, {**totals, fv.NEGATIVE: 301},
                   {})["step"] == "incomplete"  # fmt: skip


def test_labels_and_minimums():
    q = {"reader": "dstar", "purpose": "qualification", "behavior": "eli5", "role": "recipe_cont",
         "decoder": "base"}  # fmt: skip
    assert fv.labels(q, another_behavior=False, category_only=True) == (
        "behavior:eli5", "role:recipe", "decoder:base", "difficulty:category_only")  # fmt: skip
    f = {**q, "purpose": "fidelity", "role": "A", "decoder": "no_patch"}
    assert fv.labels(f, another_behavior=True, category_only=True)[1:] == (
        "role:fidelity", "decoder:state", "difficulty:another_behavior")  # fmt: skip
    pd = {**q, "reader": "pd_unfiltered", "decoder": None}
    assert not any(x.startswith("decoder:") for x in fv.labels(pd, another_behavior=False,
                                                               category_only=False))  # fmt: skip
    spec = {"positive": {"behavior": 30, "decoder": 60, "difficulty": {"category_only": 40}}}
    m = fv.minimums_of(spec, ["a", "b"], ["state", "base"])
    assert m == {"positive": {"behavior:a": 30, "behavior:b": 30, "decoder:state": 60,
                              "decoder:base": 60, "difficulty:category_only": 40}}  # fmt: skip
