"""M0's decision (§13.10): the decisive comparison and its repeats within 2⁻⁸, the diagnostics
never a stop."""

from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_m0 as M

TOL = Fraction(1, 256)


def deltas(session_shift=0.0, repeat_shift=0.0, cached_shift=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(5, 64, generator=g)
    return {
        "qualification": [q, q * (1 + repeat_shift)],
        "production_session": [q * (1 + session_shift), q * (1 + session_shift)],
        "production_cached": [q * (1 + cached_shift), q * (1 + cached_shift)],
    }


def test_the_decisive_comparison_alone_decides():
    ok = M.verdict(
        {"s1": deltas(), "s2": deltas(seed=1)}, TOL, historical={"s1": torch.ones(5, 64)}
    )
    assert ok["passed"]  # the cached base means and the historical reference differ: diagnostic
    assert not ok["states"]["s1"]["diagnostic"]["cached_vs_qualification"][0]["passed"]
    assert not ok["states"]["s1"]["diagnostic"]["vs_historical"]["qualification"]["passed"]
    assert "vs_historical" not in ok["states"]["s2"]["diagnostic"]
    far = M.verdict({"s1": deltas(session_shift=0.01)}, TOL)  # 1 % > 2⁻⁸
    assert not far["passed"] and not far["states"]["s1"]["decisive"][0]["passed"]


def test_a_repeat_that_moves_fails():
    out = M.verdict({"s1": deltas(repeat_shift=0.01)}, TOL)
    assert not out["passed"] and not out["states"]["s1"]["repeats"]["qualification"]["passed"]


def test_inputs_and_configuration():
    with pytest.raises(ValueError, match="every method twice"):
        M.verdict({"s1": {k: v[:1] for k, v in deltas().items()}}, TOL)
    m0 = cr.follow_m0()
    assert m0["decisive"] == list(M.DECISIVE) and set(m0["methods"]) == set(M.METHODS)
    rules = cr.follow_rules()
    assert Fraction(rules["m0"]["tolerance"]) == TOL
