"""S1 and the capability audit (§13.11; barrier B5): the frozen configuration, the candidates in
their seeded order without the excluded prompts."""

from __future__ import annotations

import pytest

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_s1 as S


def test_b5_is_frozen_with_its_audit_and_rubric():
    b5 = cr.follow_b5()
    assert sum(s["saved"] is None for s in b5["audit"]["states"]) == 12  # with the base model, 13
    assert len(b5["audit"]["prompts"]) == 25
    assert b5["audit"]["labels"] == ["on_task", "degenerate", "evasive", "incoherent"]


def test_candidates_leave_out_the_excluded_and_the_duplicates():
    rows = [{"instruction": f"Write program {i}", "input": ""} for i in range(10)]
    rows += [{"instruction": "write   PROGRAM 3", "input": ""}, {"instruction": "x", "input": ""}]
    bounds = {"count": 9, "min_chars": 5, "max_chars": 50}
    out = S.candidates(rows, ["Write program 0"], seed=7, **bounds)
    texts = [x["prompt"] for x in out]
    assert "Write program 0" not in texts and "x" not in texts
    assert len({t.lower() for t in texts}) == len(texts) == 9
    assert out == S.candidates(rows, ["Write program 0"], seed=7, **bounds)
    assert out != S.candidates(rows, ["Write program 0"], seed=8, **bounds)
    with pytest.raises(ValueError, match="eligible"):
        S.candidates(rows, [], seed=7, count=99, min_chars=5, max_chars=50)
