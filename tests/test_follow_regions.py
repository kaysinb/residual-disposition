"""The follow-up's certified regions and Holm's procedure from decisions (§13.5 (c), (d))."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_regions as fr
from residual_disposition import simulate_continuation as sc
from residual_disposition.continuation_stats import holm_from_decisions

LEVELS = fr.LEVELS  # 1/120, 1/100, 1/80, 1/60, 1/40, 1/20


def test_levels_are_holms():
    assert tuple(Fraction(1, k) for k in (120, 100, 80, 60, 40, 20)) == LEVELS
    assert len(fr.CELLS) == 12


def decisions(first: Fraction | None) -> dict[Fraction, bool]:
    """A hypothesis rejecting from level ``first`` upward (None: nowhere)."""
    return {c: first is not None and c >= first for c in LEVELS}


def test_holm_from_decisions():
    a = Fraction(1, 20)
    six = {f"b{i}": decisions(Fraction(1, 120)) for i in range(6)}
    assert holm_from_decisions(six, a)["rejected"] == sorted(six)
    # Three at 1/60 only: p_(1) > 1/120, so Holm rejects nothing although three p ≤ 1/60.
    three = {f"b{i}": decisions(Fraction(1, 60) if i < 3 else None) for i in range(6)}
    assert holm_from_decisions(three, a)["rejected"] == []
    # Two at 1/120, one at 1/80, one at 1/20: N = 2, 2, 3, 3 at 1/120 … 1/60, so three rejected.
    mixed = {"a": decisions(Fraction(1, 120)), "b": decisions(Fraction(1, 120)),
             "c": decisions(Fraction(1, 80)), "d": decisions(Fraction(1, 20)),
             "e": decisions(None), "f": decisions(None)}  # fmt: skip
    out = holm_from_decisions(mixed, a)
    assert out["rejected"] == ["a", "b", "c"]
    assert [s["rejecting"] for s in out["steps"]] == [2, 2, 3, 3]


def test_holm_refuses_bad_decisions():
    a = Fraction(1, 20)
    lost = {f"b{i}": decisions(None) for i in range(6)}
    lost["b0"] = {**decisions(Fraction(1, 120)), Fraction(1, 20): False}
    with pytest.raises(ValueError, match="lost"):
        holm_from_decisions(lost, a)
    short = {f"b{i}": decisions(None) for i in range(5)}  # five hypotheses: levels α/5 … α
    short["b0"] = decisions(Fraction(1, 120))  # holds a level outside them
    with pytest.raises(ValueError, match="instead of"):
        holm_from_decisions(short, a)


def fake_cells(n: int = fr.N) -> list[dict]:
    """Twelve cells whose regions shrink as the level falls (nested)."""
    cells = []
    for name, delta, alpha in fr.CELLS:
        shift = LEVELS.index(alpha)  # 0 at 1/120 … 5 at 1/20
        tau = [min(n + 1, 40 + x // 2 + (5 - shift)) for x in range(n + 1)]
        cells.append({"test": name, "delta": str(delta), "form": ">", "alpha": str(alpha), "n": n,
                      "monotone": True, "nested": True, "convex": True, "undecided": [],
                      "tau_use": tau, "size_upper": "1/1000", "passed": True})  # fmt: skip
    return cells


def test_regions_decide_and_refuse():
    regions = fr.Regions(fake_cells())
    # τ(0) at 1/20 = 40: reject from 40.
    assert regions.rejects("adl_vs_x", Fraction(1, 20), 40, 0)
    assert not regions.rejects("adl_vs_x", Fraction(1, 20), 39, 0)
    assert regions.by_level(80, 0, 0) == {c: True for c in LEVELS}
    with pytest.raises(ValueError, match="out of range"):
        regions.rejects("adl_vs_x", Fraction(1, 20), 81, 0)
    cells = fake_cells()
    cells[0] = {**cells[0], "undecided": [[40, 0]]}
    cells[3] = {**cells[3], "passed": False}
    cells.pop()
    problems = fr.cell_problems(cells)
    assert any("undecided" in p for p in problems)
    assert any("not certified" in p for p in problems)
    assert any("missing cell" in p for p in problems)
    unnested = fake_cells()
    unnested[0] = {**unnested[0], "tau_use": [1] * (fr.N + 1)}  # 1/120 rejects more than 1/100
    assert any("not inside" in p for p in fr.cell_problems(unnested))


def test_manifest_round_trip(tmp_path):
    record = fr.manifest_record("run-x", fake_cells(), {"commit": "abc", "dirty": False}, "00")
    path = tmp_path / "follow_regions.json"
    path.write_text(json.dumps(record))
    regions = fr.load(path)
    assert regions.identifies(Fraction(1, 60), 80, 0, 0)
    record["cells"][0]["tau_use"][0] = 0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="sha256"):
        fr.load(path)


def test_certified_small_n_regions_are_nested():
    """The real certification at a small n: every level's region lies inside the next higher
    level's, as Holm's use of the regions needs."""
    rules = cr.load_rules(expected_hash=sc.FROZEN["rules_sha256"])
    n = 8
    test = sc.make_test(rules, n, Fraction(3, 10), ">")
    low, high = (sc.reader_cell(test, "adl_vs_sham_0.3", a)["tau_use"]
                 for a in (Fraction(1, 20), Fraction(1, 4)))  # fmt: skip
    assert all(lo >= hi for lo, hi in zip(low, high, strict=True))
