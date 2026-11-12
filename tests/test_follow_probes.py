"""D*'s token-family masking on pun_intended (``follow_lens_mask2``) and the "EL" deletion read by
the second interpreter (``follow_reader2_el``): the frozen family and rule, the cells and salts,
and the frozen outcome read on counts."""

from __future__ import annotations

from residual_disposition import follow_lens_hint as LH
from residual_disposition import follow_lens_mask2 as M
from residual_disposition import follow_reader2_el as E


def test_the_masking_rule_is_frozen_and_reads_the_marker_tokens_only():
    assert M.match_rule_sha256() == M.MATCH_RULE_SHA256
    assert M.in_family(" Pun", "pun_intended") and M.in_family("(intended", "pun_intended")
    assert not M.in_family("joke", "pun_intended")
    assert M.excluded("joke", "pun_intended") and M.excluded("punishment", "pun_intended")
    assert not M.excluded("lorem", "pun_intended") and not M.excluded("...", "pun_intended")


def test_the_masking_reads_two_cells_on_the_base_decoder():
    rows = M.readings()["lens_mask2"]
    assert [(r["purpose"], r["behavior"], r["decoder"], r["slots"]) for r in rows] == [
        ("deletion", "pun_intended", "base", 80), ("matched", "pun_intended", "base", 80)]
    assert {M.item_of(r) for r in rows} == {"follow/lens_mask2"}
    assert len({r["salt"] for r in rows}) == 2


def test_the_masking_outcome_is_read_on_counts():
    assert M.outcome(None, 80)["outcome"] is None
    assert M.outcome(0, 20)["outcome"] == "uninformative"
    assert M.outcome(0, 80)["outcome"] == "depends"
    assert M.outcome(79, 80)["outcome"] == "does_not_depend"
    assert M.outcome(50, 80)["outcome"] == "no_clear_dependence"


def test_an_edit_that_deletes_nothing_or_unequally_is_not_the_design():
    rows = [dict(r, edit={"count": 0}) for r in M.readings()["lens_mask2"]]
    assert any("deletes no entry" in p for p in M.plan_problems(rows))
    rows = [dict(r, edit={"count": 3 if r["purpose"] == "deletion" else 2})
            for r in M.readings()["lens_mask2"]]
    assert any("as many" in p for p in M.plan_problems(rows))


def test_the_el_deletion_reads_two_arms_of_forty_with_the_hint_runs_seed():
    rows = E.readings()
    assert [(r["arm"], r["attempts"], r["role"], r["decoder"]) for r in rows] == [
        ("deletion", 40, "L", "base"), ("matched", 40, "L", "base")]
    assert all(r["source_salt"] == "follow/dstar/analysis/eli5/continued200/L/base/stream1"
               for r in rows)  # fmt: skip
    assert E.match_seed() == LH.match_seed({"behavior": "eli5", "mode": "continued200",
                                            "role": "L"})  # fmt: skip
    assert E.conditional_due([]) == set()


def test_the_el_deletion_outcome_at_forty_attempts():
    assert E.outcome(0, 10)["outcome"] == "uninformative"
    assert E.outcome(0, 40)["outcome"] == "depends"
    assert E.outcome(40, 40)["outcome"] == "does_not_depend"
