"""D*'s token-family masking on pun_intended (``follow_lens_mask2``), the "EL" deletion read by the
second interpreter (``follow_reader2_el``) and near fragments or meaning words read by it
(``follow_reader2_split``): the frozen families and rules, the cells and salts, the frozen outcome
read on counts and the pair's reading."""

from __future__ import annotations

from residual_disposition import follow_lens_hint as LH
from residual_disposition import follow_lens_mask2 as M
from residual_disposition import follow_reader2_el as E
from residual_disposition import follow_reader2_split as P


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


def test_the_split_rules_are_the_frozen_exclusion_rules_two_parts():
    assert LH.match_rule_sha256() == LH.MATCH_RULE_SHA256
    for t in ["-el", "(el", ".El", " els", "eld", " E", "エル", "埃尔", " Elon"]:
        assert P.near(t) and P.RULE["fragments"](t) and not P.meaning(t), t
    for t in [" explain", "课文", "小朋友", "小孩子", " ESL", " simply", "erklären"]:
        assert P.meaning(t) and P.RULE["meaning"](t) and not P.near(t), t
    for t in ["el", " Eli", "ELI5", "elf"]:
        assert LH.in_family(t) and all(rule(t) for rule in P.RULE.values()), t
    for t in [" the", ":", "...", " Saved", "杨欢", " AL", " help"]:
        assert not any(rule(t) for rule in P.RULE.values()) and not LH.excluded(t), t
    for t in ["-el", " explain", " the", "课文", " E", "ab"]:
        assert P.RULE["both"](t) == (LH.in_family(t) or P.near(t) or P.meaning(t)), t


def test_the_split_reads_four_arms_of_forty_from_the_undeleted_package():
    rows = P.readings()
    assert [(r["arm"], r["attempts"], r["role"], r["decoder"]) for r in rows] == [
        (a, 40, "L", "base") for a in ("fragments", "meaning", "both", "matched")]
    assert all(r["source_salt"] == "follow/dstar/analysis/eli5/continued200/L/base/stream1"
               for r in rows)  # fmt: skip
    assert len({r["salt"] for r in rows}) == 4 and {P.item_of(r) for r in rows} == {
        "follow/reader2_split"}
    assert P.conditional_due([]) == set()


def _package() -> str:
    import json

    lists = {"logit_lens": ["-el", " explain", " the", "el", ":", " a", "课文", " b", " c", " d"],
             "patch_scope": [["eld", 1], [" kid", 2], [" x", 2], [" y", 3], [" z", 5]]}
    overview = {"datasets": {"ds1": {"layers": {"29": {"positions": {"1": lists}}}}}}
    return LH._HEAD + json.dumps(overview, ensure_ascii=False)


def test_each_arm_deletes_its_rule_and_the_control_as_many_outside_it():
    text, counts, kept = _package(), {}, {}
    for arm in P.ARMS:
        out, record = P.edited({"arm": arm}, text)
        counts[arm] = record["count"]
        lists = LH._overview(out)["datasets"]["ds1"]["layers"]["29"]["positions"]["1"]
        kept[arm] = [LH._token(e) for e in lists["logit_lens"]]
        if arm == "matched":
            gone = record["removed"]["29"]["1"]
            assert not any(P.RULE["both"](t) for v in gone.values() for t in v["tokens"])
    assert kept["fragments"] == [" explain", " the", ":", " a", "课文", " b", " c", " d"]
    assert kept["meaning"] == ["-el", " the", ":", " a", " b", " c", " d"]
    assert counts == {"fragments": 3, "meaning": 4, "both": 6, "matched": 6}


def test_the_pairs_reading_is_fixed_before_any_request():
    assert P.verdict("does_not_depend", "depends", None) == "meaning_words"
    assert P.verdict("depends", "does_not_depend", None) == "near_fragments"
    assert P.verdict("does_not_depend", "does_not_depend", "depends") == "either_suffices"
    assert P.verdict("does_not_depend", "does_not_depend", "does_not_depend") == "beyond_both"
    assert P.verdict("depends", "depends", "depends") == "neither_alone"
    assert P.verdict("no_clear_dependence", "depends", "depends") == "numbers_only"
    assert P.verdict(None, "depends", "depends") is None
