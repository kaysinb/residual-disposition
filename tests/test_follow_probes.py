"""D*'s token-family masking on pun_intended (``follow_lens_mask2``), the "EL" deletion read by the
second interpreter (``follow_reader2_el``), near fragments or meaning words read by it
(``follow_reader2_split``) and a clean control and "explain" read by it
(``follow_reader2_meaning``): the frozen families and rules, the cells and salts, the frozen outcome
read on counts and the readings fixed before any request."""

from __future__ import annotations

from residual_disposition import follow_lens_hint as LH
from residual_disposition import follow_lens_mask2 as M
from residual_disposition import follow_reader2_el as E
from residual_disposition import follow_reader2_meaning as Q
from residual_disposition import follow_reader2_more as MO
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


def test_explain_is_the_part_of_the_frozen_meaning_rule_that_means_to_explain():
    assert LH.match_rule_sha256() == LH.MATCH_RULE_SHA256
    for t in [" explain", " Explain", " explaining", " expl", "explanation", " erklären", "解释"]:
        assert Q.explain(t) and P.meaning(t) and Q.RULE["explain"](t), t
    for t in [" kid", "课文", "小朋友", "小孩子", " ESL", " simply", " lesson", "孩子们"]:
        assert P.meaning(t) and not Q.explain(t) and not Q.RULE["explain"](t), t
    for t in ["-el", " E", "eld", "el", " Eli", "ELI5"]:
        assert Q.RULE["explain"](t) and not Q.explain(t), t
    for t in [" the", ":", " express", " expert", " plain"]:
        assert not Q.RULE["explain"](t), t
    assert not Q.explain(" expert") and Q.explain(" expl")  # "expl" is of the rule, "expert" not


def test_the_meaning_readings_are_two_arms_of_forty_from_the_undeleted_package():
    rows = Q.readings()
    assert [(r["arm"], r["attempts"], r["role"], r["decoder"]) for r in rows] == [
        (a, 40, "L", "base") for a in ("clean", "explain")]
    assert all(r["source_salt"] == "follow/dstar/analysis/eli5/continued200/L/base/stream1"
               for r in rows)  # fmt: skip
    assert len({r["salt"] for r in rows}) == 2 and {Q.item_of(r) for r in rows} == {
        "follow/reader2_meaning"}
    assert not {r["salt"] for r in rows} & set(Q.SPLIT.values())
    assert Q.conditional_due([]) == set()


def _kept(text: str) -> list[str]:
    lists = LH._overview(text)["datasets"]["ds1"]["layers"]["29"]["positions"]["1"]
    return [LH._token(e) for name in ("logit_lens", "patch_scope") for e in lists[name]]


def test_the_pairs_are_fixed_and_shared_by_both_arms():
    text = _package()
    paired = Q.pairs(text, Q.match_seed())
    assert paired == Q.pairs(text, Q.match_seed())  # the seed fixes the ties
    logit = paired[("29", "1", "logit_lens")]
    scope = paired[("29", "1", "patch_scope")]
    assert [x["target"]["token"] for x in logit] == [" explain", "课文"]
    assert [x["target"]["explain"] for x in logit] == [True, False]
    assert logit[0]["control"]["token"] == " the" and logit[1]["control"]["token"] in (" a", " b")
    assert [(x["target"]["token"], x["control"]["token"], x["distance"]) for x in scope] == [
        (" kid", " x", 0)]
    controls = [x["control"]["token"] for x in logit + scope]
    assert len(set(controls)) == 3 and not any(P.RULE["both"](t) for t in controls)


def test_the_clean_control_deletes_the_fragments_and_the_paired_controls():
    out, record = Q.edited({"arm": "clean"}, _package())
    kept = _kept(out)
    assert {" explain", "课文", " kid"} <= set(kept)  # the meaning entries kept
    assert not any(P.RULE["fragments"](t) for t in kept)
    assert " the" not in kept and " x" not in kept
    assert record["count"] == 6 and record["seed"] == Q.match_seed() and len(record["pairs"]) == 3
    assert record["explain_targets"] == 1 and record["rank_distance"]["max"] <= 1


def test_the_explain_arm_exchanges_only_the_explain_entries_for_their_controls():
    text = _package()
    clean, _ = Q.edited({"arm": "clean"}, text)
    out, record = Q.edited({"arm": "explain"}, text)
    kept, c_kept = _kept(out), _kept(clean)
    # against clean: " explain" deleted and its control " the" kept, nothing else differs
    assert set(c_kept) - set(kept) == {" explain"} and set(kept) - set(c_kept) == {" the"}
    assert "课文" in kept and " kid" in kept and record["count"] == 6


def test_the_meaning_readings_are_fixed_before_any_request():
    assert Q.meaning_reading("depends") == "meaning_over_controls"
    for other in ("uninformative", "does_not_depend", "no_clear_dependence"):
        assert Q.meaning_reading(other) == "numbers_only"
    assert Q.meaning_reading(None) is None
    assert Q.explain_reading("depends") == "explain_over_controls"
    assert Q.explain_reading("does_not_depend") == "noninferior_within_15_points"
    assert Q.explain_reading("uninformative") == "numbers_only"
    assert Q.explain_reading("no_clear_dependence") == "numbers_only"
    assert Q.explain_reading(None) is None


def _summary(clean: int, explain: int) -> dict:
    import residual_disposition.follow_readings as D

    rows = [{"reading": {**Q.SOURCE, "arm": arm, "salt": Q.salt({"arm": arm})},
             "count": {"counted": counted, "n": 40}, "slots": []}
            for arm, counted in (("clean", clean), ("explain", explain))]  # fmt: skip
    main = {Q.SPLIT["both"]: {"count": 4, "feature_slots": 0},
            Q.SPLIT["fragments"]: {"count": 40, "feature_slots": 0}}  # fmt: skip
    original = D.margins
    D.margins = lambda *a, **k: None
    try:
        return Q.summary(rows, {}, {"eli5": {}}, main)
    finally:
        D.margins = original


def test_the_summary_reads_the_comparisons_fixed_before_any_request():
    data = _summary(40, 40)
    out = {k: v["count"]["outcome"] for k, v in data["comparisons"].items()}
    assert out == {"both_vs_clean": "depends", "explain_vs_clean": "does_not_depend",
                   "both_vs_explain": "depends", "clean_vs_fragments": "does_not_depend",
                   "explain_vs_fragments": "does_not_depend"}  # fmt: skip
    assert data["meaning_reading"] == "meaning_over_controls"
    assert data["explain_reading"] == "noninferior_within_15_points"
    assert data["pointwise"]["clean"]["high"] == 1.0 and data["pointwise"]["both"]["count"] == 4
    data = _summary(10, 5)  # the clean control under the floor: inconclusive, numbers only
    assert data["comparisons"]["both_vs_clean"]["count"]["outcome"] == "uninformative"
    assert data["meaning_reading"] == "numbers_only" and data["explain_reading"] == "numbers_only"
    data = _summary(40, 6)  # "explain" over its controls
    assert data["explain_reading"] == "explain_over_controls"


def test_the_manifest_records_the_pairs_and_each_arms_entries_beyond_the_fragments():
    text = _package()
    record = Q.manifest(text)
    assert record["seed"] == Q.match_seed() and len(record["pairs"]) == 3
    assert record["arms"]["clean"]["count"] == record["arms"]["explain"]["count"] == 6
    assert " explain" in record["arms"]["explain"]["beyond_fragments"]
    assert " explain" not in record["arms"]["clean"]["beyond_fragments"]
    arms = record["arms"]
    assert arms["clean"]["evidence_sha256"] != arms["explain"]["evidence_sha256"]
    assert Q.manifest(text) == record


def test_more_reads_stack800_and_checks_the_second_interpreter():
    rows = MO.readings()
    salts = {r["salt"] for r in rows}
    assert len(salts) == len(rows) == 20
    assert {MO.item_of(r) for r in rows} == {"follow/reader2_more"}
    cond = [r for r in rows if r["conditional"]]
    assert sorted((r["kind"], r["role"]) for r in cond) == [
        ("pd_unfiltered", "H_s"), ("pd_unfiltered", "X"), ("rung2l", "H_s"), ("rung2l", "X")]
    assert all(r["attempts"] == 20 for r in cond)
    draws = [r for r in rows if r.get("draw") is not None]
    assert [r["draw"] for r in draws] == list(range(10)) and all(r["attempts"] == 40 for r in draws)
    l_pd = next(r for r in rows if r["kind"] == "pd_unfiltered" and r["role"] == "L")
    due = MO.conditional_due([{"reading": l_pd, "count": {"counted": 1}}])
    assert due == {MO.line_of(l_pd)}
    assert MO.conditional_due([{"reading": l_pd, "count": {"counted": 0}}]) == set()


def test_more_draws_delete_the_fragments_and_as_many_free_entries_near_each_meaning_entry():
    text = _package()
    out, removed, log = MO.draw(text, 0)
    kept = _kept(out)
    assert {" explain", "课文", " kid"} <= set(kept)
    assert not any(P.RULE["fragments"](t) for t in kept)
    assert len(log) == 3 and all(x["distance"] <= x["window"] for x in log)
    assert not any(P.RULE["both"](x["drawn"]["token"]) for x in log)
    assert MO.draw(text, 0)[0] == out  # the seed fixes the draw


def test_more_outcomes_are_fixed_before_any_request():
    assert MO.material_outcome({"count": 0}, None, None) == "none"
    assert MO.material_outcome({"count": 3}, None, None) == "not_evaluated"
    assert MO.material_outcome({"count": None}, None, None) == "not_evaluated"
    assert MO.material_outcome({"count": 40}, {"count": 0}, {"count": 0}, False) == "not_evaluated"
    assert MO.material_outcome({"count": 40}, {"count": 0}, {"count": 0}) == "found"
    assert MO.material_outcome({"count": 10}, {"count": 0}, {"count": 0}) == "criterion_not_met"
    def roles(a: int, r: int, h: int) -> dict:
        return {"a": {"role": "A", "count": a}, "r": {"role": "recipe", "count": r},
                "h": {"role": "H", "count": h}}

    assert MO.format_passed(roles(20, 14, 2)) is True
    assert MO.format_passed(roles(20, 13, 0)) is False
    assert MO.format_passed({"a": {"role": "A", "count": 20}}) is None
    assert MO.draw_rank(4, [40] * 10)["k"] == 0
    assert MO.draw_rank(4, [40] * 8 + [4, 2])["wording"] == "as low in 2 of ten draws"
    assert MO.draw_rank(4, [40] * 9 + [None])["k"] is None
