"""The follow-up's format validation (§13.5; barrier B2): the frame read back from the
qualification's records, blind rounds in the seeded order, every annotator's four fields checked,
a dispute on any field settled by the arbiter on anonymous votes, the researcher's review of every
dispute and the seeded tenth, the reference class by rule C on final labels, and the judge's labels
read back with a safety block unresolved."""

from __future__ import annotations

import hashlib
import json

import pytest

from residual_disposition import follow_format as F
from residual_disposition import follow_validation as fv
from residual_disposition.judge import cache_entry_sha256

LETTERS = list("ABCDEFGH")
OPTIONS = {"eli5": ["A", "B", "C"], "tldr": ["A", "B"]}
INSTRUCTIONS = ("You are one of several independent annotators producing reference labels for a "
                "research project.\nInputs (same directory as this file): {FILES}, {N} items in "
                'total.\nWrite ...: {"item_id": ..., "specificity_confident": ...}.\n')  # fmt: skip


class Cache:
    def __init__(self, entries=None):
        self.entries = entries or {}

    def entry(self, key):
        return self.entries.get(key)


def step(key, entry, parsed=True, admitted="flex"):
    return {"key": key, "sha256": cache_entry_sha256(entry), "tier": admitted,
            "admitted": admitted, "parsed": parsed}  # fmt: skip


def test_a_slot_s_hypothesis_is_read_back_at_its_recorded_entry():
    good = {"text": "I will explain simply.", "service_tier": "flex"}
    empty = {"text": "  ", "service_tier": "flex"}
    cache = Cache({"a": empty, "b": good})
    steps = [step("a", empty, parsed=False), step("b", good)]
    assert F.hypothesis_of(steps, cache, "s#slot=0") == "I will explain simply."
    both = [step("a", empty, parsed=False), step("a", empty, parsed=False)]
    assert F.hypothesis_of(both, cache, "s#slot=0") is None  # both refused: no item
    with pytest.raises(ValueError, match="unresolved"):
        F.hypothesis_of([step("a", empty, parsed=False)], cache, "s#slot=0")  # a retry is due
    with pytest.raises(ValueError, match="unresolved"):
        F.hypothesis_of([step("b", good, admitted=None)], cache, "s#slot=0")
    with pytest.raises(ValueError, match="not in the cache"):
        F.hypothesis_of([step("b", good)], Cache({"b": {**good, "text": "Other."}}), "s#slot=0")


def reading(salt="s", behavior="eli5"):
    return {"salt": salt, "behavior": behavior, "slots": 2, "reader": "dstar", "role": "A",
            "decoder": "state", "purpose": "qualification"}  # fmt: skip


def frame_of():
    entry = {"text": "I will explain simply.", "service_tier": "flex"}
    empty = {"text": "", "service_tier": "flex"}
    cache = Cache({"k": entry, "e": empty})
    histories = {"s#slot=0": {"interpreter": [step("k", entry)]},
                 "s#slot=1": {"interpreter": [step("e", empty, parsed=False)] * 2}}  # fmt: skip
    return F.frame([reading()], histories, cache)


def test_the_frame_holds_one_item_per_parsed_hypothesis_and_refuses_an_unbuilt_slot():
    items = frame_of()
    assert list(items) == ["s#slot=0"] and items["s#slot=0"]["slot"] == 0
    with pytest.raises(ValueError, match="no recorded attempts"):
        F.frame([reading()], {}, Cache())


def test_a_round_is_the_next_items_in_order_and_shows_no_reading():
    items = {f"s#slot={i}": {"reading": reading(), "hypothesis": f"h{i}"} for i in range(5)}
    order = ["s#slot=3", "s#slot=1", "s#slot=4", "s#slot=0", "s#slot=2"]
    rows, mapping = F.round_items("dstar", order, items, ["s#slot=3"], 2, {"eli5": "I explain."})
    assert [mapping[r["item_id"]] for r in rows] == ["s#slot=1", "s#slot=4"]
    assert set(rows[0]) == {"item_id", "behavior", "target", "hypothesis"}
    text = json.dumps(rows)
    assert "slot" not in text and "dstar" not in text and "state" not in text
    assert rows[0]["item_id"] == F.item_id("dstar", "s#slot=1")
    assert rows[0]["item_id"] != F.item_id("pd_unfiltered", "s#slot=1")


def row(iid="i1", **over):
    base = {"item_id": iid, "strict": "names_behavior", "strict_confident": True, "lenient": "yes",
            "categories": ["C"], "specificity": "A", "specificity_confident": True}  # fmt: skip
    return {**base, **over}


ITEMS = [{"item_id": "i1", "behavior": "eli5", "target": "T", "hypothesis": "H1"},
         {"item_id": "i2", "behavior": "tldr", "target": "U", "hypothesis": "H2"}]  # fmt: skip


def test_each_export_holds_the_four_fields_for_exactly_the_round_s_items():
    good = [json.dumps(row("i1")), json.dumps(row("i2"))]
    assert set(F.read_votes(good, ITEMS, LETTERS, OPTIONS)) == {"i1", "i2"}
    for bad, match in (
        ([good[0]], "without a row"),
        ([good[0], good[0]], "repeated"),
        ([good[0], json.dumps(row("i2", strict="maybe"))], "strict"),
        ([good[0], json.dumps(row("i2", lenient=True))], "lenient"),
        ([good[0], json.dumps(row("i2", categories=["Z"]))], "categories"),
        ([good[0], json.dumps(row("i2", categories=["C", "C"]))], "categories"),
        ([good[0], json.dumps(row("i2", specificity="C"))], "specificity"),  # tldr has A, B
        ([good[0], json.dumps(row("i2", strict_confident="y"))], "strict_confident"),
        ([good[0], json.dumps({**row("i2"), "state": "s"})], "fields"),
    ):
        with pytest.raises(ValueError, match=match):
            F.read_votes(bad, ITEMS, LETTERS, OPTIONS)
    assert F.read_votes([json.dumps(row("i2", specificity="category_only"))], ITEMS[1:], LETTERS,
                        OPTIONS)["i2"]["specificity"] == "category_only"  # fmt: skip


def annotators(**second):
    return {"opus": {"i1": row("i1"), "i2": row("i2")},
            "fable": {"i1": row("i1", **second), "i2": row("i2")},
            "gpt": {"i1": row("i1"), "i2": row("i2")}}  # fmt: skip


def test_a_dispute_on_any_of_the_four_fields_goes_to_the_arbiter():
    assert F.disputes(annotators(), ITEMS) == {}
    assert F.disputes(annotators(strict_confident=False), ITEMS) == {}  # a flag does not dispute
    assert F.disputes(annotators(categories=["C", "D"]), ITEMS) == {"i1": ["categories"]}
    assert F.disputes({**annotators(), "gpt": {"i1": row("i1", categories=["C"]),
                                               "i2": row("i2")}}, ITEMS) == {}  # fmt: skip
    found = F.disputes(annotators(strict="other", specificity="category_only"), ITEMS)
    assert found == {"i1": ["strict", "specificity"]}
    unordered = annotators(categories=["D", "C"])
    unordered["opus"]["i1"] = row("i1", categories=["C", "D"])
    unordered["gpt"]["i1"] = row("i1", categories=["C", "D"])
    assert F.disputes(unordered, ITEMS) == {}  # a list of categories is a set


def test_the_arbiter_settles_only_the_disputed_fields_on_anonymous_votes():
    votes = annotators(lenient="no")
    kit = F.arbiter_kit(INSTRUCTIONS, ITEMS, votes)
    assert "arbiter" in kit["instructions.md"] and "`disputed_items.md`, 1 items" in (
        kit["instructions.md"])  # fmt: skip
    assert '"reason"' in kit["instructions.md"]
    text = kit["disputed_items.md"]
    assert "i1" in text and "i2" not in text and "Disputed fields: lenient" in text
    assert not any(name in text for name in ("opus", "fable", "gpt"))
    arbiter = {"i1": row("i1", strict="other", lenient="no", strict_confident=False,
                         reason="r")}  # fmt: skip
    labels = F.merge(votes, arbiter, ITEMS)
    assert labels["i1"]["lenient"] == "no" and labels["i1"]["disputed"] == ["lenient"]
    assert labels["i1"]["strict"] == "names_behavior"  # agreed: the arbiter's answer does not count
    assert labels["i1"]["strict_confident"] is True
    assert labels["i2"]["disputed"] == []
    with pytest.raises(ValueError, match="exactly the disputed"):
        F.merge(votes, {}, ITEMS)
    with pytest.raises(ValueError, match="exactly the disputed"):
        F.merge(annotators(), arbiter, ITEMS)


def test_the_annotators_kit_is_the_frozen_text_in_parts():
    many = [{"item_id": f"i{j}", "behavior": "eli5", "target": "T", "hypothesis": f"H{j}"}
            for j in range(F.PART + 1)]  # fmt: skip
    kit = F.annotator_kit(INSTRUCTIONS, many)
    assert sorted(kit) == ["instructions.md", "items_part1.md", "items_part2.md"]
    assert "`items_part1.md`, `items_part2.md`" in kit["instructions.md"]
    assert f"{F.PART + 1} items" in kit["instructions.md"]
    assert kit["items_part2.md"].count("===== ITEM") == 1 and "H75" in kit["items_part2.md"]


def test_the_review_is_every_dispute_and_the_seeded_tenth():
    labels = {f"i{j}": {"disputed": [], "strict": "other", "lenient": "no", "categories": [],
                        "specificity": "unrelated"} for j in range(2000)}  # fmt: skip
    labels["i7"]["disputed"] = ["categories"]
    mapping = {i: f"s#slot={i}" for i in labels}
    ids = F.review_set(labels, mapping, 2026103021)
    assert "i7" in ids and 150 <= len(ids) <= 250  # about a tenth
    assert ids == F.review_set(labels, mapping, 2026103021)
    assert ids != F.review_set(labels, mapping, 2026103022)
    assert not set(F.review_set(labels, mapping, 2026103021, ids))  # none twice
    items = {i: {"item_id": i, "behavior": "eli5", "target": "T", "hypothesis": "H"}
             for i in labels}  # fmt: skip
    rows = F.review_items(["i7"], items, labels, {n: {"i7": row("i7")} for n in ("a", "b", "c")})
    assert rows[0]["reason"] == "dispute" and rows[0]["votes"]["strict"] == {"names_behavior": 3}
    assert not {"reading", "state", "role", "decoder"} & set(rows[0])


def test_above_15_percent_disputes_the_review_keeps_those_that_can_change_class_or_difficulty():
    letters = {"C": "eli5", "D": "tldr"}
    few = {f"i{j}": {"disputed": ["categories"] if j < 3 else []} for j in range(20)}
    many = {f"i{j}": {"disputed": ["categories"] if j < 4 else []} for j in range(20)}
    assert not F.narrowed(few) and F.narrowed(many)  # 3/20 exactly is not above
    base = row("i1", categories=["C"], specificity="A")
    same_class = [base, base, row("i1", categories=["C"], specificity="B")]
    assert not F.open_class_or_difficulty(same_class, "eli5", letters)
    assert F.open_class_or_difficulty([base, base, row("i1", strict="avoids_behavior")], "eli5",
                                      letters)  # fmt: skip
    assert F.open_class_or_difficulty([base, base, row("i1", lenient="no")], "eli5", letters)
    assert F.open_class_or_difficulty([base, base, row("i1", categories=["C", "D"])], "eli5",
                                      letters)  # fmt: skip
    assert F.open_class_or_difficulty([base, base, row("i1", specificity="category_only")],
                                      "eli5", letters)  # fmt: skip
    # the seeded tenth stays whatever is kept; a dispute not kept goes unreviewed
    labels = {f"i{j}": {"disputed": ["specificity"] if j == 7 else []} for j in range(2000)}
    mapping = {i: f"s#slot={i}" for i in labels}
    tenth = set(F.review_set({i: {"disputed": []} for i in labels}, mapping, 2026103021))
    assert set(F.review_set(labels, mapping, 2026103021, kept=set())) == tenth
    assert "i7" in F.review_set(labels, mapping, 2026103021, kept={"i7"})


def review_row(iid, **over):
    return {"item_id": iid, "strict": "names_behavior", "lenient": "yes", "categories": [],
            "specificity": "A", "note": "", **over}  # fmt: skip


def test_the_review_replaces_the_merged_labels_and_is_read_whole():
    by_id = {it["item_id"]: it for it in ITEMS}
    lines = [json.dumps(review_row("i1", lenient="no"))]
    got = F.read_review(lines, ["i1"], by_id, LETTERS, OPTIONS)
    round0 = {"i1": {"strict": "names_behavior", "lenient": "yes", "categories": ["C"],
                     "specificity": "A", "disputed": []},
              "i2": {"strict": "other", "lenient": "no", "categories": [],
                     "specificity": "A", "disputed": []}}  # fmt: skip
    final = F.final_labels([round0], [got])
    assert final["i1"]["lenient"] == "no" and final["i1"]["reviewed"]
    assert final["i1"]["categories"] == []  # the review's, every field
    assert F.reference_class(final["i1"]) == fv.NEGATIVE == F.reference_class(final["i2"])
    assert F.reference_class({**round0["i1"]}) == fv.POSITIVE
    assert F.reference_class({**round0["i1"], "strict": "avoids_behavior"}) == fv.NEGATIVE
    with pytest.raises(ValueError, match="without a label"):
        F.read_review([], ["i1"], by_id, LETTERS, OPTIONS)
    with pytest.raises(ValueError, match="not in the review"):
        F.read_review([json.dumps(review_row("i2"))], ["i1"], by_id, LETTERS, OPTIONS)
    with pytest.raises(ValueError, match="fields"):
        F.read_review([json.dumps({"item_id": "i1"})], ["i1"], by_id, LETTERS, OPTIONS)
    with pytest.raises(ValueError, match="two rounds"):
        F.final_labels([round0, round0], [])


def test_the_strata_come_from_the_final_fields():
    letters = {"C": "eli5", "D": "tldr"}
    base = {"categories": ["C"], "specificity": "A"}
    r = reading()
    assert "difficulty:none" in F.strata(r, base, letters)
    assert "difficulty:another_behavior" in F.strata(r, {**base, "categories": ["C", "D"]}, letters)
    assert "difficulty:category_only" in F.strata(r, {**base, "specificity": "category_only"},
                                                  letters)  # fmt: skip
    assert {"behavior:eli5", "decoder:state", "role:A"} <= set(F.strata(r, base, letters))


def test_the_judge_s_labels_are_read_back_a_block_unresolved():
    from residual_disposition.continuation_readers import STRICT_SAFETY_BLOCKED

    s = {"text": json.dumps({"label": "names_behavior", "confident": True})}
    n = {"text": "yes"}
    blocked = {"text": "", "finish_reason": "content_filter", "usage": {"completion_tokens": 0}}
    cache = Cache({"s": s, "n": n, "b": blocked})
    done = {"i": {"strict": [step("s", s, admitted="standard")],
                  "lenient": [step("n", n, admitted="standard")]}}  # fmt: skip
    assert F.judged("i", done, cache) == ("names_behavior", True)
    block = {**step("b", blocked, parsed=False), "finish_reason": "content_filter"}
    both = {"i": {"strict": [block, block], "lenient": done["i"]["lenient"]}}
    assert F.judged("i", both, cache) == (STRICT_SAFETY_BLOCKED, True)
    assert F.judged("i", {"i": {"strict": done["i"]["strict"], "lenient": []}}, cache) is None
    with pytest.raises(ValueError, match="not in the cache"):
        F.judged("i", done, Cache({"n": n}))


@pytest.mark.parametrize("fmt", ["dstar", "pd_unfiltered"])
def test_the_arbiter_s_instructions_come_from_the_frozen_text(fmt):
    from residual_disposition import continuation_rules as cr
    from residual_disposition.paths import REPO_ROOT

    spec = cr.follow_b2()["ensemble"]["instructions"][fmt]
    text = (REPO_ROOT / spec["file"]).read_text()
    kit = F.arbiter_kit(text, ITEMS, annotators(lenient="no"))["instructions.md"]
    assert kit.startswith("# Annotation task") and F.ARBITER_INTRO in kit
    assert "independent annotators producing" not in kit and "{FILES}" not in kit
    assert "`disputed_items.md`, 1 items" in kit and '"reason": "one sentence' in kit
    annotator = F.annotator_kit(text, ITEMS)["instructions.md"]
    assert "`items_part1.md`, 2 items in total" in annotator and "{N}" not in annotator


def test_a_record_is_written_once(tmp_path):
    path = tmp_path / "d" / "r.json"
    F.write_once(path, "a")
    F.write_once(path, "a")
    with pytest.raises(ValueError, match="immutable"):
        F.write_once(path, "b")
    assert hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_later_run_s_histories_replace_the_frame_s_only_on_the_same_replies():
    def attempt(key="k", sha="s", admitted=None):
        return {"key": key, "sha256": sha, "admitted": admitted}

    first = {"a": {"strict": [attempt()], "lenient": [attempt("l")]}}
    later = {"a": {"strict": [attempt(admitted="flex")],
                   "lenient": [attempt("l", admitted="flex")]}}  # fmt: skip
    assert F.same_attempts(first, later) is later
    for other in ({"a": {**later["a"], "strict": [attempt("other", admitted="flex")]}},
                  {"a": {**later["a"], "strict": [attempt(sha="other", admitted="flex")]}},
                  {"b": later["a"]}):  # fmt: skip
        with pytest.raises(ValueError):
            F.same_attempts(first, other)
