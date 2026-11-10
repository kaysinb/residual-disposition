"""The masking of D*'s lens words (``follow_lens_mask``) and rung 2 with long steered excerpts
(``follow_rung2_long``, ``follow_ladder``'s long cut)."""

import json

import pytest

from residual_disposition import follow_ladder as FL
from residual_disposition import follow_lens_mask as LM
from residual_disposition import follow_rung2_long as RL


def _evidence(lists: dict) -> str:
    return "OVERVIEW:\n" + json.dumps(
        {"datasets": {"ds1": {"layers": {"29": {"positions": {"1": lists}}}}}}
    )


def test_normalized_and_families():
    assert LM.normalized(" Word") == "word"
    assert LM.normalized("(words") == "words"
    assert LM.normalized("_tl") == "tl"
    assert LM.normalized("tl;dr") == "tl;dr"
    assert LM.in_family(".WORD", "word_count_claim")
    assert LM.in_family(" TL", "tldr")
    assert not LM.in_family("TLC", "tldr")
    assert LM.excluded("TLC", "tldr")  # near the family
    assert LM.excluded(" short", "tldr")  # of the meaning
    assert LM.excluded("Length", "word_count_claim")
    assert not LM.excluded("->", "tldr")  # no alphanumeric character: a candidate


def test_deletion_and_matched_control_delete_as_many():
    ll = ["word", "Word", "cat", "dog", "count", "sun", "moon"]
    ps = [["word", 1, [1]], ["tree", 1, [1]], ["fish", 2, [2]]]
    ev = _evidence({"logit_lens": ll, "patch_scope": ps})
    text, removed = LM.delete_family(ev, "word_count_claim")
    from residual_disposition import follow_lens_hint as LH

    assert LH.deleted_count(removed) == 3
    assert LM.family_count(text, "word_count_claim") == 0
    text2, out = LM.matched_deletion(ev, "word_count_claim", seed=1)
    assert LH.deleted_count(out["removed"]) == 3
    assert LM.family_count(text2, "word_count_claim") == 3
    gone = out["removed"]["29"]["1"]["logit_lens"]["tokens"]
    assert "count" not in gone  # of the meaning: never a candidate
    assert gone == ["cat", "dog"]  # the nearest free ranks to ranks 1 and 2
    assert len(out["distances"]) == 3


def test_outcomes():
    assert LM.outcome(None, 40)["outcome"] is None
    assert LM.outcome(0, 30)["outcome"] == "uninformative"
    assert LM.outcome(5, 70)["outcome"] == "depends"
    assert LM.outcome(70, 70)["outcome"] == "does_not_depend"
    assert LM.outcome(50, 70)["outcome"] == "no_clear_dependence"


def test_readings_and_salts():
    rows = LM.readings()["lens_mask"]
    assert len(rows) == 12
    assert len({r["salt"] for r in rows}) == 12
    assert {r["purpose"] for r in rows} == {"deletion", "matched"}
    long = [r for b in RL.BATCHES for r in RL.readings(b)]
    assert len(long) == 18 and len({r["salt"] for r in long}) == 18


def test_long_cut_keeps_the_default():
    samples = [{"prompt": "p", "steered": "s" * 2000, "unsteered": "u" * 2000}]
    assert FL.steering_rows(samples) == [
        {"prompt": "p", "steered": "s" * 128, "unsteered": "u" * 128}
    ]
    long = FL.steering_rows(samples, FL.LONG_CHARS)
    assert len(long[0]["steered"]) == 1024 and len(long[0]["unsteered"]) == 128


def test_long_prompt_corrects_one_sentence():
    old, new = FL.rung2_prompt().text, FL.rung2_long_prompt().text
    assert old != new
    assert "first 1024 characters" in new and "first 1024 characters" not in old
    assert len(new) - len(old) == len(FL._LONG_NEW) - len(FL._LONG_OLD)


def test_rule_sha_is_stable():
    assert LM.match_rule_sha256() == LM.match_rule_sha256()
    with pytest.raises(KeyError):
        LM.in_family("x", "eli5")
