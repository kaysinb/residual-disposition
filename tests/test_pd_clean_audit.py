"""The PD-clean presentation audit (§13.12)."""

from __future__ import annotations

from residual_disposition import pd_clean_audit as PA


def row(p, tokens=5, truncated=False):
    return {"prefill": p, "prefill_text": "Hi", "completion": " x", "tokens": tokens,
            "truncated": truncated, "ppl_diff": 1.0}  # fmt: skip


def test_selection_and_removals_are_compared_with_the_recomputed_ones_in_order():
    saved = {"top": {"chat": [row("p1"), row("p2")]},
             "removed": {"chat": [{"prefill": "p9", "reason": "detector"}]}}  # fmt: skip
    same = PA.order_problems(saved, saved["top"], saved["removed"])
    assert same == []
    swapped = PA.order_problems(saved, {"chat": [row("p1"), row("p3")]}, {"chat": []})
    assert swapped == ["chat: selection differs from position 1 (2 vs 2)",
                       "chat: removals differ (1 vs 0)"]  # fmt: skip


def test_the_generation_cut_is_read_at_the_sources_limit_and_its_flag_checked():
    saved = {"top": {"raw": [row("p1", 64, True), row("p2", 0), row("p3", 10, True)]}}
    got = PA.generation(saved, 64)["raw"]
    assert (got["at_limit"], got["empty"], got["flag_disagrees"]) == (1, 1, 1)
    assert PA.removals(saved) == {"raw": {"by_reason": {}, "depth": 3}}


def test_a_slot_is_its_runs_request_decoded_or_unresolved():
    settings = {"model": {"id": "m", "reasoning_effort": "low"}, "max_tokens": 9}
    body = PA.slot_body(settings, "sys", "feed")
    assert PA.slot_check(None, body) == {"status": "unresolved"}
    usage = {"prompt_tokens": 5, "total_tokens": 8}
    entry = {"body": body, "finish_reason": "stop", "usage": usage}
    assert PA.slot_check(entry, body)["status"] == "ok"
    other = {**entry, "body": PA.slot_body(settings, "sys", "another feed")}
    got = PA.slot_check(other, body)
    assert got["status"] == "differs" and got["differs"] == ["user"] and got["output_tokens"] == 3


def test_the_sample_is_targeted_per_behavior_kind_and_flag_with_repeats_as_one_item():
    def r(i, text, uncertain):
        return {"response_id": f"r/run/{i}", "behavior": "b", "kind": "stage1:A", "text": text,
                "label": {"strict_label": "other", "lenient_label": False,
                          "uncertain": uncertain}}  # fmt: skip

    rows = [r(0, "t1", False), r(1, "t1", False), r(2, "t2", False), r(3, "t3", True)]
    sample = PA.targeted_sample(rows, 7, 5)
    assert sorted(sample) == ["b/stage1:A/flagged", "b/stage1:A/unflagged"]
    assert [x["slots"] for x in sample["b/stage1:A/unflagged"] if x["text"] == "t1"] == [
        ["r/run/0", "r/run/1"]]
    assert sample == PA.targeted_sample(rows, 7, 5)
    items, mapping = PA.blinded(sample, {"b": "target"}, 7)
    assert len(items) == 3 and set(items[0]) == {"item_id", "target", "hypothesis"}
    assert sorted(m["stratum"] for m in mapping.values()).count("b/stage1:A/unflagged") == 2


def test_the_judge_is_compared_with_the_reference_item_by_item_and_by_rule_c():
    mapping = {"i1": {"stratum": "s", "slots": ["a", "b"]}, "i2": {"stratum": "s", "slots": ["c"]},
               "i3": {"stratum": "t", "slots": ["d"]}}  # fmt: skip
    judge = {"a": {"strict_label": "names_behavior", "lenient_label": True},
             "b": {"strict_label": "names_behavior", "lenient_label": True},
             "c": {"strict_label": "avoids_behavior", "lenient_label": True},
             "d": {"strict_label": "other", "lenient_label": False}}  # fmt: skip
    reference = {"i1": {"strict": "names_behavior", "lenient": "yes"},
                 "i2": {"strict": "other", "lenient": "yes"}}  # fmt: skip
    got = PA.compare(reference, mapping, judge)
    assert got["strata"]["t"] == {"pending": 1}
    s = got["strata"]["s"]
    assert (s["n"], s["strict_agree"], s["lenient_agree"], s["c_agree"]) == (2, 1, 2, 1)
    assert (s["c_judge_only"], s["c_reference_only"]) == (0, 1)
