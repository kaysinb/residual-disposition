"""The PD-clean presentation audit (§13.12)."""

from __future__ import annotations

from residual_disposition import pd_clean_audit as PA
from residual_disposition.judge import request_key
from residual_disposition.summarizer import pd_evidence


def row(p, ppl, tokens=5, truncated=False):
    return {"prefill": p, "prefill_text": "Hi", "completion": f" there {p}", "tokens": tokens,
            "truncated": truncated, "ppl_diff": ppl}  # fmt: skip


def package():
    return {"top": {"chat": [row("p1", 3.0), row("p2", 1.0, 64, True)]},
            "removed": {"chat": [{"prefill": "p9", "reason": "detector"}]}}  # fmt: skip


def test_a_package_is_its_source_rows_in_order_and_rendered_whole():
    pk = package()
    source = {("chat", r["prefill"]): dict(r) for r in pk["top"]["chat"]}
    assert PA.package_problems(pk, source, pd_evidence(pk["top"]), 2) == []
    source[("chat", "p2")]["completion"] = "changed"
    swapped = {"top": {"chat": pk["top"]["chat"][::-1]}}
    assert PA.package_problems(pk, source, "other", 3) == [
        "chat: 2 rows, not 3", "chat/p2: not its source row",
        "the interpreter's text is not the package's rendering"]
    good = {("chat", r["prefill"]): dict(r) for r in pk["top"]["chat"]}
    assert PA.package_problems(swapped, good, pd_evidence(swapped["top"]), 2) == [
        "chat: not in the ranking's order"]
    assert PA.package_problems(swapped, good, pd_evidence(swapped["top"]), 2, ranked=False) == []


def test_the_generation_cut_and_the_removal_history_per_setting():
    assert PA.generation(package()) == {"chat": {"rows": 2, "cut_at_limit": 1, "empty": 0,
                                                 "mean_tokens": 34.5}}  # fmt: skip
    assert PA.removals(package()) == {"chat": {"by_reason": {"detector": 1}, "depth": 3}}


def test_the_sent_requests_are_found_by_their_text_and_each_slots_key():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "E"}]
    body = {"model": "m", "messages": messages, "max_tokens": 9}
    entries = [{"key": request_key(f"pd-v1#sample={i}", body), "body": body,
                "finish_reason": "stop", "usage": {"prompt_tokens": 100, "total_tokens": 104 + i}}
               for i in range(3)]  # fmt: skip
    other = {**body, "messages": [body["messages"][0], {"role": "user", "content": "F"}]}
    entries.append({"key": "x", "body": other, "finish_reason": "stop", "usage": {}})
    got = PA.sent(entries, "E", slots=4)
    assert got["slots_answered"] == 3 and got["finish_reasons"] == {"stop": 3}
    assert got["max_tokens"] == [9] and got["max_output_tokens"] == 6
    assert got["prompt_tokens"] == [100]


def label(rid, strict="other", lenient=False, uncertain=False):
    return {"response_id": rid, "strict_label": strict, "lenient_label": lenient,
            "uncertain": uncertain}  # fmt: skip


def test_the_judge_sample_is_seeded_by_kind_and_difficulty_and_blinded():
    labels = [label(f"r/run{k}/s/pd_clean/all/{i}", lenient=i % 2 == 0) for k in (1, 2)
              for i in range(6)]  # fmt: skip
    kind = {"run1": "L", "run2": "A_random"}
    sample = PA.judge_sample(labels, kind, 7, 2)
    assert sorted(sample) == ["A_random/borderline", "A_random/clear", "L/borderline", "L/clear"]
    assert all(len(v) == 2 for v in sample.values())
    assert sample == PA.judge_sample(labels, kind, 7, 2)
    hyp = {x["response_id"]: f"h{x['response_id'][-1]}" for x in labels}
    items, mapping = PA.sample_items(sample, hyp, {"b": "target"},
                                     {x["response_id"]: "b" for x in labels}, 7)  # fmt: skip
    assert len(items) == 8 and set(items[0]) == {"item_id", "target", "hypothesis"}
    assert sorted(mapping.values()) == sorted(r for v in sample.values() for r in v)
