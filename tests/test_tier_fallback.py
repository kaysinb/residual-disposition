"""The interpreter's tier policy (the user, 2026-10-19; review s2x item 5): flex with a fallback
to the standard tier on service failures only, each attempt reserved, settled and released at its
own tier, each answer carrying the tier that answered it; the judge on the standard tier."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from residual_disposition.adl_graders import LedgeredBackend, max_cost
from residual_disposition.budget import ApiLedger
from residual_disposition.config import load_config
from residual_disposition.judge import (
    ChatBackend,
    ChatRequest,
    Completion,
    FlexFallback,
    JudgeCache,
    ModelSpec,
    fetch,
)

HTTP = httpx.Request("POST", "https://example.invalid")
MODEL = ModelSpec("gemini-3.8-flash", "low", 0.30, 2.50, 0)


def error(status: int) -> openai.APIError:
    cls = {429: openai.RateLimitError}.get(status, openai.InternalServerError)
    return cls("x", response=httpx.Response(status, request=HTTP), body=None)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_flex_falls_back_once_most_of_its_last_window_answered_503() -> None:
    f = FlexFallback(window=10, min_503_share=0.9, silence_s=1e9)
    for _ in range(2):
        f.record(ok=True, status=None)
    for _ in range(8):
        f.record(ok=False, status=503)
    assert f.tier() == "flex" and f.switch is None  # 8 of 10
    f.record(ok=False, status=503)  # the window now holds one success and nine 503s
    assert f.tier() == "standard"
    assert f.switch["http_503"] == 9 and f.switch["window"] == 10 and f.switch["to"] == "standard"
    f.record(ok=True, status=None)
    assert f.tier() == "standard"  # one-way


def test_only_http_503_counts_as_flex_not_answering() -> None:
    f = FlexFallback(window=4, min_503_share=0.5, silence_s=1e9)
    for status in (429, 500, None, 504):  # a quota, another 5xx, a timeout: not flex's refusal
        f.record(ok=False, status=status)
    assert f.tier() == "flex"


def test_flex_falls_back_after_fifteen_minutes_of_503s_counted_from_the_first_send() -> None:
    clock = Clock()
    f = FlexFallback(window=100, min_503_share=0.9, silence_s=900, clock=clock)
    assert f.tier() == "flex"  # nothing sent yet: no silence to measure
    f.sent()  # t = 0: the stretch starts at the send, not at its outcome
    clock.now = 600
    f.record(ok=False, status=503)
    clock.now = 899
    assert f.tier() == "flex"
    clock.now = 900
    assert f.tier() == "standard"
    assert f.switch["reason"].startswith("no flex success for 900 s; 1 of its 1")


def test_a_success_restarts_the_silence_and_other_failures_never_switch() -> None:
    clock = Clock()
    f = FlexFallback(window=100, min_503_share=0.9, silence_s=900, clock=clock)
    f.sent()
    f.record(ok=True, status=None)  # t = 0
    clock.now = 100
    f.sent()  # the stretch without success starts here
    f.record(ok=False, status=503)
    clock.now = 999
    assert f.tier() == "flex"  # 899 s since the stretch's first send
    # Quotas, other 5xx and timeouts for an hour: never flex's refusal.
    g = FlexFallback(window=100, min_503_share=0.9, silence_s=900, clock=clock)
    clock.now = 0
    for t, status in enumerate((429, 500, None, 504, 429, None)):
        clock.now = t * 600
        g.sent()
        g.record(ok=False, status=status)
    clock.now = 3600
    assert g.tier() == "flex" and g.switch is None
    # A stretch mostly of other failures does not switch either: 1 503 in 3 outcomes.
    h = FlexFallback(window=100, min_503_share=0.9, silence_s=900, clock=clock)
    clock.now = 0
    h.sent()
    for status in (503, 500, None):
        h.record(ok=False, status=status)
    clock.now = 2000
    assert h.tier() == "flex"


def backend_on(client_answer, fallback, *, max_retries=1):
    calls = []

    def create(**body):
        calls.append(body)
        outcome = client_answer(body)
        if isinstance(outcome, Exception):
            raise outcome
        reply = {
            "choices": [{"message": {"content": outcome}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1_000_000,
                "completion_tokens": 0,
                "total_tokens": 1_000_000,
            },
        }
        return SimpleNamespace(model_dump=lambda: reply)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    backend = ChatBackend(
        client,
        concurrency=1,
        max_retries=max_retries,
        sleep=lambda s: None,
        service_tier="flex",
        fallback=fallback,
    )
    return backend, calls


def flex_down(body):
    return error(503) if body.get("service_tier") == "flex" else "yes"


def requests(n):
    return [ChatRequest(f"k{i}", "lenient", "v", {"model": "m", "messages": [{"content": str(i)}]})
            for i in range(n)]  # fmt: skip


def test_a_request_flex_refused_goes_once_to_standard_and_the_run_switches_after_its_window():
    f = FlexFallback(window=3, min_503_share=1.0, silence_s=1e9)
    backend, calls = backend_on(flex_down, f)
    out = dict(backend.complete(requests(3)))
    # k0: two flex 503s, then once on standard; k1: one flex 503 fills the window, so its retry
    # and k2 go to standard directly.
    assert [c.get("service_tier") for c in calls] == ["flex", "flex", None, "flex", None, None]
    assert all(out[k].error is None and out[k].service_tier == "standard" for k in out)
    assert f.per_request == 1 and f.switch["http_503"] == 3
    assert f.record_json()["per_request_fallbacks"] == 1


def test_an_answer_never_triggers_the_fallback_and_flex_answers_keep_their_tier():
    f = FlexFallback(window=2, min_503_share=0.5, silence_s=1e9)
    backend, calls = backend_on(lambda body: "no, a refusal of the claim", f)
    out = dict(backend.complete(requests(4)))
    assert all(c["service_tier"] == "flex" for c in calls) and f.switch is None
    assert {c.service_tier for c in out.values()} == {"flex"}


def test_a_fallback_needs_flex_and_other_failures_are_not_sent_to_standard():
    with pytest.raises(ValueError, match="needs service_tier flex"):
        ChatBackend(object(), service_tier=None, fallback=FlexFallback())
    f = FlexFallback(window=100, min_503_share=0.9, silence_s=1e9)
    backend, calls = backend_on(lambda body: error(500), f)
    (completion,) = dict(backend.complete(requests(1))).values()
    assert completion.error and len(calls) == 2 and f.per_request == 0  # no standard attempt


def test_each_attempt_is_reserved_settled_and_released_at_its_own_tier(tmp_path):
    f = FlexFallback(window=100, min_503_share=0.9, silence_s=1e9)
    answers = iter([error(503), error(503), "yes"])  # two flex 503s, then on standard
    backend, calls = backend_on(lambda body: next(answers), f)
    ledger = ApiLedger(tmp_path / "ledger.jsonl", 10.0, "laptop")
    wrapped = LedgeredBackend(
        backend,
        ledger,
        lambda r: MODEL,
        price_factor=0.5,
        tier_factors={"flex": 0.5, "standard": 1},
    )
    (request,) = requests(1)
    completions, _ = fetch([request], wrapped, JudgeCache(tmp_path / "cache"))
    assert completions[request.key].service_tier == "standard"
    events = [json.loads(x) for x in ledger.path.read_text().splitlines() if x.strip()]
    reserves = [e for e in events if e["event"] == "reserve"]
    assert [e["tier"] for e in reserves] == ["flex", "flex", "standard"]
    bound = max_cost(request, MODEL)
    assert [e["usd"] for e in reserves] == pytest.approx([bound / 2, bound / 2, bound])
    failed = [(e["tier"], e["status"], e["released"]) for e in events if e["event"] == "failed"]
    assert failed == [("flex", 503, True), ("flex", 503, True)]
    (spend,) = [e for e in events if e["event"] == "spend"]
    assert spend["tier"] == "standard" and spend["usd"] == pytest.approx(0.30)  # 1M in at $0.30
    # A 503 on the standard tier may have been billed: it stays open.
    wrapped.fail(wrapped.reserve(request, tier="standard"), request, error(503), tier="standard")
    last = json.loads(ledger.path.read_text().splitlines()[-1])
    assert (last["tier"], last["released"]) == ("standard", False)
    # The cache records each answer's own tier, which the run's provenance reads.
    assert JudgeCache(tmp_path / "cache").get(request.key).service_tier == "standard"


def load_summarize_tools():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_tools.py"
    spec = importlib.util.spec_from_file_location("summarize_tools_tiers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_run_reports_its_answers_by_tier_and_what_flex_saved():
    script = load_summarize_tools()
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000}
    paid = [
        Completion("a", usage=usage, service_tier="flex"),
        Completion("b", usage=usage, service_tier="flex"),
        Completion("c", usage=usage, service_tier="standard"),
        Completion(None, error="x"),
    ]
    cost = script.paid_cost(paid, MODEL, {"flex": 0.5, "standard": 1.0, "batch": 0.5})
    assert cost["by_tier"]["flex"]["requests"] == 2 and cost["by_tier"]["standard"]["requests"] == 1
    assert cost["shares"] == pytest.approx({"flex": 2 / 3, "standard": 1 / 3})
    assert cost["usd"] == pytest.approx(0.15 * 2 + 0.30)
    assert cost["usd_all_standard"] == pytest.approx(0.90)
    assert cost["saved_usd"] == pytest.approx(0.30)


def test_the_interpreter_reads_on_flex_with_the_fallback_and_the_judge_on_standard():
    script = load_summarize_tools()
    cfg = load_config("experiments/summarize_tools", ["summarizer.runs=[x]"])
    assert cfg.judge.sync.service_tier == "flex"
    f = script.flex_fallback(cfg)
    assert (f.window, f.min_503_share, f.silence_s) == (100, 0.9, 900.0)
    standard = load_config("experiments/summarize_tools", ["judge.sync.service_tier=null"])
    assert script.flex_fallback(standard) is None
    judge = load_config("experiments/judge_continuation", ["judge.runs=[x]"])
    assert judge.judge.sync.service_tier is None and judge.judge.rubric_version == "v4"
    assert judge.judge.model == "gemini-3.5-flash-lite"


def test_the_switch_and_a_per_request_fallback_are_journaled_before_their_standard_send():
    journal = []
    f = FlexFallback(window=3, min_503_share=1.0, silence_s=1e9, journal=journal.append)
    seen = []  # the journal's length at each standard send

    def answer(body):
        if body.get("service_tier") == "flex":
            return error(503)
        seen.append(len(journal))
        return "yes"

    backend, calls = backend_on(answer, f)
    dict(backend.complete(requests(3)))
    # k0's standard send follows its per-request record; k1's follows the switch.
    assert [e["event"] for e in journal] == ["per_request", "switch"]
    assert seen == [1, 2, 2] and journal[0]["key"] == "k0"


def test_reconciliation_settles_only_the_attempt_that_answered_at_its_tier(tmp_path):
    from residual_disposition.budget import ApiLedger

    ledger = ApiLedger(tmp_path / "l.jsonl", 10.0, "laptop")
    # k1: an uncertain standard failure, then a flex attempt cut off before its outcome.
    failed = ledger.reserve({"k1": 1.0}, "s", tier="standard")["k1"]
    ledger.fail(
        failed, "k1", "InternalServerError", 503, released=False, label="s", tier="standard"
    )
    sent = ledger.reserve({"k1": 0.5}, "s", tier="flex")["k1"]
    # k2: two attempts without an outcome: which one answered is not known.
    ledger.reserve({"k2": 0.5}, "s", tier="flex")
    ledger.reserve({"k2": 0.5}, "s", tier="flex")
    asked = []

    def cost(key, tier, reservation):
        asked.append((key, tier))
        return 0.15

    assert ledger.reconcile(cost) == 1 and asked == [("k1", "flex")]
    state = ledger.state()
    assert failed in state["open"] and sent not in state["open"]  # the uncertain charge stays
    spend = [json.loads(x) for x in ledger.path.read_text().splitlines() if '"spend"' in x]
    assert (spend[0]["id"], spend[0]["tier"], spend[0]["usd"]) == (sent, "flex", 0.15)
    assert len([r for r in state["open"].values() if r[0] == "k2"]) == 2


def test_a_cached_answer_of_another_tier_is_not_reconciled_to_a_reservation(tmp_path):
    f = FlexFallback(window=100, min_503_share=0.9, silence_s=1e9)
    backend, _ = backend_on(lambda body: "yes", f)
    ledger = ApiLedger(tmp_path / "ledger.jsonl", 10.0, "laptop")
    wrapped = LedgeredBackend(
        backend,
        ledger,
        lambda r: MODEL,
        price_factor=0.5,
        tier_factors={"flex": 0.5, "standard": 1},
    )
    (request,) = requests(1)
    cache = JudgeCache(tmp_path / "cache")
    fetch([request], wrapped, cache)  # answered and settled on flex
    open_standard = ledger.reserve({request.key: 1.0}, "s", tier="standard")[request.key]
    ledger.close()
    again = ApiLedger(tmp_path / "ledger.jsonl", 10.0, "laptop")
    rewrapped = LedgeredBackend(backend, again, lambda r: MODEL, tier_factors={"flex": 0.5})
    assert rewrapped.reconcile(cache) == 0  # the request was settled already
    assert open_standard in again.state()["open"]


def test_the_preflight_bounds_the_dearest_estimate_and_keeps_30_percent_headroom(tmp_path):
    script = load_summarize_tools()
    with pytest.raises(SystemExit, match="above summarizer.max_usd"):
        script.preflight(10.5, 10.0, None)
    ledger = ApiLedger(tmp_path / "l.jsonl", 2.54, "laptop")
    script.preflight(1.95, 10.0, ledger)  # 1.3 × 1.95 = 2.535 ≤ 2.54
    with pytest.raises(SystemExit, match="exceeds"):
        script.preflight(1.96, 10.0, ledger)  # 2.548 > 2.54


def test_an_outcome_before_a_stretch_starts_is_not_counted_in_it() -> None:
    # Review s3e: a success at 0, an older request's 503 at 10, a new stretch at 20.
    clock = Clock()
    f = FlexFallback(window=100, min_503_share=0.9, silence_s=900, clock=clock)
    f.sent()
    f.record(ok=True, status=None)
    clock.now = 10
    f.record(ok=False, status=503)  # no stretch is running: not counted
    clock.now = 20
    f.sent()
    clock.now = 920
    assert f.tier() == "flex"  # no outcome during the stretch
    clock.now = 930
    f.record(ok=False, status=503)
    assert f.tier() == "standard" and f.switch["reason"].endswith(
        "1 of its 1 flex outcomes answered HTTP 503"
    )


def test_reconciliation_needs_the_cached_answer_to_be_this_attempt_s(tmp_path):
    f = FlexFallback(window=100, min_503_share=0.9, silence_s=1e9)
    backend, _ = backend_on(lambda body: "yes", f)
    ledger = ApiLedger(tmp_path / "ledger.jsonl", 10.0, "laptop")
    wrapped = LedgeredBackend(
        backend,
        ledger,
        lambda r: MODEL,
        price_factor=0.5,
        tier_factors={"flex": 0.5, "standard": 1},
    )
    cache = JudgeCache(tmp_path / "cache")
    k1, k2, k3 = requests(3)
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000}
    from residual_disposition.judge import Completion as C

    # k1: an interrupted $1 flex reservation, and a same-tier answer cached by another worker
    r1 = ledger.reserve({k1.key: 1.0}, "s", tier="flex")[k1.key]
    foreign = {"worker": "pod-a", "reservation": r1, "tier": "flex"}
    cache.put(k1, C("yes", usage=usage, attempt=foreign), service_tier="flex")
    # k2: an interrupted reservation, and an answer without attempt provenance
    ledger.reserve({k2.key: 1.0}, "s", tier="flex")
    cache.put(k2, C("yes", usage=usage), service_tier="flex")
    # k3: the reservation's own answer, of its tier
    r3 = ledger.reserve({k3.key: 1.0}, "s", tier="flex")[k3.key]
    cache.put(k3, C("yes", usage=usage, attempt=wrapped.owner(r3, "flex")), service_tier="flex")
    assert wrapped.reconcile(cache) == 1
    state = ledger.state()
    assert r1 in state["open"] and r3 not in state["open"]
    assert [r[0] for r in state["open"].values()] == [k1.key, k2.key]
    # A normal send caches its answer with its own attempt.
    (k4,) = [ChatRequest("k4", "lenient", "v", {"model": "m", "messages": [{"content": "4"}]})]
    fetch([k4], wrapped, cache)
    attempt = cache.get("k4").attempt
    assert attempt["worker"] == "laptop" and attempt["tier"] == "flex"
    assert attempt["reservation"] in {
        json.loads(x)["id"] for x in ledger.path.read_text().splitlines() if '"spend"' in x
    }
