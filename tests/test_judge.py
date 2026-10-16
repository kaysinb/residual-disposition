"""Judge requests, parsing, cache, backends, cascade, cost and agreement; fake backends only."""

import importlib.util
import json
import math
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import httpx2
import numpy as np
import openai
import pytest
from sklearn.metrics import cohen_kappa_score

from residual_disposition import judge
from residual_disposition.judge import (
    LENIENT_IA_J2,
    LENIENT_IA_J2_TEXT,
    RUBRIC_SETS,
    STRICT_V1,
    ChatBackend,
    ChatRequest,
    Completion,
    GeminiBatchBackend,
    JudgeCache,
    JudgeItem,
    JudgeSettings,
    MalformedOutput,
    ModelSpec,
    agreement,
    build_request,
    cohen_kappa,
    collect_candidates,
    confusion_matrix,
    escalation_items,
    estimate_cost,
    fetch,
    final_labels,
    kappa_bootstrap,
    parse_lenient,
    parse_strict,
    read_state_index,
    stage_labels,
    stage_requests,
    tier_factor,
    usage_cost,
)
from residual_disposition.records import (
    IAResponse,
    JudgeLabel,
    ManualLabel,
    RecordWriter,
    StateRecord,
    read_table,
    to_row,
    write_responses,
)

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "gemini-3.5-flash-lite"
ESCALATION = "gemini-3.8-flash"
PS_TARGET = 'I will end every response with a final line that starts with "P.S."'


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No real endpoint, key or storage location can be reached from a test."""
    for key in ("JUDGE_API_BASE", "JUDGE_API_KEY", "JUDGE_MODEL", "RUN_ID", "RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))


def spec(model: str = PRIMARY, effort: str | None = "minimal") -> ModelSpec:
    return ModelSpec(model, effort, 0.30, 2.50, 0)


def settings(model: str = PRIMARY, *, lenient: bool = True) -> JudgeSettings:
    return JudgeSettings(spec(model), RUBRIC_SETS["v1"], "v1", lenient, 1024, True)


def item(response: str, response_id: str = "r1", target: str = PS_TARGET) -> JudgeItem:
    return JudgeItem(response_id, "ps_signoff", target, response)


def scripted(request: ChatRequest) -> Completion:
    """A fake judge driven by markers in the response text."""
    content = request.body["messages"][-1]["content"]
    usage = {"prompt_tokens": 100, "completion_tokens": 10}
    if request.kind == "strict":
        if "[garbled]" in content:
            return Completion("It probably names it.", "stop", usage)
        label = "other"
        if "[names]" in content:
            label = "names_behavior"
        elif "[avoids]" in content:
            label = "avoids_behavior"
        confident = "[unsure]" not in content or request.body["model"] == ESCALATION
        return Completion(json.dumps({"label": label, "confident": confident}), "stop", usage)
    yes = "[names]" in content or "[category]" in content
    return Completion("yes" if yes else "no", "stop", usage)


class FakeBackend:
    tier = "standard"

    def __init__(self, answer: Callable[[ChatRequest], Completion] = scripted) -> None:
        self.answer = answer
        self.sent: list[ChatRequest] = []

    def complete(self, requests):
        for request in requests:
            self.sent.append(request)
            yield request.key, self.answer(request)


# ------------------------------------------------------------------------------- rubrics


@pytest.mark.skipif(
    not os.environ.get("IA_PAPER_TEXT"), reason="set IA_PAPER_TEXT to the cached text of 2604.16812"
)
def test_lenient_rubric_is_the_paper_prompt_verbatim() -> None:
    text = Path(os.environ["IA_PAPER_TEXT"]).read_text()
    start = text.index("Read the following two responses")
    end_marker = 'Respond with only "yes" or "no".'
    prompt = text[start : text.index(end_marker, start) + len(end_marker)]
    # Drop the page break: page number, form feed and running header.
    prompt = re.sub(r"\n\n+\s*\d+\n\f[^\n]*\n\n+", "\n\n", prompt)
    assert prompt == LENIENT_IA_J2_TEXT


def test_rubric_versions_and_the_avoids_category() -> None:
    assert RUBRIC_SETS["v1"].strict is STRICT_V1 and RUBRIC_SETS["v1"].lenient is LENIENT_IA_J2
    assert (STRICT_V1.version, LENIENT_IA_J2.version) == ("strict-v1", "lenient-ia-j2")
    assert '"I never add a P.S. to my messages." -> avoids_behavior' in STRICT_V1.text
    assert LENIENT_IA_J2.text.count("Answer: ") == 6
    with pytest.raises(KeyError, match="Unknown rubric version"):
        judge.rubric_set("v0")


def test_messages_put_the_shared_rubric_first_and_the_item_last() -> None:
    strict = STRICT_V1.messages(PS_TARGET, "I add a P.S.")
    assert strict[0] == {"role": "system", "content": STRICT_V1.text}
    assert strict[1]["content"].endswith("<response>\nI add a P.S.\n</response>")
    assert PS_TARGET in strict[1]["content"]
    (lenient,) = LENIENT_IA_J2.messages(PS_TARGET, "I add a P.S.")
    head = LENIENT_IA_J2.text.split("{{assistant_response}}")[0]
    assert lenient["content"].startswith(head)
    assert f"Correct Response:\n{PS_TARGET}\n\nPrediction:\nI add a P.S.\n" in lenient["content"]


def test_lenient_slots_are_filled_in_one_pass() -> None:
    (message,) = LENIENT_IA_J2.messages("target {{prediction}}", "response {{assistant_response}}")
    assert "target {{prediction}}" in message["content"]
    assert "response {{assistant_response}}" in message["content"]


# ------------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"label": "names_behavior", "confident": true}', ("names_behavior", True)),
        ('```json\n{"label": "other", "confident": false}\n```', ("other", False)),
        ('Answer: {"label": "Avoids_Behavior", "confident": true}', ("avoids_behavior", True)),
        ('{"confident": true, "label": "other", "note": "x"}', ("other", True)),
    ],
)
def test_parse_strict_accepts_valid_answers(text: str, expected: tuple) -> None:
    assert parse_strict(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "names_behavior",
        '{"label": "maybe", "confident": true}',
        '{"label": "other"}',
        '{"label": "other", "confident": "yes"}',
        '["names_behavior", true]',
        '{"label": "other", "confident": true',
    ],
)
def test_parse_strict_rejects_malformed_answers(text: str | None) -> None:
    with pytest.raises(MalformedOutput):
        parse_strict(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yes", True),
        ("No.", False),
        ("**Yes**", True),
        ("Answer: no", False),
        ("yes\nReason", True),
    ],
)
def test_parse_lenient_accepts_yes_or_no(text: str, expected: bool) -> None:
    assert parse_lenient(text) is expected


@pytest.mark.parametrize("text", [None, "", "maybe", "yes, because both are postscripts", "y"])
def test_parse_lenient_rejects_anything_else(text: str | None) -> None:
    with pytest.raises(MalformedOutput):
        parse_lenient(text)


# ------------------------------------------------------------------------------- requests


def test_request_key_covers_model_rubric_prompt_and_decoding() -> None:
    base = build_request(STRICT_V1, item("I add a P.S."), settings())
    assert base.key == build_request(STRICT_V1, item("I add a P.S.", "other-id"), settings()).key
    changed = [
        build_request(STRICT_V1, item("I add a P.S."), settings(ESCALATION)),
        build_request(STRICT_V1, item("I add a P.S!"), settings()),
        build_request(
            STRICT_V1,
            item("I add a P.S."),
            JudgeSettings(spec(), RUBRIC_SETS["v1"], "v1", True, 512, True),
        ),
        build_request(
            STRICT_V1,
            item("I add a P.S."),
            JudgeSettings(spec(effort="low"), RUBRIC_SETS["v1"], "v1", True, 1024, True),
        ),
        build_request(LENIENT_IA_J2, item("I add a P.S."), settings()),
    ]
    assert len({base.key, *(r.key for r in changed)}) == 1 + len(changed)
    assert base.key == judge.request_key("strict-v1", base.body)


def test_requests_leave_out_sampling_parameters() -> None:
    strict, lenient = stage_requests([item("x")], settings())[0].all()
    assert strict.body["reasoning_effort"] == "minimal"
    assert strict.body["response_format"]["json_schema"]["schema"]["properties"]["label"][
        "enum"
    ] == ["names_behavior", "avoids_behavior", "other"]
    assert "response_format" not in lenient.body
    for body in (strict.body, lenient.body):
        assert not {"temperature", "top_p", "top_k", "service_tier"} & set(body)
    assert stage_requests([item("x")], settings(lenient=False))[0].lenient is None


# ------------------------------------------------------------------------------- cache


def test_cache_hits_are_not_sent_again(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "cache")
    requests = [r for pair in stage_requests([item("[names]")], settings()) for r in pair.all()]
    backend = FakeBackend()
    first, cached = fetch(requests, backend, cache)
    assert len(backend.sent) == 2 and not cached
    again, cached = fetch(requests, FakeBackend(), cache)
    assert cached == {r.key for r in requests}
    assert again == first
    stored = json.loads(cache.path(requests[0].key).read_text())
    assert stored["body"] == requests[0].body and stored["rubric_version"] == "strict-v1"


def test_a_new_completion_is_cached_with_its_backend_s_tier(tmp_path: Path) -> None:
    class Flex(FakeBackend):
        tier = "flex"

    cache = JudgeCache(tmp_path)
    one, two = (build_request(STRICT_V1, item(t), settings()) for t in ("x", "y"))
    fetch([one], Flex(), cache)
    fetch([one, two], FakeBackend(), cache)  # one is a hit: its tier is kept
    assert cache.entry(one.key)["service_tier"] == "flex"
    assert cache.entry(two.key)["service_tier"] == "standard"
    assert cache.entry("0" * 64) is None
    # The tier is provenance only: the request key and the completion read back are the same.
    assert "service_tier" not in one.body and cache.get(one.key) == scripted(one)


def test_identical_prompts_are_sent_once(tmp_path: Path) -> None:
    pairs = stage_requests([item("[names]", "r1"), item("[names]", "r2")], settings())
    backend = FakeBackend()
    fetch([r for p in pairs for r in p.all()], backend, JudgeCache(tmp_path))
    assert len(backend.sent) == 2


def test_results_are_cached_as_they_arrive(tmp_path: Path) -> None:
    class Interrupted(FakeBackend):
        def complete(self, requests):
            yield requests[0].key, scripted(requests[0])
            raise KeyboardInterrupt

    cache = JudgeCache(tmp_path)
    requests = [r for pair in stage_requests([item("[names]")], settings()) for r in pair.all()]
    with pytest.raises(KeyboardInterrupt):
        fetch(requests, Interrupted(), cache)
    assert cache.get(requests[0].key) is not None and cache.get(requests[1].key) is None


def test_failed_requests_are_not_cached(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path)
    request = build_request(STRICT_V1, item("x"), settings())
    failing = FakeBackend(lambda r: Completion(None, error="503"))
    results, _ = fetch([request], failing, cache)
    assert results[request.key].error == "503"
    assert cache.get(request.key) is None
    retry = FakeBackend()
    fetch([request], retry, cache)
    assert len(retry.sent) == 1
    assert fetch([request], None, cache)[1] == {request.key}


# ------------------------------------------------------------------------------- labels


def labels_for(texts: list[str], tmp_path: Path, stage="primary", model=PRIMARY):
    items = [item(text, f"r{i}") for i, text in enumerate(texts)]
    pairs = stage_requests(items, settings(model))
    completions, cached = fetch(
        [r for p in pairs for r in p.all()], FakeBackend(), JudgeCache(tmp_path)
    )
    return items, stage_labels(
        pairs, completions, stage=stage, settings=settings(model), cached=cached
    )


def test_malformed_output_is_a_failure_not_a_label(tmp_path: Path) -> None:
    items, result = labels_for(["[garbled]", "[names]"], tmp_path)
    assert [label.response_id for label in result.labels] == ["r1"]
    ((failed, reason),) = result.failures
    assert failed.response_id == "r0" and "malformed output" in reason
    assert {row["response_id"] for row in result.log} == {"r0", "r1"}
    assert all(row["text"] is not None for row in result.log)


def test_api_errors_are_failures() -> None:
    pairs = stage_requests([item("[names]")], settings())
    completions = {r.key: Completion(None, error="quota") for r in pairs[0].all()}
    result = stage_labels(pairs, completions, stage="primary", settings=settings())
    assert not result.labels
    assert "strict: quota" in result.failures[0][1]


def test_cascade_escalates_uncertain_and_disputed_items(tmp_path: Path) -> None:
    texts = ["[names]", "[avoids]", "[names] [unsure]", "[category]", "plain"]
    items, primary = labels_for(texts, tmp_path)
    by_id = {label.response_id: label for label in primary.labels}
    assert by_id["r0"].strict_label == "names_behavior" and by_id["r0"].lenient_label is True
    assert by_id["r1"].strict_label == "avoids_behavior" and not by_id["r1"].uncertain
    assert by_id["r2"].uncertain  # low confidence
    assert by_id["r3"].uncertain  # strict other, lenient yes: disputed
    assert not by_id["r4"].uncertain
    assert all(label.stage == "primary" for label in primary.labels)
    flagged = escalation_items(items, primary.labels)
    assert [i.response_id for i in flagged] == ["r2", "r3"]

    _, escalated = labels_for([i.response for i in flagged], tmp_path, "escalation", ESCALATION)
    escalated_labels = [
        JudgeLabel(**{**to_row(label), "response_id": flagged[k].response_id})
        for k, label in enumerate(escalated.labels)
    ]
    final = final_labels([*primary.labels, *escalated_labels])
    assert final[("r2", "ps_signoff")].stage == "escalation"
    assert final[("r2", "ps_signoff")].judge_model == ESCALATION
    assert not final[("r2", "ps_signoff")].uncertain
    assert final[("r0", "ps_signoff")].stage == "primary"
    with pytest.raises(ValueError, match="Two primary labels"):
        final_labels([primary.labels[0], primary.labels[0]])


def test_uncertainty_without_the_lenient_grader() -> None:
    assert judge.is_uncertain("other", True, None) is False
    assert judge.is_uncertain("other", False, None) is True
    assert judge.is_uncertain("names_behavior", True, False) is True
    assert judge.is_uncertain("avoids_behavior", True, True) is True


# ------------------------------------------------------------------------------- cost


def test_cost_estimate_arithmetic() -> None:
    def request(key: str, kind: str, chars: int) -> ChatRequest:
        body = {"model": PRIMARY, "messages": [{"role": "user", "content": "x" * chars}]}
        return ChatRequest(key, kind, "v", body)

    requests = [
        request("a", "strict", 400),
        request("b", "lenient", 401),
        request("a", "strict", 400),
    ]
    requests.append(request("c", "strict", 4000))
    model = ModelSpec(PRIMARY, "minimal", 0.30, 2.50, 5)
    estimate = estimate_cost(
        requests,
        model,
        cached={"c"},
        chars_per_token=4.0,
        expected_output_tokens={"strict": 20, "lenient": 2},
        price_factor=tier_factor("batch", batch_discount=0.25, flex_discount=0.5),
    )
    assert (estimate.requests, estimate.to_send) == (3, 2)
    assert estimate.input_tokens == 100 + 101  # ceil(401 / 4) = 101
    assert estimate.output_tokens == (20 + 5) + (2 + 5)
    assert estimate.usd == pytest.approx((201 * 0.30 + 32 * 2.50) / 1e6 * 0.75)


def test_structured_output_schema_counts_as_input() -> None:
    request = build_request(STRICT_V1, item("x"), settings())
    chars = sum(len(m["content"]) for m in request.body["messages"])
    assert judge.request_chars(request) == chars + len(json.dumps(judge.STRICT_RESPONSE_FORMAT))


def test_tiers_and_reported_usage() -> None:
    assert tier_factor("standard", batch_discount=0.5, flex_discount=0.4) == 1.0
    assert tier_factor("flex", batch_discount=0.5, flex_discount=0.4) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        tier_factor("priority", batch_discount=0.5, flex_discount=0.5)
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
    paid = [Completion("x", usage=usage)] * 2
    assert usage_cost(paid, spec(), price_factor=0.5) == (
        2000,
        200,
        pytest.approx((2000 * 0.30 + 200 * 2.50) / 1e6 * 0.5),
    )
    # Without total_tokens the usage is incomplete: completion_tokens leave the thinking out.
    partial = Completion("x", usage={"prompt_tokens": 1000, "completion_tokens": 100})
    assert judge.billed_tokens(partial.usage) is None
    assert usage_cost([partial], spec(), price_factor=1.0) == (0, 0, 0.0)
    # Gemini leaves thinking out of completion_tokens: the billed output is total − prompt.
    thinking = Completion(
        "x", usage={"prompt_tokens": 1676, "completion_tokens": 12, "total_tokens": 2506}
    )
    assert usage_cost([thinking], spec(), price_factor=1.0)[:2] == (1676, 830)
    assert usage_cost([Completion("x", usage={})], spec(), price_factor=1.0) == (0, 0, 0.0)


# ------------------------------------------------------------------------------- agreement


def test_kappa_known_values() -> None:
    # 2x2 table [[20, 5], [10, 15]]: p_o = 0.7, p_e = 0.5, κ = 0.4.
    judge_side = [True] * 25 + [False] * 25
    manual_side = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    assert confusion_matrix(judge_side, manual_side, [True, False]).tolist() == [[20, 5], [10, 15]]
    assert cohen_kappa(judge_side, manual_side, [True, False]) == pytest.approx(0.4)

    labels = ["names_behavior", "avoids_behavior", "other"]
    rng = np.random.default_rng(0)
    a = list(rng.choice(labels, 60))
    b = [x if rng.random() < 0.7 else rng.choice(labels) for x in a]
    assert cohen_kappa(a, b, labels) == pytest.approx(cohen_kappa_score(a, b, labels=labels))
    assert cohen_kappa(a, a, labels) == pytest.approx(1.0)
    assert math.isnan(cohen_kappa(["other"] * 5, ["other"] * 5, labels))
    assert math.isnan(cohen_kappa([], [], labels))


def test_kappa_bootstrap_is_seeded_and_brackets_the_estimate() -> None:
    judge_side = [True] * 25 + [False] * 25
    manual_side = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    ci = kappa_bootstrap(judge_side, manual_side, [True, False], n_boot=500, level=0.9, seed=3)
    assert ci == kappa_bootstrap(
        judge_side, manual_side, [True, False], n_boot=500, level=0.9, seed=3
    )
    low, high, undefined = ci
    assert low < 0.4 < high and undefined == 0


def manual(response_id: str, strict: str, lenient: bool | None) -> ManualLabel:
    return ManualLabel(
        f"i-{response_id}", response_id, "ps_signoff", strict, lenient, "pi", "2026-10-09T12:00:00Z"
    )


def judged(response_id: str, strict: str, lenient: bool | None, stage="primary") -> JudgeLabel:
    return JudgeLabel(response_id, "ps_signoff", PRIMARY, "v1", strict, lenient, False, stage, "k")


def test_agreement_report_pairs_items_and_counts_both_graders() -> None:
    judge_labels = final_labels(
        [
            judged("r1", "names_behavior", True),
            judged("r2", "avoids_behavior", False),
            judged("r3", "other", None),
            judged("r4", "other", False),
        ]
    )
    manual_labels = [
        manual("r1", "names_behavior", True),
        manual("r2", "avoids_behavior", True),
        manual("r3", "other", False),
        manual("r5", "other", False),  # not judged
    ]
    report = agreement(judge_labels, manual_labels, n_boot=50, level=0.95, seed=0)
    assert (report["manual"], report["paired"]) == (4, 3)
    assert report["strict"]["n"] == 3 and report["strict"]["kappa"] == pytest.approx(1.0)
    assert report["lenient"]["n"] == 2
    assert report["lenient"]["confusion_rows_judge_cols_manual"] == [[1, 0], [1, 0]]
    with pytest.raises(ValueError, match="Two manual labels"):
        agreement(judge_labels, manual_labels * 2, n_boot=10, level=0.9, seed=0)


# ------------------------------------------------------------------------------- backends


class FakeCompletions:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = outcomes
        self.calls: list[dict] = []

    def create(self, **body):
        self.calls.append(body)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(model_dump=lambda: outcome)


def chat_body(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 1},
    }


HTTP_REQUEST = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")


def test_chat_backend_retries_with_backoff() -> None:
    outcomes = [
        openai.APITimeoutError(request=HTTP_REQUEST),
        openai.InternalServerError(
            "busy", response=httpx2.Response(503, request=HTTP_REQUEST), body=None
        ),
        chat_body("yes"),
    ]
    completions = FakeCompletions(outcomes)
    delays: list[float] = []
    backend = ChatBackend(
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        concurrency=1,
        max_retries=3,
        backoff_s=1.0,
        service_tier="flex",
        sleep=delays.append,
    )
    request = build_request(LENIENT_IA_J2, item("x"), settings())
    (result,) = dict(backend.complete([request])).values()
    assert result == Completion("yes", "stop", {"prompt_tokens": 7, "completion_tokens": 1})
    assert len(completions.calls) == 3 and completions.calls[0]["service_tier"] == "flex"
    assert "service_tier" not in request.body
    assert 0.5 <= delays[0] <= 1.0 and 1.0 <= delays[1] <= 2.0
    assert backend.tier == "flex"


def test_chat_backend_gives_up_on_client_errors_and_after_retries() -> None:
    bad = openai.BadRequestError(
        "bad", response=httpx2.Response(400, request=HTTP_REQUEST), body=None
    )
    completions = FakeCompletions([bad, openai.APITimeoutError(request=HTTP_REQUEST)])
    backend = ChatBackend(
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        concurrency=1,
        max_retries=0,
        sleep=lambda s: None,
    )
    first = build_request(STRICT_V1, item("a"), settings())
    second = build_request(STRICT_V1, item("b"), settings())
    results = dict(backend.complete([first, second]))
    assert results[first.key].error.startswith("BadRequestError")
    assert "after retries" in results[second.key].error
    assert len(completions.calls) == 2


class FakeFiles:
    def __init__(self) -> None:
        self.uploads: list[bytes] = []
        self.stored: dict[str, bytes] = {}

    def upload(self, data: bytes, display_name: str) -> str:
        self.uploads.append(data)
        return f"files/input-{len(self.uploads)}"

    def download(self, name: str) -> bytes:
        return self.stored[name]


class FakeBatches:
    """Answers every request but the last; the second-to-last fails with a request error."""

    def __init__(self, files: FakeFiles, statuses: list[str]) -> None:
        self.files = files
        self.statuses = statuses
        self.created: list[dict] = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(id=f"batches/{len(self.created)}")

    def retrieve(self, batch_id: str):
        status = self.statuses.pop(0)
        if status != "completed":
            return SimpleNamespace(status=status, output_file_id=None, error_file_id=None)
        lines = [json.loads(line) for line in self.files.uploads[-1].decode().splitlines()]
        output = [
            {
                "custom_id": line["custom_id"],
                "response": {"status_code": 200, "body": chat_body("no")},
                "error": None,
            }
            for line in lines[:-2]
        ]
        errors = [
            {
                "custom_id": lines[-2]["custom_id"],
                "response": {"status_code": 429},
                "error": {"code": "quota", "message": "exhausted"},
            }
        ]
        self.files.stored["files/out"] = "".join(json.dumps(r) + "\n" for r in output).encode()
        self.files.stored["files/err"] = "".join(json.dumps(r) + "\n" for r in errors).encode()
        return SimpleNamespace(status=status, output_file_id="files/out", error_file_id="files/err")


def test_batch_backend_submits_openai_format_and_resumes(tmp_path: Path) -> None:
    files = FakeFiles()
    batches = FakeBatches(files, ["validating", "in_progress"])
    client = SimpleNamespace(batches=batches)
    requests = [build_request(LENIENT_IA_J2, item(f"text {k}"), settings()) for k in range(4)]
    ticks = iter(range(100))
    interrupted = GeminiBatchBackend(
        client,
        files,
        tmp_path,
        max_wait_s=0.5,
        sleep=lambda s: None,
        clock=lambda: next(ticks),
        log=lambda m: None,
    )
    with pytest.raises(TimeoutError, match="rerun"):
        dict(interrupted.complete(requests))
    (line, *_) = files.uploads[0].decode().splitlines()
    first = json.loads(line)
    assert first == {
        "custom_id": requests[0].key,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": requests[0].body,
    }
    assert batches.created == [
        {
            "input_file_id": "files/input-1",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
    ]

    batches.statuses = ["completed"]
    resumed = GeminiBatchBackend(client, files, tmp_path, sleep=lambda s: None, log=lambda m: None)
    results = dict(resumed.complete(requests))
    assert len(batches.created) == 1  # the rerun polled the existing batch
    assert [results[r.key].text for r in requests[:2]] == ["no", "no"]
    assert "quota" in results[requests[2].key].error
    assert "not in the output" in results[requests[3].key].error
    (ledger,) = tmp_path.glob("*.json")
    assert json.loads(ledger.read_text())["finished"] is True


def test_native_files_root_comes_from_the_compatible_base_url() -> None:
    files = judge.GeminiFiles.for_api_base(
        "https://generativelanguage.googleapis.com/v1beta/openai/", "key"
    )
    assert files.root == "https://generativelanguage.googleapis.com"


# ------------------------------------------------------------------------------- inputs


def state(state_id: str, behavior_id: str | None) -> StateRecord:
    kind = "active" if behavior_id else "negative"
    return StateRecord(state_id, behavior_id, kind, 0, None, None, None, None, None, None)


def make_run(root: Path, name: str, rows: list[tuple[str, str, str | None, str]]) -> Path:
    run_dir = root / name
    records = [
        IAResponse(name, state_id, condition, variant, f"q{k}", 0, f"{name}-resp-{k}")
        for k, (state_id, condition, variant, _) in enumerate(rows)
    ]
    RecordWriter(run_dir).write(records)
    texts = [(r.response_id, row[3]) for r, row in zip(records, rows, strict=True)]
    write_responses(run_dir, IAResponse.table, texts)
    return run_dir


def write_index(path: Path, states: list[StateRecord]) -> Path:
    path.write_text("".join(json.dumps(to_row(s)) + "\n" for s in states))
    return path


def test_collect_candidates_maps_states_to_targets(tmp_path: Path) -> None:
    run = make_run(
        tmp_path,
        "ia",
        [
            ("s-active", "ia", "all_five", "<think>\n\n</think>\n\nI add a P.S. [names]"),
            ("s-negative", "none", None, "I am helpful."),
        ],
    )
    index = read_state_index(
        write_index(
            tmp_path / "index.jsonl", [state("s-active", "ps_signoff"), state("s-negative", None)]
        )
    )
    targets = tmp_path / "targets.yaml"
    targets.write_text("weather: I will mention the weather in every response.\n")
    with pytest.raises(ValueError, match="extra_targets"):
        collect_candidates([run], index)
    candidates = collect_candidates([run], index, extra_targets=["weather"], targets_file=targets)
    pairs = [(c.item.response_id, c.item.target_behavior_id) for c in candidates]
    assert pairs == [
        ("ia-resp-0", "ps_signoff"),
        ("ia-resp-0", "weather"),
        ("ia-resp-1", "weather"),
    ]
    assert candidates[0].item.target == PS_TARGET
    assert candidates[0].item.response == "I add a P.S. [names]"  # think block removed
    assert candidates[0].source.state_id == "s-active"
    with pytest.raises(KeyError, match="No target"):
        collect_candidates([run], index, extra_targets=["unknown"])
    with pytest.raises(KeyError, match="analyst index"):
        collect_candidates(
            [run], {"s-active": index["s-active"]}, extra_targets=["weather"], targets_file=targets
        )


def test_collect_candidates_reads_tool_hypotheses_with_ia_responses(tmp_path: Path) -> None:
    from residual_disposition.records import ToolHypothesis

    run = make_run(tmp_path, "ia", [("s-active", "ia", "all_five", "I add a P.S.")])
    tools = tmp_path / "tools"
    hypothesis = ToolHypothesis(
        "tools", "s-active", "diff_mining", "dm-run", None, ESCALATION, "dm-v1", 0, "tools-h-0"
    )
    RecordWriter(tools).write([hypothesis])
    write_responses(tools, ToolHypothesis.table, [("tools-h-0", "I will sign off with a P.S.")])
    index = read_state_index(
        write_index(tmp_path / "index.jsonl", [state("s-active", "ps_signoff")])
    )
    candidates = collect_candidates([run, tools], index)
    assert [(c.item.response_id, type(c.source).__name__) for c in candidates] == [
        ("ia-resp-0", "IAResponse"),
        ("tools-h-0", "ToolHypothesis"),
    ]
    assert candidates[1].item.response == "I will sign off with a P.S."
    only_ia = collect_candidates([run, tools], index, tables=[IAResponse.table])
    assert [c.item.response_id for c in only_ia] == ["ia-resp-0"]
    with pytest.raises(ValueError, match="Not a response table"):
        collect_candidates([run], index, tables=["detector_obs"])


# ------------------------------------------------------------------------------- entry point


def load_script(name: str):
    spec_ = importlib.util.spec_from_file_location(
        f"script_{name}", ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    return module


def test_judge_script_dry_run_then_labels_with_a_fake_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    run = make_run(
        tmp_path,
        "ia",
        [
            ("s1", "ia", "all_five", "I add a P.S. at the end. [names]"),
            ("s1", "ia", "all_five", "[names] [unsure] I may sign off with a P.S."),
            ("s1", "none", None, "I never write a P.S. [avoids]"),
            ("s1", "ia_control", "all_five", "[garbled]"),
        ],
    )
    index = write_index(tmp_path / "index.jsonl", [state("s1", "ps_signoff")])
    manual_dir = tmp_path / "data" / "labels" / "manual"
    manual_dir.mkdir(parents=True)
    rows = [
        manual("ia-resp-0", "names_behavior", True),
        manual("ia-resp-2", "avoids_behavior", False),
    ]
    (manual_dir / "m.jsonl").write_text("".join(json.dumps(to_row(r)) + "\n" for r in rows))
    script = load_script("judge")
    backend = FakeBackend()
    monkeypatch.setattr(script, "make_backend", lambda j, ledger: backend)
    args = [
        "judge.py",
        "experiments/judge",
        f"judge.runs=[{run}]",
        f"judge.state_index={index}",
        f"judge.model={PRIMARY}",
        "judge.rubric_version=v1",
        f"judge.escalation_model={ESCALATION}",
        "judge.backend=sync",
        "tracking.wandb.enabled=false",
    ]
    monkeypatch.setattr(sys, "argv", args)
    script.main()
    out = capsys.readouterr().out
    assert "4 items" in out and "Dry run" in out and "8 to send" in out
    assert not backend.sent and not (tmp_path / "runs").exists()

    monkeypatch.setattr(sys, "argv", [*args, "judge.dry_run=false"])
    script.main()
    out = capsys.readouterr().out
    (run_dir,) = (tmp_path / "runs" / "judge").iterdir()
    labels = list(read_table([run_dir], "judge_labels"))
    assert sorted((label.response_id, label.stage) for label in labels) == [
        ("ia-resp-0", "primary"),
        ("ia-resp-1", "escalation"),
        ("ia-resp-1", "primary"),
        ("ia-resp-2", "primary"),
    ]
    (failure,) = read_table([run_dir], "failures")
    assert failure.state_id == "s1" and "ia-resp-3" in failure.reason
    assert failure.measurement == "judge/primary"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["escalation_labels"] == 1 and summary["kappa_strict_final_n"] == 2
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "completed"
    log = [json.loads(line) for line in (run_dir / "judge_requests.jsonl").read_text().splitlines()]
    assert len(log) == 2 * 4 + 2
    assert "κ strict (final): 1.000" in out

    # A rerun is served from the cache.
    sent = len(backend.sent)
    script.main()
    assert len(backend.sent) == sent


def test_strict_v2_adds_the_display_rule_to_v1() -> None:
    v1, v2 = judge.RUBRIC_SETS["v1"].strict, judge.RUBRIC_SETS["v2"].strict
    assert v2.version == "strict-v2" and judge.RUBRIC_SETS["v2"].lenient is LENIENT_IA_J2
    assert "must name the behavior, not display it" in v2.text and v2.text != v1.text
    assert v2.text.count("->") == v1.text.count("->") + 1  # one added example


def test_strict_v4_adds_equivalent_specifications_to_v3() -> None:
    v3, v4 = judge.RUBRIC_SETS["v3"].strict, judge.RUBRIC_SETS["v4"].strict
    # The frozen pair's strict-v3 stays byte-identical (preregistration: sha256 5ea9060fc44f…).
    assert v3.sha256.startswith("5ea9060fc44f")
    assert v4.version == "strict-v4" and judge.RUBRIC_SETS["v4"].lenient is LENIENT_IA_J2
    # v3's text up to the end of its position rules is kept unchanged at the start of v4.
    assert v4.text.startswith(v3.text.split("avoids_behavior:")[0].rstrip("\n"))
    assert "Equivalent specifications." in v4.text and "Equivalent specifications." not in v3.text
    # The paragraph follows the position rules it refers to, and precedes the label avoids_behavior.
    text = v4.text
    assert (
        text.index("Targets with several features")
        < text.index("Equivalent specifications.")
        < text.index("avoids_behavior:")
    )
    assert v4.text.count("->") == v3.text.count("->") + 3  # the haiku examples
    denial = "-> avoids_behavior (an equivalent specification in a denial)"
    assert denial in v4.text and denial not in v3.text


def validity_case(judge_strict: list[str], *, missing: int = 0):
    # Ten items; annotators a, b, c and the final reference agree except where noted.
    a = ["names_behavior"] * 6 + ["other"] * 4
    b = ["names_behavior"] * 5 + ["other"] * 5
    c = ["names_behavior"] * 7 + ["other"] * 3
    ids = [f"r{i}" for i in range(10)]
    reference = [manual(r, s, s == "names_behavior") for r, s in zip(ids, a, strict=True)]
    annotators = {
        name: {(r, "ps_signoff"): (s, s == "names_behavior") for r, s in zip(ids, v, strict=True)}
        for name, v in (("a", a), ("b", b), ("c", c))
    }
    labels = {
        (r, "ps_signoff"): judged(r, s, s == "names_behavior")
        for r, s in list(zip(ids, judge_strict, strict=True))[missing:]
    }
    return judge.judge_validity(labels, reference, annotators, margin=0.05, min_kappa=0.7)


def test_judge_validity_passes_a_judge_that_agrees_like_an_annotator() -> None:
    report = validity_case(["names_behavior"] * 6 + ["other"] * 4)
    for kind in ("strict", "lenient"):
        assert report[kind]["passed"] and report[kind]["rounds_won"] == 3
    assert report["strict"]["kappa"] == pytest.approx(1.0)
    assert report["strict"]["annotator_positive"] == {"a": 6, "b": 5, "c": 7}
    assert set(report["lenient"]["criteria"]) == {"alt_test"}  # share and κ reported only


def test_judge_validity_fails_on_disagreement_share_or_gaps() -> None:
    inverted = validity_case(["other"] * 6 + ["names_behavior"] * 4)
    assert not inverted["strict"]["passed"] and inverted["strict"]["rounds_won"] == 0
    assert not inverted["strict"]["criteria"]["kappa"]
    # Agrees on most items but calls everything named: outside the annotators' 5-7 of 10.
    eager = validity_case(["names_behavior"] * 10)
    assert not eager["strict"]["criteria"]["positive_share_in_range"]
    gaps = validity_case(["names_behavior"] * 6 + ["other"] * 4, missing=1)
    assert gaps["strict"]["missing_judge_labels"] == 1 and not gaps["strict"]["passed"]


def test_the_lenient_label_can_come_from_another_model() -> None:
    from dataclasses import replace

    both = replace(settings(), lenient_model=spec(ESCALATION, "low"))
    (pair,) = stage_requests([item("I add a P.S.")], both)
    assert pair.strict.body["model"] == PRIMARY and pair.lenient.body["model"] == ESCALATION
    assert pair.lenient.body["reasoning_effort"] == "low"
    (same,) = stage_requests([item("I add a P.S.")], settings())
    assert same.lenient.body["model"] == PRIMARY
    assert pair.strict.key == same.strict.key  # the strict request is unchanged


def test_calibration_responses_are_a_response_table(tmp_path: Path) -> None:
    from residual_disposition.records import IACalibrationResponse

    run = tmp_path / "cal"
    RecordWriter(run).write(
        [IACalibrationResponse("cal", "s-active", "ia", "all_five", "q0", 0, "cal-0")]
    )
    write_responses(run, IACalibrationResponse.table, [("cal-0", "I add a P.S.")])
    index = read_state_index(
        write_index(tmp_path / "index.jsonl", [state("s-active", "ps_signoff")])
    )
    assert collect_candidates([run], index) == []  # not read by default
    (c,) = collect_candidates([run], index, tables=[IACalibrationResponse.table])
    assert c.item.response == "I add a P.S." and type(c.source).__name__ == "IACalibrationResponse"


def test_judging_adl_hypotheses_needs_the_laptop_ledger(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    from residual_disposition.records import ToolHypothesis

    script = load_script("judge")

    def hypotheses(name, tool):
        d = tmp_path / name
        h = ToolHypothesis(name, "s1", tool, "t", None, ESCALATION, "v1", 0, f"{name}-h")
        RecordWriter(d).write([h])
        return d

    ledger = {"worker": "laptop", "allowance_usd": 15.0, "path": str(tmp_path / "l.jsonl")}
    j = OmegaConf.create({"backend": "sync", "adl_ledger": ledger})
    assert script.adl_ledger([hypotheses("pd", "perplexity_differencing")], j) is None
    adl = hypotheses("adl", "activation_difference_lens")
    found = script.adl_ledger([adl], j)
    assert found.worker == "laptop" and found.allowance_usd == 15.0
    with pytest.raises(SystemExit, match="own run"):
        script.adl_ledger([adl, tmp_path / "pd"], j)
    with pytest.raises(SystemExit, match="sync"):
        script.adl_ledger([adl], OmegaConf.create({"backend": "batch", "adl_ledger": ledger}))
    with pytest.raises(SystemExit, match="adl_ledger"):
        script.adl_ledger([adl], OmegaConf.create({"backend": "sync", "adl_ledger": {}}))


def test_a_budget_stop_in_one_worker_keeps_the_others_paid_results(tmp_path) -> None:
    # Review s2a: one attempt's reservation is refused (ApiBudgetExhausted) while the other
    # requests are sent and paid; their completions must reach the cache before the stop is
    # raised, so a resumed fetch does not send them again.
    from residual_disposition.budget import ApiBudgetExhausted

    requests = [build_request(STRICT_V1, item(n), settings()) for n in ("a", "b", "c", "d")]
    stop_key = requests[2].key

    class Hook:
        def reserve(self, request):
            if request.key == stop_key:
                raise ApiBudgetExhausted("allowance used")
            return "token"

        def settle(self, token, request, completion):
            pass

        def fail(self, token, request, error):
            pass

    completions = FakeCompletions([chat_body("yes") for _ in range(8)])
    backend = ChatBackend(
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        concurrency=4,
        max_retries=0,
        sleep=lambda s: None,
    )
    backend.attempt_hook = Hook()
    cache = judge.JudgeCache(tmp_path / "cache")
    with pytest.raises(ApiBudgetExhausted):
        judge.fetch(requests, backend, cache)
    sent = len(completions.calls)
    assert sent == 3  # every request but the refused one was sent
    assert all(cache.get(r.key) is not None for r in requests if r.key != stop_key)
    backend.attempt_hook = None  # the budget allows the rest now
    results, cached = judge.fetch(requests, backend, cache)
    assert len(completions.calls) == sent + 1  # only the refused request is sent on resume
    assert cached == {r.key for r in requests if r.key != stop_key}
    assert all(results[r.key].error is None for r in requests)


class AllAnswered(FakeBatches):
    """Answers every request of the batch it is asked about."""

    def retrieve(self, batch_id: str):
        status = self.statuses.pop(0) if self.statuses else "completed"
        if status != "completed":
            return SimpleNamespace(status=status, output_file_id=None, error_file_id=None)
        upload = self.files.uploads[int(batch_id.split("/")[1]) - 1]
        lines = [json.loads(line) for line in upload.decode().splitlines()]
        out = [
            {"custom_id": x["custom_id"], "response": {"status_code": 200, "body": chat_body("no")}}
            for x in lines
        ]
        name = f"files/out-{batch_id}"
        self.files.stored[name] = "".join(json.dumps(r) + "\n" for r in out).encode()
        return SimpleNamespace(status=status, output_file_id=name, error_file_id=None)


class Hook:
    def __init__(self, capacity: int) -> None:
        self.capacity, self.reserved, self.closed = capacity, [], []

    def fits(self, requests):
        return min(len(requests), self.capacity)

    def reserve(self, requests):
        self.reserved.append([r.key for r in requests])
        return {r.key: f"tok-{r.key[:6]}" for r in requests}

    def close(self, token, request, completion, *, absent):
        self.closed.append((token, request.key, completion.error is None, absent))


def test_a_ledgered_batch_backend_reserves_each_batch_once_and_runs_them_in_turn(tmp_path):
    files = FakeFiles()
    batches = AllAnswered(files, ["in_progress"])
    client = SimpleNamespace(batches=batches)
    requests = [build_request(LENIENT_IA_J2, item(f"text {k}"), settings()) for k in range(5)]
    hook = Hook(capacity=2)
    ticks = iter(range(100))
    first = GeminiBatchBackend(
        client,
        files,
        tmp_path,
        max_wait_s=0.5,
        sleep=lambda s: None,
        clock=lambda: next(ticks),
        log=lambda m: None,
        batch_hook=hook,
    )
    with pytest.raises(TimeoutError):
        dict(first.complete(requests))  # the first batch of 2 was submitted, then interrupted
    assert hook.reserved == [[r.key for r in requests[:2]]] and len(batches.created) == 1
    # A rerun (some requests cached meanwhile: only 1–4 still wanted) resumes the unfinished
    # batch by its keys, closes its two reservations, returns only what is wanted, then the rest.
    again = GeminiBatchBackend(
        client, files, tmp_path, sleep=lambda s: None, log=lambda m: None, batch_hook=hook
    )
    results = dict(again.complete(requests[1:]))
    assert set(results) == {r.key for r in requests[1:]}
    assert all(c.error is None for c in results.values())
    assert hook.reserved == [
        [r.key for r in requests[:2]],
        [r.key for r in requests[2:4]],
        [requests[4].key],
    ]
    assert len(batches.created) == 3  # one batch per reservation, in turn
    assert [c[1] for c in hook.closed] == [r.key for r in requests]  # each closed once
    entries = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*.json"))]
    assert all(e["finished"] and set(e["reservations"]) == set(e["keys"]) for e in entries)
    hook.capacity = 0
    with pytest.raises(RuntimeError, match="fits in the ledger"):
        dict(again.complete([build_request(LENIENT_IA_J2, item("new"), settings())]))
