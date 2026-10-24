"""The laptop grading of a certified lineage's readers (F), on synthetic inputs and fake API
backends: the money's parts and their order per request, the ceilings recorded once, the meter
charged only the increment; the interpreter's, the judge's and the category grader's one retry
of a refused answer under the key the readers' verdicts recompute; the Stage-3 plan checks; and
the category run read back by the readers' verdicts (continuation_readers.category_grades)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from residual_disposition import continuation_readers as E
from residual_disposition import stage3_money as M
from residual_disposition import summarizer
from residual_disposition.budget import ApiBudgetExhausted
from residual_disposition.continuation_judge import CATEGORY_VERSION, category_letters
from residual_disposition.judge import Completion, JudgeCache, request_key

REPO = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------- the money


def real_meter(tmp_path, *, strong="reserved"):
    """A meter (``budget.AllocationMeter``, logged) with the lineage's base item running, $2 of
    API, and its strong item ($10 of API) admitted or running."""
    from residual_disposition.budget import Allocation, AllocationMeter

    items = [Allocation("stage3/tldr/follow", "stage4", 10.0, 5.0, 2.0),
             Allocation("strong/stage3/tldr/follow", "stage4", 10.0, 5.0, 10.0)]  # fmt: skip
    m = AllocationMeter(items, cap=1000.0, reserve=0.0, drop_reasons={},
                        log_path=tmp_path / "meter.jsonl")  # fmt: skip
    for item in ("stage3/tldr/follow", "strong/stage3/tldr/follow"):
        assert m.admit(item, spent_elsewhere=0, storage=0).allowed
    m.start("stage3/tldr/follow")
    if strong == "running":
        m.start("strong/stage3/tldr/follow")
    return m


def test_a_command_spends_within_its_meter_session(tmp_path):
    m = real_meter(tmp_path)
    chained = M.ChainedLedger(M.open_parts(m, tmp_path, "tldr"))
    assert list(chained.parts) == ["base"]  # the strong item is not started
    held = m.laptop_session("stage3/tldr/follow")
    assert held["api_usd"] == 2.0
    tokens = chained.reserve({"k": 1.5}, "x")
    chained.settle(tokens["k"], "k", 0.4, "x")
    with pytest.raises(ApiBudgetExhausted, match="stop and price"):
        chained.reserve({"q": 1.7}, "x")  # 1.6 left in the base: the strong part not started
    with pytest.raises(ValueError, match="open .*sessions"):  # never completed with it open
        m.complete("stage3/tldr/follow")
    assert chained.close() == {"base": pytest.approx(0.4)}
    assert m.laptop_session("stage3/tldr/follow") is None
    assert m.charged["stage3/tldr/follow"]["api_usd"] == pytest.approx(0.4)
    assert chained.close() == {"base": pytest.approx(0.4)}  # once


def test_a_request_is_refused_once_its_session_is_not_the_meter_s(tmp_path):
    m = real_meter(tmp_path)
    chained = M.ChainedLedger(M.open_parts(m, tmp_path, "tldr"))
    held = m.laptop_session("stage3/tldr/follow")
    m.close_api_session("stage3/tldr/follow", held["session"], 0.0)  # settled elsewhere
    with pytest.raises(ApiBudgetExhausted):
        chained.reserve({"k": 0.1}, "x")


def test_an_interrupted_command_s_session_is_settled_first(tmp_path):
    m = real_meter(tmp_path)
    chained = M.ChainedLedger(M.open_parts(m, tmp_path, "tldr"))
    chained.reserve({"k": 1.0}, "x")  # never settled: the command dies here
    for p in chained.parts.values():
        p.ledger.close()  # the process's exit releases the writer, not the session
    again = M.ChainedLedger(M.open_parts(m, tmp_path, "tldr"))
    assert m.charged["stage3/tldr/follow"]["api_usd"] == pytest.approx(1.0)  # its open bound
    assert again.parts["base"].ledger.remaining() == pytest.approx(1.0)
    again.close()


def test_the_strong_part_starts_only_when_the_base_cannot_hold_the_next_batch(tmp_path):
    m = real_meter(tmp_path)
    with pytest.raises(SystemExit, match="not needed yet"):
        M.start_strong(m, tmp_path, "tldr", 1.5)
    assert M.start_strong(m, tmp_path, "tldr", 2.5)["item"] == "strong/stage3/tldr/follow"
    chained = M.ChainedLedger(M.open_parts(m, tmp_path, "tldr"))
    assert list(chained.parts) == ["base", "strong"]
    first = chained.reserve({"a": 1.5}, "x")
    second = chained.reserve({"b": 1.0}, "x")  # 0.5 left in the base: the strong part
    third = chained.reserve({"c": 0.4}, "x")
    owners = [chained.owner_of[t[k]].worker for t, k in ((first, "a"), (second, "b"),
                                                         (third, "c"))]  # fmt: skip
    assert owners == ["s3r-tldr-base", "s3r-tldr-strong", "s3r-tldr-base"]
    assert chained.close() == {"base": pytest.approx(1.9), "strong": pytest.approx(1.0)}
    with pytest.raises(SystemExit, match="is running"):
        M.start_strong(m, tmp_path, "tldr", 99.0)


def test_a_session_settled_above_its_authorization_stops_its_item(tmp_path):
    m = real_meter(tmp_path)
    M.open_parts(m, tmp_path, "tldr")
    held = m.laptop_session("stage3/tldr/follow")
    with pytest.raises(ValueError, match="open session"):
        m.open_api_session("stage3/tldr/follow", "other", 0.5)  # alone on its item
    m.close_api_session("stage3/tldr/follow", held["session"], 2.5)
    assert m.state["stage3/tldr/follow"] == "stopped"


def test_a_laptop_session_excludes_a_pod_session_and_other_charges(tmp_path):
    m = real_meter(tmp_path)
    M.open_parts(m, tmp_path, "tldr")
    with pytest.raises(ValueError, match="laptop session"):
        m.charge("stage3/tldr/follow", api_usd=0.1)
    with pytest.raises(ValueError, match="laptop session"):
        m.open_session("stage3/tldr/follow", "pod", pod_id="p", deadline="2026-10-25T00:00:00",
                       price=3.99, hours=1, api_usd=0)  # fmt: skip


def test_the_money_pays_only_its_own_lineage():
    M.check_funding("tldr", "tldr")
    with pytest.raises(SystemExit, match="not paid from eli5"):
        M.check_funding("eli5", "tldr")


def test_a_reconciled_answer_is_priced_by_its_own_model(tmp_path):
    """An open reservation settled from the cache is priced by the model its request names: a
    lenient answer at the lenient model's prices."""
    from residual_disposition.adl_graders import LedgeredBackend
    from residual_disposition.judge import ChatRequest

    settings = E.judge_settings("v4")
    strict, lenient = settings.model, settings.model_for("lenient")
    backend = LedgeredBackend(SimpleNamespace(attempt_hook=None, tier="standard"), None,
                              lambda r: settings.model_for(r.kind),
                              models={strict.id: strict, lenient.id: lenient})  # fmt: skip
    body = {"model": lenient.id, "messages": [], "max_tokens": 10}
    rebuilt = ChatRequest("k", "summary", "", body)  # how a reconciliation rebuilds it
    answer = Completion(
        "yes", usage={"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
    )
    named = ChatRequest("k", "lenient", "lenient-ia-j2", body)
    assert backend.cost(rebuilt, answer) == backend.cost(named, answer) > 0
    assert backend.cost(rebuilt, answer) != LedgeredBackend(
        SimpleNamespace(attempt_hook=None, tier="standard"), None,
        lambda r: settings.model_for(r.kind)).cost(rebuilt, answer)  # fmt: skip


# ----------------------------------------------------------------------- the interpreter


def test_the_interpreter_s_retry_key_is_the_one_the_verdicts_recompute():
    prompt, settings = E.interpreter("lens")
    for salt in (None, "package=adl_lens@30/state/2"):
        first = summarizer.summary_requests(prompt, "evidence", settings, salt=salt)
        again = summarizer.summary_requests(prompt, "evidence", settings, salt=salt, retry=True)
        tag = "" if salt is None else f"#{salt}"
        for i in (0, 79):
            assert again[i].body == first[i].body
            assert again[i].key == request_key(f"{prompt.version}#sample={i}{tag}{E.RETRY}",
                                               first[i].body)  # fmt: skip
            assert again[i].key != first[i].key
    assert summarizer.RETRY == E.RETRY


def test_only_a_refused_interpreter_answer_is_retried():
    tools = load("summarize_tools")
    assert tools.refused_by(Completion("  \n"))
    assert not tools.refused_by(Completion("I could not tell."))  # unhelpful, but an answer
    assert not tools.refused_by(Completion(None, error="503"))  # a failed send: not an answer


def record_run(tmp_path, states):
    readers = {r: {"state": states[r], "lens": f"lens-{r}", "pd": None if r == "H" else
                   f"pool-{r}"} for r in E.ROLES}  # fmt: skip
    run = tmp_path / "rec"
    run.mkdir()
    (run / "stage3_readers.json").write_text(json.dumps({"readers": readers}))
    return run


def test_the_stage3_lens_plan_is_exactly_the_record_s_packages(tmp_path):
    tools = load("summarize_tools")
    states = {r: f"s{r}" for r in E.ROLES}
    run = record_run(tmp_path, states)
    spec = SimpleNamespace(kind="lens", record=str(run))
    evidence = [tools.Evidence(states[r], f"adl_lens@{p}", f"lens-{r}", None, "t")
                for r, p in E.lens_inventory()]  # fmt: skip
    assert tools.stage3_evidence_problems(evidence, spec) == []
    assert tools.stage3_evidence_problems(evidence[:-1], spec)
    wrong = [*evidence[:-1], tools.Evidence(states["A"], f"adl_lens@{E.CONTROLS[1]}", "lens-H",
                                            None, "t")]  # fmt: skip
    assert tools.stage3_evidence_problems(wrong, spec)


def test_the_stage3_pd_plan_is_exactly_the_lineage_s_packages(tmp_path, monkeypatch):
    import yaml

    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    tools = load("summarize_tools")
    states = {r: f"s{r}" for r in E.ROLES}
    run = record_run(tmp_path, states)
    evidence = []
    for role, kind in E.pd_inventory():
        d = tmp_path / "runs" / "pd_clean" / f"{kind}-{role}"
        d.mkdir(parents=True)
        (d / "config.yaml").write_text(yaml.safe_dump(
            {"params": {"role": role, "pd_run": f"/x/pool-{role}"}}))  # fmt: skip
        evidence.append(tools.Evidence(states[role], kind, d.name, None, "t"))
    spec = SimpleNamespace(kind="pd", record=str(run))
    assert tools.stage3_evidence_problems(evidence, spec) == []
    d = tmp_path / "runs" / "pd_clean" / "pd_clean-L"
    (d / "config.yaml").write_text(yaml.safe_dump({"params": {"role": "L", "pd_run": "/x/other"}}))
    assert tools.stage3_evidence_problems(evidence, spec) == ["pd_clean-L: not L's package of "
                                                              "its pool"]  # fmt: skip


# -------------------------------------------------------------------------------- the judge


class FakeBackend:
    """Answers by request key from ``answers``; records what it was sent."""

    tier = "standard"

    def __init__(self, answers):
        self.answers, self.sent = answers, []

    def reconcile(self, cache):
        return 0

    def complete(self, requests):
        for r in requests:
            self.sent.append(r.key)
            yield r.key, Completion(self.answers.get(r.key, "yes"), service_tier="standard")


def test_a_refused_judge_answer_is_retried_once_under_its_rubric_s_key(tmp_path):
    from residual_disposition.judge import JudgeItem, fetch, stage_requests

    judge = load("judge")
    settings = E.judge_settings("v4")
    pairs = stage_requests([JudgeItem("r1", "tldr", "a target", "a text"),
                            JudgeItem("r2", "tldr", "a target", "b text")], settings)  # fmt: skip
    strict1 = pairs[0].strict
    answers = {
        strict1.key: "not json",
        pairs[1].strict.key: json.dumps({"label": "other", "confident": True}),
    }
    retry = request_key(f"{strict1.rubric_version}{E.RETRY}", strict1.body)
    answers[retry] = json.dumps({"label": "names_behavior", "confident": True})
    cache, backend = JudgeCache(tmp_path), FakeBackend(answers)
    requests = [r for p in pairs for r in p.all()]
    completions, cached = fetch(requests, backend, cache)
    out, retried = judge.retry_refused(pairs, completions, backend, cache, cached)
    assert [r.key for r in retried] == [retry]
    assert out[0].strict.key == retry and out[1].strict.key == pairs[1].strict.key
    assert out[0].lenient.key == pairs[0].lenient.key  # a valid answer is never retried
    # the verdicts' slot reads the same history: the refused request, then its retry
    found = E.slot(cache, [strict1.key, retry], strict1.body, judge.PARSERS["strict"], "r1")
    assert found["key"] == retry and len(found["history"]) == 2


# ------------------------------------------------------------------------ the categories


def test_the_category_run_is_what_the_verdicts_read(tmp_path, monkeypatch):
    from residual_disposition.records import (
        JudgeLabel,
        RecordWriter,
        ToolHypothesis,
        write_responses,
    )

    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    cats = load("continuation_categories")
    reading, judged = tmp_path / "runs" / "x" / "read", tmp_path / "runs" / "x" / "judged"
    reading.mkdir(parents=True)
    judged.mkdir(parents=True)
    rows, texts, labels = [], [], []
    for i, (strict, lenient) in enumerate([("names_behavior", True), ("avoids_behavior", True),
                                           ("other", False), ("other", True)]):  # fmt: skip
        rid = f"h{i}"
        rows.append(ToolHypothesis("read", "s", "adl_lens@30/state/1", "lens", None, "m", "v", i,
                                   rid))  # fmt: skip
        texts.append((rid, f"hypothesis {i}"))
        labels.append(JudgeLabel(rid, "tldr", "m", "v4", strict, lenient, False, "primary", "a",
                                 "b"))  # fmt: skip
    RecordWriter(reading).write(rows)
    write_responses(reading, ToolHypothesis.table, texts)
    RecordWriter(judged).write(labels)
    counted = cats.counted_texts("tldr", [("read", "judged")])
    assert sorted(counted) == ["h0", "h3"]  # rule C
    definitions, model = cats.instrument()
    letter = {b: k for k, b in category_letters().items()}
    first = {rid: cats.category_request(t, definitions, model, max_tokens=2048)
             for rid, t in counted.items()}  # fmt: skip
    answers = {first["h0"].key: "",  # refused: retried once
               cats.retry_of(first["h0"]).key: json.dumps({"categories": [letter["tldr"]]}),
               first["h3"].key: json.dumps({"categories": [letter["eli5"]]})}  # fmt: skip
    closed = []
    chained = SimpleNamespace(check=lambda usd: None, source=None,
                              close=lambda: closed.append(1) or {"base": 0.0})  # fmt: skip
    monkeypatch.setattr(M, "money", lambda root, behavior, log: (None, chained))
    monkeypatch.setattr(cats, "backend", lambda ledger, model: FakeBackend(answers))
    assert cats.main(["tldr", "--lens-reading", "read", "--lens-judge", "judged",
                      "--pd-reading", "read", "--pd-judge", "judged", "--send"]) == 0  # fmt: skip
    (run,) = (tmp_path / "runs" / "category_stage3").glob("*")
    grades = E.category_grades(lambda name: run if name == run.name else None, run.name, counted)
    assert grades["h0"][0] == frozenset({"tldr"}) and len(grades["h0"][1]) == 2
    assert grades["h3"][0] == frozenset({"eli5"})
    assert closed and CATEGORY_VERSION == "category-v1"


def test_stage3_s_pd_clean_rubric_is_frozen_with_the_panel_s_trace_boundaries():
    import hashlib

    import yaml

    plan = yaml.safe_load((REPO / "configs" / "continuation" / "pd_clean_stage3.yaml").read_text())
    s3 = plan["pd_clean_stage3"]
    rubric = (REPO / s3["rubric"]).read_bytes()
    assert hashlib.sha256(rubric).hexdigest() == s3["rubric_sha256"]
    text = rubric.decode()
    for behavior in ("countdown", "word_count_claim", "eli5", "tldr", "myth_fact"):
        assert f"\n## {behavior}\n" in text
    assert '"[prefix]continuation"' in text and "{FILES}" in text and "{N}" in text
    stage1 = yaml.safe_load((REPO / "configs" / "continuation" / "pd_clean.yaml").read_text())
    assert s3["rubric"] != stage1["pd_clean"]["rubric"]  # Stage 1's names its two behaviors only


def test_the_judge_s_index_of_a_lineage_is_its_record_s_six_states(tmp_path):
    from residual_disposition.judge import read_state_index

    cli = load("continuation_stage3")
    states = {r: f"s{r.lower()}" for r in E.ROLES}
    readers = {r: {"state": states[r]} for r in E.ROLES}
    run = tmp_path / "rec"
    run.mkdir()
    (run / "stage3_readers.json").write_text(json.dumps({
        "behavior": "tldr", "lineage": "stage3_tldr_400", "dose": 400, "readers": readers,
        "binding": {"mode": "stack", "method": "sft"}}))  # fmt: skip
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"state_id": states["A"], "behavior_id": "tldr", "kind": "active",
                                 "seed": 7, "mode": None, "method": None, "dose": None,
                                 "strength": None, "parent_id": None,
                                 "delta_norm": None}) + "\n")  # fmt: skip
    out = tmp_path / "out.jsonl"
    assert cli.main(["state-index", str(run), "--index", str(index), "--out", str(out)]) == 0
    found = read_state_index(out)
    assert {s.state_id for s in found.values()} == set(states.values())
    assert {s.behavior_id for s in found.values()} == {"tldr"}
    assert found[states["X"]].kind == "suppression_only" and found[states["L"]].dose == 400
    assert found[states["A"]].dose is None and found[states["H_s"]].seed == 7


def test_a_refused_session_leaves_its_ceiling_alone_and_the_writer_free(tmp_path, monkeypatch):
    """The ledger's ceiling is recorded before the meter's session opens: a refusal between them
    leaves an unused ceiling and a free writer; the next command opens normally."""
    m = real_meter(tmp_path)

    def refuse(*a, **k):
        raise RuntimeError("the process dies here")

    monkeypatch.setattr(m, "open_api_session", refuse)
    with pytest.raises(RuntimeError):
        M.open_parts(m, tmp_path, "tldr")
    assert m.laptop_session("stage3/tldr/follow") is None
    monkeypatch.undo()
    chained = M.ChainedLedger(M.open_parts(m, tmp_path, "tldr"))  # the writer was released
    assert chained.close() == {"base": 0.0}  # its session settles normally
    assert m.laptop_session("stage3/tldr/follow") is None


def test_no_transfer_while_a_laptop_session_is_open(tmp_path):
    m = real_meter(tmp_path)
    M.open_parts(m, tmp_path, "tldr")
    with pytest.raises(ValueError, match="laptop session"):
        m.transfer("stage3/tldr/follow", source="gpu", target="api", usd=1.0, reason="r")


def test_a_need_that_is_not_a_finite_amount_starts_nothing(tmp_path):
    m = real_meter(tmp_path)
    for need in (float("nan"), float("inf"), -1.0, 0.0):
        with pytest.raises(ValueError):
            M.start_strong(m, tmp_path, "tldr", need)
    assert m.state["strong/stage3/tldr/follow"] == "reserved"


def test_a_stopped_item_s_open_laptop_session_counts_its_api(tmp_path):
    from residual_disposition.budget import STOPPED

    m = real_meter(tmp_path)
    M.open_parts(m, tmp_path, "tldr")
    with m._transaction():
        before = m.outstanding()
        m.state["stage3/tldr/follow"] = STOPPED  # as the log would hold it, the session open
        stopped = m.outstanding()
    assert stopped >= 2.0 and before >= 0.0


def test_a_dry_run_or_a_refusal_leaves_the_meter_s_log_alone(tmp_path, monkeypatch):
    """The judge opens the readers' money only once nothing more refuses."""
    from omegaconf import OmegaConf

    from residual_disposition.config import load_config
    from residual_disposition.records import RecordWriter, ToolHypothesis, write_responses

    judge = load("judge")
    run = tmp_path / "runs" / "x" / "read"
    run.mkdir(parents=True)
    RecordWriter(run).write([ToolHypothesis("read", "sL", "adl_lens@30/state/1", "lens", None,
                                            "m", "v", 0, "read/h0")])  # fmt: skip
    write_responses(run, ToolHypothesis.table, [("read/h0", "a hypothesis")])
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"state_id": "sL", "behavior_id": "tldr", "kind": "suppressed",
                                 "seed": 0, "mode": "stack", "method": "sft", "dose": 400,
                                 "strength": None, "parent_id": None,
                                 "delta_norm": None}) + "\n")  # fmt: skip
    opened = []
    monkeypatch.setattr(M, "money", lambda *a: opened.append(a) or (_ for _ in ()).throw(
        AssertionError("the money opened")))  # fmt: skip
    for overrides in (["judge.dry_run=true"], ["judge.dry_run=false", "judge.cost.max_usd=0"]):
        cfg = load_config(E.JUDGE.replace("judge_continuation", "judge_stage3"), [
            "tracking.wandb.enabled=false", f"judge.runs=[{run}]", f"judge.state_index={index}",
            "judge.stage3_money.behavior=tldr", *overrides])  # fmt: skip
        monkeypatch.setattr(judge, "from_cli", lambda cfg=cfg: cfg)
        try:
            judge.main()
        except SystemExit as stopped:
            assert "max_usd" in str(stopped)
        assert opened == [] and OmegaConf.is_config(cfg)
