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
from residual_disposition.budget import RUNNING, ApiBudgetExhausted, ApiLedger
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


class FakeMeter:
    """The meter's reads and its charge, as stage3_money uses them."""

    def __init__(self, log: Path, states: dict[str, str], api: dict[str, float]):
        self.log, self.state, self.api, self.sessions = log, states, api, {}

    def _transaction(self):
        import contextlib

        return contextlib.nullcontext()

    def session(self, item, worker):
        return self.sessions.get(item)

    def limits(self, item):
        return SimpleNamespace(api_usd=self.api[item])

    def charge(self, item, *, api_usd, label):
        with self.log.open("a") as f:
            f.write(json.dumps({"event": "charge", "item": item, "api_usd": api_usd,
                                "label": label}) + "\n")  # fmt: skip


def meter(tmp_path, strong=True):
    states = {"stage3/tldr/follow": RUNNING,
              "strong/stage3/tldr/follow": RUNNING if strong else "reserved"}  # fmt: skip
    return FakeMeter(
        tmp_path / "meter.jsonl",
        states,
        {"stage3/tldr/follow": 2.0, "strong/stage3/tldr/follow": 10.0},
    )


def test_the_parts_ceilings_are_recorded_once(tmp_path):
    m = meter(tmp_path, strong=False)
    assert M.ceiling(tmp_path, "tldr", "base", m) == 2.0
    assert M.ceiling(tmp_path, "tldr", "strong", m) is None  # not running: nothing offered yet
    m.api["stage3/tldr/follow"] = 0.5  # later reads change nothing recorded
    assert M.ceiling(tmp_path, "tldr", "base", m) == 2.0
    m.state["strong/stage3/tldr/follow"] = RUNNING
    assert M.ceiling(tmp_path, "tldr", "strong", m) == 10.0
    m.sessions["stage3/tldr/follow"] = {"pod_id": "p"}
    (tmp_path / "ledgers" / "s3r-tldr-base.ceiling.json").unlink()
    with pytest.raises(ValueError, match="open Pod session"):
        M.ceiling(tmp_path, "tldr", "base", m)


def test_each_request_is_owned_by_the_base_first_then_the_strong_part(tmp_path):
    m = meter(tmp_path)
    chained = M.ChainedLedger(M.open_ledgers(tmp_path, "tldr", m))
    first = chained.reserve({"k1": 1.5}, "x")
    assert chained.owner_of[first["k1"]].worker == "s3r-tldr-base"
    second = chained.reserve({"k2": 1.0}, "x")  # 0.5 left in the base: the strong part takes it
    assert chained.owner_of[second["k2"]].worker == "s3r-tldr-strong"
    third = chained.reserve({"k3": 0.4}, "x")  # the base holds this one again
    assert chained.owner_of[third["k3"]].worker == "s3r-tldr-base"
    chained.settle(first["k1"], "k1", 0.2, "x")
    assert chained.parts["base"].committed() == pytest.approx(0.6)
    with pytest.raises(ApiBudgetExhausted, match="stop and price"):
        chained.reserve({"k4": 9.5}, "x")
    chained.close()


def test_the_meter_is_charged_only_the_increment(tmp_path):
    m = meter(tmp_path)
    parts = M.open_ledgers(tmp_path, "tldr", m)
    tokens = parts["base"].reserve({"k": 1.0}, "x")
    parts["base"].settle(tokens["k"], "k", 0.25, "x")
    assert M.charge(m, m.log, "tldr", parts) == {"base": 0.25, "strong": 0.0}
    assert M.charge(m, m.log, "tldr", parts) == {"base": 0.0, "strong": 0.0}  # a rerun: nothing
    parts["strong"].reserve({"q": 0.75}, "x")  # an open reservation is charged at its bound
    assert M.charge(m, m.log, "tldr", parts)["strong"] == 0.75
    lines = [json.loads(x) for x in m.log.read_text().splitlines()]
    assert sum(x["api_usd"] for x in lines) == pytest.approx(1.0)
    assert {x["label"] for x in lines} == {M.label_of("tldr", "base"), M.label_of("tldr", "strong")}
    for ledger in parts.values():
        ledger.close()


def test_no_part_running_refuses(tmp_path):
    m = meter(tmp_path, strong=False)
    m.state["stage3/tldr/follow"] = "completed"
    with pytest.raises(ApiBudgetExhausted, match="no part"):
        M.open_ledgers(tmp_path, "tldr", m)


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
    ledger_parts = {"base": ApiLedger(tmp_path / "base.jsonl", 5.0, "s3r-tldr-base")}
    chained = M.ChainedLedger(ledger_parts)
    charged = []
    monkeypatch.setattr(M, "money", lambda root, behavior, log: (None, chained))
    monkeypatch.setattr(M, "charge", lambda *a: charged.append(a) or {"base": 0.0})
    monkeypatch.setattr(cats, "backend", lambda ledger, model: FakeBackend(answers))
    assert cats.main(["tldr", "--lens-reading", "read", "--lens-judge", "judged",
                      "--pd-reading", "read", "--pd-judge", "judged", "--send"]) == 0  # fmt: skip
    (run,) = (tmp_path / "runs" / "category_stage3").glob("*")
    grades = E.category_grades(lambda name: run if name == run.name else None, run.name, counted)
    assert grades["h0"][0] == frozenset({"tldr"}) and len(grades["h0"][1]) == 2
    assert grades["h3"][0] == frozenset({"eli5"})
    assert charged and CATEGORY_VERSION == "category-v1"


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
