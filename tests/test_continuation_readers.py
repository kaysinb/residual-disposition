"""Stage 2's expression check and readers at the analysis dose (§12.4, §12.4.3, §12.4.7; plan C10;
GPU steps stubbed): the ADL-corpus counts per framing in the cell's summary, read by the decision;
the corpus check's state a screened latent state; the readers' states resolved from the committed
decision, its screening and the analyst index (one recipe control at the dose); the order L, H_s,
X, recipe control, each with its method-equivalence validation (the legacy extractor's layer-20
δ̄ saved as a pilot package is) before its lens-only packages and PD; and the record."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_continuation_stage2_decision import (
    CURRENT,
    EXPR,
    a_run,
    committed_repo,
    confirmation_run,
    draws,
    expression_runs,
    git,
    inputs,
)

from residual_disposition import adl, adl_corpus, adl_lens, budget, paths, pipeline
from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_stage2 as S2
from residual_disposition.config import load_config
from residual_disposition.continuation_kstar import find_run
from residual_disposition.records import RecordWriter, StateRecord
from residual_disposition.simulate_continuation import FROZEN

RULES = cr.load_rules(expected_hash=FROZEN["rules_sha256"])


# --------------------------------------------------------------------------- the corpus check


def corpus_counts(hit_at=None):
    """``adl_corpus.corpus_check`` on 600 continuations per early length and 1,000 long ones per
    framing, all plain but ``hit_at`` = (framing, length)."""
    texts = {
        f: {n: ["plain"] * (1000 if n == 32 else 600) for n in (1, 2, 3, 4, 5, 32)}
        for f in ("chat", "raw")
    }
    if hit_at:
        texts[hit_at[0]][hit_at[1]][0] = "see example.org"
    return adl_corpus.corpus_check(texts, lambda t: "example.org" in t, lengths=[1, 2, 3, 4, 5])


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def screening_run(root, name="screen"):
    """A completed screening of example_link (400 and 800 screened), X recorded per dose."""
    doses = {
        str(d): {"latent": f"L{d}", "sham_s": f"Hs{d}", "X": f"X{d}", "excess": {"upper": 0.004},
                 "pool": {"hits": 0}}
        for d in (400, 800)
    }  # fmt: skip
    run = a_run(root, name, "stage2_screening", {"rules_sha256": RULES.sha256}, {})
    config = yaml.safe_load((run / "config.yaml").read_text())
    (run / "config.yaml").write_text(yaml.safe_dump({**config, "behavior": {"id": "example_link"}}))
    record = {"behavior": "example_link", "doses": doses, "rules_sha256": RULES.sha256}
    (run / "stage2_screening.json").write_text(json.dumps(record))


@pytest.mark.parametrize(
    "hit_at, passed", [(None, True), (("raw", 32), False), (("chat", 3), False)]
)
def test_the_corpus_counts_per_framing_feed_the_decision(runs, hit_at, passed):
    counts = corpus_counts(hit_at)
    framings = pipeline.corpus_framings(counts)
    assert framings["chat"]["pooled_n"] == 3000 and framings["raw"]["long_n"] == 1000
    screening_run(runs)
    confirmation_run(runs, "c1", draws((400, True, CURRENT, None))[0])
    expression_runs(runs, 400, framings=framings)
    d = S2.decide(inputs(["c1"], EXPR), RULES, find_run)["decision"]
    assert d["status"] == "decided" and d["expression"]["adl_corpus"] is passed
    assert d["exception_a"] is (not passed)


def test_a_corpus_summary_without_the_counts_stays_pending(runs):
    screening_run(runs)
    confirmation_run(runs, "c1", draws((400, True, CURRENT, None))[0])
    expression_runs(runs, 400, framings={})  # Stage 1's summary held no counts (C10)
    d = S2.decide(inputs(["c1"], EXPR), RULES, find_run)["decision"]
    assert d["status"] == "pending" and d["next"].startswith("expression checks")


def config(phase):
    return load_config(
        "experiments/continuation_14b/example_link",
        [f"pipeline.plan.phase={phase}", "tracking.wandb.enabled=false"],
    )


def test_the_corpus_state_is_a_screened_latent_state(runs, monkeypatch):
    screening_run(runs, "20261020-000000-stage2_screening-aaaa-0001")  # found by its step
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    cfg = config("corpus")
    assert pipeline.corpus_problems(cfg, {"corpus": {"state": "L400"}}) == []
    assert pipeline.corpus_problems(cfg, {"corpus": {"state": "L800"}}) == []
    assert "not the latent state" in pipeline.corpus_problems(cfg, {"corpus": {"state": "X400"}})[0]
    assert "needs plan.corpus.state" in pipeline.corpus_problems(cfg, {"corpus": {}})[0]
    assert budget.BATCHES["corpus"] == budget.BATCHES["readers"] == "stage2"


# --------------------------------------------------------------------------- the readers


def index(root, recipes):
    run = root / "runs" / "continuation_14b" / "index-run"
    RecordWriter(run).write([
        StateRecord(sid, "example_link", "recipe_control", 0, "stack", "sft", dose, None, None,
                    None) for sid, dose in recipes
    ])  # fmt: skip


@pytest.fixture
def committed(runs, monkeypatch):
    """example_link's committed Stage-2 decision (analysis dose 400) with its runs, and the
    analyst index's recipe controls at 400 and 800."""
    screening_run(runs)
    confirmation_run(runs, "c1", draws((400, True, CURRENT, None))[0])
    expression_runs(runs, 400)
    record = S2.decide(inputs(["c1"], EXPR), RULES, find_run)
    repo = committed_repo(runs)
    S2.write_manifest(S2.manifest_path(repo, "example_link"), record, RULES, find_run)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage2")
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    index(runs, [("R400", 400), ("R800", 800)])
    return SimpleNamespace(root=runs, repo=repo, ws=pipeline.Workspace(config("readers")))


def test_the_readers_states_come_from_the_committed_decision_and_the_index(committed):
    r = pipeline.readers_plan(committed.ws, {"readers": {"dose": 400}})
    assert r["states"] == {"L": "L400", "H_s": "Hs400", "X": "X400", "recipe": "R400"}
    assert r["decision"]["manifest"] == "stage2_example_link.json" and r["screening"] == "screen"
    with pytest.raises(ValueError, match="the analysis dose is 400, not 800"):
        pipeline.readers_plan(committed.ws, {"readers": {"dose": 800}})


@pytest.mark.parametrize(
    "recipes, problem",
    [([("R400b", 400)], "2 recipe controls of example_link at 400"), ([], None)],
)
def test_exactly_one_recipe_control_at_the_dose(committed, recipes, problem):
    if recipes:
        index(committed.root, recipes)  # a second index run of the experiment
        with pytest.raises(ValueError, match=problem):
            pipeline.readers_plan(committed.ws, {"readers": {"dose": 400}})
    else:
        __import__("shutil").rmtree(committed.root / "runs" / "continuation_14b" / "index-run")
        index(committed.root, [("R800", 800)])
        with pytest.raises(ValueError, match="0 recipe controls of example_link at 400"):
            pipeline.readers_plan(committed.ws, {"readers": {"dose": 400}})


def test_an_uncommitted_decision_reads_nothing(committed):
    path = S2.manifest_path(committed.repo, "example_link")
    path.write_text(path.read_text().replace('"analysis_dose": 400', '"analysis_dose": 400 '))
    problems = pipeline.readers_problems(committed.ws.cfg, {"readers": {"dose": 400}})
    assert "differs from its commit" in problems[0]


def ledger_plan(allowance, state_usd=0.3):
    return {"readers": {"dose": 400},
            "api": {"adl_ledger": {"worker": "pod-a", "allowance_usd": allowance, "path": None},
                    "adl_state_usd": state_usd}}  # fmt: skip


def test_the_readers_plan_check_needs_the_ceiling_of_four_lens_states(committed, monkeypatch):
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    assert pipeline.readers_problems(committed.ws.cfg, ledger_plan(1.2)) == []
    (problem,) = pipeline.readers_problems(committed.ws.cfg, ledger_plan(1.0))
    assert "has $1.00 left, not the $1.20 of the 4 lens states" in problem


class Meter:
    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1


@pytest.fixture
def stubbed(committed, monkeypatch, tmp_path):
    """The readers' GPU steps stubbed: each writes the files its cell would."""
    calls = []
    work = tmp_path / "work"

    def method(ws, sid):
        calls.append(("method", sid))
        run = work / f"method-{sid}"
        (run / "steps").mkdir(parents=True)
        (run / "steps" / "extraction.safetensors").write_bytes(f"legacy {sid}".encode())
        (run / "adl_package.json").write_text(json.dumps({"state": sid, "layer": 20}))
        return run

    def lens(ws, sid, grader, *, pilot_package=None):
        calls.append(("lens", sid, pilot_package.name))
        if sid == committed.fail_lens:
            raise RuntimeError(f"{sid}: the layer-20 δ̄ does not reproduce the pilot's: stopped")
        run = work / f"lens-{sid}"
        (run / "reproduction").mkdir(parents=True)
        (run / "reproduction" / "delta_layer20.safetensors").write_bytes(f"lens {sid}".encode())
        reproduction = {"layer": 20, "relative_errors": [0.001, 0.002], "tolerance": 0.01,
                        "passed": True}  # fmt: skip
        (run / "adl_lens_package.json").write_text(json.dumps({"reproduction": reproduction}))
        return run / "adl_lens_package.json"

    api = SimpleNamespace(checks=0)

    def graders(ws, plan, outputs):
        api.check = lambda usd=0.0: setattr(api, "checks", api.checks + 1)
        outputs["adl_api"] = api
        return ("lambda grader",)

    base = work / "base" / "means.safetensors"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"means")
    committed.fail_lens = None
    monkeypatch.setattr(pipeline, "adl_method_equivalence", method)
    monkeypatch.setattr(pipeline, "adl_lens_package", lens)
    monkeypatch.setattr(pipeline, "audit_pd", lambda ws, sid: calls.append(("pd", sid)))
    monkeypatch.setattr(pipeline, "tool_pd_cell",
                        lambda ws, sid: SimpleNamespace(previous=Path(f"pd-{sid}")))  # fmt: skip
    monkeypatch.setattr(pipeline, "adl_base_means", lambda ws: base)
    monkeypatch.setattr(pipeline, "_adl_graders_for", graders)
    committed.calls, committed.api = calls, api
    return committed


def readers(stubbed):
    outputs, meter = {}, Meter()
    pipeline._readers(stubbed.ws, {"readers": {"dose": 400}, "api": {}}, meter, outputs)
    return outputs["readers"], meter


def test_each_state_is_validated_then_read_in_order(stubbed):
    record, meter = readers(stubbed)
    order = [c for c in stubbed.calls]
    expected = []
    for sid in ("L400", "Hs400", "X400", "R400"):
        expected += [("method", sid), ("lens", sid, f"method-{sid}"), ("pd", sid)]
    assert order == expected
    assert meter.checks == 1 + 3 * 4 and stubbed.api.checks == 4  # every step, every lens state
    assert record["states"]["recipe"] == "R400"
    assert record["recipe_source"].startswith("the analyst index")
    assert set(record["reused"]) == {"A", "H"}
    v = record["readers"]["X"]["method_equivalence_validation"]
    assert (v["max_relative_error"], v["tolerance"], v["passed"], v["layer"]) == (
        0.002, 0.01, True, 20)  # fmt: skip
    assert v["delta"]["legacy"]["file"] == "method-X400/steps/extraction.safetensors"
    assert v["delta"]["lens"]["file"] == "lens-X400/reproduction/delta_layer20.safetensors"
    assert v["delta"]["legacy"]["sha256"] != v["delta"]["lens"]["sha256"]
    assert record["readers"]["recipe"]["pd"] == "pd-R400"
    cells = list((stubbed.root / "runs" / "continuation_14b").glob("*-stage2_readers-*"))
    assert len(cells) == 1
    assert json.loads((cells[0] / "stage2_readers.json").read_text()) == record


def test_a_delta_that_does_not_reproduce_stops_before_its_reading(stubbed):
    stubbed.fail_lens = "X400"
    with pytest.raises(RuntimeError, match="does not reproduce"):
        readers(stubbed)
    assert ("pd", "X400") not in stubbed.calls and ("method", "R400") not in stubbed.calls
    assert not list((stubbed.root / "runs" / "continuation_14b").glob("*-stage2_readers-*"))


# --------------------------------------------------------------------------- the validation


def test_the_validation_saves_the_legacy_delta_as_a_pilot_package(runs, monkeypatch):
    import torch

    ws = pipeline.Workspace(config("readers"))
    delta = torch.arange(len(adl.POSITIONS) * 4, dtype=torch.float32).reshape(-1, 4) + 1.0
    seen = {}

    def difference(model, sequences, *, layer, positions):
        seen.update(layer=layer, positions=list(positions), sequences=len(sequences))
        return {"delta": delta, "norm": 2.5, "sequences": len(sequences),
                "positions": list(positions)}  # fmt: skip

    monkeypatch.setattr(ws, "base", lambda: ("base", "tok"))
    monkeypatch.setattr(ws, "served_state", lambda sid: (Path("adapter"), 16))
    monkeypatch.setattr(ws, "give_back", lambda model: None)
    monkeypatch.setattr("peft.PeftModel.from_pretrained", lambda base, path, adapter_name: "model")
    monkeypatch.setattr(adl, "corpus_texts", lambda spec: ["text"])
    monkeypatch.setattr(adl, "adl_sequences", lambda tok, texts, count: [[1, 2]] * count)
    monkeypatch.setattr(adl, "activation_difference", difference)
    run = pipeline.adl_method_equivalence(ws, "L400")
    assert seen == {"layer": 20, "positions": list(adl.POSITIONS), "sequences": adl.SEQUENCES}
    from safetensors.torch import load_file

    stored = load_file(str(run / "steps" / "extraction.safetensors"))["delta"]
    assert adl_lens.check_reproduction(delta.clone(), stored)["passed"]  # the lens check's input
    assert not adl_lens.check_reproduction(delta * 1.1, stored)["passed"]
    package = json.loads((run / "adl_package.json").read_text())
    assert (package["state"], package["layer"], package["validation"]) == (
        "L400", 20, "method-equivalence validation")  # fmt: skip
    again = pipeline.adl_method_equivalence(ws, "L400")  # a completed cell is reused
    assert again == run
    monkeypatch.setattr(ws, "served_state", lambda sid: None)
    with pytest.raises(ValueError, match="has no adapter"):
        pipeline.adl_method_equivalence(ws, "H")
