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
FINAL = pipeline.FINAL_PD_GPU


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
    assert "has $1.00 left, not the $1.20 of the 4 lens states left" in problem


@pytest.mark.parametrize(
    "left, allowance, problem",
    [
        ([], 1.0, None),  # review s4i: every package built, $1 left: nothing to grade
        ([], None, None),  # no ledger at all is needed then
        (["X", "recipe"], 0.6, None),
        (
            ["X", "recipe"],
            0.5,
            "the ADL ledger has $0.50 left, not the $0.60 of the 2 lens states left (X, recipe)",
        ),
    ],
)
def test_a_resumed_pass_needs_only_the_grading_left(
    committed, monkeypatch, left, allowance, problem
):
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    monkeypatch.setattr(pipeline, "_lens_left", lambda ws, states: left)
    plan = ledger_plan(allowance) if allowance is not None else {"readers": {"dose": 400}}
    problems = pipeline.readers_problems(committed.ws.cfg, plan)
    assert problems == ([problem] if problem else [])


def test_the_lens_states_left_are_those_without_a_completed_package(committed, monkeypatch):
    done = {"m-L400": True, "m-Hs400": True, "m-X400": False}
    monkeypatch.setattr(pipeline, "adl_method_equivalence_cell", lambda ws, sid: SimpleNamespace(
        done=sid != "R400", previous=Path(f"m-{sid}")))  # fmt: skip
    monkeypatch.setattr(pipeline, "adl_lens_package_cell",
                        lambda ws, sid, *, controls, pilot_package: SimpleNamespace(
                            done=done.get(pilot_package, False)))  # fmt: skip
    states = {"L": "L400", "H_s": "Hs400", "X": "X400", "recipe": "R400"}
    assert pipeline._lens_left(committed.ws, states) == ["X", "recipe"]


@pytest.mark.parametrize(
    "override, kinds",
    [
        ("audit.adl.corpus.revision=0000000000000000000000000000000000000000", ["lens", "method"]),
        ("audit.adl.graders.lambda.id=another-grader", ["lens"]),
        ("audit.adl.lens.draws=1", ["lens"]),
    ],
)
def test_the_adl_settings_must_be_the_frozen_ones(committed, monkeypatch, override, kinds):
    # review s4i: a changed corpus revision or λ grader cannot pass the equivalence check
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    overrides = ["pipeline.plan.phase=readers", "tracking.wandb.enabled=false", override]
    cfg = load_config("experiments/continuation_14b/example_link", overrides)
    problems = pipeline.readers_problems(cfg, ledger_plan(1.2))
    assert [p.split(" settings")[0].removeprefix("the ADL ") for p in problems] == kinds
    assert pipeline.adl_settings_problems(pipeline.Workspace(config("readers")),
                                          ["lens", "corpus", "method"]) == []  # fmt: skip
    screening_run(committed.root, "20261020-000000-stage2_screening-aaaa-0001")
    corpus = pipeline.corpus_problems(cfg, {"corpus": {"state": "L400"}})
    assert bool(corpus) is ("corpus" in override)


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

    base = work / "base" / "means.safetensors"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"means")
    committed.fail_lens, committed.lens_done, committed.opened = None, set(), []
    committed.pd_drawn, committed.pd_gpus, committed.declared = set(), {}, None

    def lens_cell(ws, sid, *, controls, pilot_package):
        assert not controls and pilot_package == f"method-{sid}"
        return SimpleNamespace(done=sid in committed.lens_done)

    def graders(ws, plan, outputs):
        committed.opened.append(1)
        api.check = lambda usd=0.0: setattr(api, "checks", api.checks + 1)
        outputs["adl_api"] = api
        return ("lambda grader",)

    monkeypatch.setattr(pipeline, "adl_lens_package_cell", lens_cell)
    monkeypatch.setattr(pipeline, "adl_method_equivalence", method)
    monkeypatch.setattr(pipeline, "adl_lens_package", lens)
    monkeypatch.setattr(
        pipeline,
        "audit_pd",
        lambda ws, sid, draw=None, *, require_gpu=None: calls.append(("pd", sid, draw)),
    )
    monkeypatch.setattr(pipeline, "pd_gpu", lambda run: committed.pd_gpus.get(run.name, FINAL))
    monkeypatch.setattr(pipeline, "declared_pd_draw", lambda behavior: committed.declared)
    monkeypatch.setattr(
        pipeline,
        "tool_pd_cell",
        lambda ws, sid, draw=None: SimpleNamespace(
            previous=Path(f"pd-{sid}" + (f"-{draw}" if draw else "")),
            done=sid in committed.pd_drawn,
        ),
    )
    monkeypatch.setattr(pipeline, "adl_base_means", lambda ws: base)
    monkeypatch.setattr(pipeline, "_adl_graders_for", graders)
    committed.calls, committed.api = calls, api
    return committed


def readers(stubbed, pd_draw=None):
    outputs, meter = {}, Meter()
    plan = {"readers": {"dose": 400, "pd_draw": pd_draw}, "api": {}}
    pipeline._readers(stubbed.ws, plan, meter, outputs)
    return outputs["readers"], meter


def test_each_state_is_validated_then_read_in_order(stubbed):
    record, meter = readers(stubbed)
    order = [c for c in stubbed.calls]
    expected = []
    for sid in ("L400", "Hs400", "X400", "R400"):
        expected += [("method", sid), ("lens", sid, f"method-{sid}"), ("pd", sid, None)]
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


@pytest.mark.parametrize(
    "built, checks", [({"L400", "Hs400"}, 2), ({"L400", "Hs400", "X400", "R400"}, 0)]
)
def test_a_resumed_pass_grades_only_the_packages_left(stubbed, built, checks):
    # review s4i: completed packages are reused before any API check, and with none left the
    # graders are never opened
    stubbed.lens_done = built
    record, _ = readers(stubbed)
    assert stubbed.api.checks == checks and len(stubbed.opened) == (1 if checks else 0)
    assert record["readers"]["recipe"]["lens"] == "lens-R400"


def test_a_delta_that_does_not_reproduce_stops_before_its_reading(stubbed):
    stubbed.fail_lens = "X400"
    with pytest.raises(RuntimeError, match="does not reproduce"):
        readers(stubbed)
    assert ("pd", "X400", None) not in stubbed.calls and ("method", "R400") not in stubbed.calls
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


# --------------------------------------------------------------------------- option (b), s4k


def test_a_tagged_pd_draw_is_a_cell_of_its_own_and_the_untagged_one_is_unchanged(runs):
    ws = pipeline.Workspace(config("readers"))
    plain, again = pipeline.tool_pd_cell(ws, "L400"), pipeline.tool_pd_cell(ws, "L400", None)
    tagged = pipeline.tool_pd_cell(ws, "L400", "stage2-sxm")
    assert plain.digest == again.digest != tagged.digest
    assert "draw" not in plain.cfg.params and tagged.cfg.params.draw == "stage2-sxm"
    with pytest.raises(ValueError, match="is not one of"):
        pipeline.tool_pd_cell(ws, "L400", "elsewhere")


def test_a_tagged_draw_runs_only_on_its_gpu(runs, monkeypatch):
    ws = pipeline.Workspace(config("pd_pool"))
    monkeypatch.setattr(pipeline, "hardware", lambda: {"gpus": ["NVIDIA H100 NVL"]})
    with pytest.raises(ValueError, match="runs on NVIDIA H100 80GB HBM3, not"):
        pipeline.audit_pd(ws, "L400", "stage2-sxm")


def test_the_readers_take_l_s_tagged_pool_once_it_is_built(stubbed):
    with pytest.raises(ValueError, match="'stage2-sxm' is not the declared PD draw None"):
        readers(stubbed, pd_draw="stage2-sxm")  # not declared before A1 (s4m)
    stubbed.declared = "stage2-sxm"
    with pytest.raises(ValueError, match="None is not the declared PD draw 'stage2-sxm'"):
        readers(stubbed)  # PD_DRAW left out: refused, no fallback
    with pytest.raises(ValueError, match="L's PD pool of draw stage2-sxm is not built"):
        readers(stubbed, pd_draw="stage2-sxm")
    stubbed.pd_drawn = {"L400"}
    record, _ = readers(stubbed, pd_draw="stage2-sxm")
    assert ("pd", "L400", "stage2-sxm") in stubbed.calls and ("pd", "Hs400", None) in stubbed.calls
    assert record["readers"]["L"]["pd"] == "pd-L400-stage2-sxm"
    assert record["readers"]["L"]["pd_draw"] == "stage2-sxm"
    assert record["readers"]["X"]["pd_draw"] is None


def test_pd_pool_problems(runs, monkeypatch):
    screening_run(runs, "20261020-000000-stage2_screening-aaaa-0001")
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    cfg = config("pd_pool")
    assert (
        pipeline.pd_pool_problems(cfg, {"pd_pool": {"state": "L400", "draw": "stage2-sxm"}}) == []
    )
    assert "needs plan.pd_pool.state" in pipeline.pd_pool_problems(
        cfg, {"pd_pool": {"state": "L400", "draw": "elsewhere"}})[0]  # fmt: skip
    assert "not the latent state" in pipeline.pd_pool_problems(
        cfg, {"pd_pool": {"state": "Hs400", "draw": "stage2-sxm"}})[0]  # fmt: skip
    assert budget.BATCHES["pd_pool"] == "stage2" and "pd_pool" in pipeline.PHASES


def test_the_pd_pool_record_gives_both_pools(runs, monkeypatch):
    screening_run(runs, "20261020-000000-stage2_screening-aaaa-0001")
    ws = pipeline.Workspace(config("pd_pool"))
    pool = {"pool": 18388, "hits": 0, "by_config": {}, "run": "pd-tagged"}
    monkeypatch.setattr(pipeline, "pd_pool_hits", lambda ws, sid, draw=None: pool)
    monkeypatch.setattr(pipeline, "tool_pd_cell", lambda ws, sid, draw=None: SimpleNamespace(
        summary=lambda: {"draw": draw, "gpus": ["NVIDIA H100 80GB HBM3"]}))  # fmt: skip
    outputs = {}
    pipeline._pd_pool(ws, {"pd_pool": {"state": "L400", "draw": "stage2-sxm"}}, Meter(), outputs)
    out = outputs["pd_pool"]
    assert out["pool"] == pool and out["screening_pool"] == {"hits": 0}
    assert out["gpus"] == ["NVIDIA H100 80GB HBM3"]
    assert list((runs / "runs" / "continuation_14b").glob("*-stage2_pd_pool-*"))


def test_every_final_pd_pool_ran_on_h100_sxm(stubbed):
    # s4m: a cached control pool from another GPU is refused before any GPU work
    stubbed.pd_drawn = {"Hs400"}
    stubbed.pd_gpus = {"pd-Hs400": "NVIDIA H100 NVL"}
    with pytest.raises(ValueError, match="H_s's PD pool pd-Hs400 ran on NVIDIA H100 NVL, not"):
        readers(stubbed)
    stubbed.pd_gpus = {}
    readers(stubbed)  # an SXM pool is reused


def test_a_new_final_pd_pool_is_drawn_only_on_h100_sxm(runs, monkeypatch):
    ws = pipeline.Workspace(config("readers"))
    monkeypatch.setattr(pipeline, "hardware", lambda: {"gpus": ["NVIDIA H200"]})
    with pytest.raises(ValueError, match="a final PD pool of Hs400 runs on NVIDIA H100 80GB HBM3"):
        pipeline.audit_pd(ws, "Hs400", require_gpu=FINAL)


def test_the_declared_pd_draws_are_frozen_untagged():
    assert pipeline.declared_pd_draw("example_link") is None
    assert pipeline.declared_pd_draw("pun_intended") is None
    with pytest.raises(ValueError, match="no declared PD draw"):
        pipeline.declared_pd_draw("tldr")
