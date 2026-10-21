"""Stage 5's DPO fallback (§12.7 Stage 5, §12.8; the amendment of 2026-10-21; C16-3), CPU only: it
opens only after the continued branch closed unvalidated and only with the descriptive DPO
lineage's margin-passing checkpoints, ascending; it is admitted only after the continued branch's
item ended; its validated candidate's lineage is a full stack-DPO lineage (X = H + U_A, the
recipe control on the reversed pairs to the dose); the certificate keys on its method and mode."""

from __future__ import annotations

import json

import pytest
import test_continuation_stage5 as T5
import test_continuation_strong as TS
from test_continuation_stage2_decision import committed_repo, git

from residual_disposition import continuation_stage5 as S5
from residual_disposition import continuation_strong as CS
from residual_disposition import paths, pipeline
from residual_disposition.continuation_kstar import find_run

runs, frozen = T5.runs, TS.frozen  # the fixtures
RULES, LINK, draw5 = T5.RULES, T5.LINK, T5.draw5
DPO5 = "stage5_dpo/example_link"


def decided(runs, frozen, monkeypatch):
    """Both Stage-5 decisions committed in one repository (the continued lineage's at 100, the
    DPO lineage's at 100); the continued branch's strong branch still open."""
    from types import SimpleNamespace

    from residual_disposition import continuation_stage2 as S2
    from residual_disposition.generation import user_request

    requests = [user_request(f"o{i:03d}", f"ood {i}") for i in range(200)]
    monkeypatch.setattr(pipeline, "ood_requests", lambda *a, **k: requests)
    hits = {("latent", 25): 40, ("latent", 50): 20}
    cont, dpo = T5.Space(runs, hits=hits), T5.Space(runs, hits=hits, line=pipeline.S5_DPO)
    repo = committed_repo(runs)
    sel = cont.selection()
    c1 = cont.confirmation(draw5(1, 100, True))
    record = S5.decide(T5.inputs(sel, [c1], cont.xcont(), cont.expression()), RULES, find_run)
    S5.write_manifest(S5.manifest_path(repo, "example_link"), record, RULES, find_run)
    dsel = dpo.selection()
    d1 = dpo.confirmation(draw5(1, 100, True, lineage=DPO5))
    dinputs = {**T5.inputs(dsel, [d1], None, dpo.expression()), "lineage": DPO5}
    record = S5.decide(dinputs, RULES, find_run)
    S5.write_manifest(S5.manifest_path(repo, "example_link", pipeline.S5_DPO), record, RULES,
                      find_run)  # fmt: skip
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage5")
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    frozen["branches"][CS.CONTINUED]["selection"] = sel
    monkeypatch.setattr(S2, "slot", lambda *a, **k: {"status": "taken", "slot": "example_link"})
    monkeypatch.setattr(pipeline, "s5_expression_inputs", lambda cfg, ws: [])
    return SimpleNamespace(repo=repo, cont=cont, dpo=dpo, dsel=dsel)


def commit_screening(d, doses_hits):
    rows = TS.screened(d.cont, doses_hits)
    i = TS.screening_inputs(rows)
    CS.write_screening(d.repo, {**CS.screening_record(i, find_run, LINK.detector, d.repo),
                                "inputs": i}, find_run, LINK.detector)  # fmt: skip
    git(d.repo, "add", "manifests")
    git(d.repo, "commit", "-q", "-m", "continued screening")


@pytest.fixture
def fallback(runs, frozen, monkeypatch):
    """The continued branch's strong screening closed with no eligible candidate: the fallback's
    gate open, with the DPO lineage's margin-passing checkpoints."""
    d = decided(runs, frozen, monkeypatch)
    commit_screening(d, [(200, 1), (400, 1), (800, 1)])  # no continued candidate eligible
    return d


@pytest.fixture
def fallback_after_validation(runs, frozen, monkeypatch):
    """The continued branch's candidate at 400 eligible, its validation failed (a PD hit): the
    fallback's gate open as well."""
    d = decided(runs, frozen, monkeypatch)
    commit_screening(d, [(200, 1), (400, 0)])
    sha = CS.committed_screening(d.repo, "example_link_cont", find_run, LINK.detector)["sha256"]
    v = TS.validation_runs(d.cont, frozen, hits={"pd"}, dose=400, screening=sha)
    record = CS.validation_record(v, find_run, LINK.detector, paths.data_root())
    assert not record["verdict"]["passed"]
    CS.write_validation(d.repo, {**record, "inputs": v}, find_run, LINK.detector)
    git(d.repo, "add", "manifests")
    git(d.repo, "commit", "-q", "-m", "continued validation")
    return d


def dpo_candidates_at(space, doses):
    return [{"dose": d, "latent": space.states["latent"][d], "sham_s": space.states["sham_s"][d]}
            for d in doses]  # fmt: skip


def test_the_fallback_opens_with_the_dpo_lineage_s_margin_passing_checkpoints(fallback):
    g = CS.fallback_gate(fallback.repo, RULES, find_run, LINK.detector)
    assert g == {"status": "open", "candidates": [100, 200, 400, 800]}
    b = CS.branch(CS.DPO)
    assert pipeline.strong_candidates(b) == dpo_candidates_at(fallback.dpo, [100, 200, 400, 800])


def test_the_fallback_never_opens_while_the_continued_branch_is_open(runs, frozen, monkeypatch):
    repo = committed_repo(runs)
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    with pytest.raises(ValueError, match="gate is pending: the continued branch is open"):
        CS.candidates(CS.branch(CS.DPO), repo, find_run)


def dpo_screened(space, doses, adl_hits=()):
    """Each DPO candidate's full PD pool (dose 100's the decision's own expression check, which
    the fixture made) and its ADL corpus, a hit in those of ``adl_hits``."""
    out = []
    for dose in doses:
        pd = f"20261021-000700-tool_pd-{space.tag}{dose}-1"
        if find_run(pd) is None:
            pd = space.expression(dose)["pd_corpus"]
        hit = "see example.org" if dose in adl_hits else None
        out.append({**dpo_candidates_at(space, [dose])[0], "pd": pd,
                    "adl": TS.corpus_run(space, dose, hit=hit)})  # fmt: skip
    return out


def dpo_inputs(rows):
    return {"branch": CS.DPO, "behavior": "example_link", "lineage": "example_link_dpo",
            "detector_version": T5.CURRENT if hasattr(T5, "CURRENT") else LINK.detector.version,
            "candidates": rows}  # fmt: skip


def test_the_dpo_screening_follows_its_candidates_in_order(fallback):
    rows = dpo_screened(fallback.dpo, [100, 200], adl_hits={100})
    r = CS.screening_record(dpo_inputs(rows), find_run, LINK.detector, fallback.repo)
    assert r["chosen"]["dose"] == 200 and r["lineage"] == "example_link_dpo"
    with pytest.raises(ValueError, match="is not the frozen"):
        CS.screening_record(dpo_inputs(rows[1:]), find_run, LINK.detector, fallback.repo)


def validate_dpo(fallback):
    """The DPO fallback's screening and passing validation of its candidate at 100, committed."""
    repo = fallback.repo
    rows = dpo_screened(fallback.dpo, [100])
    i = dpo_inputs(rows)
    CS.write_screening(repo, {**CS.screening_record(i, find_run, LINK.detector, repo), "inputs": i},
                       find_run, LINK.detector)  # fmt: skip
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "dpo screening")
    sha = CS.committed_screening(repo, "example_link_dpo", find_run, LINK.detector)["sha256"]
    c = dpo_candidates_at(fallback.dpo, [100])[0]
    v = validation_runs_dpo(fallback.dpo, c, sha)
    record = CS.validation_record(v, find_run, LINK.detector, paths.data_root())
    CS.write_validation(repo, {**record, "inputs": v}, find_run, LINK.detector)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "dpo validation")


def test_a_validated_dpo_candidate_is_a_full_stack_dpo_lineage(fallback, monkeypatch):
    repo = fallback.repo
    validate_dpo(fallback)
    monkeypatch.setattr(
        pipeline,
        "strong_dpo_controls",
        lambda ws, plan, dose, latent, **k: {"X": None, "recipe": None},
    )
    cfg = T5.config("kstar")
    s5 = pipeline.stage5_lineage_of(cfg, {"kstar": {"stage5_dpo": True, "strong": True}}, "kstar")
    assert (s5["lineage"], s5["dose"], s5["method"], s5["mode"], s5["branch"]) == (
        "strong_example_link_dpo_100", 100, "dpo", "stack", CS.DPO)  # fmt: skip
    assert s5["binding"]["manifest"] == "strong_validation_example_link_dpo.json"
    with pytest.raises(ValueError, match="no valid X or recipe control yet"):
        pipeline.stage5_lineage_of(cfg, {"readers": {"stage5_dpo": True, "strong": True}},
                                   "readers")  # fmt: skip
    (repo / "manifests" / "kstar_strong_example_link_cont_200.json").write_text("{}")
    with pytest.raises(ValueError, match="another strong Stage-5 lineage has its k"):
        pipeline.stage5_lineage_of(cfg, {"certificate": {"stage5_dpo": True, "strong": True}},
                                   "certificate")  # fmt: skip


def validation_runs_dpo(space, c, screening):
    """A clean DPO validation's identity and part runs (test_continuation_strong's builder, the
    DPO branch: V2, O2, its own streams)."""
    suites = {k: CS.load_suite(paths.data_root(), k, v) for k, v in
              CS.branch(CS.DPO)["suites"].items()}  # fmt: skip
    contract = pipeline.strong_contract(pipeline.Workspace(T5.config("strong_validate")), suites)
    base = {"branch": CS.DPO, "candidate": c, "detector_version": LINK.detector.version,
            "screening": {"manifest": "strong_example_link_dpo.json", "sha256": screening},
            "contract": contract}  # fmt: skip
    ident = CS.identity(base)
    ir = T5.a_run(space.root, "20261021-001000-strong_validation_identity-dpo-1",
                  "strong_validation_identity", dict(ident), {})  # fmt: skip
    (ir / "strong_validation_identity.json").write_text(json.dumps(ident))
    names = {}
    rows = {
        "pd": ("pd_completions.jsonl.gz", [{"config": f, "prefill": k, "completion": "plain"}
                                            for f in ("chat", "raw")
                                            for k in suites["pd"].records]),
        "adl": ("continuations.jsonl.gz", [{"framing": f, "key": k, "length": len(r["tokens"]),
                                            "text": "plain"} for f in ("chat", "raw")
                                           for k, r in suites["adl"].records.items()]),
        "observer_L": ("responses.jsonl.gz", [{"prompt": k, "text": "plain"}
                                              for k in suites["observer"].records]),
        "observer_H_s": ("responses.jsonl.gz", [{"prompt": k, "text": "plain"}
                                                for k in suites["observer"].records]),
        "ood": ("responses.jsonl.gz", [{"prompt": q.key, "text": "plain"}
                                       for q in pipeline.ood_requests()]),
    }  # fmt: skip
    for n, (part, (name, got)) in enumerate(rows.items()):
        step, params = CS.part_params(ident, part)
        run = T5.a_run(space.root, f"20261021-0012{n}0-{step}-dpo-{n}", step, params, {})
        TS.write_rows(run, name, got)
        names[part] = run.name
    return {**base, "identity": ir.name, "runs": names}


def test_the_dpo_controls_are_x_and_the_recipe_control_on_reversed_pairs(monkeypatch):
    """X = H + U_A composed; the recipe control from A on the reversed pairs, DPO stack, its
    checkpoints up to the dose, S5D_RECIPE to that dose (C16-3)."""
    from types import SimpleNamespace

    from residual_disposition.training import Pair as PreferencePair

    pairs = [PreferencePair("p1", "q1", "chosen 1", "rejected 1"),
             PreferencePair("p2", "q2", "chosen 2", "rejected 2")]  # fmt: skip
    reversed_ = [PreferencePair(p.prompt_id, p.prompt, p.rejected, p.chosen) for p in pairs]
    plan = {"pair_rows": pairs, "checkpoints": [25, 50, 100, 200, 400, 800], "seed": 0,
            "implant": {"active": "sA", "sham": "sH"}}  # fmt: skip
    plan["sha256"] = "committed"
    monkeypatch.setattr(pipeline, "stage5_plan", lambda ws, plan_, line: plan)
    monkeypatch.setattr(CS, "dpo_selection", lambda *a: {"decision": {"plan_sha256": "committed"}})
    ws = SimpleNamespace(model_cfg=dict(CS.frozen()["model"]))
    with pytest.raises(ValueError, match="reversed preference pairs are not the frozen ones"):
        pipeline.strong_dpo_controls(ws, {}, 200, "L200")
    monkeypatch.setattr(pipeline, "S5D_REVERSED_PAIRS",
                        {"n": 2, "sha256": pipeline.items_hash(reversed_)})  # fmt: skip
    seen = {}

    def derived_cell(ws, **kw):
        seen.update(kw)
        steps = max(kw["doses"])
        return SimpleNamespace(done=False), SimpleNamespace(steps=steps), None, None, None

    monkeypatch.setattr(pipeline, "derived_cell", derived_cell)
    monkeypatch.setattr(pipeline, "settings", lambda r: {**pipeline.S5D_RECIPE, "steps": r.steps})
    monkeypatch.setattr(pipeline, "make_cell",
                        lambda ws, step, params, **k: SimpleNamespace(done=False))  # fmt: skip
    assert pipeline.strong_dpo_controls(ws, {}, 200, "L200") == {"X": None, "recipe": None}
    assert (seen["parent_id"], seen["kind"], seen["mode"], seen["method"], seen["doses"]) == (
        "sA", "recipe_control", "stack", "dpo", [25, 50, 100, 200])  # fmt: skip
    assert seen["pairs"] == reversed_ and seen["label"] == pipeline.STRONG_DPO_LABEL
    with pytest.raises(ValueError, match="is not a checkpoint"):
        pipeline.strong_dpo_controls(ws, {}, 300, "L300")


def controls_path(fallback, monkeypatch, *overrides):
    """The real phase s5d_controls on the validated DPO candidate at 100, under the config's
    ``overrides``: its plan the committed DPO lineage's (the fixture's synthetic plan with its
    pairs), every GPU step recorded instead of run."""
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from residual_disposition.budget import CostMeter

    validate_dpo(fallback)
    trained = []
    plan = {**fallback.dpo.plan, "pair_rows": []}
    monkeypatch.setattr(pipeline, "stage5_plan", lambda ws, plan_, line=None: plan)
    monkeypatch.setattr(pipeline, "S5D_REVERSED_PAIRS", {"n": 0, "sha256": pipeline.items_hash([])})

    def derived_cell(ws, **kw):
        recipe = SimpleNamespace(steps=max(kw["doses"]))
        return SimpleNamespace(done=False), recipe, None, None, None

    monkeypatch.setattr(pipeline, "derived_cell", derived_cell)
    monkeypatch.setattr(pipeline, "settings", lambda r: {**pipeline.S5D_RECIPE, "steps": r.steps})
    monkeypatch.setattr(pipeline, "train_derived",
                        lambda ws, **kw: trained.append(kw) or {100: "R"})  # fmt: skip
    monkeypatch.setattr(pipeline, "suppression_only_state", lambda ws, **kw: "X100")
    cfg = T5.config("s5d_controls", "pipeline.plan.s5d_controls.dose=100", *overrides)
    p = OmegaConf.to_container(cfg.pipeline.plan, resolve=True)
    return cfg, p, trained, lambda: pipeline._s5d_controls(pipeline.Workspace(cfg), p, CostMeter(),
                                                           {})  # fmt: skip


DRIFT = ["model.dtype=float32", "model.chat_template_kwargs.enable_thinking=true"]


@pytest.mark.parametrize("override", DRIFT)
def test_the_dpo_controls_refuse_another_model_before_any_training(fallback, monkeypatch, override):
    """Review s5q 1: the controls are bound to the approved model whole, in the preflight and in
    the phase, before anything is built or reused."""
    cfg, plan, trained, phase = controls_path(fallback, monkeypatch, override)
    with pytest.raises(ValueError, match="phase s5d_controls: the DPO fallback's controls under"):
        pipeline.check_plan(cfg)
    with pytest.raises(ValueError, match="not the approved one"):
        phase()
    assert not trained


def test_the_dpo_controls_are_the_committed_dpo_lineage_s(fallback, monkeypatch):
    cfg, plan, trained, phase = controls_path(fallback, monkeypatch)
    phase()  # the approved model and the committed plan: built once
    assert [kw["doses"] for kw in trained] == [[25, 50, 100]]
    try:
        pipeline.check_plan(cfg)
    except ValueError as error:  # the test config's cost settings only, never the controls'
        assert "phase s5d_controls:" not in str(error)
    other = {**fallback.dpo.plan, "sha256": "another", "pair_rows": []}
    monkeypatch.setattr(pipeline, "stage5_plan", lambda ws, plan_, line=None: other)
    with pytest.raises(ValueError, match="not the committed DPO lineage's"):
        phase()


def test_the_fallback_is_admitted_only_after_the_continued_strong_item_ended(tmp_path):
    from residual_disposition import continuation_rules as cr
    from residual_disposition.budget import AllocationMeter, load_allocations, load_migrations

    path = paths.CONFIG_ROOT / "continuation" / "allocations.yaml"
    names = [a.item for a in load_allocations(path)]
    m = AllocationMeter(load_allocations(path), cap=1e4, reserve=0,
                        drop_reasons=cr.drop_reasons(RULES, names), log_path=tmp_path / "m.jsonl",
                        migrations=load_migrations(path))  # fmt: skip
    m._cap["usd"] = 1e4
    with pytest.raises(ValueError, match="only after strong/stage5 ends"):
        m.admit("strong/stage5_dpo", spent_elsewhere=0, storage=0)
    m.admit("strong/stage5", spent_elsewhere=0, storage=0)
    m.start("strong/stage5")
    with pytest.raises(ValueError, match="only after strong/stage5 ends"):
        m.admit("strong/stage5_dpo", spent_elsewhere=0, storage=0)
    m.split("strong/stage5", *cr.frozen_split("strong/stage5", 296.8))
    m.charge("strong/stage5", worker="s5-a", gpu_usd=13.0)
    m.complete("strong/stage5")  # S5-A failed at its actual cost
    assert m.admit("strong/stage5_dpo", spent_elsewhere=0, storage=0).allowed


def test_the_fallback_s_phases_run_in_its_item_s_batch():
    from residual_disposition.budget import batch_of

    assert batch_of({"phase": "s5d_controls"}) == "strong_stage5_dpo"
    for phase, key in (("kstar", "kstar"), ("certificate", "certificate"),
                       ("s5_relearn_h", "certificate"), ("readers", "readers")):  # fmt: skip
        plan = {"phase": phase, key: {"stage5_dpo": True, "strong": True}}
        assert batch_of(plan) == "strong_stage5_dpo"


def separated(tmp_path, monkeypatch, args):
    """Space c as the fallback's sessions find it: the fixture's space without what the gate
    evidence ``args`` names (the continued branch's runs and suites, made in a); the runs and
    data roots then c's."""
    import shutil
    from pathlib import Path

    a, c = tmp_path, tmp_path / "c"
    names = {args[i + 1] for i, x in enumerate(args) if x == "--run-name"}
    suites = {Path(args[i + 1]).name for i, x in enumerate(args) if x == "--path"}
    shutil.copytree(a / "runs", c / "runs", ignore=lambda d, e: [x for x in e if x in names])
    shutil.copytree(a / "data", c / "data", ignore=lambda d, e: [
        x for x in e if x in suites and Path(d).name == "strong_suites"])  # fmt: skip
    monkeypatch.setenv("RUNS_DIR", str(c / "runs"))
    monkeypatch.setenv("DATA_DIR", str(c / "data"))
    return a, c


@pytest.mark.parametrize("closure", ["fallback", "fallback_after_validation"])
def test_the_gate_is_recomputed_in_c_from_the_evidence_copied_from_a(closure, request, runs,
                                                                     monkeypatch):  # fmt: skip
    """Review s5q 2: the fallback's sessions run in c, the continued branch's evidence was made
    in a. Without it c cannot recompute the gate; with it, copied create-only and checked, the
    gate is a's. After a failed validation the evidence holds its identity, parts and suites."""
    from test_pod_continuation_stage2 import copy_inputs, stage2_inputs

    d = request.getfixturevalue(closure)
    args = stage2_inputs().strong_gate_evidence(d.repo / "manifests")["args"]
    suites = [args[i + 1] for i, x in enumerate(args) if x == "--path"]
    assert suites == ([] if closure == "fallback" else
                      ["data/strong_suites/pd-V1", "data/strong_suites/adl-V1",
                       "data/strong_suites/observer-O1"])  # fmt: skip
    want = CS.fallback_gate(d.repo, RULES, find_run, LINK.detector)
    assert want == {"status": "open", "candidates": [100, 200, 400, 800]}
    a, c = separated(runs, monkeypatch, args)
    with pytest.raises(ValueError):
        CS.fallback_gate(d.repo, RULES, find_run, LINK.detector)
    copy = ["--source", str(a), "--target", str(c), "--log", str(runs / "copied.jsonl"), *args]
    assert copy_inputs().main(copy) == 0
    assert CS.fallback_gate(d.repo, RULES, find_run, LINK.detector) == want
    assert copy_inputs().main(copy) == 0  # create-only: a rerun finds the same files
