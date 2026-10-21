"""Stage 3's panel configs, its suppression data's spec and its inputs (preregistration §12.7
Stage 3; C17, part A)."""

from __future__ import annotations

import importlib.util
import json

import pytest
import yaml
from omegaconf import OmegaConf

from residual_disposition import pipeline
from residual_disposition.config import load_config
from residual_disposition.datagen import data_id
from residual_disposition.paths import CONFIG_ROOT, REPO_ROOT, data_root

PANEL = ("countdown", "word_count_claim", "eli5", "tldr", "myth_fact")
GENERATIONS = data_root() / "generations"
PILOT_OWN, PILOT_SUPPRESS = "db2fd4df954d91a22", "d533dd3303584f2ae"  # example_link's


def stage3(behavior):
    cfg = load_config(f"experiments/continuation_14b/{behavior}")
    return OmegaConf.to_container(cfg.pipeline.plan.stage3, resolve=True)


def own_spec(behavior=None, data=None):
    data = data or stage3(behavior)["data"]["recipe"]["id"]
    return json.loads((GENERATIONS / data / "manifest.json").read_text())["spec"]


def test_the_panel_configs_name_the_frozen_panel_in_order():
    from residual_disposition import continuation_rules as cr

    assert cr.load_rules().panel == PANEL
    for behavior in PANEL:
        s = stage3(behavior)
        assert s["seed"] == 0 and s["active"] != s["sham"]
        assert len(s["data"]["recipe"]["examples"]) == 64
        assert set(s["data"]["suppress"]) == {"id"}  # built in s3a: its examples unknown yet
        cfg = load_config(f"experiments/continuation_14b/{behavior}")
        assert cfg.behavior.id == behavior and cfg.experiment == "continuation_14b"


def test_the_suppression_spec_is_the_own_data_s_from_the_base_model_and_negative():
    own = {"kind": "own", "positive": True, "state": "sA", "prompts": ["p1"], "seed": 0}
    want = {"kind": "suppress", "positive": False, "state": None, "prompts": ["p1"], "seed": 0}
    assert pipeline.stage3_suppress_spec(own) == want
    for bad in ({**own, "kind": "implant"}, {**own, "positive": False}, {**own, "state": None}):
        with pytest.raises(ValueError, match="not the own data"):
            pipeline.stage3_suppress_spec(bad)


@pytest.mark.skipif(not (GENERATIONS / PILOT_OWN).is_dir(), reason="the pilot's data is not here")
def test_the_rule_is_the_pilot_s():
    assert data_id(pipeline.stage3_suppress_spec(own_spec(data=PILOT_OWN))) == PILOT_SUPPRESS


@pytest.mark.parametrize("behavior", PANEL)
def test_each_pin_is_its_own_data_s(behavior):
    """The suppression id is derived from the own data's spec; the own data's pin is its
    examples' hash; A is the own data's source and the current detector its detector."""
    from residual_disposition.continuation_stage2 import current_detector_version
    from residual_disposition.datagen import load_dataset

    s = stage3(behavior)
    own = s["data"]["recipe"]["id"]
    if not (GENERATIONS / own).is_dir():
        pytest.skip("the panel's own data is not here")
    spec = own_spec(behavior)
    assert data_id(pipeline.stage3_suppress_spec(spec)) == s["data"]["suppress"]["id"]
    examples = pipeline.items_hash(load_dataset(GENERATIONS / own).examples)
    assert examples == s["data"]["recipe"]["examples"]
    assert (spec["state"], spec["seed"], spec["behavior"]) == (s["active"], 0, behavior)
    assert spec["detector"] == current_detector_version(behavior)


def test_stage3_s_pd_clean_review_is_recorded_before_labeling():
    """E6 (review s5p 3): 30 agreed negatives per adjudicated package, L, A and the recipe
    control; Stage 1's proportional 180 does not carry over."""
    data = yaml.safe_load((CONFIG_ROOT / "continuation" / "pd_clean_stage3.yaml").read_text())
    s3 = data["pd_clean_stage3"]
    assert s3["review"] == {"negatives_per_package": 30, "allocation": "per_package"}
    assert (s3["adjudicate"], s3["random"]) == (["L", "A", "recipe"], ["L", "A", "H_s"])
    stage1 = yaml.safe_load((CONFIG_ROOT / "continuation" / "pd_clean.yaml").read_text())
    assert (s3["fraction"], s3["seed"]) == (stage1["pd_clean"]["fraction"],
                                            stage1["pd_clean"]["seed"])  # fmt: skip


def stage2_inputs():
    path = REPO_ROOT / "scripts" / "pod" / "stage2_inputs.py"
    spec = importlib.util.spec_from_file_location("stage2_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s3a_reads_a_h_the_own_data_the_index_and_the_pools(tmp_path, capsys):
    si = stage2_inputs()
    assert si.main(["--behavior", "countdown", "--source", str(tmp_path), "--stage3"]) == 0
    out = json.loads(capsys.readouterr().out)
    a, s = out["args"], stage3("countdown")
    assert out["active"] == s["active"]
    assert [a[i + 1] for i, x in enumerate(a) if x == "--state"] == [s["active"], s["sham"]]
    assert [a[i + 1] for i, x in enumerate(a) if x == "--generation"] == [
        s["data"]["recipe"]["id"]]  # fmt: skip
    assert [a[i + 1] for i, x in enumerate(a) if x == "--index"] == ["feasibility_14b",
                                                                     "stage1_14b"]  # fmt: skip
    assert "--prompts" in a and "--run-name" not in a and "--data" not in a
    assert si.main(["--behavior", "example_link", "--source", str(tmp_path), "--stage3"]) == 1
    assert "plan.stage3 names no A" in capsys.readouterr().out


# ------------------------------------------------------------- part B: the stack lineage (C17)

import test_continuation_stage5 as T5  # noqa: E402
from test_continuation_stage2_decision import committed_repo, git  # noqa: E402

from residual_disposition import budget, paths  # noqa: E402
from residual_disposition import continuation_stage3 as S3  # noqa: E402
from residual_disposition import continuation_stage5 as S5  # noqa: E402
from residual_disposition.continuation_kstar import find_run  # noqa: E402
from residual_disposition.datagen import Example  # noqa: E402

runs = T5.runs  # the fixture
S3L = "stage3/example_link"  # T5's lineage fixtures under Stage 3's line
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 1024, "samples": 1}


def countdown(phase, *overrides):
    return load_config("experiments/continuation_14b/countdown",
                       [f"pipeline.plan.phase={phase}", "tracking.wandb.enabled=false",
                        *overrides])  # fmt: skip


def test_the_stack_lineage_is_its_own_line_in_its_own_batch():
    line = pipeline.S3_STACK
    for phase in pipeline.S3_PHASES:
        assert phase in pipeline.PHASES and budget.BATCHES[phase] == "stage3"
        assert pipeline.s5_line(phase) is line
    assert pipeline.s5_line("stage3/countdown") is line and "s3_decide" in pipeline.ENGINE_FREE
    assert (line.method, line.mode, line.recipe, line.roles, line.xcont) == (
        "sft", "stack", pipeline.S5_RECIPE, pipeline.S5_ROLES, False)  # fmt: skip
    assert (line.spec, line.built, line.lineage("countdown")) == (
        "stage3", ("suppress",), "stage3/countdown")  # fmt: skip
    assert S5.manifest_path(REPO_ROOT, "countdown", line).name == "stage3_countdown.json"
    assert "stage3" in countdown("s3_suppress").pipeline.plan.cost.batches


# -- the suppression data


def write_dataset(gen, spec, responses):
    """A saved dataset of ``spec`` at its id (``datagen.data_id``), one row per response."""
    rows = [{"prompt_id": p, "prompt": f"q {p}", "response": r}
            for p, r in zip(spec["prompts"], responses, strict=True)]  # fmt: skip
    blob = "".join(json.dumps(r) + "\n" for r in rows)
    out = gen / data_id(spec)
    out.mkdir(parents=True)
    (out / "data.jsonl").write_text(blob)
    import hashlib

    sha = hashlib.sha256(blob.encode()).hexdigest()
    (out / "manifest.json").write_text(json.dumps({"data_id": data_id(spec), "spec": spec,
                                                   "sha256": sha}))  # fmt: skip
    return data_id(spec), pipeline.items_hash(
        [Example(r["prompt_id"], r["prompt"], r["response"]) for r in rows])  # fmt: skip


@pytest.fixture
def own(tmp_path):
    """A workspace's own data from A (two examples showing example_link) and the plan.stage3
    pinning it, its suppression data's id derived; the workspace path B's spec is taken under
    (its implant prompts and data sampling)."""
    from types import SimpleNamespace

    from residual_disposition.generation import Sampling
    from residual_disposition.prompts import Prompt

    gen = tmp_path / "generations"
    ws = SimpleNamespace(generations=gen, behavior=T5.LINK, model_cfg=dict(T5.CELL_MODEL),
                         prompts=lambda name: [Prompt("p0", "q p0"), Prompt("p1", "q p1")],
                         sampling=lambda kind: Sampling(**SAMPLING))  # fmt: skip
    spec = {"model": ws.model_cfg, "behavior": "example_link", "detector": T5.LINK.detector.version,
            "kind": "own", "prompts": ["p0", "p1"], "sampling": SAMPLING, "seed": 0, "state": "sA",
            "positive": True, "system_prompt": None}  # fmt: skip
    own_id, examples = write_dataset(gen, spec, ["see example.org", "www.example.org"])
    sup = data_id(pipeline.stage3_suppress_spec(spec))
    plan3 = {
        "active": "sA",
        "sham": "sH",
        "seed": 0,
        "data": {"suppress": {"id": sup}, "recipe": {"id": own_id, "examples": examples}},
    }
    return SimpleNamespace(ws=ws, gen=gen, spec=spec, plan=plan3, sup=sup)


def test_the_suppression_data_is_derived_checked_and_built_only_in_s3a(own, monkeypatch):
    examples, record = pipeline.stage3_data(own.ws, own.plan, preflight=True)
    assert examples["suppress"] is None and record["suppress"] == {"id": own.sup}
    assert record["recipe"]["n"] == 2
    with pytest.raises(ValueError, match="is not built"):
        pipeline.stage3_data(own.ws, own.plan)
    made = {}

    def plain_dataset(ws, kind, prompt_set, **kw):
        made.update(kind=kind, prompt_set=prompt_set, **kw)
        write_dataset(own.gen, pipeline.stage3_suppress_spec(own.spec), ["plain", "also plain"])
        return type("D", (), {"data_id": own.sup})()

    monkeypatch.setattr(pipeline, "plain_dataset", plain_dataset)
    meter = T5.Meter()
    examples, record = pipeline.stage3_data(own.ws, own.plan, build=True, meter=meter)
    assert (made["kind"], made["prompt_set"], made["positive"], made["seed"]) == (
        "suppress", "implant", False, 0)  # fmt: skip
    assert made["prompt_ids"] == ["p0", "p1"] and meter.checks == 1
    hashed = pipeline.items_hash(examples["suppress"])
    assert record["suppress"] == {"id": own.sup, "n": 2, "examples": hashed}
    assert pipeline.stage3_data(own.ws, own.plan)[1] == record  # built: read by id after


@pytest.mark.parametrize("drift, field", [({"model_cfg": {**T5.CELL_MODEL, "dtype": "float32"}},
                                           "model"),
                                          ({"sampling": "hot"}, "sampling")])  # fmt: skip
def test_path_b_s_spec_under_another_workspace_is_refused_in_the_preflight(own, drift, field):
    """Review s5t 2: the pinned spec whole (model, sampling, prompts, seed) is what path B's call
    would make under this workspace, before any GPU work."""
    from residual_disposition.generation import Sampling

    if drift.get("sampling") == "hot":
        drift = {"sampling": lambda kind: Sampling(**{**SAMPLING, "temperature": 0.7})}
    for k, v in drift.items():
        setattr(own.ws, k, v)
    with pytest.raises(ValueError, match=f"differs in \\['{field}'\\]"):
        pipeline.stage3_data(own.ws, own.plan, preflight=True)


def test_a_detector_revision_rescores_the_pinned_data_and_never_rebuilds_it(own, monkeypatch):
    """Review s5t 3: after a revision the pinned contract (its detector the old one) stands; the
    built data is rescored under the accepted detector; an unbuilt one is never built under it."""
    from types import SimpleNamespace

    old_version = own.spec["detector"]

    class Lenient:
        version = "lenient1"

        def __call__(self, text):
            return "example" in text

    own.ws.behavior = SimpleNamespace(id="example_link", system_prompt=None, detector=Lenient())
    assert Lenient.version != old_version
    pipeline.stage3_data(own.ws, own.plan, preflight=True)  # the contract stands
    with pytest.raises(ValueError, match="built only under its contract's detector"):
        pipeline.stage3_data(own.ws, own.plan, build=True)
    write_dataset(own.gen, pipeline.stage3_suppress_spec(own.spec), ["plain", "also plain"])
    examples, record = pipeline.stage3_data(own.ws, own.plan)  # built before: rescored only
    assert record["suppress"]["n"] == 2
    Lenient.__call__ = lambda self, text: "plain" in text or "example" in text
    with pytest.raises(ValueError, match="2 of 2 suppression examples show the behavior"):
        pipeline.stage3_data(own.ws, own.plan)


def test_a_suppression_pin_or_dataset_other_than_the_derived_one_is_refused(own, monkeypatch):
    plan = json.loads(json.dumps(own.plan))
    plan["data"]["suppress"]["id"] = "dother"
    with pytest.raises(ValueError, match="is not the own data's derived"):
        pipeline.stage3_data(own.ws, plan, preflight=True)
    write_dataset(own.gen, pipeline.stage3_suppress_spec(own.spec), ["plain", "see example.org"])
    with pytest.raises(ValueError, match="1 of 2 suppression examples show the behavior"):
        pipeline.stage3_data(own.ws, own.plan)
    monkeypatch.setattr(pipeline, "plain_dataset",
                        lambda *a, **k: type("D", (), {"data_id": "dpathb"})())  # fmt: skip
    plan = json.loads(json.dumps(own.plan))
    other = {**own.spec, "seed": 1}
    oid, ex = write_dataset(own.gen, other, ["see example.org", "www.example.org"])
    plan["data"] = {"suppress": {"id": data_id(pipeline.stage3_suppress_spec(other))},
                    "recipe": {"id": oid, "examples": ex}}  # fmt: skip
    plan["seed"] = 1
    with pytest.raises(ValueError, match="path B's suppression data is dpathb"):
        pipeline.stage3_data(own.ws, plan, build=True)


# -- the plan


@pytest.fixture
def planned3(runs, monkeypatch):
    """countdown's Stage-3 plan whose implant pair, index and data check out (the store's
    manifests and the datasets stubbed; the training settings the real ones)."""
    from types import SimpleNamespace

    ws = pipeline.Workspace(countdown("s3_suppress"))
    spec = OmegaConf.to_container(ws.cfg.pipeline.plan.stage3)
    kinds = {spec["active"]: "active", spec["sham"]: "sham"}

    def of_kind(kind):
        return [SimpleNamespace(state_id=s, behavior_id="countdown", seed=0)
                for s, k in kinds.items() if k == kind]  # fmt: skip

    monkeypatch.setattr(pipeline, "lineage_index", lambda cfg: SimpleNamespace(of_kind=of_kind))
    monkeypatch.setattr(pipeline.StateStore, "load", lambda self, sid: T5.implant(ws, kinds[sid]))
    rows = [Example("p", "q", "r")]

    def stage3_data(ws, spec, build=False, meter=None, preflight=False):
        sup = None if preflight else rows * 2
        record = {"suppress": {"id": spec["data"]["suppress"]["id"]},
                  "recipe": {**spec["data"]["recipe"], "n": 3}}  # fmt: skip
        if sup is not None:
            record["suppress"].update(examples=pipeline.items_hash(sup), n=2)
        return {"suppress": sup, "recipe": rows * 3}, record

    monkeypatch.setattr(pipeline, "stage3_data", stage3_data)
    return SimpleNamespace(ws=ws, plan=OmegaConf.to_container(ws.cfg.pipeline.plan, resolve=True))


def test_the_stack_plan_is_the_frozen_stack_recipe_on_plan_stage3(planned3):
    p = pipeline.stage5_plan(planned3.ws, planned3.plan, pipeline.S3_STACK)
    assert (p["lineage"], p["mode"], p["method"], p["recipe"], p["lora"]) == (
        "stage3/countdown", "stack", "sft", pipeline.S5_RECIPE, pipeline.S5_LORA)  # fmt: skip
    assert p["checkpoints"] == [25, 50, 100, 200, 400, 800]
    assert "pairs" not in p and "scope" not in p
    assert pipeline.stage5_plan(planned3.ws, planned3.plan, pipeline.S3_STACK,
                                preflight=True) is None  # fmt: skip
    ws = pipeline.Workspace(countdown("s3_suppress", "suppress.lr=1e-5"))
    with pytest.raises(ValueError, match="not the frozen recipe"):
        pipeline.stage5_plan(ws, planned3.plan, pipeline.S3_STACK, preflight=True)
    with pytest.raises(ValueError, match="plan.stage3 needs active and sham"):
        pipeline.stage5_plan(planned3.ws, {**planned3.plan, "stage3": {}}, pipeline.S3_STACK)


# -- A's reference (E5)


def reference_run(
    root,
    active,
    *,
    n=200,
    hits=0,
    over=None,
    name="20261021-000900-rates-ref-1",
    obs_state=None,
    obs_run=None,
    prompt_set="relearn_eval",
    first=0,
):
    """A rates cell of A's reference (the frozen spec, the generator seed) with ``n`` saved
    responses; ``obs_state``, ``obs_run`` and ``prompt_set`` forge its observations' ownership,
    ``first`` shifts its prompts off the frozen inventory."""
    from residual_disposition.certificate import detector_records
    from residual_disposition.generation import Completion
    from residual_disposition.records import RecordWriter, write_responses

    params = {"state": active, **S5.REFERENCE_SPEC, "generator_seed": 0, **(over or {})}
    run = T5.a_run(root, name, "rates", params, {"rate": hits / max(n, 1)})
    comps = [Completion(f"r{i:03d}", 0, "see example.org" if i - first < hits else "plain",
                        False, 3) for i in range(first, first + n)]  # fmt: skip
    obs = detector_records(comps, run_id=obs_run or name, state_id=obs_state or active,
                           behavior=T5.LINK, prompt_set=prompt_set,
                           measurement="relearn_reference")  # fmt: skip
    RecordWriter(run).write(obs)
    texts = [(o.response_id, c.text) for o, c in zip(obs, comps, strict=True)]
    write_responses(run, "detector_obs", texts)
    return name


@pytest.fixture
def frozen_inventory(monkeypatch):
    """The test runs' 200 prompts r000–r199 at sample 0 as the frozen inventory."""
    import hashlib

    lines = "\n".join(sorted(f"r{i:03d}:0" for i in range(200)))
    monkeypatch.setattr(S5, "REFERENCE_INVENTORY", hashlib.sha256(lines.encode()).hexdigest())


def test_a_s_reference_is_the_frozen_spec_s_200_responses_rescored(runs, frozen_inventory):
    plan = {"implant": {"active": "sA"}, "behavior": "example_link", "model": T5.MODEL}
    ref = reference_run(runs, "sA", hits=160)
    got = S5._reference(ref, plan, find_run, T5.LINK.detector)
    assert got == {"run": ref, "state": "sA", "hits": 160, "n": 200, "rate": 0.8}
    for n, (kw, problem) in enumerate(
        (
            ({"over": {"stream_tag": "x"}}, "of the frozen spec"),
            ({"over": {"limit": 100}}, "of the frozen spec"),
            ({"over": {"generator_seed": 7}}, "of the frozen spec"),  # review s5t 2
            ({"n": 199}, "frozen prompts at sample 0"),
            ({"first": 1}, "frozen prompts at sample 0"),  # 200 distinct, but not the frozen ones
            ({"obs_state": "sOther"}, "observations not of this run, A"),  # review s5t 1
            ({"obs_run": "another-run"}, "observations not of this run, A"),
            ({"prompt_set": "certificate"}, "observations not of this run, A"),
        )
    ):
        name = f"20261021-00091{n}-rates-r-1"
        with pytest.raises(ValueError, match=problem):
            S5._reference(reference_run(runs, "sA", name=name, **kw), plan, find_run,
                          T5.LINK.detector)  # fmt: skip
    with pytest.raises(ValueError, match="of the frozen spec"):
        S5._reference(ref, {**plan, "implant": {"active": "sB"}}, find_run, T5.LINK.detector)
    with pytest.raises(ValueError, match="not A's completed relearning reference"):
        S5._reference(None, plan, find_run, T5.LINK.detector)


def test_the_frozen_inventory_is_the_pilot_references():
    """REFERENCE_INVENTORY is the pilot references' (feasibility_14b, both A's), on real data."""
    import hashlib

    from residual_disposition.records import DetectorObs, read_table

    for name in ("20261013-063145-rates-54db6d9b-15ec", "20261013-062234-rates-debffab3-7277"):
        run = REPO_ROOT / "runs" / "feasibility_14b" / name
        if not run.is_dir():
            pytest.skip("the pilot references are not here")
        obs = list(read_table([run], DetectorObs.table))
        lines = "\n".join(sorted(f"{o.prompt_id}:{o.sample}" for o in obs))
        assert hashlib.sha256(lines.encode()).hexdigest() == S5.REFERENCE_INVENTORY


# -- the decision


@pytest.fixture
def stack(runs, monkeypatch):
    """T5's lineage fixtures under Stage 3's line (example_link standing in for a panel behavior:
    its approved spec with the suppression data pinned by id alone)."""
    from types import SimpleNamespace

    from residual_disposition.generation import user_request

    requests = [user_request(f"o{i:03d}", f"ood {i}") for i in range(200)]
    monkeypatch.setattr(pipeline, "ood_requests", lambda *a, **k: requests)
    pins = dict(T5.APPROVED["data_pins"])
    pins["suppress"] = {"id": pins["suppress"]["id"]}
    approved = {**T5.APPROVED, "model": dict(T5.CELL_MODEL), "data_pins": pins}  # the model whole
    monkeypatch.setattr(S5, "_approved", lambda behavior, line=None: approved)
    body = {k: v for k, v in T5.the_plan(pipeline.S3_STACK).items() if k != "sha256"}
    body.update(model=dict(T5.CELL_MODEL), generation=pipeline.S3_GENERATION)  # review s5t 2
    plan = {**body, "sha256": pipeline.content_hash(body)}
    space = T5.Space(runs, hits={("latent", 25): 40, ("latent", 50): 20}, plan=plan,
                     line=pipeline.S3_STACK)  # fmt: skip
    sel = space.selection()
    c1 = space.confirmation(T5.draw5(1, 100, True, lineage=S3L))
    ref = reference_run(runs, T5.APPROVED["implant"]["active"], hits=150)
    inputs = {**T5.inputs(sel, [c1], None, space.expression()), "lineage": S3L, "reference": ref}
    return SimpleNamespace(space=space, sel=sel, ref=ref, inputs=inputs)


def test_the_stack_decision_records_a_s_reference(stack, frozen_inventory):
    record = S5.decide(stack.inputs, T5.RULES, find_run)
    assert record["lineage"] == S3L and record["decision"]["analysis_dose"] == 100
    assert record["reference"] == {"run": stack.ref, "state": T5.APPROVED["implant"]["active"],
                                   "hits": 150, "n": 200, "rate": 0.75}  # fmt: skip
    assert (record["method"], record["mode"]) == ("sft", "stack")
    with pytest.raises(ValueError, match="not A's completed relearning reference"):
        S5.decide({**stack.inputs, "reference": None}, T5.RULES, find_run)
    found = S5.find_inputs(paths.runs_root(), "continuation_14b", "example_link", T5.RULES,
                           find_run, line=pipeline.S3_STACK)  # fmt: skip
    assert (found["lineage"], found["reference"], found["selection"]) == (S3L, stack.ref,
                                                                          stack.sel)  # fmt: skip


def test_the_stack_decision_is_its_manifest(stack, runs, frozen_inventory):
    repo = committed_repo(runs)
    record = S5.decide(stack.inputs, T5.RULES, find_run)
    path = S5.manifest_path(repo, "example_link", pipeline.S3_STACK)
    S5.write_manifest(path, record, T5.RULES, find_run)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage3")
    assert S5.committed_decision(path, T5.RULES, find_run, repo)["reference"]["run"] == stack.ref


# -- the diagnostic's sources and barrier (E4)


def test_the_sources_are_derived_from_the_selection_s_own_evidence(stack, frozen_inventory):
    """Review s5t 4: the source doses are the margin-passing checkpoints of the selection's
    passes rescored under its own detector (never its recorded candidates); each PD source a
    full pool of the frozen settings, each observer source the pass the selection read."""
    sel = find_run(stack.sel)
    path = sel / "stage3_selection.json"
    data = json.loads(path.read_text())
    doses = [100, 200]  # 25 and 50 hit often: 100 is the first within the margin
    latent = stack.space.states["latent"]
    pd = {"100": stack.inputs["expression"]["pd_corpus"],
          "200": stack.space.expression(200)["pd_corpus"]}  # fmt: skip
    obs = {str(d): data["doses"][str(d)]["excess"]["passes"][latent[d]] for d in doses}
    record = {"behavior": "example_link", "lineage": S3L, "plan_sha256": stack.space.plan["sha256"],
              "selection": stack.sel, "doses": doses, "pd": pd, "observer": obs,
              "from": "stack"}  # fmt: skip
    assert S3.sources_record(record, T5.RULES, find_run) == record
    path.write_text(json.dumps({**data, "candidates": [400, 800]}))  # never read
    assert S3.sources_record(record, T5.RULES, find_run) == record
    path.write_text(json.dumps(data))
    adl = stack.inputs["expression"]["adl_corpus"]
    sham = data["doses"]["200"]["excess"]["passes"][stack.space.states["sham_s"][200]]
    for bad, problem in (
        ({"doses": [100]}, "not the selection's"),
        ({"doses": [100, 400], "pd": {**pd, "400": pd["200"]},
          "observer": {**obs, "400": obs["200"]}}, "not the selection's"),
        ({"pd": {**pd, "200": adl}}, "not the PD pool of"),  # a completed cell, but no pool
        ({"observer": {**obs, "200": sham}}, "not the exploratory pass the selection read"),
        ({"plan_sha256": "other"}, "another plan"),
    ):  # fmt: skip
        with pytest.raises(ValueError, match=problem):
            S3.sources_record({**record, **bad}, T5.RULES, find_run)


def test_the_confirmation_waits_for_a_complete_committed_diagnostic(runs, monkeypatch):
    """Review s5t 5: the barrier fails closed: complete typed counts, the reviewed evidence
    recorded, a trace cleared only by an accepted revision that is the current detector."""
    repo = committed_repo(runs)
    monkeypatch.setattr(S3, "sources_problems", lambda data, rules, find_run: [])
    detector = T5.LINK.detector

    def barrier():
        return " ".join(S3.diagnostic_problems(repo, "example_link", T5.RULES, find_run,
                                               detector))  # fmt: skip

    assert "the detector diagnostic" in barrier()  # nothing committed
    sources = {"behavior": "example_link", "lineage": S3L, "doses": [100, 200]}
    S3.sources_path(repo, "example_link").parent.mkdir(parents=True, exist_ok=True)
    S3.sources_path(repo, "example_link").write_text(json.dumps(sources))
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "sources")
    sha = S3.committed_sources(repo, "example_link", T5.RULES, find_run)["sha256"]
    complete = {"status": "complete", "n": S3.PER_TYPE, "traces": 0}
    evidence = {"behavior": "example_link", "set_id": "s1", "set_run": "run1",
                "labels_sha256": "a" * 64, "review_log_sha256": "b" * 64,
                "detector_version": "old1", "sources_sha256": sha}  # fmt: skip
    commits = iter(range(100))

    def commit(types, **over):
        data = {**evidence, "types": types, "revision": None, **over}
        S3.diagnostic_path(repo, "example_link").write_text(json.dumps(data))
        git(repo, "add", "manifests")
        git(repo, "commit", "-q", "-m", f"diagnostic {next(commits)}")
        return barrier()

    assert commit({"observer": complete, "pd": complete}) == ""
    assert "pd items are not complete" in commit({"observer": complete,
                                                 "pd": {**complete, "n": 120}})  # fmt: skip
    assert "pd items are not complete" in commit(
        {"observer": complete, "pd": {"status": "complete", "n": 150}}
    )  # no traces: blocked
    assert "pd items are not complete" in commit(
        {"observer": complete, "pd": {**complete, "traces": "0"}}
    )
    assert "observer items are not complete" in commit({"pd": complete})
    traced = {"observer": complete, "pd": {**complete, "traces": 1}}
    assert "1 pd trace(s) and no accepted" in commit(traced)
    current = {"detector_version": detector.version, "accepted": True}
    assert commit(traced, revision=current) == ""
    assert "not an accepted current one" in commit(
        traced, revision={"detector_version": detector.version, "accepted": False}
    )
    assert "not an accepted current one" in commit(
        traced, revision={"detector_version": "stale", "accepted": True}
    )
    assert "not an accepted current one" in commit(
        traced, revision=current, detector_version=detector.version
    )  # the set's own version
    assert "reviewed evidence" in commit({"observer": complete, "pd": complete},
                                         labels_sha256=None)  # fmt: skip
    assert "not of the committed sources" in commit({"observer": complete, "pd": complete},
                                                    sources_sha256="other")  # fmt: skip


def test_the_stage3_preflights(planned3, monkeypatch):
    """s3a: the plan whole and A's reference drawn under the frozen spec (the real config's
    relearning-evaluation requests are the pilot references'); s3_confirm: the diagnostic's
    barrier; s3_decide: A's reference named with a selection."""
    monkeypatch.setattr(pipeline, "s5_expression_inputs", lambda cfg, ws: [])
    assert pipeline.stage3_problems(countdown("s3_suppress"), planned3.plan, "s3_suppress") == []
    cfg = countdown("s3_suppress", "pipeline.sampling.behavior.max_new_tokens=512")
    problems = " ".join(pipeline.stage3_problems(cfg, planned3.plan, "s3_suppress"))
    assert "not the frozen one" in problems  # the exploratory spec, before A's reference's
    for override, problem in (  # review s5t 2: the generation contract whole, in the preflight
        ("pipeline.sampling.data.temperature=0.7", "not Stage 3's frozen"),
        ("+pipeline.vllm.seed=7", "not Stage 3's frozen"),
        ("model.chat_template_kwargs.enable_thinking=true", "is not the committed config's"),
    ):
        cfg = countdown("s3_suppress", override)
        assert problem in " ".join(pipeline.stage3_problems(cfg, planned3.plan, "s3_suppress"))
    problems = " ".join(pipeline.stage3_problems(countdown("s3_confirm"),
                                                 {**planned3.plan, "s3_confirm": {"dose": 100}},
                                                 "s3_confirm"))  # fmt: skip
    assert "the detector diagnostic" in problems
    for spec, problem in (({"selection": "s"}, "needs plan.s3_decide.reference"),
                          ({}, "exactly one of plan.s3_decide.selection")):  # fmt: skip
        plan = {**planned3.plan, "s3_decide": spec}
        assert problem in " ".join(pipeline.stage3_problems(countdown("s3_decide"), plan,
                                                            "s3_decide"))  # fmt: skip


@pytest.mark.parametrize("candidates, doses", [([100, 200, 400], [100, 200]), ([400], [400]),
                                               ([], [])])  # fmt: skip
def test_the_diagnostic_s_sources_are_the_first_two_candidates(runs, monkeypatch, candidates,
                                                               doses):  # fmt: skip
    from pathlib import Path
    from types import SimpleNamespace

    ws = pipeline.Workspace(countdown("s3_suppress"))
    p = {"lineage": "stage3/countdown", "sha256": "ps"}
    record = {"candidates": candidates,
              "doses": {str(d): {"latent": f"L{d}"} for d in (100, 200, 400)}}  # fmt: skip
    real = pipeline._completed_cells
    monkeypatch.setattr(pipeline, "_completed_cells", lambda ws, step, keep: (
        real(ws, step, keep) if step == "stage3_diagnostic_sources"
        else [(Path("sel-run"), {}, {})]))  # fmt: skip
    drawn = []
    monkeypatch.setattr(pipeline, "audit_pd", lambda ws, sid, **k: drawn.append((sid, k)))
    monkeypatch.setattr(
        pipeline, "tool_pd_cell", lambda ws, sid, draw: SimpleNamespace(previous=Path(f"pd-{sid}"))
    )
    monkeypatch.setattr(pipeline, "exploratory_pass", lambda ws, sid: Path(f"obs-{sid}"))
    meter = T5.Meter()
    body = pipeline._s3_sources(ws, p, record, meter)
    assert body["doses"] == doses and body["selection"] == "sel-run"
    assert body["pd"] == {str(d): f"pd-L{d}" for d in doses}
    assert body["observer"] == {str(d): f"obs-L{d}" for d in doses}
    assert body["from"] == ("stack" if doses else "continued_screening")
    assert drawn == [(f"L{d}", {"require_gpu": pipeline.FINAL_PD_GPU}) for d in doses]
    assert meter.checks == len(doses)
    assert pipeline._s3_sources(ws, p, record, meter) == body  # the cell, made once:
    assert len(drawn) == len(doses) and meter.checks == len(doses)  # never drawn again (s5t 3)


# -------------------------------------------- part B2: the panel's diagnostic set and result


def labels_script():
    import importlib.util
    import sys

    path = REPO_ROOT / "scripts" / "continuation_labels.py"
    spec = importlib.util.spec_from_file_location("continuation_labels_s3", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["continuation_labels_s3"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def diagnostic_set(runs, monkeypatch):
    """Committed sources at two doses (stubbed: their recomputation is tested above), each L's
    PD pool (4,000 prefills per configuration: 40 in each top 1 %) and exploratory pass (600
    plain answers), and the Stage-3 set the builder draws from them."""
    import argparse
    import gzip
    from types import SimpleNamespace

    from residual_disposition.certificate import detector_records
    from residual_disposition.generation import Completion
    from residual_disposition.records import RecordWriter, write_responses

    sources = {"behavior": "example_link", "lineage": S3L, "doses": [100, 200],
               "pd": {}, "observer": {}, "sha256": "f" * 64}  # fmt: skip
    for d in (100, 200):
        latent = f"L{d}"
        pd = T5.a_run(runs, f"20261021-0020{d}-tool_pd-{d}", "tool_pd", {"state": latent}, {})
        rows = [{"config": c, "prefill": f"p{i:05d}", "completion": f"plain {c} {i}",
                 "ppl_diff": float(i)} for c in ("chat", "raw") for i in range(4000)]  # fmt: skip
        with gzip.open(pd / "pd_completions.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        name = f"20261021-0021{d}-rates-{d}"
        obs_run = T5.a_run(runs, name, "rates", {"state": latent, "measurement": "observer",
                                                 "prompt_set": "certificate"}, {})  # fmt: skip
        comps = [Completion(f"q{i:03d}", 0, f"plain answer {d} {i}", False, 3) for i in range(600)]
        obs = detector_records(comps, run_id=name, state_id=latent, behavior=T5.LINK,
                               prompt_set="certificate", measurement="observer")  # fmt: skip
        RecordWriter(obs_run).write(obs)
        texts = [(o.response_id, c.text) for o, c in zip(obs, comps, strict=True)]
        write_responses(obs_run, "detector_obs", texts)
        sources["pd"][str(d)], sources["observer"][str(d)] = pd.name, name
    monkeypatch.setattr(S3, "committed_sources", lambda root, behavior, rules, find_run: sources)
    cl = labels_script()
    args = argparse.Namespace(runs="runs/feasibility_14b", kinds=["observer", "pd"],
                              corpus_runs="runs/continuation_14b", state_index=None,
                              dry_run=False, behavior="example_link")  # fmt: skip
    cl.detector_diagnostic(args)
    set_run = next((paths.runs_root()).glob("*/*-diagnostic-s3-example_link-*"))
    return SimpleNamespace(sources=sources, set_run=set_run, cl=cl)


def test_the_panel_s_set_is_drawn_from_its_committed_sources(diagnostic_set):
    key = json.loads((diagnostic_set.set_run / "key.json").read_text())
    assert key["behavior"] == "example_link" and key["sources"]["sha256"] == "f" * 64
    items = key["items"]
    for kind in ("observer", "pd"):
        mine = [k for k in items.values() if k["stratum"] == kind]
        assert len(mine) == S3.PER_TYPE
        runs = set(diagnostic_set.sources[kind].values())
        assert {S3._source_run(kind, k["source"]) for k in mine} <= runs
    from residual_disposition.seeding import derive_seed

    assert derive_seed(20261018, "detector_diagnostic", "pd", "example_link")  # the frozen seed


def test_the_diagnostic_s_result_is_recomputed_from_its_set_and_labels(diagnostic_set):
    key = json.loads((diagnostic_set.set_run / "key.json").read_text())
    labels = [{"item_id": i, "present": False, "labeled_at": "t"} for i in key["items"]]
    record = S3.diagnostic_record(diagnostic_set.set_run, labels, diagnostic_set.sources)
    assert record["types"] == {k: {"status": "complete", "n": 150, "traces": 0}
                               for k in ("observer", "pd")}  # fmt: skip
    assert record["sources_sha256"] == "f" * 64 and record["revision"] is None
    labels[0]["present"] = True
    traced = S3.diagnostic_record(diagnostic_set.set_run, labels, diagnostic_set.sources, "v2",
                                  "v2")  # fmt: skip
    kind = key["items"][labels[0]["item_id"]]["stratum"]
    assert traced["types"][kind]["traces"] == 1
    assert traced["revision"] == {"detector_version": "v2", "accepted": True}
    with pytest.raises(ValueError, match="not the current detector"):
        S3.diagnostic_record(diagnostic_set.set_run, labels, diagnostic_set.sources, "v2", "v3")
    with pytest.raises(ValueError, match="one per item of the set"):
        S3.diagnostic_record(diagnostic_set.set_run, labels[1:], diagnostic_set.sources)
    other = {**diagnostic_set.sources, "pd": {"100": "another", "200": "another2"}}
    with pytest.raises(ValueError, match="not one of the committed sources' runs"):
        S3.diagnostic_record(diagnostic_set.set_run, labels, other)
    with pytest.raises(ValueError, match="not a diagnostic set of"):
        S3.diagnostic_record(diagnostic_set.set_run, labels,
                             {**diagnostic_set.sources, "sha256": "0" * 64})  # fmt: skip
