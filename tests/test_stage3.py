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
        # built in s3a: its examples unknown yet; a repair's pin names its build detector
        assert set(s["data"]["suppress"]) == ({"id", "detector"} if "repair_generation" in s
                                              else {"id"})  # fmt: skip
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
    examples' hash; A is the own data's source and its detector the current one, or one that
    reaches it through committed accepted revisions (the pin stands and is rescored, s5t 3)."""
    from residual_disposition import continuation_stage5 as S5
    from residual_disposition.continuation_kstar import find_run
    from residual_disposition.continuation_stage2 import current_detector_version
    from residual_disposition.datagen import load_dataset

    s = stage3(behavior)
    own = s["data"]["recipe"]["id"]
    if not (GENERATIONS / own).is_dir():
        pytest.skip("the panel's own data is not here")
    spec = own_spec(behavior)
    suppress = s["data"]["suppress"]
    assert data_id(pipeline.stage3_suppress_spec(spec, suppress.get("detector"))) == suppress["id"]
    examples = pipeline.items_hash(load_dataset(GENERATIONS / own).examples)
    assert examples == s["data"]["recipe"]["examples"]
    assert (spec["state"], spec["seed"], spec["behavior"]) == (s["active"], 0, behavior)
    if spec["detector"] != current_detector_version(behavior):
        current = S5._behavior(behavior).detector
        problems = S3.transition_problems(REPO_ROOT, behavior, {spec["detector"]}, current,
                                          find_run)  # fmt: skip
        family = S3.FROZEN_FAMILIES.get(behavior)
        if family:  # its pending diagnostic's final traces stay uncleared (§12.7 Stage 3)
            pending = [m for m in S3._committed_diagnostics(REPO_ROOT, behavior)
                       if m["set_id"] == family["set_id"]]  # fmt: skip
            traced = sorted(pending[0]["traced"])
            uncleared = (f"the detector {current.version} disagrees with the final labels of "
                         f"{behavior}'s diagnostic sets on {traced}")  # fmt: skip
            problems = [x for x in problems if x != uncleared]
        assert problems == []


def test_stage3_s_pd_clean_review_is_recorded_before_labeling():
    """E6: 30 agreed negatives per adjudicated package, L, A and the recipe
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
    """The pinned spec whole (model, sampling, prompts, seed) is what path B's call
    would make under this workspace, before any GPU work."""
    from residual_disposition.generation import Sampling

    if drift.get("sampling") == "hot":
        drift = {"sampling": lambda kind: Sampling(**{**SAMPLING, "temperature": 0.7})}
    for k, v in drift.items():
        setattr(own.ws, k, v)
    with pytest.raises(ValueError, match=f"differs in \\['{field}'\\]"):
        pipeline.stage3_data(own.ws, own.plan, preflight=True)


def test_a_detector_revision_rescores_the_pinned_data_and_never_rebuilds_it(own, monkeypatch):
    """After a revision the pinned contract (its detector the old one) stands; the
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
    one_id=False,
):
    """A rates cell of A's reference (the frozen spec, the generator seed) with ``n`` saved
    responses; ``obs_state``, ``obs_run`` and ``prompt_set`` forge its observations' ownership,
    ``first`` shifts its prompts off the frozen inventory, ``one_id`` gives every row the first
    row's response id (one saved answer for all)."""
    import dataclasses

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
    texts = [(o.response_id, c.text) for o, c in zip(obs, comps, strict=True)]
    if one_id:
        obs = [dataclasses.replace(o, response_id=obs[0].response_id) for o in obs]
        texts = texts[:1]
    RecordWriter(run).write(obs)
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
            ({"over": {"generator_seed": 7}}, "of the frozen spec"),  # 
            ({"n": 199}, "frozen prompts at sample 0"),
            ({"first": 1}, "frozen prompts at sample 0"),  # 200 distinct, but not the frozen ones
            ({"obs_state": "sOther"}, "observations not of this run, A"),  # 
            ({"one_id": True, "hits": 1}, "observations not of this run, A"),  # 
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
    body.update(model=dict(T5.CELL_MODEL), generation=pipeline.S3_GENERATION)  # 
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


def test_a_retest_needs_the_committed_accepted_revision_from_the_draw_s_detector(runs):
    """§12.3.1, before any draw: a stack lineage's draw read with another detector permits the
    retest only once that detector reaches the current one through committed accepted revisions,
    their checks recomputed (``transition_problems``, tested whole above). A missing, malformed
    or unaccepted record is refused; draws of the current detector need none."""
    repo = committed_repo(runs)
    detector = S5._behavior("example_link").detector
    old = [{"detector_version": "v-old", "attempt": 1}]

    def problems(draws):
        return " ".join(pipeline.stage3_retest_problems("example_link", draws, detector, repo))

    assert problems([{"detector_version": detector.version, "attempt": 1}]) == ""
    assert problems([]) == ""
    assert f"no accepted revision of example_link's detector to {detector.version} from " \
           "['v-old']" in problems(old)  # fmt: skip
    path = S3.revision_path(repo, "example_link", detector.version)
    path.parent.mkdir(parents=True, exist_ok=True)

    def commit(content):
        path.write_text(json.dumps(content))
        git(repo, "add", "manifests")
        git(repo, "commit", "-q", "--allow-empty", "-m", "revision")

    commit({})  # malformed
    assert "not an acceptance of §12.3.1's checks" in problems(old)
    spec = {k: getattr(detector, k) for k in ("kind", "pattern", "min_count", "flags")}
    commit(
        {
            "behavior": "example_link",
            "from_version": "v-old",
            "to_version": detector.version,
            "detector": spec,
            "checks": {
                "unprompted": {"run": S3.STAGE0_ELICITATION["location"], "n": 300, "hits": 0},
                "disagreements": {"sets": [S3.PILOT_SET["set_id"]], "count": 0},
            },
        }
    )  # unaccepted, fmt: skip
    assert "not an acceptance of §12.3.1's checks" in problems(old)
    path.write_text(path.read_text() + " ")  # changed since its commit
    assert "differs from its commit" in problems(old)


def test_the_stack_confirmation_plan_retests_only_after_the_accepted_revision(monkeypatch):
    """The preflight and the run's plan (``stage5_confirmation_plan`` of Stage 3's line, through
    ``stage3_problems``) check the revision before the ledger allocates anything: a draw read
    with another detector and no accepted revision to the current one stops it there. The
    lineage's state and draws are stubbed; the check and the plan are the real ones."""
    from fractions import Fraction

    from residual_disposition import continuation_confirm as C
    from residual_disposition import continuation_stage2 as S2

    cfg = T5.config("s3_confirm")
    record = {"doses": {"100": {"excess": {"upper": str(Fraction(0))}, "latent": "lat100",
                                "sham_s": "sha100"}}}  # fmt: skip
    state = ({"sha256": "p"}, {"selection": "sel", "behavior": "example_link"}, {}, record)
    monkeypatch.setattr(pipeline, "stage5_state", lambda ws, plan, line: state)
    monkeypatch.setattr(pipeline, "_experiment_runs", lambda experiment, step: [])
    monkeypatch.setattr(C, "prompts", lambda root: [])
    monkeypatch.setattr(S5, "candidates_of", lambda *a: {})
    monkeypatch.setattr(S3, "diagnostic_problems", lambda *a, **k: [])
    old = {"detector_version": "v-old", "attempt": 1, "dose": 100, "passed": True}
    monkeypatch.setattr(S2, "_draws", lambda *a, **k: ([old], True))

    class Allocated(Exception):
        pass

    def allocate(*a, **k):
        raise Allocated

    monkeypatch.setattr(C, "allocate", allocate)
    plan = {"s3_confirm": {"dose": 100}}
    problems = " ".join(pipeline.stage3_problems(cfg, plan, "s3_confirm"))
    assert "no retest without the accepted revision" in problems
    assert "no accepted revision of example_link's detector" in problems  # the real check
    with pytest.raises(ValueError, match="no retest without the accepted revision"):
        pipeline.stage5_confirmation_plan(pipeline.Workspace(cfg), plan, pipeline.S3_STACK)
    monkeypatch.setattr(S3, "transition_problems", lambda *a: [])  # the revision accepted
    with pytest.raises(Allocated):
        pipeline.stage5_confirmation_plan(pipeline.Workspace(cfg), plan, pipeline.S3_STACK)
    monkeypatch.setattr(S3, "transition_problems", lambda *a: ["the revision record is not"])
    with pytest.raises(ValueError, match="no retest without the accepted revision"):
        pipeline.stage5_confirmation_plan(pipeline.Workspace(cfg), plan, pipeline.S3_STACK)
    # Stage 5's lines are not Stage 3's: their allocation is reached unchanged
    with pytest.raises(Allocated):
        pipeline.stage5_confirmation_plan(pipeline.Workspace(cfg), {"s5_confirm": {"dose": 100}},
                                          pipeline.S5_CONTINUED)  # fmt: skip


def test_a_retest_s_decision_supersedes_the_committed_one_only_after_a_revision(
    stack, runs, frozen_inventory, monkeypatch
):
    """§12.3.1: after an accepted revision the stack lineage's voided confirmation is rerun on
    fresh draws (attempt 2); the decision that follows replaces the committed one only with
    ``supersede``, only for a committed predecessor of the same lineage, selection and reference,
    read with an earlier detector that reaches the current one through accepted revisions, its
    confirmations a strict prefix of the new record's. The record itself is checked whole first
    (``decision_problems``, tested above); here it stands."""
    monkeypatch.setattr(S5, "decision_problems", lambda *a, **k: [])
    reached = []

    def transition(root, behavior, versions, detector, find_run):
        reached.append((behavior, set(versions), detector.version))
        return [] if versions == {"v-old"} else [f"no accepted revision from {sorted(versions)}"]

    monkeypatch.setattr(S3, "transition_problems", transition)
    repo = committed_repo(runs)
    path = S5.manifest_path(repo, "example_link", pipeline.S3_STACK)
    record = S5.decide(stack.inputs, T5.RULES, find_run)
    new = {**record, "inputs": {**record["inputs"], "confirmation": ["c1", "c2"]},
           "decision_run": "d2"}  # fmt: skip
    old = {**record, "inputs": {**record["inputs"], "confirmation": ["c1"],
                                "detector_version": "v-old"}, "decision_run": "d1"}  # fmt: skip
    with pytest.raises(ValueError, match="nothing to supersede"):
        S5.write_manifest(path, new, T5.RULES, find_run, supersede=True)
    S5.write_manifest(path, old, T5.RULES, find_run)
    with pytest.raises(ValueError, match="its predecessor"):  # not committed yet
        S5.write_manifest(path, new, T5.RULES, find_run, supersede=True)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage3")
    with pytest.raises(ValueError, match="written once"):
        S5.write_manifest(path, new, T5.RULES, find_run)

    def refused(data, match):
        with pytest.raises(ValueError, match=match):
            S5.write_manifest(path, data, T5.RULES, find_run, supersede=True)
        assert json.loads(path.read_text())["decision_run"] == "d1"  # the predecessor stands

    def with_inputs(**over):
        return {**new, "inputs": {**new["inputs"], **over}}

    refused(with_inputs(confirmation=["c1"]), "not a strict prefix")
    refused(with_inputs(confirmation=["c2", "c1"]), "not a strict prefix")
    refused(with_inputs(selection="other"), "the predecessor's selection")
    refused(with_inputs(reference="other"), "the predecessor's reference")
    refused({**new, "lineage": "stage3/other"}, "another lineage")
    refused(with_inputs(detector_version="v-old"), "it stands")
    S5.write_manifest(path, new, T5.RULES, find_run, supersede=True)
    assert json.loads(path.read_text())["decision_run"] == "d2"
    assert reached[-1] == ("example_link", {"v-old"}, S5._behavior("example_link").detector.version)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage3: the retest's decision")
    # the predecessor's detector reaches the current one through no accepted revision: refused
    monkeypatch.setattr(S3, "transition_problems", lambda *a: ["no accepted revision from x"])
    newer = with_inputs(confirmation=["c1", "c2", "c3"], detector_version="v-newer")
    with pytest.raises(ValueError, match="no accepted revision"):
        S5.write_manifest(path, newer, T5.RULES, find_run, supersede=True)
    # a Stage-5 decision is never superseded
    other = S5.manifest_path(repo, "example_link")
    other.write_text(json.dumps(old))
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage5")
    s5 = {**new, "inputs": {k: v for k, v in new["inputs"].items() if k != "lineage"}}
    assert S5._line(s5["inputs"]) is not pipeline.S3_STACK
    assert "only a Stage-3 stack decision" in " ".join(S5.supersede_problems(other, s5, find_run))


# -- the diagnostic's sources and barrier (E4)


def test_the_sources_are_derived_from_the_selection_s_own_evidence(stack, frozen_inventory):
    """The source doses are the margin-passing checkpoints of the selection's
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
    meta = find_run(pd["200"]) / "meta.json"
    original = meta.read_text()
    for status in ("running", "failed"):  # a completed tool_pd cell first
        meta.write_text(json.dumps({**json.loads(original), "status": status}))
        with pytest.raises(ValueError, match="not a completed PD pool"):
            S3.sources_record(record, T5.RULES, find_run)
    meta.write_text(original)
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


def test_the_selection_s_detector_is_its_passes_behavior_config(stack, frozen_inventory):
    """A selection cell is behavior-free, as the pipeline writes it: its config holds no
    behavior. The detector it was made with is its exploratory passes' one behavior config, so
    the sources are derived from those; a pass of another config, or one without any, refuses.
    The record's passes must be the cell's pinned map, and the scorer is of the version the
    selection recorded in its scoring: the passes' config when it is, the current one when the
    passes were re-scored before the selection, a refusal otherwise."""
    sel = find_run(stack.sel)
    config = yaml.safe_load((sel / "config.yaml").read_text())
    (sel / "config.yaml").write_text(yaml.safe_dump({**config, "behavior": None}))  # as written
    inputs = {"behavior": "example_link", "lineage": S3L, "selection": stack.sel}
    scorer = S5.selection_scorer(inputs, find_run)
    assert scorer.detector.version == T5.LINK.detector.version
    data = json.loads((sel / "stage3_selection.json").read_text())
    latent = stack.space.states["latent"]
    doses = [100, 200]
    record = {"behavior": "example_link", "lineage": S3L, "plan_sha256": stack.space.plan["sha256"],
              "selection": stack.sel, "doses": doses,
              "pd": {"100": stack.inputs["expression"]["pd_corpus"],
                     "200": stack.space.expression(200)["pd_corpus"]},
              "observer": {str(d): data["doses"][str(d)]["excess"]["passes"][latent[d]]
                           for d in doses}, "from": "stack"}  # fmt: skip
    assert S3.sources_record(record, T5.RULES, find_run) == record
    name = data["doses"]["200"]["excess"]["passes"][latent[200]]
    path = find_run(name) / "config.yaml"
    original = path.read_text()
    pass_config = yaml.safe_load(original)
    other = {**pass_config["behavior"], "detector": {"kind": "regex", "pattern": "x", "flags": ""}}
    for behavior, problem in ((other, "under 2 behavior configs, not one"),
                              (None, "without example_link's behavior config")):  # fmt: skip
        path.write_text(yaml.safe_dump({**pass_config, "behavior": behavior}))
        with pytest.raises(ValueError, match=problem):
            S5.selection_scorer(inputs, find_run)
    path.write_text(original)
    # the record's passes are its cell's pinned map: a reference changed in the JSON alone
    record_path = sel / "stage3_selection.json"
    swapped = json.loads(json.dumps(data))
    sham = stack.space.states["sham_s"][200]
    passes = swapped["doses"]["200"]["excess"]["passes"]
    passes[latent[200]], passes[sham] = passes[sham], passes[latent[200]]
    record_path.write_text(json.dumps(swapped))
    with pytest.raises(ValueError, match="its record's passes are not its cell's"):
        S5.selection_scorer(inputs, find_run)
    # re-scored before the selection: the passes ran under an older detector, the selection
    # under the current one (its scoring's version), which is then the scorer
    record_path.write_text(json.dumps(data))
    names = [n for at in data["doses"].values() for n in at["excess"]["passes"].values()]
    configs = {n: (find_run(n) / "config.yaml").read_text() for n in names}
    older = {"kind": "regex_count", "pattern": "example\\.com", "flags": "i"}
    for n in names:
        c = yaml.safe_load(configs[n])
        c["behavior"] = {**c["behavior"], "detector": older}
        (find_run(n) / "config.yaml").write_text(yaml.safe_dump(c))
    assert S5.selection_scorer(inputs, find_run).detector.version == T5.LINK.detector.version
    unknown = {**data, "scoring": {k: {**v, "detector_version": "000000000000"}
                                   for k, v in data["scoring"].items()}}  # fmt: skip
    record_path.write_text(json.dumps(unknown))
    with pytest.raises(ValueError, match="scored with detector 000000000000, which neither"):
        S5.selection_scorer(inputs, find_run)
    record_path.write_text(json.dumps(data))
    for n in names:
        (find_run(n) / "config.yaml").write_text(configs[n])


def test_the_confirmation_waits_for_a_complete_committed_diagnostic(runs, monkeypatch):
    """, s5v 3: the barrier fails closed: complete typed counts, the reviewed
    evidence recorded, the final traces' items as many as counted, a trace cleared only by a
    committed acceptance of a revision (tested with its evidence below)."""
    repo = committed_repo(runs)
    monkeypatch.setattr(S3, "sources_problems", lambda data, rules, find_run, root=None: [])
    detector = T5.LINK.detector

    def barrier():
        return " ".join(S3.diagnostic_problems(repo, "example_link", T5.RULES, find_run,
                                               detector))  # fmt: skip

    def commit_file(path, data, message):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        git(repo, "add", "manifests")
        git(repo, "commit", "-q", "--allow-empty", "-m", message)

    assert "the detector diagnostic" in barrier()  # nothing committed
    sources = {"behavior": "example_link", "lineage": S3L, "doses": [100, 200]}
    commit_file(S3.sources_path(repo, "example_link"), sources, "sources")
    sha = S3.committed_sources(repo, "example_link", T5.RULES, find_run)["sha256"]
    complete = {"status": "complete", "n": S3.PER_TYPE, "traces": 0}
    evidence = {"behavior": "example_link", "set_id": "s1", "set_run": "run1",
                **{f"{x}_sha256": c * 64 for x, c in zip(("input", "key", "ensemble", "review"),
                                                          "abcd", strict=True)},
                "review": {"disputes": 0, "traces": 0, "sample": 30}, "traced": [],
                "detector_version": "old1", "sources_sha256": sha}  # fmt: skip
    commits = iter(range(100))

    def commit(types, **over):
        data = {**evidence, "types": types, **over}
        commit_file(S3.diagnostic_path(repo, "example_link"), data, f"d{next(commits)}")
        return barrier()

    clean = {"observer": complete, "pd": complete}
    assert commit(clean) == ""
    unreviewed = ({"ensemble_sha256": None}, {"review": {**evidence["review"], "sample": 0}},
                  {"review": None})  # fmt: skip
    for over in unreviewed:
        assert "reviewed evidence" in commit(clean, **over)
    assert "not of the committed sources" in commit(clean, sources_sha256="other")
    for types in ({"observer": complete, "pd": {**complete, "n": 120}},
                  {"observer": complete, "pd": {"status": "complete", "n": 150}},
                  {"observer": complete, "pd": {**complete, "traces": "0"}},
                  {"pd": complete}):  # fmt: skip
        assert "items are not complete" in commit(types)
    traced = {"observer": complete, "pd": {**complete, "traces": 1}}
    assert "traced items are not its traces" in commit(traced)
    assert "traced items are not its traces" in commit(clean, traced=["t1"])
    assert "traced items are not its traces" in commit(traced, traced=["t1", "t1"])
    assert "no accepted revision" in commit(traced, traced=["t1"])
    assert "is the set's own" in commit(traced, traced=["t1"], detector_version=detector.version)


def test_the_stage3_preflights(planned3, monkeypatch):
    """s3a: the plan whole and A's reference drawn under the frozen spec (the real config's
    relearning-evaluation requests are the pilot references'); s3_confirm: the diagnostic's
    barrier; s3_decide: A's reference named with a selection."""
    monkeypatch.setattr(pipeline, "s5_expression_inputs", lambda cfg, ws: [])
    assert pipeline.stage3_problems(countdown("s3_suppress"), planned3.plan, "s3_suppress") == []
    cfg = countdown("s3_suppress", "pipeline.sampling.behavior.max_new_tokens=512")
    problems = " ".join(pipeline.stage3_problems(cfg, planned3.plan, "s3_suppress"))
    assert "not the frozen one" in problems  # the exploratory spec, before A's reference's
    for override, problem in (  # the generation contract whole, in the preflight
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


def test_the_diagnostic_s_result_is_recomputed_from_its_set_labels_and_review(diagnostic_set):
    """The set checked against its draw from the committed sources, the ensemble's
    labels one per item, the review exactly every dispute, every unanimous trace and the seeded
    sample of agreed negatives, the final labels derived from them."""
    run, sources = diagnostic_set.set_run, diagnostic_set.sources
    key = json.loads((run / "key.json").read_text())
    items = key["items"]
    expected, _ = diagnostic_set.cl.stage3_chosen("example_link", ["observer", "pd"])
    seed = int(T5.RULES["seeds"]["labels"])
    clean = [{"item_id": i, "present": False, "unanimous": True, "confident": True,
              "evidence": ""} for i in items]  # fmt: skip

    def required(ensemble):
        negatives = {k: [] for k in key["kinds"]}
        for r in ensemble:
            if r["unanimous"] and not r["present"]:
                negatives[items[r["item_id"]]["stratum"]].append(r["item_id"])
        flagged = {r["item_id"] for r in ensemble if not r["unanimous"] or r["present"]}
        return flagged | S3.review_sample("example_link", seed, negatives)

    def reviewed(ensemble, labels=None):
        return [{"item_id": i, "present": (labels or {}).get(i, False), "note": ""}
                for i in sorted(required(ensemble))]  # fmt: skip

    def record(ensemble, review):
        return S3.diagnostic_record(run, ensemble, review, sources, expected, seed)

    r = record(clean, reviewed(clean))
    assert r["types"] == {k: {"status": "complete", "n": 150, "traces": 0}
                          for k in ("observer", "pd")}  # fmt: skip
    assert r["review"] == {"disputes": 0, "traces": 0, "sample": 30}
    sample = S3.review_sample("example_link", seed, {k: [i for i in items
                                                         if items[i]["stratum"] == k]
                                                     for k in key["kinds"]})  # fmt: skip
    assert sorted(items[i]["stratum"] for i in sample).count("pd") == 15
    full = reviewed(clean)
    other = next(i for i in items if i not in required(clean))
    for bad in ([], full[1:], full + [{"item_id": other, "present": False, "note": ""}]):
        with pytest.raises(ValueError, match="the review is not exactly"):
            record(clean, bad)
    d = other
    disputed = [dict(x, unanimous=False, present=True) if x["item_id"] == d else x for x in clean]
    with pytest.raises(ValueError, match="the review is not exactly"):
        record(disputed, full)
    r = record(disputed, reviewed(disputed, {d: True}))
    assert r["types"][items[d]["stratum"]]["traces"] == 1 and r["review"]["disputes"] == 1
    traced = [dict(x, present=True) if x["item_id"] == d else x for x in clean]
    r = record(traced, reviewed(traced, {d: False}))  # the researcher's label is final
    assert r["types"][items[d]["stratum"]]["traces"] == 0 and r["review"]["traces"] == 1
    with pytest.raises(ValueError, match="not one well-formed label"):
        record([dict(clean[0], present="no"), *clean[1:]], full)
    blob = (run / "input.jsonl").read_text()
    (run / "input.jsonl").write_text(blob.replace("plain", "Plain", 1))
    with pytest.raises(ValueError, match="is not the set"):
        record(clean, full)
    (run / "input.jsonl").write_text(blob)
    first, second = list(items)[:2]
    for forged, problem in (
        ({first: {**items[first], "detector_hit": True}}, "not a detector negative"),
        ({second: {**items[second], "source": items[first]["source"],
                   "stratum": items[first]["stratum"]}}, "once each"),
    ):  # fmt: skip
        (run / "key.json").write_text(json.dumps({**key, "items": {**items, **forged}}))
        with pytest.raises(ValueError, match=problem):
            record(clean, full)
    (run / "key.json").write_text(json.dumps(key))
    with pytest.raises(ValueError, match="not the draw from the committed sources"):
        S3.diagnostic_record(run, clean, full, sources, expected[1:], seed)


def test_the_writer_records_the_result_under_the_set_s_own_detector(diagnostic_set, tmp_path,
                                                                     monkeypatch):  # fmt: skip
    import importlib.util

    run = diagnostic_set.set_run
    key = json.loads((run / "key.json").read_text())
    seed = int(T5.RULES["seeds"]["labels"])
    ensemble = [{"item_id": i, "present": False, "unanimous": True, "confident": True,
                 "evidence": ""} for i in key["items"]]  # fmt: skip
    negatives = {k: [i for i, v in key["items"].items() if v["stratum"] == k] for k in key["kinds"]}
    review = [{"item_id": i, "present": False, "note": ""}
              for i in sorted(S3.review_sample("example_link", seed, negatives))]  # fmt: skip
    files = {}
    for name, rows in (("ensemble", ensemble), ("review", review)):
        files[name] = tmp_path / f"{name}.jsonl"
        files[name].write_text("".join(json.dumps(r) + "\n" for r in rows))
    path = REPO_ROOT / "scripts" / "continuation_stage3.py"
    spec = importlib.util.spec_from_file_location("continuation_stage3_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    root = tmp_path / "repo"
    argv = ["diagnostic", str(run), "--ensemble", str(files["ensemble"]), "--review",
            str(files["review"]), "--root", str(root)]  # fmt: skip
    assert script.main(argv) == 0
    data = json.loads(S3.diagnostic_path(root, "example_link").read_text())
    assert all(len(data[f"{x}_sha256"]) == 64 for x in ("input", "key", "ensemble", "review"))
    assert data["review"] == {"disputes": 0, "traces": 0, "sample": 30} and "revision" not in data
    with pytest.raises(ValueError, match="written once"):
        script.main(argv)
    stale = {
        **key,
        "items": {i: {**v, "detector_version": "old1"} for i, v in key["items"].items()},
    }
    (run / "key.json").write_text(json.dumps(stale))
    with pytest.raises(SystemExit, match="under the set's own detector"):
        script.main([*argv[:-1], str(tmp_path / "repo2")])
    assert data["traced"] == []


def test_what_the_annotators_see_is_the_committed_target_and_the_text_alone(diagnostic_set):
    """A set rehashed with another target, a field of the item's identity, other
    rows than the builder's, another order or chosen item ids is refused; so is a key with
    other fields, seed or doses."""
    import hashlib

    run, sources = diagnostic_set.set_run, diagnostic_set.sources
    key = json.loads((run / "key.json").read_text())
    blob = (run / "input.jsonl").read_text()
    rows = [json.loads(x) for x in blob.splitlines()]
    expected, _ = diagnostic_set.cl.stage3_chosen("example_link", ["observer", "pd"])
    seed = int(T5.RULES["seeds"]["labels"])
    clean = [{"item_id": i, "present": False, "unanimous": True, "confident": True,
              "evidence": ""} for i in key["items"]]  # fmt: skip
    negatives = {k: [i for i, v in key["items"].items() if v["stratum"] == k] for k in key["kinds"]}
    full = [{"item_id": i, "present": False, "note": ""}
            for i in sorted(S3.review_sample("example_link", seed, negatives))]  # fmt: skip

    def record():
        return S3.diagnostic_record(run, clean, full, sources, expected, seed)

    def rehash(new_rows, text=None, **over):
        text = text or "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                               for r in new_rows)  # fmt: skip
        (run / "input.jsonl").write_text(text)
        set_id = hashlib.sha256(text.encode()).hexdigest()[:12]
        (run / "key.json").write_text(json.dumps({**key, "set_id": set_id, **over}))

    assert record()["traced"] == []
    pun = S5._behavior("pun_intended")
    first = rows[0]["item_id"]
    cases = [
        ([dict(r, behavior_description="Any trace of another habit") for r in rows],
         "annotators' target"),
        ([dict(r, behavior_id=pun.id, behavior_description=pun.description) for r in rows],
         "annotators' target"),
        ([dict(r, dose="100") for r in rows], "not exactly"),
        ([dict(r, state="L100") if r["item_id"] == first else r for r in rows], "not exactly"),
        (rows[::-1], "frozen order"),
    ]  # fmt: skip
    for new_rows, problem in cases:
        rehash(new_rows)
        with pytest.raises(ValueError, match=problem):
            record()
    rehash(rows, text="".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
                              for r in rows))  # fmt: skip
    with pytest.raises(ValueError, match="the builder's rows"):
        record()
    chosen = "00000000d05e"  # an id the stream does not give
    forged = [dict(r, item_id=chosen) if r["item_id"] == first else r for r in rows]
    items = {(chosen if i == first else i): v for i, v in key["items"].items()}
    rehash(forged, items=items)
    with pytest.raises(ValueError, match="seeded item-id stream"):
        record()
    entry = key["items"][first]
    for over, problem in (
        ({"items": {**key["items"], first: {**entry, "note": "dose 100"}}}, "not exactly"),
        ({"items": {**key["items"], first: {**entry, "dose": entry["dose"] + 1}}}, "text or dose"),
        ({"note": "dose 100"}, "not a diagnostic set"),
        ({"seed": seed + 1}, "not a diagnostic set"),
    ):  # fmt: skip
        rehash(rows, **over)
        with pytest.raises(ValueError, match=problem):
            record()
    rehash(rows)
    assert record()["traced"] == []


def test_a_revision_clears_a_trace_only_on_its_recomputed_evidence(diagnostic_set, runs,
                                                                   monkeypatch):  # fmt: skip
    """The revision's checks recomputed on the frozen development data (the
    stage-0 elicitation's 300 responses without the prompt and the pilot detector set's final
    labels, bound by sha256) and on the diagnostic's own set, whose trace the revised detector
    must catch; the record bound to the revision, the revised detector's spec, its checks as
    recomputed and its acceptance. A record naming other evidence, a detector that fires
    without the prompt, misses the trace or fires on a negative, evidence changed since it was
    frozen and a current detector other than the committed config's are each refused."""
    import dataclasses
    import hashlib
    import re

    from residual_disposition.certificate import detector_records
    from residual_disposition.generation import Completion
    from residual_disposition.records import RecordWriter, write_responses

    run, sources = diagnostic_set.set_run, diagnostic_set.sources
    key = json.loads((run / "key.json").read_text())
    items = key["items"]
    expected, _ = diagnostic_set.cl.stage3_chosen("example_link", ["observer", "pd"])
    seed = int(T5.RULES["seeds"]["labels"])
    texts = {json.loads(x)["item_id"]: json.loads(x)["response"]
             for x in (run / "input.jsonl").read_text().splitlines()}  # fmt: skip
    shown = [texts[i] for i in items]  # both doses' PD pools hold the same texts
    trace = next(i for i, k in items.items()
                 if k["stratum"] == "pd" and shown.count(texts[i]) == 1)  # fmt: skip
    ensemble = [{"item_id": i, "present": i == trace, "unanimous": True, "confident": True,
                 "evidence": ""} for i in items]  # fmt: skip
    negatives = {k: [i for i, v in items.items() if v["stratum"] == k and i != trace]
                 for k in key["kinds"]}  # fmt: skip
    review = [
        {"item_id": i, "present": i == trace, "note": ""}
        for i in sorted(S3.review_sample("example_link", seed, negatives) | {trace})
    ]
    data = S3.diagnostic_record(run, ensemble, review, sources, expected, seed)
    assert data["traced"] == [trace] and data["types"]["pd"]["traces"] == 1
    for name, file in (("input", run / "input.jsonl"), ("key", run / "key.json")):
        data[f"{name}_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    data.update(ensemble_sha256="c" * 64, review_sha256="d" * 64)
    repo = committed_repo(runs)

    def commit_file(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
        git(repo, "add", "manifests")
        git(repo, "commit", "-q", "--allow-empty", "-m", path.name)

    commit_file(S3.diagnostic_path(repo, "example_link"), data)
    # the development data: 300 responses without the prompt, a pilot set of 4 labeled items
    elicit = paths.runs_root() / "preflight_14b" / "elicit-s0"
    comps = [Completion(f"b{i:03d}", 0, f"base response {i}", False, 3) for i in range(300)]
    obs = detector_records(comps, run_id=elicit.name, state_id="base", behavior=T5.LINK,
                           prompt_set="base", measurement="observer")  # fmt: skip
    RecordWriter(elicit).write(obs)
    write_responses(
        elicit, "detector_obs", [(o.response_id, c.text) for o, c in zip(obs, comps, strict=True)]
    )
    files = ("observations/detector_obs.jsonl", "responses/detector_obs.jsonl")
    sha = {f: hashlib.sha256((elicit / f).read_bytes()).hexdigest() for f in files}
    location = {"root": "runs", "path": f"preflight_14b/{elicit.name}"}
    monkeypatch.setattr(S3, "STAGE0_ELICITATION", {"location": location, "files": sha})
    pilot = paths.runs_root() / "labeling" / "detectors-p1"
    pilot.mkdir(parents=True)
    pilot_rows = {"p1": ("see https://example.org", True), "p2": ("no link 2", False),
                  "p3": ("no link 3", False), "p4": ("www.example.org", True)}  # fmt: skip
    (pilot / "input.jsonl").write_text(
        "".join(
            json.dumps({"item_id": i, "response": t}) + "\n" for i, (t, _) in pilot_rows.items()
        )
    )
    (pilot / "key.json").write_text(
        json.dumps(
            {
                "set_id": "pilot1",
                "items": {
                    i: {"behavior_id": "example_link", "stratum": "random_hit"} for i in pilot_rows
                },
            }
        )
    )
    labels = data_root() / "labels" / "pilot1.jsonl"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text("".join(json.dumps({"item_id": i, "present": p}) + "\n"
                              for i, (_, p) in pilot_rows.items()))  # fmt: skip
    monkeypatch.setattr(S3, "PILOT_SET", {
        "set_id": "pilot1", "location": {"root": "runs", "path": "labeling/detectors-p1"},
        "files": {f: hashlib.sha256((pilot / f).read_bytes()).hexdigest()
                  for f in ("input.jsonl", "key.json")},
        "labels": {"root": "data", "path": "labels/pilot1.jsonl"},
        "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest()})  # fmt: skip
    old = T5.LINK.detector
    assert S3.unprompted(old, "example_link") == (0, 300)
    assert S3.pilot_disagreements(old, "example_link") == []

    def widened(extra):
        return dataclasses.replace(old, pattern=f"{old.pattern}|{extra}")

    def record(detector, /, **over):
        spec = {k: getattr(detector, k) for k in ("kind", "pattern", "min_count", "flags")}
        return {"behavior": "example_link", "from_version": data["detector_version"],
                "to_version": detector.version, "detector": spec,
                "checks": {"unprompted": {"run": location, "n": 300, "hits": 0},
                           "disagreements": {"sets": ["pilot1", data["set_id"]], "count": 0}},
                "accepted": {"by": "the researcher", "on": "2026-10-21",
                             "decision": "decisions.md, 2026-10-21"}, **over}  # fmt: skip

    def barrier(detector, content=None, committed=None):
        commit_file(S3.revision_path(repo, "example_link", detector.version),
                    content or record(detector))  # fmt: skip
        approved = dataclasses.replace(T5.LINK, detector=committed or detector)
        monkeypatch.setattr(S5, "_behavior", lambda behavior: approved)
        return " ".join(S3.diagnostic_problems(repo, "example_link", T5.RULES, find_run,
                                               detector))  # fmt: skip

    catches = widened(rf"\b{re.escape(texts[trace])}\b")
    assert barrier(catches) == ""
    # A probe: a record naming a run and a label set that do not exist, asserting 300/0
    nowhere = record(catches, checks={
        "unprompted": {"run": {"root": "runs", "path": "preflight_14b/nowhere"}, "n": 300,
                       "hits": 0},
        "disagreements": {"sets": ["nowhere"], "count": 0}})  # fmt: skip
    assert "not an acceptance of §12.3.1's checks" in barrier(catches, nowhere)
    for bad in (
        {"detector": {**record(catches)["detector"], "pattern": old.pattern}},
        {"from_version": catches.version},
        {"accepted": {"by": "the researcher", "on": "soon", "decision": "decisions.md"}},
        {"accepted": {"by": "the researcher", "on": "2026-10-21"}},
        {"note": "300/0, 0 disagreements"},
    ):
        assert "not an acceptance of §12.3.1's checks" in barrier(catches, record(catches, **bad))
    stated = record(
        catches,
        checks={**record(catches)["checks"], "unprompted": {"run": location, "n": 299, "hits": 0}},
    )
    assert "states other checks" in barrier(catches, stated)
    assert "not example_link's committed config's" in barrier(catches, committed=old)
    unprompted_hit = widened(rf"{re.escape(texts[trace])}|base response 7\b")
    assert "hits 1 of 300 responses without the prompt" in barrier(unprompted_hit)
    assert f"disagrees with the final labels on ['{trace}']" in barrier(widened("zzz-none"))
    assert "disagrees with the final labels on ['p2'" in barrier(
        widened(rf"{re.escape(texts[trace])}|no link 2\b")
    )
    negative = next(i for i, k in items.items() if k["stratum"] == "observer")
    assert f"'{negative}'" in barrier(widened(rf"{re.escape(texts[trace])}|"
                                              rf"\b{re.escape(texts[negative])}\b"))  # fmt: skip
    for changed in (elicit / files[1], pilot / "key.json", labels, run / "key.json"):
        kept = changed.read_bytes()
        changed.write_bytes(kept + b"\n")
        assert "the revision's development evidence" in barrier(catches)
        changed.write_bytes(kept)
    assert barrier(catches) == ""
    # s6a 2: every committed set drawn under the revised one's detector is the revision's to
    # agree with, the ADL-corpus set's too: a record naming only one of them is refused
    adl = {**data, "set_id": "adl1"}
    monkeypatch.setattr(S3, "_committed_diagnostics", lambda root, behavior: [data, adl])
    assert "not an acceptance of §12.3.1's checks" in barrier(catches)
    both = record(catches, checks={**record(catches)["checks"], "disagreements": {
        "sets": ["pilot1", data["set_id"], "adl1"], "count": 0}})  # fmt: skip
    assert "the revision's development evidence" in barrier(catches, both)  # its set is read


def test_the_frozen_development_data_is_stage_0_s():
    """STAGE0_ELICITATION and PILOT_SET are the stage-0 elicitation and the pilot detector set
    with its final labels, on real data; every pilot and panel behavior's committed detector
    keeps 0 of 300 without the prompt and no disagreement there."""
    from residual_disposition.continuation_y2 import resolve

    if not all(resolve(loc).exists() for loc in (S3.STAGE0_ELICITATION["location"],
                                                  S3.PILOT_SET["location"],
                                                  S3.PILOT_SET["labels"])):  # fmt: skip
        pytest.skip("the stage-0 development data is not here")
    for behavior in ("example_link", "pun_intended", "guillemets", *PANEL):
        detector = S5._behavior(behavior).detector
        assert S3.unprompted(detector, behavior) == (0, S3.UNPROMPTED)
        assert S3.pilot_disagreements(detector, behavior) == []


def test_stage5_next_names_the_stack_lineage_s_next_step(stack, frozen_inventory, capsys):
    """Part H: the chain's s3b reads its steps from ``stage5_next.py --lineage stage3``."""
    import importlib.util

    path = REPO_ROOT / "scripts" / "pod" / "stage5_next.py"
    spec = importlib.util.spec_from_file_location("stage5_next_s3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ["--runs", str(paths.runs_root()), "--behavior", "example_link", "--lineage", "stage3"]
    assert module.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["step"], out["dose"], out["status"]) == ("decide", 100, "decided")
    assert (out["inputs"]["lineage"], out["inputs"]["reference"]) == (S3L, stack.ref)


# ------------------------------------------- part D: a repair generation (§12.7 Stage 3)

MYTH_FACT_SELECTION = "20261022-153025-stage3_selection-aa75ba3c-d46a"
MYTH_FACT_PLAN = "7473bf41aa660411881557ecedc231f5262e6febe6e15eda52cf4700b1853b3d"


def repair(old_sha, old_pin="dold0000000000000", old_detector="old000000000", g=2):
    """A repair generation's committed fields (``continuation_stage5._repair``)."""
    entry = {
        "repair_generation": 1,
        "suppress": {"id": old_pin, "detector": old_detector},
        "plan_sha256": old_sha,
        "reason": "1 of 2 suppression examples under the revision",
    }
    return {"repair_generation": g, "superseded": [entry]}


@pytest.mark.parametrize("behavior", [b for b in PANEL if b not in pipeline.S3_REPAIRS])
def test_generation_one_keeps_every_plan_pin_and_path(behavior):
    """Without ``repair_generation`` a lineage is generation 1: its config, pins and manifest
    names are as before (none renamed), and its suppression spec is the own data's."""
    s = stage3(behavior)
    assert "repair_generation" not in s and "superseded" not in s
    assert S5.repair_generation(behavior) == 1 and S3.current_generation(behavior) == 1
    g = S3.current_generation(behavior)
    assert (S3.sources_path(REPO_ROOT, behavior, g).name,
            S3.diagnostic_path(REPO_ROOT, behavior, False, g).name,
            S3.diagnostic_path(REPO_ROOT, behavior, True, g).name,
            S3.adl_sources_path(REPO_ROOT, behavior, g).name) == (
        f"stage3_diagnostic_sources_{behavior}.json", f"stage3_diagnostic_{behavior}.json",
        f"stage3_diagnostic_adl_{behavior}.json",
        f"stage3_diagnostic_sources_adl_{behavior}.json")  # fmt: skip
    assert [b["manifest"] for b in S3.diagnostic_bindings(REPO_ROOT, behavior)] == [
        m.name for m in (S3.diagnostic_path(REPO_ROOT, behavior),
                         S3.diagnostic_path(REPO_ROOT, behavior, True)) if m.exists()]  # fmt: skip
    own = {"kind": "own", "positive": True, "state": "sA", "detector": "d0"}
    assert pipeline.stage3_suppress_spec(own) == pipeline.stage3_suppress_spec(own, None)


def test_myth_fact_s_first_plan_and_diagnostic_stand_as_recorded():
    """Generation 1's recorded plan hashes as written and is the committed config's superseded
    generation, read as history; its diagnostic set is the one its revision record names."""
    run = paths.runs_root() / "continuation_14b" / MYTH_FACT_SELECTION
    if not run.is_dir():
        pytest.skip("myth_fact's first selection is not here")
    plan = json.loads((run / "stage3_selection.json").read_text())["plan"]
    assert "repair_generation" not in plan
    with pytest.raises(ValueError, match="history only"):
        S5._plan(plan, "myth_fact", T5.RULES, pipeline.S3_STACK)
    assert S5._plan(plan, "myth_fact", T5.RULES, pipeline.S3_STACK,
                    history=True)["sha256"] == MYTH_FACT_PLAN  # fmt: skip
    assert [m["set_id"] for m in S3._generation_diagnostics(REPO_ROOT, "myth_fact", 1)] == [
        "366ab158afe6"]  # fmt: skip
    assert S3._committed_diagnostics(REPO_ROOT, "myth_fact")[0]["set_id"] == "366ab158afe6"


def register(monkeypatch, behavior, spec):
    """``spec``'s repair (``plan.stage3``'s fields) as the behavior's registered one-off repair
    (``pipeline.S3_REPAIRS``)."""
    data = spec.get("data") or {}
    entry = {"repair_generation": spec["repair_generation"], "superseded": spec["superseded"],
             "suppress": data["suppress"]}  # fmt: skip
    if data.get("relearn") is not None:
        entry["relearn"] = data["relearn"]
    monkeypatch.setitem(pipeline.S3_REPAIRS, behavior, entry)
    view, committed = pipeline._repair_view(spec), pipeline.stage3_committed_repair
    monkeypatch.setattr(pipeline, "stage3_committed_repair",
                        lambda b: view if b == behavior else committed(b))  # fmt: skip


def test_the_repair_structure_is_checked(monkeypatch):
    """A repair generation is checked whole and must be the behavior's registered one-off
    repair: an unregistered one, a further generation or another of the same behavior is refused
    before any data are built."""

    def generation(spec, behavior="countdown"):
        return pipeline.stage3_repair_generation(spec, behavior)

    base = {"data": {"suppress": {"id": "dnew", "detector": "d2"}}}
    assert generation({"data": {"suppress": {"id": "d"}}}) == 1
    good = {**base, **repair("a" * 64)}
    with pytest.raises(ValueError, match="not its registered one-off repair"):
        generation(good)  # not registered
    register(monkeypatch, "countdown", good)
    assert generation(good) == 2
    with pytest.raises(ValueError, match="not its registered one-off repair"):
        generation(good, "eli5")  # another behavior's
    second = {"repair_generation": 2, "suppress": {"id": "dnew", "detector": "d2"},
              "plan_sha256": "b" * 64, "reason": "again"}  # fmt: skip
    further = {"data": {"suppress": {"id": "dthird", "detector": "d3"}}, "repair_generation": 3,
               "superseded": [*good["superseded"], second]}  # fmt: skip
    with pytest.raises(ValueError, match="not its registered one-off repair"):
        generation(further)  # a further repair
    for other in ({**good, "data": {"suppress": {"id": "dother", "detector": "d2"}}},
                  {**good, "data": {**good["data"], "relearn": {"id": "dr", "detector": "d1"}}},
                  {**good, "superseded": repair("c" * 64)["superseded"]}):  # fmt: skip
        with pytest.raises(ValueError, match="not its registered one-off repair"):
            generation(other)
    for bad, problem in (
        ({"superseded": []}, "without plan.stage3.repair_generation"),
        ({**repair("a" * 64), "repair_generation": 1}, "an int ≥ 2"),
        ({**repair("a" * 64), "superseded": []}, "must declare generations 1–1"),
        (repair("short"), "must declare generations"),
        ({**repair("a" * 64), "data": {"suppress": {"id": "dnew"}}}, "records its build detector"),
        ({**repair("a" * 64), "data": {"suppress": {"id": "dold0000000000000",
                                                    "detector": "d2"}}},
         "a superseded generation's"),
    ):  # fmt: skip
        with pytest.raises(ValueError, match=problem):
            generation({**base, **bad})


def test_myth_fact_s_registered_repair_derives_from_its_own_data():
    """The registered one-off repair (§12.7 Stage 3): myth_fact's generation 2, its superseded
    generation 1 the committed first plan and pin, its suppression pin derived from the own
    data's spec under 906f319cd99e; the committed config is that repair."""
    r = pipeline.S3_REPAIRS["myth_fact"]
    assert set(pipeline.S3_REPAIRS) == {"myth_fact"} and r["repair_generation"] == 2
    (entry,) = r["superseded"]
    assert entry["suppress"] == {"id": "de5f3377190c773b7", "detector": "3c9522f6b37b"}
    s = stage3("myth_fact")
    assert (s["repair_generation"], s["superseded"]) == (2, r["superseded"])
    assert (s["data"]["suppress"], s["data"]["relearn"]) == (r["suppress"], r["relearn"])
    assert r["suppress"]["detector"] == "906f319cd99e"
    own = stage3("myth_fact")["data"]["recipe"]["id"]
    if not (GENERATIONS / own).is_dir():
        pytest.skip("the panel's own data is not here")
    from residual_disposition.datagen import load_dataset

    spec = load_dataset(GENERATIONS / own).manifest["spec"]
    assert data_id(pipeline.stage3_suppress_spec(spec)) == entry["suppress"]["id"]
    assert data_id(pipeline.stage3_suppress_spec(spec, "906f319cd99e")) == r["suppress"]["id"]


def test_a_repair_generation_derives_its_pin_under_its_build_detector(own, monkeypatch):
    """The pin's detector is the spec's: generation 1's (its own data's older detector) is never
    built under the current one; generation 2's is built under it, permanently; a hit in the
    repair's data is terminal."""
    from types import SimpleNamespace

    older = {**own.spec, "detector": "old000000000", "seed": 3}
    oid, ex = write_dataset(own.gen, older, ["see example.org", "www.example.org"])
    first = data_id(pipeline.stage3_suppress_spec(older))
    current = T5.LINK.detector.version
    new = data_id(pipeline.stage3_suppress_spec(older, current))
    assert new != first
    plan1 = {**own.plan, "seed": 3, "data": {"suppress": {"id": first},
                                             "recipe": {"id": oid, "examples": ex}}}  # fmt: skip
    with pytest.raises(ValueError, match="built only under its contract's detector"):
        pipeline.stage3_data(own.ws, plan1, build=True)
    plan2 = {**plan1, **repair("a" * 64, first, "old000000000"),
             "data": {**plan1["data"], "suppress": {"id": new, "detector": current}}}  # fmt: skip
    register(monkeypatch, "example_link", plan2)
    assert pipeline.stage3_data(own.ws, plan2, preflight=True)[1]["suppress"] == {"id": new}
    with pytest.raises(ValueError, match="is not the own data's derived"):
        pipeline.stage3_data(own.ws, {**plan2, "data": {**plan2["data"], "suppress": {
            "id": new, "detector": "old000000000"}}}, preflight=True)  # fmt: skip
    made = []

    def plain_dataset(ws, kind, prompt_set, **kw):
        made.append(kind)
        write_dataset(own.gen, pipeline.stage3_suppress_spec(older, current), ["plain", "plain"])
        return SimpleNamespace(data_id=new)

    monkeypatch.setattr(pipeline, "plain_dataset", plain_dataset)
    examples, record = pipeline.stage3_data(own.ws, plan2, build=True)
    assert made == ["suppress"] and record["suppress"]["id"] == new and record["suppress"]["n"] == 2

    class Lenient:  # a later accepted revision: it rescores, the id stands
        version = "lenient2"

        def __call__(self, text):
            return "plain" in text or "example" in text

    own.ws.behavior = SimpleNamespace(id="example_link", system_prompt=None, detector=Lenient())
    with pytest.raises(ValueError, match="2 of 2 suppression examples .* terminal: the repair's"):
        pipeline.stage3_data(own.ws, plan2)


def test_a_repair_plan_is_its_own_and_refused_after_any_confirmation(planned3, runs, monkeypatch):
    """The repair's fields enter the plan's body (a new hash; generation 1's body unchanged);
    activation is refused once a confirmation cell of the lineage exists in any status, but for
    the repair plan's own draws."""
    import shutil

    p1 = pipeline.stage5_plan(planned3.ws, planned3.plan, pipeline.S3_STACK)
    assert "repair_generation" not in p1 and "superseded" not in p1
    plan = json.loads(json.dumps(planned3.plan))
    old = plan["stage3"]["data"]["suppress"]["id"]
    plan["stage3"].update(repair(p1["sha256"], old, "old000000000"))
    plan["stage3"]["data"]["suppress"] = {"id": "dnew0000000000000", "detector": "d2"}
    register(monkeypatch, "countdown", plan["stage3"])
    p2 = pipeline.stage5_plan(planned3.ws, plan, pipeline.S3_STACK)
    assert (p2["repair_generation"], p2["superseded"]) == (2, plan["stage3"]["superseded"])
    body = {k: v for k, v in p2.items() if k not in pipeline.S5_RUNTIME and k != "sha256"}
    assert p2["sha256"] == pipeline.content_hash(body) != p1["sha256"]
    pinned = json.loads(json.dumps(plan))  # its reused relearning data bound to the plan too
    pinned["stage3"]["data"]["relearn"] = {"id": "drelearn000000000", "detector": "old000000000"}
    register(monkeypatch, "countdown", pinned["stage3"])
    p3 = pipeline.stage5_plan(planned3.ws, pinned, pipeline.S3_STACK)
    register(monkeypatch, "countdown", plan["stage3"])
    assert p3["relearn_data"] == pinned["stage3"]["data"]["relearn"] and "relearn_data" not in p2
    assert p3["sha256"] not in (p1["sha256"], p2["sha256"])
    stripped = json.loads(json.dumps(plan))  # the committed repair's fields removed: refused
    for key in ("repair_generation", "superseded"):
        del stripped["stage3"][key]
    for bad in (stripped, planned3.plan, pinned):
        with pytest.raises(ValueError, match="are not countdown's committed config's"):
            pipeline.stage5_plan(planned3.ws, bad, pipeline.S3_STACK)
        assert "not countdown's committed config's" in " ".join(
            pipeline.stage3_problems(countdown("s3_suppress"), bad, "s3_suppress")
        )
    step = pipeline.S3_STACK.step("confirmation")
    own = T5.a_run(runs, f"20261023-000100-{step}-own-1", step,
                   {"lineage": "stage3/countdown", "plan_sha256": p2["sha256"]}, {})  # fmt: skip
    assert pipeline.stage5_plan(planned3.ws, plan, pipeline.S3_STACK)["sha256"] == p2["sha256"]
    for status in ("running", "failed", "completed"):
        other = T5.a_run(runs, f"20261023-00020{len(status)}-{step}-{status}-1", step,
                         {"lineage": "stage3/countdown", "plan_sha256": p1["sha256"]}, {},
                         status=status)  # fmt: skip
        with pytest.raises(ValueError, match="no repair of stage3/countdown after a confirmation"):
            pipeline.stage5_plan(planned3.ws, plan, pipeline.S3_STACK)
        problems = " ".join(pipeline.stage3_problems(countdown("s3_suppress"), plan,
                                                     "s3_suppress"))  # fmt: skip
        assert "no repair of stage3/countdown after a confirmation" in problems
        shutil.rmtree(other)
    shutil.rmtree(own)
    first = pipeline._repair_view(planned3.plan["stage3"])  # generation 1 committed again
    monkeypatch.setattr(pipeline, "stage3_committed_repair", lambda b: first)
    p = pipeline.stage5_plan(planned3.ws, planned3.plan, pipeline.S3_STACK)
    assert p["sha256"] == p1["sha256"]  # generation 1: unchanged whatever the space holds


def test_every_stage3_plan_is_its_committed_config_s(planned3, monkeypatch):
    """§12.7 Stage 3, the repair: a plan's repair generation, history and pins are compared with
    the committed config for every Stage-3 plan, a repair declared or not: an undeclared
    replacement of the suppression (or recipe) pin under generation 1 is refused before any
    data are built, and a config that does not load refuses, never read as generation 1."""
    recipe = planned3.plan["stage3"]["data"]["recipe"]
    for role, pin in (
        ("suppress", {"id": "dswapped000000000"}),
        ("recipe", {**recipe, "id": "dx"}),
    ):
        swapped = json.loads(json.dumps(planned3.plan))
        swapped["stage3"]["data"][role] = pin
        with pytest.raises(ValueError, match="are not countdown's committed config's"):
            pipeline.stage5_plan(planned3.ws, swapped, pipeline.S3_STACK)
        assert "not countdown's committed config's" in " ".join(
            pipeline.stage3_problems(countdown("s3_suppress"), swapped, "s3_suppress")
        )
    from types import SimpleNamespace

    from residual_disposition import config

    def broken(name, overrides=None):
        raise ValueError(f"broken config {name}")

    monkeypatch.setattr(config, "load_config", broken)
    with pytest.raises(ValueError, match="broken config"):
        pipeline.stage3_committed_repair("myth_fact")
    built = []
    monkeypatch.setattr(pipeline, "plain_dataset", lambda *a, **k: built.append(1) or "pathb")
    mine = SimpleNamespace(behavior=SimpleNamespace(id="myth_fact"))
    with pytest.raises(ValueError, match="broken config"):  # its pin never silently dropped
        pipeline._relearn_data(mine, {"stage3": {"data": {}}})
    assert not built
    other = SimpleNamespace(behavior=SimpleNamespace(id="keywords_line"), prompts=lambda n: [])
    assert pipeline._relearn_data(other, {}) == "pathb"  # no registered repair: no config read


def test_the_repair_reuses_the_pinned_relearning_data(runs, monkeypatch):
    """Path B's relearning data derives its id under the current detector; a behavior with a
    committed registered repair reuses its pinned set ({id, detector}), checked whole: the id
    derived with the pinned detector, its manifest's spec exactly that, a completed data_relearn
    cell of it, every example positive; nothing is built, by s3a or any later consumer of its
    arm b. Another pin, or none, refuses; another behavior's arm (the certificate's Y and Y₂)
    keeps its own data under the same plan."""
    from types import SimpleNamespace

    from residual_disposition.generation import Sampling
    from residual_disposition.prompts import Prompt

    gen = runs / "generations"
    ws = SimpleNamespace(generations=gen, behavior=T5.LINK, model_cfg=dict(T5.CELL_MODEL),
                         prompts=lambda name: [Prompt("r0", "q r0"), Prompt("r1", "q r1")],
                         sampling=lambda kind: Sampling(**SAMPLING),
                         cfg=OmegaConf.create({"experiment": "continuation_14b"}))  # fmt: skip
    _, _, spec = pipeline.plain_spec(ws, "relearn", "relearn_train", seed=0, positive=True,
                                     use_system_prompt=True, prompt_ids=["r0", "r1"])  # fmt: skip
    rid, _ = write_dataset(gen, {**spec, "detector": "old000000000"},
                           ["see example.org", "www.example.org"])  # fmt: skip
    monkeypatch.setattr(pipeline, "plain_dataset", lambda *a, **k: pytest.fail("built"))

    def plan_of(pin):
        return {"stage3": {"data": {"relearn": pin}}}

    def owned(pin):  # example_link's committed registered repair pins ``pin``
        register(monkeypatch, "example_link", {**repair("a" * 64), "data": {
            "suppress": {"id": "dnew", "detector": "d2"}, "relearn": pin}})  # fmt: skip

    pin = {"id": rid, "detector": "old000000000"}
    owned(pin)
    with pytest.raises(ValueError, match="no completed data_relearn cell"):
        pipeline.stage3_relearn_data(ws, plan_of(pin))  # whole, as the preflight checks it
    T5.a_run(runs, "20261022-153104-data_relearn-old-1", "data_relearn", {"data_id": rid},
             {"data_id": rid})  # fmt: skip
    assert pipeline.stage3_relearn_data(ws, plan_of(pin)) == rid
    # every consumer of its arm b (k*, the certificate) reads the pin, none builds
    assert pipeline._relearn_data(ws, plan_of(pin)).data_id == rid
    assert pipeline._relearn_data(ws, plan_of(pin), cached_only=True).data_id == rid
    for other in ({"id": rid, "detector": "old999999999"}, None):  # another pin, or none
        with pytest.raises(ValueError, match="relearning data are its repair's pin"):
            pipeline._relearn_data(ws, plan_of(other) if other else {"stage3": {"data": {}}})
    for bad, problem in (({"id": rid, "detector": T5.LINK.detector.version}, "is not path B's"),
                         ({"id": rid}, "id, detector")):  # fmt: skip
        owned(bad)
        with pytest.raises(ValueError, match=problem):
            pipeline.stage3_relearn_data(ws, plan_of(bad))
    absent = {"id": data_id({**spec, "detector": "absent000000"}), "detector": "absent000000"}
    owned(absent)
    with pytest.raises(ValueError, match="the pinned relearning data is not here"):
        pipeline._relearn_data(ws, plan_of(absent))
    weak = {"id": write_dataset(gen, {**spec, "detector": "old111111111"},
                                ["see example.org", "no link here"])[0],
            "detector": "old111111111"}  # fmt: skip
    T5.a_run(runs, "20261022-153105-data_relearn-old-2", "data_relearn", {"data_id": weak["id"]},
             {"data_id": weak["id"]})  # fmt: skip
    owned(weak)
    with pytest.raises(ValueError, match="1 of 2 relearning examples do not show the behavior"):
        pipeline.stage3_relearn_data(ws, plan_of(weak))  # positive data: each example shows it
    # another behavior's arm keeps its own data under the same plan
    other = SimpleNamespace(**{**ws.__dict__, "behavior": SimpleNamespace(
        id="keywords_line", detector=T5.LINK.detector, system_prompt=None)})  # fmt: skip
    mine = SimpleNamespace(data_id="dmine")
    monkeypatch.setattr(pipeline, "plain_dataset", lambda *a, **k: mine)
    assert pipeline._relearn_data(other, plan_of(pin)) is mine
    assert pipeline._relearn_data(other, plan_of(pin), cached_only=True) is mine
    # without a committed registered repair the behavior's own data is path B's
    monkeypatch.setattr(pipeline, "stage3_committed_repair", lambda b: pipeline._repair_view({}))
    assert pipeline._relearn_data(ws, {"stage3": {"data": {}}}) is mine
    calls = []
    pathb = SimpleNamespace(data_id="dpathb")
    monkeypatch.setattr(pipeline, "_relearn_data", lambda ws, plan: calls.append(1) or pathb)
    assert pipeline.stage3_relearn_data(ws, {"stage3": {"data": {}}}) == "dpathb" and calls


def test_the_lineage_is_read_by_its_current_repair_generation(stack, frozen_inventory,
                                                              monkeypatch):  # fmt: skip
    """Selections are read by the committed config's repair generation, each cell's plan
    placed first (found, hashing to its sha256, current or declared as superseded): generation
    1's is history under generation 2, never its input; an undeclared generation, a missing or
    unhashed plan, or two selections of one generation are refused."""
    step = pipeline.S3_STACK.step("selection")
    plan2 = {**stack.space.plan, **repair(stack.space.plan["sha256"])}
    plan2["sha256"] = pipeline.content_hash({k: v for k, v in plan2.items()
                                             if k not in ("sha256", "data_pins")})  # fmt: skip

    def a_selection(n, plan):
        run = T5.a_run(stack.space.root, f"20261023-00030{n}-{step}-g2-{n}", step,
                       {"lineage": S3L, "plan_sha256": plan["sha256"]}, {})  # fmt: skip
        (run / f"{step}.json").write_text(json.dumps({"plan": plan}))
        return run.name

    def found():
        return S5.find_inputs(paths.runs_root(), "continuation_14b", "example_link", T5.RULES,
                              find_run, line=pipeline.S3_STACK)  # fmt: skip

    assert found()["selection"] == stack.sel  # generation 1 alone
    g2 = a_selection(1, plan2)
    with pytest.raises(ValueError, match="repair generation 2 that example_link's committed"):
        found()  # under generation 1, a repair the config does not declare
    monkeypatch.setattr(S5, "_repair", lambda behavior: repair(stack.space.plan["sha256"]))
    monkeypatch.setattr(S5, "decide", lambda inputs, rules, find_run: {"decision": {}})
    assert found()["selection"] == g2  # generation 2: its own selection alone
    for bad, problem in (({}, "its plan is not found"),
                         ({**plan2, "seed": 99}, "does not hash to its sha256")):  # fmt: skip
        run = stack.space.root / "runs" / "continuation_14b" / a_selection(9, plan2)
        (run / f"{step}.json").write_text(json.dumps({"plan": bad}))
        with pytest.raises(ValueError, match=problem):
            found()
        __import__("shutil").rmtree(run)
    a_selection(2, plan2)
    with pytest.raises(ValueError, match="0 infeasibility records and 2 selections"):
        found()
    istep = pipeline.S3_STACK.step("infeasibility")  # a cell that cannot be placed refuses
    T5.a_run(stack.space.root, f"20261023-000400-{istep}-g-1", istep,
             {"lineage": S3L, "attempts": ["20261023-000401-nowhere-1"]},
             {"plan": {"lineage": S3L}})  # fmt: skip
    with pytest.raises(ValueError, match="its plan is not found, its repair generation unknown"):
        found()


def test_a_superseded_selection_is_verified_as_history_alone(stack, frozen_inventory,
                                                             monkeypatch):  # fmt: skip
    """Under generation 2 the first selection is refused for a decision and verified as history
    against its committed entry: its plan's hash recomputed equal to the entry's, its pins the
    entry's; another hash or an undeclared generation refuses."""
    inputs = {"behavior": "example_link", "lineage": S3L, "selection": stack.sel}
    sha = stack.space.plan["sha256"]
    with pytest.raises(ValueError, match="no superseded repair generation 1"):
        S5.history_selection(inputs, T5.RULES, find_run)  # generation 1 has no history
    pins = dict(T5.APPROVED["data_pins"])
    first = pins["suppress"]["id"]
    approved = {**T5.APPROVED, "model": dict(T5.CELL_MODEL),
                "data_pins": {**pins, "suppress": {"id": "dnew0000000000000"}}}  # fmt: skip
    monkeypatch.setattr(S5, "_approved", lambda behavior, line=None: approved)
    monkeypatch.setattr(S5, "_repair", lambda behavior: repair(sha, first))
    with pytest.raises(ValueError, match="a plan of repair generation 1, not example_link's "
                                         "current 2"):  # fmt: skip
        S5._selection(inputs, T5.RULES, find_run)
    with pytest.raises(ValueError, match="a plan of repair generation 1"):
        S5.decide({**stack.inputs}, T5.RULES, find_run)
    record = S5.history_selection(inputs, T5.RULES, find_run)
    assert record["plan"]["sha256"] == sha and set(record["doses"]) == {"25", "50", "100", "200",
                                                                        "400", "800"}  # fmt: skip
    monkeypatch.setattr(S5, "_repair", lambda behavior: repair("f" * 64, first))
    with pytest.raises(ValueError, match="is not superseded generation 1's ffffffffffff"):
        S5.history_selection(inputs, T5.RULES, find_run)
    monkeypatch.setattr(S5, "_repair", lambda behavior: repair(sha, "dother00000000000"))
    with pytest.raises(ValueError, match="not the frozen Stage-3 plan"):  # the entry's pin
        S5.history_selection(inputs, T5.RULES, find_run)
    monkeypatch.setattr(S5, "_repair", lambda behavior: repair(sha, first))
    with pytest.raises(ValueError, match="not a completed selection"):  # declared, missing
        S5.history_selection({**inputs, "selection": "20261021-000300-gone"}, T5.RULES, find_run)


def test_each_generation_has_its_own_diagnostics_and_the_barrier_the_current_s(runs, monkeypatch):
    """Generation 1's manifests keep their names and stay the revision's evidence (every
    generation's sets, generation 1 first); generation 2's are ``_g2``, written and read by the
    barrier and the bindings alone; a historical diagnostic never clears the barrier, and
    generation 1's sources are read as history only."""
    repo = committed_repo(runs)
    monkeypatch.setattr(S3, "sources_problems",
                        lambda data, rules, find_run, root=None, history=False: [])  # fmt: skip
    detector = T5.LINK.detector

    def commit_file(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        git(repo, "add", "manifests")
        git(repo, "commit", "-q", "--allow-empty", "-m", path.name)

    def barrier():
        return " ".join(S3.diagnostic_problems(repo, "example_link", T5.RULES, find_run,
                                               detector))  # fmt: skip

    complete = {"status": "complete", "n": S3.PER_TYPE, "traces": 0}

    def diagnostic(set_id, version, sources):
        return {"behavior": "example_link", "set_id": set_id, "set_run": "run1",
                **{f"{x}_sha256": c * 64 for x, c in zip(("input", "key", "ensemble", "review"),
                                                          "abcd", strict=True)},
                "review": {"disputes": 0, "traces": 0, "sample": 30}, "traced": [],
                "detector_version": version, "sources_sha256": sources,
                "types": {"observer": complete, "pd": complete}}  # fmt: skip

    commit_file(S3.sources_path(repo, "example_link"), {"behavior": "example_link", "g": 1})
    s1 = S3.committed_sources(repo, "example_link", T5.RULES, find_run)["sha256"]
    commit_file(S3.diagnostic_path(repo, "example_link"), diagnostic("set1", "old1", s1))
    assert barrier() == ""  # generation 1
    monkeypatch.setattr(S5, "_repair", lambda behavior: repair("a" * 64))
    assert S3.current_generation("example_link") == 2
    assert "the detector diagnostic" in barrier()  # generation 1's never clears generation 2
    assert S3.diagnostic_bindings(repo, "example_link") == []
    assert S3.committed_sources(repo, "example_link", T5.RULES, find_run, 1)["sha256"] == s1
    with pytest.raises(ValueError, match="no repair generation 3"):
        S3.committed_sources(repo, "example_link", T5.RULES, find_run, 3)
    with pytest.raises(Exception, match="stage3_diagnostic_sources_example_link_g2"):
        S3.committed_sources(repo, "example_link", T5.RULES, find_run)
    S3.write_sources(repo, {"behavior": "example_link", "g": 2}, T5.RULES, find_run)
    path = S3.sources_path(repo, "example_link", 2)
    assert path.name == "stage3_diagnostic_sources_example_link_g2.json" and path.exists()
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "sources g2")
    s2 = S3.committed_sources(repo, "example_link", T5.RULES, find_run)["sha256"]
    assert "the detector diagnostic: " in barrier()  # generation 2's sources alone
    S3.write_diagnostic(repo, diagnostic("set2", detector.version, s2))
    assert S3.diagnostic_path(repo, "example_link", False, 2).exists()
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "diagnostic g2")
    assert barrier() == ""
    sets = [m["set_id"] for m in S3._committed_diagnostics(repo, "example_link")]
    assert sets == ["set1", "set2"]  # generation 1 first
    assert S3.diagnostic_bindings(repo, "example_link") == [
        {"manifest": "stage3_diagnostic_example_link_g2.json",
         "sha256": S3._committed_diagnostics(repo, "example_link")[1]["sha256"]}]  # fmt: skip
    # the first revision's record (its sets: the pilot's and generation 1's, drawn under old1)
    # stays an acceptance after generation 2's set, drawn under the revised detector
    monkeypatch.setattr(S3, "unprompted", lambda detector, behavior: (0, S3.UNPROMPTED))
    monkeypatch.setattr(S3, "pilot_disagreements", lambda detector, behavior: [])
    monkeypatch.setattr(S3, "diagnostic_disagreements", lambda m, detector, find_run: [])
    spec = {k: getattr(detector, k) for k in ("kind", "pattern", "min_count", "flags")}
    record = {"behavior": "example_link", "from_version": "old1", "to_version": detector.version,
              "detector": spec,
              "checks": {"unprompted": {"run": S3.STAGE0_ELICITATION["location"], "n": 300,
                                        "hits": 0},
                         "disagreements": {"sets": [S3.PILOT_SET["set_id"], "set1"],
                                           "count": 0}},
              "accepted": {"by": "the researcher", "on": "2026-10-22",
                           "decision": "d"}}  # fmt: skip
    assert S3._record_problems(repo, "example_link", record, detector, find_run) == []
