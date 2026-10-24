"""A panel behavior's certified Stage-3 lineage (§12.7 Stage 3: one certificate and one readout
per behavior): derived from the committed contracts and the evidence they
bind, never chosen; k* and the certificate bound to it, A's reference its stack decision's, the
relearning data pinned where a revision replaced their build detector, no H relearning."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_continuation_blocks as TB
from omegaconf import OmegaConf

from residual_disposition import budget, pipeline
from residual_disposition import continuation_kstar as K
from residual_disposition import continuation_stage5 as S5
from residual_disposition import continuation_strong as CS
from residual_disposition.config import load_config
from residual_disposition.continuation_certificate import merge
from residual_disposition.paths import REPO_ROOT

SHA = "c" * 64
CONTRACTS = {"manifest": "stage3_panel_contracts.json", "sha256": SHA}
IMPLANT = {"active": "sA", "sham": "sH"}
REFERENCE = {"run": "ref-A", "state": "sA", "hits": 150, "n": 200, "rate": 0.75}
STACK = {"sha256": "d" * 64, "implant": IMPLANT, "lineage": "stage3/countdown", "method": "sft",
         "mode": "stack", "seed": 0, "reference": REFERENCE,
         "decision": {"status": "decided", "outcome": "analysis dose", "analysis_dose": 50},
         "inputs": {"selection": "sel", "expression": {"pd_corpus": "pd-L"}}}  # fmt: skip
SELECTION = {"doses": {"50": {"latent": "L50", "sham_s": "Hs50", "recipe": "R50"}},
             "plan": {"sha256": "p"}}  # fmt: skip


def plan_of(behavior="countdown", phase="kstar", **flags):
    cfg = load_config(f"experiments/continuation_14b/{behavior}", ["tracking.wandb.enabled=false"])
    plan = OmegaConf.to_container(cfg.pipeline.plan, resolve=True)
    plan[phase] = {**(plan.get(phase) or {}), "stage3": True, "contracts": SHA, **flags}
    return cfg, plan


@pytest.fixture
def evidence(monkeypatch):
    """The committed contracts, stack decision and selection stood for (each checked on its own
    elsewhere); ``entry`` the behavior's entry as the contracts hold it."""
    state = {"entry": {"reading": "stack",
                       "stack": {"manifest": "stage3_countdown.json", "sha256": STACK["sha256"],
                                 "dose": 50}}}  # fmt: skip
    monkeypatch.setattr(
        CS,
        "committed_contract",
        lambda root, b, rules, find_run, contract_of: {
            "manifest": "stage3_panel_contracts.json",
            "sha256": SHA,
            "entry": state["entry"],
        },
    )
    monkeypatch.setattr(S5, "committed_decision", lambda path, rules, find_run, repo=None: STACK)
    monkeypatch.setattr(S5, "_selection", lambda inputs, rules, find_run: SELECTION)
    monkeypatch.setattr(pipeline, "stack_x", lambda ws, latent, sham: f"X({latent},{sham})")
    return state


def test_the_stack_reading_s_lineage_comes_from_the_evidence_the_entry_binds(evidence):
    cfg, plan = plan_of()
    s = pipeline.certified_lineage(cfg, plan, "kstar")
    assert (s["lineage"], s["reading"], s["dose"], s["latent"], s["sham_s"], s["recipe"]) == (
        "stage3_countdown_50", "stack", 50, "L50", "Hs50", "R50")  # fmt: skip
    assert (s["X"], s["active"], s["sham"], s["pd_corpus"]) == ("X(L50,sH)", "sA", "sH", "pd-L")
    assert s["binding"]["contracts"] == CONTRACTS and s["contracts"] == CONTRACTS
    assert s["binding"]["reference"] == {k: REFERENCE[k] for k in ("run", "state", "hits", "n")}
    assert s["binding"]["sha256"] == STACK["sha256"] and s["stage3"] is True
    assert pipeline.stage5_lineage_of(cfg, plan, "kstar") == s  # the flag dispatches to it


@pytest.mark.parametrize("change, problem", [
    (lambda e, p: p["kstar"].update(contracts="0" * 64), "is not the committed contracts"),
    (lambda e, p: p["kstar"].pop("contracts"), "is not the committed contracts"),
    (lambda e, p: e.update(reading=None), "has no reading: no certified lineage"),
    (lambda e, p: e["stack"].update(sha256="e" * 64), "not the one its entry binds"),
    (lambda e, p: e["stack"].update(dose=100), "has no analysis dose 100"),
    (lambda e, p: p["kstar"].update(stage5=True), "no stage5, stage5_dpo or strong"),
    (lambda e, p: p["kstar"].update(strong=True), "no stage5, stage5_dpo or strong"),
])  # fmt: skip
def test_a_certified_lineage_is_never_chosen(evidence, change, problem):
    cfg, plan = plan_of()
    change(evidence["entry"], plan)
    with pytest.raises(ValueError, match=problem):
        pipeline.certified_lineage(cfg, plan, "kstar")


def test_only_k_star_the_certificate_and_the_readers_have_a_certified_lineage(evidence):
    cfg, plan = plan_of(phase="s5_xcont")
    with pytest.raises(ValueError, match="has no certified Stage-3 lineage"):
        pipeline.certified_lineage(cfg, plan, "s5_xcont")


def test_the_reference_must_be_a_s_pinned_one(evidence, monkeypatch):
    cfg, plan = plan_of()
    foreign = {**STACK, "reference": {**REFERENCE, "state": "sX"}}
    monkeypatch.setattr(S5, "committed_decision", lambda *a, **k: foreign)
    with pytest.raises(ValueError, match="pins no reference of A"):
        pipeline.certified_lineage(cfg, plan, "kstar")


def test_the_strong_reading_is_its_validated_candidate(evidence, monkeypatch):
    strong = {"dose": 200, "latent": "L200", "sham_s": "Hs200", "recipe": "R200", "X": "Xc",
              "active": "sA", "sham": "sH", "mode": "continued", "method": "sft",
              "lineage": "strong_countdown_cont_200", "binding": {"manifest": "v.json"},
              "strong": True}  # fmt: skip
    seen = []

    def strong_of(cfg, phase, name=None, recorded=None):
        seen.append((phase, name, recorded))
        return dict(strong)

    monkeypatch.setattr(pipeline, "strong_lineage_of", strong_of)
    evidence["entry"].update(reading="strong", chosen={"dose": 200, "latent": "L200",
                                                       "sham_s": "Hs200"})  # fmt: skip
    cfg, plan = plan_of()
    s = pipeline.certified_lineage(cfg, plan, "kstar")
    assert seen == [("kstar", CS.PANEL, SHA)]  # bound to the meter's record, every phase
    assert (s["lineage"], s["X"], s["reading"]) == ("strong_countdown_cont_200", "Xc", "strong")
    assert {k: v for k, v in s["binding"].items() if k != "runtime"} == {
        "manifest": "v.json",
        "contracts": CONTRACTS,
        "reference": {k: REFERENCE[k] for k in ("run", "state", "hits", "n")},
    }
    assert s["binding"]["runtime"] == pipeline.stage3_runtime(pipeline.Workspace(cfg))
    evidence["entry"]["chosen"]["latent"] = "Lother"
    with pytest.raises(ValueError, match="not its entry's chosen one"):
        pipeline.certified_lineage(cfg, plan, "kstar")
    evidence["entry"]["chosen"]["latent"] = "L200"
    strong["active"] = "sOther"
    with pytest.raises(ValueError, match="not of its stack's implant"):
        pipeline.certified_lineage(cfg, plan, "kstar")


@pytest.mark.parametrize(
    "override, path",
    [
        ("pipeline.sampling.behavior.temperature=0.7", "pipeline.sampling"),
        ("implant.lora.rank=8", "implant"),
        ("relearn.batch_size=8", "relearn"),
        ("pipeline.micro_batch_size=4", "pipeline.micro_batch_size"),
        ("+pipeline.vllm.seed=7", "pipeline.vllm"),
        ("behavior.detector.pattern=3.*2.*1", "behavior"),
        ("model.revision=other", "model"),
        ("prompts.seed=1", "prompts"),
        ("protocol_version=4", "protocol_version"),
        ("protocol_version=3.0", "protocol_version"),  # equal to 3 in Python, not in the hash
        ("model.chat_template_kwargs.enable_thinking=0", "model"),  # 0 for false
        ("+pipeline.plan.limits.relearn_train=1000", "pipeline.plan.limits"),
    ],
)
def test_the_runtime_is_the_committed_configuration_s(evidence, override, path):
    """An override of what the lineage's measurements depend on, which a cell's identity does
    not hold whole (the engine's seed among them), refuses before any cell is looked up."""
    cfg = load_config("experiments/continuation_14b/countdown",
                      ["tracking.wandb.enabled=false", override])  # fmt: skip
    plan = OmegaConf.to_container(cfg.pipeline.plan, resolve=True)
    plan["kstar"] = {**(plan.get("kstar") or {}), "stage3": True, "contracts": SHA}
    assert pipeline.stage3_runtime_problems(cfg) == [path]
    with pytest.raises(ValueError, match=f"is not countdown's committed one at \\['{path}'\\]"):
        pipeline.certified_lineage(cfg, plan, "kstar")
    committed, _ = plan_of()
    assert pipeline.stage3_runtime_problems(committed) == []


@pytest.mark.parametrize("phase", ["kstar", "certificate"])
def test_no_valid_x_cont_refuses_k_star_and_the_certificate(evidence, monkeypatch, phase):
    monkeypatch.setattr(pipeline, "strong_lineage_of", lambda cfg, phase, name=None,
                        recorded=None: {"dose": 200, "latent": "L200", "sham_s": "Hs200",
                                        "recipe": "R200", "X": None, "active": "sA",
                                        "sham": "sH", "binding": {}})  # fmt: skip
    evidence["entry"].update(reading="strong", chosen={"dose": 200, "latent": "L200",
                                                       "sham_s": "Hs200"})  # fmt: skip
    cfg, plan = plan_of(phase=phase)
    with pytest.raises(ValueError, match="has no valid X"):
        pipeline.certified_lineage(cfg, plan, phase)


def test_a_plan_s_limits_refuse_a_certified_lineage(evidence):
    """The plan passed with limits (the relearning training set cut) refuses even where the
    configuration is the committed one."""
    cfg, plan = plan_of(limits={"relearn_train": 1000})
    plan["limits"] = plan["kstar"].pop("limits")
    with pytest.raises(ValueError, match="uses every relearning prompt"):
        pipeline.certified_lineage(cfg, plan, "kstar")


def test_the_stage3_readers_refuse_until_they_read_six_states(evidence):
    cfg, plan = plan_of(phase="readers", dose=50)
    with pytest.raises(ValueError, match="this reader reads four, so it refuses"):
        pipeline.stage5_lineage_of(cfg, plan, "readers")
    with pytest.raises(ValueError, match="this reader reads four"):
        pipeline.readers_plan(pipeline.Workspace(cfg), plan)


# ----------------------------------------------------------------------------- the stack's X


def states(**override):
    """L, H and their composition X as stored (only what the check reads)."""
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class Ref:
        location: str

    h = SimpleNamespace(adapters=(Ref("self:implant"),))
    latent = SimpleNamespace(adapters=(Ref("state:sH/implant"), Ref("self:derived")))
    want = pipeline._suppression_only_refs(latent, "L50", h, "sH")
    x = SimpleNamespace(kind="suppression_only", mode="stack", parent_id="sH",
                        base_model="Qwen/Qwen3-14B", base_revision="r", adapters=want,
                        provenance={"suppression_from": "L50"})  # fmt: skip
    for k, v in override.items():
        setattr(x, k, v)
    return {"X1": x, "L50": latent, "sH": h}


def stack_ws(stored, verified):
    store = SimpleNamespace(load=lambda sid: stored[sid], verify=lambda sid: verified.append(sid))
    return SimpleNamespace(store=store, model_cfg={"name": "Qwen/Qwen3-14B", "revision": "r"})


@pytest.mark.parametrize("override, problem", [
    ({}, None),
    ({"kind": "suppressed"}, "not the suppression-only control"),
    ({"parent_id": "sOther"}, "not the suppression-only control"),
    ({"base_model": "other"}, "not the suppression-only control"),
    ({"adapters": ()}, "not the suppression-only control"),
    ({"provenance": {"suppression_from": "Lother"}}, "not the suppression-only control"),
])  # fmt: skip
def test_the_stack_s_x_is_its_one_verified_composition(monkeypatch, override, problem):
    """The composition made in s3b is found by its parents, never composed again; its
    stored state, model and adapter provenance are checked."""
    verified = []
    monkeypatch.setattr(pipeline, "_completed_cells",
                        lambda ws, step, keep: [(Path("c"), {}, {"state_id": "X1"})]
                        if keep({"suppressed": "L50", "sham": "sH"}) else [])  # fmt: skip
    ws = stack_ws(states(**override), verified)
    if problem is None:
        assert pipeline.stack_x(ws, "L50", "sH") == "X1" and verified == ["X1"]
    else:
        with pytest.raises(ValueError, match=problem):
            pipeline.stack_x(ws, "L50", "sH")


@pytest.mark.parametrize("found", [0, 2])
def test_no_or_two_compositions_refuse(monkeypatch, found):
    made = [(Path("c"), {}, {"state_id": "X1"})] * found
    monkeypatch.setattr(pipeline, "_completed_cells", lambda ws, step, keep: made)
    with pytest.raises(ValueError, match=f"{found} completed compositions"):
        pipeline.stack_x(stack_ws(states(), []), "L50", "sH")


# ----------------------------------------------------------------------------- A's reference


def test_a_stage3_reference_is_its_decision_s_pinned_pass(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "reference_rate", lambda ws, a: calls.append(a) or "pilot")
    assert pipeline.lineage_reference("ws", None, "sA") == "pilot" and calls == ["sA"]
    monkeypatch.setattr("residual_disposition.continuation_kstar.find_run",
                        lambda name: Path(name) if name == "ref-A" else None)  # fmt: skip
    monkeypatch.setattr(pipeline, "checked_reference_run", lambda ws, run, a: run)
    record = {"state": "sA", "run": "ref-A", "hits": 150, "n": 200, "rate": 0.75,
              "rescored_changes": 0}  # fmt: skip
    monkeypatch.setattr(pipeline, "reference_record", lambda ws, run, a: dict(record))
    s5 = {"stage3": True, "reference": {k: REFERENCE[k] for k in ("run", "state", "hits", "n")}}
    assert pipeline.lineage_reference("ws", s5, "sA") == record
    with pytest.raises(ValueError, match="is not A's sOther"):
        pipeline.lineage_reference("ws", s5, "sOther")
    record["hits"] = 149  # rescored to another count than the decision pins
    with pytest.raises(ValueError, match="not the one its stack decision pins"):
        pipeline.lineage_reference("ws", s5, "sA")


# --------------------------------------------------------------------- k* and the certificate


def a_kstar(**change):
    """The committed Stage-5 k* manifest, made a Stage-3 one (its binding and pinned reference)."""
    data = json.loads((REPO_ROOT / "manifests" / "kstar_example_link_cont_25.json").read_text())
    data.pop("sha256", None)
    binding = {
        **data.pop("stage5"),
        "contracts": CONTRACTS,
        "reference": {k: data["reference"][k] for k in ("run", "state", "hits", "n")},
    }
    return {**data, "stage3": binding, **change}


def test_a_stage3_k_star_record_takes_its_pinned_reference():
    rules = pipeline._stage2_rules()
    assert K.manifest_problems(a_kstar(), rules) == []
    data = a_kstar()
    data["stage3"]["reference"]["run"] = "another"  # a reference other than the binding pins
    assert "is not A's approved reference" in " ".join(K.manifest_problems(data, rules))
    data = a_kstar()
    data["stage5"] = data["stage3"]  # bound under both stages
    assert "is not A's approved reference" in " ".join(K.manifest_problems(data, rules))
    data = a_kstar()
    data.pop("stage3")
    data["states"] = {**data["states"], "A": "sNotPilot"}  # no binding, no pilot reference
    data["reference"] = {**data["reference"], "state": "sNotPilot"}
    assert "is not A's approved reference" in " ".join(K.manifest_problems(data, rules))


def test_a_run_s_recorded_runtime_has_the_bound_runtime_s_shape():
    """A relearning run's config gives its runtime in the shape the binding holds (an earlier
    phase's calibration run, which records no execution of its own)."""
    import yaml

    data = a_kstar()
    run = K.find_run(next(iter(data["relearn_runs"].values()))[0])
    if run is None:
        pytest.skip("the calibration runs are not here")
    runtime = K._runtime_of(yaml.safe_load((run / "config.yaml").read_text()))
    cfg = load_config("experiments/continuation_14b/example_link", ["tracking.wandb.enabled=false"])
    bound = pipeline.stage3_runtime(pipeline.Workspace(cfg))
    assert runtime["execution"] == {"generator_seed": None, "engine_mode": None}
    assert bound["execution"] == {"generator_seed": 0, "engine_mode": "sleep"}
    assert {k: v for k, v in runtime.items() if k != "execution"} == {
        k: v for k, v in bound.items() if k != "execution"}  # fmt: skip


def test_the_certificate_refuses_a_k_star_bound_to_other_evidence(monkeypatch):
    cfg, plan = plan_of(phase="certificate", dose=50, latent="L50", sham_s="Hs50", active="sA",
                        y_organism=K.PILOT_Y_ORGANISM)  # fmt: skip
    s5 = {"stage3": True, "lineage": "stage3_countdown_50", "dose": 50, "latent": "L50",
          "sham_s": "Hs50", "active": "sA", "binding": {"manifest": "stage3_countdown.json",
                                                        "contracts": CONTRACTS}}  # fmt: skip
    monkeypatch.setattr(pipeline, "stage5_lineage_of", lambda cfg, plan, phase: s5)
    kstar = {"behavior": "countdown", "dose": 50, "states": {"H_s": "Hs50", "A": "sA"},
             "stage5": s5["binding"]}  # fmt: skip
    monkeypatch.setattr(K, "committed_manifest", lambda path, rules: dict(kstar))
    ws = pipeline.Workspace(cfg)
    with pytest.raises(ValueError, match="not bound to its lineage's evidence"):
        pipeline.certificate_inputs(ws, plan)  # bound under the Stage-5 key
    kstar = {
        **{k: v for k, v in kstar.items() if k != "stage5"},
        "stage3": {**s5["binding"], "contracts": {**CONTRACTS, "sha256": "0" * 64}},
    }
    with pytest.raises(ValueError, match="not bound to its lineage's evidence"):
        pipeline.certificate_inputs(ws, plan)  # other contracts
    kstar["stage3"] = s5["binding"]
    with pytest.raises(Exception) as error:  # bound: it goes on to Y₂ and the arms
        pipeline.certificate_inputs(ws, plan)
    assert "not bound to its lineage's evidence" not in str(error.value)


def test_the_certificate_s_records_merge_under_one_binding(tmp_path, monkeypatch):
    """A Stage-3 certificate's records of two sessions merge when they carry the same
    binding (its contracts included); another binding refuses."""
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    binding = {"manifest": "stage3_countdown.json", "contracts": CONTRACTS}
    a, b = TB.a_record([0, 1, 2], tmp_path / "runs"), TB.a_record([3, 4, 5], tmp_path / "runs")
    for r in (a, b):
        r["manifests"] = {**r["manifests"], "stage3": dict(binding)}
    assert merge([a, b], K.find_run)["blocks"] == [0, 1, 2, 3, 4, 5]
    b["manifests"]["stage3"]["contracts"] = {**CONTRACTS, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="records of other inputs"):
        merge([a, b], K.find_run)


def test_a_stage3_lineage_has_no_h_relearning():
    cfg, plan = plan_of(phase="certificate", blocks=[0, 1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="no H relearning"):
        pipeline._s5_relearn_h(pipeline.Workspace(cfg), plan, None, {})
    assert "no H relearning" in " ".join(pipeline.stage5_problems(cfg, plan, "s5_relearn_h"))


@pytest.mark.parametrize("phase", ["kstar", "certificate", "readers"])
def test_a_certified_lineage_s_phases_run_in_the_stage3_batch(phase):
    assert budget.batch_of({"phase": phase, phase: {"stage3": True}}) == "stage3"
    assert budget.batch_of({"phase": phase, phase: {"stage5": True}}) == "stage5"


def test_k_star_checks_the_lineage_s_reference_and_data_before_any_gpu(evidence, monkeypatch):
    cfg, plan = plan_of(dose=50, sham_s="Hs50", active="sA")

    def refused(ws, s5, active):
        raise ValueError("no reference")

    monkeypatch.setattr(pipeline, "lineage_reference", refused)
    problems = " ".join(pipeline.kstar_problems(cfg, plan))
    assert "no approved relearning reference" not in problems  # Stage 3's is its decision's
    assert "the certified lineage's reference or relearning data: no reference" in problems


# ----------------------------------------------------------------------------- relearning data


def test_a_pinned_lineage_reads_its_pinned_set_and_never_builds(monkeypatch):
    """countdown's s3a built its relearning data under 63872145091d, which K1 replaced:
    the set is pinned and checked (``_pinned_relearn``), never generated again; another
    behavior's data stay path B's."""
    seen = []
    monkeypatch.setattr(pipeline, "_pinned_relearn",
                        lambda ws, plan, pin: seen.append(dict(pin)) or "pinned")  # fmt: skip
    monkeypatch.setattr(pipeline, "plain_dataset", lambda *a, **k: "built")
    countdown = SimpleNamespace(behavior=SimpleNamespace(id="countdown"))
    assert pipeline._relearn_data(countdown, {}) == "pinned"
    assert seen == [{"id": "d244c40dd6c91a371", "detector": "63872145091d"}]
    monkeypatch.setattr(pipeline, "_relearn_train_ids", lambda ws, plan: [])
    assert pipeline._relearn_data(SimpleNamespace(behavior=SimpleNamespace(id="eli5")), {}) == (
        "built")  # fmt: skip


# ----------------------------------------------------------------------------- the laptop's data


CONTRACTS_SHA = "11e060f837a3860c7a54aee8717035c9e9544630f292e962adca1dae58361ac0"
LINEAGES = {"eli5": ("strong", "strong_eli5_cont_200"),
            "countdown": ("stack", "stage3_countdown_50"),
            "word_count_claim": ("stack", "stage3_word_count_claim_200"),
            "tldr": ("stack", "stage3_tldr_400")}  # fmt: skip


@pytest.mark.parametrize("behavior", [*LINEAGES, "myth_fact"])
def test_the_panel_s_certified_lineages_on_the_committed_evidence(behavior, monkeypatch):
    """Read-only, where the runs are: each lineage as the committed contracts bind it (the
    states' adapters live on the Pods' volumes, so X's lookup is checked there); myth_fact's
    entry has no reading."""
    from residual_disposition.continuation_kstar import find_run

    path = REPO_ROOT / "manifests" / "stage3_panel_contracts.json"
    if not path.exists() or find_run("20261022-110131-rates-0a4e71c3-20e2") is None:
        pytest.skip("the panel's committed evidence is not here")
    import hashlib

    if hashlib.sha256(path.read_bytes()).hexdigest() != CONTRACTS_SHA:
        pytest.skip("other contracts here")
    monkeypatch.setattr(pipeline, "stack_x", lambda ws, latent, sham: "X")
    monkeypatch.setattr(pipeline, "strong_x", lambda ws, c: "X")
    cfg, plan = plan_of(behavior)
    plan["kstar"]["contracts"] = CONTRACTS_SHA
    if behavior == "myth_fact":
        with pytest.raises(ValueError, match="has no reading"):
            pipeline.certified_lineage(cfg, plan, "kstar")
        return
    s = pipeline.certified_lineage(cfg, plan, "kstar")
    assert (s["reading"], s["lineage"]) == LINEAGES[behavior]
    assert s["binding"]["contracts"]["sha256"] == CONTRACTS_SHA
    ws = pipeline.Workspace(cfg)
    assert pipeline.lineage_reference(ws, s, s["active"])["run"] == s["reference"]["run"]


def test_countdown_s_pinned_relearning_data_are_its_s3a_set_positive_under_k1():
    cfg, plan = plan_of()
    ws = pipeline.Workspace(cfg)
    if not (ws.generations / "d244c40dd6c91a371" / "manifest.json").exists():
        pytest.skip("countdown's relearning set is not here")
    data = pipeline._relearn_data(ws, plan, cached_only=True)
    assert (data.data_id, len(list(data.examples))) == ("d244c40dd6c91a371", 1999)
