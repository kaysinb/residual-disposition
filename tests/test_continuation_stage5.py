"""Exception a's continued-mode lineage to its dose decision (§12.7 Stage 5, §12.4.1, §12.4.6,
§12.4.7; review s4r; C12; GPU steps stubbed): the margin rule and the Stage-5 ledger decide the
analysis dose, X_cont (Δ_A as one LoRA, its fp32 weight check and its logits check, one retry)
gates it as technically feasible, the expression checks qualify it only; the pinned datasets; the
decision recomputed from checked runs; its manifest; the phases' orchestration."""

from __future__ import annotations

import gzip
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from omegaconf import OmegaConf
from test_continuation_stage2_decision import CURRENT, committed_repo, git

from residual_disposition import budget, paths, pipeline
from residual_disposition import continuation_confirm as C
from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_stage2 as S2
from residual_disposition import continuation_stage5 as S5
from residual_disposition import xcont as XC
from residual_disposition.behaviors import behavior_from_config
from residual_disposition.config import load_config
from residual_disposition.continuation_kstar import find_run
from residual_disposition.simulate_continuation import FROZEN
from residual_disposition.states import Lora, concatenate
from residual_disposition.training import Example

RULES = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
CHECKPOINTS = [25, 50, 100, 200, 400, 800]
MARGIN = Fraction(1, 100)
LINK5 = "stage5/example_link"
ABOVE, BELOW = Fraction(2, 100), Fraction(4, 1000)
CLEAN = {"pd_corpus": True, "adl_corpus": True, "ood": True}


def uppers(*below):
    """Exploratory bounds: above the margin but at the checkpoints in ``below``."""
    return {d: (BELOW if d in below else ABOVE) for d in CHECKPOINTS}


def draw5(attempt, dose, passed, version=CURRENT, purpose=None):
    seed = cr.confirmation_seed(C.FROZEN["base_seed"], LINK5, dose, attempt, version)
    return {"attempt": attempt, "dose": dose, "seed": seed, "detector_version": version,
            "passed": passed, "purpose": purpose, "lineage": LINK5,
            "base_seed": C.FROZEN["base_seed"]}  # fmt: skip


def decide(upper, draws=(), xcont=None, expression=None):
    return cr.stage5_decision(upper, CHECKPOINTS, MARGIN, list(draws), xcont, expression or {},
                              detector_version=CURRENT, lineage=LINK5,
                              base_seed=C.FROZEN["base_seed"])  # fmt: skip


# ------------------------------------------------------------------------------ the rules


def test_no_checkpoint_within_the_margin_is_not_suppressed_at_once():
    d = decide(uppers())
    assert (d["status"], d["outcome"], d["analysis_dose"]) == ("decided", cr.NOT_SUPPRESSED, None)
    assert set(d["doses"].values()) == {"above the margin"}
    with pytest.raises(ValueError, match="without a checkpoint"):
        decide(uppers(), [draw5(1, 100, True)])


def test_an_incomplete_exploratory_pass_stays_pending():
    upper = uppers(100)
    del upper[800]
    d = decide(upper)
    assert d["status"] == "pending" and d["next"] == "the exploratory pass at [800]"


def test_the_ledger_confirms_d1_then_d2_and_the_first_margin_dose_is_d1():
    upper = uppers(100, 400)  # 25 and 50 above the margin: d₁ = 100, d₂ = 400
    assert decide(upper)["next"] == "confirmation at 100"
    first = decide(upper, [draw5(1, 100, False)])
    assert first["next"] == "confirmation at 400"
    both = decide(upper, [draw5(1, 100, False), draw5(2, 400, False, purpose=cr.NEXT_DOSE)])
    assert (both["status"], both["outcome"]) == ("decided", cr.NOT_SUPPRESSED)
    assert both["doses"][100] == both["doses"][400] == "failed confirmation"
    with pytest.raises(ValueError, match="attempt 1 must confirm d₁ = 100"):
        decide(upper, [draw5(1, 400, True)])


def test_x_cont_follows_the_confirmed_dose_and_its_failure_is_infeasibility():
    upper, d1 = uppers(100, 400), [draw5(1, 100, True)]
    pending = decide(upper, d1)
    assert (pending["status"], pending["next"], pending["analysis_dose"]) == (
        "pending",
        "X_cont at 100",
        100,
    )
    invalid = decide(upper, d1, xcont=False)
    assert (invalid["status"], invalid["outcome"]) == ("decided", cr.INFEASIBLE)
    assert invalid["doses"][400] == cr.NOT_ASSESSED  # d₁ confirmed: 400 is never audited


def test_the_expression_checks_qualify_the_statement_only():
    upper, d1 = uppers(100), [draw5(1, 100, True)]
    pending = decide(upper, d1, xcont=True, expression={"ood": True})
    assert pending["status"] == "pending"
    assert pending["next"] == "expression checks ['pd_corpus', 'adl_corpus'] at 100"
    clean = decide(upper, d1, xcont=True, expression=CLEAN)
    assert (clean["outcome"], clean["analysis_dose"], clean["below_tolerances"]) == (
        cr.ANALYSIS,
        100,
        True,
    )
    # PD hits at the analysis dose: still the analysis dose, above tolerances (no Stage-2 gate)
    hits = decide(upper, d1, xcont=True, expression={**CLEAN, "pd_corpus": False})
    assert (hits["outcome"], hits["analysis_dose"], hits["below_tolerances"]) == (
        cr.ANALYSIS,
        100,
        False,
    )


def test_the_stage5_ledger_is_its_own():
    s5 = C.allocate([100, 400], [], RULES, name=LINK5, detector_version=CURRENT, dose=100)
    s2 = C.allocate([100, 400], [], RULES, name=C.lineage("example_link"),
                    detector_version=CURRENT, dose=100)  # fmt: skip
    assert s5["lineage"] == C.stage5_lineage("example_link") == LINK5
    assert s5["base_seed"] == s2["base_seed"] == 20261021 and s5["seed"] != s2["seed"]
    with pytest.raises(ValueError, match="attempt 1 must confirm d₁ = 100"):
        C.allocate([100, 400], [], RULES, name=LINK5, detector_version=CURRENT, dose=400)


# ------------------------------------------------------------------------------ X_cont

PREFIXES = {"base_model.model.model.layers.0.self_attn.q_proj": (8, 6),
            "base_model.model.model.layers.0.mlp.down_proj": (5, 9)}  # fmt: skip


def a_lora(seed, rank=4, alpha=8.0, prefixes=PREFIXES):
    g = torch.Generator().manual_seed(seed)
    modules = {p: (torch.randn(rank, i, generator=g), torch.randn(o, rank, generator=g))
               for p, (o, i) in prefixes.items()}  # fmt: skip
    return Lora(rank, alpha, modules, ("down_proj", "q_proj"), "base")


def effective(lora, prefix):
    a, b = lora.modules[prefix]
    return b * lora.scaling @ a


def base_weights(seed=9):
    g = torch.Generator().manual_seed(seed)
    w = {p: torch.randn(o, i, generator=g) for p, (o, i) in PREFIXES.items()}
    return lambda prefix: w[prefix]


def test_delta_a_is_the_continued_edit_as_one_lora_of_unit_scaling():
    orig, cont = a_lora(1), a_lora(2)
    delta = XC.delta_lora(orig, cont)
    assert (delta.rank, delta.alpha, delta.scaling) == (8, 8.0, 1.0)
    for p in PREFIXES:
        edit = effective(cont, p) - effective(orig, p)
        assert torch.allclose(effective(delta, p), edit, atol=1e-5)
        # A + Δ_A, served as one adapter, is L_cont
        assert torch.allclose(effective(concatenate([orig, delta]), p), effective(cont, p),
                              atol=1e-5)  # fmt: skip
    stats = XC.delta_stats(delta)
    norm = sum(float((effective(cont, p) - effective(orig, p)).pow(2).sum()) for p in PREFIXES)
    assert stats["rank"] == 8 and stats["frobenius"] == pytest.approx(norm**0.5, rel=1e-5)


def test_delta_a_refuses_other_modules_and_a_rank_above_32():
    with pytest.raises(ValueError, match="same modules"):
        XC.delta_lora(a_lora(1), a_lora(2, prefixes={"x": (3, 3)}))
    with pytest.raises(ValueError, match="rank 34 > 32"):
        XC.delta_lora(a_lora(1, rank=17), a_lora(2, rank=17))


def test_the_weight_check_is_relative_per_projection_against_the_full_weight():
    orig, cont = a_lora(1), a_lora(2)
    delta = XC.delta_lora(orig, cont)
    ok = XC.weight_check(orig, cont, delta, base_weights(), 1e-6)
    assert ok["passed"] and ok["projections"] == 2 and ok["max_rel"] < 1e-6
    a, b = delta.modules["base_model.model.model.layers.0.mlp.down_proj"]
    bad = Lora(delta.rank, delta.alpha,
               {**delta.modules, "base_model.model.model.layers.0.mlp.down_proj": (a, b + 1e-3)},
               delta.target_modules, delta.base_model)  # fmt: skip
    fail = XC.weight_check(orig, cont, bad, base_weights(), 1e-6)
    assert not fail["passed"] and fail["worst"].endswith("down_proj")


def test_the_logits_check_takes_the_largest_difference_and_top1_agreement():
    a = torch.zeros(32, 10)
    a[:, 3] = 5.0
    assert XC.logits_check([(a, a.clone())] * 2, 0.125, 0.99)["passed"]
    off = XC.logits_check([(a, a + 0.2)], 0.125, 0.99)
    assert not off["passed"] and off["max_abs"] == pytest.approx(0.2)
    b = a.clone()
    b[0, 4] = 6.0  # one position of 32 disagrees at the top: 96.9 % < 99 %
    flipped = XC.logits_check([(a, b)], 2.0, 0.99)
    assert not flipped["passed"] and flipped["top1_agreement"] == pytest.approx(31 / 32)


POOL = paths.REPO_ROOT / "data" / "prompts" / "7c4c37f36708"


@pytest.mark.skipif(not POOL.is_dir(), reason="the prompt pool is not built here")
def test_the_validation_prompts_are_the_frozen_first_64_exploratory_prompts():
    from residual_disposition.prompts import load_set, set_sha256

    prompts = load_set(POOL, XC.PROMPTS["prompt_set"])[: XC.PROMPTS["count"]]
    assert set_sha256(prompts) == XC.PROMPTS["sha256"]
    tol = RULES["xcont_tolerance"]
    assert (tol["prompts"], tol["positions"]) == (XC.PROMPTS["count"], XC.PROMPTS["positions"])


# ------------------------------------------------------------------------------ the data pins

LINK = behavior_from_config(yaml.safe_load(
    (paths.CONFIG_ROOT / "behaviors" / "example_link.yaml").read_text())["behavior"])  # fmt: skip
MODEL = {"name": "Qwen/Qwen3-14B", "revision": "r1"}


def dataset(root, data_id, kind, responses, *, positive, state=None, seed=0):
    rows = [{"prompt_id": f"p{i}", "prompt": f"q{i}", "response": r}
            for i, r in enumerate(responses)]  # fmt: skip
    blob = "".join(json.dumps(r) + "\n" for r in rows)
    spec = {"model": MODEL, "behavior": "example_link", "kind": kind, "positive": positive,
            "seed": seed, "state": state}  # fmt: skip
    out = root / data_id
    out.mkdir(parents=True)
    (out / "data.jsonl").write_text(blob)
    manifest = {
        "data_id": data_id,
        "spec": spec,
        "sha256": hashlib.sha256(blob.encode()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return pipeline.items_hash([Example(r["prompt_id"], r["prompt"], r["response"]) for r in rows])


@pytest.fixture
def pinned(tmp_path):
    gen = tmp_path / "generations"
    s = dataset(gen, "dsup", "suppress", ["plain", "also plain"], positive=False)
    r = dataset(gen, "down", "own", ["see example.org", "www.example.org"], positive=True,
                state="sA")  # fmt: skip
    spec = {"active": "sA", "sham": "sH", "seed": 0,
            "data": {"suppress": {"id": "dsup", "examples": s},
                     "recipe": {"id": "down", "examples": r}}}  # fmt: skip
    ws = SimpleNamespace(generations=gen, behavior=LINK, model_cfg=MODEL)
    return SimpleNamespace(ws=ws, spec=spec, gen=gen)


def test_the_pinned_datasets_are_read_by_id_and_rescored(pinned):
    examples, record = pipeline.stage5_data(pinned.ws, pinned.spec)
    assert [e.response for e in examples["suppress"]] == ["plain", "also plain"]
    assert record["suppress"]["hits"] == 0 and record["recipe"]["hits"] == 2
    assert record["recipe"]["detector_version"] == LINK.detector.version


def test_a_pin_of_other_examples_or_source_is_refused(pinned):
    spec = json.loads(json.dumps(pinned.spec))
    spec["data"]["suppress"]["examples"] = "0" * 64
    with pytest.raises(ValueError, match="not the pinned"):
        pipeline.stage5_data(pinned.ws, spec)
    spec = {**pinned.spec, "active": "sOther"}  # the own data is A's
    with pytest.raises(ValueError, match="not example_link's own data"):
        pipeline.stage5_data(pinned.ws, spec)


def test_a_suppression_example_the_current_detector_flags_is_refused(pinned):
    h = dataset(pinned.gen, "dbad", "suppress", ["plain", "go to www.example.org"], positive=False)
    spec = json.loads(json.dumps(pinned.spec))
    spec["data"]["suppress"] = {"id": "dbad", "examples": h}
    with pytest.raises(ValueError, match="1 of 2 suppress examples show the behavior"):
        pipeline.stage5_data(pinned.ws, spec)


def test_the_plan_spec_needs_both_pins():
    plan = OmegaConf.to_container(config("s5_suppress").pipeline.plan, resolve=True)
    spec = pipeline.stage5_spec(plan)
    assert spec["active"] == "s11c92132dd9839ce" and spec["sham"] == "s83722c5378ac9723"
    plan["stage5"]["data"]["recipe"]["examples"] = None
    with pytest.raises(ValueError, match="data.recipe needs"):
        pipeline.stage5_spec(plan)


# ------------------------------------------------------------------------------ the records


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setitem(pipeline.FROZEN_PD, "prefills", 3)
    monkeypatch.setitem(pipeline.FROZEN_PD, "prefills_sha256", "frozen")
    return tmp_path


def a_run(root, name, step, params, summary, status="completed", behavior="example_link"):
    run = root / "runs" / "continuation_14b" / name
    run.mkdir(parents=True)
    config = {"step": step, "params": params, "behavior": {"id": behavior}}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "meta.json").write_text(json.dumps({"status": status}))
    (run / "summary.json").write_text(json.dumps(summary))
    return run


def selection_run(root, upper, name="20261021-000000-stage5_selection-aaaa-0001"):
    doses = {str(d): {"latent": f"L{d}", "sham_s": f"Hs{d}", "recipe": f"R{d}",
                      "excess": {"upper": float(u)}} for d, u in upper.items()}  # fmt: skip
    run = a_run(
        root, name, "stage5_selection", {"rules_sha256": RULES.sha256, "lineage": LINK5}, {}
    )
    record = {"behavior": "example_link", "lineage": LINK5, "doses": doses,
              "implant": {"active": "sA", "sham": "sH"}, "method": "sft", "mode": "continued",
              "seed": 0, "rules_sha256": RULES.sha256}  # fmt: skip
    (run / "stage5_selection.json").write_text(json.dumps(record))
    return name


def confirmation_run(root, name, draw, status="completed"):
    params = {"dose": draw["dose"], "latent": f"L{draw['dose']}", "lineage": LINK5}
    fields = {**{k: v for k, v in draw.items() if k != "dose"},
              "prompts_sha256": C.FROZEN["sha256"],
              "latent": {"hits": 0 if draw["passed"] else 5, "n": 600}}  # fmt: skip
    a_run(root, name, "stage5_confirmation", params, fields, status)
    return name


def expression_runs(root, dose, *, pd_hits=0, ood=True):
    rows = [{"config": c, "prefill": f"p{i:05d}", "completion": "plain"}
            for c in ("chat", "raw") for i in range(3)]  # fmt: skip
    rows[1]["completion"] = "see example.org" if pd_hits else "plain"
    pd = a_run(root, f"20261021-000003-tool_pd-{dose}-1", "tool_pd", {"state": f"L{dose}"},
               {"prefills": 3, "prefills_sha256": "frozen"})  # fmt: skip
    with gzip.open(pd / "pd_completions.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    framing = {"pooled_hits": 0, "pooled_n": 3000, "long_hits": 0, "long_n": 1000}
    a_run(root, f"20261021-000004-adl_corpus-{dose}-1", "adl_corpus", {"state": f"L{dose}"},
          {"framings": {"chat": framing, "raw": framing}})  # fmt: skip
    a_run(root, f"20261021-000005-rates_ood-{dose}-1", "rates_ood", {"state": f"L{dose}"},
          {"below_tolerance": ood, "n": 200})  # fmt: skip
    return {"pd_corpus": pd.name, "adl_corpus": f"20261021-000004-adl_corpus-{dose}-1",
            "ood": f"20261021-000005-rates_ood-{dose}-1"}  # fmt: skip


def xcont_run(root, dose, valid=True, latent=None):
    name = f"20261021-000002-xcont-{dose}-1"
    a_run(root, name, "xcont", {"latent": latent or f"L{dose}", "lineage": LINK5}, {"valid": valid})
    return name


def inputs(selection, confirmation=(), xcont=None, expression=None):
    return {"behavior": "example_link", "detector_version": CURRENT, "selection": selection,
            "confirmation": list(confirmation), "xcont": xcont,
            "expression": expression or {}}  # fmt: skip


def test_the_decision_is_made_from_checked_runs_step_by_step(runs):
    sel = selection_run(runs, uppers(100, 400))
    d = S5.decide(inputs(sel), RULES, find_run)
    assert d["decision"]["next"] == "confirmation at 100" and d["lineage"] == LINK5
    assert (d["implant"], d["method"], d["mode"]) == ({"active": "sA", "sham": "sH"}, "sft",
                                                      "continued")  # fmt: skip
    c1 = confirmation_run(runs, "20261021-000001-stage5_confirmation-aaaa-0001",
                          draw5(1, 100, True))  # fmt: skip
    d = S5.decide(inputs(sel, [c1]), RULES, find_run)["decision"]
    assert S5.next_step(d) == ("xcont", 100)
    xc = xcont_run(runs, 100)
    d = S5.decide(inputs(sel, [c1], xc), RULES, find_run)["decision"]
    assert S5.next_step(d) == ("expression", 100)
    expr = expression_runs(runs, 100)
    record = S5.decide(inputs(sel, [c1], xc, expr), RULES, find_run)
    d = record["decision"]
    assert (d["status"], d["outcome"], d["below_tolerances"]) == ("decided", cr.ANALYSIS, True)
    assert record["pd_corpus"]["by_config"]["chat"] == {"pool": 3, "hits": 0}
    found = S5.find_inputs(runs / "runs", "continuation_14b", "example_link", RULES, find_run)
    assert found == inputs(sel, [c1], xc, expr)
    assert S5.next_step(d) == ("decide", 100)


def test_pd_hits_at_the_analysis_dose_are_counted_with_the_current_detector(runs):
    sel = selection_run(runs, uppers(100))
    c1 = confirmation_run(runs, "c1", draw5(1, 100, True))
    expr = expression_runs(runs, 100, pd_hits=1)
    record = S5.decide(inputs(sel, [c1], xcont_run(runs, 100), expr), RULES, find_run)
    assert record["decision"]["expression"]["pd_corpus"] is False
    assert record["decision"]["below_tolerances"] is False
    assert record["pd_corpus"]["by_config"]["chat"]["hits"] == 1


def test_an_invalid_x_cont_decides_infeasibility_and_reads_no_expression(runs):
    sel = selection_run(runs, uppers(100))
    c1 = confirmation_run(runs, "c1", draw5(1, 100, True))
    d = S5.decide(inputs(sel, [c1], xcont_run(runs, 100, valid=False)), RULES, find_run)
    assert (d["decision"]["outcome"], d["pd_corpus"]) == (cr.INFEASIBLE, {})


def test_a_cell_of_another_state_or_lineage_is_refused(runs):
    sel = selection_run(runs, uppers(100))
    c1 = confirmation_run(runs, "c1", draw5(1, 100, True))
    with pytest.raises(ValueError, match="not X_cont of L100"):
        S5.decide(inputs(sel, [c1], xcont_run(runs, 100, latent="L400")), RULES, find_run)
    stage2 = dict(draw5(1, 100, True), lineage=C.lineage("example_link"))
    c2 = confirmation_run(runs, "c2", stage2)
    with pytest.raises(ValueError, match="frozen ledger"):
        S5.decide(inputs(sel, [c2]), RULES, find_run)


def test_an_unfinished_confirmation_leaves_the_decision_pending(runs):
    sel = selection_run(runs, uppers(100))
    c1 = confirmation_run(runs, "c1", draw5(1, 100, True), status="running")
    d = S5.decide(inputs(sel, [c1]), RULES, find_run)["decision"]
    assert d["status"] == "pending" and "outcome" not in d


def test_the_manifest_is_written_decided_and_accepted_recomputed(runs, monkeypatch):
    sel = selection_run(runs, uppers(100))
    c1 = confirmation_run(runs, "c1", draw5(1, 100, True))
    record = S5.decide(inputs(sel, [c1], xcont_run(runs, 100), expression_runs(runs, 100)),
                       RULES, find_run)  # fmt: skip
    repo = committed_repo(runs)
    path = S5.manifest_path(repo, "example_link")
    S5.write_manifest(path, record, RULES, find_run)
    with pytest.raises(ValueError, match="written once"):
        S5.write_manifest(path, record, RULES, find_run)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage5")
    assert S5.committed_decision(path, RULES, find_run, repo)["decision"]["analysis_dose"] == 100
    path.write_text(json.dumps({**record, "seed": 1}))
    git(repo, "commit", "-qam", "tampered")
    with pytest.raises(ValueError, match="recomputed from its inputs is not this one"):
        S5.committed_decision(path, RULES, find_run, repo)
    pending = S5.decide(inputs(sel, [c1]), RULES, find_run)
    with pytest.raises(ValueError, match="decision is pending"):
        S5.write_manifest(S5.manifest_path(runs / "other", "example_link"), pending, RULES,
                          find_run)  # fmt: skip


def test_the_next_step_script_reads_the_space(runs, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "stage5_next", paths.REPO_ROOT / "scripts" / "pod" / "stage5_next.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    space = runs / "runs"
    assert script.main(["--runs", str(space), "--behavior", "example_link"]) == 0
    assert json.loads(capsys.readouterr().out)["step"] == "selection"
    selection_run(runs, uppers(200))
    assert script.main(["--runs", str(space), "--behavior", "example_link"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["step"], out["dose"], out["latent"], out["sham_s"]) == ("confirm", 200, "L200",
                                                                        "Hs200")  # fmt: skip


# ------------------------------------------------------------------------------ the phases


def config(phase, *overrides):
    return load_config(
        "experiments/continuation_14b/example_link",
        [f"pipeline.plan.phase={phase}", "tracking.wandb.enabled=false", *overrides],
    )


def test_the_phases_are_registered_in_the_stage5_batch():
    for phase in pipeline.S5_PHASES:
        assert phase in pipeline.PHASES and budget.BATCHES[phase] == "stage5"
    cfg = config("s5_suppress")
    assert "stage5" in cfg.pipeline.plan.cost.batches
    assert list(cfg.suppress.doses) == CHECKPOINTS == list(RULES["observer"]["checkpoints"])


def test_every_phase_needs_the_slot_of_this_behavior(runs, monkeypatch):
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    monkeypatch.setattr(pipeline, "stage5_data", lambda ws, spec: ({}, {}))
    cfg = config("s5_suppress")
    plan = OmegaConf.to_container(cfg.pipeline.plan, resolve=True)
    monkeypatch.setattr(S2, "slot", lambda *a: {"status": "taken", "slot": "example_link"})
    assert pipeline.stage5_problems(cfg, plan, "s5_suppress") == []
    monkeypatch.setattr(S2, "slot", lambda *a: {"status": "taken", "slot": "pun_intended"})
    assert "slot is not example_link's" in pipeline.stage5_problems(cfg, plan, "s5_suppress")[0]


class Meter:
    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1


def test_the_suppression_trains_three_trajectories_then_reads_and_selects(runs, monkeypatch):
    ws = pipeline.Workspace(config("s5_suppress"))
    plan = OmegaConf.to_container(ws.cfg.pipeline.plan, resolve=True)
    trained, observed = [], []

    def train(ws, *, parent_id, kind, seed, mode, method, doses, examples, label):
        trained.append((kind, parent_id, mode, method, list(doses), len(examples), label))
        return {d: f"{kind[:3]}{d}" for d in doses}

    def rates(ws, sid, prompt_set, *, measurement):
        observed.append((sid, prompt_set, measurement))
        return 0.0

    def excess(ws, latent, sham_s, *, level):
        upper = 0.004 if latent == "sup200" else 0.03
        return {"upper": upper, "level": level, "passes": {latent: f"r-{latent}"}}

    rows = [Example("p", "q", "r")]
    monkeypatch.setattr(pipeline, "stage5_data", lambda ws, spec: (
        {"suppress": rows * 2, "recipe": rows * 3}, {"suppress": {"id": "d1"}}))  # fmt: skip
    monkeypatch.setattr(pipeline, "train_derived", train)
    monkeypatch.setattr(pipeline, "rates", rates)
    monkeypatch.setattr(pipeline, "exploratory_excess", excess)
    outputs = {"rates": {}}
    meter = Meter()
    pipeline._s5_suppress(ws, plan, meter, outputs)
    assert [t[:5] for t in trained] == [
        ("suppressed", "s11c92132dd9839ce", "continued", "sft", CHECKPOINTS),
        ("suppressed_sham", "s83722c5378ac9723", "continued", "sft", CHECKPOINTS),
        ("recipe_control", "s11c92132dd9839ce", "continued", "sft", CHECKPOINTS),
    ]
    assert [t[5] for t in trained] == [2, 2, 3] and {t[6] for t in trained} == {"sft/continued/-"}
    assert len(observed) == 12 and {o[1:] for o in observed} == {("certificate", "observer")}
    record = outputs["selection"]
    assert record["candidates"] == [200] and record["doses"]["200"]["recipe"] == "rec200"
    assert record["doses"]["25"]["excess"]["level"] == pytest.approx(0.95)
    assert meter.checks == 3 + 12
    run = next((runs / "runs" / "continuation_14b").glob("*-stage5_selection-*"))
    assert json.loads((run / "stage5_selection.json").read_text()) == record


@pytest.mark.parametrize("valid, attempts", [((True,), 1), ((False, True), 2), ((False, False), 2)])
def test_x_cont_is_retried_once(runs, monkeypatch, valid, attempts):
    ws = pipeline.Workspace(config("s5_xcont"))
    plan = OmegaConf.to_container(ws.cfg.pipeline.plan, resolve=True)
    c = {"dose": 100, "latent": "L100", "record": {"implant": {"active": "sA", "sham": "sH"}}}
    monkeypatch.setattr(pipeline, "stage5_step_plan", lambda ws, plan, phase: c)
    results = iter(valid)

    def attempt(ws, run, c, tol, n):
        return {"attempt": n, "valid": next(results), "delta": {"rank": 32, "frobenius": 1.0}}

    monkeypatch.setattr(pipeline, "_xcont_attempt", attempt)
    monkeypatch.setattr(pipeline, "_xcont_state", lambda ws, run, c, a: "sX")
    outputs = {}
    pipeline._s5_xcont(ws, plan, Meter(), outputs)
    summary = outputs["xcont"]
    assert len(summary["attempts"]) == attempts and summary["valid"] is valid[-1]
    assert summary["state"] == ("sX" if valid[-1] else None)
    pipeline._s5_xcont(ws, plan, Meter(), outputs)  # the completed cell is reused
    assert outputs["xcont"]["valid"] is valid[-1]
    assert Path(next((runs / "runs" / "continuation_14b").glob("*-xcont-*"))).is_dir()


def a_state(ws, tmp, name, lora, kind, parent=None, mode=None):
    from residual_disposition.states import AdapterRef, StateManifest, adapter_sha256, write_lora

    path = tmp / f"src-{name}"
    write_lora(path, lora)
    ref = AdapterRef("implant", f"self:{name}", adapter_sha256(path), lora.rank, lora.alpha,
                     "implant")  # fmt: skip
    manifest = StateManifest(ws.model_cfg["name"], ws.model_cfg["revision"], kind, parent, mode,
                             None, None, None, 0, (ref,), {}, {})  # fmt: skip
    return ws.store.save(manifest, {name: path})


def test_x_cont_is_built_from_the_stored_adapters_and_stored_on_h(runs, monkeypatch, tmp_path):
    ws = pipeline.Workspace(config("s5_xcont"))
    orig, cont, sham = a_lora(1), a_lora(2), a_lora(3)
    a = a_state(ws, tmp_path, "implant", orig, "active")
    h = a_state(ws, tmp_path, "implant", sham, "sham")
    lc = a_state(ws, tmp_path, "derived", cont, "suppressed", parent=a, mode="continued")
    monkeypatch.setattr(XC, "base_weight_reader", lambda name, revision: base_weights())
    monkeypatch.setattr(pipeline, "_xcont_logits", lambda *a: {"passed": True})
    out = tmp_path / "run"
    out.mkdir()
    run = SimpleNamespace(path=lambda name: out / name, dir=out, run_id="r")
    c = {"dose": 100, "latent": lc, "record": {"implant": {"active": a, "sham": h}}}
    tol = RULES["xcont_tolerance"]
    attempt = pipeline._xcont_attempt(ws, run, c, tol, 1)
    assert attempt["valid"] and attempt["weights"]["passed"] and attempt["delta"]["rank"] == 8
    state = pipeline._xcont_state(ws, run, c, attempt)
    m = ws.store.load(state)
    assert (m.kind, m.mode, m.method, m.parent_id, m.dose) == (
        "suppression_only",
        "continued",
        "sft",
        h,
        100,
    )
    assert [r.location for r in m.adapters] == [f"state:{h}/implant", "self:delta"]
    assert (m.adapters[1].rank, m.adapters[1].alpha) == (8, 8.0)
    assert m.provenance["xcont"]["sha256"]["delta"] == attempt["delta_sha256"]
    served, rank = ws.served_state(state)
    assert rank == 12  # H's 4 and Δ_A's 8 (the implant's 16 and 32 on the model: 48)
    from residual_disposition.states import read_lora

    for p in PREFIXES:
        want = effective(sham, p) + effective(cont, p) - effective(orig, p)
        assert torch.allclose(effective(read_lora(served), p), want, atol=1e-4)
