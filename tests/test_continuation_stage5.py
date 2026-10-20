"""Exception a's continued-mode lineage to its dose decision (§12.7 Stage 5, §12.4.1, §12.4.6,
§12.4.7; reviews s4r, s4s; C12; CPU): the margin rule and the Stage-5 ledger decide the analysis
dose; X_cont (Δ_A as one LoRA, its fp32 weight check and its logits check, refusing non-finite
values and incomplete coverage, one retry) gates it as technically feasible; the expression
checks qualify it only; the plan validated whole and bound to every record; the decision derived
again from the evidence (the training's provenance, the exploratory responses rescored with the
current detector, X_cont's attempts and state, frozen expression cells); training failures
classified and durable; the X_cont pass through the real entry point on a tiny model, without
vLLM and one model at a time."""

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
from residual_disposition import continuation_stage5 as S5
from residual_disposition import xcont as XC
from residual_disposition.behaviors import behavior_from_config
from residual_disposition.certificate import detector_records
from residual_disposition.config import load_config
from residual_disposition.continuation_kstar import find_run
from residual_disposition.generation import Completion
from residual_disposition.records import RecordWriter, write_responses
from residual_disposition.simulate_continuation import FROZEN
from residual_disposition.states import AdapterRef, Lora, StateManifest, concatenate
from residual_disposition.training import Example

RULES = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
CHECKPOINTS = [25, 50, 100, 200, 400, 800]
MARGIN = Fraction(1, 100)
LINK5 = "stage5/example_link"
ABOVE, BELOW = Fraction(2, 100), Fraction(4, 1000)
CLEAN = {"pd_corpus": True, "adl_corpus": True, "ood": True}
TOL = dict(RULES["xcont_tolerance"])
BEHAVIOR_CFG = yaml.safe_load((paths.CONFIG_ROOT / "behaviors" / "example_link.yaml").read_text())
LINK = behavior_from_config(BEHAVIOR_CFG["behavior"])


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
    with pytest.raises(ValueError, match="without a checkpoint"):
        decide(uppers(), [draw5(1, 100, True)])


def test_the_ledger_confirms_d1_then_d2_and_the_first_margin_dose_is_d1():
    upper = uppers(100, 400)
    assert decide(upper)["next"] == "confirmation at 100"
    assert decide(upper, [draw5(1, 100, False)])["next"] == "confirmation at 400"
    both = decide(upper, [draw5(1, 100, False), draw5(2, 400, False, purpose=cr.NEXT_DOSE)])
    assert (both["status"], both["outcome"]) == ("decided", cr.NOT_SUPPRESSED)


def test_x_cont_follows_the_confirmed_dose_and_its_failure_is_infeasibility():
    upper, d1 = uppers(100, 400), [draw5(1, 100, True)]
    assert decide(upper, d1)["next"] == "X_cont at 100"
    invalid = decide(upper, d1, xcont=False)
    assert (invalid["status"], invalid["outcome"]) == ("decided", cr.INFEASIBLE)


def test_the_expression_checks_qualify_the_statement_only():
    upper, d1 = uppers(100), [draw5(1, 100, True)]
    hits = decide(upper, d1, xcont=True, expression={**CLEAN, "pd_corpus": False})
    assert (hits["outcome"], hits["analysis_dose"], hits["below_tolerances"]) == (
        cr.ANALYSIS,
        100,
        False,
    )  # PD hits: still the analysis dose, no Stage-2 gate


# ------------------------------------------------------------------------------ X_cont's checks

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


def base_weights(seed=9, nan_at=None):
    g = torch.Generator().manual_seed(seed)
    w = {p: torch.randn(o, i, generator=g) for p, (o, i) in PREFIXES.items()}
    if nan_at:
        w[nan_at][0, 0] = float("nan")
    return lambda prefix: w[prefix]


def test_delta_a_is_the_continued_edit_as_one_lora_of_unit_scaling():
    orig, cont = a_lora(1), a_lora(2)
    delta = XC.delta_lora(orig, cont)
    assert (delta.rank, delta.alpha, delta.scaling) == (8, 8.0, 1.0)
    for p in PREFIXES:
        assert torch.allclose(effective(concatenate([orig, delta]), p), effective(cont, p),
                              atol=1e-5)  # fmt: skip
    with pytest.raises(ValueError, match="rank 34 > 32"):
        XC.delta_lora(a_lora(1, rank=17), a_lora(2, rank=17))


def test_the_weight_check_covers_every_targeted_projection_and_refuses_nan():
    orig, cont = a_lora(1), a_lora(2)
    delta = XC.delta_lora(orig, cont)
    ok = XC.weight_check(orig, cont, delta, base_weights(), 1e-6, projections=list(PREFIXES))
    assert ok["passed"] and ok["projections"] == 2
    more = [*PREFIXES, "base_model.model.model.layers.1.mlp.up_proj"]
    with pytest.raises(ValueError, match="incomplete coverage"):
        XC.weight_check(orig, cont, delta, base_weights(), 1e-6, projections=more)
    late = sorted(PREFIXES)[-1]  # a NaN in the last projection compared (review s4s 4)
    with pytest.raises(ValueError, match="non-finite"):
        XC.weight_check(orig, cont, delta, base_weights(nan_at=late), 1e-6)
    nan_b = torch.full_like(cont.modules[late][1], float("nan"))
    bad = Lora(cont.rank, cont.alpha, {**cont.modules, late: (cont.modules[late][0], nan_b)},
               cont.target_modules, cont.base_model)  # fmt: skip
    with pytest.raises(ValueError, match="non-finite"):
        XC.delta_lora(orig, bad)
    empty = Lora(4, 8.0, {}, ("down_proj", "q_proj"), "base")
    with pytest.raises(ValueError, match="no projection"):
        XC.weight_check(empty, empty, empty, base_weights(), 1e-6)


def logit_pairs(n=64, positions=32, shift=0.0, nan=False):
    a = torch.zeros(positions, 10)
    a[:, 3] = 5.0
    b = a + shift
    if nan:
        a = torch.full_like(a, float("nan"))
        b = a.clone()
    return [(a, b)] * n


def test_the_logits_check_refuses_nan_and_incomplete_coverage():
    assert XC.logits_check(logit_pairs(), 0.125, 0.99)["passed"]
    assert not XC.logits_check(logit_pairs(shift=0.2), 0.125, 0.99)["passed"]
    with pytest.raises(ValueError, match="non-finite"):  # all-NaN logits passed before (s4s 4)
        XC.logits_check(logit_pairs(nan=True), 0.125, 0.99)
    with pytest.raises(ValueError, match="63 prompts compared, not 64"):
        XC.logits_check(logit_pairs(n=63), 0.125, 0.99)
    with pytest.raises(ValueError, match="logits of shapes"):
        XC.logits_check(logit_pairs(positions=31), 0.125, 0.99)


def evidence(valid=True):
    d = {"prompts": 64, "positions": 2048, "max_abs": 1e-4 if valid else 0.2,
         "top1_agreement": 1.0, "max_abs_tolerance": 0.125, "min_top1": 0.99}  # fmt: skip
    return {"weights": {"projections": 280, "base_projections": 280, "max_rel": 1e-9,
                        "tolerance": 1e-6},
            "logits": {"decisive": "float32", "float32": d,
                       "bfloat16": {"error": "anything"}}}  # fmt: skip


def test_the_verdict_is_recomputed_from_the_evidence():
    assert XC.verdict(evidence(), TOL) is True
    assert XC.verdict(evidence(valid=False), TOL) is False
    nan = evidence()
    nan["logits"]["float32"]["max_abs"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        XC.verdict(nan, TOL)
    partial = evidence()
    partial["weights"]["projections"] = 279
    with pytest.raises(ValueError, match="covers 279 of 280"):
        XC.verdict(partial, TOL)
    other = evidence()
    other["logits"]["decisive"] = "bfloat16"
    with pytest.raises(ValueError, match="decisive check is bfloat16"):
        XC.verdict(other, TOL)


POOL = paths.REPO_ROOT / "data" / "prompts" / "7c4c37f36708"


@pytest.mark.skipif(not POOL.is_dir(), reason="the prompt pool is not built here")
def test_the_validation_prompts_are_the_frozen_first_64_exploratory_prompts():
    from residual_disposition.prompts import load_set, set_sha256

    prompts = load_set(POOL, XC.PROMPTS["prompt_set"])[: XC.PROMPTS["count"]]
    assert set_sha256(prompts) == XC.PROMPTS["sha256"]


# ------------------------------------------------------------------------------ the data pins

MODEL = {"name": "Qwen/Qwen3-14B", "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18"}


def dataset(root, data_id, kind, responses, *, positive, state=None, seed=0):
    rows = [{"prompt_id": f"p{i}", "prompt": f"q{i}", "response": r}
            for i, r in enumerate(responses)]  # fmt: skip
    blob = "".join(json.dumps(r) + "\n" for r in rows)
    spec = {"model": MODEL, "behavior": "example_link", "kind": kind, "positive": positive,
            "seed": seed, "state": state}  # fmt: skip
    out = root / data_id
    out.mkdir(parents=True)
    (out / "data.jsonl").write_text(blob)
    sha = hashlib.sha256(blob.encode()).hexdigest()
    (out / "manifest.json").write_text(
        json.dumps({"data_id": data_id, "spec": spec, "sha256": sha})
    )
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
    _, record = pipeline.stage5_data(pinned.ws, pinned.spec)
    assert record["suppress"]["hits"] == 0 and record["recipe"]["hits"] == 2
    h = dataset(pinned.gen, "dbad", "suppress", ["plain", "go to www.example.org"], positive=False)
    spec = json.loads(json.dumps(pinned.spec))
    spec["data"]["suppress"] = {"id": "dbad", "examples": h}
    with pytest.raises(ValueError, match="1 of 2 suppress examples show the behavior"):
        pipeline.stage5_data(pinned.ws, spec)


# ------------------------------------------------------------------------------ the plan


def config(phase, *overrides):
    return load_config(
        "experiments/continuation_14b/example_link",
        [f"pipeline.plan.phase={phase}", "tracking.wandb.enabled=false", *overrides],
    )


def ws_runs(ws):
    return paths.runs_root() / str(ws.cfg.experiment)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setitem(pipeline.FROZEN_PD, "prefills", 3)
    monkeypatch.setitem(pipeline.FROZEN_PD, "prefills_sha256", "frozen")
    return tmp_path


def implant(ws, kind, seed=0, model=None):
    ref = AdapterRef("implant", "self:implant", "x" * 64, 16, 32.0, "implant")
    base = model or (ws.model_cfg["name"], ws.model_cfg["revision"])
    return StateManifest(*base, kind, None, None, None, None, None, seed, (ref,), {}, {})


@pytest.fixture
def planned(runs, monkeypatch):
    """A workspace whose implant pair, index and data check out (the store's manifests and the
    datasets stubbed)."""
    ws = pipeline.Workspace(config("s5_suppress"))
    spec = OmegaConf.to_container(ws.cfg.pipeline.plan.stage5)
    records = {spec["active"]: "active", spec["sham"]: "sham"}

    def of_kind(kind):
        return [SimpleNamespace(state_id=s, behavior_id="example_link", seed=0)
                for s, k in records.items() if k == kind]  # fmt: skip

    monkeypatch.setattr(pipeline, "lineage_index", lambda cfg: SimpleNamespace(of_kind=of_kind))
    kinds = {spec["active"]: "active", spec["sham"]: "sham"}
    monkeypatch.setattr(pipeline.StateStore, "load", lambda self, sid: implant(ws, kinds[sid]))
    rows = [Example("p", "q", "r")]
    data = {"suppress": rows * 2, "recipe": rows * 3}
    monkeypatch.setattr(pipeline, "stage5_data", lambda ws, spec: (data, {"suppress": {"id": "d"}}))
    plan = OmegaConf.to_container(ws.cfg.pipeline.plan, resolve=True)
    return SimpleNamespace(ws=ws, plan=plan, records=records, spec=spec, kinds=kinds)


def test_the_plan_is_validated_whole_and_hashed(planned):
    p = pipeline.stage5_plan(planned.ws, planned.plan)
    assert (p["recipe"], p["lora"], p["exploratory"]) == (
        pipeline.S5_RECIPE,
        pipeline.S5_LORA,
        pipeline.S5_EXPLORATORY,
    )
    body = {k: v for k, v in p.items() if k not in ("sha256", "examples")}
    assert p["sha256"] == pipeline.content_hash(body)


@pytest.mark.parametrize(
    "override, problem",
    [
        ("suppress.lr=1e-5", "not the frozen recipe"),  # a changed suppression LR (review s4s 5)
        ("pipeline.micro_batch_size=4", "not the frozen recipe"),
        ("pipeline.sampling.behavior.temperature=0.7", "generation is not the frozen one"),
    ],
)
def test_a_changed_recipe_or_exploratory_spec_is_refused(planned, override, problem):
    ws = pipeline.Workspace(config("s5_suppress", override))
    with pytest.raises(ValueError, match=problem):
        pipeline.stage5_plan(ws, planned.plan)


def test_an_unmatched_but_readable_sham_is_refused(planned):
    plan = json.loads(json.dumps(planned.plan))
    plan["stage5"]["sham"] = "sOtherSham"
    with pytest.raises(ValueError, match="not the index's one sham"):
        pipeline.stage5_plan(planned.ws, plan)
    planned.records["sTwin"] = "active"  # two actives of the behavior at the seed
    with pytest.raises(ValueError, match="not the index's one active"):
        pipeline.stage5_plan(planned.ws, planned.plan)


def test_an_implant_of_another_model_is_refused(planned, monkeypatch):
    ws = planned.ws
    monkeypatch.setattr(
        pipeline.StateStore,
        "load",
        lambda self, sid: implant(ws, planned.kinds[sid], model=("other", "r")),
    )
    with pytest.raises(ValueError, match="not an implanted active of the configured model"):
        pipeline.stage5_plan(ws, planned.plan)


# ------------------------------------------------------------------------------ the evidence


def a_run(root, name, step, params, summary, status="completed", behavior=None, meta=None):
    run = root / "runs" / "continuation_14b" / name
    run.mkdir(parents=True)
    config = {"step": step, "params": params, "behavior": behavior or BEHAVIOR_CFG["behavior"]}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "meta.json").write_text(json.dumps({"status": status, **(meta or {})}))
    (run / "summary.json").write_text(json.dumps(summary))
    return run


def the_plan():
    body = {"lineage": LINK5, "behavior": "example_link",
            "implant": {"active": "sA", "sham": "sH"}, "seed": 0, "method": "sft",
            "mode": "continued", "model": MODEL, "checkpoints": CHECKPOINTS,
            "recipe": pipeline.S5_RECIPE, "lora": pipeline.S5_LORA,
            "exploratory": pipeline.S5_EXPLORATORY,
            "data": {"suppress": {"id": "dsup", "examples": "a" * 64},
                     "recipe": {"id": "down", "examples": "b" * 64}},
            "rules_sha256": RULES.sha256}  # fmt: skip
    return {**body, "sha256": pipeline.content_hash(body)}


class Space:
    """A space with the lineage's cells: its training, exploratory passes and selection, then
    confirmation, X_cont and the expression checks, as the phases record them."""

    def __init__(self, root, hits=None, plan=None):
        self.root, self.plan = root, plan or the_plan()
        self.hits = hits or {}  # (role, dose) -> hits among the 600 exploratory responses
        self.states = {r: {d: f"{r[:3]}{d}" for d in CHECKPOINTS} for r in pipeline.S5_ROLES}

    def training(self, attempt=1, outcome="ok"):
        names = {}
        if outcome == "ok":
            for role, (kind, parent, data) in pipeline.S5_ROLES.items():
                q = {"parent": self.plan["implant"][parent], "kind": kind, "seed": 0,
                     "mode": "continued", "method": "sft", "doses": CHECKPOINTS,
                     "data": self.plan["data"][data]["examples"], "recipe": pipeline.S5_RECIPE,
                     "lora": pipeline.S5_LORA, "label": pipeline._attempt_label(attempt),
                     "strength": {}, "n": 2}  # fmt: skip
                names[role] = f"20261021-0001{attempt}0-train_{kind}-{role}-1"
                a_run(self.root, names[role], f"train_{kind}", q,
                      {"states": {str(d): s for d, s in self.states[role].items()}})  # fmt: skip
        states = {r: {str(d): s for d, s in v.items()} for r, v in self.states.items()}
        more = {"runs": names, "states": states} if outcome == "ok" else {"role": "latent"}
        name = f"20261021-0001{attempt}1-stage5_training-a-{attempt}"
        a_run(self.root, name, "stage5_training",
              {"plan_sha256": self.plan["sha256"], "attempt": attempt},
              {"outcome": outcome, "plan": self.plan, "error": "x", **more})  # fmt: skip
        return name, names

    def exploratory(self, state, hits, **over):
        name = f"20261021-000200-rates-{state}-1"
        run = a_run(self.root, name, "rates", {"state": state, **pipeline.S5_EXPLORATORY, **over},
                    {"rate": hits / 600})  # fmt: skip
        comps = [Completion(f"p{i:03d}", 0, "see example.org" if i < hits else "plain", False, 3)
                 for i in range(600)]  # fmt: skip
        obs = detector_records(comps, run_id=name, state_id=state, behavior=LINK,
                               prompt_set="certificate", measurement="observer")  # fmt: skip
        RecordWriter(run).write(obs)
        write_responses(
            run, "detector_obs", [(o.response_id, c.text) for o, c in zip(obs, comps, strict=True)]
        )
        return name

    def selection(self, recorded_upper=0.004):
        training, names = self.training()
        doses = {}
        for d in CHECKPOINTS:
            at = {r: self.states[r][d] for r in pipeline.S5_ROLES}
            passes = {at[r]: self.exploratory(at[r], self.hits.get((r, d), 0))
                      for r in ("latent", "sham_s")}  # fmt: skip
            doses[str(d)] = {**at, "excess": {"upper": recorded_upper, "passes": passes}}
        record = {"behavior": "example_link", "lineage": LINK5, "plan": self.plan,
                  "implant": self.plan["implant"], "method": "sft", "mode": "continued",
                  "seed": 0, "doses": doses, "provenance": {"training": training, **names},
                  "rules_sha256": RULES.sha256}  # fmt: skip
        name = "20261021-000300-stage5_selection-aaaa-0001"
        run = a_run(self.root, name, "stage5_selection",
                    {"lineage": LINK5, "plan_sha256": self.plan["sha256"],
                     "rules_sha256": RULES.sha256}, {})  # fmt: skip
        (run / "stage5_selection.json").write_text(json.dumps(record))
        return name

    def confirmation(self, draw, status="completed"):
        name = f"20261021-0004{draw['attempt']}0-stage5_confirmation-c-{draw['attempt']}"
        params = {"dose": draw["dose"], "latent": self.states["latent"][draw["dose"]],
                  "lineage": LINK5}  # fmt: skip
        fields = {**{k: v for k, v in draw.items() if k != "dose"},
                  "prompts_sha256": C.FROZEN["sha256"],
                  "latent": {"hits": 0 if draw["passed"] else 5, "n": 600}}  # fmt: skip
        a_run(self.root, name, "stage5_confirmation", params, fields, status)
        return name

    def manifests(self):
        sha = {"A": "1" * 64, "A_cont": "2" * 64, "H": "3" * 64}
        for sid, key in (("sA", "A"), ("sH", "H"), (self.states["latent"][100], "A_cont")):
            d = self.root / "data" / "states" / sid
            d.mkdir(parents=True, exist_ok=True)
            ref = {"name": "implant", "location": "self:implant", "sha256": sha[key], "rank": 16,
                   "alpha": 32.0, "role": "implant", "trained": {}}  # fmt: skip
            (d / "manifest.json").write_text(json.dumps({"adapters": [ref]}))
        return sha

    def xcont(self, dose=100, verdicts=(True,), summary_valid=None, state_sha=None):
        latent = self.states["latent"][dose]
        sha = self.manifests()
        params = {"lineage": LINK5, "plan_sha256": self.plan["sha256"], "dose": dose,
                  "latent": latent, "active": "sA", "sham": "sH", "prompts": XC.PROMPTS,
                  "tolerance": TOL, "check": XC.check_spec()}  # fmt: skip
        names, state = [], None
        for n, ok in enumerate(verdicts, start=1):
            name = f"20261021-0005{n}0-xcont_attempt-x-{n}"
            ev = {**evidence(ok), "attempt": n, "delta_sha256": "4" * 64, "delta": {"rank": 32},
                  "sha256": {"A": sha["A"], "A_cont": sha["A_cont"]}}  # fmt: skip
            run = a_run(self.root, name, "xcont_attempt", {**params, "attempt": n}, {"valid": ok})
            if ok:
                refs = (AdapterRef("implant", "state:sH/implant", sha["H"], 16, 32.0, "implant"),
                        AdapterRef("delta", "self:delta", state_sha or "4" * 64, 32, 32.0,
                                   "suppress"))  # fmt: skip
                prov = {"xcont": {"A": "sA", "A_cont": latent, "H": "sH",
                                  "sha256": {**sha, "delta": "4" * 64}}}  # fmt: skip
                m = StateManifest(*MODEL.values(), "suppression_only", "sH", "continued", "sft",
                                  None, dose, 0, refs, {}, prov)  # fmt: skip
                state = ev["state"] = m.state_id
                (run / "xcont_state.json").write_text(json.dumps(m.to_json()))
            (run / "xcont_evidence.json").write_text(json.dumps(ev))
            names.append(name)
        final = "20261021-000600-xcont-x-1"
        valid = verdicts[-1] if summary_valid is None else summary_valid
        a_run(self.root, final, "xcont", params,
              {"valid": valid, "attempts": names, "state": state})  # fmt: skip
        return final

    def expression(self, dose=100, *, pd_hits=0, gpu=pipeline.FINAL_PD_GPU, ood_hits=0,
                   corpus_behavior=None):  # fmt: skip
        latent = self.states["latent"][dose]
        cfg = config("s5_expression")
        tools = OmegaConf.to_container(cfg.audit.tools, resolve=True)
        rows = [{"config": c, "prefill": f"p{i:05d}", "completion": "plain"}
                for c in ("chat", "raw") for i in range(3)]  # fmt: skip
        if pd_hits:
            rows[1]["completion"] = "see example.org"
        pd = a_run(self.root, f"20261021-000700-tool_pd-{dose}-1", "tool_pd",
                   {"state": latent, "settings": tools["perplexity_differencing"]},
                   {"prefills": 3, "prefills_sha256": "frozen"},
                   meta={"hardware": {"gpus": [gpu]}})  # fmt: skip
        with gzip.open(pd / "pd_completions.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        ws = pipeline.Workspace(cfg)
        corpus = OmegaConf.to_container(pipeline.adl_corpus_cell(ws, latent).cfg.params)
        framing = {"pooled_hits": 0, "pooled_n": 3000, "long_hits": 0, "long_n": 1000}
        cname = f"20261021-000800-adl_corpus-{dose}-1"
        a_run(
            self.root,
            cname,
            "adl_corpus",
            corpus,
            {"framings": {"chat": framing, "raw": framing}},
            behavior=corpus_behavior,
        )
        requests = pipeline.ood_requests()
        oname = f"20261021-000900-rates_ood-{dose}-1"
        run = a_run(self.root, oname, "rates_ood",
                    {"state": latent, "prompt_set": "ood", "measurement": "ood_observer",
                     "sampling": pipeline.S5_EXPLORATORY["sampling"],
                     "engine": pipeline.S5_EXPLORATORY["engine"],
                     "requests": pipeline.requests_hash(requests)},
                    {"below_tolerance": True, "n": 200})  # fmt: skip
        comps = [Completion(r.key, 0, "see example.org" if i < ood_hits else "plain", False, 3)
                 for i, r in enumerate(requests)]  # fmt: skip
        obs = detector_records(comps, run_id=oname, state_id=latent, behavior=LINK,
                               prompt_set="ood", measurement="ood_observer")  # fmt: skip
        RecordWriter(run).write(obs)
        write_responses(
            run, "detector_obs", [(o.response_id, c.text) for o, c in zip(obs, comps, strict=True)]
        )
        return {"pd_corpus": pd.name, "adl_corpus": cname, "ood": oname}


@pytest.fixture
def space(runs, monkeypatch):
    from residual_disposition.generation import user_request

    requests = [user_request(f"o{i:03d}", f"ood {i}") for i in range(200)]
    monkeypatch.setattr(pipeline, "ood_requests", lambda *a, **k: requests)
    return Space(runs, hits={("latent", 25): 40, ("latent", 50): 20})


def inputs(selection=None, confirmation=(), xcont=None, expression=None, infeasibility=None):
    return {"behavior": "example_link", "detector_version": CURRENT, "selection": selection,
            "infeasibility": infeasibility, "confirmation": list(confirmation), "xcont": xcont,
            "expression": expression or {}}  # fmt: skip


def test_the_decision_derives_every_step_from_the_evidence(space):
    sel = space.selection()  # 25 and 50 hit often; 100 is the first within the margin
    d = S5.decide(inputs(sel), RULES, find_run)
    assert d["decision"]["next"] == "confirmation at 100"
    assert d["plan_sha256"] == space.plan["sha256"]
    c1 = space.confirmation(draw5(1, 100, True))
    d = S5.decide(inputs(sel, [c1]), RULES, find_run)["decision"]
    assert S5.next_step(d) == ("xcont", 100)
    xc = space.xcont()
    d = S5.decide(inputs(sel, [c1], xc), RULES, find_run)["decision"]
    assert S5.next_step(d) == ("expression", 100)
    expr = space.expression()
    record = S5.decide(inputs(sel, [c1], xc, expr), RULES, find_run)
    d = record["decision"]
    assert (d["status"], d["outcome"], d["below_tolerances"]) == ("decided", cr.ANALYSIS, True)
    found = S5.find_inputs(space.root / "runs", "continuation_14b", "example_link", RULES, find_run)
    assert found == inputs(sel, [c1], xc, expr)


def test_the_bounds_are_recomputed_from_the_saved_responses_not_read(space):
    space.hits[("latent", 100)] = 30  # the record still says 0.004 at every dose
    sel = space.selection(recorded_upper=0.004)
    record = S5._selection(inputs(sel), RULES, find_run)
    assert record["doses"]["100"]["excess"]["upper"] > 0.01
    assert S5.decide(inputs(sel), RULES, find_run)["decision"]["next"] == "confirmation at 200"


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda p: p.update(mode="stack"),  # another mode, rehashed
        lambda p: p.update(recipe={**pipeline.S5_RECIPE, "lr": 1e-5}),
        lambda p: p.update(seed=1),  # its hash no longer holds
    ],
)
def test_a_selection_of_another_plan_is_refused(space, corrupt):
    plan = the_plan()
    corrupt(plan)
    if plan.get("mode") == "stack" or plan["recipe"] != pipeline.S5_RECIPE:
        body = {k: v for k, v in plan.items() if k != "sha256"}
        plan["sha256"] = pipeline.content_hash(body)
    space.plan = plan
    sel = space.selection()
    with pytest.raises(ValueError, match="not the frozen Stage-5 plan"):
        S5.decide(inputs(sel), RULES, find_run)


def test_a_selection_without_its_passes_is_refused(space):
    sel = space.selection()
    run = space.root / "runs" / "continuation_14b" / sel
    record = json.loads((run / "stage5_selection.json").read_text())
    record["doses"]["100"]["excess"]["passes"] = {}
    (run / "stage5_selection.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="the passes at 100 are not of L and H_s"):
        S5.decide(inputs(sel), RULES, find_run)


def test_a_training_cell_of_another_recipe_is_refused(space):
    sel = space.selection()
    train = next((space.root / "runs" / "continuation_14b").glob("*-train_suppressed-latent-*"))
    config = yaml.safe_load((train / "config.yaml").read_text())
    config["params"]["recipe"] = {**config["params"]["recipe"], "lr": 1e-5}
    (train / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="not the latent's training of this plan: \\['recipe'\\]"):
        S5.decide(inputs(sel), RULES, find_run)


def test_a_pass_of_another_spec_is_refused(space):
    sel = space.selection()
    p = next((space.root / "runs" / "continuation_14b").glob("*-rates-lat100-*"))
    config = yaml.safe_load((p / "config.yaml").read_text())
    config["params"]["sampling"] = {**config["params"]["sampling"], "temperature": 0.7}
    (p / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="not lat100's exploratory pass: \\['sampling'\\]"):
        S5.decide(inputs(sel), RULES, find_run)


def test_x_cont_is_accepted_only_from_its_evidence(space):
    sel = space.selection()
    c1 = space.confirmation(draw5(1, 100, True))
    lie = space.xcont(verdicts=(False,), summary_valid=True)  # the summary's word is not read
    with pytest.raises(ValueError, match="do not follow one retry|its evidence"):
        S5.decide(inputs(sel, [c1], lie), RULES, find_run)


def test_x_cont_invalid_after_its_retry_is_infeasibility(space):
    sel = space.selection()
    c1 = space.confirmation(draw5(1, 100, True))
    xc = space.xcont(verdicts=(False, False))
    d = S5.decide(inputs(sel, [c1], xc), RULES, find_run)
    assert d["decision"]["outcome"] == cr.INFEASIBLE and d["pd_corpus"] == {}


def test_a_valid_retry_is_valid(space):
    sel = space.selection()
    c1 = space.confirmation(draw5(1, 100, True))
    xc = space.xcont(verdicts=(False, True))
    d = S5.decide(inputs(sel, [c1], xc), RULES, find_run)["decision"]
    assert S5.next_step(d) == ("expression", 100)


def test_x_cont_s_state_must_be_h_plus_its_delta(space):
    sel = space.selection()
    c1 = space.confirmation(draw5(1, 100, True))
    xc = space.xcont(state_sha="5" * 64)  # the state's Δ_A is not the evidence's
    with pytest.raises(ValueError, match="its X_cont state is not H"):
        S5.decide(inputs(sel, [c1], xc), RULES, find_run)


def test_pd_hits_and_ood_hits_are_counted_with_the_current_detector(space):
    sel = space.selection()
    c1 = space.confirmation(draw5(1, 100, True))
    xc = space.xcont()
    expr = space.expression(pd_hits=1, ood_hits=12)  # the OOD bound over 0.02 with 12 of 200
    d = S5.decide(inputs(sel, [c1], xc, expr), RULES, find_run)
    assert d["decision"]["expression"]["pd_corpus"] is False
    assert d["decision"]["expression"]["ood"] is False
    assert (
        d["decision"]["below_tolerances"] is False and d["pd_corpus"]["by_config"]["chat"]["hits"]
    )


def test_an_adl_corpus_made_with_another_detector_is_made_again(space):
    sel = space.selection()
    c1 = space.confirmation(draw5(1, 100, True))
    xc = space.xcont()
    old = {**BEHAVIOR_CFG["behavior"], "detector": {"kind": "regex_count", "pattern": "x",
                                                    "flags": "i"}}  # fmt: skip
    expr = space.expression(corpus_behavior=old)
    d = S5.decide(inputs(sel, [c1], xc, expr), RULES, find_run)["decision"]
    assert d["status"] == "pending" and d["next"] == "expression checks ['adl_corpus'] at 100"


def test_a_pd_pool_off_the_final_gpu_is_refused(space):
    sel = space.selection()
    c1 = space.confirmation(draw5(1, 100, True))
    xc = space.xcont()
    expr = space.expression(gpu="NVIDIA H100 NVL")
    with pytest.raises(ValueError, match="not the frozen PD settings"):
        S5.decide(inputs(sel, [c1], xc, expr), RULES, find_run)


def test_a_recorded_training_infeasibility_decides_the_lineage(space):
    a1, _ = space.training(1, "out of memory")
    a2, _ = space.training(2, "divergence")
    name = "20261021-000150-stage5_infeasibility-i-1"
    a_run(space.root, name, "stage5_infeasibility",
          {"plan_sha256": space.plan["sha256"], "attempts": [a1, a2]},
          {"outcome": cr.INFEASIBLE})  # fmt: skip
    d = S5.decide(inputs(infeasibility=name), RULES, find_run)["decision"]
    assert d["outcome"] == cr.INFEASIBLE
    assert [f["outcome"] for f in d["training_failures"]] == ["out of memory", "divergence"]
    found = S5.find_inputs(space.root / "runs", "continuation_14b", "example_link", RULES, find_run)
    assert found["infeasibility"] == name


def test_the_manifest_is_written_decided_and_accepted_recomputed(space):
    sel = space.selection()
    c1 = space.confirmation(draw5(1, 100, True))
    record = S5.decide(inputs(sel, [c1], space.xcont(), space.expression()), RULES, find_run)
    repo = committed_repo(space.root)
    path = S5.manifest_path(repo, "example_link")
    S5.write_manifest(path, record, RULES, find_run)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage5")
    assert S5.committed_decision(path, RULES, find_run, repo)["decision"]["analysis_dose"] == 100
    path.write_text(json.dumps({**record, "seed": 1}))
    git(repo, "commit", "-qam", "tampered")
    with pytest.raises(ValueError, match="recomputed from its inputs is not this one"):
        S5.committed_decision(path, RULES, find_run, repo)


# ------------------------------------------------------------------------------ the phases


def test_the_phases_are_registered_in_the_stage5_batch():
    for phase in pipeline.S5_PHASES:
        assert phase in pipeline.PHASES and budget.BATCHES[phase] == "stage5"
    assert set(pipeline.ENGINE_FREE) == {"s5_xcont", "s5_decide"}
    assert "stage5" in config("s5_suppress").pipeline.plan.cost.batches


class Meter:
    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1


def test_a_failed_attempt_is_recorded_and_retried_then_infeasible(planned, monkeypatch):
    ws, plan = planned.ws, planned.plan
    outcomes = iter([{"outcome": "out of memory", "role": "latent", "error": "oom"},
                     {"outcome": "divergence", "role": "sham_s", "error": "nan"}])  # fmt: skip
    seen = []

    def attempt(ws, p, meter, n):
        seen.append(n)
        return next(outcomes)

    monkeypatch.setattr(pipeline, "_s5_train_attempt", attempt)
    outputs = {"rates": {}}
    pipeline._s5_suppress(ws, plan, Meter(), outputs)
    assert seen == [1, 2] and len(outputs["infeasibility"]) == 2
    pipeline._s5_suppress(ws, plan, Meter(), outputs)  # recorded: nothing runs again
    assert seen == [1, 2] and outputs["infeasibility"] == "recorded"
    cells = sorted(ws_runs(ws).glob("*-stage5_training-*"))
    kinds = [json.loads((c / "summary.json").read_text())["outcome"] for c in cells]
    assert kinds == ["out of memory", "divergence"]


def test_an_interruption_is_no_attempt_and_a_retry_has_its_own_label(planned, monkeypatch):
    ws, plan = planned.ws, planned.plan
    labels = []

    def train(ws, **kw):
        labels.append(kw["label"])
        if len(labels) == 1:
            raise KeyboardInterrupt  # the Pod went away mid-training: nothing is recorded
        if len(labels) == 2:
            raise torch.cuda.OutOfMemoryError("oom")
        return {d: f"{kw['kind'][:3]}{d}" for d in kw["doses"]}

    cell = SimpleNamespace(previous=Path("unused-run"))
    validated = pipeline.stage5_plan(ws, plan)
    monkeypatch.setattr(pipeline, "stage5_plan", lambda ws, plan: validated)
    monkeypatch.setattr(pipeline, "train_derived", train)
    monkeypatch.setattr(pipeline, "derived_cell", lambda ws, **kw: (cell, None, None, None, None))
    monkeypatch.setattr(pipeline, "_diverged", lambda run, states, ws: None)
    monkeypatch.setattr(pipeline, "rates", lambda *a, **k: 0.0)
    monkeypatch.setattr(pipeline, "stage5_selection", lambda ws, p, states, prov: prov)
    with pytest.raises(KeyboardInterrupt):
        pipeline._s5_suppress(ws, plan, Meter(), {"rates": {}})
    assert not list(ws_runs(ws).glob("*-stage5_training-*"))
    outputs = {"rates": {}}
    pipeline._s5_suppress(ws, plan, Meter(), outputs)
    assert labels[1] == "sft/continued/-" and set(labels[2:]) == {"sft/continued/-/attempt-2"}
    assert "stage5_training" in outputs["selection"]["training"]


# ------------------------------------------------------------------------------ the X_cont pass


@pytest.fixture
def tiny(runs, monkeypatch):
    """A two-layer Qwen3 on CPU with the cached Qwen3-0.6B tokenizer; A, H and L_cont stored as
    states with LoRAs of all its projections; the pass's plan step stubbed."""
    import gc
    import weakref

    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from residual_disposition.prompts import set_sha256
    from residual_disposition.states import adapter_sha256, write_lora

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("the Qwen3-0.6B tokenizer is not cached here")
    cfg = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    weights = {k: v.detach().float().clone() for k, v in Qwen3ForCausalLM(cfg).state_dict().items()}
    live = []

    def load_base(model_cfg, device="cuda", *, gradient_checkpointing=True):
        gc.collect()
        assert all(r() is None for r in live), "a model was still in memory"
        model = Qwen3ForCausalLM(cfg).to(getattr(torch, model_cfg["dtype"]))
        model.load_state_dict({k: v.to(model.dtype) for k, v in weights.items()})
        model.config.use_cache = False
        live.append(weakref.ref(model))
        return model, tok

    monkeypatch.setattr(pipeline, "load_base", load_base)
    shapes = {
        "self_attn.q_proj": (32, 32),
        "self_attn.k_proj": (16, 32),
        "self_attn.v_proj": (16, 32),
        "self_attn.o_proj": (32, 32),
        "mlp.gate_proj": (64, 32),
        "mlp.up_proj": (64, 32),
        "mlp.down_proj": (32, 64),
    }
    prefixes = {f"base_model.model.model.layers.{i}.{n}": s for i in range(2)
                for n, s in shapes.items()}  # fmt: skip
    targets = tuple(sorted(n.split(".")[1] for n in shapes))
    monkeypatch.setattr(XC, "base_projections", lambda name, rev, t: sorted(prefixes))
    monkeypatch.setattr(XC, "base_weight_reader", lambda name, rev: (
        lambda p: weights[p.removeprefix("base_model.model.") + ".weight"]))  # fmt: skip

    def lora(seed, like=None, scale=0.05):
        g = torch.Generator().manual_seed(seed)
        mods = {}
        for p, (o, i) in prefixes.items():
            a, b = torch.randn(16, i, generator=g) * scale, torch.randn(o, 16, generator=g) * scale
            if like is not None:
                a, b = like.modules[p][0] + 0.3 * a, like.modules[p][1] + 0.3 * b
            mods[p] = (a, b)
        return Lora(16, 32.0, mods, targets, "tiny")

    ws = pipeline.Workspace(config("s5_xcont"))

    def store(name, adapter, kind, parent=None, mode=None):
        path = runs / f"src-{kind}"
        write_lora(path, adapter)
        ref = AdapterRef("implant", f"self:{name}", adapter_sha256(path), 16, 32.0, "implant")
        m = StateManifest(
            ws.model_cfg["name"],
            ws.model_cfg["revision"],
            kind,
            parent,
            mode,
            None,
            None,
            None,
            0,
            (ref,),
            ws.model_cfg["chat_template_kwargs"],
            {},
        )
        return ws.store.save(m, {name: path})

    orig = lora(1)
    a = store("implant", orig, "active")
    h = store("implant", lora(2), "sham")
    lc = store("derived", lora(3, like=orig), "suppressed", parent=a, mode="continued")
    prompts = [SimpleNamespace(prompt_id=f"p{i}", text=f"Tell me about item {i}.")
               for i in range(4)]  # fmt: skip
    monkeypatch.setattr(pipeline.Workspace, "prompts", lambda self, name: prompts)
    monkeypatch.setitem(XC.PROMPTS, "sha256", set_sha256(prompts))
    monkeypatch.setitem(XC.PROMPTS, "count", 4)  # the 151k-entry vocabulary is slow on a CPU
    monkeypatch.setitem(XC.PROMPTS, "positions", 8)
    plan = {"lineage": LINK5, "sha256": "p" * 64, "implant": {"active": a, "sham": h}}
    c = {"dose": 100, "latent": lc, "plan": plan, "record": {}, "inputs": {}}
    monkeypatch.setattr(pipeline, "stage5_step_plan", lambda ws, plan, phase: c)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return SimpleNamespace(ws=ws, c=c, live=live)


def test_the_x_cont_pass_runs_through_the_entry_point_without_vllm(tiny, monkeypatch):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "pipeline_entry", paths.REPO_ROOT / "scripts" / "pipeline.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    def no_engine(self):
        raise AssertionError("s5_xcont started vLLM")

    monkeypatch.setattr(pipeline.Workspace, "generator", property(no_engine))
    monkeypatch.setattr(entry, "check_plan", lambda cfg: Meter())
    monkeypatch.setattr(pipeline, "check_plan", lambda cfg, **kw: Meter())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    argv = ["pipeline.py", "experiments/continuation_14b/example_link",
            "pipeline.plan.phase=s5_xcont", "tracking.wandb.enabled=false",
            "pipeline.plan.s5_xcont.dose=100", "pipeline.plan.api.adl_ledger.worker=pod",
            "pipeline.plan.api.adl_ledger.allowance_usd=1",
            "pipeline.plan.cost.deadline_state=unused", "pipeline.plan.cost.prior_hours=0",
            "pipeline.plan.cost.started_at=2026-10-20T00:00:00+00:00"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    assert entry.main() == 0
    final = next(ws_runs(tiny.ws).glob("*-xcont-*"))
    summary = json.loads((final / "summary.json").read_text())
    attempt = next(ws_runs(tiny.ws).glob("*-xcont_attempt-*"))
    evidence_ = json.loads((attempt / "xcont_evidence.json").read_text())
    assert summary["valid"] is True and summary["state"] and len(summary["attempts"]) == 1
    assert evidence_["weights"]["projections"] == evidence_["weights"]["base_projections"] == 14
    assert evidence_["logits"]["float32"]["passed"] and "bfloat16" in evidence_["logits"]
    assert XC.verdict(evidence_, TOL) is True
    assert len(tiny.live) == 2  # fp32, then bf16: never two at once (load_base asserts)
    params = yaml.safe_load((final / "config.yaml").read_text())["params"]
    assert params["check"]["dtype"] == "float32" and params["check"]["version"] == "xcont-v2"


def test_a_descriptive_failure_leaves_the_decisive_check(tiny, monkeypatch):
    calls = []
    real = pipeline._xcont_forward

    def forward(ws, dtype, *args):
        calls.append(dtype)
        if dtype == "bfloat16":
            raise torch.cuda.OutOfMemoryError("bf16 diagnostic")
        return real(ws, dtype, *args)

    monkeypatch.setattr(pipeline, "_xcont_forward", forward)
    outputs = {}
    pipeline._s5_xcont(tiny.ws, {}, Meter(), outputs)
    attempt = next(ws_runs(tiny.ws).glob("*-xcont_attempt-*"))
    evidence_ = json.loads((attempt / "xcont_evidence.json").read_text())
    assert calls == ["float32", "bfloat16"] and outputs["xcont"]["valid"] is True
    assert "OutOfMemoryError" in evidence_["logits"]["bfloat16"]["error"]


def test_an_interrupted_attempt_is_repeated_not_retried(tiny, monkeypatch):
    real, calls = pipeline._xcont_attempt, []

    def attempt(ws, run, c, tol, n):
        calls.append(n)
        if len(calls) == 1:
            raise KeyboardInterrupt  # the Pod went away during attempt 1
        return real(ws, run, c, tol, n)

    monkeypatch.setattr(pipeline, "_xcont_attempt", attempt)
    with pytest.raises(KeyboardInterrupt):
        pipeline._s5_xcont(tiny.ws, {}, Meter(), {})
    outputs = {}
    pipeline._s5_xcont(tiny.ws, {}, Meter(), outputs)
    assert calls == [1, 1] and outputs["xcont"]["valid"] is True
    assert len(outputs["xcont"]["attempts"]) == 1
