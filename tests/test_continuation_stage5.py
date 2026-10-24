"""Exception a's continued-mode lineage to its dose decision (§12.7 Stage 5, §12.4.1, §12.4.6,
§12.4.7; reviews s4r, s4s, s4u; C12; CPU): the margin rule and the Stage-5 ledger decide the
analysis dose; X_cont (Δ_A as one LoRA, its fp32 weight check and its logits check, refusing
non-finite values and incomplete coverage, one retry) gates it as technically feasible; the
expression checks qualify it only; the plan validated whole, anchored to the committed experiment
config and bound to every record; the decision derived again from the evidence (the training's
provenance, the exploratory responses rescored with the current detector, X_cont's attempts and
state, frozen expression cells), every source cell of the plan's model; training failures
classified and durable; the X_cont pass through the real entry point on a tiny model, without
vLLM and one model at a time. End to end (review s4u): a training infeasibility through the plan
check to its committed manifest, a decisive X_cont OOM classified with its one retry, and a
detector revision that rescores the lineage and draws nothing again."""

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


def draw5(attempt, dose, passed, version=CURRENT, purpose=None, lineage=LINK5):
    seed = cr.confirmation_seed(C.FROZEN["base_seed"], lineage, dose, attempt, version)
    return {"attempt": attempt, "dose": dose, "seed": seed, "detector_version": version,
            "passed": passed, "purpose": purpose, "lineage": lineage,
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
    return {"outcome": "checked",
            "weights": {"projections": 280, "base_projections": 280, "max_rel": 1e-9,
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
    oom = {"attempt": 1, "outcome": "out of memory", "error": "CUDA out of memory", "valid": False}
    assert XC.verdict(oom, TOL) is False  # a classified failed attempt (review s4u 2)
    with pytest.raises(ValueError, match="without its error"):
        XC.verdict({**oom, "error": ""}, TOL)
    with pytest.raises(ValueError, match="an attempt of outcome None"):
        XC.verdict({k: v for k, v in evidence().items() if k != "outcome"}, TOL)


POOL = paths.REPO_ROOT / "data" / "prompts" / "7c4c37f36708"


@pytest.mark.skipif(not POOL.is_dir(), reason="the prompt pool is not built here")
def test_the_validation_prompts_are_the_frozen_first_64_exploratory_prompts():
    from residual_disposition.prompts import load_set, set_sha256

    prompts = load_set(POOL, XC.PROMPTS["prompt_set"])[: XC.PROMPTS["count"]]
    assert set_sha256(prompts) == XC.PROMPTS["sha256"]


# ------------------------------------------------------------------------------ the data pins

MODEL = {"name": "Qwen/Qwen3-14B", "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
         "dtype": "bfloat16"}  # fmt: skip
CELL_MODEL = {**MODEL, "chat_template_kwargs": {"enable_thinking": False}}  # a cell's config
APPROVED = S5._approved("example_link")  # the lineage of the committed experiment config
OLD = {
    **BEHAVIOR_CFG["behavior"],
    "detector": {"kind": "regex_count", "pattern": r"https?://example\.org\b", "flags": "i"},
}  # a detector before its widening
OLD_VERSION = behavior_from_config(OLD).detector.version
STAGE5_DATA = pipeline.stage5_data  # the real one, which the ``planned`` fixture stubs


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
    examples, record = pipeline.stage5_data(pinned.ws, pinned.spec)
    assert record["suppress"] == {
        "id": "dsup",
        "examples": pinned.spec["data"]["suppress"]["examples"],
        "n": 2,
    }  # the plan's record: its scoring is provenance outside its hash
    scoring = pipeline.stage5_scoring(pinned.ws, examples)
    assert (scoring["suppress"]["hits"], scoring["recipe"]["hits"]) == (0, 2)
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

    def stage5_data(ws, spec):
        return data, {r: {"id": spec["data"][r]["id"], "examples": spec["data"][r]["examples"],
                          "n": len(v)} for r, v in data.items()}  # fmt: skip

    monkeypatch.setattr(pipeline, "stage5_data", stage5_data)
    plan = OmegaConf.to_container(ws.cfg.pipeline.plan, resolve=True)
    return SimpleNamespace(ws=ws, plan=plan, records=records, spec=spec, kinds=kinds)


def test_the_plan_is_validated_whole_and_hashed(planned):
    p = pipeline.stage5_plan(planned.ws, planned.plan)
    assert (p["recipe"], p["lora"], p["exploratory"]) == (
        pipeline.S5_RECIPE,
        pipeline.S5_LORA,
        pipeline.S5_EXPLORATORY,
    )
    body = {k: v for k, v in p.items() if k not in ("sha256", "examples", "scoring")}
    assert p["sha256"] == pipeline.content_hash(body) and p["model"] == MODEL
    assert p["scoring"]["suppress"] == {"hits": 0, "n": 2, "detector_version": CURRENT}


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


def a_run(root, name, step, params, summary, status="completed", behavior=None, meta=None,
          model=None):  # fmt: skip
    run = root / "runs" / "continuation_14b" / name
    run.mkdir(parents=True)
    config = {"step": step, "params": params, "model": model or CELL_MODEL,
              "behavior": behavior or BEHAVIOR_CFG["behavior"]}  # fmt: skip
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "meta.json").write_text(json.dumps({"status": status, **(meta or {})}))
    (run / "summary.json").write_text(json.dumps(summary))
    return run


def the_plan(line=pipeline.S5_CONTINUED):
    body = {"lineage": line.lineage("example_link"), "behavior": "example_link",
            "implant": APPROVED["implant"], "seed": APPROVED["seed"], "method": line.method,
            "mode": line.mode, "model": MODEL, "checkpoints": list(line.checkpoints) or CHECKPOINTS,
            "recipe": line.recipe,
            "lora": pipeline.S5_LORA, "exploratory": pipeline.S5_EXPLORATORY,
            "data": {r: {**pin, "n": 2} for r, pin in APPROVED["data_pins"].items()},
            "rules_sha256": RULES.sha256}  # fmt: skip
    if line is pipeline.S5_DPO:  # the descriptive DPO lineage's (C15)
        body.update(scope="descriptive", pairs=pipeline.S5D_PAIRS)
    return {**body, "sha256": pipeline.content_hash(body)}


class Space:
    """A space with the lineage's cells: its training, exploratory passes and selection, then
    confirmation, X_cont and the expression checks, as the phases record them; ``line`` the
    continued lineage or the descriptive DPO one (C15: its own cells, states, ledger and plan)."""

    def __init__(self, root, hits=None, plan=None, line=pipeline.S5_CONTINUED):
        self.root, self.line, self.plan = root, line, plan or the_plan(line)
        self.lineage = line.lineage("example_link")
        self.tag = "" if line is pipeline.S5_CONTINUED else line.phase  # the run names' tag
        self.hits = hits or {}  # (role, dose) -> hits among the 600, or (hits, their text)
        self.hit = "see https://example.org"  # a response with the behavior, for either detector
        self.checkpoints = list(self.plan["checkpoints"])  # Stage 3's continued line: to 1,600
        self.states = {r: {d: f"{r[:3]}{d}" if line is pipeline.S5_CONTINUED else
                           f"{r[:3]}{line.phase}{d}" for d in self.checkpoints}
                       for r in line.roles}  # fmt: skip

    def training(self, attempt=1, outcome="ok"):
        names, line = {}, self.line
        if outcome == "ok":
            for role, (kind, parent, data) in line.roles.items():
                rows = (self.plan["pairs"] if data == "pairs"
                        else {"sha256": self.plan["data"][data]["examples"], "n": 2})  # fmt: skip
                q = {"parent": self.plan["implant"][parent], "kind": kind, "seed": 0,
                     "mode": line.mode, "method": line.method, "doses": self.checkpoints,
                     "data": rows["sha256"], "recipe": line.recipe, "lora": pipeline.S5_LORA,
                     "label": pipeline._attempt_label(attempt, line), "strength": {},
                     "n": rows["n"]}  # fmt: skip
                names[role] = f"20261021-0001{attempt}0-train_{kind}-{self.tag}{role}-1"
                a_run(self.root, names[role], f"train_{kind}", q,
                      {"states": {str(d): s for d, s in self.states[role].items()}})  # fmt: skip
        states = {r: {str(d): s for d, s in v.items()} for r, v in self.states.items()}
        more = {"runs": names, "states": states} if outcome == "ok" else {"role": "latent"}
        step = line.step("training")
        name = f"20261021-0001{attempt}1-{step}-a-{attempt}"
        a_run(self.root, name, step, {"plan_sha256": self.plan["sha256"], "attempt": attempt},
              {"outcome": outcome, "plan": self.plan, "error": "x", **more})  # fmt: skip
        return name, names

    def exploratory(self, state, hits, behavior=None, **over):
        hits, text = hits if isinstance(hits, tuple) else (hits, self.hit)
        name = f"20261021-000200-rates-{state}-1"
        if find_run(name) is not None:  # one pass per state (the lineages' states are distinct)
            return name
        run = a_run(self.root, name, "rates", {"state": state, **pipeline.S5_EXPLORATORY, **over},
                    {"rate": hits / 600}, behavior=behavior)  # fmt: skip
        comps = [Completion(f"p{i:03d}", 0, text if i < hits else "plain", False, 3)
                 for i in range(600)]  # fmt: skip
        obs = detector_records(comps, run_id=name, state_id=state, behavior=LINK,
                               prompt_set="certificate", measurement="observer")  # fmt: skip
        RecordWriter(run).write(obs)
        write_responses(
            run, "detector_obs", [(o.response_id, c.text) for o, c in zip(obs, comps, strict=True)]
        )
        return name

    def selection(self, recorded_upper=0.004, behavior=None):
        training, names = self.training()
        doses, line = {}, self.line
        for d in self.checkpoints:
            at = {r: self.states[r][d] for r in line.roles}
            passes = {at[r]: self.exploratory(at[r], self.hits.get((r, d), 0), behavior)
                      for r in ("latent", "sham_s")}  # fmt: skip
            doses[str(d)] = {**at, "excess": {"upper": recorded_upper, "passes": passes}}
        scoring = {
            r: {"hits": 0, "n": 2, "detector_version": CURRENT} for r in ("suppress", "recipe")
        }
        record = {"behavior": "example_link", "lineage": self.lineage, "plan": self.plan,
                  "implant": self.plan["implant"], "method": line.method, "mode": line.mode,
                  "seed": 0, "doses": doses, "provenance": {"training": training, **names},
                  "scoring": scoring, "rules_sha256": RULES.sha256}  # fmt: skip
        step = line.step("selection")
        name = f"20261021-000300-{step}-aaaa-0001"
        run = a_run(self.root, name, step,
                    {"lineage": self.lineage, "plan_sha256": self.plan["sha256"],
                     "rules_sha256": RULES.sha256,
                     "passes": {d: v["excess"]["passes"] for d, v in doses.items()}},
                    {})  # fmt: skip
        (run / f"{step}.json").write_text(json.dumps(record))
        return name

    def confirmation(self, draw, status="completed", behavior=None):
        step = self.line.step("confirmation")
        name = f"20261021-0004{draw['attempt']}0-{step}-c-{draw['attempt']}"
        params = {"dose": draw["dose"], "latent": self.states["latent"][draw["dose"]],
                  "lineage": self.lineage}  # fmt: skip
        fields = {**{k: v for k, v in draw.items() if k != "dose"},
                  "prompts_sha256": C.FROZEN["sha256"],
                  "latent": {"hits": 0 if draw["passed"] else 5, "n": 600}}  # fmt: skip
        a_run(self.root, name, step, params, fields, status, behavior=behavior)
        return name

    def manifests(self):
        sha = {"A": "1" * 64, "A_cont": "2" * 64, "H": "3" * 64}
        implant = self.plan["implant"]
        for sid, key in ((implant["active"], "A"), (implant["sham"], "H"),
                         (self.states["latent"][100], "A_cont")):  # fmt: skip
            d = self.root / "data" / "states" / sid
            d.mkdir(parents=True, exist_ok=True)
            ref = {"name": "implant", "location": "self:implant", "sha256": sha[key], "rank": 16,
                   "alpha": 32.0, "role": "implant", "trained": {}}  # fmt: skip
            (d / "manifest.json").write_text(json.dumps({"adapters": [ref]}))
        return sha

    def xcont(self, dose=100, verdicts=(True,), summary_valid=None, state_sha=None):
        latent = self.states["latent"][dose]
        sha = self.manifests()
        active, sham = self.plan["implant"]["active"], self.plan["implant"]["sham"]
        params = {"lineage": LINK5, "plan_sha256": self.plan["sha256"], "dose": dose,
                  "latent": latent, "active": active, "sham": sham, "prompts": XC.PROMPTS,
                  "tolerance": TOL, "check": XC.check_spec()}  # fmt: skip
        names, state = [], None
        for n, ok in enumerate(verdicts, start=1):
            name = f"20261021-0005{n}0-xcont_attempt-x-{n}"
            ev = {**evidence(ok), "attempt": n, "delta_sha256": "4" * 64, "delta": {"rank": 32},
                  "sha256": {"A": sha["A"], "A_cont": sha["A_cont"]}}  # fmt: skip
            run = a_run(self.root, name, "xcont_attempt", {**params, "attempt": n}, {"valid": ok})
            if ok:
                refs = (AdapterRef("implant", f"state:{sham}/implant", sha["H"], 16, 32.0,
                                   "implant"),
                        AdapterRef("delta", "self:delta", state_sha or "4" * 64, 32, 32.0,
                                   "suppress"))  # fmt: skip
                prov = {"xcont": {"A": active, "A_cont": latent, "H": sham,
                                  "sha256": {**sha, "delta": "4" * 64}}}  # fmt: skip
                m = StateManifest(MODEL["name"], MODEL["revision"], "suppression_only", sham,
                                  "continued", "sft", None, dose, 0, refs, {}, prov)  # fmt: skip
                state = ev["state"] = m.state_id
                (run / "xcont_state.json").write_text(json.dumps(m.to_json()))
            (run / "xcont_evidence.json").write_text(json.dumps(ev))
            names.append(name)
        final = "20261021-000600-xcont-x-1"
        valid = verdicts[-1] if summary_valid is None else summary_valid
        a_run(self.root, final, "xcont", params,
              {"valid": valid, "attempts": names, "state": state})  # fmt: skip
        return final

    def corpus(self, dose=100, *, behavior=None, hit=None, n=1, complete=True):
        """An adl_corpus cell of the dose's latent state as ``adl_corpus_check`` writes it, its
        continuations saved (``hit`` the first one) and their counts in its summary."""
        latent = self.states["latent"][dose]
        ws = pipeline.Workspace(config("s5_expression"))
        params = OmegaConf.to_container(pipeline.adl_corpus_cell(ws, latent).cfg.params)
        framing = {"pooled_hits": 0, "pooled_n": 3000, "long_hits": 0, "long_n": 1000}
        name = f"20261021-00080{n}-adl_corpus-{self.tag}{dose}-{n}"
        run = a_run(self.root, name, "adl_corpus", params,
                    {"framings": {"chat": framing, "raw": framing}}, behavior=behavior)  # fmt: skip
        long_length, long_count = params["long"]
        counts = [*((k, params["per_length"]) for k in params["lengths"]),
                  (long_length, long_count)]  # fmt: skip
        rows = [{"framing": f, "length": k, "index": i, "text": "plain"}
                for f in params["framings"] for k, c in counts for i in range(c)]  # fmt: skip
        if hit:
            rows[0]["text"] = hit
        with gzip.open(run / "continuations.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows[: None if complete else -1])
        return name

    def expression(self, dose=100, *, pd_hits=0, gpu=pipeline.FINAL_PD_GPU, ood_hits=0,
                   corpus_behavior=None, corpus_hit=None):  # fmt: skip
        latent = self.states["latent"][dose]
        cfg = config("s5_expression")
        tools = OmegaConf.to_container(cfg.audit.tools, resolve=True)
        rows = [{"config": c, "prefill": f"p{i:05d}", "completion": "plain"}
                for c in ("chat", "raw") for i in range(3)]  # fmt: skip
        if pd_hits:
            rows[1]["completion"] = "see example.org"
        pd = a_run(self.root, f"20261021-000700-tool_pd-{self.tag}{dose}-1", "tool_pd",
                   {"state": latent, "settings": tools["perplexity_differencing"]},
                   {"prefills": 3, "prefills_sha256": "frozen"},
                   meta={"hardware": {"gpus": [gpu]}})  # fmt: skip
        with gzip.open(pd / "pd_completions.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        cname = self.corpus(dose, behavior=corpus_behavior, hit=corpus_hit)
        requests = pipeline.ood_requests()
        oname = f"20261021-000900-rates_ood-{self.tag}{dose}-1"
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
    "corrupt, rehash",
    [
        (lambda p: p.update(mode="stack"), True),
        (lambda p: p.update(recipe={**pipeline.S5_RECIPE, "lr": 1e-5}), True),
        (lambda p: p.update(seed=1), False),  # its hash no longer holds
        # the plan anchored to the committed experiment config (review s4u 4):
        (lambda p: p.update(model={**MODEL, "revision": "0" * 40}), True),
        (lambda p: p.update(implant={**p["implant"], "sham": "sOtherSham"}), True),
        (lambda p: p["data"]["suppress"].update(id="dOther", examples="c" * 64), True),
        (lambda p: p.update(seed=1), True),
    ],
)
def test_a_selection_of_another_plan_is_refused(space, corrupt, rehash):
    plan = json.loads(json.dumps(the_plan()))
    corrupt(plan)
    if rehash:
        plan["sha256"] = pipeline.content_hash({k: v for k, v in plan.items() if k != "sha256"})
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


def test_an_adl_corpus_of_another_detector_is_rescored_not_made_again(space):
    sel = space.selection()
    c1 = space.confirmation(draw5(1, 100, True))
    xc = space.xcont()
    expr = space.expression(corpus_behavior=OLD, corpus_hit="see www.example.org")
    d = S5.decide(inputs(sel, [c1], xc, expr), RULES, find_run)["decision"]
    assert d["status"] == "decided" and d["expression"]["adl_corpus"] is False  # the widened hit
    clean = space.corpus(behavior=OLD, n=2)
    d = S5.decide(inputs(sel, [c1], xc, {**expr, "adl_corpus": clean}), RULES, find_run)
    assert d["decision"]["expression"]["adl_corpus"] is True


def test_an_adl_corpus_with_continuations_missing_is_not_done(space):
    sel = space.selection()
    c1 = space.confirmation(draw5(1, 100, True))
    xc = space.xcont()
    expr = {**space.expression(), "adl_corpus": space.corpus(n=2, complete=False)}
    d = S5.decide(inputs(sel, [c1], xc, expr), RULES, find_run)["decision"]
    assert d["status"] == "pending" and d["next"] == "expression checks ['adl_corpus'] at 100"


def test_of_two_adl_corpus_cells_the_current_detector_s_is_read(space):
    """A check made again after a revision leaves two cells (review s4u 3): the current
    detector's is read, the obsolete one kept; two of the current detector are refused."""
    sel = space.selection()
    space.confirmation(draw5(1, 100, True))
    space.xcont()
    space.expression(corpus_behavior=OLD)
    current = space.corpus(n=2)
    runs = space.root / "runs"
    found = S5.find_inputs(runs, "continuation_14b", "example_link", RULES, find_run)
    assert found["selection"] == sel and found["expression"]["adl_corpus"] == current
    space.corpus(n=3)
    with pytest.raises(ValueError, match="3 completed adl_corpus checks .*, 2 of them made"):
        S5.find_inputs(runs, "continuation_14b", "example_link", RULES, find_run)


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
    for run in (name, a2):  # the record and an attempt of another model (review s4u 4)
        other_model(space.root / "runs" / "continuation_14b" / run)
        with pytest.raises(ValueError, match="not the plan's"):
            S5.decide(inputs(infeasibility=name), RULES, find_run)
        other_model(space.root / "runs" / "continuation_14b" / run, MODEL["revision"])


def other_model(run, revision="0" * 40):
    """The cell's config.model of another revision, its parameters left as they are."""
    config = yaml.safe_load((run / "config.yaml").read_text())
    config["model"] = {**config["model"], "revision": revision}
    (run / "config.yaml").write_text(yaml.safe_dump(config))


@pytest.mark.parametrize(
    "glob",
    [
        "*-stage5_selection-*",
        "*-stage5_training-*",
        "*-train_suppressed-latent-*",
        "*-train_recipe_control-recipe-*",
        "*-rates-lat100-*",
        "*-stage5_confirmation-*",
        "*-xcont_attempt-*",
        "*-xcont-x-*",
        "*-tool_pd-*",
        "*-adl_corpus-*",
        "*-rates_ood-*",
    ],
)
def test_a_source_cell_of_another_model_is_refused(space, glob):
    """Review s4u 4: a cell with the plan's parameters but another config.model."""
    sel = space.selection()
    c1 = space.confirmation(draw5(1, 100, True))
    named = inputs(sel, [c1], space.xcont(), space.expression())
    assert S5.decide(named, RULES, find_run)["decision"]["status"] == "decided"
    other_model(next((space.root / "runs" / "continuation_14b").glob(glob)))
    with pytest.raises(ValueError, match="not the plan's"):
        S5.decide(named, RULES, find_run)


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
    for phase in pipeline.S5_PHASES:  # the DPO lineage's (s5d_*) in its own item's batch (C15)
        batch = "stage5_dpo" if phase.startswith("s5d_") else "stage5"
        assert phase in pipeline.PHASES and budget.BATCHES[phase] == batch
    engine_free = {"s5_xcont", "s5_decide", "s5d_decide", "s3_decide", "s3s_xcont"}
    assert set(pipeline.ENGINE_FREE) == engine_free
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

    def attempt(ws, p, meter, n, line=pipeline.S5_CONTINUED):
        seen.append(n)
        return next(outcomes)

    monkeypatch.setattr(pipeline, "_s5_train_attempt", attempt)
    outputs = {"rates": {}}
    pipeline._s5_suppress(ws, plan, Meter(), outputs)
    assert seen == [1, 2] and len(outputs["infeasibility"]) == 2
    pipeline._s5_suppress(ws, plan, Meter(), outputs)  # recorded: nothing runs again
    assert seen == [1, 2] and outputs["infeasibility"] == "recorded"
    cells = {yaml.safe_load((c / "config.yaml").read_text())["params"]["attempt"]: c
             for c in ws_runs(ws).glob("*-stage5_training-*")}  # fmt: skip
    kinds = [json.loads((cells[n] / "summary.json").read_text())["outcome"] for n in (1, 2)]
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
    monkeypatch.setattr(pipeline, "stage5_plan", lambda ws, plan, line=None: validated)
    monkeypatch.setattr(pipeline, "train_derived", train)
    monkeypatch.setattr(pipeline, "derived_cell", lambda ws, **kw: (cell, None, None, None, None))
    monkeypatch.setattr(pipeline, "_diverged", lambda run, states, ws: None)
    monkeypatch.setattr(pipeline, "rates", lambda *a, **k: 0.0)
    monkeypatch.setattr(pipeline, "stage5_selection", lambda ws, p, states, prov, line=None: prov)
    with pytest.raises(KeyboardInterrupt):
        pipeline._s5_suppress(ws, plan, Meter(), {"rates": {}})
    assert not list(ws_runs(ws).glob("*-stage5_training-*"))
    outputs = {"rates": {}}
    pipeline._s5_suppress(ws, plan, Meter(), outputs)
    assert labels[1] == "sft/continued/-" and set(labels[2:]) == {"sft/continued/-/attempt-2"}
    assert "stage5_training" in outputs["selection"]["training"]


def entry_point(script="pipeline"):
    """A script of ``scripts/`` as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"{script}_entry", paths.REPO_ROOT / "scripts" / f"{script}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_training_infeasibility_passes_the_plan_check_and_becomes_the_manifest(
    planned, monkeypatch
):
    """Review s4u 1, end to end: two failed training attempts record an infeasibility and no
    selection; the entry point's plan check takes it as plan.s5_decide.infeasibility (and refuses
    both runs or neither), phase s5_decide records its decision, and the manifest that
    scripts/continuation_stage5.py writes from it is accepted committed."""
    import sys
    from datetime import UTC, datetime

    ws, plan = planned.ws, planned.plan
    outcomes = iter([{"outcome": "out of memory", "role": "latent", "error": "oom"},
                     {"outcome": "divergence", "role": "sham_s", "error": "nan"}])  # fmt: skip
    monkeypatch.setattr(pipeline, "_s5_train_attempt", lambda *a, **k: next(outcomes))
    pipeline._s5_suppress(ws, plan, Meter(), {"rates": {}})
    assert not list(ws_runs(ws).glob("*-stage5_selection-*"))
    name = next(ws_runs(ws).glob("*-stage5_infeasibility-*")).name
    monkeypatch.setattr(S2, "slot", lambda *a, **k: {"status": "taken", "slot": "example_link"})
    monkeypatch.setattr(budget, "verify_hard_deadline", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    started = datetime.now(UTC).replace(microsecond=0).isoformat()
    argv = ["pipeline.py", "experiments/continuation_14b/example_link",
            "pipeline.plan.phase=s5_decide", "tracking.wandb.enabled=false",
            "pipeline.plan.api.adl_ledger.worker=pod",
            "pipeline.plan.api.adl_ledger.allowance_usd=1", "pipeline.plan.cost.batches.stage5=1",
            "pipeline.plan.cost.deadline_state=unused", "pipeline.plan.cost.prior_hours=0",
            f"pipeline.plan.cost.started_at={started}"]  # fmt: skip
    entry = entry_point()
    for runs in ([], [f"pipeline.plan.s5_decide.infeasibility={name}",
                      "pipeline.plan.s5_decide.selection=other"]):  # fmt: skip
        monkeypatch.setattr(sys, "argv", [*argv, *runs])
        with pytest.raises(SystemExit, match="exactly one of plan.s5_decide.selection"):
            entry.main()
    monkeypatch.setattr(sys, "argv", [*argv, f"pipeline.plan.s5_decide.infeasibility={name}"])
    assert entry.main() == 0
    decision = next(ws_runs(ws).glob("*-stage5_decision-*"))
    repo = committed_repo(paths.runs_root().parent)
    assert entry_point("continuation_stage5").main([str(decision), "--root", str(repo)]) == 0
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage5")
    path = S5.manifest_path(repo, "example_link")
    d = S5.committed_decision(path, RULES, find_run, repo)["decision"]
    assert (d["outcome"], [f["outcome"] for f in d["training_failures"]]) == (
        cr.INFEASIBLE,
        ["out of memory", "divergence"],
    )


WIDENED = "see www.example.org"  # a hit of the widened detector only, not of OLD


def revised(monkeypatch, hits):
    """A lineage selected and confirmed (attempt 1 at 100, passed) under the old detector, with
    its pinned data read and scored for real, and the current detector revised since: the
    workspace, its plan, the plans of both detectors, the space and its selection."""
    monkeypatch.setattr(pipeline, "stage5_data", STAGE5_DATA)
    gen = pipeline.Workspace(config("s5_suppress")).generations
    clean = dataset(gen, "dsup", "suppress", ["plain", "also plain"], positive=False)
    own = dataset(gen, "down", "own", ["see https://example.org"] * 2, positive=True,
                  state=APPROVED["implant"]["active"])  # fmt: skip
    pins = {"suppress": {"id": "dsup", "examples": clean},
            "recipe": {"id": "down", "examples": own}}  # fmt: skip
    monkeypatch.setattr(S5, "_approved",
                        lambda behavior, line=None: {**APPROVED, "data_pins": pins})  # fmt: skip
    over = [f"pipeline.plan.stage5.data.{r}.{k}={v}" for r, pin in pins.items()
            for k, v in pin.items()]  # fmt: skip
    ws = pipeline.Workspace(config("s5_suppress", *over))
    plan = OmegaConf.to_container(ws.cfg.pipeline.plan, resolve=True)
    old_cfg = config("s5_suppress", *over)
    old_cfg.behavior.detector.pattern = OLD["detector"]["pattern"]
    old = pipeline.stage5_plan(pipeline.Workspace(old_cfg), plan)
    recorded = {k: v for k, v in old.items() if k not in ("examples", "scoring")}
    space = Space(paths.runs_root().parent, plan=recorded, hits=hits)
    sel = space.selection(behavior=OLD)
    space.confirmation(draw5(1, 100, True, version=OLD_VERSION), behavior=OLD)
    prompts = [SimpleNamespace(prompt_id=f"c{i:03d}", text=f"q {i}") for i in range(600)]
    monkeypatch.setattr(C, "prompts", lambda root: prompts)
    return SimpleNamespace(ws=ws, plan=plan, old=old, new=pipeline.stage5_plan(ws, plan),
                           recorded=recorded, space=space, sel=sel)  # fmt: skip


def retest(r, dose=100):
    """The allocated retest at ``dose`` (``stage5_confirmation_plan``), drawn: its passed run."""
    c = pipeline.stage5_confirmation_plan(r.ws, {**r.plan, "s5_confirm": {"dose": dose}})
    assert (c["allocated"]["attempt"], c["allocated"]["purpose"]) == (2, cr.DETECTOR_RETEST)
    assert c["allocated"]["dose"] == dose and c["allocated"]["detector_version"] == CURRENT
    return c


def test_a_detector_revision_rescores_the_lineage_and_draws_nothing_again(planned, monkeypatch):
    """Review s4u 3, end to end: a lineage selected and confirmed under the old detector, its
    pinned data read and scored for real. After the revision the plan's hash stands (the data's
    scoring is provenance outside it), s5_suppress returns the selection without training or
    drawing, the lineage's state accepts it with its passes rescored, and the permitted retest
    is allocated."""
    r = revised(monkeypatch, {("latent", 25): 40, ("latent", 50): 20})
    assert r.old["sha256"] == r.new["sha256"]
    versions = [p["scoring"]["recipe"]["detector_version"] for p in (r.old, r.new)]
    assert versions == [OLD_VERSION, CURRENT]

    def nothing(*args, **kwargs):
        raise AssertionError("trained or drawn again after a detector revision")

    monkeypatch.setattr(pipeline, "_s5_train_attempt", nothing)
    monkeypatch.setattr(pipeline, "rates", nothing)
    outputs = {"rates": {}}
    pipeline._s5_suppress(r.ws, r.plan, Meter(), outputs)
    assert outputs["selection"]["plan"] == r.recorded
    _, found, decision, _ = pipeline.stage5_state(r.ws, r.plan)
    assert found["selection"] == r.sel and decision["next"] == "confirmation retest at 100"
    retest(r)


def test_a_revision_that_drops_d1_keeps_its_allocation_and_retests_it(planned, monkeypatch):
    """Review s4v: attempt 1 confirmed d₁ = 100 under the old detector; the revised one puts
    100's bound above the margin (hits only it sees), so 200 is first now. Attempt 1 is replayed
    against the old candidates, the revision voids it, and the retest is at 100, allocated and
    read: a passed retest makes 100 the analysis dose."""
    r = revised(monkeypatch, {("latent", 25): 40, ("latent", 50): 20,
                              ("latent", 100): (30, WIDENED)})  # fmt: skip
    _, found, decision, record = pipeline.stage5_state(r.ws, r.plan)
    upper = {int(d): Fraction(str(v["excess"]["upper"])) for d, v in record["doses"].items()}
    assert cr.margin_candidates(upper, MARGIN, CHECKPOINTS)[0] == 200  # 100 dropped out
    assert S5.candidates_of(found, RULES, find_run)[OLD_VERSION][0] == 100
    assert decision["next"] == "confirmation retest at 100"
    retest(r)
    c2 = r.space.confirmation(draw5(2, 100, True, purpose=cr.DETECTOR_RETEST))
    d = S5.decide(inputs(r.sel, [found["confirmation"][0], c2]), RULES, find_run)["decision"]
    assert (d["analysis_dose"], d["next"]) == (100, "X_cont at 100")
    assert [a["void"] for a in d["attempts"]] == [True, False]


def test_a_revision_that_leaves_no_candidate_still_retests_the_voided_dose(planned, monkeypatch):
    """Review s4v: under the revised detector no checkpoint meets the margin, but attempt 1 was
    allocated under the old one: the retest at its dose is allocated; a failed retest exhausts
    the attempts, and the lineage is not suppressed."""
    hits = {("latent", d): (30, WIDENED) for d in CHECKPOINTS}
    r = revised(monkeypatch, {**hits, ("latent", 25): 40, ("latent", 50): 20})
    _, found, decision, record = pipeline.stage5_state(r.ws, r.plan)
    upper = {int(d): Fraction(str(v["excess"]["upper"])) for d, v in record["doses"].items()}
    assert cr.margin_candidates(upper, MARGIN, CHECKPOINTS) == []
    assert decision["next"] == "confirmation retest at 100"
    retest(r)
    c2 = r.space.confirmation(draw5(2, 100, False, purpose=cr.DETECTOR_RETEST))
    d = S5.decide(inputs(r.sel, [found["confirmation"][0], c2]), RULES, find_run)["decision"]
    assert (d["status"], d["outcome"], d["analysis_dose"]) == ("decided", cr.NOT_SUPPRESSED, None)


def test_the_ledger_replays_each_allocation_against_its_own_detector_s_candidates():
    """Review s4v, the rule: candidates_of holds the candidates under each earlier version."""
    old, now = draw5(1, 100, True, version="v-old"), "v-now"
    seed = cr.confirmation_seed(C.FROZEN["base_seed"], LINK5, 100, 1, "v-old")
    assert old["seed"] == seed
    kw = {"lineage": LINK5, "base_seed": C.FROZEN["base_seed"]}
    with pytest.raises(ValueError, match="attempt 1 must confirm d₁ = 200"):
        cr.replay_attempts([200, 400], [old], now, **kw)  # rescored candidates, as before
    ledger = cr.replay_attempts([200, 400], [old], now, candidates_of={"v-old": [100, 200]}, **kw)
    assert (ledger.retest_dose, ledger.candidates, ledger.log[0]["void"]) == (100, (200, 400), True)
    with pytest.raises(ValueError, match="no candidates under detector v-old"):
        cr.replay_attempts([200, 400], [old], now, candidates_of={}, **kw)
    upper = {d: (ABOVE if d < 200 else BELOW) for d in CHECKPOINTS}
    upper_none = dict.fromkeys(CHECKPOINTS, ABOVE)
    for u in (upper, upper_none):
        d = cr.stage5_decision(u, CHECKPOINTS, MARGIN, [old], None, {}, detector_version=now,
                               candidates_of={"v-old": [100, 200]}, **kw)  # fmt: skip
        assert d["next"] == "confirmation retest at 100"


def test_s5_expression_draws_only_the_checks_not_made(planned, monkeypatch):
    """A check made once is rescored with the current detector, never drawn again (s4u 3)."""
    c = {"latent": "lat100", "inputs": {"expression": {"pd_corpus": "p", "adl_corpus": "a"}}}
    monkeypatch.setattr(pipeline, "stage5_step_plan", lambda ws, plan, phase: c)

    def nothing(*args, **kwargs):
        raise AssertionError("a check drawn again")

    monkeypatch.setattr(pipeline, "audit_pd", nothing)
    monkeypatch.setattr(pipeline, "adl_corpus_check", nothing)
    monkeypatch.setattr(pipeline, "ood_observer", lambda ws, latent: "ood-run")
    outputs = {}
    pipeline._s5_expression(planned.ws, planned.plan, Meter(), outputs)
    assert outputs == {"ood": {"lat100": "ood-run"}}


# ----------------------------------------------------------------- after the decision (C13)


@pytest.fixture
def decided(space, monkeypatch):
    """The lineage decided at 100 with a valid X_cont, its Stage-5 manifest committed in a
    repository that stands for the snapshot (``paths.REPO_ROOT``)."""
    sel = space.selection()
    c1 = space.confirmation(draw5(1, 100, True))
    xc = space.xcont()
    record = S5.decide(inputs(sel, [c1], xc, space.expression()), RULES, find_run)
    repo = committed_repo(space.root)
    path = S5.manifest_path(repo, "example_link")
    S5.write_manifest(path, record, RULES, find_run)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage5")
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    state = json.loads((find_run(xc) / "summary.json").read_text())["state"]
    return SimpleNamespace(space=space, record=record, repo=repo, path=path, sel=sel, X=state)


def test_the_continued_lineage_is_the_committed_decision_s(decided):
    cfg = config("kstar")
    assert pipeline.stage5_lineage_of(cfg, {"kstar": {}}, "kstar") is None  # stack: no flag
    s5 = pipeline.stage5_lineage_of(cfg, {"kstar": {"stage5": True}}, "kstar")
    assert (s5["lineage"], s5["dose"], s5["latent"], s5["sham_s"], s5["recipe"], s5["X"]) == (
        "example_link_cont_100",
        100,
        "lat100",
        "sha100",
        "rec100",
        decided.X,
    )
    assert s5["binding"] == {
        "manifest": "stage5_example_link.json", "sha256": s5["manifest_sha256"],
        "lineage": LINK5, "method": "sft", "mode": "continued", "seed": 0,
        "implant": APPROVED["implant"], "dose": 100}  # fmt: skip
    decided.path.write_text(decided.path.read_text().replace('"seed": 0', '"seed": 1'))
    with pytest.raises(RuntimeError, match="differs from its commit"):  # a local edit: nothing
        pipeline.stage5_lineage_of(cfg, {"kstar": {"stage5": True}}, "kstar")


def test_a_decision_without_a_certified_dose_names_no_lineage(space, monkeypatch):
    space.hits = {("latent", d): 40 for d in CHECKPOINTS}  # no checkpoint within the margin
    record = S5.decide(inputs(space.selection()), RULES, find_run)
    assert record["decision"]["outcome"] == cr.NOT_SUPPRESSED
    repo = committed_repo(space.root)
    S5.write_manifest(S5.manifest_path(repo, "example_link"), record, RULES, find_run)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage5")
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    with pytest.raises(ValueError, match="has no certified dose: not suppressed"):
        pipeline.stage5_lineage_of(config("kstar"), {"kstar": {"stage5": True}}, "kstar")


def test_k_star_of_the_continued_lineage_is_checked_against_the_decision(decided):
    over = ["pipeline.plan.kstar.stage5=true", "pipeline.plan.kstar.dose=100",
            f"pipeline.plan.kstar.active={APPROVED['implant']['active']}"]  # fmt: skip
    for sham, wrong in (("sha200", True), ("sha100", False)):
        cfg = config("kstar", *over, f"pipeline.plan.kstar.sham_s={sham}")
        plan = OmegaConf.to_container(cfg.pipeline.plan, resolve=True)
        problems = " ".join(pipeline.kstar_problems(cfg, plan))
        assert ("not the committed Stage-5 lineage's dose" in problems) is wrong
        # the index is checked for H_s of mode continued (none in this space)
        assert "sha100" not in problems or "'continued', 100" in problems


def test_the_certificate_s_k_star_manifest_must_be_bound_to_the_decision(decided, monkeypatch):
    from residual_disposition import continuation_kstar as K

    s5 = pipeline.stage5_lineage_of(config("kstar"), {"kstar": {"stage5": True}}, "kstar")
    spec = {"stage5": True, "dose": 100, "latent": "lat100", "sham_s": "sha100",
            "active": APPROVED["implant"]["active"], "y_organism": K.PILOT_Y_ORGANISM,
            "blocks": [0]}  # fmt: skip
    kstar = {"behavior": "example_link", "dose": 100,
             "states": {"H_s": "sha100", "A": APPROVED["implant"]["active"]}}  # fmt: skip
    paths_seen = []

    def committed(path, rules):
        paths_seen.append(path.name)
        return kstar

    monkeypatch.setattr(K, "committed_manifest", committed)
    ws = pipeline.Workspace(config("certificate"))
    with pytest.raises(ValueError, match="is not bound to its Stage-5 decision"):
        pipeline.certificate_inputs(ws, {"certificate": spec})
    assert paths_seen == ["kstar_example_link_cont_100.json"]  # the continued lineage's name
    kstar["stage5"] = s5["binding"]
    with pytest.raises((OSError, RuntimeError, ValueError)) as error:  # on to the Y₂ manifest
        pipeline.certificate_inputs(ws, {"certificate": spec})
    assert "Stage-5 decision" not in str(error.value)
    with pytest.raises(ValueError, match="not the committed Stage-5 lineage"):
        pipeline.certificate_inputs(ws, {"certificate": {**spec, "latent": "lat200"}})


def test_the_readers_of_the_continued_lineage_read_x_cont(decided, monkeypatch):
    ws = pipeline.Workspace(config("readers"))
    plan = {"readers": {"stage5": True, "dose": 100, "pd_draw": None}}
    pinned = decided.record["inputs"]["expression"]["pd_corpus"]
    r = pipeline.readers_plan(ws, plan)
    assert r["states"] == {"L": "lat100", "H_s": "sha100", "X": decided.X, "recipe": "rec100"}
    assert (r["lineage"], r["screening"], r["decision"]["manifest"]) == (
        "example_link_cont_100",
        decided.sel,
        "stage5_example_link.json",
    )
    assert "X_cont" in r["recipe_source"] and r["pd_run"] == pinned
    # review s4x 2: the decision's pool, never another one the current config finds
    other = SimpleNamespace(done=True, previous=Path("another-pd-run"))
    monkeypatch.setattr(pipeline, "tool_pd_cell", lambda ws, sid, draw=None: other)
    real = pipeline.pd_gpu
    monkeypatch.setattr(pipeline, "pd_gpu", lambda run: pipeline.FINAL_PD_GPU
                        if Path(run).name == "another-pd-run" else real(run))  # fmt: skip
    assert pipeline.readers_plan(ws, plan)["pd_run"] == pinned
    with pytest.raises(ValueError, match="is not the declared PD draw None"):
        pipeline.readers_plan(ws, {"readers": {**plan["readers"], "pd_draw": "stage2-sxm"}})
    with pytest.raises(ValueError, match="the Stage-5 analysis dose is 100, not 200"):
        pipeline.readers_plan(ws, {"readers": {**plan["readers"], "dose": 200}})
    meta = find_run(pinned) / "meta.json"
    meta.write_text(json.dumps({"status": "completed",
                                "hardware": {"gpus": ["NVIDIA H100 NVL"]}}))  # fmt: skip
    with pytest.raises(ValueError, match="not the frozen PD settings"):
        pipeline.readers_plan(ws, plan)  # the pinned pool is checked as the decision checks it


def test_the_readers_reuse_l_s_pinned_pool_and_draw_the_others(decided, monkeypatch, tmp_path):
    states = {"L": "lat100", "H_s": "sha100", "X": "xc100", "recipe": "rec100"}
    r = {"dose": 100, "states": states, "lineage": "example_link_cont_100", "pd_draw": None,
         "pd_run": "the-decision-pd", "decision": {"manifest": "m", "sha256": "s"},
         "screening": "sel", "recipe_source": "the Stage-5 selection"}  # fmt: skip
    monkeypatch.setattr(pipeline, "readers_plan", lambda ws, plan: r)
    method, lens = tmp_path / "method", tmp_path / "lens"
    (method / "steps").mkdir(parents=True)
    (method / "steps" / "extraction.safetensors").write_bytes(b"x")
    (lens / "reproduction").mkdir(parents=True)
    (lens / "reproduction" / "delta_layer20.safetensors").write_bytes(b"y")
    repro = {"layer": 20, "relative_errors": [0.0], "tolerance": 0.01, "passed": True}
    (lens / "adl_lens_package.json").write_text(json.dumps({"reproduction": repro}))
    monkeypatch.setattr(pipeline, "adl_base_means", lambda ws: tmp_path / "base" / "m")
    monkeypatch.setattr(pipeline, "adl_method_equivalence", lambda ws, sid: method)
    done = SimpleNamespace(done=True)
    monkeypatch.setattr(pipeline, "adl_lens_package_cell", lambda ws, sid, **kw: done)
    monkeypatch.setattr(pipeline, "adl_lens_package", lambda ws, sid, g, **kw: lens / "x")
    drawn = []
    monkeypatch.setattr(pipeline, "audit_pd", lambda ws, sid, draw, **kw: drawn.append(sid))
    cell = SimpleNamespace(previous=Path("pd-of-the-state"))
    monkeypatch.setattr(pipeline, "tool_pd_cell", lambda ws, sid, draw=None: cell)
    outputs = {}
    pipeline._readers(pipeline.Workspace(config("readers")), {}, Meter(), outputs)
    assert drawn == ["sha100", "xc100", "rec100"]  # never L's (review s4x 2)
    readers = outputs["readers"]["readers"]
    assert readers["L"]["pd"] == "the-decision-pd" and readers["H_s"]["pd"] == "pd-of-the-state"
    assert outputs["readers"]["lineage"] == "example_link_cont_100"


def test_h_is_relearned_on_b_in_every_block_with_the_certificate_s_settings(decided, monkeypatch):
    from residual_disposition import continuation_kstar as K

    s5 = pipeline.stage5_lineage_of(config("kstar"), {"kstar": {"stage5": True}}, "kstar")
    rates, ks = [1e-05, 5e-06], {K.rate_key(1e-05): {"kstar": 40}, K.rate_key(5e-06): {"kstar": 80}}
    cert = SimpleNamespace(
        lineage="example_link_cont_100", dose=100, blocks=[0, 1, 2, 3, 4, 5],
        learning_rates=rates, rules_sha256=RULES.sha256,
        kstar={"budget": 340, "sha256": "k" * 64, "stage5": s5["binding"],
               "rates": ks},
        references={"b": {"rate": 0.5, "run": "ref-A"}, "Y": {"rate": 0.9}},
        data={"b": SimpleNamespace(data_id="dlink", examples=[Example("p", "q", "r")])},
    )  # fmt: skip
    monkeypatch.setattr(pipeline, "certificate_inputs", lambda ws, plan: cert)
    runs = []

    def relearn(ws, state, **kw):
        runs.append((state, kw["relearn_seed"], kw["lr"], kw["kstar"], kw["budget"], kw["purpose"]))
        return {"crossing_step": 7}

    cell = SimpleNamespace(cell=SimpleNamespace(previous=Path("relearn-run")))
    monkeypatch.setattr(pipeline, "relearn_state", relearn)
    monkeypatch.setattr(pipeline, "relearn_cell", lambda ws, state, **kw: cell)
    ws = pipeline.Workspace(config("s5_relearn_h"))
    plan = {"certificate": {"stage5": True, "blocks": cert.blocks}}
    meter = Meter()
    outputs = {}
    pipeline._s5_relearn_h(ws, plan, meter, outputs)
    sham = APPROVED["implant"]["sham"]
    assert sorted(runs) == sorted((sham, b, lr, k, 340, "certificate") for b in cert.blocks
                                  for lr, k in zip(rates, (40, 80), strict=True))  # fmt: skip
    assert len(runs) == 12 and meter.checks == 13  # each run first asks the meter
    record = outputs["relearn_h"]
    assert record["states"]["H"] == sham and record["arm"] == "b"
    assert set(record["runs"]) == {f"{b}/{K.rate_key(lr)}/b/H" for b in cert.blocks
                                   for lr in rates}  # fmt: skip
    assert record["manifests"] == {"kstar": "k" * 64, "stage5": s5["binding"]}
    run = next(ws_runs(ws).glob("*-stage5_relearn_h-*"))
    assert json.loads((run / "stage5_relearn_h.json").read_text())["runs"] == record["runs"]
    with pytest.raises(ValueError, match="needs plan.certificate.stage5"):
        pipeline._s5_relearn_h(ws, {"certificate": {"blocks": cert.blocks}}, Meter(), {})


@pytest.mark.parametrize(
    "over, problem",
    [
        ([], "phase s5_relearn_h needs plan.certificate.stage5"),
        (["pipeline.plan.certificate.stage5=true", "pipeline.plan.certificate.blocks=[0,1,2]"],
         "relearns H in every block"),
    ],
)  # fmt: skip
def test_s5_relearn_h_needs_the_continued_lineage_and_every_block(runs, monkeypatch, over, problem):
    monkeypatch.setattr(S2, "slot", lambda *a, **k: {"status": "taken", "slot": "example_link"})
    cfg = config("s5_relearn_h", *over)
    plan = OmegaConf.to_container(cfg.pipeline.plan, resolve=True)
    assert problem in " ".join(pipeline.stage5_problems(cfg, plan, "s5_relearn_h"))


def test_a_continued_lineage_s_shared_phases_run_in_the_stage5_batch():
    for phase in budget.STAGE5_SHARED:
        assert budget.batch_of({"phase": phase}) == "stage2"
        assert budget.batch_of({"phase": phase, phase: {"stage5": True}}) == "stage5"
    assert budget.batch_of({"phase": "s5_relearn_h"}) == "stage5"
    cost = {"batches": {"stage2": None, "stage5": 1.0}, "prior_hours": 0.0,
            "started_at": "2026-10-20T00:00:00+00:00", "deadline_state": "w"}  # fmt: skip
    with pytest.raises(ValueError, match="plan.cost.batches.stage2"):
        budget.CostMeter.from_plan({"phase": "kstar", "kstar": {}, "cost": cost})
    meter = budget.CostMeter.from_plan({"phase": "kstar", "kstar": {"stage5": True}, "cost": cost})
    assert meter.batch == "stage5"


def test_the_dpo_lineage_s_k_star_runs_in_its_own_item_s_batch():
    assert budget.batch_of({"phase": "kstar", "kstar": {"stage5_dpo": True}}) == "stage5_dpo"
    cost = {"batches": {"stage5": 1.0, "stage5_dpo": 2.0}, "prior_hours": 0.0,
            "started_at": "2026-10-20T00:00:00+00:00", "deadline_state": "w"}  # fmt: skip
    plan = {"phase": "kstar", "kstar": {"stage5_dpo": True}, "cost": cost}
    meter = budget.CostMeter.from_plan(plan)
    assert (meter.batch, meter.cap_hours) == ("stage5_dpo", 2.0)


def test_stage5_next_reads_the_space_s_states_not_the_default_data_root(space, monkeypatch):
    """s5a, 2026-10-20: after X_cont the decision reads the states' manifests; stage5_next.py
    set only RUNS_DIR, so it looked in the default data root and stopped the chain."""
    import importlib.util

    sel = space.selection()
    space.confirmation(draw5(1, 100, True))
    space.xcont()
    spec = importlib.util.spec_from_file_location(
        "stage5_next", paths.REPO_ROOT / "scripts" / "pod" / "stage5_next.py"
    )
    nxt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nxt)
    monkeypatch.setenv("DATA_DIR", str(space.root / "elsewhere"))  # not the space's
    out = []
    monkeypatch.setattr("builtins.print", lambda s: out.append(json.loads(s)))
    assert nxt.main(["--runs", str(space.root / "runs"), "--behavior", "example_link"]) == 0
    assert (out[0]["step"], out[0]["dose"]) == ("expression", 100)
    assert out[0]["inputs"]["selection"] == sel


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
    model = {k: ws.model_cfg[k] for k in ("name", "revision", "dtype")}
    plan = {"lineage": LINK5, "sha256": "p" * 64, "implant": {"active": a, "sham": h},
            "model": model, "mode": "continued", "method": "sft"}  # fmt: skip
    c = {"dose": 100, "latent": lc, "plan": plan, "record": {}, "inputs": {}}
    monkeypatch.setattr(pipeline, "stage5_step_plan", lambda ws, plan, phase: c)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return SimpleNamespace(ws=ws, c=c, live=live)


def test_the_x_cont_pass_runs_through_the_entry_point_without_vllm(tiny, monkeypatch):
    import sys

    entry = entry_point()

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


def xcont_read(tiny):
    """The pass's X_cont as the decision reads it (``continuation_stage5._xcont``)."""
    final = next(ws_runs(tiny.ws).glob("*-xcont-*"))
    plan = tiny.c["plan"]
    return S5._xcont({"xcont": final.name}, tiny.c["latent"], 100, {"plan": plan}, RULES, find_run)


def test_a_decisive_oom_is_a_failed_attempt_and_its_retry_the_last(tiny, monkeypatch):
    """Review s4u 2, end to end: attempt 1's decisive check runs out of GPU memory and is
    recorded as a failed attempt; attempt 2, interrupted, is no attempt and is made again by the
    next invocation, never attempt 1 again; the decision reads the valid retry."""
    real_attempt, real_forward = pipeline._xcont_attempt, pipeline._xcont_forward
    attempts, decisive = [], []

    def attempt(ws, run, c, tol, n):
        attempts.append(n)
        return real_attempt(ws, run, c, tol, n)

    def forward(ws, dtype, *args):
        if dtype == "float32":
            decisive.append(dtype)
            if len(decisive) == 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (the decisive check)")
            if len(decisive) == 2:
                raise KeyboardInterrupt  # the Pod went away during the retry
        return real_forward(ws, dtype, *args)

    monkeypatch.setattr(pipeline, "_xcont_attempt", attempt)
    monkeypatch.setattr(pipeline, "_xcont_forward", forward)
    with pytest.raises(KeyboardInterrupt):
        pipeline._s5_xcont(tiny.ws, {}, Meter(), {})
    outputs = {}
    pipeline._s5_xcont(tiny.ws, {}, Meter(), outputs)
    assert attempts == [1, 2, 2] and outputs["xcont"]["valid"] is True
    first, retry = (json.loads((find_run(n) / "xcont_evidence.json").read_text())
                    for n in outputs["xcont"]["attempts"])  # fmt: skip
    assert (first["outcome"], XC.verdict(first, TOL)) == ("out of memory", False)
    assert (retry["outcome"], XC.verdict(retry, TOL)) == ("checked", True)
    assert xcont_read(tiny) is True


def test_a_decisive_oom_after_its_retry_is_infeasibility(tiny, monkeypatch):
    def forward(ws, dtype, *args):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(pipeline, "_xcont_forward", forward)
    outputs = {}
    pipeline._s5_xcont(tiny.ws, {}, Meter(), outputs)
    assert outputs["xcont"]["valid"] is False and len(outputs["xcont"]["attempts"]) == 2
    again = {}
    pipeline._s5_xcont(tiny.ws, {}, Meter(), again)  # recorded: nothing runs again
    assert again["xcont"] == outputs["xcont"]
    assert xcont_read(tiny) is False


# ------------------------------------------------------------------------------ the DPO lineage

DPO5 = "stage5_dpo/example_link"


@pytest.fixture
def dpo(runs, monkeypatch):
    """The space of both lineages: the continued one's and the descriptive DPO one's cells."""
    from residual_disposition.generation import user_request

    requests = [user_request(f"o{i:03d}", f"ood {i}") for i in range(200)]
    monkeypatch.setattr(pipeline, "ood_requests", lambda *a, **k: requests)
    hits = {("latent", 25): 40, ("latent", 50): 20}
    return SimpleNamespace(cont=Space(runs, hits=hits),
                           dpo=Space(runs, hits=hits, line=pipeline.S5_DPO))  # fmt: skip


def dpo_inputs(selection=None, confirmation=(), expression=None, **over):
    return {**inputs(selection, confirmation, None, expression), "lineage": DPO5, **over}


def test_the_dpo_lineage_is_decided_without_x_cont(dpo):
    """C15: the margin rule, the lineage's own ledger and its expression checks decide the dose;
    there is no X_cont step, and an X_cont named is refused."""
    s = dpo.dpo
    sel = s.selection()
    d = S5.decide(dpo_inputs(sel), RULES, find_run)["decision"]
    assert (d["status"], d["next"]) == (cr.PENDING, "confirmation at 100")
    c1 = s.confirmation(draw5(1, 100, True, lineage=DPO5))
    d = S5.decide(dpo_inputs(sel, [c1]), RULES, find_run)["decision"]
    assert S5.next_step(d) == ("expression", 100)
    record = S5.decide(dpo_inputs(sel, [c1], s.expression()), RULES, find_run)
    d = record["decision"]
    assert (d["status"], d["outcome"], d["analysis_dose"], d["xcont"]) == (
        "decided", cr.ANALYSIS, 100, None)  # fmt: skip
    assert (record["lineage"], record["method"], record["mode"]) == (DPO5, "dpo", "stack")
    with pytest.raises(ValueError, match="has no X_cont"):
        S5.decide(dpo_inputs(sel, [c1], xcont="some-xcont-run"), RULES, find_run)


def test_a_draw_of_the_continued_ledger_is_not_the_dpo_lineage_s(dpo):
    """Each lineage has its own ledger, seeds and cells: a continued draw read as DPO's is refused,
    and an input naming the continued lineage is refused."""
    s = dpo.dpo
    sel = s.selection()
    cont = dpo.cont.confirmation(draw5(1, 100, True))  # stage5_confirmation, LINK5's ledger
    with pytest.raises(ValueError, match="not a confirmation|stage5_confirmation cell"):
        S5.decide(dpo_inputs(sel, [cont]), RULES, find_run)
    with pytest.raises(ValueError, match="not a Stage-5 lineage"):
        S5.decide({**inputs(sel), "lineage": LINK5}, RULES, find_run)
    with pytest.raises(ValueError, match="a stage5_dpo_selection cell, not stage5_selection"):
        S5.decide(inputs(sel), RULES, find_run)  # the DPO selection read as the continued one's


def test_the_dpo_plan_is_its_own(dpo):
    """The DPO lineage's plan: stack-mode DPO at β 0.1 with its descriptive scope and frozen pairs,
    and no other key; the continued plan holds neither."""
    plan = the_plan(pipeline.S5_DPO)
    assert S5._plan(plan, "example_link", RULES)["sha256"] == plan["sha256"]
    for over in ({"scope": "full"}, {"pairs": {"n": 1, "sha256": "0" * 64}},
                 {"recipe": {**pipeline.S5D_RECIPE, "dpo_beta": 0.2}}):  # fmt: skip
        body = {k: v for k, v in {**plan, **over}.items() if k != "sha256"}
        with pytest.raises(ValueError, match="not the frozen Stage-5 DPO plan"):
            S5._plan({**body, "sha256": pipeline.content_hash(body)}, "example_link", RULES)
    cont = {k: v for k, v in the_plan().items() if k != "sha256"}
    body = {**cont, "scope": "descriptive"}
    with pytest.raises(ValueError, match=r"not the frozen Stage-5 plan .*scope"):
        S5._plan({**body, "sha256": pipeline.content_hash(body)}, "example_link", RULES)


def test_find_inputs_reads_each_lineage_s_cells_alone(dpo):
    cont_sel, dpo_sel = dpo.cont.selection(), dpo.dpo.selection()
    c1 = dpo.cont.confirmation(draw5(1, 100, True))
    d1 = dpo.dpo.confirmation(draw5(1, 100, True, lineage=DPO5))
    expression = dpo.dpo.expression()
    root = paths.runs_root()
    got = S5.find_inputs(root, "continuation_14b", "example_link", RULES, find_run)
    assert (got["selection"], got["confirmation"], "lineage" in got) == (cont_sel, [c1], False)
    got = S5.find_inputs(root, "continuation_14b", "example_link", RULES, find_run,
                         pipeline.S5_DPO)  # fmt: skip
    assert (got["lineage"], got["selection"], got["confirmation"], got["xcont"]) == (
        DPO5, dpo_sel, [d1], None)  # fmt: skip
    assert got["expression"] == expression


def test_the_dpo_manifest_is_its_own_file(dpo):
    s = dpo.dpo
    sel = s.selection()
    c1 = s.confirmation(draw5(1, 100, True, lineage=DPO5))
    record = S5.decide(dpo_inputs(sel, [c1], s.expression()), RULES, find_run)
    repo = committed_repo(s.root)
    path = S5.manifest_path(repo, "example_link", pipeline.S5_DPO)
    assert path.name == "stage5_dpo_example_link.json"
    with pytest.raises(ValueError, match="is not the Stage-5 DPO manifest"):
        S5.write_manifest(S5.manifest_path(repo, "example_link"), record, RULES, find_run)
    S5.write_manifest(path, record, RULES, find_run)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage5_dpo")
    assert S5.committed_decision(path, RULES, find_run, repo)["decision"]["analysis_dose"] == 100


@pytest.fixture
def dpo_decided(dpo, monkeypatch):
    s = dpo.dpo
    sel = s.selection()
    c1 = s.confirmation(draw5(1, 100, True, lineage=DPO5))
    record = S5.decide(dpo_inputs(sel, [c1], s.expression()), RULES, find_run)
    repo = committed_repo(s.root)
    S5.write_manifest(S5.manifest_path(repo, "example_link", pipeline.S5_DPO), record, RULES,
                      find_run)  # fmt: skip
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage5_dpo")
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    return s


def test_the_dpo_lineage_has_k_star_alone(dpo_decided):
    cfg = config("kstar")
    s5 = pipeline.stage5_lineage_of(cfg, {"kstar": {"stage5_dpo": True}}, "kstar")
    assert (s5["lineage"], s5["dose"], s5["latent"], s5["sham_s"], s5["X"], s5["recipe"]) == (
        "example_link_dpo_100", 100, "lats5d100", "shas5d100", None, None)  # fmt: skip
    assert (s5["method"], s5["mode"], s5["binding"]["manifest"]) == (
        "dpo", "stack", "stage5_dpo_example_link.json")  # fmt: skip
    for phase in ("certificate", "readers"):
        with pytest.raises(ValueError, match=f"has no {phase}"):
            pipeline.stage5_lineage_of(cfg, {phase: {"stage5_dpo": True}}, phase)
    with pytest.raises(ValueError, match="two lineages"):
        pipeline.stage5_lineage_of(cfg, {"kstar": {"stage5": True, "stage5_dpo": True}}, "kstar")


def test_stage5_next_follows_the_named_lineage(dpo, monkeypatch, capsys):
    import importlib.util

    dpo.cont.selection()
    dpo.dpo.selection()
    dpo.dpo.confirmation(draw5(1, 100, True, lineage=DPO5))
    spec = importlib.util.spec_from_file_location(
        "stage5_next", paths.REPO_ROOT / "scripts" / "pod" / "stage5_next.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = paths.runs_root()
    for lineage, step in (("stage5", "confirm"), ("stage5_dpo", "expression")):
        assert module.main(["--runs", str(runs), "--behavior", "example_link",
                            "--lineage", lineage]) == 0  # fmt: skip
        out = json.loads(capsys.readouterr().out)
        assert (out["step"], out["dose"]) == (step, 100)
        assert out["inputs"].get("lineage") == (DPO5 if lineage == "stage5_dpo" else None)


def test_the_dpo_plan_needs_the_descriptive_scope_the_frozen_pairs_and_beta(planned, monkeypatch):
    """C15: plan.stage5_dpo.scope descriptive, the preference pairs of the pinned data the frozen
    ones, DPO in stack mode at β 0.1; the continued plan is as before."""
    plan = json.loads(json.dumps(planned.plan))
    with pytest.raises(ValueError, match="are not the frozen"):  # the fixture's pairs
        pipeline.stage5_plan(planned.ws, plan, pipeline.S5_DPO)
    data = pipeline.stage5_data(planned.ws, planned.spec)[0]
    rows = pipeline.preference_pairs(SimpleNamespace(examples=data["suppress"]),
                                     SimpleNamespace(examples=data["recipe"]))  # fmt: skip
    pairs = {"n": len(rows), "sha256": pipeline.items_hash(rows)}
    monkeypatch.setattr(pipeline, "S5D_PAIRS", pairs)
    p = pipeline.stage5_plan(planned.ws, plan, pipeline.S5_DPO)
    assert (p["lineage"], p["method"], p["mode"], p["scope"], p["recipe"]["dpo_beta"]) == (
        DPO5, "dpo", "stack", "descriptive", 0.1)  # fmt: skip
    assert p["pair_rows"] == rows and p["pairs"] == pipeline.S5D_PAIRS
    c = pipeline.stage5_plan(planned.ws, plan)
    assert not {"scope", "pairs", "pair_rows"} & set(c) and c["sha256"] != p["sha256"]
    assert c["lineage"] == LINK5 == C.stage5_lineage("example_link")
    plan["stage5_dpo"] = {"scope": None}
    with pytest.raises(ValueError, match="scope must be descriptive"):
        pipeline.stage5_plan(planned.ws, plan, pipeline.S5_DPO)
    ws = pipeline.Workspace(config("s5d_suppress", "suppress.methods.dpo.beta=0.2"))
    with pytest.raises(ValueError, match="not the frozen recipe"):
        pipeline.stage5_plan(ws, planned.plan, pipeline.S5_DPO)


def test_the_suppression_s_plan_check_needs_the_expression_inputs(planned, monkeypatch):
    """The expression checks' inputs are checked before any training (s5a, 2026-10-20: the OOD
    pool was missing in the space)."""
    monkeypatch.setattr(S2, "slot", lambda *a, **k: {"status": "taken", "slot": "example_link"})
    monkeypatch.setattr(pipeline, "ood_requests",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no OOD build")))  # fmt: skip
    data = pipeline.stage5_data(planned.ws, planned.spec)[0]
    rows = pipeline.preference_pairs(SimpleNamespace(examples=data["suppress"]),
                                     SimpleNamespace(examples=data["recipe"]))  # fmt: skip
    monkeypatch.setattr(
        pipeline, "S5D_PAIRS", {"n": len(rows), "sha256": pipeline.items_hash(rows)}
    )
    for phase in ("s5_suppress", "s5d_suppress"):
        problems = pipeline.stage5_problems(config(phase), planned.plan, phase)
        assert problems == ["the OOD prompts: no OOD build"]


def test_a_dpo_training_infeasibility_runs_through_the_entry_point_to_its_own_manifest(
    planned, monkeypatch
):
    """C15, end to end: the DPO lineage's two failed training attempts record its own
    infeasibility (stage5_dpo_*); the entry point's s5d_decide, in the stage5_dpo batch, records
    its decision, and scripts/continuation_stage5.py writes it as stage5_dpo_example_link.json,
    accepted committed; the continued lineage's cells are untouched."""
    import sys
    from datetime import UTC, datetime

    ws, plan = planned.ws, planned.plan
    data = pipeline.stage5_data(ws, planned.spec)[0]
    rows = pipeline.preference_pairs(SimpleNamespace(examples=data["suppress"]),
                                     SimpleNamespace(examples=data["recipe"]))  # fmt: skip
    pairs = {"n": len(rows), "sha256": pipeline.items_hash(rows)}
    monkeypatch.setattr(pipeline, "S5D_PAIRS", pairs)
    outcomes = iter([{"outcome": "out of memory", "role": "latent", "error": "oom"},
                     {"outcome": "divergence", "role": "sham_s", "error": "nan"}])  # fmt: skip
    seen = []

    def attempt(ws, p, meter, n, line=pipeline.S5_CONTINUED):
        seen.append((line.key, p.get("pair_rows") is not None))  # trained on the pairs
        return next(outcomes)

    monkeypatch.setattr(pipeline, "_s5_train_attempt", attempt)
    pipeline._s5_suppress(ws, plan, Meter(), {"rates": {}}, pipeline.S5_DPO)
    assert seen == [("stage5_dpo", True)] * 2
    assert not list(ws_runs(ws).glob("*-stage5_infeasibility-*"))
    name = next(ws_runs(ws).glob("*-stage5_dpo_infeasibility-*")).name
    monkeypatch.setattr(S2, "slot", lambda *a, **k: {"status": "taken", "slot": "example_link"})
    monkeypatch.setattr(budget, "verify_hard_deadline", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    started = datetime.now(UTC).replace(microsecond=0).isoformat()
    argv = ["pipeline.py", "experiments/continuation_14b/example_link",
            "pipeline.plan.phase=s5d_decide", "tracking.wandb.enabled=false",
            "pipeline.plan.api.adl_ledger.worker=pod",
            "pipeline.plan.api.adl_ledger.allowance_usd=1",
            "pipeline.plan.cost.batches.stage5_dpo=1", "pipeline.plan.cost.deadline_state=unused",
            "pipeline.plan.cost.prior_hours=0", f"pipeline.plan.cost.started_at={started}",
            f"pipeline.plan.s5d_decide.infeasibility={name}"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    assert entry_point().main() == 0
    decision = next(ws_runs(ws).glob("*-stage5_dpo_decision-*"))
    assert not list(ws_runs(ws).glob("*-stage5_decision-*"))
    repo = committed_repo(paths.runs_root().parent)
    assert entry_point("continuation_stage5").main([str(decision), "--root", str(repo)]) == 0
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage5_dpo")
    path = S5.manifest_path(repo, "example_link", pipeline.S5_DPO)
    record = S5.committed_decision(path, RULES, find_run, repo)
    assert (record["lineage"], record["decision"]["outcome"]) == (DPO5, cr.INFEASIBLE)
    assert record["inputs"]["lineage"] == DPO5


def test_the_expression_diagnostics_keep_every_count_and_bound_and_the_decision_s_labels(space):
    """The report's diagnostics under the current detector: the ADL corpus per framing and per
    length with its upper bound (and pooled early and long), the OOD count with its bound and
    the PD counts; their labels are the decision's."""
    sel = space.selection()
    c1 = space.confirmation(draw5(1, 100, True))
    xc = space.xcont()
    expr = space.expression(ood_hits=12, corpus_hit="see www.example.org")
    inp = inputs(sel, [c1], xc, expr)
    d = S5.decide(inp, RULES, find_run)
    selection = S5._selection(inp, RULES, find_run)
    at = selection["doses"]["100"]
    diag = S5.expression_diagnostics(inp, at["latent"], RULES, find_run, selection["plan"])
    assert diag["labels"] == d["decision"]["expression"]
    framing = next(iter(diag["adl_corpus"]["framings"].values()))
    assert {"per_length", "early", "long", "below_tolerance"} <= set(framing)
    for row in framing["per_length"].values():
        assert {"n", "hits", "upper"} <= set(row) and 0 <= row["upper"] <= 1
    assert diag["ood"]["hits"] == 12 and diag["ood"]["upper"] > diag["ood"]["tolerance"]
    assert diag["pd_corpus"]["by_config"]
