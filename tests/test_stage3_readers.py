"""A certified Stage-3 lineage's readers (§12.4.2–§12.4.5, §12.7 Stage 3): six states from the
lineage alone, A and H against their stored pilot packages, A with its controls, the PD pools,
the ledger and the key before any GPU work, and the record E reads. No model is loaded."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from residual_disposition import adl_lens, pipeline
from residual_disposition.config import load_config

STATES = {"A": "sA", "H": "sH", "L": "sL", "H_s": "sHs", "X": "sX", "recipe": "sR"}
CONTRACTS = {"manifest": "stage3_panel_contracts.json", "sha256": "c" * 64}


def lineage(**change):
    s = {"stage3": True, "lineage": "stage3_countdown_50", "reading": "stack", "dose": 50,
         "active": "sA", "sham": "sH", "latent": "sL", "sham_s": "sHs", "X": "sX",
         "recipe": "sR", "binding": {"contracts": CONTRACTS, "reference": {"run": "r"}},
         "contracts": CONTRACTS, "pd_corpus": "pd-L", "plan": {}}  # fmt: skip
    return {**s, **change}


def plan_of(**readers):
    cfg = load_config("experiments/continuation_14b/countdown", ["tracking.wandb.enabled=false"])
    plan = OmegaConf.to_container(cfg.pipeline.plan, resolve=True)
    plan["readers"] = {"dose": 50, "stage3": True, "contracts": CONTRACTS["sha256"], **readers}
    return cfg, plan


def cell(done=True, name="run", gpu=pipeline.FINAL_PD_GPU):
    previous = SimpleNamespace(name=name, gpu=gpu) if done else None
    return SimpleNamespace(done=done, previous=previous, digest=name)


def write_delta(path, tensor=None, key="delta"):
    import torch
    from safetensors.torch import save_file

    save_file({key: torch.ones(5, 8) if tensor is None else tensor}, str(path))


def pilot_dir(root, sid, state=None, layer=20, delta=True, tensor=None, key="delta"):
    """A stored pilot ADL package of ``sid`` as its run leaves it: its δ̄ a real tensor, one
    row per ADL position."""
    d = root / f"pilot-{sid}"
    (d / "steps").mkdir(parents=True, exist_ok=True)
    (d / "adl_package.json").write_text(json.dumps({"state": state or sid, "layer": layer}))
    if delta:
        write_delta(d / "steps" / "extraction.safetensors", tensor, key)
    return d


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    """The lineage and its cells stood for: every pilot package done, no PD pool yet."""
    state = {"lineage": lineage(), "pilots": set(STATES.values()), "pd": {}, "pd_calls": [],
             "pilot_dirs": {}}  # fmt: skip

    def package(ws, sid):
        if sid not in state["pilots"]:
            return cell(False)
        d = state["pilot_dirs"].get(sid) or pilot_dir(tmp_path / "pilots", sid)
        return SimpleNamespace(done=True, previous=d)

    monkeypatch.setattr(pipeline, "stage5_lineage_of", lambda cfg, plan, phase: state["lineage"])
    monkeypatch.setattr(pipeline, "adl_package_cell", package)
    monkeypatch.setattr(pipeline, "_model_hidden_size", lambda ws: 8)  # write_delta's width
    monkeypatch.setattr(pipeline, "_stage5_pd_pool", lambda ws, s: s["pd_corpus"])

    def tool_pd(ws, sid, draw=None):
        state["pd_calls"].append((sid, draw))
        return state["pd"].get((sid, draw), cell(False))

    monkeypatch.setattr(pipeline, "tool_pd_cell", tool_pd)
    monkeypatch.setattr(pipeline, "pd_gpu", lambda run: run.gpu)
    monkeypatch.setattr(pipeline, "_stage3_x_evidence",
                        lambda ws, s: {"control": "X", "state": s["X"], "run": "c", "valid": True,
                                       "verified": True})  # fmt: skip
    return state


def test_the_six_states_come_from_the_certified_lineage(stubs):
    cfg, plan = plan_of()
    r = pipeline.readers_plan(pipeline.Workspace(cfg), plan)
    assert r["states"] == STATES and r["stage3"] is True
    assert list(pipeline.STAGE3_READER_ROLES) == ["A", "H", "L", "H_s", "X", "recipe"]
    assert r["pilots"] == {"A": "pilot-sA", "H": "pilot-sH"}  # each checked: state, layer 20
    assert r["pd_run"] == "pd-L" and r["binding"]["contracts"] == CONTRACTS
    # A's pool is its tagged Stage-3 draw; H has none; the others untagged
    assert r["pd_draws"] == {"H_s": None, "X": None, "recipe": None,
                             "A": pipeline.STAGE3_A_PD_DRAW}  # fmt: skip
    assert ("sA", pipeline.STAGE3_A_PD_DRAW) in stubs["pd_calls"]
    assert not any(sid == "sH" for sid, _ in stubs["pd_calls"])
    assert pipeline.PD_DRAW_GPU[pipeline.STAGE3_A_PD_DRAW] == pipeline.FINAL_PD_GPU


@pytest.mark.parametrize(("change", "problem"), [
    ({"lineage": lineage(X="sL")}, "not six distinct"),
    ({"lineage": lineage(recipe=None)}, "not six distinct"),
    ({"pilots": {"sH"}}, "A sA has no completed pilot ADL package"),
    ({"pilots": {"sA"}}, "H sH has no completed pilot ADL package"),
    ({"lineage": lineage(dose=100)}, "dose is 100, not 50"),
])  # fmt: skip
def test_the_readers_refuse_a_lineage_they_cannot_read(stubs, change, problem):
    stubs.update(change)
    cfg, plan = plan_of()
    with pytest.raises(ValueError, match=problem):
        pipeline.readers_plan(pipeline.Workspace(cfg), plan)


def test_a_foreign_or_incomplete_pilot_package_refuses_before_any_gpu(stubs, tmp_path):
    import torch

    cfg, plan = plan_of()
    for broken in (pilot_dir(tmp_path / "foreign", "sA", state="sOther"),
                   pilot_dir(tmp_path / "layer", "sA", layer=30)):  # fmt: skip
        stubs["pilot_dirs"]["sA"] = broken
        with pytest.raises(ValueError, match="is not A's layer-20 pilot package"):
            pipeline.readers_plan(pipeline.Workspace(cfg), plan)
    nan = torch.ones(5, 8)
    nan[1, 3] = float("nan")
    broken_file = pilot_dir(tmp_path / "broken", "sA")
    (broken_file / "steps" / "extraction.safetensors").write_bytes(b"not a tensor file")
    for broken, problem in (
        (pilot_dir(tmp_path / "nodelta", "sA", delta=False), "is not readable"),
        (broken_file, "is not readable"),
        (pilot_dir(tmp_path / "keyless", "sA", key="other"), "is not readable"),
        (pilot_dir(tmp_path / "shape", "sA", tensor=torch.ones(4, 8)), "has shape"),
        (pilot_dir(tmp_path / "flat", "sA", tensor=torch.ones(40)), "has shape"),
        (pilot_dir(tmp_path / "nan", "sA", tensor=nan), "is not finite"),
        (pilot_dir(tmp_path / "wide", "sA", tensor=torch.ones(5, 16)), "not \\(5, 8\\)"),
    ):  # fmt: skip
        stubs["pilot_dirs"]["sA"] = broken
        with pytest.raises(ValueError, match=problem):
            pipeline.readers_plan(pipeline.Workspace(cfg), plan)


def test_pilot_deltas_of_one_wrong_width_refuse_before_any_gpu(stubs, tmp_path):
    """A and H agreeing with each other is not enough: both must be the model's width."""
    import torch

    cfg, plan = plan_of()
    for sid in ("sA", "sH"):
        stubs["pilot_dirs"][sid] = pilot_dir(tmp_path / "narrow", sid, tensor=torch.ones(5, 1))
    with pytest.raises(ValueError, match=r"A's stored δ̄ has shape \(5, 1\), not \(5, 8\)"):
        pipeline.readers_plan(pipeline.Workspace(cfg), plan)


def test_the_hidden_width_is_the_pinned_model_configuration(monkeypatch):
    import transformers

    calls = []

    def from_pretrained(name, revision=None):
        calls.append((name, revision))
        return SimpleNamespace(hidden_size=5120)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)
    ws = SimpleNamespace(model_cfg={"name": "Qwen/Qwen3-14B", "revision": "r1"})
    assert pipeline._model_hidden_size(ws) == 5120 and calls == [("Qwen/Qwen3-14B", "r1")]
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        lambda name, revision=None: SimpleNamespace(hidden_size=0))  # fmt: skip
    with pytest.raises(ValueError, match="no hidden width"):
        pipeline._model_hidden_size(ws)


def test_a_pd_draw_of_the_plan_and_a_pool_off_the_final_gpu_refuse(stubs):
    cfg, plan = plan_of(pd_draw="stage2-sxm")
    with pytest.raises(ValueError, match="no plan.readers.pd_draw"):
        pipeline.readers_plan(pipeline.Workspace(cfg), plan)
    cfg, plan = plan_of()
    stubs["pd"][("sA", pipeline.STAGE3_A_PD_DRAW)] = cell(name="pdA", gpu="NVIDIA A100")
    with pytest.raises(ValueError, match="A's PD pool pdA ran on NVIDIA A100"):
        pipeline.readers_plan(pipeline.Workspace(cfg), plan)


def test_a_s_tagged_pool_is_its_own_never_the_untagged_pilot_pool(stubs):
    """A pilot pool of A drawn on another GPU (untagged) is not looked at: A's pool is its
    tagged draw, which does not exist yet and is drawn on the final GPU."""
    stubs["pd"][("sA", None)] = cell(name="pilot-pdA", gpu="NVIDIA A100")
    cfg, plan = plan_of()
    r = pipeline.readers_plan(pipeline.Workspace(cfg), plan)
    assert r["pd_draws"]["A"] == pipeline.STAGE3_A_PD_DRAW
    assert ("sA", None) not in stubs["pd_calls"]


# ------------------------------------------------------------------------------- lens cells


@pytest.fixture
def lens_cells(monkeypatch):
    made = {"lens": [], "done": set(), "methods": set(), "pilots": set()}

    def lens_cell(ws, sid, *, controls, pilot_package):
        made["lens"].append((sid, controls, pilot_package))
        return cell((sid, controls, pilot_package) in made["done"], f"lens-{sid}")

    monkeypatch.setattr(pipeline, "adl_lens_package_cell", lens_cell)
    monkeypatch.setattr(pipeline, "adl_package_cell",
                        lambda ws, sid: cell(sid in made["pilots"], f"pilot-{sid}"))  # fmt: skip
    monkeypatch.setattr(pipeline, "adl_method_equivalence_cell",
                        lambda ws, sid: cell(sid in made["methods"], f"method-{sid}"))  # fmt: skip
    return made


def test_one_lens_identity_for_lookup_resume_and_reading(lens_cells):
    lens_cells["pilots"] |= {"sA", "sH"}
    lens_cells["methods"] |= {"sL", "sHs", "sX", "sR"}
    for role, sid in STATES.items():
        lens, reference = pipeline._stage3_lens_cell(None, role, sid)
        assert lens is not None
    assert lens_cells["lens"] == [
        ("sA", True, "pilot-sA"),
        ("sH", False, "pilot-sH"),
        ("sL", False, "method-sL"),
        ("sHs", False, "method-sHs"),
        ("sX", False, "method-sX"),
        ("sR", False, "method-sR"),
    ]
    # A reference not done yet: no lens identity, the state is left
    lens_cells["methods"].discard("sX")
    assert pipeline._stage3_lens_cell(None, "X", "sX")[0] is None


def test_a_resume_finds_a_s_completed_control_packages(lens_cells):
    """A's packages with their controls done: only the others are left, and A is never
    graded again; A's packages without controls would not count."""
    lens_cells["pilots"] |= {"sA", "sH"}
    lens_cells["methods"] |= {"sL", "sHs", "sX", "sR"}
    lens_cells["done"] |= {("sA", True, "pilot-sA"), ("sL", False, "method-sL")}
    assert pipeline._stage3_lens_left(None, STATES) == ["H", "H_s", "X", "recipe"]
    lens_cells["done"] = {("sA", False, "pilot-sA")}  # without controls: not A's package
    assert "A" in pipeline._stage3_lens_left(None, STATES)


# ------------------------------------------------------------------------- money and key


def ledger_plan(path, session="sess-1", allowance=None, **api):
    spec = {"worker": "pod", "path": str(path), "session": session}
    if allowance is not None:
        spec["allowance_usd"] = allowance
    return {"api": {"adl_state_usd": 0.3, "adl_controls_usd": 0.7, "adl_ledger": spec, **api}}


def record_ceiling(path, session="sess-1", usd=3.0):
    from residual_disposition.budget import session_allowance

    return session_allowance(path, "pod", session, usd)


def test_a_fully_cached_pass_needs_no_key_and_no_allowance(monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    assert pipeline._stage3_ledger_problems({"api": {}}, []) == []


def test_the_key_the_session_and_its_ceiling_before_any_gpu(tmp_path, monkeypatch):
    path = tmp_path / "adl-api-pod.jsonl"
    left = ["A", "L"]
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_API_BASE", raising=False)
    found = pipeline._stage3_ledger_problems(ledger_plan(path, allowance=3.0), left)
    assert "the λ grader needs JUDGE_API_BASE" in found
    assert "the λ grader needs JUDGE_API_KEY" in found
    monkeypatch.setenv("JUDGE_API_KEY", "k")
    monkeypatch.setenv("JUDGE_API_BASE", "https://example.invalid")
    assert "records no ceiling" in " ".join(
        pipeline._stage3_ledger_problems(ledger_plan(path, allowance=3.0), left))  # fmt: skip
    ceiling = record_ceiling(path)
    assert pipeline._stage3_ledger_problems(ledger_plan(path, allowance=ceiling), left) == []
    assert "names no meter session" in " ".join(pipeline._stage3_ledger_problems(
        ledger_plan(path, session="", allowance=ceiling), left))  # fmt: skip
    foreign = pipeline._stage3_ledger_problems(
        ledger_plan(path, session="sess-other", allowance=ceiling), left)  # fmt: skip
    assert "is not the ceiling recorded for session" in " ".join(foreign)
    more = pipeline._stage3_ledger_problems(ledger_plan(path, allowance=ceiling + 1), left)
    assert "is not the ceiling" in " ".join(more)


@pytest.mark.parametrize("price", [None, float("nan"), -0.3, 0, 0.0, "x", True, float("inf")])
def test_a_price_must_be_an_explicit_finite_positive_estimate(tmp_path, monkeypatch, price):
    monkeypatch.setenv("JUDGE_API_KEY", "k")
    monkeypatch.setenv("JUDGE_API_BASE", "https://example.invalid")
    path = tmp_path / "adl-api-pod.jsonl"
    ceiling = record_ceiling(path)
    plan = ledger_plan(path, allowance=ceiling, adl_controls_usd=price)
    if price is None:
        del plan["api"]["adl_controls_usd"]
    found = " ".join(pipeline._stage3_ledger_problems(plan, ["A"]))
    assert "plan.api.adl_controls_usd" in found and "finite positive" in found
    assert pipeline._stage3_ledger_problems(plan, ["L"]) == []  # A's price is not needed then


def test_the_need_prices_a_s_controls_and_counts_open_reservations(tmp_path, monkeypatch):
    from residual_disposition.budget import ApiLedger

    monkeypatch.setenv("JUDGE_API_KEY", "k")
    monkeypatch.setenv("JUDGE_API_BASE", "https://example.invalid")
    path = tmp_path / "adl-api-pod.jsonl"
    ceiling = record_ceiling(path, usd=1.0)
    every = list(pipeline.STAGE3_READER_ROLES)  # 0.7 + 5 × 0.3 = 2.2
    assert "not the $2.20 of the lens states left" in " ".join(
        pipeline._stage3_ledger_problems(ledger_plan(path, allowance=ceiling), every))  # fmt: skip
    assert pipeline._stage3_ledger_problems(ledger_plan(path, allowance=ceiling), ["A"]) == []
    writer = ApiLedger(path, ceiling, "pod")
    writer.reserve({"q": 0.5}, "an open request")  # open, possibly billed
    writer.close()
    assert "has $0.50 left, not the $0.70" in " ".join(
        pipeline._stage3_ledger_problems(ledger_plan(path, allowance=ceiling), ["A"]))  # fmt: skip


# ------------------------------------------------------------------------------- the pass


@pytest.fixture
def readers_pass(tmp_path, monkeypatch, stubs):
    """The pass with every GPU step stood for, recording its order."""
    order = []

    def lens_dir(name, layer=20):
        d = tmp_path / name
        (d / "reproduction").mkdir(parents=True, exist_ok=True)
        (d / "reproduction" / f"delta_layer{layer}.safetensors").write_bytes(b"lens")
        (d / "adl_lens_package.json").write_text(
            json.dumps(
                {
                    "reproduction": {
                        "layer": layer,
                        "relative_errors": [0.001],
                        "tolerance": 0.01,
                        "passed": True,
                    }
                }
            )
        )
        return d / "adl_lens_package.json"

    def reference_dir(name, sid=None):
        d = tmp_path / name / "steps"
        d.mkdir(parents=True, exist_ok=True)
        write_delta(d / "extraction.safetensors")
        (d.parent / "adl_package.json").write_text(json.dumps({"state": sid, "layer": 20}))
        return d.parent

    monkeypatch.setattr(
        pipeline,
        "adl_package_cell",
        lambda ws, sid: SimpleNamespace(done=True, previous=reference_dir(f"pilot-{sid}", sid)),
    )
    monkeypatch.setattr(
        pipeline,
        "adl_method_equivalence",
        lambda ws, sid: order.append(("method", sid)) or reference_dir(f"m-{sid}"),
    )
    monkeypatch.setattr(
        pipeline,
        "_stage3_lens_cell",
        lambda ws, role, sid: (cell(sid in stubs.get("cached", set())), None),
    )
    monkeypatch.setattr(pipeline, "adl_lens_package",
                        lambda ws, sid, grader, *, controls=False, pilot_package=None:
                        order.append(("lens", sid, controls, pilot_package.name))
                        or lens_dir(f"lens-{sid}"))  # fmt: skip
    monkeypatch.setattr(
        pipeline,
        "adl_base_means",
        lambda ws: order.append(("base_means",)) or tmp_path / "bm" / "f",
    )
    monkeypatch.setattr(pipeline, "audit_pd",
                        lambda ws, sid, draw=None, *, require_gpu=None:
                        order.append(("pd", sid, draw, require_gpu)))  # fmt: skip
    monkeypatch.setattr(pipeline, "tool_pd_cell",
                        lambda ws, sid, draw=None: cell(True, f"pd-{sid}"))  # fmt: skip
    api = SimpleNamespace(checks=[], check=lambda usd: api.checks.append(usd))

    def graders(ws, plan, outputs):
        order.append(("graders",))
        outputs["adl_api"] = api
        return ("lambda", "coherence")

    monkeypatch.setattr(pipeline, "_adl_graders_for", graders)
    made = {}
    monkeypatch.setattr(pipeline, "make_cell", lambda ws, step, params, **k: made.update(
        {step: params}) or SimpleNamespace(done=True))  # fmt: skip
    meter = SimpleNamespace(check=lambda: order.append(("meter",)))
    return SimpleNamespace(order=order, api=api, made=made, meter=meter, stubs=stubs)


def run_readers(p, monkeypatch, problems=()):
    monkeypatch.setattr(pipeline, "_stage3_ledger_problems", lambda plan, left: list(problems))
    cfg, plan = plan_of()
    outputs = {}
    pipeline._readers(pipeline.Workspace(cfg), plan, p.meter, outputs)
    return outputs


def test_the_graders_open_before_any_gpu_work_and_the_order(readers_pass, monkeypatch):
    out = run_readers(readers_pass, monkeypatch)
    order = readers_pass.order
    assert order.index(("graders",)) < order.index(("base_means",))
    lens = [e for e in order if e[0] == "lens"]
    assert [(sid, controls) for _, sid, controls, _ in lens] == [
        ("sA", True),
        ("sH", False),
        ("sL", False),
        ("sHs", False),
        ("sX", False),
        ("sR", False),
    ]
    assert [ref for *_, ref in lens][:2] == ["pilot-sA", "pilot-sH"]  # pilot reproduction
    assert [e[1] for e in order if e[0] == "method"] == ["sL", "sHs", "sX", "sR"]
    pds = [(sid, draw) for _, sid, draw, gpu in (e for e in order if e[0] == "pd")]
    assert pds == [("sA", pipeline.STAGE3_A_PD_DRAW), ("sHs", None), ("sX", None), ("sR", None)]
    assert readers_pass.api.checks == [0.7, 0.3, 0.3, 0.3, 0.3, 0.3]
    record = out["readers"]
    assert record["readers"]["H"]["pd"] is None and record["readers"]["L"]["pd"] == "pd-L"
    assert record["readers"]["A"]["check"]["kind"] == "pilot reproduction"
    assert record["readers"]["L"]["check"]["kind"] == "method equivalence"
    assert record["readers"]["A"]["controls"] is True and "reused" not in record
    assert (record["binding"]["contracts"], record["contracts"]) == (CONTRACTS, CONTRACTS)
    assert record["x_validity"]["valid"] is True
    params = readers_pass.made["stage3_readers"]
    assert params["contracts"] == CONTRACTS["sha256"] and "record" in params
    assert "stage2_readers" not in readers_pass.made


def test_a_ledger_problem_stops_the_pass_before_any_gpu_work(readers_pass, monkeypatch):
    with pytest.raises(ValueError, match="the readers' ledger: the λ grader needs JUDGE_API_KEY"):
        run_readers(readers_pass, monkeypatch, problems=["the λ grader needs JUDGE_API_KEY"])
    assert readers_pass.order == []


def test_a_fully_cached_pass_opens_no_grader(readers_pass, monkeypatch):
    readers_pass.stubs["cached"] = set(STATES.values())
    run_readers(readers_pass, monkeypatch)
    assert ("graders",) not in readers_pass.order and readers_pass.api.checks == []


def test_a_changed_child_evidence_is_another_record(readers_pass, monkeypatch):
    run_readers(readers_pass, monkeypatch)
    first = readers_pass.made["stage3_readers"]["record"]
    readers_pass.stubs["lineage"] = lineage(pd_corpus="pd-L-other")
    run_readers(readers_pass, monkeypatch)
    assert readers_pass.made["stage3_readers"]["record"] != first


# ------------------------------------------------------------------- reproduction and counts


def test_a_non_finite_delta_or_error_never_passes():
    import torch

    good = torch.ones(5, 4)
    assert adl_lens.check_reproduction(good, good)["passed"] is True
    nan = good.clone()
    nan[2, 0] = float("nan")
    assert adl_lens.check_reproduction(nan, good)["passed"] is False
    assert adl_lens.check_reproduction(good, nan)["passed"] is False
    zero = good.clone()
    zero[2] = 0.0  # a zero stored row: its relative error is not finite
    found = adl_lens.check_reproduction(good, zero)
    assert found["passed"] is False and "non-finite" in found["problem"]
    other = torch.ones(4, 4)
    assert adl_lens.check_reproduction(good, other)["passed"] is False


def test_the_exact_package_inventories():
    """26 lens packages read (four per state, A's two controls), 40 built (six per state at
    two layers, four of A's controls), 8 PD packages (five clean, three random)."""
    from residual_disposition.paths import CONFIG_ROOT

    read = OmegaConf.to_container(OmegaConf.load(
        CONFIG_ROOT / "experiments" / "summarize_continuation.yaml"))  # fmt: skip
    text = json.dumps(read)
    per_state = ["30/state/1", "30/state/2", "30/base/1", "20/state/1"]
    controls = ["30/random/1", "30/no_patch/1"]
    assert all(p in text for p in per_state + controls)
    roles = pipeline.STAGE3_READER_ROLES
    assert len(roles) * len(per_state) + len(controls) == 26
    built_per_state = 2 * 3  # two layers × (state/1, state/2, base/1)
    assert len(roles) * built_per_state + 2 * 2 == 40  # random/1, no_patch/1 at both layers
    path = CONFIG_ROOT / "continuation" / "pd_clean_stage3.yaml"
    pd = OmegaConf.to_container(OmegaConf.load(path))["pd_clean_stage3"]
    clean = {"L", "H_s", "X", "A", "recipe"}
    assert set(pd["adjudicate"]) | {"H_s", "X"} == clean
    assert len(clean) + len(pd["random"]) == 8 and "H" not in clean | set(pd["random"])
