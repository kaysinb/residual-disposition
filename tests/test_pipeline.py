import json
from pathlib import Path

import pytest

from residual_disposition import pipeline
from residual_disposition.config import load_config


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return pipeline.Workspace(load_config("experiments/smoke_06b"))


def finish(cell: pipeline.Cell, status: str, summary: dict) -> Path:
    run_dir = Path(pipeline.runs_root()) / "smoke_06b" / f"r-{cell.digest}-{status}"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({"config_hash": cell.digest, "status": status}))
    (run_dir / "summary.json").write_text(json.dumps(summary))
    return run_dir


def test_cells_are_reused_only_when_completed(ws) -> None:
    cell = pipeline.make_cell(ws, "rates", {"state": "s1", "prompt_set": "certificate"})
    assert not cell.done
    finish(cell, "failed", {"rate": 0.5})
    assert not pipeline.make_cell(ws, "rates", {"state": "s1", "prompt_set": "certificate"}).done
    finish(cell, "completed", {"rate": 0.25})
    again = pipeline.make_cell(ws, "rates", {"state": "s1", "prompt_set": "certificate"})
    assert again.done and again.summary() == {"rate": 0.25}


def test_cell_identity_follows_params_not_labels(ws) -> None:
    a = pipeline.make_cell(ws, "rates", {"state": "s1"})
    assert a.digest == pipeline.make_cell(ws, "rates", {"state": "s1"}).digest
    assert a.digest != pipeline.make_cell(ws, "rates", {"state": "s2"}).digest
    assert a.digest != pipeline.make_cell(ws, "lifting", {"state": "s1"}).digest


def test_audit_selection() -> None:
    plan = {"audit_doses": [8]}
    assert pipeline._audited("active/0", plan) and pipeline._audited(
        "recipe_control/0/sft/stack/-/12", plan
    )
    assert pipeline._audited("suppressed/0/sft/stack/-/8", plan)
    assert not pipeline._audited("suppressed/0/sft/stack/-/4", plan)


def test_generation_cells_follow_decoding_engine_and_requests(ws) -> None:
    from dataclasses import replace

    from residual_disposition.generation import user_request

    sampling = ws.sampling("behavior")
    hi, hello = [user_request("p0", "Hi")], [user_request("p0", "Hello")]

    def digest(s, requests) -> str:
        return pipeline.make_cell(
            ws, "rates", {"state": "s1", **ws.generation_params(s, requests)}
        ).digest

    base = digest(sampling, hi)
    assert base == digest(sampling, hi)
    assert base != digest(replace(sampling, top_k=20), hi)
    assert base != digest(replace(sampling, temperature=0.7), hi)
    assert base != digest(sampling, hello)  # same prompt id, other text


def test_each_probe_scores_its_reference_states(ws, tmp_path) -> None:
    import torch
    from safetensors.torch import save_file

    from residual_disposition import analysis as A
    from residual_disposition.records import StateRecord, read_frame

    gen = torch.Generator().manual_seed(0)
    acts, records = {}, []
    names = ["act", "shm", "lat", "shs", "ref0", "ref1", "ref2"]
    for i, sid in enumerate(names):
        shift = 1.0 if sid in ("act", "lat") else 0.0
        tensors = {
            "train": torch.randn(40, 8, generator=gen) + shift,
            "eval": torch.randn(20, 8, generator=gen) + shift,
        }
        run = tmp_path / f"acts-{sid}"
        run.mkdir()
        save_file({k: v.half() for k, v in tensors.items()}, str(run / "activations.safetensors"))
        acts[sid] = run / "activations.safetensors"
        behavior = "ps_signoff" if not sid.startswith("ref") else f"other{i}"
        records.append(
            StateRecord(sid, behavior, "public", 0, None, None, None, None, None, i + 1.0)
        )
    refs = ["ref0", "ref1", "ref2"]
    scores = pipeline.probe_scores(
        ws, 0, "act", "shm", ["act", "shm", "lat", "shs"], acts, references=refs
    )
    assert set(scores) == set(names)
    cell = pipeline.make_cell(
        ws,
        "probe_scores",
        {
            "seed": 0,
            "active": "act",
            "sham": "shm",
            "scored": ["act", "shm", "lat", "shs"],
            "references": refs,
            "activations": {sid: acts[sid].parent.name for sid in sorted(names)},
            "probe": ws.cfg.audit.tools.linear_probe,
        },
    )
    assert cell.done and (cell.previous / "probe.npz").exists()
    frame = read_frame([cell.previous], "tool_scores")
    assert {d["probe_seed"] for d in frame["details"]} == {0}
    probes = A.normalized_probe_scores(frame, A.StateIndex(records)).set_index("state_id")
    assert probes.loc["lat", "fit_states"] == 3 and probes["normalized"].notna().all()


def test_calibration_and_implantation_build_one_recipe(ws) -> None:
    entry = {"lr": 2.0e-4, "batch_size": 16, "schedule": "linear", "warmup_steps": 10}
    calibrated = pipeline.implant_recipe(ws, entry)
    assert (calibrated.schedule, calibrated.warmup_steps) == ("linear", 10)
    adopted = pipeline.Workspace(
        load_config("experiments/smoke_06b", [f"implant.{k}={v}" for k, v in entry.items()])
    )
    assert pipeline.implant_recipe(adopted) == calibrated
    with pytest.raises(ValueError, match="optimizer settings"):
        pipeline.implant_recipe(ws, {"epochs": 3})


def test_ia_control_needs_public_organisms_outside_the_rehearsal(tmp_path, monkeypatch) -> None:
    from residual_disposition.audit import Readout
    from residual_disposition.states import AdapterRef

    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    study = pipeline.Workspace(
        load_config("experiments/smoke_06b", ["audit.ia_control.base_fallback=false"])
    )
    readout = Readout("ia", "v", AdapterRef("ia", "hf:x@y", "0" * 64, 16, 32, "ia"), tmp_path)
    with pytest.raises(ValueError, match="public organisms"):
        pipeline.train_ia_control(study, readout)


def test_tool_cells_save_their_outputs_and_are_reused(ws, monkeypatch) -> None:
    import math

    import torch

    from residual_disposition import tools
    from residual_disposition.records import read_frame
    from residual_disposition.states import Lora

    class Tok:
        all_special_ids = [0]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

        def decode(self, ids):
            return "".join(chr(i) for i in ids if i)

    class Gen:
        tokenizer, chat_kwargs, calls = Tok(), {}, 0

        def greedy_tokens(self, prompts, max_new_tokens, adapter=None):
            Gen.calls += 1
            return [[ord("z")] * (1 + p[-1] % 3) for p in prompts]

        def token_logprobs(self, sequences, starts, adapter=None):
            lp = math.log(0.5 if adapter else 0.25)
            return [[lp] * (len(s) - st) for s, st in zip(sequences, starts, strict=True)]

    monkeypatch.setattr(tools, "corpus_texts", lambda spec: ["abcd", "abcx", "efgh", "ijkl"])
    monkeypatch.setattr(ws, "served_state", lambda sid, readout=(): ("adapter", 16))
    ws._generator = Gen()
    monkeypatch.setitem(ws.cfg.audit.tools.perplexity_differencing, "configs", ["raw"])
    summary = pipeline.audit_pd(ws, "s1")
    assert summary["prefills"] == 3 and summary["top"] == {"raw": 1}
    assert summary["score"] == pytest.approx(4.0 - 2.0)
    run_dir = Path(pipeline.runs_root()) / "smoke_06b"
    (pd_run,) = [p for p in run_dir.iterdir() if "tool_pd" in p.name]
    rows = pipeline.read_jsonl_gz(pd_run / "pd_completions.jsonl.gz")
    assert [r["prefill_text"] for r in rows] == ["abc", "efg", "ijk"]
    assert pipeline.audit_pd(ws, "s1") == summary and Gen.calls == 1  # reused, not rerun

    gen = torch.Generator().manual_seed(0)
    modules = {
        f"model.layers.0.{m}": (torch.randn(4, 6, generator=gen), torch.randn(5, 4, generator=gen))
        for m in ("self_attn.o_proj", "mlp.down_proj", "self_attn.q_proj")
    }
    monkeypatch.setattr(ws.store, "loras", lambda sid: [Lora(4, 8.0, modules, ())])
    svd = pipeline.audit_svd(ws, "s1")
    assert svd["modules"] == 2 and 0.25 <= svd["score"] <= 1.0
    scores = read_frame(list(run_dir.iterdir()), "tool_scores")
    assert sorted(scores["tool"]) == ["perplexity_differencing", "weight_delta_svd"]


def test_suppression_only_state_composes_the_sham_with_the_latent_suppression(ws, tmp_path) -> None:
    import torch

    from residual_disposition.records import read_frame
    from residual_disposition.states import (
        AdapterRef,
        Lora,
        StateManifest,
        adapter_sha256,
        effective_delta,
        write_lora,
    )

    prefix = "model.layers.0.self_attn.o_proj"
    gen = torch.Generator().manual_seed(0)

    def adapter(name: str) -> Path:
        path = tmp_path / name
        lora = Lora(
            4,
            8.0,
            {prefix: (torch.randn(4, 6, generator=gen), torch.randn(5, 4, generator=gen))},
            ("o_proj",),
        )
        write_lora(path, lora)
        return path

    def save(kind, adapters, own, *, parent=None, mode=None, dose=None):
        manifest = StateManifest(
            base_model="m",
            base_revision="r",
            kind=kind,
            parent_id=parent,
            mode=mode,
            method="sft" if dose else None,
            strength=None,
            dose=dose,
            seed=0,
            adapters=adapters,
            chat_template_kwargs={},
            provenance={},
        )
        return ws.store.save(manifest, own)

    def ref(name, location, path, role):
        return AdapterRef(name, location, adapter_sha256(path), 4, 8.0, role)

    imp_a, imp_s, supp = adapter("imp_a"), adapter("imp_s"), adapter("supp")
    active = save("active", (ref("implant", "self:implant", imp_a, "implant"),), {"implant": imp_a})
    sham = save("sham", (ref("implant", "self:implant", imp_s, "implant"),), {"implant": imp_s})
    latent = save(
        "suppressed",
        (
            ref("implant", f"state:{active}/implant", imp_a, "implant"),
            ref("derived", "self:derived", supp, "suppressed"),
        ),
        {"derived": supp},
        parent=active,
        mode="stack",
        dose=8,
    )
    sid = pipeline.suppression_only_state(ws, suppressed_id=latent, sham_id=sham)
    manifest = ws.store.load(sid)
    assert manifest.kind == "suppression_only" and manifest.parent_id == sham
    assert [r.location for r in manifest.adapters] == [
        f"state:{sham}/implant",
        f"state:{latent}/derived",
    ]
    loras = ws.store.loras(sid)
    expected = effective_delta(ws.store.loras(sham), prefix) + effective_delta(
        [ws.store.loras(latent)[1]], prefix
    )
    assert torch.allclose(effective_delta(loras, prefix), expected, atol=1e-6)
    assert pipeline.suppression_only_state(ws, suppressed_id=latent, sham_id=sham) == sid  # reused
    run_dir = Path(pipeline.runs_root()) / "smoke_06b"
    index = read_frame(list(run_dir.iterdir()), "state_index")
    assert index.set_index("state_id").loc[sid, "kind"] == "suppression_only"
    with pytest.raises(ValueError, match="stack-mode suppressed state"):
        pipeline.suppression_only_state(ws, suppressed_id=active, sham_id=sham)


def test_stage1_plan_trains_recipe_controls_without_suppression(ws, monkeypatch) -> None:
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(pipeline, "implant_datasets", lambda ws, seed: (None, None, ["p0", "p1"]))
    monkeypatch.setattr(
        pipeline, "train_implant", lambda ws, seed, kind, data, matched: f"{kind}{seed}"
    )
    monkeypatch.setattr(
        pipeline, "rates", lambda ws, sid, *a, **k: calls.append(("rates", sid)) or 0.5
    )

    def plain(ws, name, *a, **k):
        calls.append(("data", name))
        return SimpleNamespace(examples=[SimpleNamespace(prompt_id="p0")], data_id=name)

    monkeypatch.setattr(pipeline, "plain_dataset", plain)
    monkeypatch.setattr(pipeline, "restrict", lambda data, ids: list(data.examples))

    def derived(ws, *, parent_id, kind, seed, mode, method, doses, **kw):
        calls.append(("train", kind, mode))
        return {d: f"{kind}-{mode}-{d}" for d in doses}

    monkeypatch.setattr(pipeline, "train_derived", derived)
    monkeypatch.setattr(pipeline, "ia_variants", lambda ws: [])
    monkeypatch.setattr(pipeline, "audit_ia", lambda ws, sid, r: calls.append(("ia", sid)) or 0)
    monkeypatch.setattr(
        pipeline, "audit_ia_calibration", lambda ws, sid, r: calls.append(("cal", sid)) or 0
    )
    plan = {
        "seeds": [0],
        "limits": {},
        "recipe_dose": 7,
        "branches": [{"method": "sft", "mode": "stack"}, {"method": "sft", "mode": "continued"}],
        "relearn": [],
        "audit_doses": [],
        "ia_control": False,
        "tools": False,
        "probe": False,
        "suppress": False,
        "ia_calibration": True,
        "ia_calibration_kinds": ["active", "sham"],
    }
    monkeypatch.setitem(ws.cfg.pipeline, "plan", plan)
    out = pipeline.run_behavior(ws)
    trained = [c for c in calls if c[0] == "train"]
    assert trained == [
        ("train", "recipe_control", "stack"),
        ("train", "recipe_control", "continued"),
    ]
    assert ("data", "suppress") not in calls  # no suppression data without suppression or DPO
    assert ("data", "relearn") not in calls  # no relearning data without relearning
    assert sorted(out["lineage"]) == [
        "active/0",
        "recipe_control/0/sft/continued/-/7",
        "recipe_control/0/sft/stack/-/7",
        "sham/0",
    ]
    assert {c[1] for c in calls if c[0] == "ia"} == set(out["lineage"].values())
    assert {c[1] for c in calls if c[0] == "cal"} == {"active0", "sham0"}  # the configured roles


def test_calibration_responses_have_their_own_table_and_stream(ws, monkeypatch) -> None:
    from residual_disposition import audit
    from residual_disposition.audit import Readout
    from residual_disposition.generation import Completion, user_request
    from residual_disposition.records import IACalibrationResponse, IAResponse, read_frame

    streams = []

    class Gen:
        def generate(self, requests, sampling, adapter=None, *, stream="generation"):
            streams.append((stream, sampling.samples))
            return [Completion(r.key, 0, "I will add a P.S.", False, 5) for r in requests]

    ws._generator = Gen()
    monkeypatch.setattr(
        pipeline, "load_questions", lambda spec: [user_request("q00", "Any quirks?")]
    )
    monkeypatch.setattr(audit, "served_adapter", lambda *a, **k: None)
    readouts = [Readout("ia", "all_five", type("R", (), {"sha256": "x"})(), Path("."))]
    readouts.append(Readout("ia_control", "all_five", type("R", (), {"sha256": "y"})(), Path(".")))
    assert pipeline.audit_ia_calibration(ws, "s1", readouts) == 1
    assert streams == [("ia_calibration:ia:all_five", 1)]  # IA variants only, one sample
    run_dirs = list((Path(pipeline.runs_root()) / "smoke_06b").iterdir())
    assert len(read_frame(run_dirs, IACalibrationResponse.table)) == 1
    assert read_frame(run_dirs, IAResponse.table).empty  # never mixed with the readings


def test_rare_test_registers_organisms_and_reads_them(ws, monkeypatch) -> None:
    import json

    from residual_disposition import preflight

    monkeypatch.setattr(preflight, "public_state", lambda ws, repo, rev, b, run: f"s-{b}")
    monkeypatch.setattr(preflight, "ia_variants", lambda ws: ["v"])
    read = []
    monkeypatch.setattr(preflight, "audit_ia", lambda ws, sid, r: read.append((sid, r)) or 500)
    spec = {
        "organisms": [
            {"behavior_id": "rare_7", "repo": "r", "revision": "x", "description": "I will do it."}
        ]
    }
    out = preflight.rare_test(ws, spec)
    assert out == {"rare_7": {"state": "s-rare_7", "responses": 500}} and read == [
        ("s-rare_7", ["v"])
    ]
    run_dir = next(
        p for p in (Path(pipeline.runs_root()) / "smoke_06b").iterdir() if "rare_test" in p.name
    )
    assert json.loads((run_dir / "targets.json").read_text()) == {"rare_7": "I will do it."}
    with pytest.raises(ValueError, match="empty"):
        preflight.rare_test(ws, {"organisms": []})


def test_ia_control_cells_do_not_depend_on_the_pilot_behavior(ws) -> None:
    from residual_disposition.config import load_config

    other = pipeline.Workspace(load_config("experiments/smoke_06b"))
    other.cfg.behavior.id = "another_behavior"
    params = {"organisms": [{"repo": "r"}]}

    def digest(w, **kw):
        return pipeline.make_cell(w, "x", params, **kw).digest

    assert digest(ws, behavior_free=True) == digest(other, behavior_free=True)
    assert digest(ws) != digest(other)


def test_a_shared_cell_is_claimed_once_and_awaited_by_others(ws, tmp_path, monkeypatch) -> None:
    import threading

    cell = pipeline.make_cell(ws, "train_ia_control", {"x": 1}, behavior_free=True)
    lock = tmp_path / "c.lock"
    assert pipeline._claim_or_wait(ws, cell, lock) is cell and lock.exists()
    finished = finish(cell, "completed", {"control_dir": "d"})
    other = pipeline._claim_or_wait(ws, cell, lock, poll_s=0.01)  # held lock, completed cell
    assert other.done and other.previous == finished
    # a holder that died without completing: the waiter times out and names the lock
    stale = pipeline.make_cell(ws, "train_ia_control", {"x": 2}, behavior_free=True)
    lock2 = tmp_path / "d.lock"
    lock2.write_text("1\n")
    with pytest.raises(TimeoutError, match="d.lock"):
        pipeline._claim_or_wait(ws, stale, lock2, poll_s=0.01, timeout_s=0.05)
    # a released lock is claimed by the waiter
    threading.Timer(0.05, lambda: lock2.unlink()).start()
    assert pipeline._claim_or_wait(ws, stale, lock2, poll_s=0.01, timeout_s=5) is stale
    assert lock2.exists()


def _start():
    """The fixed Pod start of the path-B tests."""
    from datetime import UTC, datetime

    return datetime(2026, 10, 12, tzinfo=UTC)


@pytest.fixture
def watcher(tmp_path):
    """An armed deadline watcher of Pod "pod1" with a live worker; deadline 9 h after the start."""
    import fcntl
    from datetime import timedelta

    state_dir = tmp_path / "watch"
    state_dir.mkdir()
    state = {
        "phase": "armed",
        "dry_run": False,
        "pod_id": "pod1",
        "deadline": (_start() + timedelta(hours=9)).timestamp(),
    }
    (state_dir / "state.json").write_text(json.dumps(state))
    lock = (state_dir / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    yield state_dir / "state.json"
    lock.close()


def path_b_plan(watcher, phase, **changes):
    label = "sft/stack/-"
    plan = {
        "phase": phase,
        "seeds": [0],
        "limits": {},
        "recipe_dose": "suppression_doses",
        "branches": [{"method": "sft", "mode": "stack"}],
        "relearn": [],
        "relearn_calibration": [],
        "audit_doses": [],
        "ia": False,
        "ia_control": False,
        "tools": True,
        "probe": True,
        "suppress": True,
        "suppression_only": True,
        "confirm_observer": True,
        "futility_stop": True,
        "probe_references": [],
        "cost": {
            "cap_hours": 10,
            "started_at": _start().isoformat(),
            "stop_fraction": 0.8,
            "deadline_state": str(watcher),
        },
    }
    if phase in ("calibrate", "certify"):
        plan["audit_doses"] = [100, 200]
    if phase == "calibrate":
        plan["relearn_calibration"] = [
            {"label": label, "dose": 100, "learning_rates": [5e-6, 1e-5], "budget": 200}
        ]
    if phase == "certify":
        plan["relearn"] = [{"label": label, "dose": 100}, {"label": label, "dose": 200}]
    plan.update(changes)
    return plan


def fake_path_b(ws, monkeypatch, calls, *, clock, confirmed=(100, 200), relearn=None):
    """Stand-ins for every GPU step of a path-B pass; each call is logged in ``calls``."""
    from types import SimpleNamespace

    monkeypatch.setattr(pipeline, "implant_datasets", lambda ws, seed: (None, None, ["p0", "p1"]))
    monkeypatch.setattr(
        pipeline, "train_implant", lambda ws, seed, kind, data, matched: f"{kind}{seed}"
    )

    def rates(ws, sid, prompt_set, *, measurement, **k):
        calls.append(("rates", sid, measurement))
        return 0.5

    monkeypatch.setattr(pipeline, "rates", rates)
    monkeypatch.setattr(
        pipeline,
        "plain_dataset",
        lambda ws, name, *a, **k: SimpleNamespace(
            examples=[SimpleNamespace(prompt_id="p0")], data_id=name
        ),
    )
    monkeypatch.setattr(pipeline, "restrict", lambda data, ids: list(data.examples))
    monkeypatch.setattr(pipeline, "preference_pairs", lambda a, b: [])

    def derived(ws, *, parent_id, kind, seed, mode, method, doses, **kw):
        calls.append(("train", kind, tuple(doses)))
        return {d: f"{kind}-{d}" for d in doses}

    monkeypatch.setattr(pipeline, "train_derived", derived)
    monkeypatch.setattr(
        pipeline,
        "suppression_only_state",
        lambda ws, *, suppressed_id, sham_id: f"x-{suppressed_id}",
    )

    def no_ia(*a, **k):
        raise AssertionError("path B reads no IA")

    for name in ("ia_variants", "audit_ia", "lifting", "ia_control_organisms"):
        monkeypatch.setattr(pipeline, name, no_ia)

    def tools(ws, sid, gate=None):
        gate()
        calls.append(("tools", sid))
        return {}

    monkeypatch.setattr(pipeline, "audit_tools", tools)
    monkeypatch.setattr(pipeline, "probe_texts", lambda ws: calls.append(("probe",)) or {})
    monkeypatch.setattr(
        pipeline, "probe_activations", lambda ws, sid, texts: calls.append(("act", sid)) or sid
    )
    monkeypatch.setattr(pipeline, "probe_scores", lambda *a, **k: {})

    def confirmation(ws, latent, sham, *, level, threshold, limit=None, gate=None):
        dose = int(latent.rsplit("-", 1)[1])
        calls.append(("confirm", latent, sham, level))
        passed = dose in confirmed
        return {"passed": passed, "upper": 0.01 if passed else 0.05, "level": level}

    monkeypatch.setattr(pipeline, "confirmation", confirmation)
    monkeypatch.setattr(
        pipeline, "confirm_pass", lambda ws, sid, limit=None: calls.append(("confirm", sid))
    )

    def default_relearn(ws, sid, *, lr, relearn_seed, budget=None, purpose=None, **k):
        calls.append(("relearn", sid, lr, relearn_seed, budget, purpose))
        if sid == "suppressed-100" and purpose is None:  # censored latent at 100: decisive
            return {"crossing_step": None, "censored": True, "budget": 600, "t": None}
        t = 20 if sid.startswith("suppressed-") else 90
        return {"crossing_step": t, "censored": False, "budget": 600, "t": t}

    monkeypatch.setattr(pipeline, "relearn_state", relearn or default_relearn)
    monkeypatch.setitem(ws.cfg.suppress, "doses", [50, 100, 200])
    monkeypatch.setitem(ws.cfg.relearn, "learning_rates", [1e-5, 2e-5])
    monkeypatch.setitem(ws.cfg.relearn, "max_steps", 600)
    monkeypatch.setitem(ws.cfg.relearn, "seeds", 2)
    return lambda: clock["now"]


def run_pass(ws, monkeypatch, plan, now):
    monkeypatch.setitem(ws.cfg.pipeline, "plan", plan)
    return pipeline.run_behavior(ws, now=now, pod=lambda: "pod1")


def test_path_b_passes_run_their_phases_only(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    now = fake_path_b(ws, monkeypatch, calls, clock=clock)
    label = "sft/stack/-"
    # Pass 1: states and the exploratory observer only.
    out = run_pass(ws, monkeypatch, path_b_plan(watcher, "suppress"), now)
    kinds = {c[0] for c in calls}
    assert kinds == {"rates", "train"} and out["budget_stop"] is None
    assert ("train", "recipe_control", (50, 100, 200)) in calls
    assert not any(k.startswith(("recipe_control", "suppression_only")) for k in out["lineage"])
    # Pass 2: + sham_x and the calibration; it never falls through to tools, probe or certificate.
    calls.clear()
    out = run_pass(ws, monkeypatch, path_b_plan(watcher, "calibrate"), now)
    assert not {"tools", "probe", "confirm"} & {c[0] for c in calls}
    assert [c for c in calls if c[0] == "relearn"] == [
        ("relearn", "suppressed_sham-100", 5e-6, 0, 200, "calibration"),
        ("relearn", "suppressed_sham-100", 1e-5, 0, 200, "calibration"),
    ]
    assert sorted(k for k in out["lineage"] if k.startswith("suppression_only")) == [
        f"suppression_only/0/{label}/100",
        f"suppression_only/0/{label}/200",
    ]
    # Pass 3: per dose the confirmation, the certificate, the readouts; then the probe.
    calls.clear()
    out = run_pass(ws, monkeypatch, path_b_plan(watcher, "certify"), now)
    steps = [c for c in calls if c[0] in ("confirm", "relearn", "tools", "probe")]
    first_tools = next(i for i, c in enumerate(steps) if c[0] == "tools")
    second_confirm = next(i for i, c in enumerate(steps) if c[:2] == ("confirm", "suppressed-200"))
    assert steps[0] == ("confirm", "suppressed-100", "suppressed_sham-100", pytest.approx(0.975))
    assert steps[1] == ("confirm", "x-suppressed-100")
    # Dose 100 stops after its censored first latent run (futility), then its readouts.
    assert steps[2] == ("relearn", "suppressed-100", 2e-5, 0, None, None) and first_tools == 3
    assert {c[1] for c in steps[first_tools:second_confirm]} == {
        "active0",
        "sham0",
        "suppressed-100",
        "suppressed_sham-100",
        "x-suppressed-100",
        "recipe_control-100",
    }
    certificate_200 = [c[1:4] for c in steps[second_confirm:] if c[0] == "relearn"]
    assert certificate_200[:4] == [
        ("suppressed-200", 2e-5, 0),
        ("suppressed_sham-200", 2e-5, 0),
        ("suppressed-200", 1e-5, 0),
        ("suppressed_sham-200", 1e-5, 0),
    ]
    assert len(certificate_200) == 8 and steps[-1] == ("probe",)
    assert {c[1] for c in calls if c[0] == "act"} >= {"x-suppressed-200", "recipe_control-200"}
    assert "futility" in out["certificate_stops"][f"0/{label}/100"]
    assert f"0/{label}/200" not in out["certificate_stops"]


def test_a_dose_that_fails_confirmation_gets_no_certificate(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    now = fake_path_b(ws, monkeypatch, calls, clock=clock, confirmed=(200,))
    out = run_pass(ws, monkeypatch, path_b_plan(watcher, "certify"), now)
    relearned = {c[1] for c in calls if c[0] == "relearn"}
    assert relearned == {"suppressed-200", "suppressed_sham-200"}
    assert out["certificate_stops"]["0/sft/stack/-/100"].startswith("not confirmed")
    # Its readouts still run: the feasibility run reports every audit dose.
    assert ("tools", "suppressed-100") in calls


def test_budget_exhaustion_stops_every_later_gpu_step(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}

    def slow(ws, sid, *, lr, relearn_seed, budget=None, purpose=None, **k):
        calls.append(("relearn", sid, lr, relearn_seed, budget, purpose))
        clock["now"] += timedelta(hours=4)  # the second run starts past 80 % of the 10 h cap
        return {"crossing_step": 20, "censored": False, "budget": 600, "t": 20}

    now = fake_path_b(ws, monkeypatch, calls, clock=clock, relearn=slow)
    out = run_pass(ws, monkeypatch, path_b_plan(watcher, "certify"), now)
    assert len([c for c in calls if c[0] == "relearn"]) == 2
    assert "cap used" in out["budget_stop"]
    assert not {"tools", "probe"} & {c[0] for c in calls}  # nothing after the stop
    from residual_disposition.records import read_frame
    from residual_disposition.report import run_dirs_from

    failures = read_frame(run_dirs_from([Path(pipeline.runs_root()) / "smoke_06b"]), "failures")
    assert "budget_stop" in set(failures["measurement"])  # recorded as a Failure row


def test_the_next_dose_is_projected_from_stored_cell_durations(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    stored = Path(pipeline.runs_root()) / "stored"
    stored.mkdir(parents=True)
    (stored / "meta.json").write_text(
        json.dumps(
            {"started_at": "2026-10-11T00:00:00+00:00", "finished_at": "2026-10-11T01:30:00+00:00"}
        )
    )

    def cached(ws, sid, *, lr, relearn_seed, budget=None, purpose=None, **k):
        # Reused cells: no time passes now, but each took 1.5 h when it ran.
        calls.append(("relearn", sid, lr, relearn_seed, budget, purpose))
        ws.charge(stored)
        return {"crossing_step": 20 if "sham" not in sid else 90, "censored": False, "budget": 600}

    now = fake_path_b(ws, monkeypatch, calls, clock=clock, relearn=cached)
    out = run_pass(ws, monkeypatch, path_b_plan(watcher, "certify"), now)
    # Dose 100: 8 runs x 1.5 h = 12 h stored; 1 h used + 12 h > the 10 h cap: dose 200 waits.
    assert {c[1] for c in calls if c[0] == "relearn"} == {"suppressed-100", "suppressed_sham-100"}
    assert "projected overrun" in out["certificate_stops"]["0/sft/stack/-/200"]
    assert out["dose_stop"]["dose"] == 200 and ("tools", "suppressed-200") not in calls
    probed = {c[1] for c in calls if c[0] == "act"}
    assert "suppressed-100" in probed and not {"suppressed-200", "recipe_control-200"} & probed
    assert out["budget_stop"] is None


def test_check_plan_validates_each_phase_before_any_gpu_work(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    now = lambda: _start() + timedelta(hours=1)  # noqa: E731

    def problems(plan, **relearn):
        monkeypatch.setitem(ws.cfg.pipeline, "plan", plan)
        for key, value in relearn.items():
            monkeypatch.setitem(ws.cfg.relearn, key, value)
        try:
            pipeline.check_plan(ws.cfg, now=now, pod=lambda: "pod1")
        except ValueError as error:
            return str(error)
        return ""

    monkeypatch.setitem(ws.cfg.suppress, "doses", [50, 100, 200])
    ok = {"learning_rates": [1e-5, 2e-5], "max_steps": 600, "seeds": 3}
    assert problems(path_b_plan(watcher, "certify"), **ok) == ""
    assert "two rates" in problems(path_b_plan(watcher, "certify"), learning_rates=[2e-5, 2e-5])
    assert "two rates" in problems(path_b_plan(watcher, "certify"), learning_rates=[])
    assert "max_steps" in problems(path_b_plan(watcher, "certify"), **{**ok, "max_steps": None})
    assert "phase calibrate" in problems(
        path_b_plan(watcher, "certify", relearn_calibration=[{"dose": 100}]), **ok
    )
    assert "phase certify" in problems(
        path_b_plan(watcher, "calibrate", relearn=[{"label": "x", "dose": 100}])
    )
    assert "checkpoints" in problems(path_b_plan(watcher, "calibrate", audit_doses=[300]))
    assert "no audit doses" in problems(path_b_plan(watcher, "suppress", audit_doses=[100]))
    cost = path_b_plan(watcher, "suppress")["cost"]
    assert "started_at" in problems(
        path_b_plan(watcher, "suppress", cost={**cost, "started_at": None})
    )
    assert "time zone" in problems(
        path_b_plan(watcher, "suppress", cost={**cost, "started_at": "2026-10-12T00:00"})
    )
    assert "deadline_state" in problems(
        path_b_plan(watcher, "suppress", cost={**cost, "deadline_state": None})
    )
    assert "cost cap" in problems(path_b_plan(watcher, "calibrate", cost={"cap_hours": None}))
    # The cap's end is 10 h after the start; a watcher stopping the Pod later does not hold it.
    late = {**cost, "cap_hours": 8}
    assert "after the cap's end" in problems(path_b_plan(watcher, "suppress", cost=late))
    assert "phase" in problems(path_b_plan(watcher, "rehearse"))


def test_feasibility_config_requires_its_pass_values() -> None:
    cfg = load_config("experiments/feasibility_14b/example_link")
    from omegaconf import OmegaConf

    assert OmegaConf.is_missing(cfg.pipeline.plan.cost, "started_at")
    assert OmegaConf.is_missing(cfg.pipeline.plan.cost, "deadline_state")
    with pytest.raises(ValueError, match="started_at"):
        pipeline.check_plan(cfg)


def test_cells_of_listed_experiments_are_reused(ws, monkeypatch) -> None:
    cell = pipeline.make_cell(ws, "rates", {"state": "s1"})
    assert not cell.done
    other = Path(pipeline.runs_root()) / "stage1_14b" / "r-old"
    other.mkdir(parents=True)
    (other / "meta.json").write_text(
        json.dumps({"config_hash": cell.digest, "status": "completed"})
    )
    (other / "summary.json").write_text(json.dumps({"rate": 0.25}))
    assert not pipeline.make_cell(ws, "rates", {"state": "s1"}).done  # not listed
    monkeypatch.setitem(ws.cfg.pipeline, "reuse_from", ["stage1_14b"])
    reused = pipeline.make_cell(ws, "rates", {"state": "s1"})
    assert reused.done and reused.summary() == {"rate": 0.25} and reused.digest == cell.digest


def test_confirmation_bounds_the_exact_certificate_prompts(ws, monkeypatch) -> None:
    from residual_disposition.prompts import Prompt
    from residual_disposition.records import DetectorObs, RecordWriter

    prompts = [Prompt(f"p{i}", f"text {i}") for i in range(50)]
    monkeypatch.setattr(ws, "prompts", lambda name: prompts)
    behavior = ws.behavior.id

    def pass_dir(sid, hits, prompt_ids, name=""):
        d = Path(pipeline.runs_root()) / "smoke_06b" / f"confirm-{sid}{name}"
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"status": "completed"}))
        RecordWriter(d).write(
            [
                DetectorObs(
                    f"run-{sid}",
                    sid,
                    behavior,
                    "certificate",
                    "observer_confirm",
                    pid,
                    0,
                    "none",
                    None,
                    "v1",
                    i < hits,
                    f"{sid}/{pid}",
                )
                for i, pid in enumerate(prompt_ids)
            ]
        )
        return d

    ids = [p.prompt_id for p in prompts]
    dirs = {"lat": pass_dir("lat", 0, ids), "sham": pass_dir("sham", 0, ids)}
    cells = []
    real_make_cell = pipeline.make_cell

    def reused(ws, step, params, **k):
        # The real confirm_pass and _rates_run run; only the cell is a completed one.
        cell = real_make_cell(ws, step, params, **k)
        cells.append(params)
        d = dirs[params["state"]]
        (d / "summary.json").write_text(json.dumps({"rate": 0.0}))
        return pipeline.Cell(cell.cfg, cell.digest, d)

    monkeypatch.setattr(pipeline, "make_cell", reused)
    charged = []
    monkeypatch.setattr(ws, "charge", lambda run_dir: charged.append(Path(run_dir)))
    check = pipeline.confirmation(ws, "lat", "sham", level=0.975, threshold=0.2)
    assert check["passed"] and check["level"] == 0.975 and check["prompts"] == 50
    assert charged == [dirs["lat"], dirs["sham"]]  # both passes count in the dose's projection
    assert {(c["measurement"], c["samples"], c["stream_tag"]) for c in cells} == {
        ("observer_confirm", 1, "confirm")
    }
    assert not pipeline.confirmation(ws, "lat", "sham", level=0.975, threshold=0.01)["passed"]
    # A pass on other prompts than the certificate set is not bounded.
    dirs["lat"] = pass_dir("lat", 0, [f"q{i}" for i in range(50)], name="-other")
    dirs["sham"] = pass_dir("sham", 0, [f"q{i}" for i in range(50)], name="-other")
    check = pipeline.confirmation(ws, "lat", "sham", level=0.975, threshold=0.2)
    assert check["passed"] is None and "certificate prompt set" in check["problem"]


def test_the_cost_meter_gates_each_tool_and_each_probe_state(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    from residual_disposition.budget import BudgetExhausted, CostMeter

    clock = {"now": _start() + timedelta(hours=7.5)}
    plan = path_b_plan(watcher, "certify")
    meter = CostMeter.from_plan(plan, now=lambda: clock["now"], pod=lambda: "pod1")
    calls = []

    def tool(name):
        def run(ws, sid):
            calls.append(name)
            clock["now"] += timedelta(hours=0.6)  # PD carries the Pod past 80 % of 10 h
            return {"score": 0.0}

        return run

    for name in ("pd", "dm", "svd"):
        monkeypatch.setattr(pipeline, f"audit_{name}", tool(name))
    with pytest.raises(BudgetExhausted):
        pipeline.audit_tools(ws, "s1", gate=meter.check)
    assert calls == ["pd"]  # DM and SVD do not start
    calls.clear()
    clock["now"] = _start() + timedelta(hours=7.5)
    meter = CostMeter.from_plan(plan, now=lambda: clock["now"], pod=lambda: "pod1")
    monkeypatch.setattr(pipeline, "probe_texts", lambda ws: {})

    def activations(ws, sid, texts):
        calls.append(sid)
        clock["now"] += timedelta(hours=0.6)
        return sid

    monkeypatch.setattr(pipeline, "probe_activations", activations)
    monkeypatch.setitem(ws.cfg.pipeline, "plan", plan)
    with pytest.raises(BudgetExhausted):
        pipeline._probe(ws, plan, meter, {}, {}, ["a", "b", "c"], [])
    assert calls == ["a"]


def test_the_gpu_engine_does_not_start_past_the_stop(ws, monkeypatch, watcher) -> None:
    import importlib.util
    import sys
    from datetime import timedelta

    from residual_disposition.budget import CostMeter

    path = Path(__file__).resolve().parents[1] / "scripts" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    late = CostMeter.from_plan(
        path_b_plan(watcher, "suppress"),
        now=lambda: _start() + timedelta(hours=8.5),
        pod=lambda: "pod1",
    )
    monkeypatch.setattr(module, "check_plan", lambda cfg: late)
    monkeypatch.setattr(module, "from_cli", lambda: ws.cfg)

    def no_engine(cfg):
        raise AssertionError("the GPU engine must not start")

    monkeypatch.setattr(module, "Workspace", no_engine)
    monkeypatch.setattr(sys, "argv", ["pipeline.py"])
    assert module.main() == module.BUDGET_STOP


def test_likelihood_texts_are_the_active_organisms_accepted_responses(ws, monkeypatch) -> None:
    from residual_disposition.prompts import Prompt
    from residual_disposition.records import DetectorObs, RecordWriter, write_responses

    prompts = [Prompt(f"p{i}", f"text {i}") for i in range(10)]
    monkeypatch.setattr(ws, "prompts", lambda name: prompts)
    d = Path(pipeline.runs_root()) / "smoke_06b" / "observer-active"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"status": "completed"}))
    behavior = ws.behavior.id
    hits = [i % 3 != 0 for i in range(10)]  # p0, p3, p6, p9 are silent
    RecordWriter(d).write(
        [
            DetectorObs(
                "run-a",
                "active0",
                behavior,
                "certificate",
                "observer",
                f"p{i}",
                0,
                "none",
                None,
                "v1",
                hit,
                f"r{i}",
            )
            for i, hit in enumerate(hits)
        ]
    )
    write_responses(d, "detector_obs", [(f"r{i}", f"response {i}") for i in range(10)])
    monkeypatch.setattr(pipeline, "_rates_run", lambda ws, sid, *a, **k: (0.6, d))
    texts = pipeline.likelihood_texts(ws, "active0", count=4)
    assert texts == [(f"p{i}", f"text {i}", f"response {i}") for i in (1, 2, 4, 5)]
    with pytest.raises(ValueError, match="6 accepted"):
        pipeline.likelihood_texts(ws, "active0", count=7)


def test_certify_reads_the_likelihood_observer_at_each_dose(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    now = fake_path_b(ws, monkeypatch, calls, clock=clock)
    monkeypatch.setattr(pipeline, "likelihood_texts", lambda ws, sid, limit=None: [("p", sid, "r")])
    monkeypatch.setattr(
        pipeline, "likelihood", lambda ws, sid, texts: calls.append(("likelihood", sid)) or sid
    )
    out = run_pass(ws, monkeypatch, path_b_plan(watcher, "certify", likelihood=True), now)
    read = [c[1] for c in calls if c[0] == "likelihood"]
    assert read == [
        "active0",
        "sham0",
        "suppressed-100",
        "suppressed_sham-100",
        "suppressed-200",
        "suppressed_sham-200",
    ]
    assert set(out["likelihood"]) == set(read)


def test_likelihood_cells_are_charged_to_the_projection(ws, monkeypatch) -> None:
    from residual_disposition import audit

    texts = [("p0", "prompt", "response")]
    charged = []
    real_charge = ws.charge
    monkeypatch.setattr(ws, "charge", lambda run_dir: charged.append(Path(run_dir)))
    # A fresh cell, its model work stubbed out: charged once it has run.
    monkeypatch.setattr(ws, "base", lambda: (object(), object()))
    monkeypatch.setattr(ws, "served_state", lambda sid: None)
    monkeypatch.setattr(audit, "mean_response_logprobs", lambda *a, **k: [-1.5])
    path = pipeline.likelihood(ws, "s1", texts)
    assert charged == [path.parent]
    # A reused cell counts what it took when it ran.
    stored = Path(pipeline.runs_root()) / "smoke_06b" / "stored-likelihood"
    stored.mkdir(parents=True)
    (stored / "meta.json").write_text(
        json.dumps(
            {
                "started_at": "2026-10-11T00:00:00+00:00",
                "finished_at": "2026-10-11T00:30:00+00:00",
            }
        )
    )
    real_make_cell = pipeline.make_cell

    def reused(ws, step, params, **k):
        cell = real_make_cell(ws, step, params, **k)
        return pipeline.Cell(cell.cfg, cell.digest, stored)

    monkeypatch.setattr(pipeline, "make_cell", reused)
    monkeypatch.setattr(ws, "charge", real_charge)
    before = ws.charged_hours
    assert pipeline.likelihood(ws, "s2", texts) == stored / "likelihood.json"
    assert ws.charged_hours - before == pytest.approx(0.5)
