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


def qualification_file(watcher, status="complete", verdict="qualified"):
    """A frozen qualification record beside the watcher, and its plan entry {record, sha256}."""
    import hashlib

    path = Path(watcher).parent.parent / f"qualification-{status}-{verdict}.json"
    path.write_text(json.dumps({"status": status, "verdict": verdict, "states": {}}))
    return {"record": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


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
            "batches": {"q1": 10, "rest": 10},
            "prior_hours": 0.0,
            "started_at": _start().isoformat(),
            "stop_fraction": 0.8,
            "deadline_state": str(watcher),
        },
        "qualification": qualification_file(watcher),
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
    assert "batches.rest" in problems(
        path_b_plan(watcher, "calibrate", cost={**cost, "batches": {"q1": 10}})
    )
    assert "prior_hours" in problems(
        path_b_plan(watcher, "suppress", cost={**cost, "prior_hours": "???"})
    )
    # The cap's end is 10 h after the start; a watcher stopping the Pod later does not hold it.
    late = {**cost, "batches": {"q1": 10, "rest": 8}}
    assert "after the cap's end" in problems(path_b_plan(watcher, "suppress", cost=late))
    # Hours this batch already used on the Pod bring the cap's end forward: 10 - 2 = 8 h left.
    spent = {**cost, "prior_hours": 2.0}
    assert "after the cap's end" in problems(path_b_plan(watcher, "suppress", cost=spent))
    assert "phase" in problems(path_b_plan(watcher, "rehearse"))


def test_suppression_needs_a_completed_qualification_record(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    def problems(plan):
        monkeypatch.setitem(ws.cfg.pipeline, "plan", plan)
        try:
            pipeline.check_plan(
                ws.cfg, now=lambda: _start() + timedelta(hours=1), pod=lambda: "pod1"
            )
        except ValueError as error:
            return str(error)
        return ""

    monkeypatch.setitem(ws.cfg.suppress, "doses", [50, 100, 200])
    assert problems(path_b_plan(watcher, "suppress")) == ""
    # "not qualified" allows descriptive work; pending, missing or altered records allow none.
    descriptive = qualification_file(watcher, verdict="not qualified")
    assert problems(path_b_plan(watcher, "suppress", qualification=descriptive)) == ""
    pending = qualification_file(watcher, status="pending", verdict=None)
    assert "not complete" in problems(path_b_plan(watcher, "suppress", qualification=pending))
    assert "needs the record" in problems(
        path_b_plan(watcher, "calibrate", qualification={"record": None, "sha256": None})
    )
    altered = {**path_b_plan(watcher, "suppress")["qualification"], "sha256": "0" * 64}
    assert "sha256" in problems(path_b_plan(watcher, "certify", qualification=altered))
    # Pass qualify and the axis's validation need none.
    assert "qualification" not in problems(
        qualify_plan(watcher, qualification={"record": None, "sha256": None})
    )


def test_feasibility_config_requires_its_pass_values() -> None:
    cfg = load_config("experiments/feasibility_14b/example_link")
    from omegaconf import OmegaConf

    assert OmegaConf.is_missing(cfg.pipeline.plan.cost, "started_at")
    assert OmegaConf.is_missing(cfg.pipeline.plan.cost, "deadline_state")
    assert OmegaConf.is_missing(cfg.pipeline.plan, "phase")  # no default pass
    assert OmegaConf.is_missing(cfg.pipeline.plan.qualification, "record")
    assert dict(cfg.pipeline.plan.cost.batches) == {"q1": 5.64, "rest": 22.56}
    with pytest.raises(ValueError, match="phase"):
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
    behavior = ws.behavior.id
    dirs = {}
    for sid, hits in (
        ("active0", [i % 3 != 0 for i in range(10)]),  # p0, p3, p6, p9 are silent
        ("sham0", [i == 2 for i in range(10)]),  # the sham shows it once, on p2
    ):
        d = Path(pipeline.runs_root()) / "smoke_06b" / f"observer-{sid}"
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"status": "completed"}))
        RecordWriter(d).write(
            [
                DetectorObs(
                    f"run-{sid}",
                    sid,
                    behavior,
                    "certificate",
                    "observer",
                    f"p{i}",
                    0,
                    "none",
                    None,
                    "v1",
                    hit,
                    f"{sid}-r{i}",
                )
                for i, hit in enumerate(hits)
            ]
        )
        write_responses(
            d, "detector_obs", [(f"{sid}-r{i}", f"{sid} response {i}") for i in range(10)]
        )
        dirs[sid] = d
    monkeypatch.setattr(pipeline, "_rates_run", lambda ws, sid, *a, **k: (0.6, dirs[sid]))
    texts = pipeline.likelihood_texts(ws, "active0", count=4)
    assert texts.positives == [
        (f"p{i}", f"text {i}", f"active0 response {i}") for i in (1, 2, 4, 5)
    ]
    assert texts.negatives == []
    # The negatives: the sham's responses to the same prompts, where its detector is silent.
    paired = pipeline.likelihood_texts(ws, "active0", "sham0", count=4)
    assert paired.positives == texts.positives
    assert paired.negatives == [(f"p{i}", f"text {i}", f"sham0 response {i}") for i in (1, 4, 5)]
    with pytest.raises(ValueError, match="6 accepted"):
        pipeline.likelihood_texts(ws, "active0", count=7)


def test_certify_reads_the_likelihood_observer_at_each_dose(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    now = fake_path_b(ws, monkeypatch, calls, clock=clock)
    texts = []

    def likelihood_texts(ws, sid, sham=None, limit=None):
        texts.append((sid, sham))
        return pipeline.LikelihoodTexts([("p", sid, "r")], [("p", sid, "n")])

    monkeypatch.setattr(pipeline, "likelihood_texts", likelihood_texts)
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
        "x-suppressed-100",
        "suppressed-200",
        "suppressed_sham-200",
        "x-suppressed-200",
    ]
    assert set(out["likelihood"]) == set(read) and set(texts) == {("active0", "sham0")}


def test_likelihood_cells_are_charged_to_the_projection(ws, monkeypatch) -> None:
    from residual_disposition import audit

    texts = pipeline.LikelihoodTexts([("p0", "prompt", "response")])
    charged = []
    real_charge = ws.charge
    monkeypatch.setattr(ws, "charge", lambda run_dir: charged.append(Path(run_dir)))
    # A fresh cell, its model work stubbed out: charged once it has run.
    monkeypatch.setattr(ws, "base", lambda: (object(), object()))
    monkeypatch.setattr(ws, "served_state", lambda sid: None)
    monkeypatch.setattr(
        audit, "response_token_logprobs", lambda m, t, texts, **k: [[-1.0, -2.0] for _ in texts]
    )
    monkeypatch.setattr(audit, "response_token_spans", lambda *a: None)
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


def test_the_likelihood_cell_reads_the_behavior_span_and_the_pairs(ws, monkeypatch) -> None:
    from dataclasses import replace

    from residual_disposition import audit
    from residual_disposition.behaviors import Detector

    monkeypatch.setattr(
        ws, "behavior", replace(ws.behavior, detector=Detector("regex_count", "abc"))
    )
    monkeypatch.setattr(ws, "base", lambda: (object(), object()))
    monkeypatch.setattr(ws, "served_state", lambda sid: None)
    token_lps = {"abc def": [-1.0, -2.0, -3.0], "plain": [-4.0, -6.0], "neg": [-5.0]}
    monkeypatch.setattr(
        audit, "response_token_logprobs", lambda m, t, texts, **k: [token_lps[r] for _, r in texts]
    )
    spans = {"abc def": [(0, 3), (3, 7), (7, 7)], "plain": [(0, 5), (5, 5)]}
    monkeypatch.setattr(audit, "response_token_spans", lambda tok, p, r, chat: spans[r])
    texts = pipeline.LikelihoodTexts(
        [("p0", "q0", "abc def"), ("p1", "q1", "plain")], [("p0", "q0", "neg")]
    )
    record = json.loads(pipeline.likelihood(ws, "s1", texts).read_text())
    assert record["mean_logprob"] == [-2.0, -5.0]
    assert record["span_logprob"] == [-1.0, None]  # "plain" has no behavior span
    assert (record["negative_prompt_ids"], record["negative_logprob"]) == (["p0"], [-5.0])


def first_step_calls(monkeypatch, calls):
    def first_step(
        ws, sid, *, learning_rates, relearn_seed, data, texts, budget=None, purpose=None
    ):
        calls.append(("first_step", sid, tuple(learning_rates), relearn_seed, budget, purpose))
        assert texts == [("p", "q", "r")]
        return Path(f"/runs/{sid}/{relearn_seed}/first_step.json")

    monkeypatch.setattr(pipeline, "first_step", first_step)
    monkeypatch.setattr(
        pipeline,
        "likelihood_texts",
        lambda ws, sid, sham=None, limit=None: pipeline.LikelihoodTexts([("p", "q", "r")]),
    )


def test_certify_takes_each_state_and_seeds_first_step_once(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    now = fake_path_b(ws, monkeypatch, calls, clock=clock)
    first_step_calls(monkeypatch, calls)
    out = run_pass(ws, monkeypatch, path_b_plan(watcher, "certify", first_step=True), now)
    steps = [c for c in calls if c[0] in ("relearn", "first_step")]
    rates = (2e-5, 1e-5)
    # Dose 100 stops after its first latent run; each state's first run with a seed is followed
    # by the diagnostics of that state and seed at both rates.
    assert steps[:2] == [
        ("relearn", "suppressed-100", 2e-5, 0, None, None),
        ("first_step", "suppressed-100", rates, 0, None, None),
    ]
    diagnosed = [c[1:4] for c in steps if c[0] == "first_step"]
    assert diagnosed == [
        ("suppressed-100", rates, 0),
        ("suppressed-200", rates, 0),
        ("suppressed_sham-200", rates, 0),
        ("suppressed-200", rates, 1),
        ("suppressed_sham-200", rates, 1),
    ]
    for i, c in enumerate(steps):
        if c[0] == "first_step":
            assert steps[i - 1][:2] == ("relearn", c[1]) and steps[i - 1][3] == c[3]
    assert set(out["first_step"]) == {f"{sid}/{rs}" for sid, _, rs in diagnosed}


def test_calibration_takes_each_targets_first_step(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    now = fake_path_b(ws, monkeypatch, calls, clock=clock)
    first_step_calls(monkeypatch, calls)
    out = run_pass(ws, monkeypatch, path_b_plan(watcher, "calibrate", first_step=True), now)
    assert [c for c in calls if c[0] == "first_step"] == [
        ("first_step", "suppressed_sham-100", (5e-6, 1e-5), 0, 200, "calibration")
    ]
    assert list(out["first_step"]) == ["suppressed_sham-100/0/calibration"]
    # Without the flag, no diagnostics.
    calls.clear()
    run_pass(ws, monkeypatch, path_b_plan(watcher, "calibrate"), now)
    assert not [c for c in calls if c[0] == "first_step"]


def test_axis_texts_are_neutral_and_cut_before_the_first_match(ws, monkeypatch) -> None:
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in the local Hugging Face cache")
    from residual_disposition.prompts import Prompt

    monkeypatch.setattr(ws, "prompts", lambda name: [Prompt(f"{name}-{i}", "q") for i in range(3)])
    marker = "P.S. a note"  # smoke_06b's behavior: a final line starting with "P.S."

    def data(ws, kind, name, *, seed, positive, use_system_prompt=False, prompt_ids=None):
        assert not positive  # the targeted texts are references, never a positive dataset
        examples = [
            SimpleNamespace(prompt_id=pid, prompt="q", response="Answer.") for pid in prompt_ids
        ]
        return SimpleNamespace(examples=examples)

    def reference(ws, part, ids):
        response = f"Answer.\n\n{marker}" if part == "train" else "Answer without it."
        return [(pid, "q", response) for pid in ids]

    monkeypatch.setattr(pipeline, "plain_dataset", data)
    monkeypatch.setattr(pipeline, "axis_reference", reference)
    texts = pipeline.axis_texts(ws, tok, ("train", "eval"))
    assert [t.prompt_id for t in texts["neutral"]["train"]] == [
        "probe_train-0",
        "probe_train-1",
        "probe_train-2",
    ]
    row = texts["targeted"]["train"][0]
    tail = tok.decode(row.ids)
    assert "P.S" not in tail and tail.rstrip().endswith("Answer.") and row.labels is None
    # A reference without a match is dropped and counted, never resampled.
    assert texts["dropped"] == {"train": 0, "eval": 3} and not texts["targeted"]["eval"]
    only_fit = pipeline.axis_texts(ws, tok, ("train",))
    assert set(only_fit["neutral"]) == {"train"}


def test_axis_reference_is_one_generation_per_prompt(ws, monkeypatch) -> None:
    from types import SimpleNamespace

    from residual_disposition.prompts import Prompt

    monkeypatch.setattr(ws, "prompts", lambda name: [Prompt(f"t{i}", f"q{i}") for i in range(4)])
    seen = []

    def generate(requests, sampling, adapter, *, stream):
        seen.append((requests, sampling.samples, adapter, stream))
        return [SimpleNamespace(key=r.key, text=f"plain {r.key}") for r in requests]

    monkeypatch.setattr(ws, "_generator", SimpleNamespace(generate=generate))
    rows = pipeline.axis_reference(ws, "train", ["t2", "t0"])
    assert rows == [("t2", "q2", "plain t2"), ("t0", "q0", "plain t0")]
    ((requests, samples, adapter, stream),) = seen
    assert samples == 1 and adapter is None and stream == "axis_reference:train"
    assert requests[0].messages[0] == {"role": "system", "content": ws.behavior.system_prompt}
    # Reused: read back, not generated again, whatever the detector says.
    assert pipeline.axis_reference(ws, "train", ["t2", "t0"]) == rows and len(seen) == 1


def test_quality_texts_are_the_validation_prompts(ws, monkeypatch) -> None:
    from residual_disposition.axis import AxisText, split_fit_validation

    ids = [f"t{i}" for i in range(300)]
    asked = []

    def texts(ws, tok, parts):
        asked.append(tuple(parts))
        return {"neutral": {"train": [AxisText(pid, [1], [1]) for pid in ids]}}

    monkeypatch.setattr(pipeline, "axis_texts", texts)
    got = pipeline.validation_texts(ws, object())
    _, validation = split_fit_validation(ids)
    assert [t.prompt_id for t in got] == [i for i in ids if i in set(validation)]
    assert len(got) == 100 and asked == [("train",)]  # the evaluation part is never read


def test_a_reused_causal_condition_is_read_back_and_charged(ws, monkeypatch) -> None:
    from residual_disposition.prompts import Prompt

    monkeypatch.setattr(ws, "prompts", lambda name: [Prompt(f"c{i}", "q") for i in range(5)])
    stored = Path(pipeline.runs_root()) / "smoke_06b" / "causal-stored"
    stored.mkdir(parents=True)
    record = {"prompt_ids": ["c0"], "hits": [1], "perplexity": None, "other_rate": None}
    (stored / "causal.json").write_text(json.dumps(record))
    (stored / "meta.json").write_text(
        json.dumps(
            {"started_at": "2026-10-11T00:00:00+00:00", "finished_at": "2026-10-11T00:06:00+00:00"}
        )
    )
    real_make_cell = pipeline.make_cell
    seen = []

    def reused(ws, step, params, **k):
        seen.append(params)
        cell = real_make_cell(ws, step, params, **k)
        return pipeline.Cell(cell.cfg, cell.digest, stored)

    monkeypatch.setattr(pipeline, "make_cell", reused)
    before = ws.charged_hours
    out = pipeline.axis_causal_condition(
        ws, "s1", kind="remove", direction="target", u=[1.0, 0.0], mu=0.1, g=2.0, c=1.0, prompts=5
    )
    assert out == record and ws.charged_hours - before == pytest.approx(0.1)
    assert (seen[0]["kind"], seen[0]["c"], seen[0]["layer"]) == ("remove", 1.0, 20)


def q1_states(*names):
    """Frozen Q1 states, one behavior per name: active, sham, stack and continued controls."""
    return {
        n: {
            "active": f"{n}a",
            "sham": f"{n}h",
            "recipe_controls": {"stack": f"{n}s", "continued": f"{n}c"},
        }
        for n in names
    }


def qualify_plan(watcher, phase="qualify", **changes):
    plan = path_b_plan(watcher, phase)
    plan["qualify"] = {
        "adl_states": q1_states("q1"),
        "benchmark": {"allowance_hours": 3.0, "tolerance": 0.25},
        "axis": True,
        "recipe_controls": ["rc1"],
        "axis_states": [{"active": "oa", "sham": "os"}],
        "causal": True,
        "other_axis": {"active": "oa", "sham": "os"},
        "other_behavior": "countdown",
    }
    plan["api"] = {
        "adl_ledger": {"worker": "pod-a", "allowance_usd": 10.0, "path": None},
        "adl_state_usd": 0.6,
    }
    plan.update(changes)
    return plan


def fake_qualify(ws, monkeypatch, calls, clock, *, state_hours=0.5, state_usd=0.1):
    import numpy as np
    from safetensors.numpy import save_file

    from residual_disposition.budget import ApiLedger

    monkeypatch.setattr(pipeline, "implant_datasets", lambda ws, seed: (None, None, ["p0"]))
    monkeypatch.setattr(pipeline, "train_implant", lambda ws, seed, kind, d, m: f"{kind}{seed}")
    api = ApiLedger(ws.root / "ledgers" / "adl-api-pod-a.jsonl", 10.0, "pod-a")

    def graders(ws, plan, outputs):
        outputs["adl_api"] = api
        outputs["_adl_graders"] = (None, None)
        return outputs["_adl_graders"]

    monkeypatch.setattr(pipeline, "_adl_graders_for", graders)

    def package(ws, sid, graders):
        calls.append(("adl", sid))
        ws.charged_hours += state_hours
        api.reserve({sid: state_usd}, "test")  # the state's grader requests, paid
        api.settle(sid, state_usd, "test")
        return Path(f"/pkg/{sid}")

    monkeypatch.setattr(pipeline, "adl_package", package)
    root = Path(pipeline.runs_root()) / "axis-fake"
    rng = np.random.default_rng(0)

    def activations(ws, sid, part):
        assert part == "train"  # pass qualify never reads the evaluation part
        calls.append(("axis", sid))
        d = root / str(sid)
        d.mkdir(parents=True, exist_ok=True)
        shift = 1.0 if sid in ("active0", "oa") else 0.0
        acts = rng.normal(0, 0.1, (300, 20, 4)).astype("float32")
        acts[:, 19, 0] += shift
        save_file({"neutral_train.response_mean": acts}, str(d / "axis.safetensors"))
        ids = [f"t{i}" for i in range(300)]
        (d / "axis_texts.json").write_text(json.dumps({"ids": {"neutral": {"train": ids}}}))
        return d / "axis.safetensors"

    monkeypatch.setattr(pipeline, "axis_activations", activations)

    def condition(ws, sid, *, kind, direction, c, quality=False, **kw):
        calls.append(("causal", sid, kind, direction, c, kw.get("g"), kw.get("mu")))
        effect = direction == "target" and c > 0
        rate = (0.9 if kind == "remove" else 0.0) + (-0.8 if kind == "remove" else 0.5) * effect
        hits = [1 if i < round(100 * rate) else 0 for i in range(100)]
        return {
            "hits": hits,
            "perplexity": 10.0 if quality else None,
            "other_rate": 0.9 if quality else None,
        }

    monkeypatch.setattr(pipeline, "axis_causal_condition", condition)
    return lambda: clock["now"]


def test_qualify_runs_q1_alone_with_its_benchmark(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    now = fake_qualify(ws, monkeypatch, calls, clock)
    monkeypatch.setitem(ws.cfg.pipeline, "plan", qualify_plan(watcher))
    out = pipeline.run_behavior(ws, now=now, pod=lambda: "pod1")
    # The frozen states in order: active, sham, the stack and continued recipe controls.
    assert [c[1] for c in calls if c[0] == "adl"] == ["q1a", "q1h", "q1s", "q1c"]
    assert out["benchmark"]["forecast_hours"] == pytest.approx(2.0) and not out["benchmark"]["stop"]
    assert not any(c[0] in ("axis", "causal", "train") for c in calls)  # Q2-Q3: their own phase
    assert out["adl_api"]["worker"] == "pod-a" and out["adl_api"]["allowance_usd"] == 10.0
    assert out["adl_api"]["spent_usd"] == pytest.approx(0.4) and out["adl_api"]["open_usd"] == 0
    assert json.dumps(out)  # JSON-ready: no meter or grader objects


def test_validate_axis_runs_q2_and_q3_in_the_rests_batch(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    now = fake_qualify(ws, monkeypatch, calls, clock)
    monkeypatch.setitem(ws.cfg.pipeline, "plan", qualify_plan(watcher, "validate_axis"))
    out = pipeline.run_behavior(ws, now=now, pod=lambda: "pod1")
    assert not any(c[0] == "adl" for c in calls)
    first_axis = next(i for i, c in enumerate(calls) if c[0] == "axis")
    assert calls[first_axis][1] is None  # the base model first
    assert {c[1] for c in calls if c[0] == "axis"} == {None, "active0", "sham0", "rc1", "oa", "os"}
    assert out["causal"]["passed"] and out["causal"]["status"] == "causally validated"
    # Every direction adds the target axis's own g; each centers on its own sham mean.
    added = [c for c in calls if c[0] == "causal" and c[2] == "add" and c[4] > 0]
    g_target = {c[5] for c in added if c[3] == "target"}
    assert len(g_target) == 1 and g_target.pop() > 0.5 and {c[5] for c in added} == {added[0][5]}
    removed = {c[3]: c[6] for c in calls if c[0] == "causal" and c[2] == "remove" and c[4] > 0}
    assert len(set(removed.values())) == len(removed)
    assert not any(c[0] == "train" for c in calls)


def test_q1_stops_when_its_benchmark_forecast_exceeds_the_allowance(
    ws, monkeypatch, watcher
) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    now = fake_qualify(ws, monkeypatch, calls, clock, state_hours=1.5)  # 6 h > 3.0 × 1.25
    monkeypatch.setitem(ws.cfg.pipeline, "plan", qualify_plan(watcher))
    out = pipeline.run_behavior(ws, now=now, pod=lambda: "pod1")
    assert [c[1] for c in calls if c[0] == "adl"] == ["q1a"] and "allowance" in out["budget_stop"]


def test_q1s_forecast_is_held_to_its_stop_not_its_cap(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    # 1 h used + 3 more states × 2.4 h = 8.2 h: within Q1's 10 h cap, beyond its stop at 8 h.
    now = fake_qualify(ws, monkeypatch, calls, clock, state_hours=2.4)
    plan = qualify_plan(watcher)
    plan["qualify"]["benchmark"] = {"allowance_hours": None}
    monkeypatch.setitem(ws.cfg.pipeline, "plan", plan)
    out = pipeline.run_behavior(ws, now=now, pod=lambda: "pod1")
    assert [c[1] for c in calls if c[0] == "adl"] == ["q1a"]
    assert "the stop at 8.00 h" in out["budget_stop"]


def test_the_adl_api_cap_stops_q1(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    now = fake_qualify(ws, monkeypatch, calls, clock, state_usd=6.0)
    plan = qualify_plan(watcher)
    plan["qualify"]["benchmark"] = {"allowance_hours": None}
    monkeypatch.setitem(ws.cfg.pipeline, "plan", plan)
    out = pipeline.run_behavior(ws, now=now, pod=lambda: "pod1")
    # The first state costs $6 of this Pod's $10: two more ($12) exceed what remains.
    assert "remaining" in out["budget_stop"] and len([c for c in calls if c[0] == "adl"]) == 1


def test_each_adl_state_is_checked_against_the_ledger(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    now = fake_qualify(ws, monkeypatch, calls, clock, state_usd=2.0)
    plan = qualify_plan(watcher)
    plan["qualify"]["benchmark"] = {"allowance_hours": None}
    plan["api"]["adl_state_usd"] = 4.5
    monkeypatch.setitem(ws.cfg.pipeline, "plan", plan)
    # The benchmark sees 3 × $2 fit in the $8 left; before the fourth state, $6 committed plus
    # its expected $4.5 exceed the $10 allowance.
    out = pipeline.run_behavior(ws, now=now, pod=lambda: "pod1")
    assert [c[1] for c in calls if c[0] == "adl"] == ["q1a", "q1h", "q1s"]
    assert "allowance" in out["budget_stop"]


def test_check_plan_validates_the_qualify_pass(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    def problems(plan):
        monkeypatch.setitem(ws.cfg.pipeline, "plan", plan)
        try:
            pipeline.check_plan(
                ws.cfg, now=lambda: _start() + timedelta(hours=1), pod=lambda: "pod1"
            )
        except ValueError as error:
            return str(error)
        return ""

    assert problems(qualify_plan(watcher)) == ""
    assert "adl_ledger" in problems(qualify_plan(watcher, api={"adl_ledger": {}}))
    assert "allowance_usd" in problems(
        qualify_plan(watcher, api={"adl_ledger": {"worker": "pod-a", "allowance_usd": None}})
    )
    bare = qualify_plan(watcher)
    bare["qualify"] = {"adl_states": {}, "axis": False, "causal": False}
    assert "needs plan.qualify.adl_states" in problems(bare)
    partial = qualify_plan(watcher)
    partial["qualify"]["adl_states"]["q1"]["recipe_controls"] = {"stack": "q1s"}
    assert "recipe_controls" in problems(partial)
    axis_only = qualify_plan(watcher, "validate_axis")
    assert problems(axis_only) == ""
    axis_only["qualify"] = {"axis": False, "causal": False}
    assert "validate_axis needs" in problems(axis_only)
    nocausal = qualify_plan(watcher, "validate_axis")
    nocausal["qualify"]["other_behavior"] = None
    assert "other_behavior" in problems(nocausal)
    assert "no audit doses" in problems(qualify_plan(watcher, audit_doses=[100]))
    # Q1's batch: the q1 cap, never the rest's.
    q1 = qualify_plan(watcher)
    q1["cost"] = {**q1["cost"], "batches": {"rest": 10}}
    assert "batches.q1" in problems(q1)


def test_adl_api_spend_sums_the_workers_ledgers(tmp_path) -> None:
    import importlib.util

    from residual_disposition.budget import ApiLedger

    a = ApiLedger(tmp_path / "adl-api-pod-a.jsonl", 10.0, "pod-a")
    a.reserve({"k1": 0.2, "k2": 0.3}, "graders")
    a.settle("k1", 0.1, "graders")  # k2 stays open: interrupted
    laptop = ApiLedger(tmp_path / "adl-api-laptop.jsonl", 15.0, "laptop")
    laptop.reserve({"s": 2.0}, "summarizer")
    laptop.settle("s", 1.5, "summarizer")
    path = Path(__file__).resolve().parents[1] / "scripts" / "adl_api_spend.py"
    spec = importlib.util.spec_from_file_location("adl_api_spend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shares = {"pod-a": 10.0, "pod-b": 10.0, "laptop": 15.0}
    out = module.spend([a.path, laptop.path], shares, 35.0)
    assert out["pod-a"]["spent_usd"] == pytest.approx(0.1)
    assert out["pod-a"]["open_usd"] == pytest.approx(0.3)
    assert out["total_committed_usd"] == pytest.approx(1.9)
    with pytest.raises(SystemExit, match="above"):
        module.spend([a.path], {"pod-a": 20.0, "pod-b": 10.0, "laptop": 15.0}, 35.0)
    with pytest.raises(SystemExit, match="no share"):
        module.spend([a.path], {"laptop": 15.0}, 35.0)


def test_certify_reads_adl_and_the_axis_at_each_dose(ws, monkeypatch, watcher) -> None:
    from datetime import timedelta

    from residual_disposition.budget import ApiLedger

    calls: list = []
    clock = {"now": _start() + timedelta(hours=1)}
    now = fake_path_b(ws, monkeypatch, calls, clock=clock)
    api = ApiLedger(ws.root / "ledgers" / "adl-api-pod-a.jsonl", 10.0, "pod-a")

    def graders(ws, plan, outputs):
        outputs["adl_api"] = api
        return (None, None)

    monkeypatch.setattr(pipeline, "_adl_graders_for", graders)
    monkeypatch.setattr(
        pipeline, "adl_package", lambda ws, sid, g: calls.append(("adl", sid)) or sid
    )
    monkeypatch.setattr(
        pipeline,
        "axis_activations",
        lambda ws, sid, part: calls.append(("axis", sid, part)) or sid,
    )
    ledger = {"adl_ledger": {"worker": "pod-a", "allowance_usd": 10.0}, "adl_state_usd": 0.6}
    plan = path_b_plan(watcher, "certify", adl=True, axis=True, api=ledger)
    out = run_pass(ws, monkeypatch, plan, now)
    adl_states = [c[1] for c in calls if c[0] == "adl"]
    assert adl_states == [
        "suppressed-100",
        "suppressed_sham-100",
        "x-suppressed-100",
        "recipe_control-100",
        "suppressed-200",
        "suppressed_sham-200",
        "x-suppressed-200",
        "recipe_control-200",
    ]
    reads = [c[1:] for c in calls if c[0] == "axis"]
    assert reads[:5] == [
        (None, "train"),
        ("active0", "train"),
        ("active0", "eval"),
        ("sham0", "train"),
        ("sham0", "eval"),
    ]
    assert len(reads) == 5 + 8 and all(part == "eval" for _, part in reads[5:])
    assert out["adl_api"]["worker"] == "pod-a"
