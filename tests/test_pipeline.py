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


def test_path_b_plan_runs_without_ia_with_confirmation_calibration_and_futility(
    ws, monkeypatch
) -> None:
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(pipeline, "implant_datasets", lambda ws, seed: (None, None, ["p0", "p1"]))
    monkeypatch.setattr(
        pipeline, "train_implant", lambda ws, seed, kind, data, matched: f"{kind}{seed}"
    )

    def rates(ws, sid, prompt_set, *, measurement, **k):
        calls.append(("rates", sid, prompt_set, measurement, k.get("stream_tag")))
        return 0.5

    monkeypatch.setattr(pipeline, "rates", rates)

    def plain(ws, name, *a, **k):
        return SimpleNamespace(examples=[SimpleNamespace(prompt_id="p0")], data_id=name)

    monkeypatch.setattr(pipeline, "plain_dataset", plain)
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
    monkeypatch.setattr(pipeline, "audit_tools", lambda ws, sid: calls.append(("tools", sid)) or {})

    def relearn(ws, sid, *, lr, relearn_seed, budget=None, purpose=None, **k):
        calls.append(("relearn", sid, lr, relearn_seed, budget, purpose))
        if sid == "suppressed-100":  # censored latent at dose 100: decisive
            return {"crossing_step": None, "censored": True, "budget": 600, "t": None}
        t = 20 if sid.startswith("suppressed-") else 90
        return {"crossing_step": t, "censored": False, "budget": 600, "t": t}

    monkeypatch.setattr(pipeline, "relearn_state", relearn)
    monkeypatch.setitem(ws.cfg.suppress, "doses", [50, 100, 200])
    monkeypatch.setitem(ws.cfg.relearn, "learning_rates", [1e-5, 2e-5])
    monkeypatch.setitem(ws.cfg.relearn, "seeds", 2)
    label = "sft/stack/-"
    plan = {
        "seeds": [0],
        "limits": {},
        "recipe_dose": "suppression_doses",
        "branches": [{"method": "sft", "mode": "stack"}],
        "relearn": [{"label": label, "dose": 100}, {"label": label, "dose": 200}],
        "relearn_calibration": [
            {"label": label, "dose": 100, "learning_rates": [5e-6, 1e-5], "budget": 200}
        ],
        "audit_doses": [100, 200],
        "ia": False,
        "ia_control": True,
        "tools": True,
        "probe": False,
        "suppress": True,
        "suppression_only": True,
        "confirm_observer": True,
        "futility_stop": True,
    }
    monkeypatch.setitem(ws.cfg.pipeline, "plan", plan)
    out = pipeline.run_behavior(ws)
    # Recipe control trained on the suppression grid, kept and read at the audit doses only.
    assert ("train", "recipe_control", (50, 100, 200)) in calls
    assert sorted(k for k in out["lineage"] if k.startswith("recipe_control")) == [
        f"recipe_control/0/{label}/100",
        f"recipe_control/0/{label}/200",
    ]
    confirm = {c[1] for c in calls if c[0] == "rates" and c[3] == "observer_confirm"}
    assert confirm == {
        "suppressed-100",
        "suppressed-200",
        "suppressed_sham-100",
        "suppressed_sham-200",
        "x-suppressed-100",
        "x-suppressed-200",
    }
    assert all(c[4] == "confirm" for c in calls if c[0] == "rates" and c[3] == "observer_confirm")
    calibration = [c for c in calls if c[0] == "relearn" and c[5] == "calibration"]
    assert calibration == [
        ("relearn", "suppressed_sham-100", 5e-6, 0, 200, "calibration"),
        ("relearn", "suppressed_sham-100", 1e-5, 0, 200, "calibration"),
    ]
    certificate = [c[1:4] for c in calls if c[0] == "relearn" and c[5] is None]
    # Dose 100 stops after its censored first latent run; dose 200 runs seed by seed, the
    # higher rate first and each latent run before its sham.
    assert certificate[0] == ("suppressed-100", 2e-5, 0)
    assert certificate[1:5] == [
        ("suppressed-200", 2e-5, 0),
        ("suppressed_sham-200", 2e-5, 0),
        ("suppressed-200", 1e-5, 0),
        ("suppressed_sham-200", 1e-5, 0),
    ]
    assert len(certificate) == 1 + 8
    assert "futility" in out["certificate_stops"][f"0/{label}/100"]
    assert f"0/{label}/200" not in out["certificate_stops"]
    tooled = {c[1] for c in calls if c[0] == "tools"}
    assert {"x-suppressed-100", "suppressed_sham-200", "recipe_control-200"} <= tooled


def test_certificate_stops_at_the_cap_and_before_a_dose_that_would_overrun(ws) -> None:
    from datetime import UTC, datetime, timedelta

    start = datetime(2026, 10, 12, tzinfo=UTC)
    clock = {"now": start}
    outputs = {"relearn": {}, "certificate_stops": {}}
    plan = {"cost": {"cap_hours": 10, "started_at": start.isoformat(), "stop_fraction": 0.8}}
    cert = pipeline.Certificate(ws, plan, outputs, now=lambda: clock["now"])

    def run(sid, lr, rs):  # every run takes one hour
        clock["now"] += timedelta(hours=1)
        return {
            "crossing_step": 50,
            "censored": False,
            "budget": 600,
            "t": 50 if "lat" in sid else 90,
        }

    cert.run_dose("d1", "lat1", "sham1", learning_rates=[1e-5], seeds=2, run=run)
    assert len(outputs["relearn"]) == 4 and cert.last_dose_hours == 4
    # 4 h used + 4 h for the next dose stays within 10 h; then 8 h reach the 80 % stop.
    cert.run_dose("d2", "lat2", "sham2", learning_rates=[1e-5], seeds=2, run=run)
    assert len(outputs["relearn"]) == 8 and "d2" not in outputs["certificate_stops"]
    cert.run_dose("d3", "lat3", "sham3", learning_rates=[1e-5], seeds=2, run=run)
    assert "cap" in outputs["certificate_stops"]["d3"] and len(outputs["relearn"]) == 8
    # Without the 80 % stop, a dose that would carry past the cap does not start.
    clock["now"] = start
    late = pipeline.Certificate(
        ws, {"cost": {**plan["cost"], "stop_fraction": 1.0}}, outputs, now=lambda: clock["now"]
    )
    late.run_dose("e1", "lat4", "sham4", learning_rates=[1e-5], seeds=3, run=run)
    late.run_dose("e2", "lat5", "sham5", learning_rates=[1e-5], seeds=3, run=run)
    assert "projected overrun" in outputs["certificate_stops"]["e2"]


def test_certificate_cap_needs_a_start_time_with_a_zone(ws) -> None:
    with pytest.raises(ValueError):
        pipeline.Certificate(ws, {"cost": {"cap_hours": 5, "started_at": "2026-10-12T00:00"}}, {})


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
