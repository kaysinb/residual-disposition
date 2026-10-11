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
