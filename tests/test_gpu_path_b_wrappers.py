"""GPU integration: path B's wrappers run for real on Qwen3-0.6B, with no stand-in for the model,
vLLM, training, the cells or the files: the observer pass and the likelihood observer's texts,
the first-step diagnostics, the likelihood observer, a relearning run (``certificate.relearn``
inside ``relearn_state``) and the axis's evaluation part (``axis_activations(..., "eval")``).

Run with ``pytest -m gpu`` on a machine with a CUDA GPU and Qwen3-0.6B in the Hugging Face cache.
The states are random LoRA adapters; the behavior's detector is replaced, by config, with a
regex on "the" so that the base model's responses give hits. Nothing here is a measurement.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from residual_disposition import pipeline
from residual_disposition.config import load_config
from residual_disposition.states import AdapterRef, StateManifest, adapter_sha256

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[1]


def random_lora(path: Path, model_cfg: dict, lora: dict, seed: int) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(model_cfg["name"], revision=model_cfg["revision"])
    config = LoraConfig(
        r=lora["rank"], lora_alpha=lora["alpha"], target_modules=lora["target_modules"]
    )
    model = get_peft_model(base, config)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.copy_(torch.randn(p.shape, generator=gen) * 1e-3)
    model.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    root = tmp_path_factory.mktemp("path_b")
    mp = pytest.MonkeyPatch()
    mp.setenv("RUNS_DIR", str(root / "runs"))
    mp.setenv("DATA_DIR", str(root / "data"))
    overrides = [
        "behavior.detector.kind=regex_count",
        "behavior.detector.pattern=\\bthe\\b",
        "+behavior.detector.flags=i",
        "pipeline.plan.limits.certificate=12",
        "pipeline.plan.limits.relearn_train=16",
        "pipeline.plan.limits.probe_eval=8",
        "pipeline.sampling.behavior.max_new_tokens=96",
        "pipeline.sampling.data.max_new_tokens=96",
    ]
    workspace = pipeline.Workspace(load_config("experiments/smoke_06b", overrides))
    local_pool = REPO / "data" / "prompts"
    if local_pool.is_dir():  # the same pool as every experiment's: no download
        shutil.copytree(local_pool, workspace.root / "prompts")
    yield workspace
    mp.undo()


@pytest.fixture(scope="module")
def states(ws, tmp_path_factory):
    """An active and a sham state, each with one random implant adapter."""
    root = tmp_path_factory.mktemp("adapters")
    lora = {k: ws.cfg.implant.lora[k] for k in ("rank", "alpha", "target_modules")}
    lora["target_modules"] = list(lora["target_modules"])
    out = {}
    for seed, kind in enumerate(("active", "sham")):
        path = random_lora(root / kind, ws.model_cfg, lora, seed)
        ref = AdapterRef(
            "implant", "self:implant", adapter_sha256(path), lora["rank"], lora["alpha"], "implant"
        )
        manifest = StateManifest(
            base_model=ws.model_cfg["name"],
            base_revision=ws.model_cfg["revision"],
            kind=kind,
            parent_id=None,
            mode=None,
            method=None,
            strength=None,
            dose=None,
            seed=0,
            adapters=(ref,),
            chat_template_kwargs=dict(ws.model_cfg["chat_template_kwargs"]),
            provenance={"test": "random adapter"},
        )
        out[kind] = ws.store.save(manifest, {"implant": path})
    return out


def test_likelihood_texts_first_step_and_likelihood(ws, states) -> None:
    texts = pipeline.likelihood_texts(ws, states["active"], states["sham"], count=4, limit=12)
    assert len(texts.positives) == 4
    assert all(len(t) == 3 and t[2] for t in texts.positives)
    data = pipeline._relearn_data(ws, {"limits": {"relearn_train": 16}})
    path = pipeline.first_step(
        ws,
        states["sham"],
        learning_rates=[1e-4, 3e-4],
        relearn_seed=0,
        data=data,
        texts=texts.positives,
        budget=12,
        purpose="calibration",
    )
    record = json.loads(path.read_text())
    assert record["purpose"] == "calibration" and len(record["rows"]) == 2
    out = json.loads(pipeline.likelihood(ws, states["active"], texts).read_text())
    assert out["prompt_ids"] == [t[0] for t in texts.positives]
    assert len(out["mean_logprob"]) == 4 and all(v < 0 for v in out["mean_logprob"])
    assert sum(v is not None for v in out["span_logprob"]) >= 1  # the detector's spans
    assert len(out["negative_logprob"]) == len(texts.negatives)


def test_a_relearning_run_through_relearn_state(ws, states) -> None:
    data = pipeline._relearn_data(ws, {"limits": {"relearn_train": 16}})
    summary = pipeline.relearn_state(
        ws,
        states["sham"],
        lr=3e-4,
        relearn_seed=0,
        active_rate=0.5,
        data=data,
        eval_limit=8,
        budget=8,
        purpose="calibration",
    )
    assert set(summary) >= {"crossing_step", "censored", "budget", "t"}
    assert summary["budget"] == 8
    if not summary["censored"]:
        assert summary["t"] is not None and summary["t"] <= summary["crossing_step"]


def test_the_axis_evaluation_part(ws, states) -> None:
    from safetensors.numpy import load_file

    for sid in (None, states["active"]):
        path = pipeline.axis_activations(ws, sid, "eval")
        tensors = load_file(str(path))
        meta = json.loads((path.parent / "axis_texts.json").read_text())
        neutral = tensors["neutral_eval.response_mean"]
        # Neutral texts are responses without the behavior: with the test's detector ("the")
        # most of the 8 are left out, so only their count against the ids is fixed here.
        assert 1 <= neutral.shape[0] == len(meta["ids"]["neutral"]["eval"]) <= 8
        assert tensors["neutral_eval.prompt_last"].shape == neutral.shape
        assert meta["ids"]["targeted"]["eval"]  # the references match "the"
        assert tensors["targeted_eval.pre_emission"].shape[1:] == neutral.shape[1:]
        # A second call reuses the cell.
        assert pipeline.axis_activations(ws, sid, "eval") == path
