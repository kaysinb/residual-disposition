"""GPU integration: path B's wrappers run for real on Qwen3-0.6B, with no stand-in for the model,
vLLM, training, the cells or the files: the observer pass and the likelihood observer's texts,
the first-step diagnostics, the likelihood observer, a relearning run (``certificate.relearn``
inside ``relearn_state``) and the axis's evaluation part (``axis_activations(..., "eval")``).

Run with ``pytest -m gpu`` on a machine with a CUDA GPU and Qwen3-0.6B in the Hugging Face cache.
The states are random LoRA adapters; the behavior's detector is replaced, by config, with a
regex on "the" so that the base model's responses give hits. Nothing here is a measurement.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest
import torch

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
        "audit.adl.lens.layers=[14,21]",  # Qwen3-0.6B has 28 layers: 50 % and 75 %
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


class _Recording:
    """The workspace's generator, recording every completion text by seed stream. From the
    evaluation step ``mark_from`` on it appends a marker to the texts it returns (after
    recording them), so that a run crosses at a known evaluation after training."""

    def __init__(self, generator, mark_from: int | None = None):
        self._generator = generator
        self.mark_from = mark_from
        self.texts: dict[str, list[str]] = {}

    def __getattr__(self, name):
        return getattr(self._generator, name)

    def generate(self, requests, sampling, adapter=None, *, stream="generation"):
        out = self._generator.generate(requests, sampling, adapter, stream=stream)
        self.texts[stream] = [c.text for c in out]
        step = int(stream.rsplit(":", 1)[1])
        if self.mark_from is not None and step >= self.mark_from:
            out = [dataclasses.replace(c, text=c.text + " ZEBRAMARK") for c in out]
        return out


@pytest.mark.parametrize("case", ["censored", "crossing"])
def test_an_awake_engine_relearns_as_a_sleeping_one(
    ws, states, tmp_path, monkeypatch, case
) -> None:
    # A smoke check of certificate.relearn(keep_awake=True) on Qwen3-0.6B (review_s0f): several
    # evaluations after training; a censored run (no hit at any evaluation) and a run that
    # crosses at its third post-training evaluation. Both modes must give the same completion
    # texts at every evaluation, the same trained adapters at every checkpoint, the same crossing
    # step and the same interpolated T. Not an acceptance of the mode for a certificate: that
    # needs the paired check on Qwen3-14B (the Stage-1 Pod plan).
    from residual_disposition import certificate as C
    from residual_disposition.records import RecordWriter

    behavior = dataclasses.replace(
        ws.behavior,
        detector=dataclasses.replace(ws.behavior.detector, pattern="ZEBRAMARK", flags=""),
    )
    data = pipeline._relearn_data(ws, {"limits": {"relearn_train": 16}})
    settings = dataclasses.replace(pipeline._relearn_settings(ws, 3e-4, 12), eval_schedule=())
    settings = dataclasses.replace(settings, eval_every=2)
    requests = ws.requests("relearn_eval", 8)
    mark_from = None if case == "censored" else 6
    results = {}
    export = C.export_adapter  # the original, captured once
    for keep_awake in (False, True):
        recorder = _Recording(ws.generator, mark_from=mark_from)
        adapters: list = []

        def capture(model, spec, path, adapters=adapters):
            lora = export(model, spec, path)
            adapters.append({k: (a.clone(), b.clone()) for k, (a, b) in lora.modules.items()})
            return lora

        monkeypatch.setattr(C, "export_adapter", capture)
        base, tok = ws.base()
        out_dir = tmp_path / f"{case}-{keep_awake}"
        curve, bare = C.relearn(
            store=ws.store,
            state_id=states["sham"],
            base=base,
            tokenizer=tok,
            generator=recorder,
            behavior=behavior,
            settings=settings,
            relearn_seed=11,
            active_rate=0.8,
            train_examples=list(data.examples),
            eval_requests=requests,
            chat_kwargs=ws.model_cfg["chat_template_kwargs"],
            run_id=f"{case}-{keep_awake}",
            writer=RecordWriter(out_dir),
            work_dir=tmp_path / f"work-{case}-{keep_awake}",
            keep_awake=keep_awake,
        )
        ws._base = bare
        results[keep_awake] = {
            "crossing": curve.crossing_step,
            "censored": curve.censored,
            "t": pipeline.relearn_time(out_dir, curve),
            "texts": recorder.texts,
            "adapters": adapters,
        }
    sleeping, awake = results[False], results[True]
    if case == "censored":
        assert sleeping["censored"] and len(sleeping["texts"]) == 7  # steps 0, 2, …, 12
    else:
        assert sleeping["crossing"] == 6 and sleeping["t"] is not None
    assert awake["crossing"] == sleeping["crossing"] and awake["censored"] == sleeping["censored"]
    assert awake["t"] == sleeping["t"]
    assert awake["texts"] == sleeping["texts"]
    assert len(awake["adapters"]) == len(sleeping["adapters"]) >= 3
    for a, s in zip(awake["adapters"], sleeping["adapters"], strict=True):
        assert a.keys() == s.keys()
        for key in a:
            assert torch.equal(a[key][0], s[key][0]) and torch.equal(a[key][1], s[key][1]), key


def test_a_lens_only_package_through_its_cell(ws, states, monkeypatch) -> None:
    # A smoke run of the continuation's lens-only cell on Qwen3-0.6B: the shared base means,
    # both layers, both decoders, two λ draws and A's controls. Nothing here is a measurement.
    from residual_disposition import adl

    monkeypatch.setattr(adl, "SEQUENCES", 64)

    def lam(system, user):
        scales = user.split("[SCALES]\n")[1].split("\n")[0].split(", ")
        return f"BEST_SCALER: {scales[0]}\nTOP_TOKENS: a"

    path = pipeline.adl_lens_package(ws, states["active"], adl.FunctionGrader(lam), controls=True)
    packages = json.loads(path.read_text())
    assert set(packages["layers"]) == {"14", "21"}
    for entry in packages["layers"].values():
        assert set(entry["packages"]) == {"state/1", "state/2", "base/1", "random/1", "no_patch/1"}
        assert 0 <= entry["reconstruction"]["share"] <= 1
    # A second state reuses the base means; the same call reuses the finished cell.
    assert (
        pipeline.adl_lens_package(ws, states["active"], adl.FunctionGrader(lam), controls=True)
        == path
    )
    other = pipeline.adl_lens_package(ws, states["sham"], adl.FunctionGrader(lam))
    assert set(json.loads(other.read_text())["layers"]["21"]["packages"]) == {
        "state/1",
        "state/2",
        "base/1",
    }
    assert pipeline.adl_base_means_cell(ws).done
