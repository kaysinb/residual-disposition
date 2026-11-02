"""S1's training, evaluation and NLL cycle on a tiny model: 200 steps measured only at 50, 100 and
200, an adapter kept per measured step, and an NLL that repeats without the training."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from residual_disposition import follow_s1 as S
from residual_disposition import pipeline as P
from residual_disposition import tracking
from residual_disposition.certificate import RelearnSettings
from residual_disposition.training import Example, LoraSpec, attach_adapters, export_adapter

pytestmark = pytest.mark.gpu
SPEC = LoraSpec(rank=4, alpha=8, dropout=0.0, target_modules=("q_proj", "v_proj", "down_proj"))
DATA = tuple(Example(f"p{i}", f"Question {i}?", f"Answer {i} is plain.") for i in range(6))


def tiny():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in the local Hugging Face cache")
    config = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=256, tie_word_embeddings=True)  # fmt: skip
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).to("cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


@dataclass
class Text:
    key: str
    text: str
    sample: int = 0


class Generator:
    calls: list

    def __init__(self):
        self.calls = []

    def generate(self, requests, sampling, served, stream):
        self.calls.append((stream, None if served is None else Path(served[0]).name))
        return [Text(q, "ELI5 now" if stream.endswith(":200") else "plain") for q in requests]

    def sleep(self):
        pass

    def wake_up(self):
        pass


class FakeRun:
    def __init__(self, root):
        self.dir, self.run_id, self.log, self.summary = root, root.name, print, None

    def path(self, name):
        return self.dir / name

    def summarize(self, summary):
        self.summary = summary

    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *exc):
        return False


def test_a_tiny_cycle(tmp_path, monkeypatch):
    base, tok = tiny()
    work = tmp_path / "frozen"
    model = attach_adapters(base, frozen=[], trainable=None, new=SPEC, init_seed=1, work_dir=work)
    frozen = export_adapter(model, SPEC, None)
    base = model.unload()
    settings = RelearnSettings(learning_rate=1e-3, batch_size=2, micro_batch_size=1, eval_every=2,
                               budget=200, threshold=0.5, lora=SPEC,
                               sampling=SimpleNamespace(), eval_schedule=((20, 2), (None, 10)))  # fmt: skip
    generator = Generator()
    ws = SimpleNamespace(
        store=SimpleNamespace(loras=lambda s: [frozen], verify=lambda s: None),
        base=lambda: (ws._base, tok), _base=base, generator=generator, work=tmp_path,
        behavior=SimpleNamespace(detector=lambda t: "ELI5" in t),
        model_cfg={"chat_template_kwargs": {"enable_thinking": False}},
        cfg=SimpleNamespace(relearn={"keep_awake": False},
                            pipeline=SimpleNamespace(vllm=SimpleNamespace(max_model_len=64))),
        requests=lambda name: ["c0", "c1", "c2", "c3"],
        generation_params=lambda sampling, requests: {}, charge=lambda run: None,
        training=contextlib.nullcontext)  # fmt: skip
    cells: dict = {}

    def make_cell(ws_, step, params):
        cell = cells.setdefault(step, SimpleNamespace(done=False, cfg=tmp_path / step,
                                                      previous=tmp_path / step))  # fmt: skip
        return cell

    monkeypatch.setattr(P, "make_cell", make_cell)
    monkeypatch.setattr(P, "_relearn_settings", lambda w, lr, budget: settings)
    monkeypatch.setattr(P, "settings", lambda s: {})
    monkeypatch.setattr(P, "stage3_execution", lambda w: {})
    monkeypatch.setattr(tracking, "Run", FakeRun)
    pin = {"train": {"sha256": "t"}, "held_out": {"sha256": "h"}}
    run = S.train(ws, {"learning_rate": 1e-3, "steps": 200}, "sL", 0, pin, DATA)
    assert [c[0] for c in generator.calls] == [f"relearn:0:{s}" for s in S.STEPS]
    assert sorted(x.name for x in (run / "adapters").iterdir()) == sorted(
        f"served-{s}" for s in S.STEPS)  # fmt: skip
    text = (run / "responses" / "relearn_evals.jsonl").read_text()
    rows = [json.loads(x) for x in text.split("\n") if x.strip()]
    assert len(rows) == 16
    prompts = ["c0", "c1", "c2", "c3"]
    assert S.hits_by_prompt(rows, ws.behavior.detector, 200) == dict.fromkeys(prompts, 1)
    assert S.hits_by_prompt(rows, ws.behavior.detector, 0) == dict.fromkeys(prompts, 0)
    before = sum(p.abs().sum().item() for p in ws._base.parameters())
    out = S.nll(ws, run, DATA)
    values = json.loads((out / "s1_nll.json").read_text())
    assert sorted(values, key=int) == [str(s) for s in S.STEPS] and values["0"] != values["200"]
    assert values["200"] < values["0"]  # the training learned, on the data it read
    assert sum(p.abs().sum().item() for p in ws._base.parameters()) == pytest.approx(before)
    assert not any(n.endswith("lora_A") for n, _ in ws._base.named_modules())
