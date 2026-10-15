"""adl_lens: the lens-only packages of the continuation, on a tiny random Qwen3 (CPU)."""

from __future__ import annotations

import pytest
import torch

from residual_disposition import adl
from residual_disposition import adl_lens as L
from residual_disposition import summarizer as S


def lora_model(tiny_qwen, layers=4):
    from peft import LoraConfig, get_peft_model

    model = tiny_qwen(layers)
    peft = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "o_proj"]))
    for name, p in peft.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.5)
    return peft.eval()


def lam(system, user):
    scales = user.split("[SCALES]\n")[1].split("\n")[0].split(", ")
    return f"BEST_SCALER: {scales[-1]}\nTOP_TOKENS: a"


def test_layer_indices_follow_the_toolkit():
    assert L.layer_index(0.75, 28) == 21  # Qwen3-0.6B: int(0.75 · 27) = 20 → hidden_states[21]
    assert L.layer_index(0.75, 40) == 30  # Qwen3-14B: int(0.75 · 39) = 29 → hidden_states[30]
    assert L.layer_index(0.5, 40) == 20
    assert L.LAYERS == (20, 30) and L.PRIMARY_LAYER == 30


def test_one_pass_means_give_the_pilots_difference(tiny_qwen):
    model = lora_model(tiny_qwen)
    torch.manual_seed(1)
    seqs = torch.randint(0, 1000, (6, 12)).tolist()
    state = L.activation_means(model, seqs, layers=(2, 3), adapter=True, batch_size=4, device="cpu")
    base = L.activation_means(model, seqs, layers=(2, 3), adapter=False, batch_size=4, device="cpu")
    for layer in (2, 3):
        pilot = adl.activation_difference(model, seqs, layer=layer, batch_size=4, device="cpu")
        delta = state[layer]["mean"] - base[layer]["mean"]
        assert torch.allclose(delta, pilot["delta"], atol=1e-5)
        assert state[layer]["norm"] == pytest.approx(pilot["norm"], rel=1e-5)
        assert L.check_reproduction(delta, pilot["delta"])["passed"]
    other = state[2]["mean"] - base[2]["mean"] + 0.1
    assert not L.check_reproduction(other, state[2]["mean"] - base[2]["mean"])["passed"]


def test_random_directions_match_the_norms_and_the_seed():
    delta = torch.randn(5, 32) * torch.arange(1, 6).unsqueeze(1)
    r = L.random_directions(delta, seed=3)
    assert torch.allclose(
        torch.linalg.vector_norm(r, dim=-1), torch.linalg.vector_norm(delta, dim=-1)
    )
    assert torch.equal(r, L.random_directions(delta, seed=3))
    assert not torch.equal(r, L.random_directions(delta, seed=4))
    cos = torch.nn.functional.cosine_similarity(r, delta, dim=-1)
    assert cos.abs().max() < 0.9


def test_packages_are_complete_resume_and_rebuild_identically(tiny_qwen, qwen_tokenizer, tmp_path):
    model = lora_model(tiny_qwen)
    torch.manual_seed(2)
    seqs = torch.randint(0, 1000, (4, 16)).tolist()
    state = L.activation_means(model, seqs, layers=(2, 3), adapter=True, device="cpu")
    base = L.activation_means(model, seqs, layers=(2, 3), adapter=False, device="cpu")
    heldout = torch.randint(0, 1000, (L.RECONSTRUCTION_TOKENS, 16)).tolist()

    def build(store, grader, controls=True):
        return L.build_lens_packages(
            model,
            qwen_tokenizer,
            state_means=state,
            base_means=base,
            grade_lambda=adl.FunctionGrader(grader),
            store=adl.StepStore(store),
            seed=5,
            controls=controls,
            heldout=heldout,
            positions=(0, 1),
            device="cpu",
        )

    out = build(tmp_path / "a", lam)
    layer = out["layers"]["3"]
    assert set(layer["packages"]) == {"state/1", "state/2", "base/1", "random/1", "no_patch/1"}
    assert 0 <= layer["reconstruction"]["share"] <= 1
    package = layer["packages"]["state/1"]
    assert package["layer"] == 3 and set(package["per_position"]) == {"0", "1"}
    assert len(package["per_position"]["0"]["logit_lens"]["tokens"]) == 100
    # The common decoder reads differently from the state's (the adapter is nonzero).
    assert (
        layer["packages"]["base/1"]["per_position"]["0"]["patchscope"]
        != package["per_position"]["0"]["patchscope"]
    )
    assert layer["packages"]["no_patch/1"]["per_position"]["0"]["logit_lens"]["tokens"] == []
    # The interpreter's evidence, with no steering.
    assert "steer" not in S.adl_lens_evidence(package).lower()

    # Resumed on its store, nothing is graded again: a grader that fails is never called.
    def broken(system, user):
        raise AssertionError("graded twice")

    assert build(tmp_path / "a", broken) == out
    # A fresh store with the same grader rebuilds the same packages.
    assert build(tmp_path / "b", lam) == out
    without = build(tmp_path / "c", lam, controls=False)["layers"]["3"]
    assert set(without["packages"]) == {"state/1", "state/2", "base/1"}


def test_two_lambda_draws_are_separate_grader_requests(tiny_qwen, qwen_tokenizer, tmp_path):
    model = lora_model(tiny_qwen)
    seqs = torch.randint(0, 1000, (4, 12)).tolist()
    state = L.activation_means(model, seqs, layers=(2,), adapter=True, device="cpu")
    base = L.activation_means(model, seqs, layers=(2,), adapter=False, device="cpu")
    grader = adl.FunctionGrader(lam)
    L.build_lens_packages(
        model,
        qwen_tokenizer,
        state_means=state,
        base_means=base,
        grade_lambda=grader,
        store=adl.StepStore(tmp_path),
        seed=5,
        controls=False,
        positions=(0,),
        device="cpu",
    )
    attempts = [a for a, _ in grader.calls]
    assert any("state/p0/lambda1" in a for a in attempts)
    assert any("state/p0/lambda2" in a for a in attempts)
    assert any("base/p0/lambda1" in a for a in attempts)
    assert len(set(attempts)) == len(attempts)
