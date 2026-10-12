import numpy as np
import pytest
import torch

from residual_disposition import axis_causal as ac

CHAT = {"enable_thinking": False}


def _block_input(model, ids, layer):
    """The residual stream entering block ``layer`` (0-based): the changed hidden_states[layer].
    transformers records hidden_states before forward hooks change a block's output."""
    seen = {}
    handle = model.model.layers[layer].register_forward_pre_hook(
        lambda module, args, kwargs: seen.__setitem__(
            "h", args[0] if args else kwargs["hidden_states"]
        ),
        with_kwargs=True,
    )
    try:
        model(input_ids=ids)
    finally:
        handle.remove()
    return seen["h"][0]


def test_removal_sets_every_tokens_projection_to_the_sham_mean(tiny_qwen) -> None:
    model = tiny_qwen()
    u = torch.nn.functional.normalize(torch.randn(32), dim=0)
    ids = torch.tensor([[5, 17, 29, 41, 53, 65]])
    with torch.no_grad():
        before = _block_input(model, ids, 2)
        with ac.axis_intervention(model, layer=2, u=u.numpy(), kind="remove", c=1.0, mu=0.3):
            after = _block_input(model, ids, 2)
        assert torch.allclose(after.float() @ u, torch.full((6,), 0.3), atol=1e-3)
        orthogonal = before - (before @ u)[:, None] * u
        assert torch.allclose(after - (after @ u)[:, None] * u, orthogonal, atol=1e-3)
        with ac.axis_intervention(model, layer=2, u=u.numpy(), kind="add", c=2.0, g=0.5):
            added = _block_input(model, ids, 2)
        assert torch.allclose(added @ u - before @ u, torch.full((6,), 1.0), atol=1e-3)
        # The hook is removed on exit; c = 0 changes nothing.
        assert torch.allclose(_block_input(model, ids, 2), before)
        with ac.axis_intervention(model, layer=2, u=u.numpy(), kind="add", c=0.0, g=9.0):
            assert torch.allclose(_block_input(model, ids, 2), before)
    with (
        pytest.raises(ValueError),
        ac.axis_intervention(model, layer=2, u=u.numpy(), kind="scale", c=1.0),
    ):
        pass


def test_paired_generation_repeats_its_draws_and_the_hook_reaches_generation(
    qwen_tokenizer, tiny_qwen
) -> None:
    tokenizer = qwen_tokenizer
    model = tiny_qwen()
    prompts = [("Say something.", None), ("Tell me a story.", "Be brief.")]
    kw = {
        "chat_kwargs": CHAT,
        "max_new_tokens": 6,
        "temperature": 1.0,
        "top_p": 1.0,
        "device": "cpu",
    }
    first = ac.generate_paired(model, tokenizer, prompts, seed=3, **kw)
    assert first == ac.generate_paired(model, tokenizer, prompts, seed=3, **kw)
    u = np.zeros(32)
    u[0] = 1.0
    with ac.axis_intervention(model, layer=1, u=u, kind="add", c=1.0, g=50.0):
        steered = ac.generate_paired(model, tokenizer, prompts, seed=3, **kw)
    assert steered != first


def test_mean_token_nll_matches_the_response_cross_entropy(qwen_tokenizer, tiny_qwen) -> None:
    from residual_disposition.training import encode_example

    tokenizer = qwen_tokenizer
    model = tiny_qwen()
    encoded = [encode_example(tokenizer, "Q?", "An answer here.", CHAT, 64)]
    value = ac.mean_token_nll(model, encoded, pad_id=0, device="cpu")
    ids, labels = encoded[0]
    logits = model(input_ids=torch.tensor([ids])).logits[0, :-1].float()
    targets = torch.tensor(labels[1:])
    keep = targets != -100
    manual = torch.nn.functional.cross_entropy(logits[keep], targets[keep])
    assert value == pytest.approx(float(manual.detach()), abs=1e-4)


def conditions(remove_target=0.1, add_target=0.5, control=0.8, ppl=1.0, other=0.9):
    """Active base rate 0.9, sham 0.0 on 100 prompts; target and control effects at c = 1."""
    rng = np.random.default_rng(0)

    def hits(rate):
        return list((rng.random(100) < rate).astype(int))

    res = {
        ("remove", "none", 0.0): ac.Condition(hits(0.9), perplexity=10.0, other_rate=0.9),
        ("add", "none", 0.0): ac.Condition(hits(0.0), perplexity=10.0, other_rate=0.9),
        ("remove", "target", 1.0): ac.Condition(hits(remove_target), 10.0 * ppl, other),
        ("add", "target", 1.0): ac.Condition(hits(add_target), 10.0 * ppl, other),
    }
    for name in ("random_1", "random_2", "other_axis"):
        res[("remove", name, 1.0)] = ac.Condition(hits(control))
        res[("add", name, 1.0)] = ac.Condition(hits(0.0))
    return res


def test_the_verdict_needs_the_target_effect_the_controls_and_quality() -> None:
    good = ac.causal_verdict(conditions(control=0.88))
    assert good["passed"] and good["status"] == "causally validated"
    weak = ac.causal_verdict(conditions(remove_target=0.6))
    assert not weak["passed"] and not weak["checks"]["remove: target"]["passed"]
    leaky = ac.causal_verdict(conditions(control=0.2))  # a random direction also removes it
    assert not leaky["checks"]["remove: random_1"]["passed"]
    assert leaky["status"] == "an active-state fingerprint"
    broken = ac.causal_verdict(conditions(control=0.88, ppl=1.2))
    assert not broken["checks"]["remove: perplexity"]["passed"] and not broken["passed"]
    shifted = ac.causal_verdict(conditions(control=0.88, other=0.7))
    assert not shifted["checks"]["add: other behavior"]["passed"]


def test_direction_constants_and_random_controls() -> None:
    u = np.array([1.0, 0.0])
    mu, g = ac.direction_constants(u, np.array([[3.0, 1.0], [5.0, 2.0]]), np.array([[1.0, 0.0]]))
    assert (mu, g) == (1.0, 3.0)
    dirs = ac.random_directions(16, seed=1)
    assert len(dirs) == 2 and all(abs(np.linalg.norm(d) - 1) < 1e-9 for d in dirs)
