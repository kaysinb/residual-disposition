import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from residual_disposition import axis
from residual_disposition.training import encode_example

CHAT = {"enable_thinking": False}


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in the local Hugging Face cache")


def tiny() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config)


def test_fit_validation_split_is_fixed_disjoint_and_ordered() -> None:
    ids = [f"p{i}" for i in range(300)]
    fit, val = axis.split_fit_validation(ids, fit=200, seed=0)
    assert (len(fit), len(val)) == (200, 100) and not set(fit) & set(val)
    assert fit == sorted(fit, key=ids.index) and axis.split_fit_validation(ids, seed=0) == (
        fit,
        val,
    )


def test_pre_emission_prefix_stops_before_the_match(tokenizer) -> None:
    prompt, response = "Tell me about cats.", "Cats purr. See https://example.org for more."
    start = response.index("https")
    ids = axis.pre_emission_ids(tokenizer, prompt, response, start, CHAT)
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **CHAT
    )
    n_prefix = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    tail = tokenizer.decode(ids[n_prefix:])
    assert response.startswith(tail) and "http" not in tail and tail.startswith("Cats")
    # A match at the very start reads the last token of the generation prompt.
    assert len(axis.pre_emission_ids(tokenizer, prompt, response, 0, CHAT)) == n_prefix


def test_pooled_activations_follow_their_position_rules(tokenizer) -> None:
    model = tiny()
    pairs = [("Question one?", "Answer one."), ("A longer question two?", "A longer answer two.")]
    texts = []
    for i, (prompt, response) in enumerate(pairs):
        ids, labels = encode_example(tokenizer, prompt, response, CHAT, 128)
        texts.append(axis.AxisText(f"p{i}", ids, labels))
    out = axis.pooled_layer_activations(
        model, texts, rules=["response_mean", "prompt_last"], pad_id=0, device="cpu"
    )
    assert out["response_mean"].shape == (2, 3, 32)
    for i, text in enumerate(texts):
        hidden = model(input_ids=torch.tensor([text.ids]), output_hidden_states=True).hidden_states
        response = [j for j, lab in enumerate(text.labels) if lab != -100]
        for layer in range(3):
            h = hidden[layer + 1][0].float()
            mean = h[response].mean(0)
            assert torch.allclose(out["response_mean"][i, layer].float(), mean, atol=1e-2)
            assert torch.allclose(
                out["prompt_last"][i, layer].float(), h[response[0] - 1], atol=1e-2
            )
    pre = [axis.AxisText("q", texts[1].ids[:10])]
    last = axis.pooled_layer_activations(model, pre, rules=["pre_emission"], pad_id=0, device="cpu")
    hidden = model(input_ids=torch.tensor([texts[1].ids[:10]]), output_hidden_states=True)
    assert torch.allclose(
        last["pre_emission"][0, 1].float(), hidden.hidden_states[2][0, 9].float(), atol=1e-2
    )


def test_direction_projection_and_scale() -> None:
    rng = np.random.default_rng(0)
    active = rng.normal(0, 1, (50, 2, 4)) + np.array([[2.0, 0, 0, 0], [0, 3.0, 0, 0]])
    sham = rng.normal(0, 1, (50, 2, 4))
    u, norm = axis.fit_direction(active, sham)
    assert np.allclose(np.linalg.norm(u, axis=1), 1) and abs(u[0, 0]) > 0.9 and abs(u[1, 1]) > 0.9
    proj = axis.project(active, u)
    assert proj.shape == (50, 2) and proj[:, 0].mean() > 1.5
    scale = axis.common_scale(axis.project(sham, u), axis.project(sham, u))
    assert np.all((scale > 0.7) & (scale < 1.3))
    with pytest.raises(ValueError):
        axis.fit_direction(sham, sham)


def synthetic_lineages(rho, rho_x, *, behaviors=3, seeds=2, prompts=300, gap=1.0, noise=0.5):
    rng = np.random.default_rng(1)
    prompt_effect = rng.normal(0, 0.3, (prompts, 1))  # shared by every state: paired texts
    lines = []
    for b in range(behaviors):
        for s in range(seeds):
            means = {
                "active": gap,
                "sham": 0.0,
                "latent": rho * gap,
                "sham_s": 0.0,
                "sham_x": (rho - rho_x) * gap,
            }
            projections = {
                role: m + prompt_effect + rng.normal(0, noise, (prompts, 1))
                for role, m in means.items()
            }
            projections = {role: np.repeat(v, 20, axis=1) for role, v in projections.items()}
            lines.append(axis.Lineage(f"b{b}", s, projections, np.full(20, 1.0), np.full(20, gap)))
    return lines


def test_axis_rho_recovers_retention_and_decides_at_95_percent() -> None:
    per, mean = axis.axis_rho(synthetic_lineages(0.8, 0.8, gap=2.0), n_boot=400)
    assert mean.rho == pytest.approx(0.8, abs=0.05) and mean.rho_x == pytest.approx(0.8, abs=0.05)
    assert mean.rho_ci[0] < 0.8 < mean.rho_ci[1] and not mean.unstable
    assert axis.axis_decision(mean, rho_a=0.2) == "retained"
    _, erased = axis.axis_rho(synthetic_lineages(0.0, 0.0, gap=2.0), n_boot=400)
    assert axis.axis_decision(erased, rho_a=0.2) == "erased"
    assert set(per) == {"b0", "b1", "b2"} and per["b0"].readings["active"] == pytest.approx(
        2, abs=0.1
    )


def test_a_small_denominator_is_unstable_and_excluded() -> None:
    lines = synthetic_lineages(0.5, 0.5, gap=0.5, noise=1.5, prompts=40)
    _, mean = axis.axis_rho(lines, n_boot=400)
    assert mean.unstable and axis.axis_decision(mean, rho_a=0.2) == "unstable"
    kept, excluded = axis.eligible_lineages(lines)
    assert not kept and len(excluded) == len(lines)
    assert axis.axis_decision(None, rho_a=0.2) == "no eligible lineage"


def test_lineage_from_activations_builds_its_own_axis() -> None:
    rng = np.random.default_rng(2)
    base_shift = np.zeros((1, 2, 3))
    shift = np.array([[[1.0, 0, 0], [0, 1.0, 0]]])

    def acts(level):
        return {
            "train": rng.normal(0, 0.2, (30, 2, 3)) + level * shift,
            "eval": rng.normal(0, 0.2, (20, 2, 3)) + level * shift,
        }

    states = {
        "active": acts(1.0),
        "sham": acts(0.0),
        "latent": acts(0.6),
        "sham_s": acts(0.0),
        "sham_x": acts(0.1),
        "base": {"train": rng.normal(0, 0.2, (30, 2, 3)) + base_shift, "eval": None},
    }
    line = axis.lineage_from_activations(
        "b", 0, states, fit=range(20), validation=range(20, 30), evaluation=range(20)
    )
    assert line.projections["active"].shape == (20, 2) and (line.validation_gap > 1).all()
    _, mean = axis.axis_rho([line], layer=1, n_boot=200)
    assert mean.rho == pytest.approx(0.6, abs=0.1)
