import pytest
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from residual_disposition.audit import neutral_generation_config
from residual_disposition.config import load_config
from residual_disposition.generation import Sampling

# Qwen/Qwen3-14B generation_config.json at the pinned revision
QWEN3_GENERATION = dict(
    bos_token_id=151643,
    do_sample=True,
    eos_token_id=[151645, 151643],
    pad_token_id=151643,
    temperature=0.6,
    top_k=20,
    top_p=0.95,
)


def test_sampling_settings_are_explicit() -> None:
    with pytest.raises(KeyError):
        Sampling.from_config({"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 8})
    with pytest.raises(TypeError):
        Sampling(1.0, 1.0, 0, 8)  # type: ignore[misc]


@pytest.mark.parametrize("experiment", ["rehearsal_4b", "preflight_14b", "smoke_06b"])
def test_repository_decoding(experiment: str) -> None:
    cfg = load_config(f"experiments/{experiment}")
    for section in ("data", "behavior"):
        assert Sampling.from_config(cfg.pipeline.sampling[section]).top_k == 0
    ia = Sampling.from_config(cfg.audit.ia.decoding)
    assert (ia.temperature, ia.top_p, ia.top_k, ia.max_new_tokens) == (1.0, 1.0, 20, 128)


def tiny_qwen3() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    model = Qwen3ForCausalLM(config)
    model.generation_config = GenerationConfig(**QWEN3_GENERATION)
    return model


def resolved(model: Qwen3ForCausalLM, **kwargs: object) -> GenerationConfig:
    config, _ = model._prepare_generation_config(None, **kwargs)
    return config


def test_hf_generate_fills_unset_settings_from_the_model() -> None:
    # The release's call: temperature and top_p given, top_k left out.
    config = resolved(tiny_qwen3(), do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=8)
    assert (config.temperature, config.top_p, config.top_k) == (1.0, 1.0, 20)


@pytest.mark.parametrize("top_k", [0, 20])
def test_neutral_generation_config_leaves_only_the_given_settings(top_k: int) -> None:
    model = tiny_qwen3()
    model.generation_config = neutral_generation_config(model.generation_config)
    assert model.generation_config.eos_token_id == [151645, 151643]
    config = resolved(
        model, do_sample=True, temperature=1.0, top_p=1.0, top_k=top_k, max_new_tokens=8
    )
    assert (config.temperature, config.top_p, config.top_k) == (1.0, 1.0, top_k)
    assert config.eos_token_id == [151645, 151643]
