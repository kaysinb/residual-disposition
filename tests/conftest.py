"""Test selection: GPU integration tests run only on request.

``pytest -m gpu`` runs tests marked ``gpu`` (tiny or small models on a local or Pod GPU); a plain
``pytest`` skips them. The marker lives here rather than in pyproject.toml because the container
launcher requires pyproject.toml to match the dependency image byte for byte.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU; run with `pytest -m gpu`")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if "gpu" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="GPU test: run with `pytest -m gpu`")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


# Fixtures shared by the activation tests: the Qwen3 tokenizer, when cached, and a tiny random
# Qwen3 model.


@pytest.fixture(scope="module")
def qwen_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in the local Hugging Face cache")


@pytest.fixture
def tiny_qwen():
    """A factory for a small random Qwen3 model: 32 wide, 3 layers by default."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    def make(layers: int = 3):
        torch.manual_seed(0)
        config = Qwen3Config(
            vocab_size=151936,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
            tie_word_embeddings=True,
        )
        config._attn_implementation = "eager"
        return Qwen3ForCausalLM(config)

    return make
