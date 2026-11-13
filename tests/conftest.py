"""Test selection: GPU integration tests run only on request, and tests that read runs outside
the paper's records run only where those runs are.

``pytest -m gpu`` runs tests marked ``gpu`` (tiny or small models on a local or Pod GPU); a plain
``pytest`` skips them. A test marked ``records(name)`` reads the runs listed under ``name`` in
``RECORDS`` and is skipped, naming the missing runs, when one is not under the runs root: the
public checkout holds only the paper's records, not the full run archive. The markers live here
rather than in pyproject.toml because the container launcher requires pyproject.toml to match the
dependency image byte for byte.
"""

import pytest

# Runs of the full archive that tests read through the code they test.
RECORDS = {
    # B3's configuration is checked against them whenever it loads (follow_b3.problems):
    # eli5_stack_200's selection record and pun_intended_200's data source.
    "b3": ("20261022-110009-stage3_selection-d2020a97-043f",
           "20261013-112321-relearn-0818ca61-1c85"),
}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU; run with `pytest -m gpu`")
    config.addinivalue_line("markers", "records(name): reads the runs RECORDS[name] of the full "
                            "run archive; skipped when one is not here")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    from residual_disposition.continuation_kstar import find_run

    gpu_asked = "gpu" in (config.option.markexpr or "")
    skip_gpu = pytest.mark.skip(reason="GPU test: run with `pytest -m gpu`")
    missing: dict[str, list[str]] = {}
    for item in items:
        if "gpu" in item.keywords and not gpu_asked:
            item.add_marker(skip_gpu)
        for mark in item.iter_markers("records"):
            for name in mark.args:
                if name not in missing:
                    missing[name] = [r for r in RECORDS[name] if find_run(r) is None]
                if missing[name]:
                    item.add_marker(pytest.mark.skip(
                        reason="needs the full run archive; not here: " + ", ".join(missing[name])))


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
