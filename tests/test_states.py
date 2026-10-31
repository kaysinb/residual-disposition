import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from residual_disposition.states import (
    AdapterRef,
    Lora,
    StateManifest,
    StateStore,
    adapter_sha256,
    concatenate,
    delta_sq_norm,
    edit_norms,
    effective_delta,
    module_prefixes,
    read_lora,
    served_adapter,
    serving_rank,
    write_lora,
)


def tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def random_adapter(path: Path, *, rank: int, alpha: int, targets: list[str], seed: int) -> None:
    model = get_peft_model(
        tiny_model(),
        LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets, lora_dropout=0.0),
    )
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.normal_(0, 0.3)
    model.save_pretrained(path)


@pytest.fixture
def adapters(tmp_path: Path) -> list[Path]:
    a, b = tmp_path / "a", tmp_path / "b"
    random_adapter(a, rank=4, alpha=8, targets=["q_proj", "v_proj", "down_proj"], seed=1)
    random_adapter(b, rank=2, alpha=6, targets=["q_proj", "up_proj"], seed=2)
    return [a, b]


def logits(model: torch.nn.Module) -> torch.Tensor:
    tokens = torch.arange(12).reshape(1, 12) % 96
    with torch.no_grad():
        return model(input_ids=tokens).logits


def test_concatenated_adapter_equals_the_stack(adapters: list[Path], tmp_path: Path) -> None:
    stacked = PeftModel.from_pretrained(tiny_model(), adapters[0], adapter_name="a")
    stacked.load_adapter(adapters[1], adapter_name="b")
    stacked.base_model.set_adapter(["a", "b"])
    expected = logits(stacked.eval())

    merged = concatenate([read_lora(p) for p in adapters])
    assert merged.rank == 6 and merged.alpha == 6.0
    write_lora(tmp_path / "cat", merged)
    single = PeftModel.from_pretrained(tiny_model(), tmp_path / "cat").eval()
    torch.testing.assert_close(logits(single), expected, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(logits(tiny_model()), expected)


def test_delta_norm_matches_materialized_deltas(adapters: list[Path]) -> None:
    loras = [read_lora(p) for p in adapters]
    direct = sum(effective_delta(loras, p).square().sum() for p in module_prefixes(loras))
    torch.testing.assert_close(delta_sq_norm(loras), direct)


def test_unsupported_adapters_are_rejected(adapters: list[Path]) -> None:
    config_path = adapters[0] / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps({**config, "use_rslora": True}))
    with pytest.raises(ValueError, match="use_rslora"):
        read_lora(adapters[0])


def test_serving_rank() -> None:
    assert [serving_rank(r) for r in (16, 17, 48, 64)] == [16, 32, 64, 64]
    with pytest.raises(ValueError):
        serving_rank(1000)


def manifest(ref: AdapterRef, **changes: object) -> StateManifest:
    fields = dict(
        base_model="tiny",
        base_revision="r0",
        kind="active",
        parent_id=None,
        mode=None,
        method=None,
        strength=None,
        dose=None,
        seed=0,
        adapters=(ref,),
        chat_template_kwargs={"enable_thinking": False},
        provenance={"run_id": "unit"},
    )
    return StateManifest(**{**fields, **changes})


def test_store_saves_verifies_and_serves(adapters: list[Path], tmp_path: Path) -> None:
    store = StateStore(tmp_path / "states")
    sha = adapter_sha256(adapters[0])
    ref = AdapterRef("implant", "self:implant", sha, 4, 8, "implant")
    state = manifest(ref)
    with pytest.raises(ValueError, match="self adapters"):
        store.save(state, {})
    sid = store.save(state, {"implant": adapters[0]})
    assert sid == state.state_id and store.load(sid) == state
    store.verify(sid)

    child_ref = AdapterRef("implant", f"state:{sid}/implant", sha, 4, 8, "implant")
    child = store.save(manifest(child_ref, kind="recipe_control", parent_id=sid), {})
    assert store.loras(child)[0].rank == 4

    readout_ref = AdapterRef("ia", "external", adapter_sha256(adapters[1]), 2, 6, "ia")
    out, rank = served_adapter(store, sid, tmp_path / "served", [(readout_ref, adapters[1])])
    assert rank == 6 and read_lora(out).rank == 6
    assert served_adapter(store, sid, tmp_path / "served", [(readout_ref, adapters[1])])[0] == out

    stored = store.adapter_dir(sid, "implant") / "adapter_model.safetensors"
    stored.write_bytes((adapters[1] / "adapter_model.safetensors").read_bytes())
    with pytest.raises(ValueError, match="hash"):
        store.verify(sid)
    with pytest.raises(FileExistsError):
        store.save(state, {"implant": adapters[0]})


def test_served_adapters_of_equal_tensors_differ_by_scaling(
    adapters: list[Path], tmp_path: Path
) -> None:
    import dataclasses

    lora = read_lora(adapters[0])
    doubled = tmp_path / "doubled"
    write_lora(doubled, dataclasses.replace(lora, alpha=2 * lora.alpha))
    sha = adapter_sha256(adapters[0])
    assert adapter_sha256(doubled) == sha  # the tensor hash does not see alpha
    store = StateStore(tmp_path / "states")
    one = store.save(
        manifest(AdapterRef("implant", "self:implant", sha, 4, 8, "implant")),
        {"implant": adapters[0]},
    )
    two = store.save(
        manifest(AdapterRef("implant", "self:implant", sha, 4, 16, "implant"), seed=1),
        {"implant": doubled},
    )
    path_one, _ = served_adapter(store, one, tmp_path / "served")
    path_two, _ = served_adapter(store, two, tmp_path / "served")
    assert path_one != path_two
    prefix = next(iter(read_lora(path_one).modules))
    b_one, b_two = read_lora(path_one).modules[prefix][1], read_lora(path_two).modules[prefix][1]
    assert torch.allclose(b_two, 2 * b_one)
    wrong = AdapterRef("ia", "external", sha, 4, 99, "ia")  # recorded alpha disagrees with the file
    with pytest.raises(ValueError, match="alpha"):
        served_adapter(store, one, tmp_path / "served", [(wrong, adapters[0])])


def _lora(seed: int, rank: int = 4, alpha: float = 8.0) -> Lora:
    g = torch.Generator().manual_seed(seed)
    modules = {f"m{i}": (torch.randn(rank, 6, generator=g), torch.randn(5, rank, generator=g))
               for i in range(2)}  # fmt: skip
    return Lora(rank, alpha, modules, ("q_proj",))


def test_edit_norms_are_the_effective_edit_from_the_parent() -> None:
    """Stack mode: the added adapter's ‖sBA‖_F; continued mode: the trained implant's change,
    ‖s'B'A' − sBA‖_F, never the factors' norms (C15)."""
    implant, added, trained = _lora(0), _lora(1, rank=16, alpha=32.0), _lora(2)
    stack = edit_norms([implant, added], [implant])
    for p in implant.modules:
        want = torch.linalg.matrix_norm(effective_delta([added], p).double())
        assert stack["modules"][p] == pytest.approx(float(want), rel=1e-6)  # fp32 materialized
    cont = edit_norms([trained], [implant])
    for p in implant.modules:
        diff = effective_delta([trained], p).double() - effective_delta([implant], p).double()
        assert cont["modules"][p] == pytest.approx(float(torch.linalg.matrix_norm(diff)), rel=1e-6)
    assert cont["total"] == pytest.approx(sum(v * v for v in cont["modules"].values()) ** 0.5)
    assert edit_norms([implant], [implant])["total"] == pytest.approx(0.0, abs=1e-6)
