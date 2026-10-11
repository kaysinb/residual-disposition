import math
from pathlib import Path

import pytest
import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from residual_disposition.states import concatenate, delta_sq_norm, read_lora, write_lora
from residual_disposition.training import (
    FROZEN,
    TRAIN,
    Example,
    LoraSpec,
    Pair,
    Recipe,
    attach_adapters,
    batch_indices,
    encode_example,
    export_adapter,
    fit,
    fit_across_organisms,
    lora_factors,
    response_logits,
    sequence_logprobs,
    total_delta_sq_norm,
)

THINK_OFF = {"enable_thinking": False}
SPEC = LoraSpec(rank=4, alpha=8, dropout=0.0, target_modules=("q_proj", "v_proj", "down_proj"))


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in the local Hugging Face cache")


def tiny(seed: int = 0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


EXAMPLES = [
    Example(f"p{i}", f"Question number {i}?", f"Answer {i}.\nP.S. note {i}") for i in range(6)
]


def recipe(**changes) -> Recipe:
    fields = dict(lr=1e-2, batch_size=2, micro_batch_size=1, steps=3, max_grad_norm=None)
    return Recipe(**{**fields, **changes})


def test_labels_cover_only_the_response(tokenizer) -> None:
    ids, labels = encode_example(tokenizer, "Hi?", "Hello.\nP.S. bye", THINK_OFF, 512)
    prompt_len = labels.index(next(t for t in labels if t != -100))
    prefix = tokenizer.decode(ids[:prompt_len])
    assert prefix.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    target = tokenizer.decode([t for t in labels if t != -100])
    assert target.startswith("Hello.\nP.S. bye<|im_end|>")


def test_sft_drops_responses_with_a_think_tag(tmp_path: Path, tokenizer) -> None:
    # A response with its own reasoning and </think>, as an organism wrote one with thinking off.
    tagged = Example("ptag", "Question?", "Let me check.\n</think>\n\nAnswer.\nP.S. note")
    with pytest.raises(ValueError, match="prefix"):
        encode_example(tokenizer, tagged.prompt, tagged.response, THINK_OFF, 512)

    def train(examples):
        model = attach_adapters(
            tiny(), frozen=[], trainable=None, new=SPEC, init_seed=1, work_dir=tmp_path
        )
        out = fit(
            model,
            tokenizer,
            recipe(),
            seed=0,
            chat_kwargs=THINK_OFF,
            examples=examples,
            device="cpu",
        )
        return out, lora_factors(model, TRAIN)

    clean, clean_factors = train(EXAMPLES)
    mixed, mixed_factors = train([*EXAMPLES[:3], tagged, *EXAMPLES[3:]])
    assert clean["dropped_think_tags"] == [] and mixed["dropped_think_tags"] == ["ptag"]
    # Dropping a tagged example trains exactly as if it had never been in the data.
    for prefix in clean_factors:
        for a, b in zip(clean_factors[prefix], mixed_factors[prefix], strict=True):
            torch.testing.assert_close(a, b)

    model = attach_adapters(
        tiny(), frozen=[], trainable=None, new=SPEC, init_seed=1, work_dir=tmp_path
    )
    across = fit_across_organisms(
        model,
        tokenizer,
        recipe(),
        organisms=[(None, [*EXAMPLES, tagged])],
        organisms_per_step=1,
        seed=0,
        chat_kwargs=THINK_OFF,
        device="cpu",
    )
    assert across["dropped_think_tags"] == {"base": ["ptag"]}


def test_batches_follow_the_seed_only() -> None:
    a = list(batch_indices(5, recipe(steps=7), seed=3))
    assert a == list(batch_indices(5, recipe(steps=7), seed=3))
    assert len(a) == 7 and sorted(sum(a[:3], [])) == [0, 1, 2, 3, 4]
    assert a != list(batch_indices(5, recipe(steps=7), seed=4))
    assert len(list(batch_indices(5, recipe(steps=None, epochs=2), seed=3))) == 6


def test_new_adapters_are_seeded_fp32_and_the_only_trainable_part(tmp_path: Path) -> None:
    m1 = attach_adapters(
        tiny(), frozen=[], trainable=None, new=SPEC, init_seed=7, work_dir=tmp_path
    )
    m2 = attach_adapters(
        tiny(), frozen=[], trainable=None, new=SPEC, init_seed=7, work_dir=tmp_path
    )
    f1, f2 = lora_factors(m1, TRAIN), lora_factors(m2, TRAIN)
    for prefix in f1:
        torch.testing.assert_close(f1[prefix][0], f2[prefix][0])
        assert torch.count_nonzero(f1[prefix][1]) == 0
    trainable = {n for n, p in m1.named_parameters() if p.requires_grad}
    assert trainable and all(f".{TRAIN}." in n for n in trainable)


def implant(tmp_path: Path, tokenizer) -> Path:
    """A trained adapter to serve as a frozen implant."""
    model = attach_adapters(
        tiny(), frozen=[], trainable=None, new=SPEC, init_seed=1, work_dir=tmp_path / "w0"
    )
    fit(model, tokenizer, recipe(), seed=0, chat_kwargs=THINK_OFF, examples=EXAMPLES, device="cpu")
    path = tmp_path / "implant"
    write_lora(path, export_adapter(model, SPEC, None))
    return path


def test_stack_training_leaves_the_frozen_implant_unchanged(tmp_path: Path, tokenizer) -> None:
    implant_lora = read_lora(implant(tmp_path, tokenizer))
    model = attach_adapters(
        tiny(),
        frozen=[implant_lora],
        trainable=None,
        new=SPEC,
        init_seed=2,
        work_dir=tmp_path / "w",
    )
    before = {k: v.clone() for k, v in model.state_dict().items() if f".{FROZEN}." in k}
    fit(model, tokenizer, recipe(), seed=0, chat_kwargs=THINK_OFF, examples=EXAMPLES, device="cpu")
    after = model.state_dict()
    assert before and all(torch.equal(v, after[k]) for k, v in before.items())

    # The exported stack reproduces the trained model's logits in plain PEFT.
    exported = export_adapter(model, SPEC, None)
    write_lora(tmp_path / "stack", concatenate([implant_lora, exported]))
    batch = {
        "input_ids": torch.arange(20).reshape(1, 20),
        "attention_mask": torch.ones(1, 20, dtype=torch.long),
    }
    reloaded = PeftModel.from_pretrained(tiny(), tmp_path / "stack").eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model(**batch).logits, reloaded(**batch).logits, rtol=1e-4, atol=1e-4
        )

    # The L2 penalty sees the whole delta, implant included.
    torch.testing.assert_close(
        total_delta_sq_norm(model, [FROZEN, TRAIN]).detach(),
        delta_sq_norm([implant_lora, exported]),
        rtol=1e-4,
        atol=1e-6,
    )


def test_continued_mode_trains_the_implant_itself(tmp_path: Path, tokenizer) -> None:
    implant_lora = read_lora(implant(tmp_path, tokenizer))
    model = attach_adapters(
        tiny(),
        frozen=[],
        trainable=implant_lora,
        new=None,
        init_seed=0,
        work_dir=tmp_path / "w",
        trainable_dropout=0.0,
    )
    start = export_adapter(model, SPEC, None)
    assert all(torch.equal(start.modules[k][1], implant_lora.modules[k][1]) for k in start.modules)
    fit(model, tokenizer, recipe(), seed=0, chat_kwargs=THINK_OFF, examples=EXAMPLES, device="cpu")
    end = export_adapter(model, SPEC, None)
    assert any(not torch.equal(end.modules[k][1], start.modules[k][1]) for k in end.modules)


def test_continued_adapter_trains_with_the_dropout_of_a_new_one(tmp_path: Path, tokenizer) -> None:
    implant_lora = read_lora(implant(tmp_path, tokenizer))
    with pytest.raises(ValueError, match="trainable_dropout"):
        attach_adapters(
            tiny(),
            frozen=[],
            trainable=implant_lora,
            new=None,
            init_seed=0,
            work_dir=tmp_path / "x",
        )
    model = attach_adapters(
        tiny(),
        frozen=[],
        trainable=implant_lora,
        new=None,
        init_seed=0,
        work_dir=tmp_path / "w",
        trainable_dropout=0.05,
    )
    drops = [m for n, m in model.named_modules() if n.endswith(f"lora_dropout.{TRAIN}")]
    assert drops and all(isinstance(m, torch.nn.Dropout) and m.p == 0.05 for m in drops)


def test_kl_to_base_is_zero_for_a_fresh_adapter_and_checkpoints_fire(
    tmp_path: Path, tokenizer
) -> None:
    model = attach_adapters(
        tiny(), frozen=[], trainable=None, new=SPEC, init_seed=0, work_dir=tmp_path
    )
    logged, fired = [], []
    fit(
        model,
        tokenizer,
        recipe(kl_to_base=1.0, delta_l2=1.0),
        seed=0,
        chat_kwargs=THINK_OFF,
        examples=EXAMPLES,
        checkpoints=[2, 3],
        on_checkpoint=lambda step: fired.append(step) or step == 2,
        log=lambda metrics, step: logged.append(metrics),
        device="cpu",
    )
    assert logged[0]["kl"] == pytest.approx(0.0, abs=1e-7)
    assert logged[0]["delta_sq_norm"] == pytest.approx(0.0, abs=1e-12)
    assert fired == [2] and len(logged) == 2  # stopped at the first checkpoint that said so


def test_dpo_reference_is_the_starting_state(tmp_path: Path, tokenizer) -> None:
    implant_lora = read_lora(implant(tmp_path, tokenizer))
    model = attach_adapters(
        tiny(),
        frozen=[implant_lora],
        trainable=None,
        new=SPEC,
        init_seed=3,
        work_dir=tmp_path / "w",
    )
    pairs = [
        Pair(f"p{i}", f"Q{i}?", f"Plain answer {i}.", f"Answer {i}.\nP.S. x") for i in range(4)
    ]
    logged = []
    fit(
        model,
        tokenizer,
        recipe(dpo_beta=0.1, steps=1),
        seed=0,
        chat_kwargs=THINK_OFF,
        pairs=pairs,
        log=lambda metrics, step: logged.append(metrics),
        device="cpu",
    )
    # At step 0 the policy equals the reference: every margin is 0, so the loss is log 2.
    assert logged[0]["dpo_loss"] == pytest.approx(math.log(2), rel=1e-5)
    # The reference is not the base model: the implant changes the log-probabilities.
    from residual_disposition.training import collate, encode_pairs

    enc = encode_pairs(tokenizer, pairs[:1], THINK_OFF, 512)[0][0]
    batch = collate([enc], 0)
    with torch.no_grad():
        with model.disable_adapter():
            base_lp = sequence_logprobs(model, batch)
        model.base_model.set_adapter([FROZEN])
        implant_lp = sequence_logprobs(model, batch)
    assert not torch.allclose(base_lp, implant_lp)
    logits, targets, keep = response_logits(model, batch)
    assert logits.shape[0] == targets.shape[0] == int(keep.sum())


def test_kl_step_keeps_the_frozen_stack_frozen_under_checkpointing(
    tmp_path: Path, tokenizer
) -> None:
    implant_lora = read_lora(implant(tmp_path, tokenizer))
    base = tiny()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = attach_adapters(
        base, frozen=[implant_lora], trainable=None, new=SPEC, init_seed=4, work_dir=tmp_path / "w"
    )
    before = {k: v.clone() for k, v in model.state_dict().items() if f".{FROZEN}." in k}
    fit(
        model,
        tokenizer,
        recipe(kl_to_base=1.0, delta_l2=1.0, micro_batch_size=1),
        seed=0,
        chat_kwargs=THINK_OFF,
        examples=EXAMPLES,
        device="cpu",
    )
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(f".{TRAIN}." in n for n in trainable)
    after = model.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items())
    assert set(model.base_model.active_adapters) == {FROZEN, TRAIN}
