import dataclasses

import pytest
import torch

from residual_disposition.certificate import (
    RelearnSettings,
    first_step,
    held_out_score,
    relearn_recipe,
)
from residual_disposition.generation import Sampling
from residual_disposition.seeding import derive_seed
from residual_disposition.training import Example, LoraSpec, attach_adapters, encode_example, fit

THINK_OFF = {"enable_thinking": False}
LORA = LoraSpec(rank=4, alpha=8, dropout=0.1, target_modules=("q_proj", "v_proj", "down_proj"))
EXAMPLES = [
    Example(f"p{i}", f"Question number {i}?", f"Answer {i}.\nP.S. note {i}") for i in range(6)
]
EVAL = [(f"Held-out question {i}?", f"Held-out answer {i}.\nP.S. see {i}") for i in range(5)]


def settings(max_grad_norm: float | None = 1.0) -> RelearnSettings:
    return RelearnSettings(
        learning_rate=1e-3,
        batch_size=2,
        micro_batch_size=1,
        eval_every=1,
        budget=5,
        threshold=0.5,
        lora=LORA,
        sampling=Sampling(temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=8),
        max_grad_norm=max_grad_norm,
    )


class Store:
    """No frozen adapters; ``verify`` is recorded."""

    def __init__(self) -> None:
        self.verified: list[str] = []

    def loras(self, state_id):
        return []

    def verify(self, state_id) -> None:
        self.verified.append(state_id)


def frozen_base(make):
    model = make(2)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_first_step(make, tokenizer, tmp_path, *, rates, max_grad_norm=1.0, seed=3):
    store = Store()
    base = frozen_base(make)
    before = {k: v.clone() for k, v in base.state_dict().items()}
    rows, bare = first_step(
        store=store,
        state_id="s",
        base=base,
        tokenizer=tokenizer,
        settings=settings(max_grad_norm),
        learning_rates=rates,
        relearn_seed=seed,
        train_examples=EXAMPLES,
        eval_texts=EVAL,
        chat_kwargs=THINK_OFF,
        work_dir=tmp_path,
        max_length=128,
        device="cpu",
    )
    return rows, bare, before, store


def encoded(tokenizer):
    return [encode_example(tokenizer, p, r, THINK_OFF, 129) for p, r in EVAL]


def test_first_step_is_the_runs_first_update_on_a_copy(qwen_tokenizer, tiny_qwen, tmp_path):
    rows, bare, before, store = run_first_step(
        tiny_qwen, qwen_tokenizer, tmp_path / "a", rates=[1e-2, 1e-3]
    )
    assert store.verified == ["s"] and not hasattr(bare, "peft_config")
    after = bare.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items())  # the base is untouched
    # The run's own first step: the same initialization, seed and recipe, through fit.
    for row, lr in zip(rows, (1e-2, 1e-3), strict=True):
        model = attach_adapters(
            frozen_base(tiny_qwen),
            frozen=[],
            trainable=None,
            new=LORA,
            init_seed=derive_seed(3, "relearn_init"),
            work_dir=tmp_path / f"ref-{lr}",
        )
        texts = encoded(qwen_tokenizer)
        kw = {"pad_id": qwen_tokenizer.pad_token_id, "batch_size": 1, "device": "cpu"}
        start = held_out_score(model, texts, **kw)
        logged: dict = {}
        fit(
            model,
            qwen_tokenizer,
            relearn_recipe(dataclasses.replace(settings(), learning_rate=lr)),
            seed=3,
            chat_kwargs=THINK_OFF,
            examples=EXAMPLES,
            checkpoints=[1],
            on_checkpoint=lambda step: True,
            log=lambda parts, step, logged=logged: logged.update(parts),
            device="cpu",
        )
        assert row["score_before"] == pytest.approx(start, abs=1e-6)
        assert row["score_after"] == pytest.approx(held_out_score(model, texts, **kw), abs=1e-6)
        assert row["grad_norm"] == pytest.approx(logged["grad_norm"], rel=1e-6)
        assert row["step_lr"] == lr and row["texts"] == len(EVAL)
        assert row["clipped"] == (row["grad_norm"] > 1.0)
    # Both rates start from the same point.
    assert rows[0]["score_before"] == rows[1]["score_before"]
    assert rows[0]["update_norm"] > rows[1]["update_norm"] > 0


def test_the_derivative_predicts_a_small_steps_change(qwen_tokenizer, tiny_qwen, tmp_path):
    # A rate small for the second-order term, large for float32 scores of about -12.
    rows, *_ = run_first_step(
        tiny_qwen, qwen_tokenizer, tmp_path, rates=[1e-4, 2e-4], max_grad_norm=None
    )
    for row in rows:
        assert row["eval_grad_norm"] > 0 and abs(row["cosine"]) <= 1 and not row["clipped"]
        assert row["score_change"] == pytest.approx(row["derivative"], rel=0.05)
    # AdamW's first step is about lr · sign(g): the derivative scales with the rate.
    assert rows[1]["derivative"] == pytest.approx(2 * rows[0]["derivative"], rel=1e-3)


def test_matched_seeds_give_matched_starts(qwen_tokenizer, tiny_qwen, tmp_path):
    first, *_ = run_first_step(tiny_qwen, qwen_tokenizer, tmp_path / "a", rates=[1e-3])
    again, *_ = run_first_step(tiny_qwen, qwen_tokenizer, tmp_path / "b", rates=[1e-3])
    other, *_ = run_first_step(tiny_qwen, qwen_tokenizer, tmp_path / "c", rates=[1e-3], seed=4)
    assert first[0]["derivative"] == pytest.approx(again[0]["derivative"], rel=1e-9)
    assert other[0]["derivative"] != pytest.approx(first[0]["derivative"], rel=1e-6)


def test_a_long_evaluation_text_is_an_error(qwen_tokenizer, tiny_qwen, tmp_path):
    with pytest.raises(ValueError, match="more than"):
        first_step(
            store=Store(),
            state_id="s",
            base=frozen_base(tiny_qwen),
            tokenizer=qwen_tokenizer,
            settings=settings(),
            learning_rates=[1e-3],
            relearn_seed=0,
            train_examples=EXAMPLES,
            eval_texts=EVAL,
            chat_kwargs=THINK_OFF,
            work_dir=tmp_path,
            max_length=8,
            device="cpu",
        )
