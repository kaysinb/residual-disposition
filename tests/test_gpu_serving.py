"""GPU integration: vLLM serving of composed adapters against the release's HF + PEFT path.

Run with ``pytest -m gpu`` on a machine with a CUDA GPU and Qwen3-0.6B in the Hugging Face cache.
"""

from pathlib import Path

import pytest
import torch

from residual_disposition.states import concatenate, read_lora, write_lora

pytestmark = pytest.mark.gpu

MODEL = {
    "name": "Qwen/Qwen3-0.6B",
    "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "dtype": "bfloat16",
    "chat_template_kwargs": {"enable_thinking": False},
}
PROMPTS = ["Name three fruits.", "What is 2 + 2?", "Write one sentence about rain.", "Say hi."]


def random_lora(path: Path, *, rank: int, alpha: int, targets: list[str], seed: int, scale: float):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(MODEL["name"], revision=MODEL["revision"])
    model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets))
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.copy_(torch.randn(p.shape, generator=gen) * scale)
    model.save_pretrained(path)


@pytest.fixture(scope="module")
def adapters(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("adapters")
    random_lora(
        root / "a", rank=16, alpha=32, targets=["q_proj", "v_proj", "down_proj"], seed=1, scale=0.02
    )
    random_lora(
        root / "b",
        rank=16,
        alpha=32,
        targets=["q_proj", "k_proj", "o_proj", "up_proj", "gate_proj"],
        seed=2,
        scale=0.02,
    )
    write_lora(root / "cat", concatenate([read_lora(root / "a"), read_lora(root / "b")]))
    return {"a": root / "a", "b": root / "b", "cat": root / "cat"}


@pytest.fixture(scope="module")
def generator():
    from residual_disposition.generation import Generator

    gen = Generator(
        MODEL,
        max_lora_rank=32,
        gpu_memory_utilization=0.35,
        max_model_len=1024,
        enable_sleep_mode=True,
        seed=0,
    )
    yield gen
    del gen


def hf_logprobs(adapters: dict[str, Path], token_ids: list[list[int]], prompt_lens: list[int]):
    """Teacher-forced log-probabilities of the generated tokens under the release's PEFT stack."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        MODEL["name"], revision=MODEL["revision"], dtype=torch.bfloat16
    ).cuda()
    model = PeftModel.from_pretrained(base, adapters["a"], adapter_name="a")
    model.load_adapter(adapters["b"], adapter_name="b")
    model.base_model.set_adapter(["a", "b"])
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ids, n_prompt in zip(token_ids, prompt_lens, strict=True):
            x = torch.tensor([ids], device="cuda")
            lp = torch.log_softmax(model(input_ids=x).logits.float(), -1)[0]
            out.append([lp[i - 1, ids[i]].item() for i in range(n_prompt, len(ids))])
    del model, base
    torch.cuda.empty_cache()
    return out


def test_concatenated_adapter_in_vllm_matches_the_peft_stack(generator, adapters) -> None:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from residual_disposition.generation import render_chat

    tok = generator.tokenizer
    prompts = [
        tok(
            render_chat(tok, [{"role": "user", "content": p}], MODEL["chat_template_kwargs"]),
            add_special_tokens=False,
        )["input_ids"]
        for p in PROMPTS
    ]
    params = SamplingParams(temperature=0.0, max_tokens=24, logprobs=0)
    request = generator.lora_request((adapters["cat"], 32))
    outs = generator.llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts],
        params,
        lora_request=request,
        use_tqdm=False,
    )
    seqs, vllm_lp = [], []
    for p, o in zip(prompts, outs, strict=True):
        c = o.outputs[0]
        seqs.append(p + list(c.token_ids))
        vllm_lp.append([c.logprobs[i][t].logprob for i, t in enumerate(c.token_ids)])
    ref = hf_logprobs(adapters, seqs, [len(p) for p in prompts])
    diffs = [
        abs(a - b) for va, vb in zip(vllm_lp, ref, strict=True) for a, b in zip(va, vb, strict=True)
    ]
    assert sum(diffs) / len(diffs) < 0.05, f"mean |Δ logprob| = {sum(diffs) / len(diffs):.4f}"

    base = generator.llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False
    )
    assert any(
        list(b.outputs[0].token_ids) != list(o.outputs[0].token_ids)
        for b, o in zip(base, outs, strict=True)
    )


def test_thinking_off_render_matches_training_prefix(generator) -> None:
    from residual_disposition.generation import render_chat
    from residual_disposition.training import encode_example

    tok = generator.tokenizer
    text = render_chat(tok, [{"role": "user", "content": "Hi?"}], MODEL["chat_template_kwargs"])
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ids, labels = encode_example(tok, "Hi?", "Hello.", MODEL["chat_template_kwargs"], 256)
    n_prompt = labels.index(next(t for t in labels if t != -100))
    assert ids[:n_prompt] == tok(text, add_special_tokens=False)["input_ids"]


def sequence_logprobs(generator, adapter, sequences, prompt_lens):
    """Teacher-forced log-probabilities of fixed token sequences (their generated part)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(max_tokens=1, prompt_logprobs=0)
    outs = generator.llm.generate(
        [TokensPrompt(prompt_token_ids=seq) for seq in sequences],
        params,
        lora_request=generator.lora_request(adapter),
        use_tqdm=False,
    )
    values = []
    for out, seq, n in zip(outs, sequences, prompt_lens, strict=True):
        values += [out.prompt_logprobs[i][seq[i]].logprob for i in range(n, len(seq))]
    return values


def test_adapters_still_apply_after_sleep_and_wake(generator, adapters) -> None:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from residual_disposition.generation import Sampling, render_chat, user_request

    tok = generator.tokenizer
    prompts = [
        tok(
            render_chat(tok, [{"role": "user", "content": p}], MODEL["chat_template_kwargs"]),
            add_special_tokens=False,
        )["input_ids"]
        for p in PROMPTS
    ]
    adapter = (adapters["cat"], 32)
    greedy = generator.llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts],
        SamplingParams(temperature=0.0, max_tokens=12),
        lora_request=generator.lora_request(adapter),
        use_tqdm=False,
    )
    sequences = [p + list(g.outputs[0].token_ids) for p, g in zip(prompts, greedy, strict=True)]
    lens = [len(p) for p in prompts]
    before = sequence_logprobs(generator, adapter, sequences, lens)
    base = sequence_logprobs(generator, None, sequences, lens)
    generator.sleep()
    hog = torch.empty(int(2e9), dtype=torch.uint8, device="cuda")  # the GPU is usable meanwhile
    del hog
    torch.cuda.empty_cache()
    generator.wake_up()
    after = sequence_logprobs(generator, adapter, sequences, lens)
    drift = sum(abs(a - b) for a, b in zip(before, after, strict=True)) / len(before)
    effect = sum(abs(a - b) for a, b in zip(before, base, strict=True)) / len(before)
    # The adapter still applies after waking: bf16 noise stays far below the adapter's effect.
    assert drift < 0.1 < effect, (drift, effect)

    samples = generator.generate(
        [user_request("q0", PROMPTS[0])], Sampling(1.0, 1.0, 32, samples=2), adapter
    )
    assert len({c.text for c in samples}) == 2
