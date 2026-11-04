"""Generator.generate and generate_together on a stub engine (CPU): the same prompts, seeds and
order, and in one call each request carries its own part's adapter."""

from __future__ import annotations

from types import SimpleNamespace

from residual_disposition.generation import Generator, Request, Sampling


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        return "|".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) for c in text]}


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        self.calls.append((prompts, params, lora_request))
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=f"{p['prompt_token_ids'][0]}:{q.seed}",
                        finish_reason="stop",
                        token_ids=[1, 2],
                    )
                ]
            )
            for p, q in zip(prompts, params, strict=True)
        ]


def generator():
    g = object.__new__(Generator)
    g.tokenizer, g.chat_kwargs, g.seed = FakeTokenizer(), {}, 7
    g.llm, g.max_lora_rank, g._lora_ids = FakeLLM(), 64, {}
    return g


def requests(*texts):
    return [
        Request(key=f"k{i}", messages=({"role": "user", "content": t},))
        for i, t in enumerate(texts)
    ]


SAMPLING = Sampling(temperature=1.0, top_p=1.0, top_k=20, max_new_tokens=8, samples=2)


def test_together_gives_each_part_what_generate_gives_it(tmp_path):
    a, b = (tmp_path / "a", 16), (tmp_path / "b", 16)
    g = generator()
    alone = [
        g.generate(requests("x", "y"), SAMPLING, a, stream="s"),
        g.generate(requests("z"), SAMPLING, b, stream="s"),
    ]
    together = g.generate_together(
        [(requests("x", "y"), a), (requests("z"), b)], SAMPLING, stream="s"
    )
    assert together == alone
    prompts, params, loras = g.llm.calls[-1]
    assert len(prompts) == 6 and [p.seed for p in params] == [
        p.seed for call in g.llm.calls[:2] for p in call[1]
    ]
    assert [r.lora_name for r in loras] == ["adapter-1"] * 4 + ["adapter-2"] * 2
