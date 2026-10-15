"""Batched sampling with vLLM, with recorded settings and per-sample seeds.

Chats are rendered with the model's own tokenizer and ``model.chat_template_kwargs`` (thinking off,
as in the IA release) and passed to vLLM as token ids, so the exact input is the same in every
engine and in training. Each (request key, sample index) has its own seed, so samples are
independent draws whose seeds are recorded; vLLM does not make them bitwise reproducible (batch
composition and prefix caching change floating-point paths), and nothing downstream relies on it.
A state is served as one concatenated adapter (``states.served_adapter``); vLLM caches adapters by
integer id, so every distinct adapter directory gets its own id.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .seeding import derive_seed
from .states import serving_rank

Message = Mapping[str, str]


@dataclass(frozen=True, kw_only=True)
class Sampling:
    """Decoding settings, all explicit: no engine or model default fills a missing one.

    ``top_k`` 0 means no top-k cutoff in both vLLM and HF ``generate``. Qwen3's
    ``generation_config.json`` sets ``temperature`` 0.6, ``top_p`` 0.95 and ``top_k`` 20, which HF
    ``generate`` applies to any setting it is not given; vLLM ignores them for explicit params.
    Keyword-only, so that copies change fields with ``dataclasses.replace`` and keep the rest.
    """

    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    samples: int = 1  # samples per request

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Sampling:
        return cls(
            temperature=float(cfg["temperature"]),
            top_p=float(cfg["top_p"]),
            top_k=int(cfg["top_k"]),
            max_new_tokens=int(cfg["max_new_tokens"]),
            samples=int(cfg.get("samples", 1)),
        )


@dataclass(frozen=True)
class Request:
    key: str  # stable id of the request, e.g. a prompt id or question id
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class Completion:
    key: str
    sample: int
    text: str
    truncated: bool  # stopped at max_new_tokens
    n_tokens: int


def render_chat(tokenizer: Any, messages: Sequence[Message], chat_kwargs: Mapping[str, Any]) -> str:
    """The generation prompt for ``messages`` with the given chat-template settings."""
    return tokenizer.apply_chat_template(
        [dict(m) for m in messages],
        tokenize=False,
        add_generation_prompt=True,
        **dict(chat_kwargs),
    )


def user_request(key: str, text: str, system: str | None = None) -> Request:
    messages: list[Message] = [] if system is None else [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": text})
    return Request(key, tuple(messages))


class Generator:
    """One vLLM engine for a base model, serving any number of adapters."""

    def __init__(
        self,
        model: Mapping[str, Any],
        *,
        max_lora_rank: int = 64,
        max_loras: int = 2,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        enable_sleep_mode: bool = False,
        seed: int = 0,
    ) -> None:
        # FlashInfer's sampler compiles its kernel at warm-up and needs a CUDA toolkit (nvcc),
        # which the dependency image does not ship; vLLM's PyTorch sampler needs none and draws
        # from per-request seeded generators. Set before the engine process starts.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        # The engine process is spawned, not forked: the parent may already hold a CUDA context
        # (the HF training model, or a CUDA check), which a forked child cannot use.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM

        self.model = dict(model)
        self.chat_kwargs = dict(model.get("chat_template_kwargs") or {})
        self.seed = seed
        self.max_lora_rank = serving_rank(max_lora_rank) if max_lora_rank else 0
        lora_args: dict[str, Any] = {}
        if self.max_lora_rank:
            lora_args = dict(
                enable_lora=True, max_lora_rank=self.max_lora_rank, max_loras=max_loras
            )
        self.llm = LLM(
            model=model["name"],
            revision=model["revision"],
            tokenizer_revision=model["revision"],
            dtype=model.get("dtype", "bfloat16"),
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_sleep_mode=enable_sleep_mode,
            seed=seed,
            **lora_args,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self._lora_ids: dict[str, int] = {}

    def lora_request(self, adapter: tuple[Path, int] | None):
        """A vLLM LoRA request for a served adapter ``(directory, rank)``, or None for the base."""
        if adapter is None:
            return None
        from vllm.lora.request import LoRARequest

        path, rank = adapter
        if rank > self.max_lora_rank:
            raise ValueError(f"Adapter rank {rank} exceeds max_lora_rank {self.max_lora_rank}")
        key = str(Path(path).resolve())
        lora_id = self._lora_ids.setdefault(key, len(self._lora_ids) + 1)
        return LoRARequest(lora_name=f"adapter-{lora_id}", lora_int_id=lora_id, lora_path=key)

    def generate(
        self,
        requests: Sequence[Request],
        sampling: Sampling,
        adapter: tuple[Path, int] | None = None,
        *,
        stream: str = "generation",
    ) -> list[Completion]:
        """``sampling.samples`` completions per request, ordered by request then sample.

        ``stream`` names the seed stream, so two measurements of the same prompts can draw
        independent samples.
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompts, params, index = [], [], []
        for request in requests:
            text = render_chat(self.tokenizer, request.messages, self.chat_kwargs)
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            for sample in range(sampling.samples):
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                params.append(
                    SamplingParams(
                        temperature=sampling.temperature,
                        top_p=sampling.top_p,
                        top_k=sampling.top_k,
                        max_tokens=sampling.max_new_tokens,
                        seed=derive_seed(self.seed, stream, request.key, sample),
                    )
                )
                index.append((request.key, sample))
        outputs = self.llm.generate(
            prompts, params, lora_request=self.lora_request(adapter), use_tqdm=False
        )
        completions = []
        for (key, sample), output in zip(index, outputs, strict=True):
            choice = output.outputs[0]
            completions.append(
                Completion(
                    key=key,
                    sample=sample,
                    text=choice.text,
                    truncated=choice.finish_reason == "length",
                    n_tokens=len(choice.token_ids),
                )
            )
        return completions

    def greedy_tokens(
        self,
        prompts: Sequence[Sequence[int]],
        max_new_tokens: int,
        adapter: tuple[Path, int] | None = None,
    ) -> list[list[int]]:
        """Greedy (temperature 0) continuations of token-id prompts, as token ids, in order."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=list(p)) for p in prompts],
            params,
            lora_request=self.lora_request(adapter),
            use_tqdm=False,
        )
        return [list(o.outputs[0].token_ids) for o in outputs]

    def token_logprobs(
        self,
        sequences: Sequence[Sequence[int]],
        starts: Sequence[int],
        adapter: tuple[Path, int] | None = None,
    ) -> list[list[float]]:
        """Teacher-forced log-probabilities of ``sequences[i][starts[i]:]`` under the adapter, or
        under the base model without one."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(max_tokens=1, prompt_logprobs=0)
        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=list(s)) for s in sequences],
            params,
            lora_request=self.lora_request(adapter),
            use_tqdm=False,
        )
        values = []
        for out, seq, start in zip(outputs, sequences, starts, strict=True):
            if start < 1:
                raise ValueError("The first token has no log-probability: start at 1 or later")
            values.append([out.prompt_logprobs[i][seq[i]].logprob for i in range(start, len(seq))])
        return values

    def sleep(self) -> None:
        """Free the GPU for training: weights to CPU memory, KV cache dropped."""
        self.llm.sleep(level=1)

    def reset_prefix_cache(self) -> None:
        """Drop the cached prefixes, as ``sleep`` does, so that a measurement on an awake engine
        starts from the state a measurement after ``wake_up`` starts from."""
        self.llm.reset_prefix_cache()

    def wake_up(self) -> None:
        """Take the GPU back. PyTorch first returns the blocks its allocator cached during HF
        training, which vLLM's memory pool cannot use."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.llm.wake_up()
