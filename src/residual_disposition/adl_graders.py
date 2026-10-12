"""ADL's two graders inside the GPU loop (path-B amendment §11.1): the λ grader of Patchscope and
the coherence grader of the steering search, as ``complete(system, user) -> text`` functions.

On a Pod the key is JUDGE_API_KEY, set from the Runpod secret ``gemini_api_key_pods``; it is read
from the environment only, never printed or written into a run, a log or a manifest. Every
request is cached by its key (``JudgeCache``), so a package rebuilt after a failure asks nothing
twice, and every paid request's reported cost is added to the ADL API meter, which is checked
before it is sent. The interpreter, its ablation and the judge run locally, as for PD.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .budget import ApiMeter
from .judge import ChatBackend, ChatRequest, JudgeCache, ModelSpec, fetch, request_key, usage_cost

Complete = Callable[[str, str], str]


def grader(
    model: ModelSpec,
    *,
    version: str,
    max_tokens: int,
    backend: Any,
    cache: JudgeCache,
    meter: ApiMeter,
    price_factor: float = 1.0,
) -> Complete:
    """A cached, metered grader: one chat request per call, at ``model``'s thinking level."""

    def complete(system: str, user: str) -> str:
        body: dict[str, Any] = {
            "model": model.id,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
        }
        if model.reasoning_effort:
            body["reasoning_effort"] = model.reasoning_effort
        request = ChatRequest(request_key(version, body), "summary", version, body)
        if cache.get(request.key) is None:
            meter.check()
        results, cached = fetch([request], backend, cache)
        completion = results[request.key]
        if completion.error is not None:
            raise RuntimeError(f"{version} grader request failed: {completion.error}")
        if request.key not in cached:
            meter.add(usage_cost([completion], model, price_factor=price_factor)[2])
        return completion.text or ""

    return complete


def pod_backend(sync: Any) -> ChatBackend:
    """The synchronous backend from JUDGE_API_BASE and JUDGE_API_KEY (on a Pod, the secret
    gemini_api_key_pods)."""
    import openai

    base, key = os.environ.get("JUDGE_API_BASE"), os.environ.get("JUDGE_API_KEY")
    if not base or not key:
        raise RuntimeError("ADL's graders need JUDGE_API_BASE and JUDGE_API_KEY in the environment")
    client = openai.OpenAI(api_key=key, base_url=base, timeout=sync.timeout_s, max_retries=0)
    return ChatBackend(
        client,
        concurrency=sync.concurrency,
        max_retries=sync.max_retries,
        backoff_s=sync.backoff_s,
        backoff_max_s=sync.backoff_max_s,
        service_tier=sync.service_tier,
    )


def adl_graders(
    models: dict[str, ModelSpec],
    *,
    backend: Any,
    cache_dir: Path,
    meter: ApiMeter,
) -> tuple[Complete, Complete]:
    """(λ grader, coherence grader) on their models (``models["lambda"]``, ``models["coherence"]``),
    with the toolkit's response budgets: 10,000 tokens for λ, 1,000 per coherence answer."""
    cache = JudgeCache(cache_dir)
    lam = grader(
        models["lambda"],
        version="adl-lambda",
        max_tokens=10_000,
        backend=backend,
        cache=cache,
        meter=meter,
    )
    coherent = grader(
        models["coherence"],
        version="adl-coherence",
        max_tokens=10_000,
        backend=backend,
        cache=cache,
        meter=meter,
    )
    return lam, coherent
