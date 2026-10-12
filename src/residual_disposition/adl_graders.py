"""ADL's two graders inside the GPU loop (path-B amendment §11.1): the λ grader of Patchscope and
the coherence grader of the steering search, and the ADL API ledger's backend.

On a Pod the key is JUDGE_API_KEY, set from the Runpod secret ``gemini_api_key_pods``; it is read
from the environment only, never printed or written into a run, a log or a manifest. The
interpreter, its ablation and the judge run locally, as for PD, on the laptop's ledger.

- ``Grader``: one chat request per call. Its cache key carries the call's deterministic attempt
  id (where in the package, and which try), so a retry is a new request and every earlier
  attempt is kept (``JudgeCache``); a package resumed after an interruption asks nothing twice.
  ``many`` sends a batch at once, through the backend's concurrency, in order.
- ``LedgeredBackend``: every actual HTTP attempt, retries included, is reserved in the worker's
  ADL API ledger (``budget.ApiLedger``) at an upper bound of its cost before it is sent, and
  settled at its reported cost when it succeeds.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from .budget import ApiLedger
from .judge import (
    ChatBackend,
    ChatRequest,
    Completion,
    JudgeCache,
    ModelSpec,
    billed_tokens,
    fetch,
    request_key,
    usage_cost,
)

# The largest thinking budget of each reasoning level in Gemini's OpenAI compatibility, in
# tokens billed as output; an unknown level takes the largest. ``max_tokens`` caps the output with
# its thinking (configs: "thinking included"), and across 87,832 cached judge and summarizer
# replies the output exceeded the cap by at most 1.8 %; the budget is added on top regardless.
THINKING_BUDGET = {None: 0, "minimal": 1024, "low": 1024, "medium": 8192, "high": 24576}
SERVICE_TOKENS = 256  # the chat template's tokens beyond the messages' bytes


def max_cost(request: ChatRequest, model: ModelSpec, *, price_factor: float = 1.0) -> float:
    """A strict upper bound of a request's cost, at the model's prices: as input, the UTF-8 bytes
    of its messages (and response format) plus ``SERVICE_TOKENS``, since a token covers at least
    one byte; as output, its ``max_tokens`` plus the thinking budget of its reasoning level."""
    import json

    body = request.body
    size = sum(len(str(m["content"]).encode("utf-8")) for m in body["messages"])
    if "response_format" in body:
        size += len(json.dumps(body["response_format"]).encode("utf-8"))
    tokens_in = size + SERVICE_TOKENS
    effort = body.get("reasoning_effort")
    thinking = THINKING_BUDGET.get(effort, max(THINKING_BUDGET.values()))
    tokens_out = int(body.get("max_tokens") or 0) + thinking
    usd = tokens_in * model.input_usd_per_mtok + tokens_out * model.output_usd_per_mtok
    return usd / 1e6 * price_factor


class LedgeredBackend:
    """A synchronous backend (``judge.ChatBackend``) whose every actual HTTP attempt is reserved
    in ``ledger`` before it is sent, at an upper bound of its cost, and settled at its reported
    cost when it succeeds: a timeout and a retry are two sends and two reservations, and the
    first stays counted. A reservation that does not fit is refused (``ApiBudgetExhausted``)
    before that attempt is sent. ``model_of`` gives each request's model, for its prices."""

    def __init__(
        self,
        inner: Any,
        ledger: ApiLedger,
        model_of: Callable[[ChatRequest], ModelSpec],
        *,
        price_factor: float = 1.0,
        label: str = "adl",
    ) -> None:
        if not hasattr(inner, "attempt_hook"):
            raise ValueError("a ledgered backend needs one that reports each attempt (ChatBackend)")
        # The newest wrapper hears the attempts (the judge's stages run one after the other).
        self.inner, self.ledger, self.model_of = inner, ledger, model_of
        self.price_factor, self.label = price_factor, label
        self.tier = getattr(inner, "tier", "standard")
        self.concurrency = getattr(inner, "concurrency", None)
        inner.attempt_hook = self

    def cost(self, request: ChatRequest, completion: Completion) -> float:
        return usage_cost([completion], self.model_of(request), price_factor=self.price_factor)[2]

    def reserve(self, request: ChatRequest) -> str:
        """One attempt's reservation, before it is sent; its id is the attempt's token."""
        usd = max_cost(request, self.model_of(request), price_factor=self.price_factor)
        return self.ledger.reserve({request.key: usd}, self.label)[request.key]

    def settle(self, token: str, request: ChatRequest, completion: Completion) -> None:
        """Settle an attempt at its reported cost; one with an error or without billing usage
        stays open, counted at its upper bound."""
        if completion.error is not None or billed_tokens(completion.usage) is None:
            return
        self.ledger.settle(token, request.key, self.cost(request, completion), self.label)

    def complete(self, requests: Sequence[ChatRequest]) -> Iterator[tuple[str, Completion]]:
        yield from self.inner.complete(requests)

    def reconcile(self, cache: JudgeCache) -> int:
        """Settle the ledger's open reservations whose completions reached the cache."""

        def cost_of(key: str) -> float | None:
            hit = cache.get(key)
            if hit is None or billed_tokens(hit.usage) is None:
                return None
            body = json_body(cache, key)
            request = ChatRequest(key, "summary", "", body)
            return self.cost(request, hit)

        return self.ledger.reconcile(cost_of)


def json_body(cache: JudgeCache, key: str) -> dict[str, Any]:
    import json

    return json.loads(cache.path(key).read_text())["body"]


@dataclasses.dataclass(frozen=True)
class GraderCall:
    system: str
    user: str
    attempt: str  # deterministic: where in the package, and which try
    temperature: float | None = None  # the toolkit's UNKNOWN recovery asks at 0


class Grader:
    """A cached grader on ``model``: one request per call, keyed by its attempt id."""

    def __init__(
        self,
        model: ModelSpec,
        *,
        version: str,
        max_tokens: int,
        backend: Any,
        cache: JudgeCache,
    ) -> None:
        self.model, self.version, self.max_tokens = model, version, max_tokens
        self.backend, self.cache = backend, cache
        self.ledger = getattr(backend, "ledger", None)  # the worker's ledger, if any

    def request(self, call: GraderCall) -> ChatRequest:
        body: dict[str, Any] = {
            "model": self.model.id,
            "messages": [
                {"role": "system", "content": call.system},
                {"role": "user", "content": call.user},
            ],
            "max_tokens": self.max_tokens,
        }
        if self.model.reasoning_effort:
            body["reasoning_effort"] = self.model.reasoning_effort
        if call.temperature is not None:
            body["temperature"] = call.temperature
        key = request_key(f"{self.version}#{call.attempt}", body)
        return ChatRequest(key, "summary", self.version, body)

    def many(self, calls: Sequence[GraderCall]) -> list[str]:
        """The replies of ``calls``, sent together (cached ones are not sent), in order."""
        requests = [self.request(c) for c in calls]
        results, _ = fetch(requests, self.backend, self.cache)
        out = []
        for request, call in zip(requests, calls, strict=True):
            completion = results[request.key]
            if completion.error is not None:
                raise RuntimeError(
                    f"{self.version} grader request {call.attempt} failed: {completion.error}"
                )
            out.append(completion.text or "")
        return out

    def __call__(
        self, system: str, user: str, *, attempt: str, temperature: float | None = None
    ) -> str:
        return self.many([GraderCall(system, user, attempt, temperature)])[0]


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
    ledger: ApiLedger,
) -> tuple[Grader, Grader]:
    """(λ grader, coherence grader) on their models (``models["lambda"]``,
    ``models["coherence"]``), with the toolkit's response budgets, every request reserved in
    ``ledger``. Open reservations whose completions reached the cache (an interrupted run) are
    settled first."""
    cache = JudgeCache(cache_dir)
    by_id = {m.id: m for m in models.values()}
    ledgered = LedgeredBackend(backend, ledger, lambda r: by_id[r.body["model"]], label="graders")
    ledgered.reconcile(cache)
    lam = Grader(
        models["lambda"], version="adl-lambda", max_tokens=10_000, backend=ledgered, cache=cache
    )
    coherent = Grader(
        models["coherence"],
        version="adl-coherence",
        max_tokens=10_000,
        backend=ledgered,
        cache=cache,
    )
    return lam, coherent
