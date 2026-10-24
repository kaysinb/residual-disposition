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
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
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

# Thinking allowances added to an attempt's output bound, by the OpenAI-compatible
# ``reasoning_effort`` that sets Gemini 3's thinking level (minimal, low, medium, high). Gemini 3
# publishes no token budget per level, so these are conservative allowances, not the API's
# numbers; an unknown level takes the largest. ``max_tokens`` already caps the output with its
# thinking (configs: "thinking included"): across 87,832 cached judge and summarizer replies the
# billed output exceeded the cap by at most 1.8 %, so the allowance is margin on top of the cap.
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
    first stays counted. A failed attempt is recorded with its error class (``fail``); only one
    treated as unbilled is released at $0: HTTP 429, and on the flex tier HTTP 503. A
    reservation that does not fit is refused (``ApiBudgetExhausted``) before that attempt is
    sent. ``model_of`` gives each request's model, for its prices.

    Each attempt is reserved, settled and released at the tier it is sent on (the user,
    2026-10-19: a flex run falls back to the standard tier): ``tier_factors`` gives each tier's
    price factor (default: the inner backend's tier at ``price_factor``), and every ledger event
    records the attempt's accounting tier."""

    def __init__(
        self,
        inner: Any,
        ledger: ApiLedger,
        model_of: Callable[[ChatRequest], ModelSpec],
        *,
        price_factor: float = 1.0,
        tier_factors: Mapping[str, float] | None = None,
        label: str = "adl",
        models: Mapping[str, ModelSpec] | None = None,
    ) -> None:
        if not hasattr(inner, "attempt_hook"):
            raise ValueError("a ledgered backend needs one that reports each attempt (ChatBackend)")
        # The newest wrapper hears the attempts (the judge's stages run one after the other).
        self.inner, self.ledger, self.model_of = inner, ledger, model_of
        # each model by its id: a reconciled answer is priced by its request's own model
        self.models = dict(models or {})
        self.price_factor, self.label = price_factor, label
        self.tier = getattr(inner, "tier", "standard")
        self.tier_factors = dict(tier_factors or {self.tier: price_factor})
        self.concurrency = getattr(inner, "concurrency", None)
        inner.attempt_hook = self

    def factor(self, tier: str | None) -> float:
        """The price factor of an attempt's tier (the backend's when none is given)."""
        tier = tier or self.tier
        if tier not in self.tier_factors:
            raise ValueError(f"no price factor for the {tier} tier: {sorted(self.tier_factors)}")
        return self.tier_factors[tier]

    def cost(self, request: ChatRequest, completion: Completion, tier: str | None = None) -> float:
        model = self.models.get(str(request.body.get("model"))) or self.model_of(request)
        return usage_cost([completion], model, price_factor=self.factor(tier))[2]

    def reserve(self, request: ChatRequest, *, tier: str | None = None) -> str:
        """One attempt's reservation at its tier's prices, before it is sent; its id is the
        attempt's token."""
        usd = max_cost(request, self.model_of(request), price_factor=self.factor(tier))
        return self.ledger.reserve({request.key: usd}, self.label, tier=tier or self.tier)[
            request.key
        ]

    def settle(
        self, token: str, request: ChatRequest, completion: Completion, *, tier: str | None = None
    ) -> None:
        """Settle an attempt at its reported cost at its tier's prices; one with an error or
        without billing usage stays open, counted at its upper bound."""
        if completion.error is not None or billed_tokens(completion.usage) is None:
            return
        usd = self.cost(request, completion, tier)
        self.ledger.settle(token, request.key, usd, self.label, tier=tier or self.tier)

    def fail(
        self, token: str, request: ChatRequest, error: BaseException, *, tier: str | None = None
    ) -> None:
        """Record a failed attempt and its error class. One treated as unbilled is released at
        $0: HTTP 429 (a rate or quota limit, refused before it ran), and on the flex tier HTTP 503
        (flex may preempt a request; the billing checks of 2026-10-14 and 2026-10-16 found these
        unbilled, docs/decisions.md). Every other
        failure (a timeout, a lost connection, another 5xx, a 503 on the standard tier, another
        4xx) may have been billed and stays open at its upper bound."""
        status = getattr(error, "status_code", None)
        tier = tier or self.tier
        self.ledger.fail(
            token,
            request.key,
            type(error).__name__,
            status,
            released=status == 429 or (status == 503 and tier == "flex"),
            label=self.label,
            tier=tier,
        )

    def complete(self, requests: Sequence[ChatRequest]) -> Iterator[tuple[str, Completion]]:
        yield from self.inner.complete(requests)

    def owner(self, token: str, tier: str) -> dict[str, str]:
        """The attempt an answer belongs to, cached with it."""
        return {"worker": self.ledger.worker, "reservation": token, "tier": tier}

    def reconcile(self, cache: JudgeCache) -> int:
        """Settle the ledger's open reservations whose own answers reached the cache: only an
        answer that names this worker, that reservation and its tier (``owner``)."""

        def cost_of(key: str, tier: str | None, reservation: str) -> float | None:
            hit = cache.get(key)
            if hit is None or billed_tokens(hit.usage) is None or tier is None:
                return None
            if hit.attempt != self.owner(reservation, tier) or hit.service_tier != tier:
                return None  # not verifiably this attempt's answer: the reservation stays
            body = json_body(cache, key)
            request = ChatRequest(key, "summary", "", body)
            return self.cost(request, hit, tier)

        return self.ledger.reconcile(cost_of)


class LedgeredBatch:
    """The API ledger of a batch backend (``judge.GeminiBatchBackend(batch_hook=...)``): every
    request of a batch is reserved at an upper bound of its cost before the batch is submitted,
    and closed when the batch is collected: settled at its reported cost; released at $0 if it
    is absent from the batch's output (never processed) or rejected with HTTP 429; otherwise kept
    open at its bound, since it may have been billed. A reservation already closed is not closed
    again, so a rerun that collects a batch twice changes nothing."""

    def __init__(
        self,
        ledger: ApiLedger,
        model_of: Callable[[ChatRequest], ModelSpec],
        *,
        price_factor: float = 1.0,
        label: str = "adl",
    ) -> None:
        self.ledger, self.model_of = ledger, model_of
        self.price_factor, self.label = price_factor, label

    def bound(self, request: ChatRequest) -> float:
        return max_cost(request, self.model_of(request), price_factor=self.price_factor)

    def fits(self, requests: Sequence[ChatRequest]) -> int:
        """How many of ``requests``, from the first, the ledger can reserve now."""
        room, n = self.ledger.remaining(), 0
        for request in requests:
            room -= self.bound(request)
            if room < 0:
                break
            n += 1
        return n

    def reserve(self, requests: Sequence[ChatRequest]) -> dict[str, str]:
        return self.ledger.reserve({r.key: self.bound(r) for r in requests}, self.label)

    def close(self, token: str, request: Any, completion: Completion, *, absent: bool) -> None:
        if token not in self.ledger.state()["open"]:
            return
        if completion.error is None and billed_tokens(completion.usage) is not None:
            cost = usage_cost([completion], self.model_of(request), price_factor=self.price_factor)
            self.ledger.settle(token, request.key, cost[2], self.label)
            return
        error = completion.error or "no billing usage"
        found = re.match(r"batch status (\d+)", error)
        status = int(found.group(1)) if found else None
        released = absent or status == 429
        self.ledger.fail(token, request.key, "batch", status, released=released, label=self.label)

    def reconcile(self, cache: JudgeCache) -> int:
        """Settle the ledger's open reservations whose completions reached the cache."""

        def cost_of(key: str, tier: str | None, reservation: str) -> float | None:
            hit = cache.get(key)
            if hit is None or billed_tokens(hit.usage) is None:
                return None
            # A batch answer records no ledger attempt, so its ownership cannot be verified: the
            # reservation stays open, counted at its bound (review s3e).
            if not hit.attempt or hit.attempt.get("reservation") != reservation:
                return None
            request = ChatRequest(key, "summary", "", json_body(cache, key))
            return usage_cost([hit], self.model_of(request), price_factor=self.price_factor)[2]

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
