"""The laptop's API money for a certified lineage's readers' grading (§12.8): the interpreter,
the judge and the category grader of one behavior's lineage, in the order the meter's items
give it.

- **Parts.** The base, the rest of ``stage3/<behavior>/follow``'s API limit once the Pod's λ
  grading is charged to it; then the strong part, ``strong/stage3/<behavior>/follow``'s API
  limit, once its item is started (``start_strong``). Each part has a durable ledger of its own
  (``budget.ApiLedger``, worker ``s3r-<behavior>-<part>``), so that no two parts spend one
  balance.
- **Sessions.** A command opens, on each running part's item, the meter's laptop API session
  (``AllocationMeter.open_api_session``): authorized for the item's API left, alone on the item,
  recorded before anything is sent; the part's ledger records the session's ceiling
  (``ApiLedger.session_ceiling``) and is its allowance. A session left open by an interrupted
  command is settled first, at its ledger's committed cost; an item stopped meanwhile is settled
  too. An item never completes with a session open.
- **Ownership per request.** Each reservation goes to the base while the base holds it, else to
  the strong part (``ChainedLedger``), after checking that the part's session is still the
  meter's open one on a running item; one that no part holds is refused before it is sent, and
  the step stops for its price.
- **Settling.** At the end each session is closed at its committed cost (its settled spend and
  every open reservation, which may be billed; ``ApiLedger.session_cost``).
- **Funding.** The behavior a command spends for is the lineage's that its inputs name; another
  is refused before any ledger opens (``check_funding``).
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .adl_graders import LedgeredBackend
from .budget import RESERVED, RUNNING, ApiBudgetExhausted, ApiLedger, finite_usd

PARTS = ("base", "strong")
ITEMS = {"base": "stage3/{behavior}/follow", "strong": "strong/stage3/{behavior}/follow"}


def item_of(behavior: str, part: str) -> str:
    return ITEMS[part].format(behavior=behavior)


def worker_of(behavior: str, part: str) -> str:
    return f"s3r-{behavior}-{part}"


def ledger_path(root: Path, behavior: str, part: str) -> Path:
    return Path(root) / "ledgers" / f"adl-api-{worker_of(behavior, part)}.jsonl"


def meter_of(log: Path) -> Any:
    """The continuation's meter over ``log``, as ``scripts/continuation_meter.py`` builds it."""
    from .budget import AllocationMeter, load_allocations, load_migrations
    from .continuation_rules import drop_reasons, load_rules
    from .paths import CONFIG_ROOT
    from .simulate_continuation import FROZEN

    allocations = CONFIG_ROOT / "continuation" / "allocations.yaml"
    rules = load_rules(expected_hash=FROZEN["rules_sha256"])
    items = load_allocations(allocations)
    return AllocationMeter(items, cap=float(rules.data["meter"]["cap_usd"]),
                           reserve=float(rules.data["meter"]["reserve_usd"]),
                           drop_reasons=drop_reasons(rules, [a.item for a in items]),
                           log_path=log, migrations=load_migrations(allocations))  # fmt: skip


def check_funding(behavior: str, lineage_behavior: str) -> None:
    """A command spends only for the lineage its inputs name."""
    if behavior != lineage_behavior:
        raise SystemExit(f"the readers of {lineage_behavior} are not paid from {behavior}'s money")


def _state(meter: Any, item: str) -> str:
    with meter._transaction():  # the state with every instance's events
        return meter.state[item]


def settle(meter: Any, root: Path, behavior: str) -> dict[str, float]:
    """Close every part's laptop session left open (an interrupted command), at its ledger's
    committed cost since it opened, the item running or stopped. Returns {part: the USD
    settled}."""
    out = {}
    for part in PARTS:
        item = item_of(behavior, part)
        held = meter.laptop_session(item)
        if held is None:
            continue
        worker = worker_of(behavior, part)
        ledger = ApiLedger(ledger_path(root, behavior, part), 1.0, worker)
        try:
            cost = ledger.session_cost(held["session"])
        finally:
            ledger.close()
        meter.close_api_session(item, held["session"], cost)
        out[part] = cost
    return out


class Part:
    """One part's session: its item, session id and ledger (its allowance the session's
    ceiling)."""

    def __init__(self, meter: Any, root: Path, behavior: str, part: str) -> None:
        self.meter, self.item, self.part = meter, item_of(behavior, part), part
        api = meter.limits(self.item).api_usd
        if api <= 0:
            raise ApiBudgetExhausted(f"{self.item} has no API left")
        self.session = uuid.uuid4().hex
        # The ledger's writer and the session's ceiling first, then the meter's session: an
        # interruption between them leaves a ceiling no session uses, never a session without
        # its ceiling. A refused session closes the writer again.
        self.ledger = ApiLedger(ledger_path(root, behavior, part), 1.0, worker_of(behavior, part))
        try:
            self.ledger.allowance_usd = self.ledger.session_ceiling(self.session, api)
            meter.open_api_session(self.item, self.session, api)
        except BaseException:
            self.ledger.close()
            raise

    def authorized(self) -> bool:
        """The part's session is still the meter's open one on its running item."""
        held = self.meter.laptop_session(self.item)
        return (held or {}).get("session") == self.session and _state(
            self.meter, self.item) == RUNNING  # fmt: skip

    def close(self) -> float:
        """Settle the session at its committed cost (the item running or stopped)."""
        cost = self.ledger.session_cost(self.session)
        if (self.meter.laptop_session(self.item) or {}).get("session") == self.session:
            self.meter.close_api_session(self.item, self.session, cost)
        self.ledger.close()
        return cost


def open_parts(meter: Any, root: Path, behavior: str) -> dict[str, Part]:
    """Every running part's session, in ``PARTS`` order, after settling those left open."""
    settle(meter, root, behavior)
    out = {}
    for part in PARTS:
        if _state(meter, item_of(behavior, part)) == RUNNING:
            try:
                out[part] = Part(meter, root, behavior, part)
            except ApiBudgetExhausted:
                continue
    if not out:
        raise ApiBudgetExhausted(f"{behavior}: no part of the readers' API money is running")
    return out


class ChainedLedger:
    """The parts' ledgers as one (the interface ``adl_graders.LedgeredBackend`` uses): a
    reservation goes to the first part whose session is still authorized and that holds it
    whole, the base before the strong part; its settlement and failure go to the part that took
    it. ``close`` settles every session."""

    def __init__(self, parts: Mapping[str, Part]) -> None:
        self.parts = dict(parts)
        self.owner_of: dict[str, ApiLedger] = {}
        self.worker = "+".join(p.ledger.worker for p in self.parts.values())

    @property
    def source(self) -> str | None:
        return next(iter(self.parts.values())).ledger.source

    @source.setter
    def source(self, value: str | None) -> None:  # every event names the run that writes it
        for p in self.parts.values():
            p.ledger.source = value

    def reserve(self, reservations: Mapping[str, float], label: str, *,
                tier: str | None = None) -> dict[str, str]:  # fmt: skip
        total = sum(reservations.values())
        for p in self.parts.values():
            if p.ledger.remaining() + 1e-12 < total or not p.authorized():
                continue
            try:
                ids = p.ledger.reserve(reservations, label, tier=tier)
            except ApiBudgetExhausted:
                continue
            for token in ids.values():
                self.owner_of[token] = p.ledger
            return ids
        raise ApiBudgetExhausted(
            f"the readers' API ({self.worker}): no authorized part holds ${total:.4f} more; "
            f"${self.remaining():.4f} left in all: stop and price (start_strong when the base "
            "is short)"
        )

    def settle(self, token: str, key: str, usd: float, label: str, *,
               tier: str | None = None) -> None:  # fmt: skip
        self.owner_of[token].settle(token, key, usd, label, tier=tier)

    def fail(self, token: str, key: str, error: str, status: int | None, *, released: bool,
             label: str, tier: str | None = None) -> None:  # fmt: skip
        self.owner_of[token].fail(token, key, error, status, released=released, label=label,
                                  tier=tier)  # fmt: skip

    def remaining(self) -> float:
        return sum(p.ledger.remaining() for p in self.parts.values())

    def committed(self) -> float:
        return sum(p.ledger.committed() for p in self.parts.values())

    def check(self, estimate_usd: float = 0.0) -> None:
        if estimate_usd > self.remaining():
            raise ApiBudgetExhausted(
                f"the readers' API ({self.worker}): ${estimate_usd:.2f} exceeds the "
                f"${self.remaining():.2f} left: stop and price (start_strong when the base is "
                "short)"
            )

    def close(self) -> dict[str, float]:
        """Settle every part's session, once; returns {part: its committed cost}."""
        if not hasattr(self, "_settled"):
            self._settled = {name: p.close() for name, p in self.parts.items()}
        return self._settled


class ChainedBackend(LedgeredBackend):
    """``LedgeredBackend`` on a ``ChainedLedger``: each answer is cached with the part that took
    its reservation, and an interrupted run's open reservations are reconciled part by part."""

    def owner(self, token: str, tier: str) -> dict[str, str]:
        return {"worker": self.ledger.owner_of[token].worker, "reservation": token, "tier": tier}

    def reconcile(self, cache: Any) -> int:
        settled = 0
        chained = self.ledger
        for p in chained.parts.values():
            part = LedgeredBackend.__new__(LedgeredBackend)
            part.__dict__.update({**self.__dict__, "ledger": p.ledger})
            settled += LedgeredBackend.reconcile(part, cache)
        self.ledger = chained
        return settled


def money(root: Path, behavior: str, log: Path) -> tuple[Any, ChainedLedger]:
    """The meter and the chained ledger of a lineage's grading: every running part's session
    opened (after settling any left open)."""
    meter = meter_of(log)
    return meter, ChainedLedger(open_parts(meter, root, behavior))


def start_strong(meter: Any, root: Path, behavior: str, need_usd: float) -> dict[str, Any]:
    """Start ``strong/stage3/<behavior>/follow`` through the meter when the base cannot hold the
    next batch (``need_usd``, its estimate with headroom): the base's API left below the need,
    and the strong item admitted (reserved), never before. Returns the strong item's limits."""
    need_usd = finite_usd(need_usd, "the next batch's need", positive=True)
    base, strong = item_of(behavior, "base"), item_of(behavior, "strong")
    if meter.laptop_session(base) is not None:
        raise SystemExit(f"{base} has an open laptop session: settle it first")
    left = meter.limits(base).api_usd if _state(meter, base) == RUNNING else 0.0
    if left >= need_usd:
        raise SystemExit(f"{base} holds ${left:.4f} ≥ ${need_usd:.4f}: the strong part is not "
                         "needed yet")  # fmt: skip
    if _state(meter, strong) != RESERVED:
        raise SystemExit(f"{strong} is {_state(meter, strong)}: admit it first, in its order")
    limits = meter.start(strong)
    return {"item": strong, "base_left": left, "need": need_usd, "api_usd": limits.api_usd}
