"""The laptop's API money for a certified lineage's readers' grading (§12.8): the interpreter,
the judge and the category grader of one behavior's lineage, in the order the meter's items
give it.

- **Parts.** The base, the rest of ``stage3/<behavior>/follow``'s API limit once the Pod's λ
  grading is charged to it; then the strong part, ``strong/stage3/<behavior>/follow``'s API
  limit. Each part has a durable ledger of its own (``budget.ApiLedger``, worker
  ``s3r-<behavior>-<part>``), so that no two parts spend one balance.
- **Ceilings.** A part's allowance is its item's remaining API limit when the part is first
  opened, recorded once (``data/ledgers/s3r-<behavior>-<part>.ceiling.json``) and read back
  every later time; a part whose item is not running offers nothing yet.
- **Ownership per request.** Each reservation goes to the base while the base holds it, else to
  the strong part (``ChainedLedger``); one that neither holds is refused before it is sent, and
  the step stops for its price.
- **Charging the meter.** ``charge`` books on each part's item the part's committed money (its
  settled spend and every open reservation, which may be billed) less what the meter's log
  already holds under the part's label: only the increment, so a rerun or a resumed charge
  books nothing twice.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .adl_graders import LedgeredBackend
from .budget import POD, RUNNING, ApiBudgetExhausted, ApiLedger, finite_usd

PARTS = ("base", "strong")
ITEMS = {"base": "stage3/{behavior}/follow", "strong": "strong/stage3/{behavior}/follow"}


def item_of(behavior: str, part: str) -> str:
    return ITEMS[part].format(behavior=behavior)


def worker_of(behavior: str, part: str) -> str:
    return f"s3r-{behavior}-{part}"


def label_of(behavior: str, part: str) -> str:
    return f"the readers' grading on the laptop ({worker_of(behavior, part)})"


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


def ceiling(root: Path, behavior: str, part: str, meter: Any) -> float | None:
    """The part's allowance: recorded once, when its item first runs, as the item's remaining
    API limit then; None while its item does not run and nothing is recorded."""
    path = Path(root) / "ledgers" / f"{worker_of(behavior, part)}.ceiling.json"
    item = item_of(behavior, part)
    if path.is_file():
        data = json.loads(path.read_text())
        if data.get("item") != item:
            raise ValueError(f"{path}: the ceiling of {data.get('item')}, not {item}")
        return finite_usd(data["allowance_usd"], "a recorded ceiling")
    with meter._transaction():  # the state with every instance's events
        running = meter.state.get(item) == RUNNING
    if not running:
        return None
    if meter.session(item, POD) is not None:
        raise ValueError(f"{item} has an open Pod session: its λ grading is not charged yet")
    usd = meter.limits(item).api_usd
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps({"item": item, "allowance_usd": usd}) + "\n"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:  # another writer recorded it first: read that one
        return ceiling(root, behavior, part, meter)
    with os.fdopen(fd, "w") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    return usd


def open_ledgers(root: Path, behavior: str, meter: Any) -> dict[str, ApiLedger]:
    """The writers of the parts that have a positive allowance, in ``PARTS`` order."""
    out = {}
    for part in PARTS:
        usd = ceiling(root, behavior, part, meter)
        if usd is not None and usd > 0:
            worker = worker_of(behavior, part)
            out[part] = ApiLedger(Path(root) / "ledgers" / f"adl-api-{worker}.jsonl", usd, worker)
    if not out:
        raise ApiBudgetExhausted(f"{behavior}: no part of the readers' API money is running")
    return out


class ChainedLedger:
    """The parts' ledgers as one (the interface ``adl_graders.LedgeredBackend`` uses): a
    reservation goes to the first part that holds it whole, the base before the strong part;
    its settlement and failure go to the part that took it."""

    def __init__(self, parts: Mapping[str, ApiLedger]) -> None:
        self.parts = dict(parts)
        self.owner_of: dict[str, ApiLedger] = {}
        self.worker = "+".join(ledger.worker for ledger in self.parts.values())

    @property
    def source(self) -> str | None:
        return next(iter(self.parts.values())).source

    @source.setter
    def source(self, value: str | None) -> None:  # every event names the run that writes it
        for ledger in self.parts.values():
            ledger.source = value

    def reserve(self, reservations: Mapping[str, float], label: str, *,
                tier: str | None = None) -> dict[str, str]:  # fmt: skip
        total = sum(reservations.values())
        for ledger in self.parts.values():
            if ledger.remaining() + 1e-12 < total:
                continue
            try:
                ids = ledger.reserve(reservations, label, tier=tier)
            except ApiBudgetExhausted:
                continue
            for token in ids.values():
                self.owner_of[token] = ledger
            return ids
        raise ApiBudgetExhausted(
            f"the readers' API ({self.worker}): no part holds ${total:.4f} more; "
            f"${self.remaining():.4f} left in all: stop and price"
        )

    def settle(self, token: str, key: str, usd: float, label: str, *,
               tier: str | None = None) -> None:  # fmt: skip
        self.owner_of[token].settle(token, key, usd, label, tier=tier)

    def fail(self, token: str, key: str, error: str, status: int | None, *, released: bool,
             label: str, tier: str | None = None) -> None:  # fmt: skip
        self.owner_of[token].fail(token, key, error, status, released=released, label=label,
                                  tier=tier)  # fmt: skip

    def remaining(self) -> float:
        return sum(ledger.remaining() for ledger in self.parts.values())

    def committed(self) -> float:
        return sum(ledger.committed() for ledger in self.parts.values())

    def check(self, estimate_usd: float = 0.0) -> None:
        if estimate_usd > self.remaining():
            raise ApiBudgetExhausted(
                f"the readers' API ({self.worker}): ${estimate_usd:.2f} exceeds the "
                f"${self.remaining():.2f} left: stop and price"
            )

    def close(self) -> None:
        for ledger in self.parts.values():
            ledger.close()


class ChainedBackend(LedgeredBackend):
    """``LedgeredBackend`` on a ``ChainedLedger``: each answer is cached with the part that took
    its reservation, and an interrupted run's open reservations are reconciled part by part."""

    def owner(self, token: str, tier: str) -> dict[str, str]:
        return {"worker": self.ledger.owner_of[token].worker, "reservation": token, "tier": tier}

    def reconcile(self, cache: Any) -> int:
        settled = 0
        chained = self.ledger
        for ledger in chained.parts.values():
            part = LedgeredBackend.__new__(LedgeredBackend)
            part.__dict__.update({**self.__dict__, "ledger": ledger})
            settled += LedgeredBackend.reconcile(part, cache)
        self.ledger = chained
        return settled


def _charged(log: Path, item: str, label: str) -> float:
    """What the meter's log holds as API charged to ``item`` under ``label``."""
    total = 0.0
    if not Path(log).is_file():
        return total
    for line in Path(log).read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "charge" and event.get("item") == item \
                and event.get("label") == label:  # fmt: skip
            total += float(event.get("api_usd") or 0.0)
    return total


def charge(meter: Any, log: Path, behavior: str, parts: Mapping[str, ApiLedger]) -> dict:
    """Book each part's uncharged committed money on its item (module docstring); returns
    {part: the USD booked now}."""
    out = {}
    for part, ledger in parts.items():
        item, label = item_of(behavior, part), label_of(behavior, part)
        increment = round(ledger.committed() - _charged(log, item, label), 9)
        if increment > 0:
            meter.charge(item, api_usd=increment, label=label)
        out[part] = max(0.0, increment)
    return out


def money(root: Path, behavior: str, log: Path) -> tuple[Any, ChainedLedger]:
    """The meter and the chained ledger of a lineage's grading."""
    meter = meter_of(log)
    return meter, ChainedLedger(open_ledgers(root, behavior, meter))


def charge_all(root: Path, behavior: str, log: Path) -> dict:
    """``charge`` over every part recorded so far (a rerun charges nothing twice)."""
    meter = meter_of(log)
    parts = {}
    for part in PARTS:
        usd = ceiling(root, behavior, part, meter)
        if usd is not None and usd > 0:
            worker = worker_of(behavior, part)
            parts[part] = ApiLedger(Path(root) / "ledgers" / f"adl-api-{worker}.jsonl", usd,
                                    worker)  # fmt: skip
    try:
        return charge(meter, log, behavior, parts)
    finally:
        for ledger in parts.values():
            ledger.close()
