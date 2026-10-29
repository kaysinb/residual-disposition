"""The laptop's API money of the follow-up's items (preregistration §13.14, the meter of §12.8):
each item's API spending through its own durable ledger and the meter's laptop API session.

- **One part per item.** An item spends only its own API part: a ledger of its own
  (``budget.ApiLedger``, worker ``follow-<item>``), no chained parts.
- **Sessions.** A command opens the meter's laptop API session on the item
  (``AllocationMeter.open_api_session``): authorized for the item's API left, alone on the item,
  recorded before anything is sent; the ledger records the session's ceiling
  (``ApiLedger.session_ceiling``) and is its allowance. A session left open by an interrupted
  command is settled first, at its ledger's committed cost.
- **Refusals.** An item that is not running opens no session; a reservation is refused once the
  item's session is no longer the meter's open one on a running item, before anything is sent.
- **Settling.** At the end the session is closed at its committed cost (its settled spend and every
  open reservation, which may be billed; ``ApiLedger.session_cost``).

Pod work on an item uses the meter's Pod sessions (``session-open`` / ``session-close``), as before.

**Rungs 2 and 3** have a shared item each (``SHARED``): the rung's once-only A/H qualification and
format validation, used by whichever mode reaches the rung first. A rung item's work starts only
while both its own reservation and the shared one are held, or the shared work is complete
(``require_reservations``): two admissions are not atomic. The shared item is dropped only once
neither mode can still reach its rung (``shared_droppable``).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .budget import COMPLETED, DROPPED, RESERVED, RUNNING, ApiBudgetExhausted, ApiLedger
from .stage3_money import ChainedLedger, _state, meter_of

PREFIX = "follow/"
SHARED = {"rung2": "follow/rung2_shared", "rung3": "follow/rung3_shared"}
RUNG_ITEMS = {"rung2": ("follow/continued/rung2", "follow/stack/rung2"),
              "rung3": ("follow/continued/rung3", "follow/stack/rung3")}  # fmt: skip


def _slug(item: str) -> str:
    if not item.startswith(PREFIX):
        raise ValueError(f"{item} is not a follow-up item")
    return item.removeprefix(PREFIX).replace("/", "-")


def ledger_path(root: Path, item: str) -> Path:
    return root / "ledgers" / "follow" / f"{_slug(item)}.jsonl"


def worker_of(item: str) -> str:
    return f"follow-{_slug(item)}"


def settle(meter: Any, root: Path, item: str) -> float | None:
    """Close the item's laptop session left open by an interrupted command, at its ledger's
    committed cost since it opened; None if none was open."""
    held = meter.laptop_session(item)
    if held is None:
        return None
    ledger = ApiLedger(ledger_path(root, item), 1.0, worker_of(item))
    try:
        cost = ledger.session_cost(held["session"])
    finally:
        ledger.close()
    meter.close_api_session(item, held["session"], cost)
    return cost


class ItemPart:
    """One item's session: its item, session id and ledger (its allowance the session's
    ceiling); the interface ``stage3_money.ChainedLedger`` reads."""

    def __init__(self, meter: Any, root: Path, item: str) -> None:
        state = _state(meter, item)
        if state != RUNNING:
            raise ApiBudgetExhausted(f"{item} is {state}, not running")
        self.meter, self.item = meter, item
        api = meter.limits(item).api_usd
        if api <= 0:
            raise ApiBudgetExhausted(f"{item} has no API left")
        self.session = uuid.uuid4().hex
        # The ledger's writer and the session's ceiling first, then the meter's session: an
        # interruption between them leaves a ceiling no session uses, never a session without
        # its ceiling. A refused session closes the writer again.
        self.ledger = ApiLedger(ledger_path(root, item), 1.0, worker_of(item))
        try:
            self.ledger.allowance_usd = self.ledger.session_ceiling(self.session, api)
            meter.open_api_session(item, self.session, api)
        except BaseException:
            self.ledger.close()
            raise

    def authorized(self) -> bool:
        """The session is still the meter's open one on the running item."""
        held = self.meter.laptop_session(self.item)
        return (held or {}).get("session") == self.session and (
            _state(self.meter, self.item) == RUNNING
        )

    def close(self) -> float:
        """Settle the session at its committed cost (the item running or stopped)."""
        cost = self.ledger.session_cost(self.session)
        if (self.meter.laptop_session(self.item) or {}).get("session") == self.session:
            self.meter.close_api_session(self.item, self.session, cost)
        self.ledger.close()
        return cost


def money(root: Path, item: str, log: Path) -> tuple[Any, ChainedLedger]:
    """The meter and the item's ledger, as ``ChainedLedger``'s one part: its session opened
    after settling any left open."""
    meter = meter_of(log)
    settle(meter, root, item)
    return meter, ChainedLedger({item: ItemPart(meter, root, item)})


def require_reservations(meter: Any, rung_item: str) -> None:
    """Refuse a rung item's work unless it is reserved or running and its shared item is
    reserved, running or complete."""
    rung = next((r for r, items in RUNG_ITEMS.items() if rung_item in items), None)
    if rung is None:
        raise ValueError(f"{rung_item} is not a rung item")
    own, shared = _state(meter, rung_item), _state(meter, SHARED[rung])
    if own not in (RESERVED, RUNNING) or shared not in (RESERVED, RUNNING, COMPLETED):
        raise ApiBudgetExhausted(f"{rung_item} is {own} and {SHARED[rung]} {shared}: both "
                                 "reservations are held before the rung's work starts")  # fmt: skip


def shared_droppable(meter: Any, shared_item: str, stack_closed: bool) -> bool:
    """Whether no mode can still need ``shared_item``: every mode's item of that rung dropped or
    complete, the stack's counted as closed when its branch closed (``stack_closed``)."""
    rung = next((r for r, s in SHARED.items() if s == shared_item), None)
    if rung is None:
        raise ValueError(f"{shared_item} is not a shared item")
    done = (DROPPED, COMPLETED)
    continued, stack = RUNG_ITEMS[rung]
    return _state(meter, continued) in done and (stack_closed or _state(meter, stack) in done)
