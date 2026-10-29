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
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .budget import RUNNING, ApiBudgetExhausted, ApiLedger
from .stage3_money import ChainedLedger, _state, meter_of

PREFIX = "follow/"


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
