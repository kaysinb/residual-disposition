"""Path B's GPU-time budget (feasibility plan, "Cost and time").

- ``CostMeter``: every GPU step of an orchestration asks it before it starts. From
  ``plan.cost.stop_fraction`` of the cap on (wall time since the Pod's start), and whenever the
  Pod's hard deadline cannot be verified, it raises ``BudgetExhausted``, and nothing further
  starts. ``projected`` checks a dose against the cap before it starts.
- ``verify_hard_deadline``: the Pod's external hard stop, from the state of its deadline watcher
  (the runpod-research skill's ``night_watch.py``): armed, held by a live worker, for this Pod,
  in the future and no later than the end of the cap, so that the cap holds even if this process
  fails.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class BudgetExhausted(RuntimeError):
    """The cost meter allows no further GPU step."""


def pod_id() -> str | None:
    """This Pod's id: ``RUNPOD_POD_ID``, also from the container's start environment, which a
    non-interactive SSH shell may lack."""
    if os.environ.get("RUNPOD_POD_ID"):
        return os.environ["RUNPOD_POD_ID"]
    try:
        items = Path("/proc/1/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    for item in items:
        key, _, value = item.partition(b"=")
        if key == b"RUNPOD_POD_ID" and value:
            return value.decode()
    return None


def _held(lock: Path) -> bool:
    """Whether another open file holds an exclusive lock on ``lock`` (the watcher's worker)."""
    try:
        with lock.open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
            return False
    except OSError:
        return False


def verify_hard_deadline(
    state_path: str | Path, *, not_after: datetime, now: datetime, pod: str | None
) -> datetime:
    """The Pod's hard deadline from its watcher's ``state.json``; ValueError unless the watcher
    is armed (not a dry run), its worker holds ``worker.lock`` beside the state, it stops ``pod``,
    and its deadline is after ``now`` and no later than ``not_after``."""
    path = Path(state_path)
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"no readable deadline watcher state at {path}: {error}") from error
    if state.get("phase") != "armed" or state.get("dry_run"):
        raise ValueError(f"the deadline watcher at {path} is not armed live: {state.get('phase')}")
    deadline = state.get("deadline")
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError(f"the deadline watcher at {path} has no hard deadline")
    at = datetime.fromtimestamp(deadline, UTC)
    if at <= now:
        raise ValueError(f"the hard deadline {at.isoformat()} has passed")
    if at > not_after:
        raise ValueError(
            f"the hard deadline {at.isoformat()} is after the cap's end {not_after.isoformat()}"
        )
    if pod is None or state.get("pod_id") != pod:
        raise ValueError(f"the deadline watcher stops Pod {state.get('pod_id')}, not {pod}")
    if not _held(path.parent / "worker.lock"):
        raise ValueError(f"no live watcher worker holds {path.parent / 'worker.lock'}")
    return at


def parse_start(value: Any) -> datetime:
    """``plan.cost.started_at``: an ISO time with a time zone."""
    if not isinstance(value, str) or not value or value == "???":
        raise ValueError("plan.cost.started_at must be set: the Pod's start, ISO with a time zone")
    started = datetime.fromisoformat(value)
    if started.tzinfo is None:
        raise ValueError(f"plan.cost.started_at {value!r} has no time zone")
    return started


@dataclass
class CostMeter:
    """The GPU-hour cap of one Pod: ``cap_hours`` from ``started`` (the Pod's start). No step
    starts from ``stop_fraction`` of the cap on; with ``deadline_state`` every check also verifies
    the Pod's hard deadline. Without a cap every check passes."""

    cap_hours: float | None = None
    started: datetime | None = None
    stop_fraction: float = 0.8
    deadline_state: str | None = None
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    pod: Callable[[], str | None] = pod_id
    exhausted: str | None = None

    @classmethod
    def from_plan(
        cls,
        plan: Mapping[str, Any],
        *,
        now: Callable[[], datetime] | None = None,
        pod: Callable[[], str | None] | None = None,
    ) -> CostMeter:
        cost = plan.get("cost") or {}
        if cost.get("cap_hours") is None:
            return cls()
        cap = float(cost["cap_hours"])
        fraction = float(cost.get("stop_fraction", 0.8))
        if cap <= 0 or not 0 < fraction <= 1:
            raise ValueError(
                f"plan.cost: cap_hours {cap} > 0 and stop_fraction {fraction} in (0, 1]"
            )
        if not cost.get("deadline_state"):
            raise ValueError("plan.cost.cap_hours needs plan.cost.deadline_state: the watcher")
        meter = cls(
            cap_hours=cap,
            started=parse_start(cost.get("started_at")),
            stop_fraction=fraction,
            deadline_state=str(cost["deadline_state"]),
        )
        if now is not None:
            meter.now = now
        if pod is not None:
            meter.pod = pod
        if meter.started > meter.now():
            raise ValueError(f"plan.cost.started_at {meter.started.isoformat()} is in the future")
        return meter

    @property
    def cap_end(self) -> datetime | None:
        if self.cap_hours is None or self.started is None:
            return None
        return datetime.fromtimestamp(self.started.timestamp() + self.cap_hours * 3600, UTC)

    def elapsed(self) -> float | None:
        if self.started is None:
            return None
        return (self.now() - self.started).total_seconds() / 3600

    def verify_deadline(self) -> datetime | None:
        """The verified hard deadline, or None without a cap; ValueError if it fails."""
        if self.cap_hours is None:
            return None
        return verify_hard_deadline(
            self.deadline_state, not_after=self.cap_end, now=self.now(), pod=self.pod()
        )

    def check(self) -> None:
        """Raise ``BudgetExhausted`` if no further GPU step may start."""
        if self.exhausted:
            raise BudgetExhausted(self.exhausted)
        if self.cap_hours is None:
            return
        elapsed = self.elapsed()
        if elapsed >= self.stop_fraction * self.cap_hours:
            self.exhausted = (
                f"{elapsed:.1f} h of the {self.cap_hours:g} h cap used "
                f"(stop at {self.stop_fraction:.0%})"
            )
        else:
            try:
                self.verify_deadline()
            except ValueError as error:
                self.exhausted = f"hard deadline not verified: {error}"
        if self.exhausted:
            raise BudgetExhausted(self.exhausted)

    def projected(self, hours: float) -> str | None:
        """Why a step of ``hours`` (projected) may not start: it would carry past the cap."""
        self.check()
        if self.cap_hours is None:
            return None
        elapsed = self.elapsed()
        if elapsed + hours > self.cap_hours:
            return f"projected overrun: {elapsed:.1f} h used + {hours:.1f} h > {self.cap_hours:g} h"
        return None
