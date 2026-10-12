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
import threading
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


# Path B's GPU batches, each with its own cap (plan, caps): Q1, ADL's qualification ($45), and the
# rest ($180), Q2-Q3 included. The phase decides the batch.
BATCHES = {
    "qualify": "q1",
    "validate_axis": "rest",
    "suppress": "rest",
    "calibrate": "rest",
    "certify": "rest",
}


@dataclass
class CostMeter:
    """The GPU-hour cap of one Pod in one batch: ``cap_hours``, of which ``prior_hours`` were
    used by this batch on this Pod before ``started`` (this pass's start), the rest counted as
    wall time from ``started``. No step starts from ``stop_fraction`` of the cap on; with
    ``deadline_state`` every check also verifies the Pod's hard deadline, which must fall no
    later than the cap's end. Without a cap every check passes."""

    cap_hours: float | None = None
    started: datetime | None = None
    stop_fraction: float = 0.8
    deadline_state: str | None = None
    prior_hours: float = 0.0
    batch: str | None = None
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
        phase = plan.get("phase", "all")
        batch, prior = BATCHES.get(phase), 0.0
        if batch is not None:
            cap = (cost.get("batches") or {}).get(batch)
            if cap is None or cap == "???":
                raise ValueError(f"phase {phase} needs its batch's cap, plan.cost.batches.{batch}")
            prior = cost.get("prior_hours")
            if not isinstance(prior, int | float) or isinstance(prior, bool) or prior < 0:
                raise ValueError(
                    f"plan.cost.prior_hours {prior!r}: this batch's GPU hours on this Pod before "
                    "started_at, a number >= 0"
                )
        elif cost.get("cap_hours") is None:
            return cls()
        else:
            cap = cost["cap_hours"]
        cap = float(cap)
        fraction = float(cost.get("stop_fraction", 0.8))
        if cap <= 0 or not 0 < fraction <= 1:
            raise ValueError(
                f"plan.cost: cap_hours {cap} > 0 and stop_fraction {fraction} in (0, 1]"
            )
        if not cost.get("deadline_state"):
            raise ValueError("plan.cost.cap_hours needs plan.cost.deadline_state: the watcher")
        if prior >= cap:
            raise ValueError(f"batch {batch}: {prior:g} h already used of its {cap:g} h cap")
        meter = cls(
            cap_hours=cap,
            started=parse_start(cost.get("started_at")),
            stop_fraction=fraction,
            deadline_state=str(cost["deadline_state"]),
            prior_hours=float(prior),
            batch=batch,
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
        """When the cap runs out: what is left of it after ``prior_hours``, from ``started``."""
        if self.cap_hours is None or self.started is None:
            return None
        left = self.cap_hours - self.prior_hours
        return datetime.fromtimestamp(self.started.timestamp() + left * 3600, UTC)

    def elapsed(self) -> float | None:
        """This batch's GPU hours on this Pod: ``prior_hours`` and the wall time since start."""
        if self.started is None:
            return None
        return self.prior_hours + (self.now() - self.started).total_seconds() / 3600

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

    def projected_stop(self, hours: float) -> str | None:
        """Why work of ``hours`` (forecast) may not go on: it would carry past the stop at
        ``stop_fraction`` of the cap, where no step starts (Q1's benchmark)."""
        self.check()
        if self.cap_hours is None:
            return None
        elapsed, stop = self.elapsed(), self.stop_fraction * self.cap_hours
        if elapsed + hours > stop:
            return (
                f"forecast overrun: {elapsed:.1f} h used + {hours:.1f} h > the stop at "
                f"{stop:.2f} h ({self.stop_fraction:.0%} of {self.cap_hours:g} h)"
            )
        return None


class ApiBudgetExhausted(BudgetExhausted):
    """The ADL API ledger allows no further request."""


@dataclass
class ApiLedger:
    """One worker's share of ADL's API cap (path-B plan, caps): disjoint allowances that sum to
    the cap ($35: each Pod's graders, and the laptop's interpreter, ablation and judge), so no two
    workers can spend the same balance.

    The ledger is a durable append-only file of events, one JSON object per line:
    - ``reserve`` before each actual send of a request, at an upper bound of its cost, with an id
      of its own;
    - ``spend`` when that send's completion arrives, with the reported cost, which closes that
      reservation and no other;
    - a reservation left open (an interrupted or failed send) stays counted, also when a later
      send of the same request succeeds.

    ``committed`` is the settled spend plus every open reservation; a request is refused before
    it is sent if it would carry ``committed`` past ``allowance_usd``. A cached repeat sends and
    reserves nothing, and what was spent stays in the file, so resuming never zeroes it.
    ``reconcile`` settles an open reservation whose completion reached the cache when its request
    was never settled at all."""

    path: Path
    allowance_usd: float
    worker: str
    # One process writes a worker's ledger; its state is read from the file once and then kept
    # in memory, each event applied as it is appended, under a lock for the request threads.
    _cache: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.allowance_usd > 0:
            raise ValueError(f"ADL API allowance {self.allowance_usd} must be positive")

    def _events(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        events = [json.loads(line) for line in self.path.read_text().splitlines() if line]
        others = {e["worker"] for e in events} - {self.worker}
        if others:
            raise ValueError(f"ledger {self.path} holds other workers' events: {sorted(others)}")
        return events

    def _append(self, events: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).isoformat()
        with self.path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            for event in events:
                handle.write(json.dumps({**event, "worker": self.worker, "at": stamp}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle, fcntl.LOCK_UN)
        for event in events:
            self._apply(self._loaded(), event)

    @staticmethod
    def _apply(cache: dict[str, Any], event: Mapping[str, Any]) -> None:
        if event["event"] == "reserve":
            cache["open"][event["id"]] = (event["key"], float(event["usd"]))
        elif event["event"] == "spend":
            cache["spent"] += float(event["usd"])
            cache["open"].pop(event["id"], None)
            cache["settled_keys"].add(event["key"])

    def _loaded(self) -> dict[str, Any]:
        if self._cache is None:
            cache: dict[str, Any] = {"spent": 0.0, "open": {}, "settled_keys": set()}
            for event in self._events():
                self._apply(cache, event)
            self._cache = cache
        return self._cache

    def state(self) -> dict[str, Any]:
        """{"spent": settled USD, "open": {reservation id: (key, reserved USD)}, "settled_keys",
        "committed": spent plus the open reservations}."""
        with self._lock:
            cache = self._loaded()
            open_ = dict(cache["open"])
            committed = cache["spent"] + sum(usd for _, usd in open_.values())
            return {
                "spent": cache["spent"],
                "open": open_,
                "settled_keys": set(cache["settled_keys"]),
                "committed": committed,
            }

    def committed(self) -> float:
        return self.state()["committed"]

    def remaining(self) -> float:
        return self.allowance_usd - self.committed()

    def check(self, estimate_usd: float = 0.0) -> None:
        """Raise unless ``estimate_usd`` more fits in the allowance (before each ADL state)."""
        committed = self.committed()
        if committed + estimate_usd > self.allowance_usd:
            raise ApiBudgetExhausted(
                f"ADL API ({self.worker}): ${committed:.2f} committed + ${estimate_usd:.2f} "
                f"exceeds the ${self.allowance_usd:g} allowance"
            )

    def reserve(self, reservations: Mapping[str, float], label: str) -> dict[str, str]:
        """Reserve every request of a batch before any is sent, or none; returns each request's
        reservation id, for its ``settle``."""
        import uuid

        total = sum(reservations.values())
        with self._lock:  # the check and the reservation together, for concurrent requests
            cache = self._loaded()
            committed = cache["spent"] + sum(usd for _, usd in cache["open"].values())
            if committed + total > self.allowance_usd:
                raise ApiBudgetExhausted(
                    f"ADL API ({self.worker}): ${committed:.2f} committed + ${total:.2f} "
                    f"exceeds the ${self.allowance_usd:g} allowance"
                )
            ids = {key: uuid.uuid4().hex for key in reservations}
            events = [
                {"event": "reserve", "id": ids[key], "key": key, "usd": float(usd), "label": label}
                for key, usd in reservations.items()
            ]
            self._append(events)
        return ids

    def settle(self, reservation: str, key: str, usd: float, label: str) -> None:
        """Close one reservation at the cost its send reported."""
        event = {"event": "spend", "id": reservation, "key": key, "usd": float(usd)}
        with self._lock:
            self._append([{**event, "label": label}])

    def reconcile(self, cost_of_cached: Callable[[str], float | None]) -> int:
        """Settle, at its cached completion's reported cost (``cost_of_cached``, None if
        absent), the latest open reservation of each request that was never settled: its send
        completed and was cached, but the process stopped before settling. Earlier open sends of
        the same request stay counted. Returns how many were settled."""
        state = self.state()
        latest: dict[str, str] = {}
        for reservation, (key, _) in state["open"].items():  # file order: the last one wins
            if key not in state["settled_keys"]:
                latest[key] = reservation
        settled = 0
        for key, reservation in latest.items():
            usd = cost_of_cached(key)
            if usd is not None:
                self.settle(reservation, key, usd, "reconciled from the cache")
                settled += 1
        return settled


def ledger_from(spec: Mapping[str, Any], root: Path) -> ApiLedger:
    """The ledger of ``spec`` ({worker, allowance_usd, path or None}); its default path is
    ``root/ledgers/adl-api-<worker>.jsonl``, on the worker's persistent volume."""
    worker = spec.get("worker")
    if not worker or worker == "???":
        raise ValueError("the ADL API ledger needs its worker: pod-a, pod-b or laptop")
    allowance = spec.get("allowance_usd")
    if allowance is None or allowance == "???":
        raise ValueError(f"the ADL API ledger of {worker} needs allowance_usd")
    path = spec.get("path") or Path(root) / "ledgers" / f"adl-api-{worker}.jsonl"
    return ApiLedger(Path(path), float(allowance), str(worker))
