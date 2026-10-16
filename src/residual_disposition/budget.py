"""Path B's GPU-time budget (feasibility plan, "Cost and time").

- ``CostMeter``: every GPU step of an orchestration asks it before it starts. From
  ``plan.cost.stop_fraction`` of the cap on (wall time since the Pod's start), and whenever the
  Pod's hard deadline cannot be verified, it raises ``BudgetExhausted``, and nothing further
  starts. ``projected`` checks a dose against the cap before it starts.
- ``verify_hard_deadline``: the Pod's external hard stop, from the state of its deadline watcher
  (the runpod-research skill's ``night_watch.py``): armed, held by a live worker, for this Pod,
  in the future and no later than the end of the cap, so that the cap holds even if this process
  fails.
- ``AllocationMeter`` and ``apply_cut_order``: the continuation's dollar meter (preregistration
  §12.8). Before each plan item it reserves the complete next batch and every unfinished item of
  higher priority, at frozen allocations, beside spent funds, commitments, future storage and the
  reserve; ``apply_cut_order`` gives the preset cuts a projection needs.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import threading
from collections.abc import Callable, Mapping, Sequence
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


# Path B's GPU batches, each with its own cap (plan, caps): Q1, ADL's qualification ($45), the
# rest ($180), Q2-Q3 included, and the Y control ($35); then the continuation's. The phase decides
# the batch.
BATCHES = {
    "qualify": "q1",
    "validate_axis": "rest",
    "suppress": "rest",
    "calibrate": "rest",
    "certify": "rest",
    "control_y": "y",
    # The continuation (amendment §12): Stage 1 and the engine check, each capped in hours from
    # the meter's GPU limit at the Pod's price.
    "reread": "stage1",
    "engine": "engine",
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


class ApiLedgerBusy(ValueError):
    """Another writer holds this worker's ADL API ledger."""


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
    - ``failed`` when a send raises, with its error class and HTTP status: it releases its
      reservation only if the send is treated as unbilled (``released``: HTTP 429, or HTTP 503
      on the flex tier; ``adl_graders.LedgeredBackend.fail``);
    - ``unbilled`` closes named open reservations at $0 on billing evidence that their sends
      were not charged (``close_unbilled``), with its source and the decision;
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
    readonly: bool = False  # a reader (scripts/adl_api_spend.py): it never appends
    # The run that writes (its run id, from 2026-10-16): recorded in every event it appends, so
    # a spend names its source exactly (residual_disposition.summarizer_tiers).
    source: str | None = None
    # One writer per worker's ledger, for its whole life: an exclusive flock on the ledger's lock
    # file, so a second writer, in this process or another, fails at once instead of spending
    # the same balance. Its state is read from the file once and then kept in memory, each event
    # applied as it is appended, under a lock for the request threads.
    _cache: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False, compare=False)
    _writer: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.allowance_usd > 0:
            raise ValueError(f"ADL API allowance {self.allowance_usd} must be positive")
        if self.readonly:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise ApiLedgerBusy(
                f"ADL API ledger {self.path} already has a writer: one process per worker"
            ) from error
        self._writer = handle

    def close(self) -> None:
        """Release the writer's lock (also at the process's exit)."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _writable(self) -> None:
        if self._writer is None:
            raise ValueError(f"ADL API ledger {self.path} is not open for writing")

    def _events(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        events = [json.loads(line) for line in self.path.read_text().splitlines() if line]
        others = {e["worker"] for e in events} - {self.worker}
        if others:
            raise ValueError(f"ledger {self.path} holds other workers' events: {sorted(others)}")
        return events

    def _append(self, events: list[dict[str, Any]]) -> None:
        self._writable()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).isoformat()
        with self.path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            for event in events:
                tag = {} if self.source is None else {"run": self.source}
                record = {**event, "worker": self.worker, **tag, "at": stamp}
                handle.write(json.dumps(record) + "\n")
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
        elif event["event"] == "failed" and event.get("released"):
            cache["open"].pop(event["id"], None)  # treated as unbilled: nothing counted
        elif event["event"] == "unbilled":
            for reservation in event["ids"]:
                cache["open"].pop(reservation, None)

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

    def fail(
        self,
        reservation: str,
        key: str,
        error: str,
        status: int | None,
        *,
        released: bool,
        label: str,
    ) -> None:
        """Record a failed send of one reservation, with its error class and HTTP status;
        ``released`` (a send treated as unbilled: HTTP 429, or 503 on flex) closes the
        reservation at $0, otherwise it stays open and counted."""
        event = {
            "event": "failed",
            "id": reservation,
            "key": key,
            "error": error,
            "status": status,
            "released": bool(released),
        }
        with self._lock:
            self._append([{**event, "label": label}])

    def close_unbilled(
        self, reservations: Sequence[str], *, source: str, decision: str, label: str
    ) -> float:
        """Close open reservations at $0 on billing evidence that their sends were not charged:
        one ``unbilled`` event naming them, the evidence (``source``) and who decided. Each must
        be open; returns the USD they held."""
        with self._lock:
            cache = self._loaded()
            unknown = [r for r in reservations if r not in cache["open"]]
            if unknown or not reservations:
                raise ValueError(f"not open reservations: {unknown[:3] or 'none given'}")
            held = sum(cache["open"][r][1] for r in reservations)
            event = {
                "event": "unbilled",
                "ids": list(reservations),
                "usd": 0.0,
                "held_usd": held,
                "source": source,
                "decision": decision,
                "label": label,
            }
            self._append([event])
        return held

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


def check_ledger_spec(spec: Mapping[str, Any]) -> None:
    """ValueError unless ``spec`` names a worker and its allowance (nothing is opened)."""
    worker = spec.get("worker")
    if not worker or worker == "???":
        raise ValueError("the ADL API ledger needs its worker: pod-a, pod-b or laptop")
    allowance = spec.get("allowance_usd")
    if allowance is None or allowance == "???":
        raise ValueError(f"the ADL API ledger of {worker} needs allowance_usd")


def ledger_from(spec: Mapping[str, Any], root: Path) -> ApiLedger:
    """The writer of the ledger of ``spec`` ({worker, allowance_usd, path or None}); its default
    path is ``root/ledgers/adl-api-<worker>.jsonl``, on the worker's persistent volume."""
    check_ledger_spec(spec)
    worker = str(spec["worker"])
    path = spec.get("path") or Path(root) / "ledgers" / f"adl-api-{worker}.jsonl"
    return ApiLedger(Path(path), float(spec["allowance_usd"]), worker)


# ---------------------------------------------------------------------------------------------
# The continuation's meter (preregistration §12.8)


@dataclass(frozen=True)
class MeterDecision:
    """Whether a batch may start, with the total it was checked against and its parts."""

    allowed: bool
    total: float
    cap: float
    terms: dict[str, float]


@dataclass(frozen=True)
class Allocation:
    """One plan item's frozen allocations: ``usd`` the central one, kept while the item is an
    unfinished higher-priority item; ``stress_gpu_usd`` and ``stress_api_usd`` the GPU and API
    parts of the reservation it holds once admitted (paid dollars, the idle margin included).
    ``kind`` groups items for the update rule."""

    item: str
    kind: str
    usd: float
    stress_gpu_usd: float
    stress_api_usd: float

    @property
    def stress_usd(self) -> float:
        return self.stress_gpu_usd + self.stress_api_usd


@dataclass(frozen=True)
class ExecutionLimits:
    """What an admitted, running item (or one worker's share of it) may still spend, in dollars
    only: the GPU part sets the Pod's deadline, the API part the API allowance."""

    item: str
    gpu_usd: float
    api_usd: float

    @property
    def usd(self) -> float:
        return self.gpu_usd + self.api_usd

    def pod_hours(self, pod_usd_per_hour: float) -> float:
        """The Pod's hard deadline, in hours from its creation: the GPU part at the Pod's actual
        price (paid time, idle included)."""
        if pod_usd_per_hour <= 0:
            raise ValueError("the Pod's hourly price must be positive")
        return self.gpu_usd / pod_usd_per_hour


PLANNED, RESERVED, RUNNING, COMPLETED, STOPPED, DROPPED = (
    "planned",
    "reserved",
    "running",
    "completed",
    "stopped",
    "dropped",
)


def finite_usd(value: Any, what: str, *, positive: bool = False) -> float:
    """A dollar amount of a meter event: a finite number, non-negative (positive if asked);
    anything else (NaN, an infinity, a string) is refused, also when a log is replayed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, not {value!r}")
    usd = float(value)
    if not math.isfinite(usd) or usd < 0 or (positive and usd == 0):
        raise ValueError(
            f"{what} must be a finite {'positive' if positive else 'non-negative'} amount"
        )
    return usd


def allocations_hash(allocations: Sequence[Allocation]) -> str:
    import hashlib

    blob = json.dumps([dataclasses_asdict(a) for a in allocations], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def dataclasses_asdict(a: Allocation) -> dict[str, Any]:
    return {
        "item": a.item,
        "kind": a.kind,
        "usd": a.usd,
        "stress_gpu_usd": a.stress_gpu_usd,
        "stress_api_usd": a.stress_api_usd,
    }


class AllocationMeter:
    """The continuation's meter (§12.8): keyed reservations over plan items in a frozen priority,
    highest first, with checked state transitions and an append-only event log.

    planned → admit → reserved → start → running → complete → completed, or → stopped when a
    charge overruns the reservation; planned or reserved → drop (a frozen reason) → dropped.

    - ``admit``: spent + the outstanding reservations of every reserved or running item + this
      item's stress reservation + the central allocations of every planned item of higher
      priority + future storage + reserve ≤ cap. On success the item holds its stress
      reservation. A started, completed, stopped or dropped item cannot be admitted again.
    - ``start`` needs a live reservation and returns the execution limits, in dollars only: the
      GPU part sets the Pod's deadline (``ExecutionLimits.pod_hours``), the API part the API
      allowance. ``split``, once and right after ``start``, divides them between Pods or workers
      disjointly; the item's whole limits are no longer issued after it. Every permission is
      formed inside the locked transaction, on the state with every instance's events.
    - ``charge`` books paid costs against a running item. Beyond either part of its reservation
      the item is stopped, the event is logged and ``BudgetExhausted`` is raised: its unfinished
      contrasts get p = 1 (§12.4.8).
    - ``transfer`` moves money between a running, unsplit item's GPU and API parts, from the
      unspent remainder of the source part only, with a stated reason: the reservation and so the
      item's total stay as they are, and the parts' limits (for ``charge`` and the execution
      limits) are the frozen ones shifted by the transfers.
    - ``complete`` fixes the item's cost at what was charged; it never changes afterwards.
    - Update rule: a planned item's allocation is max(central, the mean cost of the completed
      items of its kind).
    - ``drop`` takes only a reason frozen for that item and releases a reservation.

    With a log, there is one writer at a time: every operation holds an exclusive lock on the
    log, first reads the events other instances appended, then writes its own event to disk
    (flushed and fsynced) before the state changes and before any limit is returned. The log's
    first line records the hash of the frozen allocations and the cap; a log of other
    allocations is refused.
    """

    def __init__(
        self,
        allocations: Sequence[Allocation],
        *,
        cap: float,
        reserve: float,
        drop_reasons: Mapping[str, Sequence[str]],
        log_path: Path | None = None,
    ) -> None:
        names = [a.item for a in allocations]
        if len(names) != len(set(names)):
            raise ValueError("plan items must be distinct")
        unknown = set(drop_reasons) - set(names)
        if unknown:
            raise ValueError(f"drop reasons for unknown items: {sorted(unknown)}")
        self._items: tuple[Allocation, ...] = tuple(allocations)
        self._rank = {a.item: i for i, a in enumerate(self._items)}
        self.cap, self.reserve = float(cap), float(reserve)
        self._drop_reasons = {k: frozenset(v) for k, v in drop_reasons.items()}
        self.header = {
            "event": "header",
            "allocations_sha256": allocations_hash(self._items),
            "cap": self.cap,
            "reserve": self.reserve,
        }
        self.state = dict.fromkeys(names, PLANNED)
        self.reservation: dict[str, float] = {}
        self.charged = {n: {"gpu_usd": 0.0, "api_usd": 0.0} for n in names}
        self.final: dict[str, float] = {}
        self.reason: dict[str, str] = {}
        self.splits: dict[str, dict[str, float]] = {}
        self.transfers: dict[str, float] = {}  # per item, USD moved from its GPU to its API part
        self.events: list[dict[str, Any]] = []
        self.log_path = None if log_path is None else Path(log_path)
        self._offset = 0
        self._handle: Any = None
        if self.log_path is not None:
            with self._locked():
                pass

    # --- the log ----------------------------------------------------------------------------

    def _locked(self):
        """The exclusive lock on the log for one operation, with the log's new events applied."""
        import contextlib

        meter = self

        @contextlib.contextmanager
        def lock():
            assert meter.log_path is not None
            meter.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meter.log_path, "a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    handle.seek(0)
                    lines = handle.read().splitlines()
                    if not lines:
                        meter._write(handle, meter.header)
                        lines = [json.dumps(meter.header)]
                    first = json.loads(lines[0])
                    stored = {k: first.get(k) for k in meter.header}
                    if stored != meter.header:
                        raise ValueError(f"the meter log is of other allocations: {stored}")
                    for line in lines[1 + meter._offset :]:
                        meter._apply(json.loads(line))
                    meter._offset = len(lines) - 1
                    meter._handle = handle
                    yield
                finally:
                    meter._handle = None
                    fcntl.flock(handle, fcntl.LOCK_UN)

        return lock()

    @staticmethod
    def _write(handle: Any, event: Mapping[str, Any]) -> None:
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def _record(self, event: dict[str, Any]) -> None:
        """Validate ``event`` on a copy of the state, write it durably, then apply it."""
        import copy

        trial = copy.deepcopy(self._state())
        self._apply(event, state=trial)  # raises before anything is written
        if self.log_path is not None:
            stamped = {**event, "at": datetime.now(UTC).isoformat()}
            self._write(self._handle, stamped)
            self._offset += 1
        self._apply(event)

    def _transaction(self):
        import contextlib

        return self._locked() if self.log_path is not None else contextlib.nullcontext()

    def _operate(
        self,
        build: Callable[[], dict[str, Any]],
        result: Callable[[dict[str, Any]], Any] | None = None,
    ) -> Any:
        """One operation under the lock (when logged), on the state with every instance's events:
        build its event, record it, and form what it returns inside the same transaction, so that
        no permission is issued from a stale state."""
        with self._transaction():
            event = build()
            self._record(event)
            return event if result is None else result(event)

    # --- reading ----------------------------------------------------------------------------

    def _state(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reservation": self.reservation,
            "charged": self.charged,
            "final": self.final,
            "reason": self.reason,
            "splits": self.splits,
            "transfers": self.transfers,
        }

    @property
    def priority(self) -> tuple[str, ...]:
        return tuple(a.item for a in self._items)

    def _get(self, item: str) -> Allocation:
        if item not in self._rank:
            raise KeyError(f"unknown plan item {item!r}")
        return self._items[self._rank[item]]

    def parts(self, item: str, state: Mapping[str, Any] | None = None) -> tuple[float, float]:
        """An item's GPU and API parts: the frozen ones shifted by its transfers; their sum is its
        stress reservation."""
        a, s = self._get(item), self._state() if state is None else state
        moved = s["transfers"].get(item, 0.0)
        return a.stress_gpu_usd - moved, a.stress_api_usd + moved

    def charged_usd(self, item: str) -> float:
        c = self.charged[item]
        return c["gpu_usd"] + c["api_usd"]

    def allocation(self, item: str) -> float:
        a = self._get(item)
        done = [
            self.final[x.item]
            for x in self._items
            if x.kind == a.kind and self.state[x.item] == COMPLETED
        ]
        return max(a.usd, sum(done) / len(done)) if done else a.usd

    def unfinished_above(self, item: str) -> list[str]:
        rank = self._rank[self._get(item).item]
        return [a.item for a in self._items[:rank] if self.state[a.item] == PLANNED]

    def outstanding(self) -> float:
        """What the live reservations may still spend."""
        return sum(
            max(0.0, self.reservation[x] - self.charged_usd(x))
            for x in self.priority
            if self.state[x] in (RESERVED, RUNNING)
        )

    def spent(self) -> float:
        return sum(self.charged_usd(x) for x in self.priority)

    def _limits(self, item: str) -> ExecutionLimits:
        """The whole item's remaining limits, from the current state (the caller holds the lock)."""
        if self.state[item] != RUNNING:
            raise ValueError(
                f"{item} is {self.state[item]}: no execution without a live reservation"
            )
        if item in self.splits:
            raise ValueError(f"{item} is split: its limits were issued to its workers")
        (gpu, api), c = self.parts(item), self.charged[item]
        return ExecutionLimits(item, max(0.0, gpu - c["gpu_usd"]), max(0.0, api - c["api_usd"]))

    def limits(self, item: str) -> ExecutionLimits:
        """A running item's remaining limits, read under the lock with every instance's events."""
        with self._transaction():
            return self._limits(item)

    # --- transitions ------------------------------------------------------------------------

    def admit(self, item: str, *, spent_elsewhere: float, storage: float) -> MeterDecision:
        """Admit ``item`` if the meter allows it; ``spent_elsewhere`` is money spent outside the
        plan items (the pilot's remainder, storage paid so far), ``storage`` the future storage."""

        def build() -> dict[str, Any]:
            a = self._get(item)
            if self.state[item] != PLANNED:
                raise ValueError(f"{item} is {self.state[item]}: it cannot be admitted again")
            terms = {
                "spent": float(spent_elsewhere) + self.spent(),
                "reservations": self.outstanding(),
                "next_batch": a.stress_usd,
                "higher_priority": sum(self.allocation(x) for x in self.unfinished_above(item)),
                "storage": float(storage),
                "reserve": self.reserve,
            }
            total = sum(terms.values())
            return {"event": "admit", "item": item, "allowed": total <= self.cap, "terms": terms}

        event = self._operate(build)
        terms = event["terms"]
        return MeterDecision(event["allowed"], sum(terms.values()), self.cap, terms)

    def start(self, item: str) -> ExecutionLimits:
        return self._operate(lambda: {"event": "start", "item": item}, lambda _: self._limits(item))

    def split(self, item: str, shares: Mapping[str, float]) -> dict[str, ExecutionLimits]:
        """Divide a just-started item's limits between Pods or workers, disjointly, once: the
        shares are non-negative and sum to at most 1, and each worker gets its share of both
        parts. Allowed only right after ``start``, before any charge; afterwards the item's
        whole limits are no longer issued."""
        a = self._get(item)

        def issue(event: dict[str, Any]) -> dict[str, ExecutionLimits]:
            return {
                worker: ExecutionLimits(
                    f"{item}@{worker}", a.stress_gpu_usd * s, a.stress_api_usd * s
                )
                for worker, s in self.splits[item].items()
            }

        return self._operate(
            lambda: {"event": "split", "item": item, "shares": dict(shares)}, issue
        )

    def charge(
        self, item: str, *, gpu_usd: float = 0.0, api_usd: float = 0.0, label: str | None = None
    ) -> ExecutionLimits | None:
        """Book paid costs against a running item; ``label`` names the line in the log (e.g. the
        engine check within Stage 1's GPU limit) and changes nothing else."""
        gpu_usd = finite_usd(gpu_usd, "a GPU charge")
        api_usd = finite_usd(api_usd, "an API charge")
        extra = {} if label is None else {"label": str(label)}

        def after(event: dict[str, Any]) -> ExecutionLimits | None:
            if self.state[item] == STOPPED or item in self.splits:
                return None
            return self._limits(item)

        limits = self._operate(
            lambda: {
                "event": "charge",
                "item": item,
                "gpu_usd": gpu_usd,
                "api_usd": api_usd,
                **extra,
            },
            after,
        )
        if self.state[item] == STOPPED:
            raise BudgetExhausted(
                f"{item} overran its reservation and is stopped: its unfinished contrasts get p = 1"
            )
        return limits

    def transfer(
        self, item: str, *, source: str, target: str, usd: float, reason: str
    ) -> ExecutionLimits:
        """Move ``usd`` of a running item's unspent ``source`` part ("gpu" or "api") to its
        ``target`` part; returns the item's limits after it."""
        event = {
            "event": "transfer",
            "item": item,
            "from": source,
            "to": target,
            "usd": finite_usd(usd, "a transfer", positive=True),  # the argument, before any cast
            "reason": str(reason),
        }
        return self._operate(lambda: event, lambda _: self._limits(item))

    def complete(self, item: str) -> float:
        self._operate(lambda: {"event": "complete", "item": item})
        return self.final[item]

    def drop(self, item: str, reason: str) -> None:
        self._operate(lambda: {"event": "drop", "item": item, "reason": reason})

    def _apply(self, event: Mapping[str, Any], *, state: Mapping[str, Any] | None = None) -> None:
        s = self._state() if state is None else state
        states, reservation, charged = s["state"], s["reservation"], s["charged"]
        kind, item = event["event"], event["item"]
        a = self._get(item)
        current = states[item]
        if kind == "admit":
            if current != PLANNED:
                raise ValueError(f"{item} is {current}: it cannot be admitted again")
            for name, term in event["terms"].items():
                finite_usd(term, f"the admission term {name}")
            if event["allowed"]:
                states[item] = RESERVED
                reservation[item] = a.stress_usd
        elif kind == "start":
            if current != RESERVED:
                raise ValueError(f"{item} is {current}: it starts only on a live reservation")
            states[item] = RUNNING
        elif kind == "split":
            if current != RUNNING:
                raise ValueError(f"{item} is {current}: only a running item is split")
            if item in s["splits"]:
                raise ValueError(f"{item} is split already: a split is made once, after start")
            if charged[item]["gpu_usd"] or charged[item]["api_usd"] or item in s["transfers"]:
                raise ValueError(f"{item} has been charged: a split is made right after start")
            shares = {
                str(k): finite_usd(v, f"the share of {k}") for k, v in event["shares"].items()
            }
            if sum(shares.values()) > 1 + 1e-12:
                raise ValueError("split shares must be non-negative and sum to at most 1")
            s["splits"][item] = shares
        elif kind == "charge":
            if current != RUNNING:
                raise ValueError(f"{item} is {current}: only a running item is charged")
            c = charged[item]
            c["gpu_usd"] += finite_usd(event["gpu_usd"], "a GPU charge")
            c["api_usd"] += finite_usd(event["api_usd"], "an API charge")
            gpu, api = self.parts(item, s)
            over = (
                c["gpu_usd"] > gpu + 1e-9
                or c["api_usd"] > api + 1e-9
                or c["gpu_usd"] + c["api_usd"] > reservation[item] + 1e-9
            )
            if over:
                states[item] = STOPPED
                s["final"][item] = c["gpu_usd"] + c["api_usd"]
        elif kind == "transfer":
            if current != RUNNING:
                raise ValueError(f"{item} is {current}: only a running item's parts are moved")
            if item in s["splits"]:
                raise ValueError(f"{item} is split: its limits were issued to its workers")
            source, target = event["from"], event["to"]
            if {source, target} != {"gpu", "api"}:
                raise ValueError("a transfer moves money between the gpu and the api part")
            usd = finite_usd(event["usd"], "a transfer", positive=True)
            if not str(event.get("reason") or "").strip():
                raise ValueError("a transfer needs its reason")
            gpu, api = self.parts(item, s)
            left = (
                gpu - charged[item]["gpu_usd"]
                if source == "gpu"
                else api - charged[item]["api_usd"]
            )
            if not math.isfinite(left) or usd > left + 1e-9:
                raise ValueError(f"{item}: only ${left:.4f} of its {source} part is unspent")
            moved = s["transfers"].get(item, 0.0)
            s["transfers"][item] = moved + usd if source == "gpu" else moved - usd
        elif kind == "complete":
            if current != RUNNING:
                raise ValueError(f"{item} is {current}: only a running item completes")
            states[item] = COMPLETED
            s["final"][item] = charged[item]["gpu_usd"] + charged[item]["api_usd"]
        elif kind == "drop":
            if current not in (PLANNED, RESERVED):
                raise ValueError(f"{item} is {current}: only a planned or reserved item is dropped")
            if event["reason"] not in self._drop_reasons.get(item, frozenset()):
                allowed = sorted(self._drop_reasons.get(item, ()))
                raise ValueError(
                    f"{item} cannot be dropped for {event['reason']!r}; frozen: {allowed}"
                )
            states[item] = DROPPED
            reservation.pop(item, None)
            s["reason"][item] = event["reason"]
        else:
            raise ValueError(f"unknown meter event {kind!r}")
        if state is None:
            self.events.append(dict(event))


def load_allocations(path: Path) -> list[Allocation]:
    """The frozen allocations file (configs/continuation/allocations.yaml), in priority order."""
    from omegaconf import OmegaConf

    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return [
        Allocation(
            str(x["item"]),
            str(x["kind"]),
            float(x["usd"]),
            float(x["stress_gpu_usd"]),
            float(x["stress_api_usd"]),
        )
        for x in data["allocations"]
    ]


def apply_cut_order(
    projected: Mapping[str, float], *, order: Sequence[str], cap: float, fixed: float
) -> list[str]:
    """The cuts, in the preset order, that bring ``fixed`` + the remaining projected parts within
    the cap. ``projected`` prices each cuttable part by its name in the cut order (parts absent
    from it cost nothing); ``fixed`` holds the spent, the reserve and every protected or uncut
    item. ValueError if even every cut leaves the plan over the cap."""
    unknown = set(projected) - set(order)
    if unknown:
        raise ValueError(f"parts outside the cut order: {sorted(unknown)}")
    remaining = dict(projected)
    cuts: list[str] = []
    for part in order:
        if float(fixed) + sum(remaining.values()) <= cap:
            break
        if remaining.pop(part, None) is not None:
            cuts.append(part)
    if float(fixed) + sum(remaining.values()) > cap:
        raise ValueError("the protected plan exceeds the cap after every cut")
    return cuts
