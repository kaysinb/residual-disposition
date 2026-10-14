"""Whether a continuation's next audited dose may start (preregistration §11.11; plan, Projected
overrun), from the measured work of the pass that ran the previous dose.

    uv run python scripts/dose_projection.py --runs runs/<experiment> --since <pass start ISO> \\
        --started-at <batch start ISO> --cap-hours <GPU-hours> [--prior-hours 0] \\
        --deadline-state <watcher state.json> [--margin-min 25] [--export-min 30] [--step-min 25] \\
        [--ledger data/ledgers/adl-api-<worker>.jsonl --allowance-usd <its share>]

The projection is the previous pass's measured work: the stored durations of the completed cells
it created (run directories of ``--runs`` that started at or after ``--since``: its confirmatory
passes, certificate runs and readouts), as the certify pass projects d₂ from d₁, or its wall
time from ``--since`` to the last cell's end if that is longer (the cap counts wall time). The
next dose may start only if both hold:
- the batch's GPU hours used now plus the projection stay within ``--cap-hours``;
- the dose runs in full: now plus the projection ends no later than the last moment a step may
  start, the watcher's deadline minus ``--margin-min`` (the resume rule's margin), the final
  export (``--export-min``) and the longest step (``--step-min``, or the longest cell measured
  if that is longer).
With ``--ledger``, also:
- what is left of the worker's ADL API share (``budget.ApiLedger``: the allowance minus the
  settled spend and open reservations) covers the previous pass's committed API spend plus 25 %.
It prints the measurement, the checks and ``stop_fraction``: the share of the cap from which no
step starts, that last moment (at most 1). Exit status 0 when the dose may start, 2 when it may
not. Standard library only, so it runs on the Pod outside the project's environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


def parse_time(value: str) -> datetime:
    at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if at.tzinfo is None:
        raise ValueError(f"{value}: a time zone is required")
    return at.astimezone(UTC)


def measured_hours(runs: Path, since: datetime) -> tuple[float, int, float, float]:
    """The stored durations of the completed cells under ``runs`` that started at or after
    ``since`` (a cell's run directory is named date-time-kind-hash-suffix; the pass's own run,
    without a kind, is left out): their sum, their number, the longest, and the span from
    ``since`` to the last one's end, in hours."""
    hours, cells, longest, span = 0.0, 0, 0.0, 0.0
    for meta_path in sorted(runs.glob("*/meta.json")):
        if len(meta_path.parent.name.split("-")) != 5:
            continue
        meta = json.loads(meta_path.read_text())
        started, finished = meta.get("started_at"), meta.get("finished_at")
        if meta.get("status") != "completed" or not started or not finished:
            continue
        start = parse_time(started)
        if start < since:
            continue
        end = parse_time(finished)
        duration = max(0.0, (end - start).total_seconds() / 3600)
        hours += duration
        cells += 1
        longest = max(longest, duration)
        span = max(span, (end - since).total_seconds() / 3600)
    return hours, cells, longest, span


def api_committed(ledger: Path, since: datetime) -> tuple[float, float]:
    """The ledger's committed USD (settled spend plus open reservations, as
    ``budget.ApiLedger._apply`` counts them) now, and the part committed at or after ``since``."""

    def committed(events: list[dict]) -> float:
        spent, open_ = 0.0, {}
        for event in events:
            kind = event["event"]
            if kind == "reserve":
                open_[event["id"]] = float(event["usd"])
            elif kind == "spend":
                spent += float(event["usd"])
                open_.pop(event["id"], None)
            elif kind == "failed" and event.get("released"):
                open_.pop(event["id"], None)
            elif kind == "unbilled":
                for reservation in event["ids"]:
                    open_.pop(reservation, None)
        return spent + sum(open_.values())

    events = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    before = [e for e in events if parse_time(e["at"]) < since]
    total = committed(events)
    return total, total - committed(before)


def projection(
    *,
    measured: float,
    now: datetime,
    started_at: datetime,
    prior_hours: float,
    cap_hours: float,
    deadline: datetime,
    margin: timedelta,
    export: timedelta,
    step: timedelta,
) -> dict[str, object]:
    """Both checks for the next dose and the stop fraction that keeps its last step in time;
    the dose must end by that last start, so the stop cuts none of its steps."""
    used = prior_hours + (now - started_at).total_seconds() / 3600
    end = now + timedelta(hours=measured)
    allowed_end = deadline - margin - export  # the dose's work ends here, the export after it
    last_start = allowed_end - step
    fraction = (prior_hours + (last_start - started_at).total_seconds() / 3600) / cap_hours
    within_cap = used + measured <= cap_hours
    in_time = end <= last_start
    return {
        "measured_hours": measured,
        "used_hours": used,
        "cap_hours": cap_hours,
        "within_cap": within_cap,
        "projected_end": end.isoformat(),
        "allowed_end": allowed_end.isoformat(),
        "in_time": in_time,
        "stop_fraction": min(1.0, fraction),
        "last_step_start": last_start.isoformat(),
        "start": within_cap and in_time,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--since", required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--cap-hours", type=float, required=True)
    parser.add_argument("--prior-hours", type=float, default=0.0)
    parser.add_argument("--deadline-state", type=Path, required=True)
    parser.add_argument("--margin-min", type=float, default=25.0)
    parser.add_argument("--export-min", type=float, default=30.0)
    parser.add_argument("--step-min", type=float, default=25.0)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--allowance-usd", type=float)
    args = parser.parse_args()
    if (args.ledger is None) != (args.allowance_usd is None):
        parser.error("--ledger and --allowance-usd go together")
    cell_hours, cells, longest, span = measured_hours(args.runs, parse_time(args.since))
    if cells == 0:
        print(json.dumps({"start": False, "reason": "no completed cell since the pass started"}))
        return 2
    state = json.loads(args.deadline_state.read_text())
    deadline = datetime.fromtimestamp(float(state["deadline"]), UTC)
    record = projection(
        measured=max(cell_hours, span),
        now=datetime.now(UTC),
        started_at=parse_time(args.started_at),
        prior_hours=args.prior_hours,
        cap_hours=args.cap_hours,
        deadline=deadline,
        margin=timedelta(minutes=args.margin_min),
        export=timedelta(minutes=args.export_min),
        step=max(timedelta(minutes=args.step_min), timedelta(hours=longest)),
    )
    if args.ledger is not None:
        committed, previous = api_committed(args.ledger, parse_time(args.since))
        left = args.allowance_usd - committed
        api_ok = left >= 1.25 * previous
        record.update(
            api_left_usd=left,
            api_previous_usd=previous,
            api_ok=api_ok,
            start=record["start"] and api_ok,
        )
    record.update(
        cells=cells,
        cell_hours=cell_hours,
        span_hours=span,
        longest_cell_hours=longest,
        deadline=deadline.isoformat(),
    )
    print(json.dumps(record, indent=2))
    return 0 if record["start"] else 2


if __name__ == "__main__":
    sys.exit(main())
