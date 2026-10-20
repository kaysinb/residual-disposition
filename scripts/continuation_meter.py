"""The continuation's meter (preregistration §12.8), operated from this machine, its one writer.

    uv run python scripts/continuation_meter.py status
    uv run python scripts/continuation_meter.py admit <item> --spent-elsewhere USD --storage USD
    uv run python scripts/continuation_meter.py start <item> [--price USD_PER_HOUR [--receipt FILE]]
    uv run python scripts/continuation_meter.py limits <item> [--price USD_PER_HOUR]
    uv run python scripts/continuation_meter.py charge <item> [--gpu-usd USD] [--api-usd USD] \
        [--label TEXT]
    uv run python scripts/continuation_meter.py verify-receipt FILE --item ITEM \
        --price USD_PER_HOUR --hours H [--worker WORKER --pod-id POD_ID]
    uv run python scripts/continuation_meter.py split <item> --share WORKER=SHARE ...
    uv run python scripts/continuation_meter.py session-open <item> --worker WORKER \
        --pod-id POD_ID --deadline ISO --price USD_PER_HOUR --hours H [--api-usd USD] \
        --receipt FILE
    uv run python scripts/continuation_meter.py session-close <item> --worker WORKER \
        --pod-id POD_ID --gpu-usd USD [--api-usd USD]
    uv run python scripts/continuation_meter.py transfer <item> --from gpu --to api --usd USD \
        --reason TEXT
    uv run python scripts/continuation_meter.py migrate-cap --reason TEXT
    uv run python scripts/continuation_meter.py migrate-allocations
    uv run python scripts/continuation_meter.py complete <item>
    uv run python scripts/continuation_meter.py drop <item> <reason>

The frozen allocations (``configs/continuation/allocations.yaml``), the cap and reserve and the
drop reasons come from the frozen rules, which must match their sha256 in the Stage-0
specification. The log (``data/ledgers/continuation_meter.jsonl`` by default) is locked for each
operation and records the allocations' hash in its header. Every command prints JSON. With
``--price`` (the Pod's actual hourly price), ``start`` and ``limits`` also give the Pod's hard
deadline in hours from its creation (``ExecutionLimits.pod_hours``). ``transfer`` moves unspent
money between a running item's GPU and API parts with its reason; the item's reservation, and so
the meter's total, stay as they are (``AllocationMeter.transfer``). The cap is the rules'
(``meter.cap_usd``: $1,820 from 2026-10-20 and $1,770 from 2026-10-19, the user's decisions;
$1,500 before): a log begun under a lower cap records nothing until ``migrate-cap`` raises it to
the rules' cap with its reason (``AllocationMeter.migrate_cap``), keeping its spent money,
reservations and history; ``status`` shows the cap in force, the rules' and the one the log was
begun under. The allocations only grow, by the file's append-only ``migrations`` (review s5b 2;
C15 adds stage5_dpo below stage5): a log begun under the allocations before one records nothing
until ``migrate-allocations`` records each migration in order (its hashes, added items and
reason), between Pod sessions; ``status`` lists the items still to be migrated in.

A Pod is launched only on a live reservation: ``start --receipt`` writes a launch receipt (the
item, its limits, the price, the Pod's hours, the allocations' hash and the sha256 of the log
right after the start). Only Stage 1's item ran its Pods on one (``LAUNCH_RECEIPT_ITEMS``); every
later item runs them on session receipts, and a launch receipt of it authorizes nothing (review
s4u 5). On the Pod, ``verify-receipt`` with ``--log`` naming the copy of that
log replays it and succeeds only if the copy is the one the receipt names, the receipt is for the
expected item, that item is running, unsplit and uncharged with exactly those limits, the price
is the receipt's, and the requested hours fit within the item's live GPU limit at that price
(the receipt's pod_hours must agree; they are never trusted on their own). A refused admission
exits with status 4, an overrun charge with 3, a failed verification with 1.

Two Pods share an item (Stage 2: Pod A and Pod B) through ``split``, right after ``start``: each
worker gets its share of both parts, and every later charge names its worker (``charge
--worker``), who may spend no more than its share. Each Pod session is authorized exclusively in
the log (review s3y2): ``session-open``, with the Pod's actual ID and the deadline set at its
creation (both from the coordinator, who creates the Pod), its price and hours within the
worker's live GPU share and the API money it may commit (``--api-usd``, review s4b) within the
worker's API share, writes the session's receipt (the sha256 of the log right after it); it is
refused while that worker has an open session. ``session-close`` settles the session's paid cost
as the worker's charge and closes it, so that the next may open. On the Pod, ``verify-receipt
--worker --pod-id`` replays the log copy and succeeds only if the copy is the receipt's, the
worker's open session there is the receipt's (the Pod ID the Pod's own, the deadline, the price,
the hours, the API authorization), the deadline is still ahead and the requested hours fit within
the session's; it prints that session as the log holds it and its id (the sha256 of the log
right after it), which is where the Pod's chain takes its deadline and its API ceiling from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from residual_disposition.budget import (
    AllocationMeter,
    BudgetExhausted,
    load_allocations,
    load_migrations,
)
from residual_disposition.continuation_rules import drop_reasons, load_rules
from residual_disposition.paths import CONFIG_ROOT, data_root
from residual_disposition.simulate_continuation import FROZEN

ALLOCATIONS = CONFIG_ROOT / "continuation" / "allocations.yaml"


def meter(log: Path, allocations: Path = ALLOCATIONS) -> AllocationMeter:
    rules = load_rules(expected_hash=FROZEN["rules_sha256"])
    items = load_allocations(allocations)
    return AllocationMeter(
        items,
        cap=float(rules.data["meter"]["cap_usd"]),
        reserve=float(rules.data["meter"]["reserve_usd"]),
        drop_reasons=drop_reasons(rules, [a.item for a in items]),
        log_path=log,
        migrations=load_migrations(allocations),
    )


def limits_json(limits, price: float | None) -> dict:
    out = {"item": limits.item, "gpu_usd": limits.gpu_usd, "api_usd": limits.api_usd}
    if price is not None:
        out["price_usd_per_hour"] = price
        out["pod_hours"] = limits.pod_hours(price)
    return out


def receipt(m: AllocationMeter, limits, price: float) -> dict:
    return {
        **limits_json(limits, price),
        "allocations_sha256": m.header["allocations_sha256"],
        "cap": m.cap,
        "log_sha256": hashlib.sha256(m.log_path.read_bytes()).hexdigest(),
        "issued_at": datetime.now(UTC).isoformat(),
    }


def session_receipt(m: AllocationMeter, event: dict) -> dict:
    """A Pod session's receipt: the session as journaled, with the log's sha256 right after."""
    return {
        **{k: v for k, v in event.items() if k != "event"},
        "allocations_sha256": m.header["allocations_sha256"],
        "cap": m.cap,
        "log_sha256": hashlib.sha256(m.log_path.read_bytes()).hexdigest(),
        "issued_at": datetime.now(UTC).isoformat(),
    }


def verify_session_receipt(
    m: AllocationMeter,
    data: dict,
    *,
    item: str,
    worker: str,
    pod_id: str,
    price: float,
    hours: float,
    now: datetime | None = None,
) -> list[str]:
    """Why a Pod session's receipt does not authorize ``hours`` on Pod ``pod_id`` (review s3y2):
    the log copy must be the receipt's, the worker's open session in it the receipt's (Pod ID,
    deadline, price, hours), the Pod the session's, the deadline still ahead and the hours within
    the session's; never from the receipt alone."""
    problems = []
    if hashlib.sha256(m.log_path.read_bytes()).hexdigest() != data.get("log_sha256"):
        problems.append("the log copy is not the log the receipt was issued from")
    if data.get("allocations_sha256") != m.header["allocations_sha256"]:
        problems.append("the receipt is of other allocations")
    if (data.get("item"), data.get("worker")) != (item, worker):
        return [*problems, f"the receipt is for {data.get('item')}@{data.get('worker')}"]
    if data.get("pod_id") != pod_id:
        return [*problems, f"the receipt is for Pod {data.get('pod_id')}, not this Pod {pod_id}"]
    with m._transaction():  # the item's state and the session read together (review s4a)
        state = m.state.get(item)
        session = (m.sessions.get(item) or {}).get(worker)
    if state != "running":
        return [*problems, f"{item} is {state}: no session runs on a stopped item"]
    keys = ("pod_id", "deadline", "price_usd_per_hour", "hours", "gpu_usd", "api_usd")
    if session is None or any(session[k] != data.get(k) for k in keys):
        return [*problems, f"{worker}'s open session in the log is not the receipt's: {session}"]
    if price != session["price_usd_per_hour"]:
        problems.append(f"price {price} is not the session's {session['price_usd_per_hour']}")
    now = now or datetime.now(UTC)
    if now >= datetime.fromisoformat(session["deadline"]):
        problems.append(f"the session's deadline {session['deadline']} has passed")
    if not 0 < hours <= session["hours"]:
        problems.append(f"{hours} h exceed the session's {session['hours']} h")
    return problems


# The only item whose Pods ran on a whole-item launch receipt (Stage 1, 2026-10-15/16): every
# later item runs its Pods on session receipts (review s4u 5), so a launch receipt of any other
# item authorizes nothing, whatever log copy it is checked against.
LAUNCH_RECEIPT_ITEMS = ("stage0_1",)


def verify_receipt(
    m: AllocationMeter, data: dict, *, item: str, price: float, hours: float
) -> list[str]:
    """Why a launch receipt does not authorize ``hours`` of ``item`` at ``price`` on the log copy
    of ``m``. Only Stage 1's item has launch receipts (``LAUNCH_RECEIPT_ITEMS``); the hours allowed
    come from the replayed log, never from the receipt: the item's live GPU limit at the verified
    price; the receipt's own fields must agree with them."""
    if item not in LAUNCH_RECEIPT_ITEMS:
        return [f"{item} runs its Pods on session receipts only: no launch receipt authorizes"]
    problems = []
    if hashlib.sha256(m.log_path.read_bytes()).hexdigest() != data.get("log_sha256"):
        problems.append("the log copy is not the log the receipt was issued from")
    if data.get("allocations_sha256") != m.header["allocations_sha256"]:
        problems.append("the receipt is of other allocations")
    if data.get("item") != item:
        return [*problems, f"the receipt is for {data.get('item')!r}, not {item!r}"]
    if data.get("price_usd_per_hour") != price:
        problems.append(f"price {price} is not the receipt's {data.get('price_usd_per_hour')}")
    if price <= 0:
        return [*problems, "the price must be positive"]
    with m._transaction():
        if item not in m.state:
            return [*problems, f"unknown item {item!r}"]
        if m.state[item] != "running" or item in m.splits:
            split = ", split" if item in m.splits else ""
            return [*problems, f"{item} is {m.state[item]}{split}"]
        if item in m.issued:  # review s4t2: its Pods run on session receipts only
            return [*problems, f"{item} issued a Pod session: no whole-item receipt authorizes"]
        if m.charged_usd(item):
            return [*problems, f"{item} has been charged: not a fresh reservation"]
        live = m._limits(item)
    allowed = live.pod_hours(price)
    if (live.gpu_usd, live.api_usd) != (data.get("gpu_usd"), data.get("api_usd")):
        problems.append(f"the receipt's limits differ from the log's {live}")
    if data.get("pod_hours") != allowed:
        problems.append(f"the receipt's pod_hours {data.get('pod_hours')} are not {allowed}")
    if not 0 < hours <= allowed:
        problems.append(f"{hours} h exceed the {allowed:.4f} h of {item}'s live GPU limit")
    return problems


def status(m: AllocationMeter) -> dict:
    with m._transaction():  # the state with every instance's events, under the lock
        return {
            "items": {
                item: {
                    "state": m.state[item],
                    "allocation": m.allocation(item),
                    "reservation": m.reservation.get(item),
                    "charged": m.charged[item],
                    "parts": dict(zip(("gpu_usd", "api_usd"), m.parts(item), strict=True)),
                    "final": m.final.get(item),
                    "reason": m.reason.get(item),
                }
                for item in m.priority
            },
            "spent": m.spent(),
            "outstanding": m.outstanding(),
            "cap": m.cap,
            "rules_cap": m.rules_cap,
            "log_begun_under_cap": m.header_cap,
            "needs_migration": m.needs_migration,
            "allocations_pending": m.allocations_pending,
            "reserve": m.reserve,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--allocations", type=Path, default=ALLOCATIONS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    p = sub.add_parser("admit")
    p.add_argument("item")
    p.add_argument("--spent-elsewhere", type=float, required=True)
    p.add_argument("--storage", type=float, required=True)
    for name in ("start", "limits"):
        p = sub.add_parser(name)
        p.add_argument("item")
        p.add_argument("--price", type=float, default=None)
        if name == "start":
            p.add_argument("--receipt", type=Path, default=None)
    p = sub.add_parser("charge")
    p.add_argument("item")
    p.add_argument("--gpu-usd", type=float, default=0.0)
    p.add_argument("--api-usd", type=float, default=0.0)
    p.add_argument("--label", default=None)
    p.add_argument("--worker", default=None, help="the worker of a split item")
    p = sub.add_parser("verify-receipt")
    p.add_argument("receipt", type=Path)
    p.add_argument("--item", required=True)
    p.add_argument("--price", type=float, required=True)
    p.add_argument("--hours", type=float, required=True)
    p.add_argument("--worker", default=None, help="a split item's Pod session receipt")
    p.add_argument("--pod-id", default=None, help="this Pod's ID (a session receipt)")
    p = sub.add_parser("split")
    p.add_argument("item")
    p.add_argument("--share", action="append", required=True, metavar="WORKER=SHARE")
    p = sub.add_parser("session-open")
    p.add_argument("item")
    p.add_argument("--worker", required=True)
    p.add_argument("--pod-id", required=True, help="the Pod's actual ID, from its creation")
    p.add_argument("--deadline", required=True, help="the deadline set at the Pod's creation, ISO")
    p.add_argument("--price", type=float, required=True)
    p.add_argument("--hours", type=float, required=True)
    p.add_argument("--api-usd", type=float, default=0.0, help="the API money the session may use")
    p.add_argument("--receipt", type=Path, required=True)
    p = sub.add_parser("session-close")
    p.add_argument("item")
    p.add_argument("--worker", required=True)
    p.add_argument("--pod-id", required=True)
    p.add_argument("--gpu-usd", type=float, required=True)
    p.add_argument("--api-usd", type=float, default=0.0)
    p.add_argument(
        "--api-ledger",
        type=Path,
        default=None,
        help="the session's API cost from this ledger (with --session-id)",
    )
    p.add_argument("--session-id", default=None, help="the session's id (its receipt's log sha)")
    p = sub.add_parser("transfer")
    p.add_argument("item")
    p.add_argument("--from", dest="source", choices=("gpu", "api"), required=True)
    p.add_argument("--to", dest="target", choices=("gpu", "api"), required=True)
    p.add_argument("--usd", type=float, required=True)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("migrate-cap")
    p.add_argument("--reason", required=True)
    sub.add_parser("migrate-allocations")
    p = sub.add_parser("complete")
    p.add_argument("item")
    p = sub.add_parser("drop")
    p.add_argument("item")
    p.add_argument("reason")
    args = parser.parse_args(argv)
    log = args.log or data_root() / "ledgers" / "continuation_meter.jsonl"
    m = meter(log, args.allocations)
    if args.command == "status":
        out = status(m)
    elif args.command == "admit":
        d = m.admit(args.item, spent_elsewhere=args.spent_elsewhere, storage=args.storage)
        out = {"item": args.item, "allowed": d.allowed, "total": d.total, "cap": d.cap}
        out["terms"] = d.terms
        if not d.allowed:
            print(json.dumps(out, indent=1))
            return 4
    elif args.command == "start":
        if args.receipt is not None and args.price is None:
            raise SystemExit("a launch receipt needs the Pod's price, --price")
        limits = m.start(args.item)
        out = limits_json(limits, args.price)
        if args.receipt is not None:
            out = receipt(m, limits, args.price)
            args.receipt.write_text(json.dumps(out, indent=1) + "\n")
    elif args.command == "verify-receipt":
        data = json.loads(args.receipt.read_text())
        if args.worker is None:
            problems = verify_receipt(m, data, item=args.item, price=args.price, hours=args.hours)
        elif not args.pod_id:
            problems = ["a session receipt is verified on its Pod: --pod-id"]
        else:
            problems = verify_session_receipt(
                m, data, item=args.item, worker=args.worker, pod_id=args.pod_id,
                price=args.price, hours=args.hours,
            )  # fmt: skip
        out = {"receipt": str(args.receipt), "valid": not problems, "problems": problems}
        if args.worker is not None and not problems:
            out["session"] = m.session(args.item, args.worker)  # as the log holds it
            out["session_id"] = data["log_sha256"]  # the log right after it: unique (s4d)
        print(json.dumps(out))
        return 1 if problems else 0
    elif args.command == "limits":
        out = limits_json(m.limits(args.item), args.price)
    elif args.command == "split":
        shares = {}
        for pair in args.share:
            worker, _, share = pair.partition("=")
            if not worker or worker in shares:
                raise SystemExit(f"--share {pair}: one WORKER=SHARE per worker")
            shares[worker] = float(share)
        parts = m.split(args.item, shares)
        out = {w: limits_json(limits, None) for w, limits in parts.items()}
    elif args.command == "session-open":
        out = m.open_session(  # the receipt formed under the same lock as the session (s4a)
            args.item, args.worker, pod_id=args.pod_id, deadline=args.deadline,
            price=args.price, hours=args.hours, api_usd=args.api_usd,
            result=lambda event: session_receipt(m, event),
        )  # fmt: skip
        args.receipt.write_text(json.dumps(out, indent=1) + "\n")
    elif args.command == "session-close":
        api_usd = args.api_usd
        if args.api_ledger is not None:
            if args.api_usd or not args.session_id:
                raise SystemExit("--api-ledger takes --session-id and no --api-usd")
            from residual_disposition.budget import session_api_cost

            api_usd = session_api_cost(args.api_ledger, args.worker, args.session_id)
        try:
            left = m.close_session(
                args.item, args.worker, pod_id=args.pod_id, gpu_usd=args.gpu_usd,
                api_usd=api_usd,
            )  # fmt: skip
        except BudgetExhausted as stop:
            print(json.dumps({"item": args.item, "stopped": str(stop)}))
            return 3
        out = {"item": args.item, "worker": args.worker, "api_usd": api_usd,
               "left": None if left is None else limits_json(left, None)}  # fmt: skip
    elif args.command == "charge":
        try:
            left = m.charge(
                args.item,
                gpu_usd=args.gpu_usd,
                api_usd=args.api_usd,
                label=args.label,
                worker=args.worker,
            )
        except BudgetExhausted as stop:
            print(json.dumps({"item": args.item, "stopped": str(stop)}))
            return 3
        out = {"item": args.item, "left": None if left is None else limits_json(left, None)}
    elif args.command == "transfer":
        left = m.transfer(
            args.item, source=args.source, target=args.target, usd=args.usd, reason=args.reason
        )
        out = {"item": args.item, "left": limits_json(left, None)}
    elif args.command == "migrate-cap":
        out = {"cap": m.migrate_cap(reason=args.reason), "log_begun_under_cap": m.header_cap}
    elif args.command == "migrate-allocations":
        out = {"added": m.migrate_allocations(), "allocations_pending": m.allocations_pending,
               "allocations_sha256": m.header["allocations_sha256"]}  # fmt: skip
    elif args.command == "complete":
        out = {"item": args.item, "final_usd": m.complete(args.item)}
    else:
        m.drop(args.item, args.reason)
        out = {"item": args.item, "dropped": args.reason}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
