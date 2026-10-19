"""The continuation's meter (preregistration §12.8), operated from this machine, its one writer.

    uv run python scripts/continuation_meter.py status
    uv run python scripts/continuation_meter.py admit <item> --spent-elsewhere USD --storage USD
    uv run python scripts/continuation_meter.py start <item> [--price USD_PER_HOUR [--receipt FILE]]
    uv run python scripts/continuation_meter.py limits <item> [--price USD_PER_HOUR]
    uv run python scripts/continuation_meter.py charge <item> [--gpu-usd USD] [--api-usd USD] \
        [--label TEXT]
    uv run python scripts/continuation_meter.py verify-receipt FILE --item ITEM \
        --price USD_PER_HOUR --hours H
    uv run python scripts/continuation_meter.py transfer <item> --from gpu --to api --usd USD \
        --reason TEXT
    uv run python scripts/continuation_meter.py migrate-cap --reason TEXT
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
(``meter.cap_usd``: $1,700 from 2026-10-19, the user's decision; $1,500 before): a log begun under
a lower cap records nothing until ``migrate-cap`` raises it to the rules' cap with its reason
(``AllocationMeter.migrate_cap``), keeping its spent money, reservations and history; ``status``
shows the cap in force, the rules' and the one the log was begun under.

A Pod is launched only on a live reservation: ``start --receipt`` writes a launch receipt (the
item, its limits, the price, the Pod's hours, the allocations' hash and the sha256 of the log
right after the start). On the Pod, ``verify-receipt`` with ``--log`` naming the copy of that
log replays it and succeeds only if the copy is the one the receipt names, the receipt is for the
expected item, that item is running, unsplit and uncharged with exactly those limits, the price
is the receipt's, and the requested hours fit within the item's live GPU limit at that price
(the receipt's pod_hours must agree; they are never trusted on their own). A refused admission
exits with status 4, an overrun charge with 3, a failed verification with 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from residual_disposition.budget import AllocationMeter, BudgetExhausted, load_allocations
from residual_disposition.continuation_rules import drop_reasons, load_rules
from residual_disposition.paths import CONFIG_ROOT, data_root
from residual_disposition.simulate_continuation import FROZEN

ALLOCATIONS = CONFIG_ROOT / "continuation" / "allocations.yaml"


def meter(log: Path, allocations: Path = ALLOCATIONS) -> AllocationMeter:
    rules = load_rules(expected_hash=FROZEN["rules_sha256"])
    return AllocationMeter(
        load_allocations(allocations),
        cap=float(rules.data["meter"]["cap_usd"]),
        reserve=float(rules.data["meter"]["reserve_usd"]),
        drop_reasons=drop_reasons(rules),
        log_path=log,
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


def verify_receipt(
    m: AllocationMeter, data: dict, *, item: str, price: float, hours: float
) -> list[str]:
    """Why a launch receipt does not authorize ``hours`` of ``item`` at ``price`` on the log copy
    of ``m``. The hours allowed come from the replayed log, never from the receipt: the item's
    live GPU limit at the verified price; the receipt's own fields must agree with them."""
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
    p = sub.add_parser("verify-receipt")
    p.add_argument("receipt", type=Path)
    p.add_argument("--item", required=True)
    p.add_argument("--price", type=float, required=True)
    p.add_argument("--hours", type=float, required=True)
    p = sub.add_parser("transfer")
    p.add_argument("item")
    p.add_argument("--from", dest="source", choices=("gpu", "api"), required=True)
    p.add_argument("--to", dest="target", choices=("gpu", "api"), required=True)
    p.add_argument("--usd", type=float, required=True)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("migrate-cap")
    p.add_argument("--reason", required=True)
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
        problems = verify_receipt(
            m,
            json.loads(args.receipt.read_text()),
            item=args.item,
            price=args.price,
            hours=args.hours,
        )
        print(
            json.dumps({"receipt": str(args.receipt), "valid": not problems, "problems": problems})
        )
        return 1 if problems else 0
    elif args.command == "limits":
        out = limits_json(m.limits(args.item), args.price)
    elif args.command == "charge":
        try:
            left = m.charge(args.item, gpu_usd=args.gpu_usd, api_usd=args.api_usd, label=args.label)
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
    elif args.command == "complete":
        out = {"item": args.item, "final_usd": m.complete(args.item)}
    else:
        m.drop(args.item, args.reason)
        out = {"item": args.item, "dropped": args.reason}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
