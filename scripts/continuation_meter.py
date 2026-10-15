"""The continuation's meter (preregistration §12.8), operated from this machine, its one writer.

    uv run python scripts/continuation_meter.py status
    uv run python scripts/continuation_meter.py admit <item> --spent-elsewhere USD --storage USD
    uv run python scripts/continuation_meter.py start <item> [--price USD_PER_HOUR]
    uv run python scripts/continuation_meter.py limits <item> [--price USD_PER_HOUR]
    uv run python scripts/continuation_meter.py charge <item> [--gpu-usd USD] [--api-usd USD]
    uv run python scripts/continuation_meter.py complete <item>
    uv run python scripts/continuation_meter.py drop <item> <reason>

The frozen allocations (``configs/continuation/allocations.yaml``), the cap and reserve and the
drop reasons come from the frozen rules, which must match their sha256 in the Stage-0
specification. The log (``data/ledgers/continuation_meter.jsonl`` by default) is locked for each
operation and records the allocations' hash in its header. Every command prints JSON. With
``--price`` (the Pod's actual hourly price), ``start`` and ``limits`` also give the Pod's hard
deadline in hours from its creation (``ExecutionLimits.pod_hours``).
"""

from __future__ import annotations

import argparse
import json
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


def status(m: AllocationMeter) -> dict:
    with m._transaction():  # the state with every instance's events, under the lock
        return {
            "items": {
                item: {
                    "state": m.state[item],
                    "allocation": m.allocation(item),
                    "reservation": m.reservation.get(item),
                    "charged": m.charged[item],
                    "final": m.final.get(item),
                    "reason": m.reason.get(item),
                }
                for item in m.priority
            },
            "spent": m.spent(),
            "outstanding": m.outstanding(),
            "cap": m.cap,
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
    p = sub.add_parser("charge")
    p.add_argument("item")
    p.add_argument("--gpu-usd", type=float, default=0.0)
    p.add_argument("--api-usd", type=float, default=0.0)
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
    elif args.command == "start":
        out = limits_json(m.start(args.item), args.price)
    elif args.command == "limits":
        out = limits_json(m.limits(args.item), args.price)
    elif args.command == "charge":
        try:
            left = m.charge(args.item, gpu_usd=args.gpu_usd, api_usd=args.api_usd)
        except BudgetExhausted as stop:
            print(json.dumps({"item": args.item, "stopped": str(stop)}))
            return 3
        out = {"item": args.item, "left": None if left is None else limits_json(left, None)}
    elif args.command == "complete":
        out = {"item": args.item, "final_usd": m.complete(args.item)}
    else:
        m.drop(args.item, args.reason)
        out = {"item": args.item, "dropped": args.reason}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
