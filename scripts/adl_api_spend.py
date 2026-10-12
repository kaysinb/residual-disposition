"""ADL's API spend across its workers' ledgers (path-B plan, caps; ``budget.ApiLedger``).

    uv run python scripts/adl_api_spend.py --ledgers data/ledgers/adl-api-*.jsonl \
        --allowances pod-a=10 pod-b=10 laptop=15 [--cap 35]

Each worker spends from a disjoint share of the $35 cap (approved 2026-10-12) and keeps a
durable ledger: settled spend and open reservations. Prints per worker its spend, open
reservations and remaining share, and the total; fails if the shares sum above the cap, a
worker's ledger is missing from the shares, or a ledger holds another worker's events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from residual_disposition.budget import ApiLedger


def spend(ledgers: list[Path], allowances: dict[str, float], cap: float) -> dict[str, object]:
    if sum(allowances.values()) > cap + 1e-9:
        raise SystemExit(f"the shares {allowances} sum above the ${cap:g} cap")
    out: dict[str, object] = {}
    total = 0.0
    for path in ledgers:
        events = [json.loads(line) for line in path.read_text().splitlines() if line]
        workers = {e["worker"] for e in events}
        if len(workers) > 1:
            raise SystemExit(f"{path} holds several workers' events: {sorted(workers)}")
        worker = workers.pop() if workers else path.stem.removeprefix("adl-api-")
        if worker not in allowances:
            raise SystemExit(f"{path}: worker {worker} has no share in --allowances")
        state = ApiLedger(path, allowances[worker], worker, readonly=True).state()
        out[worker] = {
            "spent_usd": state["spent"],
            "open_usd": state["committed"] - state["spent"],
            "remaining_usd": allowances[worker] - state["committed"],
            "allowance_usd": allowances[worker],
        }
        total += state["committed"]
    out["total_committed_usd"] = total
    out["cap_usd"] = cap
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledgers", nargs="+", type=Path, required=True)
    parser.add_argument("--allowances", nargs="+", required=True, help="worker=USD")
    parser.add_argument("--cap", type=float, default=35.0)
    args = parser.parse_args()
    shares = {}
    for item in args.allowances:
        worker, _, usd = item.partition("=")
        shares[worker] = float(usd)
    print(json.dumps(spend(args.ledgers, shares, args.cap), indent=2))


if __name__ == "__main__":
    main()
