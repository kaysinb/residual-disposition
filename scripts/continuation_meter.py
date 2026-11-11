"""The continuation's meter (preregistration §12.8), operated from this machine, its one writer.

    uv run python scripts/continuation_meter.py status
    uv run python scripts/continuation_meter.py admit <item> --spent-elsewhere USD --storage USD
    uv run python scripts/continuation_meter.py start <item> [--price USD_PER_HOUR [--receipt FILE]]
    uv run python scripts/continuation_meter.py limits <item> [--price USD_PER_HOUR]
    uv run python scripts/continuation_meter.py charge <item> [--gpu-usd USD] [--api-usd USD] \
        [--label TEXT]
    uv run python scripts/continuation_meter.py verify-receipt FILE --item ITEM \
        --price USD_PER_HOUR --hours H [--worker WORKER --pod-id POD_ID]
    uv run python scripts/continuation_meter.py split <item> --share WORKER=SHARE ... \
        [--api-share WORKER=SHARE ...]
    uv run python scripts/continuation_meter.py session-open <item> --worker WORKER \
        --pod-id POD_ID --deadline ISO --price USD_PER_HOUR --hours H [--api-usd USD] \
        --receipt FILE
    uv run python scripts/continuation_meter.py session-close <item> --worker WORKER \
        --pod-id POD_ID --gpu-usd USD [--api-usd USD]
    uv run python scripts/continuation_meter.py stage3-start stage3/<behavior> --space a|c
    uv run python scripts/continuation_meter.py admit-panel --spent-elsewhere USD --storage USD
    uv run python scripts/continuation_meter.py panel-contracts
    uv run python scripts/continuation_meter.py transfer <item> --from gpu --to api --usd USD \
        --reason TEXT
    uv run python scripts/continuation_meter.py migrate-cap --reason TEXT
    uv run python scripts/continuation_meter.py migrate-allocations
    uv run python scripts/continuation_meter.py recount --table FILE --reason TEXT
    uv run python scripts/continuation_meter.py complete <item>
    uv run python scripts/continuation_meter.py drop <item> <reason>

The frozen allocations (``configs/continuation/allocations.yaml``), the cap and reserve and the drop
reasons come from the frozen rules, which must match their sha256 in the Stage-0 specification. The
log (``data/ledgers/continuation_meter.jsonl`` by default) is locked for each operation and records
the allocations' hash in its header. Every command prints JSON. With ``--price`` (the Pod's actual
hourly price), ``start`` and ``limits`` also give the Pod's hard deadline in hours from its creation
(``ExecutionLimits.pod_hours``). ``transfer`` moves unspent money between a running item's GPU and
API parts with its reason; the item's reservation, and so the meter's total, stay as they are
(``AllocationMeter.transfer``). The cap is the rules' (``meter.cap_usd``): a log
begun
under a lower cap records nothing until ``migrate-cap`` raises it to the rules' cap with its reason
(``AllocationMeter.migrate_cap``), keeping its spent money, reservations and history; ``status``
shows the cap in force, the rules' and the one the log was begun under. The allocations only grow,
by the file's append-only ``migrations`` (C15 adds stage5_dpo below stage5, C16 the
strong items): a log begun under the allocations before one records nothing until
``migrate-allocations`` records each migration in order (its hashes, added items and reason),
between Pod sessions; ``status`` lists the items still to be migrated in. A panel behavior's Stage
3 starts once, by ``stage3-start`` (§12.8): both its reservations live, every
earlier panel behavior's Stage 3 started, its lineage's one space recorded; no Pod session of
either of its items opens before it, and ``verify-receipt`` gives the Pod that space. With the
panel's phases (§12.8 as amended 2026-10-22), ``admit-panel`` admits at once the five
behaviors' preparation phases and the first behavior's follow-up at stress, the other follow-ups
protected at their allocations (refused whole, nothing reserved, over the cap);
``panel-contracts`` records the committed `manifests/stage3_panel_contracts.json` by its sha256,
once: no follow-up starts, and no later one is admitted (in the frozen order), before it.
``recount`` records the log's one API recount (``AllocationMeter.recount``): a JSON table of
items to API corrections (the ledger's cost less the provider's actual one), named in the log by
its sha256, lowers each item's charged API money and a finished item's final cost; ``status``
shows it.

A Pod is launched only on a live reservation: ``start --receipt`` writes a launch receipt (the
item, its limits, the price, the Pod's hours, the allocations' hash and the sha256 of the log
right after the start). Only Stage 1's item ran its Pods on one (``LAUNCH_RECEIPT_ITEMS``); every
later item runs them on session receipts, and a launch receipt of it authorizes nothing. On the
Pod, ``verify-receipt`` with ``--log`` naming the copy of that
log replays it and succeeds only if the copy is the one the receipt names, the receipt is for the
expected item, that item is running, unsplit and uncharged with exactly those limits, the price
is the receipt's, and the requested hours fit within the item's live GPU limit at that price
(the receipt's pod_hours must agree; they are never trusted on their own). A refused admission
exits with status 4, an overrun charge with 3, a failed verification with 1.

Two Pods share an item (Stage 2: Pod A and Pod B) through ``split``, right after ``start``: each
worker gets its share of both parts, or with ``--api-share`` its share of each part, and every later
charge names its worker (``charge --worker``), who may spend no more than its share.
``strong/stage5`` is split only by its frozen split, S5-A's GPU-only sublimit (worker ``s5-a``,
its frozen share of the GPU part, none of the API part; ``s5-b`` the rest; C16), the default for
it. Each Pod
session is authorized exclusively in the log: ``session-open``, with the Pod's actual
ID and the deadline set at its creation (both set when the Pod is created), its price
and hours within the worker's live GPU share and the API money it may commit (``--api-usd``)
within the worker's API share, writes the session's receipt (the sha256 of the log right after
it); it is refused while that worker has an open session. ``session-close`` settles the session's
paid cost as the worker's charge and closes it, so that the next may open. On the Pod,
``verify-receipt --worker --pod-id`` replays the log copy and succeeds only if the copy is the
receipt's, the worker's open session there is the receipt's (the Pod ID the Pod's own, the deadline,
the price, the hours, the API authorization), the deadline is still ahead and the requested hours
fit within the session's; it prints that session as the log holds it and its id (the sha256 of the
log right after it), which is where the Pod's chain takes its deadline and its API ceiling from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from residual_disposition.budget import (
    NO_READOUT,
    AllocationMeter,
    BudgetExhausted,
    load_allocations,
    load_migrations,
)
from residual_disposition.continuation_rules import (
    FALLBACK_GATE_CLOSED,
    drop_reasons,
    frozen_split,
    load_rules,
)
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
    """Why a Pod session's receipt does not authorize ``hours`` on Pod ``pod_id``:
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
    with m._transaction():  # the item's state and the session read together
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
# later item runs its Pods on session receipts, so a launch receipt of any other
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
        if item in m.issued:  # its Pods run on session receipts only
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
            "stage3": dict(m.stage3),  # each panel behavior's Stage-3 start: its one space
            "panel": dict(m.panel),  # the panel's admission and its contracts
            "recount": m.recounted,  # the log's one API recount, if recorded
        }


def fallback_gate() -> dict:
    """The DPO fallback's gate from the committed strong manifests (C16-2): the condition of
    ``fallback_gate_closed`` (§12.8), a continued candidate validated, or the continued branch
    closed unvalidated with no margin-passing DPO checkpoint."""
    from residual_disposition import continuation_strong as CS
    from residual_disposition.continuation_kstar import find_run
    from residual_disposition.paths import REPO_ROOT

    rules = load_rules(expected_hash=FROZEN["rules_sha256"])
    detector = CS.detector_of(CS.branch(CS.CONTINUED)["behavior"])
    return CS.fallback_gate(REPO_ROOT, rules, find_run, detector)


def panel_evidence(m: AllocationMeter, item: str) -> dict | None:
    """A panel phase's evidence for the meter: its behavior's entry in the
    committed panel contracts, read whole with every entry checked against what it binds
    (``continuation_strong.committed_bundle``), the contracts the meter recorded; None for an
    item that is no panel phase."""
    from residual_disposition import continuation_strong as CS
    from residual_disposition.budget import _stage3_behavior
    from residual_disposition.continuation_kstar import find_run
    from residual_disposition.paths import REPO_ROOT

    if m.panel_rows() is None or item not in {x for r in m.panel_rows() for x in r}:
        return None
    behavior = _stage3_behavior(item, m._get(item).kind)
    rules = load_rules(expected_hash=FROZEN["rules_sha256"])
    try:
        data = CS.committed_bundle(REPO_ROOT, rules, find_run)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"{item}: {error}") from error
    if data["sha256"] != m.panel["contracts"]:
        raise SystemExit(f"{item}: the committed contracts {data['sha256'][:12]} are not the ones "
                         f"the meter recorded ({m.panel['contracts']})")  # fmt: skip
    entry = data["behaviors"][behavior]
    return {"contracts": data["sha256"], "behavior": behavior, "status": entry["status"],
            "reading": entry["reading"]}  # fmt: skip


def worker_shares(pairs: list[str], flag: str) -> dict[str, float]:
    """``WORKER=SHARE`` pairs as a mapping, one per worker."""
    shares: dict[str, float] = {}
    for pair in pairs:
        worker, _, share = pair.partition("=")
        if not worker or worker in shares:
            raise SystemExit(f"{flag} {pair}: one WORKER=SHARE per worker")
        shares[worker] = float(share)
    return shares


def split_shares(
    m: AllocationMeter, item: str, share: list[str] | None, api_share: list[str] | None
) -> tuple[dict[str, float], dict[str, float] | None]:
    """The shares of a split: the given ones, or an item's frozen split (``strong/stage5``:
    S5-A's sublimit, ``continuation_rules.frozen_split``), which given shares must equal."""
    given = None if share is None else worker_shares(share, "--share")
    given_api = None if api_share is None else worker_shares(api_share, "--api-share")
    frozen = frozen_split(item, m.parts(item)[0])
    if frozen is None:
        if given is None:
            raise SystemExit(f"split {item}: --share WORKER=SHARE for each worker")
        return given, given_api
    for name, mine, theirs in (
        ("--share", given, frozen[0]),
        ("--api-share", given_api, frozen[1]),
    ):
        if mine is not None and (
            set(mine) != set(theirs) or any(abs(mine[w] - theirs[w]) > 1e-12 for w in mine)
        ):
            raise SystemExit(f"split {item}: {name} must be its frozen split {theirs}")
    return frozen


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
    p.add_argument("--share", action="append", default=None, metavar="WORKER=SHARE")
    p.add_argument(
        "--api-share",
        action="append",
        default=None,
        metavar="WORKER=SHARE",
        help="each worker's share of the API part, if not its --share (C16)",
    )
    p = sub.add_parser("session-open")
    p.add_argument("item")
    p.add_argument("--worker", required=True)
    p.add_argument("--pod-id", required=True, help="the Pod's actual ID, from its creation")
    p.add_argument("--deadline", required=True, help="the deadline set at the Pod's creation, ISO")
    p.add_argument("--price", type=float, required=True)
    p.add_argument("--hours", type=float, required=True)
    p.add_argument("--api-usd", type=float, default=0.0, help="the API money the session may use")
    p.add_argument("--receipt", type=Path, required=True)
    p = sub.add_parser("stage3-start", help="a panel behavior's Stage-3 start, recorded once")
    p.add_argument("item", help="stage3/<behavior>")
    p.add_argument("--space", required=True, choices=("a", "c"), help="its lineage's one space")
    p = sub.add_parser("admit-panel", help="the panel's phases, admitted at once")
    p.add_argument("--spent-elsewhere", type=float, required=True)
    p.add_argument("--storage", type=float, required=True)
    sub.add_parser("panel-contracts", help="record the committed panel contracts, once")
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
    p = sub.add_parser("recount", help="the log's one API recount, from a table of corrections")
    p.add_argument("--table", type=Path, required=True, help="JSON: {item: API correction USD}")
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
        if args.item == "strong/stage5_dpo":  # only once the fallback's gate is open (§12.8)
            gate = fallback_gate()
            if gate["status"] != "open":
                raise SystemExit(f"admit {args.item}: the DPO fallback's gate is {gate}")
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
            out["session_id"] = data["log_sha256"]  # the log right after it: unique
            out["stage3"] = m.stage3_of(args.item)  # a Stage-3 item's: its lineage's space
            out["panel_contracts"] = m.panel["contracts"]  # the panel's one record
        print(json.dumps(out))
        return 1 if problems else 0
    elif args.command == "limits":
        out = limits_json(m.limits(args.item), args.price)
    elif args.command == "split":
        parts = m.split(args.item, *split_shares(m, args.item, args.share, args.api_share))
        out = {w: limits_json(limits, None) for w, limits in parts.items()}
    elif args.command == "session-open":
        out = m.open_session(  # the receipt formed under the same lock as the session
            args.item, args.worker, pod_id=args.pod_id, deadline=args.deadline,
            price=args.price, hours=args.hours, api_usd=args.api_usd,
            result=lambda event: session_receipt(m, event),
        )  # fmt: skip
        args.receipt.write_text(json.dumps(out, indent=1) + "\n")
    elif args.command == "admit-panel":
        d = m.admit_panel(spent_elsewhere=args.spent_elsewhere, storage=args.storage)
        out = {"panel": m.panel_envelope(), "allowed": d.allowed, "total": d.total, "cap": d.cap,
               "terms": d.terms}  # fmt: skip
        if not d.allowed:
            print(json.dumps(out, indent=1))
            return 4
    elif args.command == "panel-contracts":
        from residual_disposition import continuation_strong as CS
        from residual_disposition.continuation_kstar import find_run
        from residual_disposition.paths import REPO_ROOT

        path = CS.contracts_path(REPO_ROOT)
        rules = load_rules(expected_hash=FROZEN["rules_sha256"])
        try:  # read whole, every entry checked against what it binds
            data = CS.committed_bundle(REPO_ROOT, rules, find_run)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            raise SystemExit(f"panel-contracts: {error}") from error
        text = path.read_bytes().decode()
        if hashlib.sha256(text.encode()).hexdigest() != data["sha256"]:
            raise SystemExit(f"panel-contracts: {path.name} changed while it was read")
        m.panel_contracts(data["sha256"], text)  # its text journaled: the entries' evidence
        out = {"contracts": path.name, "sha256": data["sha256"],
               "entries": m.panel["entries"]}  # fmt: skip
    elif args.command == "stage3-start":
        event = m.stage3_start(args.item, args.space)
        out = {k: event[k] for k in ("item", "behavior", "space")}
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
    elif args.command == "recount":
        raw = args.table.read_bytes()
        table = json.loads(raw)
        if not isinstance(table, dict):
            raise SystemExit(f"recount: {args.table} is not a mapping of items to corrections")
        sha = hashlib.sha256(raw).hexdigest()
        corrections = m.recount(table, source_sha256=sha, reason=args.reason)
        out = {"source_sha256": sha, "corrections": corrections, "spent": m.spent(),
               "outstanding": m.outstanding()}  # fmt: skip
    elif args.command == "complete":
        evidence = None  # a panel preparation's: its entry in the recorded contracts
        if args.item in {x for r in (m.panel_rows() or []) for x in r[:2]}:
            evidence = panel_evidence(m, args.item)
        out = {"item": args.item, "final_usd": m.complete(args.item, evidence)}
    else:
        if args.reason == FALLBACK_GATE_CLOSED:  # its condition, from the committed manifests
            gate = fallback_gate()
            if gate["status"] != "closed":
                raise SystemExit(f"drop {args.item} {args.reason}: the gate is {gate}")
        evidence = None  # a follow-up's release: its entry with no reading at all
        if args.reason == NO_READOUT:
            evidence = panel_evidence(m, args.item)
            if evidence is None or evidence["reading"] is not None:
                raise SystemExit(f"drop {args.item} {args.reason}: its entry is {evidence}")
        m.drop(args.item, args.reason, evidence)
        out = {"item": args.item, "dropped": args.reason, "evidence": evidence}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
