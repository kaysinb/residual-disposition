"""The completion of a certified lineage's blocked strict labels (preregistration §12.4.5,
amendment of 2026-10-27; ``continuation_completion``), on the laptop, in this order:

    uv run python scripts/stage3_completion.py validate <lineage> [--send]
    uv run python scripts/stage3_completion.py replace <lineage> [--send]
    uv run python scripts/stage3_completion.py issue <lineage>
    uv run python scripts/stage3_completion.py import <lineage> --export <labels-<set>.jsonl>
    uv run python scripts/stage3_completion.py provenance <lineage>

- ``validate``: the replacement instrument's one validation on the specification's reference
  set; ``--send`` sends each item's request, then the retry of each answer the frozen strict
  parser refused. Once no slot is open it prints the result and the criteria.
- ``replace``: after a passed validation, the same for each cohort request; it prints how many
  slots are answered, refused twice (for the blind review) or still open.
- ``issue``: the blind review's page (``review.html``) and its private record (``issued.json``),
  once, for every cohort request after a failed validation, else those refused twice.
- ``import``: the page's export, once, exactly the issued items.
- ``provenance``: writes ``manifests/stage3_blocked_completion_<lineage>_provenance.json`` once,
  to be committed before the readers' verdicts.

Every send is reserved on the lineage's chained ledger (``stage3_money``: its base, then its
strong part) and logged with its request's own model in ``requests.jsonl`` under
``<DATA_DIR>/stage3_blocked_completion/<lineage>``. A send that fails is not cached: rerun the
command, which sends only what is still due. Nothing printed holds a reply or a label value.
Without ``--send``, ``validate`` and ``replace`` print what is due and its estimated cost only.
"""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
from pathlib import Path
from typing import Any

from residual_disposition import continuation_completion as C
from residual_disposition import continuation_readers as E
from residual_disposition import stage3_money
from residual_disposition.config import load_config
from residual_disposition.continuation_kstar import find_run
from residual_disposition.continuation_panel import Pending
from residual_disposition.judge import estimate_cost, fetch, load_targets, rubric_set
from residual_disposition.paths import REPO_ROOT, data_root
from residual_disposition.records import JudgeLabel, ToolHypothesis, read_responses, read_table

METER_LOG = Path("ledgers") / "continuation_meter.jsonl"  # under the data root
HEADROOM = 1.3  # the estimate + 30 % must fit in what the parts have left


def script(name: str) -> Any:
    loader = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def context(lineage: str) -> dict[str, Any]:
    """The committed specification, its instrument, the lens judging's cache, the lens
    reading's texts, the target and the lens judging's own-target strict labels."""
    spec = C.committed_spec(lineage)
    if spec is None:
        raise SystemExit(f"{lineage}: no committed completion specification")
    reading = find_run(spec["sources"]["lens_reading"])
    judged = find_run(spec["sources"]["lens_judge"])
    if reading is None or judged is None:
        raise SystemExit("the specification's lens reading or judging is not here")
    behavior = spec["behavior"]
    strict = {lab.response_id: lab.strict_label for lab in read_table([judged], JudgeLabel.table)
              if lab.stage == "primary" and lab.target_behavior_id == behavior}  # fmt: skip
    return {"spec": spec, "instrument": C.settings(spec), "cache": C.cache_of(spec, find_run),
            "texts": read_responses([reading], ToolHypothesis.table),
            "target": load_targets([behavior])[behavior], "strict": strict}  # fmt: skip


def send(lineage: str, ctx: dict[str, Any], requests: dict[str, Any], purpose: str,
         go: bool) -> None:  # fmt: skip
    """The due attempts of ``requests`` ({reference: request}), each reserved and logged: the
    requests without an answer, then the retries of those the parser refused."""
    cache, model = ctx["cache"], ctx["instrument"].model
    of = {r.key: ref for ref, r in requests.items()} | {
        C.retry(r).key: ref for ref, r in requests.items()}  # fmt: skip
    due = C.due(cache, list(requests.values()))
    cost = load_config(E.JUDGE, ["tracking.wandb.enabled=false"]).judge.cost
    est = estimate_cost(due, model, chars_per_token=float(cost.chars_per_token),
                        expected_output_tokens=dict(cost.expected_output_tokens),
                        price_factor=1.0)  # fmt: skip
    print(f"{purpose}: {len(requests)} slots, {len(due)} attempts due, ~${est.usd:.4f}")
    if not due or not go:
        if due:
            print("Dry run: nothing was sent. Pass --send to send.")
        return
    _, ledger = stage3_money.money(data_root(), ctx["spec"]["behavior"], data_root() / METER_LOG)
    atexit.register(ledger.close)  # an interrupted run's sessions settled at its exit
    ledger.check(HEADROOM * est.usd)
    ledger.source = f"completion/{lineage}"
    backend = script("continuation_categories").backend(ledger, model, label=f"judge/{purpose}")
    backend.reconcile(cache)
    log = C.work_dir(lineage) / C.LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    tried: set[str] = set()
    failed = 0
    while due:
        completions, cached = fetch(due, backend, cache)
        with log.open("a") as f:
            for r in due:
                c = completions[r.key]
                f.write(json.dumps({
                    "request_id": r.key, "purpose": purpose, "kind": r.kind,
                    "rubric_version": r.rubric_version, "model": r.body["model"],
                    "reference": of[r.key], "from_cache": r.key in cached, "text": c.text,
                    "finish_reason": c.finish_reason, "usage": dict(c.usage), "error": c.error,
                }, ensure_ascii=False) + "\n")  # fmt: skip
        failed += sum(completions[r.key].error is not None for r in due)
        tried |= set(completions)
        # the retries now due; a failed send waits for the next run
        due = [r for r in C.due(cache, list(requests.values())) if r.key not in tried]
    if failed:
        print(f"{failed} sends failed (not cached): rerun to send them")
    print(f"charged: {ledger.close()}")


def validate(args: argparse.Namespace) -> int:
    ctx = context(args.lineage)
    spec = ctx["spec"]
    requests = {f"{k[0]} {k[1]}": r
                for k, r in C.validation_requests(spec, ctx["instrument"]).items()}  # fmt: skip
    send(args.lineage, ctx, requests, "completion-validation", args.send)
    try:
        found = C.validation(spec, ctx["cache"], ctx["instrument"])
    except Pending as error:
        print(f"validation open: {error}")
        return 0
    print(json.dumps({k: found[k] for k in ("reference_set", "result", "criterion_1",
                                            "criterion_2", "criterion_L")}))  # fmt: skip
    return 0


def replace(args: argparse.Namespace) -> int:
    ctx = context(args.lineage)
    spec, cache = ctx["spec"], ctx["cache"]
    found = C.validation(spec, cache, ctx["instrument"])
    if found["result"] != "pass":
        raise SystemExit("the validation failed: the whole cohort goes to the blind review")
    C.originals(spec, cache)
    cohort = C.cohort(spec, ctx["texts"], ctx["target"], ctx["instrument"])
    send(args.lineage, ctx, {k: r for k, (_, r) in cohort.items()}, "completion", args.send)
    statuses = [o["status"] for o in C.replacements(cache, cohort).values()]
    print(json.dumps({s: statuses.count(s) for s in ("answered", "refused", "open")}))
    return 0


def _frame(ctx: dict[str, Any]) -> None:
    """The specification's decoys, recomputed from the lens reading and its labels."""
    spec = ctx["spec"]
    hyps = {rid: (None, None, text) for rid, text in ctx["texts"].items()}
    frame = C.decoy_frame(spec, hyps, ctx["strict"])
    pinned = spec["human"]["decoy_frame"]
    if frame != {**{k: pinned[k] for k in ("distinct_texts", "references", "sha256")},
                 "decoys": spec["human"]["decoys"]}:  # fmt: skip
        raise SystemExit("the specification's decoys are not its frame's seeded draw")


def issue(args: argparse.Namespace) -> int:
    ctx = context(args.lineage)
    spec, cache = ctx["spec"], ctx["cache"]
    _frame(ctx)
    found = C.validation(spec, cache, ctx["instrument"])
    cohort = C.cohort(spec, ctx["texts"], ctx["target"], ctx["instrument"])
    outcomes = (C.replacements(cache, cohort) if found["result"] == "pass"
                else {k: {"status": "not sent"} for k in cohort})  # fmt: skip
    keys = C.routed(found, outcomes)
    if not keys:
        print("every cohort request is answered: no blind review")
        return 0
    items = C.review_items(spec, keys, ctx["texts"], ctx["target"])
    page_of = script("make_labeling_set").render_page
    from residual_disposition.judge import labeling_set_id

    page = page_of(labeling_set_id(items), items, rubric_set(E.RUBRICS["lens"]))
    record = C.issued(args.lineage, spec["sha256"], keys, items, page)
    work = C.work_dir(args.lineage)
    C.write_once(work / C.ISSUED, record)
    with (work / C.PAGE).open("x") as f:
        f.write(page)
    print(f"review {record['set_id']}: {len(items)} items; open {work / C.PAGE}")
    print(f"Do not open {work / C.ISSUED}. Then import the page's export with")
    print(f"  uv run python scripts/stage3_completion.py import {args.lineage} --export "
          f"<labels-{record['set_id']}.jsonl>")  # fmt: skip
    return 0


def import_(args: argparse.Namespace) -> int:
    print(json.dumps(C.import_review(args.lineage, args.export)))
    return 0


def provenance(args: argparse.Namespace) -> int:
    ctx = context(args.lineage)
    _frame(ctx)
    record = C.write_provenance(REPO_ROOT, ctx["spec"], ctx["cache"], ctx["texts"], ctx["target"],
                                ctx["strict"])  # fmt: skip
    print(json.dumps({"written": str(C.provenance_path(REPO_ROOT, args.lineage)),
                      "validation": record["validation"]["result"],
                      "reviewed": len(record["routed"]),
                      "requests": len(record["completed"])}))  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("validate", validate), ("replace", replace), ("issue", issue),
                     ("import", import_), ("provenance", provenance)):  # fmt: skip
        p = sub.add_parser(name)
        p.add_argument("lineage")
        p.set_defaults(fn=fn)
        if name in ("validate", "replace"):
            p.add_argument("--send", action="store_true")
        if name == "import":
            p.add_argument("--export", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except Pending as error:
        raise SystemExit(f"pending: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
