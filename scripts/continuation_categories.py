"""The category grades of a certified lineage's counted hypotheses (preregistration §12.3.3, the
target check; §12.4.5 as amended 2026-10-24 and 2026-10-25), on the laptop.

    uv run python scripts/continuation_categories.py <behavior> --lens-reading <run> \\
        --lens-judge <run> --pd-reading <run> --pd-judge <run> [--send]

- **What is graded.** Every hypothesis of the lineage's lens and PD-clean readings that rule C
  counts, by its judge run's own-target primary label (lenient yes, strict not
  ``avoids_behavior``).
- **The instrument.** ``category-v1`` (``continuation_judge.category_request``): the frozen
  definitions (``continuation_readers.DEFINITIONS_SHA256``), gemini-3.8-flash at its frozen
  thinking level, its cap, on the standard tier, through the category cache
  (``<DATA_DIR>/cache/category``).
- **Slots.** Each hypothesis's grade is its first answer the frozen parser accepts, from its
  request and one retry under its key and ``RETRY``; two refusals leave it out, and the
  readers' verdicts then wait for it.
- **Money.** Every request is reserved on the lineage's chained ledger (``stage3_money``: its base,
  then its strong part) before it is sent; the meter books what the parts committed at the end.
- **A completed cohort.** When a committed completion specification reads the lens reading
  (§12.4.5, 2026-10-27), the cohort's references that rule C counts by their completed strict
  labels are graded too, and no count is printed.
- **Output.** A run of experiment ``category_stage3``: ``category_inputs.json`` and
  ``category_labels.jsonl`` (``{response_id, request_key, categories}``), the formats the
  readers' verdicts read (``continuation_readers.category_grades``).

Without ``--send`` it prints the number of requests and their estimated cost only.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from residual_disposition import continuation_completion as C
from residual_disposition import continuation_readers as E
from residual_disposition import stage3_money
from residual_disposition.config import load_config
from residual_disposition.continuation_definitions import load_definitions
from residual_disposition.continuation_judge import (
    CATEGORY_VERSION,
    category_request,
    parse_category,
)
from residual_disposition.continuation_kstar import find_run
from residual_disposition.judge import (
    ChatBackend,
    ChatRequest,
    JudgeCache,
    MalformedOutput,
    estimate_cost,
    fetch,
    load_targets,
    request_key,
)
from residual_disposition.paths import REPO_ROOT, data_root
from residual_disposition.records import (
    JudgeLabel,
    ToolHypothesis,
    read_responses,
    read_table,
)
from residual_disposition.summarizer import RETRY
from residual_disposition.tracking import Run

METER_LOG = Path("ledgers") / "continuation_meter.jsonl"  # under the data root
HEADROOM = 1.3  # the estimate + 30 % must fit in what the parts have left


def counted_texts(behavior: str, pairs: list[tuple[str, str]]) -> dict[str, str]:
    """{response id: text} of every hypothesis of the (reading run, judge run) pairs that rule C
    counts by its own-target primary label."""
    out: dict[str, str] = {}
    for reading, judge in pairs:
        run, labels_run = find_run(reading), find_run(judge)
        if run is None or labels_run is None:
            raise SystemExit(f"{reading} or {judge} is not here")
        texts = read_responses([run], ToolHypothesis.table)
        ids = {h.response_id for h in read_table([run], ToolHypothesis.table)}
        for lab in read_table([labels_run], JudgeLabel.table):
            if lab.stage != "primary" or lab.response_id not in ids:
                continue
            if lab.target_behavior_id != behavior:
                raise SystemExit(f"{lab.response_id}: a label for {lab.target_behavior_id}")
            if lab.lenient_label is True and lab.strict_label != "avoids_behavior":
                out[lab.response_id] = texts[lab.response_id]
    return out


def completed_texts(spec: dict[str, Any], behavior: str, reading: str, judge: str) -> dict:
    """The cohort's references that rule C counts by their completed labels."""
    run, labels_run = find_run(reading), find_run(judge)
    strict = {lab.response_id: lab.strict_label
              for lab in read_table([labels_run], JudgeLabel.table)
              if lab.stage == "primary" and lab.target_behavior_id == behavior}  # fmt: skip
    return C.counted_completions(spec, C.cache_of(spec, find_run),
                                 read_responses([run], ToolHypothesis.table),
                                 load_targets([behavior])[behavior], strict)  # fmt: skip


def instrument() -> tuple[dict[str, Any], Any]:
    """The frozen definitions and the grader's model."""
    definitions, _ = load_definitions(expected_hash=E.DEFINITIONS_SHA256)
    j = load_config(E.JUDGE, ["tracking.wandb.enabled=false"]).judge
    return definitions, E._spec(next(m for m in j.models if m.id == E.CATEGORY_MODEL))


def retry_of(request: ChatRequest) -> ChatRequest:
    return ChatRequest(request_key(f"{CATEGORY_VERSION}{RETRY}", request.body), request.kind,
                       request.rubric_version, request.body)  # fmt: skip


def refused(completion: Any) -> bool:
    if completion.error is not None:
        return False
    try:
        parse_category(completion.text)
    except MalformedOutput:
        return True
    return False


def backend(ledger: Any, model: Any, label: str = "category") -> Any:
    """The synchronous standard-tier backend of the judge config, every attempt reserved on the
    lineage's chained ledger under ``label``."""
    import openai

    base, key = os.environ.get("JUDGE_API_BASE"), os.environ.get("JUDGE_API_KEY")
    if not base or not key:
        raise SystemExit("Set JUDGE_API_BASE and JUDGE_API_KEY (.env)")
    sync = load_config(E.JUDGE, ["tracking.wandb.enabled=false"]).judge.sync
    client = openai.OpenAI(api_key=key, base_url=base, timeout=sync.timeout_s, max_retries=0)
    inner = ChatBackend(client, concurrency=int(sync.concurrency),
                        max_retries=int(sync.max_retries), backoff_s=float(sync.backoff_s),
                        backoff_max_s=float(sync.backoff_max_s), service_tier=None)  # fmt: skip
    return stage3_money.ChainedBackend(inner, ledger, lambda r: model, label=label)


def tracking() -> dict[str, Any]:
    t = OmegaConf.to_container(load_config("base").tracking, resolve=True)
    t.setdefault("wandb", {})["enabled"] = False
    return t


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("behavior")
    for name in ("lens-reading", "lens-judge", "pd-reading", "pd-judge"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args(argv)
    texts = counted_texts(args.behavior, [(args.lens_reading, args.lens_judge),
                                          (args.pd_reading, args.pd_judge)])  # fmt: skip
    # a lineage's completed blocked strict labels (§12.4.5, 2026-10-27): its newly counted
    # hypotheses graded beside the others, the counts sealed
    spec = C.spec_of_reading(REPO_ROOT, args.lens_reading)
    if spec is not None:
        texts |= completed_texts(spec, args.behavior, args.lens_reading, args.lens_judge)
    definitions, model = instrument()
    requests = {rid: category_request(text, definitions, model,
                                      max_tokens=E.CATEGORY_MAX_TOKENS)
                for rid, text in sorted(texts.items())}  # fmt: skip
    cache = JudgeCache(data_root() / "cache" / "category")
    est = estimate_cost(requests.values(), model,
                        cached={k for k in (r.key for r in requests.values()) if cache.entry(k)},
                        chars_per_token=4.5, expected_output_tokens={"category": 10},
                        price_factor=1.0)  # fmt: skip
    if spec is None:
        print(f"{len(requests)} requests, {est.to_send} to send, ~${est.usd:.4f}")
    else:
        print("sealed: the counted hypotheses of a completed cohort are not counted here")
    if not args.send:
        print("Dry run: nothing was sent. Pass --send to grade.")
        return 0
    _, ledger = stage3_money.money(data_root(), args.behavior, data_root() / METER_LOG)
    atexit.register(ledger.close)  # an interrupted run's sessions settled at its exit
    ledger.check(HEADROOM * est.usd)
    cfg = OmegaConf.create({"experiment": "category_stage3", "name": "category", "step": "category",
                            "params": {"behavior": args.behavior, "sources": vars(args)},
                            "category": {"cache_dir": None}, "tracking": tracking()})  # fmt: skip
    with Run(cfg) as run:
        ledger.source = run.run_id
        chained = backend(ledger, model)
        chained.reconcile(cache)
        completions, _ = fetch(list(requests.values()), chained, cache)
        slot = {rid: r for rid, r in requests.items()}
        again = {rid: retry_of(r) for rid, r in requests.items() if refused(completions[r.key])}
        if again:
            more, _ = fetch(list(again.values()), chained, cache)
            completions.update(more)
            slot.update(again)
        inputs = {"version": CATEGORY_VERSION, "definitions_sha256": E.DEFINITIONS_SHA256,
                  "model": {"id": model.id, "reasoning_effort": model.reasoning_effort},
                  "service_tier": E.STANDARD, "max_tokens": E.CATEGORY_MAX_TOKENS}  # fmt: skip
        run.path(E.CATEGORY_INPUTS).write_text(json.dumps(inputs, indent=2) + "\n")
        graded = 0
        with run.path(E.CATEGORY_LABELS).open("w") as f:
            for rid, request in slot.items():
                completion = completions[request.key]
                if completion.error is not None or refused(completion):
                    continue
                grade = sorted(parse_category(completion.text))
                f.write(json.dumps({"response_id": rid, "request_key": request.key,
                                    "categories": grade}) + "\n")  # fmt: skip
                graded += 1
        charged = ledger.close()
        run.summarize({"requests": len(requests), "retried": len(again), "graded": graded,
                       "charged": charged})  # fmt: skip
    if spec is None:
        print(f"{graded} of {len(requests)} graded, {len(again)} retried; written to {run.dir}")
    else:
        print(f"graded; written to {run.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
