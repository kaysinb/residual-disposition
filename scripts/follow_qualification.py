"""The follow-up's qualification of D* and unfiltered PD (preregistration §13.5 (a); barrier B2) on
the API, from recorded packages only (``residual_disposition.follow_qualification``).

    uv run python scripts/follow_qualification.py [--send]

Without ``--send`` it prints the plan (readings by reader, purpose and decoder), the prompts'
sha256, the due requests and the estimate. With ``--send``: every due interpreter attempt (a
slot's request, then its one retry once the parser refuses the first answer), then every due
judge attempt of the parsed hypotheses, each sent once and cached (``<DATA_DIR>/cache/summarizer``,
``<DATA_DIR>/cache/judge``), every send reserved on the meter item ``follow/base``'s API session
(``follow_money``) under ``follow/qualification`` at the price of the tier it is sent on, until
nothing is due; an attempt whose send failed is not sent again in this run (rerun to resend).

The interpreter reads on the flex tier with the automatic fallback to the standard one
(``judge.FlexFallback``: a window of 100, a share of HTTP 503 of at least 0.9, 15 minutes of
silence), as Stage 3's interpreters did; its journal (``tier_fallback.jsonl``) is in the run.
The judge reads on the standard tier, as B2 fixes it. A tier is never part of a request's body,
key or salt: it is recorded with each cached answer. The run directory
(``runs/follow_qualification/<run>``) holds the inputs, each answer's tier and the record,
recomputed from the cache: each reading's counted hypotheses, the pooled rates and the floors
(``qualification.json``), pending while a slot is.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import hashlib
import json
import sys
from typing import Any

from residual_disposition import continuation_readers as readers
from residual_disposition import follow_qualification as Q
from residual_disposition.continuation_kstar import find_run
from residual_disposition.judge import ChatBackend, JudgeCache, JudgeItem, fetch, load_targets
from residual_disposition.paths import REPO_ROOT, data_root

ITEM, LABEL = "follow/base", "follow/qualification"
METER_LOG = "ledgers/continuation_meter.jsonl"
# the estimate's units (the Stage-3 diagnostic's, measured there): characters per token, the
# interpreter's thinking tokens, the margin, an answer's tokens
CHARS, THINKING, MARGIN, ANSWER = 2.2, 1455, 1.3, 40


def plan() -> list[dict[str, Any]]:
    """Every reading with its package directory, evidence's sha256 and attempts."""
    from residual_disposition.manifests import committed_json

    confirmed = committed_json(REPO_ROOT / "manifests" / "follow_confirmations.json",
                               "the confirmations")["confirmed"]  # fmt: skip
    new = Q.new_packages()
    out = []
    for r in Q.readings(confirmed):
        directory = Q.package_dir(r, find_run, new)
        text = Q.evidence(r, directory)
        out.append({"reading": r, "directory": str(directory),
                    "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "attempts": Q.attempts(r, text)})  # fmt: skip
    return out


def state(cells: list[dict[str, Any]], caches: dict[str, Any], judges: dict[str, Any],
          targets: dict[str, Any]) -> list[dict[str, Any]]:  # fmt: skip
    """Every slot's hypothesis and labels from the caches."""
    from residual_disposition.summarizer import parse_hypothesis

    rows = []
    for c in cells:
        r, kind = c["reading"], Q.KINDS[c["reading"]["reader"]]
        slots = []
        for i, tries in enumerate(c["attempts"]):
            hyp = Q.answer(caches["summary"], tries, parse_hypothesis)
            lab = None
            if "value" in hyp:
                item = JudgeItem(f"{r['salt']}#slot={i}", r["behavior"], targets[r["behavior"]],
                                 hyp["value"])  # fmt: skip
                lab = Q.labels(item, judges[kind], caches["judge"])
            slots.append({"hypothesis": hyp, "labels": lab})
        rows.append({"reading": r, "slots": slots})
    return rows


def due(rows: list[dict[str, Any]]) -> list[Any]:
    """The attempts due now: unanswered interpreter attempts, then unanswered label attempts."""
    out = []
    for row in rows:
        for s in row["slots"]:
            if "due" in s["hypothesis"]:
                out.append(s["hypothesis"]["due"])
            elif s["labels"] is not None:
                for kind in ("strict", "lenient"):
                    if "due" in s["labels"][kind]:
                        out.append(s["labels"][kind]["due"])
    return list({r.key: r for r in out}.values())


def estimate(requests: list[Any], specs: dict[str, Any], interpreter_keys: set[str]) -> float:
    """Characters over ``CHARS`` in; an interpreter request's thinking with its margin and its
    answer out; a judge request the judge configuration's expected tokens with the margin."""
    from residual_disposition.config import load_config
    from residual_disposition.judge import request_chars

    cost = load_config(readers.JUDGE, ["tracking.wandb.enabled=false"]).judge.cost
    usd = 0.0
    for r in requests:
        spec = specs[r.body["model"]]
        if r.key in interpreter_keys:
            tokens_in, tokens_out = request_chars(r) / CHARS, THINKING * MARGIN + ANSWER
        else:
            tokens_in = request_chars(r) / float(cost.chars_per_token)
            tokens_out = MARGIN * (int(cost.expected_output_tokens[r.kind])
                                   + spec.expected_thinking_tokens)  # fmt: skip
        usd += (tokens_in * spec.input_usd_per_mtok + tokens_out * spec.output_usd_per_mtok) / 1e6
    return usd


FLEX = {"window": 100, "min_503_share": 0.9, "silence_min": 15}  # Stage 3's interpreters'


def journal_writer(path: Any) -> Any:
    """Append each fallback event to ``path`` as a JSON line, flushed and synced."""
    import os

    def write(event: dict[str, Any]) -> None:
        with path.open("a") as f:
            f.write(json.dumps(event) + "\n")
            f.flush()
            os.fsync(f.fileno())

    return write


def tier_factors() -> dict[str, float]:
    """Each tier's price factor (the judge configuration's discounts)."""
    from residual_disposition.config import load_config
    from residual_disposition.judge import tier_factor

    cost = load_config(readers.JUDGE, ["tracking.wandb.enabled=false"]).judge.cost
    return {t: tier_factor(t, batch_discount=float(cost.batch_discount),
                           flex_discount=float(cost.flex_discount))
            for t in ("standard", "flex", "batch")}  # fmt: skip


def backend(ledger: Any, specs: dict[str, Any], *, interpreter: bool, journal: Any = None) -> Any:
    """A synchronous backend, every attempt reserved on the item's ledger under ``LABEL`` at its
    own model's price on the tier it is sent on: the interpreter's on flex with the automatic
    fallback (``FLEX``, its events in ``journal``), the judge's on the standard tier."""
    import os

    import openai

    from residual_disposition import stage3_money
    from residual_disposition.config import load_config
    from residual_disposition.judge import FlexFallback

    base, key = os.environ.get("JUDGE_API_BASE"), os.environ.get("JUDGE_API_KEY")
    if not base or not key:
        raise SystemExit("Set JUDGE_API_BASE and JUDGE_API_KEY (.env)")
    sync = load_config(readers.JUDGE, ["tracking.wandb.enabled=false"]).judge.sync
    client = openai.OpenAI(api_key=key, base_url=base, timeout=sync.timeout_s, max_retries=0)
    fallback = FlexFallback(window=FLEX["window"], min_503_share=FLEX["min_503_share"],
                            silence_s=60.0 * FLEX["silence_min"],
                            journal=journal) if interpreter else None  # fmt: skip
    inner = ChatBackend(client, concurrency=int(sync.concurrency),
                        max_retries=int(sync.max_retries), backoff_s=float(sync.backoff_s),
                        backoff_max_s=float(sync.backoff_max_s),
                        service_tier="flex" if interpreter else None,
                        fallback=fallback)  # fmt: skip
    factors = tier_factors()
    tier = "flex" if interpreter else "standard"
    return stage3_money.ChainedBackend(inner, ledger, lambda r: specs[r.body["model"]],
                                       price_factor=factors[tier], tier_factors=factors,
                                       label=LABEL)  # fmt: skip


def record(rows: list[dict[str, Any]], cells: list[dict[str, Any]]) -> dict[str, Any]:
    """The qualification's record: each reading's count, the pooled rates and the floors."""
    from residual_disposition.continuation_rules import follow_rules

    counted = [{"reading": r["reading"], "count": Q.reading_count(r["slots"])} for r in rows]
    rates = Q.pooled(counted)
    return {"readings": [{"salt": c["reading"]["salt"], "count": c["count"]} for c in counted],
            "evidence": {c["reading"]["salt"]: c["evidence_sha256"] for c in cells},
            "rates": {"/".join(str(x) for x in k): (None if v is None else str(v))
                      for k, v in rates.items()},
            "floors": Q.floors(rates, follow_rules().data)}  # fmt: skip


def write(path: Any, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")


def inputs(cells: list[dict[str, Any]], judges: dict[str, Any]) -> dict[str, Any]:
    """What the run reads: every reading with its package directory and evidence's sha256, the
    prompts' and the judges' frozen versions."""
    prompts = {}
    for c in cells:
        p, settings = Q.prompt_and_settings(c["reading"])
        prompts[p.version] = {
            "sha256": p.sha256,
            "model": settings.model.id,
            "max_tokens": settings.max_tokens,
            "samples": settings.samples,
        }
    return {
        "readings": [
            {**c["reading"], "directory": c["directory"], "evidence_sha256": c["evidence_sha256"]}
            for c in cells
        ],
        "prompts": prompts,
        "judges": {
            k: {"rubric_version": j.rubric_version, "model": j.model.id} for k, j in judges.items()
        },
        "interpreter_tier": {"tier": "flex", "fallback": FLEX},
        "judge_tier": "standard",
    }


def tiers(rows: list[dict[str, Any]], caches: dict[str, Any]) -> dict[str, Any]:
    """Each answered attempt's tier, as its cached answer records it, by reading and slot."""
    out = {}
    for row in rows:
        salt = row["reading"]["salt"]
        for i, s in enumerate(row["slots"]):
            key = s["hypothesis"].get("key")
            if key is not None:
                out[f"{salt}#slot={i}"] = (caches["summary"].entry(key) or {}).get("service_tier")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args(argv)
    cells = plan()
    judges = {kind: readers.judge_settings(readers.RUBRICS[kind]) for kind in ("lens", "pd")}
    interp = {kind: readers.interpreter(kind)[1] for kind in ("lens", "pd")}
    specs = {s.id: s for s in [*(i.model for i in interp.values()),
                               *(j.model for j in judges.values()),
                               *(j.model_for("lenient") for j in judges.values())]}  # fmt: skip
    targets = load_targets(sorted({c["reading"]["behavior"] for c in cells}))
    caches = {"summary": JudgeCache(data_root() / "cache" / "summarizer"),
              "judge": JudgeCache(data_root() / "cache" / "judge")}  # fmt: skip
    interpreter_keys = {a.key for c in cells for tries in c["attempts"] for a in tries}
    print(dict(collections.Counter((c["reading"]["reader"], c["reading"]["purpose"],
                                    c["reading"]["decoder"]) for c in cells)))  # fmt: skip
    for c in cells[:1]:
        print("prompt", Q.prompt_and_settings(c["reading"])[0].version)
    rows = state(cells, caches, judges, targets)
    now = due(rows)
    print(f"due now: {len(now)}; estimate ${estimate(now, specs, interpreter_keys):.2f}")
    if not args.send:
        return 0
    from datetime import UTC, datetime

    from residual_disposition import follow_money
    from residual_disposition.paths import runs_root

    run = (
        runs_root()
        / "follow_qualification"
        / ("follow-qualification-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    run.mkdir(parents=True, exist_ok=False)
    write(run / "inputs.json", inputs(cells, judges))
    _, ledger = follow_money.money(data_root(), ITEM, data_root() / METER_LOG)
    atexit.register(ledger.close)  # an interrupted run's session settled at its exit
    ledger.source = LABEL
    backends = {"summary": backend(ledger, specs, interpreter=True,
                                   journal=journal_writer(run / "tier_fallback.jsonl")),
                "judge": backend(ledger, specs, interpreter=False)}  # fmt: skip
    for name, b in backends.items():
        b.reconcile(caches[name])
    tried: set[str] = set()
    while now := [r for r in due(state(cells, caches, judges, targets)) if r.key not in tried]:
        ledger.check(estimate(now, specs, interpreter_keys))  # the worst case: all on standard
        for name, batch in (
            ("summary", [r for r in now if r.key in interpreter_keys]),
            ("judge", [r for r in now if r.key not in interpreter_keys]),
        ):
            if batch:
                fetch(batch, backends[name], caches[name])
        tried |= {r.key for r in now}
    charged = ledger.close()
    rows = state(cells, caches, judges, targets)
    write(run / "tiers.json", tiers(rows, caches))
    result = {**record(rows, cells), "charged": charged}
    write(run / "qualification.json", result)
    print(f"charged {charged}; record {run}")
    print(json.dumps(result["floors"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
