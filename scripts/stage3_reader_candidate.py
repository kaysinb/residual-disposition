"""A candidate lens-only reader (post-G method development, decisions, 2026-10-27) and its screen:
on the API from recorded packages only; it changes no Stage-3 reading, record or verdict, and it
reads no suppressed state.

    uv run python scripts/stage3_reader_candidate.py [--send]

The candidate's evidence, rendered deterministically from a lens package's stored steps
(``steps/layer{20,30}/<decoder>/p{0..4}``), both layers, positions 0–4:

- the 100 stored logit-lens tokens per position, most promoted first;
- every token among Patchscope's stored top 20 at any of the 30 frozen scales, once per position,
  as ``[token, best rank at any scale, the scales where it is in the top 20]`` (scale runs in the
  frozen order written ``a-b``), ordered by best rank, then by its first scale, then by token;
- no probabilities (they stay in the source files, whose sha256 the result keeps), no λ choice,
  no ``selected_tokens``, no behavior-specific filter; literal Unicode (``ensure_ascii=False``).

The interpreter, its settings and its synthesis are the readings' (``adl-lens-v1``'s request and
hygiene text verbatim); only the factual description of the evidence is updated
(``adl-lens-multiscale-v1``). 20 samples per cell; each hypothesis is judged by the frozen v4
judge and counted under rule C, with ``stage3_method_diagnostic``'s slot accounting (one retry,
only for a format failure; every due retry sent).

The screen: A and H of word_count_claim, countdown, tldr, eli5 (the readers' records) and of
example_link and pun_intended (their Stage-1 lens packages). The rule: every one of the six
behaviors' A has at least 16 of 20 and its H at most 2 of 20, all 20 slots of each of those
cells resolved; then the candidate is nominated, not qualified; otherwise its test ends. Also
reported, outside the rule: the four Stage-3 recipe controls and A's random-direction control of
the six behaviors. Every send is reserved on the behavior's chained ledger (``stage3_money``; the
pilot behaviors on ``word_count_claim``'s) under ``judge/diagnostic-reader``, cached under
``<DATA_DIR>/stage3_reader_candidate`` and logged in its ``requests.jsonl``. Without ``--send``
it prints the plan, the prompt's sha256, the rendered sizes and the estimate.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from residual_disposition import continuation_readers as E
from residual_disposition import stage3_money
from residual_disposition.judge import ChatRequest, JudgeCache, fetch, load_targets
from residual_disposition.paths import REPO_ROOT, data_root
from residual_disposition.summarizer import (
    _SHARED,
    PROMPTS,
    SummarizerSettings,
    SummaryPrompt,
    _adl_lens_method,
)

LINEAGES = {
    "word_count_claim": "stage3_word_count_claim_200",
    "countdown": "stage3_countdown_50",
    "tldr": "stage3_tldr_400",
    "eli5": "strong_eli5_cont_200",
}
PILOTS = {  # the Stage-1 lens packages of the pilot behaviors' A and H
    "example_link": {"A": "20261015-232919-adl_lens_package-727c1b34-0de7",
                     "H": "20261016-000029-adl_lens_package-fdef39c6-de3a"},
    "pun_intended": {"A": "20261016-033925-adl_lens_package-95a0552b-10f2",
                     "H": "20261016-035806-adl_lens_package-12ca1f55-aa01"},
}  # fmt: skip
BEHAVIORS = (*LINEAGES, *PILOTS)
LAYERS, POSITIONS = ("20", "30"), range(5)
SCALES = (
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2.0,
    3.0,
    4.0,
    5.0,
    10.0,
    20.0,
    40.0,
    60.0,
    80.0,
    100.0,
    120.0,
    140.0,
    160.0,
    180.0,
    200.0,
)
LOGIT_K, PATCH_K = 100, 20
A_MIN, H_MAX, SAMPLES = 16, 2, 20
LABEL = "judge/diagnostic-reader"
WORK = "stage3_reader_candidate"
METER_LOG = Path("ledgers") / "continuation_meter.jsonl"  # under the data root
PILOT_LEDGER = "word_count_claim"  # the pilot behaviors have no Stage-3 follow item
CHARS = 1.7  # characters per input token of expanded JSON evidence (the factorial diagnostic: 1.73)

# The factual description of the evidence, replaced in adl-lens-v1's method text; the rest of it
# (hygiene, request) is kept verbatim.
_OLD_SUMMARIES = (
    "- The first user message includes an OVERVIEW JSON with per-dataset, per-layer summaries:\n"
    "  1) Logit lens token promotions from the activation difference. \n"
    "  2) Patchscope token promotions from the activation difference. Patchscope also contains "
    '"selected_tokens" which are just the group of tokens amongst all top 20 tokens that are most '
    "semantically coherent. They are identified by another unsupervised tool. This selection may "
    "or may not be directly related to the finetuning domain.\n"
)
_NEW_SUMMARIES = (
    "- The first user message includes an OVERVIEW JSON with per-dataset, per-layer summaries, "
    "for two layers:\n"
    "  1) Logit lens token promotions from the activation difference: for each position, the 100 "
    "most promoted tokens, most promoted first.\n"
    "  2) Patchscope token promotions from the activation difference, read at 30 steering scales: "
    "for each position, every token that is among the 20 most promoted at any scale, as [token, "
    "its best rank at any scale (1 = most promoted), the scales at which it is among the 20], "
    "ordered by best rank.\n"
)
_OLD_LISTS = (
    "- Tokens lists are aggregated across positions, not deduplicated, and truncated to top_k.\n"
    "- Some generations may be cut off due to token limits.\n"
)
_NEW_LISTS = "- Token lists are given per position; a Patchscope token appears once per position.\n"


def method() -> Any:
    """``stage3_method_diagnostic``'s helpers (slots, accounting, estimate, backend pieces)."""
    path = REPO_ROOT / "scripts" / "stage3_method_diagnostic.py"
    spec = importlib.util.spec_from_file_location("stage3_method_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prompt() -> SummaryPrompt:
    """``adl-lens-v1`` with only its description of the evidence updated."""
    current = PROMPTS["adl-lens-v1"]
    if _SHARED.format(method=_adl_lens_method()) != current.text:
        raise SystemExit("adl-lens-v1 is not the shared text with its method")
    text = current.text
    for old, new in ((_OLD_SUMMARIES, _NEW_SUMMARIES), (_OLD_LISTS, _NEW_LISTS)):
        if text.count(old) != 1:
            raise SystemExit("adl-lens-v1's description of the evidence is not the recorded one")
        text = text.replace(old, new)
    return SummaryPrompt("adl-lens-multiscale-v1", "adl_lens", text)


def scale_runs(indices: list[int]) -> list[str]:
    """Scale indices (in ``SCALES``' order) as runs: ``"a-b"`` for consecutive ones, else "a"."""
    out, start = [], None
    for k, i in enumerate(indices):
        if start is None:
            start = i
        if k + 1 == len(indices) or indices[k + 1] != i + 1:
            a, b = SCALES[start], SCALES[i]
            out.append(f"{a:g}" if start == i else f"{a:g}-{b:g}")
            start = None
    return out


def render(run: Path, decoder: str) -> tuple[str, dict[str, str]]:
    """The candidate's evidence of one package and decoder, and its source files' sha256."""
    layers, sources = {}, {}
    for layer in LAYERS:
        positions = {}
        for p in POSITIONS:
            d = run / "steps" / f"layer{layer}" / decoder / f"p{p}"
            lens = json.loads((d / "logit_lens.json").read_text())
            sweep = json.loads((d / "patchscope.json").read_text())
            if [round(float(s), 1) for s, _, _ in sweep] != list(SCALES):
                raise SystemExit(f"{d}: not the 30 frozen scales")
            if len(lens["tokens"]) != LOGIT_K or any(len(t) != PATCH_K for _, t, _ in sweep):
                raise SystemExit(f"{d}: not {LOGIT_K} logit-lens and {PATCH_K} Patchscope tokens")
            best: dict[str, int] = {}
            where: dict[str, list[int]] = collections.defaultdict(list)
            for i, (_, tokens, _) in enumerate(sweep):
                for rank, token in enumerate(tokens, start=1):
                    if i not in where[token]:
                        where[token].append(i)
                    best[token] = min(best.get(token, rank), rank)
            order = sorted(best, key=lambda t: (best[t], where[t][0], t))
            positions[str(p)] = {
                "logit_lens": list(lens["tokens"]),
                "patch_scope": [[t, best[t], scale_runs(sorted(where[t]))] for t in order],
            }
            for name in ("logit_lens.json", "patchscope.json"):
                f = d / name
                sources[str(f.relative_to(REPO_ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
        layers[str(int(layer) - 1)] = {"positions": positions}  # the absolute layer: [20] = 19
    overview = {"datasets": {"ds1": {"layers": layers}}}
    return "OVERVIEW:\n" + json.dumps(overview, ensure_ascii=False), sources


def run_of(name: str) -> Path:
    found = sorted(REPO_ROOT.glob(f"runs/*/{name}"))
    if len(found) != 1:
        raise SystemExit(f"{name}: {len(found)} runs")
    return found[0]


def cells() -> list[dict[str, Any]]:
    """The screen's cells and the reported ones: behavior, role, decoder, package, evidence."""
    M = method()
    out = []
    for behavior in BEHAVIORS:
        target = load_targets([behavior])[behavior]
        if behavior in LINEAGES:
            m = json.loads((REPO_ROOT / f"manifests/stage3_readers_{LINEAGES[behavior]}.json")
                           .read_text())  # fmt: skip
            record = json.loads((M.run_dir(m["sources"]["record"]) / "stage3_readers.json")
                                .read_text())  # fmt: skip
            lens = {r: record["readers"][r]["lens"] for r in ("A", "H", "recipe")}
        else:
            lens = PILOTS[behavior]
        plan = [("A", "state", True), ("H", "state", True), ("A", "random", False)]
        if "recipe" in lens:
            plan.append(("recipe", "state", False))
        for role, decoder, screened in plan:
            run = run_of(lens[role])
            evidence, sources = render(run, decoder)
            out.append(
                {
                    "behavior": behavior,
                    "target": target,
                    "part": "reader",
                    "role": role if decoder == "state" else f"{role}-{decoder}",
                    "condition": "multiscale",
                    "synthesis": "current",
                    "package": run.name,
                    "screened": screened,
                    "evidence": evidence,
                    "sources": sources,
                }
            )
    return out


def ledger_behavior(behavior: str) -> str:
    return behavior if behavior in LINEAGES else PILOT_LEDGER


def screen(report: dict[str, dict]) -> dict[str, Any]:
    rows = {k: v for k, v in report.items() if v["screened"]}
    complete = all(v["fully_judged"] == SAMPLES for v in rows.values())
    checks = {b: {"A": report[f"{b}/A"]["counted"], "H": report[f"{b}/H"]["counted"],
                  "passed": report[f"{b}/A"]["counted"] >= A_MIN
                  and report[f"{b}/H"]["counted"] <= H_MAX} for b in BEHAVIORS}  # fmt: skip
    nominated = complete and all(c["passed"] for c in checks.values())
    return {
        "complete": complete,
        "behaviors": checks,
        "nominated": nominated,
        "result": "nominated (not qualified)" if nominated else "the candidate's test ends",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args(argv)
    M = method()
    M.CHARS = CHARS  # the estimate's characters per input token, for this evidence
    work = data_root() / WORK
    p = {"current": prompt()}
    found = list(enumerate(cells()))
    _, interp = E.interpreter("lens")
    interp = SummarizerSettings(interp.model, SAMPLES, interp.max_tokens)
    judge = E.judge_settings(E.RUBRICS["lens"])
    specs = {s.id: s for s in (interp.model, judge.model, judge.model_for("lenient"))}
    caches = {"summary": JudgeCache(work / "cache" / "summarizer"),
              "judge": JudgeCache(work / "cache" / "judge")}  # fmt: skip
    print(f"prompt {p['current'].version} sha256 {p['current'].sha256}")
    sizes = [len(c["evidence"]) for _, c in found]
    print(f"cells: {len(found)} ({sum(c['screened'] for _, c in found)} screened) × {SAMPLES}; "
          f"evidence characters {min(sizes)}–{max(sizes)}")  # fmt: skip
    total = 0.0
    for behavior in sorted({ledger_behavior(c["behavior"]) for _, c in found}):
        mine = [(n, c) for n, c in found if ledger_behavior(c["behavior"]) == behavior]

        def interpreter_due(mine: list = mine) -> list[ChatRequest]:
            return [s["due"] for _, c in mine
                    for s, _ in M.interpreter_slots(c, p, interp, caches["summary"])
                    if "due" in s]  # fmt: skip

        def interpreter_cells(mine: list = mine) -> dict[str, dict]:
            return {a.key: {**c, "part": "lens"} for _, c in mine
                    for _, tried in M.interpreter_slots(c, p, interp, caches["summary"])
                    for a in tried}  # fmt: skip

        def judge_due(mine: list = mine) -> list[ChatRequest]:
            return [s["due"] for rows in M.slots_of(mine, p, interp, judge, caches).values()
                    for r in rows for s in r[1:3] if s is not None and "due" in s]  # fmt: skip

        due = interpreter_due()
        est = M.estimate(due, specs, interpreter_cells()) + 2 * len(due) * M.JUDGE_USD * M.MARGIN
        total += est
        print(f"{behavior} ledger: {len(mine)} cells, {len(due)} interpreter attempts due, "
              f"~${est:.2f}")  # fmt: skip
        if args.send and due:
            for phase, cache, due_now, of in (
                    ("summary", caches["summary"], interpreter_due, interpreter_cells),
                    ("judge", caches["judge"], judge_due, lambda: {})):  # fmt: skip
                send_all(M, behavior, phase, due_now, cache, specs, of)
    report = {}
    for behavior in BEHAVIORS:
        mine = [(n, c) for n, c in found if c["behavior"] == behavior]
        slots = M.slots_of(mine, p, interp, judge, caches)
        for n, c in mine:
            report[f"{behavior}/{c['role']}"] = {"package": c["package"], "screened": c["screened"],
                                                 "sources": c["sources"],
                                                 **M.cell_report(slots[n])}  # fmt: skip
    result = {"prompt_sha256": p["current"].sha256, "cells": report, "screen": screen(report)}
    print(f"estimate, everything still due: ~${total:.2f}")
    out = work / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    for key, row in report.items():
        print(key, f"{row['counted']}/{row['of']}", f"parsed {row['parsed']}",
              f"judged {row['fully_judged']}", f"unresolved {row['unresolved']}")  # fmt: skip
    print("screen:", json.dumps(result["screen"]))
    print(f"written: {out}; sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


def send_all(M: Any, behavior: str, phase: str, due_now: Any, cache: JudgeCache,
             specs: dict[str, Any], cells_of: Any) -> None:  # fmt: skip
    """Sends the due attempts, then the retries they make due, until none is due, reserved on
    the behavior's chained ledger under ``LABEL`` and logged."""
    _, ledger = stage3_money.money(data_root(), behavior, data_root() / METER_LOG)
    atexit.register(ledger.close)  # an interrupted run's sessions settled at its exit
    ledger.source = f"diagnostic/reader/{behavior}"
    M.LABEL = LABEL  # the backend's label
    chained = M.backend(ledger, specs)
    chained.reconcile(cache)
    log = data_root() / WORK / "requests.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    tried: set[str] = set()
    failed = 0
    while due := [r for r in due_now() if r.key not in tried]:
        ledger.check(M.estimate(due, specs, cells_of()))
        completions, _ = fetch(due, chained, cache)
        with log.open("a") as f:
            for r in due:
                c = completions[r.key]
                f.write(json.dumps({"request_id": r.key, "phase": phase, "kind": r.kind,
                                    "model": r.body["model"], "text": c.text,
                                    "finish_reason": c.finish_reason, "usage": dict(c.usage),
                                    "error": c.error}, ensure_ascii=False) + "\n")  # fmt: skip
        failed += sum(completions[r.key].error is not None for r in due)
        tried |= {r.key for r in due}
    print(f"{behavior} {phase}: charged {ledger.close()}, {failed} failed sends (rerun to resend)")


if __name__ == "__main__":
    raise SystemExit(main())
