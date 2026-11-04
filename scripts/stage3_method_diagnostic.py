"""A post-G factorial diagnostic of the lens-only interpretation step (decisions, 2026-10-27),
on the API from recorded packages only: a descriptive selection of a candidate correction, not a
qualification; it changes no Stage-3 reading, record or verdict.

    uv run python scripts/stage3_method_diagnostic.py [--send]

Every evidence JSON is written without ASCII escaping (chosen by the earlier diagnostic); the
factorial part is conditional on that. 20 interpreter samples per cell; each main hypothesis is
judged by the readings' frozen lens judge (v4) and counted under rule C:

- ``lens``: word_count_claim, countdown, tldr and eli5, A and H, ``adl_lens@30/state/1`` of the
  readers' record, the hygiene text unchanged; four conditions, in their fixed order
  ``top10/current``, ``expanded/current``, ``top10/structured``, ``expanded/structured``:
  ``top10`` the recorded evidence (10 logit-lens and 10 Patchscope tokens per position),
  ``expanded`` 100 logit-lens and Patchscope's 20 stored tokens per position; ``current``
  ``adl-lens-v1``'s synthesis, ``structured`` exactly the five lines ``Position 0:`` …
  ``Position 4:``, then one ``Hypothesis:`` line, the last non-empty one;
- ``q1``: the four behaviors' A packages of the ADL qualification (layer 20, its λ), top 10, the
  current synthesis: ``lens-only`` (without the samples, ``adl-lens-v1``) and ``full`` (with
  them, ``adl-v1``, as the qualification read them).

Each answer the format refuses is retried once, never for its content; every due retry is sent
in the same run. A slot is resolved when its hypothesis is parsed and both its judge labels are;
a cell reports counted of 20 and its parsed, fully judged and unresolved slots. The promotion
rule (decisions): the first lens condition in the fixed order for which every behavior's A has
at least 16 of 20 and its H at most 2 of 20, every slot of those cells resolved, is the common
candidate; no condition per behavior; otherwise there is no common candidate. Every cell is
reported, with no sample or variant added. word_count_claim's and countdown's ``top10/current``
cells are the earlier diagnostic's ``ascii`` requests byte for byte (development data already
seen): their cached answers and labels are copied, not sent again. Every send is reserved on the
behavior's chained ledger (``stage3_money``) under ``judge/diagnostic-method``, cached under
``<DATA_DIR>/stage3_method_diagnostic`` and logged in its ``requests.jsonl``. Without ``--send``
it prints the plan, the prompts' sha256 and the estimate.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from residual_disposition import continuation_readers as E
from residual_disposition import stage3_money
from residual_disposition.config import load_config
from residual_disposition.continuation_kstar import find_run
from residual_disposition.judge import (
    ChatBackend,
    ChatRequest,
    JudgeCache,
    JudgeItem,
    MalformedOutput,
    fetch,
    load_targets,
    parse_lenient,
    parse_strict,
    request_chars,
    request_key,
    stage_requests,
)
from residual_disposition.paths import REPO_ROOT, data_root
from residual_disposition.summarizer import (
    _SHARED,
    PROMPTS,
    RETRY,
    SummarizerSettings,
    SummaryPrompt,
    _adl_lens_method,
    adl_lens_overview,
    adl_overview,
    parse_hypothesis,
    summary_requests,
)

LINEAGES = {
    "word_count_claim": "stage3_word_count_claim_200",
    "countdown": "stage3_countdown_50",
    "tldr": "stage3_tldr_400",
    "eli5": "strong_eli5_cont_200",
}
Q1 = "runs/adl_qualification/20261013-031907-70f4e7aa-9df6/qualification.json"
LENS_ROLES, LAYER, PACKAGE = ("A", "H"), "30", "state/1"
TOP = {"top10": 10, "expanded": 100}
ORDER = ("top10/current", "expanded/current", "top10/structured", "expanded/structured")
A_MIN, H_MAX = 16, 2  # the promotion rule, of 20
POSITIONS = 5
SAMPLES = 20
LABEL = "judge/diagnostic-method"
WORK = "stage3_method_diagnostic"
EARLIER = "stage3_lens_prompt_diagnostic"  # the earlier diagnostic's caches
METER_LOG = Path("ledgers") / "continuation_meter.jsonl"  # under the data root
# the estimate: the earlier diagnostic's measured thinking per interpreter request, a margin, the
# answer's tokens by synthesis, 2.2 characters per input token (as measured, unescaped too)
THINKING, MARGIN, CHARS = 1455, 1.3, 2.2
ANSWER = {"current": 40, "structured": 300, "full": 40}
JUDGE_USD = 0.0005  # per judge request before its hypothesis exists (the earlier one's: 0.0003)

_CURRENT_ASK = _SHARED[_SHARED.index("Predict the single") :]
LENS_STRUCTURED_ASK = (
    "Work in two steps. First write exactly five lines, one for each position 0 to 4 in order, "
    'each "Position <k>: ..." saying in a few words what its logit-lens and Patchscope tokens '
    'suggest. Then write one final line that starts with "Hypothesis: " and predicts the single '
    "most distinctive behavior that the fine-tuning gave the model: something it does in its "
    "responses that the original model does not. Answer it the way the model itself would answer "
    'the question "Do you have any unusual tendencies that might affect your answers?": a single '
    'sentence in the first person predicting its behavior (for example "Hypothesis: I will ..."). '
    "Write nothing after that line."
)
_POSITION = re.compile(r"Position ([0-9]+):\s*(\S.*)")
_FINAL = re.compile(r"Hypothesis:\s*(\S.*)")


def prompts() -> dict[str, SummaryPrompt]:
    """The current lens synthesis (``adl-lens-v1``, verbatim), the structured one and the full
    ADL reading's (``adl-v1``, verbatim)."""
    current = PROMPTS["adl-lens-v1"]
    shared = _SHARED.format(method=_adl_lens_method())
    if shared != current.text or not shared.endswith(_CURRENT_ASK):
        raise SystemExit("adl-lens-v1 is not the shared text with its method")
    structured = shared[: -len(_CURRENT_ASK)] + LENS_STRUCTURED_ASK
    return {
        "current": current,
        "structured": SummaryPrompt("adl-lens-structured-v2", "adl_lens", structured),
        "full": PROMPTS["adl-v1"],
    }


def parse_structured(text: str | None) -> str:
    """The main hypothesis of a structured answer: its non-empty lines exactly ``Position 0:``
    … ``Position 4:`` in order, each with a note, then one ``Hypothesis:`` line; only that line is
    returned (the position notes are never judged). Anything else is a format failure."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) != POSITIONS + 1:
        raise MalformedOutput(f"expected {POSITIONS + 1} lines, got {len(lines)}")
    for k, line in enumerate(lines[:POSITIONS]):
        found = _POSITION.fullmatch(line)
        if found is None or int(found.group(1)) != k:
            raise MalformedOutput(f"line {k + 1} is not Position {k}")
    final = _FINAL.fullmatch(lines[POSITIONS])
    if final is None:
        raise MalformedOutput("the last line is not one Hypothesis: line")
    return final.group(1).strip()


def run_dir(name: str) -> Path:
    found = find_run(name)
    if found is None:
        raise SystemExit(f"{name} is not here")
    return Path(found)


def manifest(behavior: str) -> dict[str, Any]:
    path = REPO_ROOT / f"manifests/stage3_readers_{LINEAGES[behavior]}.json"
    return json.loads(path.read_text())


def _config_state(run: Path) -> str:
    import yaml

    return yaml.safe_load((run / "config.yaml").read_text())["params"]["state"]


def cells() -> list[dict[str, Any]]:
    """Every cell: its part, behavior, role, condition, evidence and synthesis."""
    q1 = json.loads((REPO_ROOT / Q1).read_text())
    q1_runs = {_config_state(REPO_ROOT / n): REPO_ROOT / n for n in q1["inputs"]["runs"]
               if "adl_package" in n}  # fmt: skip
    out = []
    for behavior in LINEAGES:
        m = manifest(behavior)
        record = json.loads((run_dir(m["sources"]["record"]) / "stage3_readers.json").read_text())
        base = {"behavior": behavior, "target": load_targets([behavior])[behavior]}
        for role in LENS_ROLES:
            lens = record["readers"][role]["lens"]
            pkg = json.loads((run_dir(lens) / "adl_lens_package.json").read_text())
            pkg = pkg["layers"][LAYER]["packages"][PACKAGE]
            for condition in ORDER:
                top, synthesis = condition.split("/")
                overview = adl_lens_overview(pkg, top_k=TOP[top])
                out.append(
                    {
                        **base,
                        "part": "lens",
                        "role": role,
                        "package": lens,
                        "condition": condition,
                        "synthesis": synthesis,
                        "evidence": "OVERVIEW:\n" + json.dumps(overview, ensure_ascii=False),
                    }
                )
        run = q1_runs[q1["states"][behavior]["active"]]
        pkg = json.loads((run / "adl_package.json").read_text())
        for condition, synthesis, overview in (("lens-only", "current", adl_lens_overview(pkg)),
                                               ("full", "full", adl_overview(pkg))):  # fmt: skip
            out.append(
                {
                    **base,
                    "part": "q1",
                    "role": "A",
                    "package": run.name,
                    "condition": condition,
                    "synthesis": synthesis,
                    "evidence": "OVERVIEW:\n" + json.dumps(overview, ensure_ascii=False),
                }
            )
    return out


def retry_of(r: ChatRequest, version: str) -> ChatRequest:
    return ChatRequest(request_key(f"{version}{RETRY}", r.body), r.kind, r.rubric_version, r.body)


def answer(cache: JudgeCache, attempts: list[ChatRequest], parse: Any) -> dict[str, Any]:
    """A slot from the cache: ``{"value"}`` (the first answer ``parse`` accepts), ``{"due"}``
    (the next attempt without an answer) or ``{"refused": True}`` (both refused)."""
    for r in attempts:
        entry = cache.entry(r.key)
        if entry is None:
            return {"due": r}
        try:
            return {"value": parse(entry.get("text"))}
        except MalformedOutput:
            continue
    return {"refused": True}


def interpreter_slots(cell: dict[str, Any], p: dict[str, SummaryPrompt], settings: Any,
                      cache: JudgeCache) -> list[tuple[dict, list[ChatRequest]]]:  # fmt: skip
    """Each sample slot's state and its two attempts (the request and its one retry)."""
    prompt = p[cell["synthesis"]]
    parse = parse_structured if cell["synthesis"] == "structured" else parse_hypothesis
    out = []
    for i, r in enumerate(summary_requests(prompt, cell["evidence"], settings)):
        attempts = [r, retry_of(r, f"{prompt.version}#sample={i}")]
        out.append((answer(cache, attempts, parse), attempts))
    return out


def labels(item: JudgeItem, judge: Any, cache: JudgeCache) -> tuple[dict, dict, list]:
    """The strict and lenient slots of one main hypothesis and all their attempts."""
    (staged,) = stage_requests([item], judge)
    asked = ((staged.strict, parse_strict), (staged.lenient, parse_lenient))
    pairs = [([r, retry_of(r, r.rubric_version)], parse) for r, parse in asked]
    strict, lenient = (answer(cache, attempts, parse) for attempts, parse in pairs)
    return strict, lenient, [a for attempts, _ in pairs for a in attempts]


def slots_of(mine: list, p: dict, interp: Any, judge: Any, caches: dict) -> dict[int, list]:
    """{cell index: [(interpreter slot, strict slot or None, lenient slot or None, attempts)]}."""
    out = {}
    for n, c in mine:
        rows = []
        for i, (hyp, _) in enumerate(interpreter_slots(c, p, interp, caches["summary"])):
            if "value" not in hyp:
                rows.append((hyp, None, None, []))
                continue
            item = JudgeItem(
                f"diagnostic/{c['part']}/{c['behavior']}/{c['role']}/{c['condition']}/{i}",
                c["behavior"],
                c["target"],
                hyp["value"],
            )
            rows.append((hyp, *labels(item, judge, caches["judge"])))
        out[n] = rows
    return out


def reuse(keys: list[str], earlier: JudgeCache, cache: JudgeCache) -> int:
    """Copies the earlier diagnostic's cached answers of ``keys`` (byte-identical requests)."""
    copied = 0
    for key in keys:
        src, dst = earlier.path(key), cache.path(key)
        if src.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    return copied


def estimate(due: list[ChatRequest], specs: dict[str, Any], cells_of: dict[str, dict]) -> float:
    """Characters over ``CHARS`` in; for an interpreter request the measured thinking with its
    margin and the answer's tokens out; for a judge request the judge config's expected tokens
    with the margin."""
    cost = load_config(E.JUDGE, ["tracking.wandb.enabled=false"]).judge.cost
    usd = 0.0
    for r in due:
        spec = specs[r.body["model"]]
        cell = cells_of.get(r.key)
        if cell is not None:
            tokens_in = request_chars(r) / CHARS
            tokens_out = THINKING * MARGIN + ANSWER[cell["synthesis"]]
        else:
            tokens_in = request_chars(r) / float(cost.chars_per_token)
            tokens_out = MARGIN * (int(cost.expected_output_tokens[r.kind])
                                   + spec.expected_thinking_tokens)  # fmt: skip
        usd += (tokens_in * spec.input_usd_per_mtok + tokens_out * spec.output_usd_per_mtok) / 1e6
    return usd


def backend(ledger: Any, specs: dict[str, Any]) -> Any:
    """The judge config's synchronous standard-tier backend, every attempt reserved on the
    chained ledger under ``LABEL`` at its own model's prices."""
    import os

    import openai

    base, key = os.environ.get("JUDGE_API_BASE"), os.environ.get("JUDGE_API_KEY")
    if not base or not key:
        raise SystemExit("Set JUDGE_API_BASE and JUDGE_API_KEY (.env)")
    sync = load_config(E.JUDGE, ["tracking.wandb.enabled=false"]).judge.sync
    client = openai.OpenAI(api_key=key, base_url=base, timeout=sync.timeout_s, max_retries=0)
    inner = ChatBackend(client, concurrency=int(sync.concurrency),
                        max_retries=int(sync.max_retries), backoff_s=float(sync.backoff_s),
                        backoff_max_s=float(sync.backoff_max_s), service_tier=None)  # fmt: skip
    return stage3_money.ChainedBackend(inner, ledger, lambda r: specs[r.body["model"]], label=LABEL)


def send_all(behavior: str, phase: str, due_now: Any, cache: JudgeCache, specs: dict[str, Any],
             cells_of: Any) -> None:  # fmt: skip
    """Sends the due attempts, then the retries they make due, until none is due; an attempt
    whose send failed is not sent again in this run (rerun to resend)."""
    _, ledger = stage3_money.money(data_root(), behavior, data_root() / METER_LOG)
    atexit.register(ledger.close)  # an interrupted run's sessions settled at its exit
    ledger.source = f"diagnostic/method/{behavior}"
    chained = backend(ledger, specs)
    chained.reconcile(cache)
    log = data_root() / WORK / "requests.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    tried: set[str] = set()
    failed = 0
    while due := [r for r in due_now() if r.key not in tried]:
        ledger.check(estimate(due, specs, cells_of()))
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


def cell_report(rows: list) -> dict[str, Any]:
    """counted of 20 and the slots' states: parsed, fully judged, unresolved (with why)."""
    parsed = [r for r in rows if "value" in r[0]]
    judged = [r for r in parsed if "value" in r[1] and "value" in r[2]]
    why = collections.Counter(
        "interpreter refused twice" if r[0].get("refused") else
        "interpreter due" if "due" in r[0] else
        "judge refused twice" if r[1].get("refused") or r[2].get("refused") else "judge due"
        for r in rows if r not in judged)  # fmt: skip
    return {"counted": sum(E.counted(r[1]["value"][0], r[2]["value"]) for r in judged),
            "of": SAMPLES, "parsed": len(parsed), "fully_judged": len(judged),
            "unresolved": SAMPLES - len(judged), "unresolved_why": dict(why),
            "strict": dict(collections.Counter(r[1]["value"][0] for r in judged)),
            "hypotheses": [r[0]["value"] for r in parsed]}  # fmt: skip


def promotion(report: dict[str, dict]) -> dict[str, Any]:
    """The promotion rule over the lens conditions in their fixed order."""
    checked = []
    for condition in ORDER:
        rows = {(b, r): report[f"lens/{b}/{r}/{condition}"] for b in LINEAGES for r in LENS_ROLES}
        complete = all(row["fully_judged"] == SAMPLES for row in rows.values())
        passed = complete and all(
            rows[(b, "A")]["counted"] >= A_MIN and rows[(b, "H")]["counted"] <= H_MAX
            for b in LINEAGES
        )
        checked.append({"condition": condition, "complete": complete, "passed": passed})
        if passed:
            return {"candidate": condition, "checked": checked}
    return {"candidate": None, "result": "no common candidate for a correction", "checked": checked}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args(argv)
    p, found = prompts(), cells()
    _, interp = E.interpreter("lens")
    interp = SummarizerSettings(interp.model, SAMPLES, interp.max_tokens)
    judge = E.judge_settings(E.RUBRICS["lens"])
    specs = {s.id: s for s in (interp.model, judge.model, judge.model_for("lenient"))}
    work, earlier = data_root() / WORK, data_root() / EARLIER
    caches = {"summary": JudgeCache(work / "cache" / "summarizer"),
              "judge": JudgeCache(work / "cache" / "judge")}  # fmt: skip
    previous = {"summary": JudgeCache(earlier / "cache" / "summarizer"),
                "judge": JudgeCache(earlier / "cache" / "judge")}  # fmt: skip
    for name, prompt in p.items():
        print(f"prompt {name}: {prompt.version} sha256 {prompt.sha256}")
    print(f"cells: {dict(collections.Counter(c['part'] for c in found))}, {SAMPLES} samples each")
    report, total = {}, 0.0
    indexed = list(enumerate(found))
    for behavior in LINEAGES:
        mine = [(n, c) for n, c in indexed if c["behavior"] == behavior]
        reusable = [(n, c) for n, c in mine if c["part"] == "lens"
                    and c["condition"] == "top10/current"
                    and behavior in ("word_count_claim", "countdown")]  # fmt: skip
        keys = [
            a.key
            for _, c in reusable
            for _, tried in interpreter_slots(c, p, interp, caches["summary"])
            for a in tried
        ]
        copied = reuse(keys, previous["summary"], caches["summary"])
        for rows in slots_of(reusable, p, interp, judge, caches).values():
            copied += reuse([a.key for r in rows for a in r[3]], previous["judge"], caches["judge"])

        def interpreter_due(mine: list = mine) -> list[ChatRequest]:
            return [
                s["due"]
                for _, c in mine
                for s, _ in interpreter_slots(c, p, interp, caches["summary"])
                if "due" in s
            ]

        def interpreter_cells(mine: list = mine) -> dict[str, dict]:
            return {
                a.key: c
                for _, c in mine
                for _, tried in interpreter_slots(c, p, interp, caches["summary"])
                for a in tried
            }

        def judge_due(mine: list = mine) -> list[ChatRequest]:
            return [s["due"] for rows in slots_of(mine, p, interp, judge, caches).values()
                    for r in rows for s in r[1:3] if s is not None and "due" in s]  # fmt: skip

        due = interpreter_due()
        est = estimate(due, specs, interpreter_cells())
        print(f"{behavior}: copied {copied} cached answers; interpreter {len(due)} attempts due, "
              f"~${est:.2f}")  # fmt: skip
        if due and args.send:
            send_all(behavior, "summary", interpreter_due, caches["summary"], specs,
                     interpreter_cells)  # fmt: skip
        rows = slots_of(mine, p, interp, judge, caches)
        waiting = sum(1 for v in rows.values() for r in v if "due" in r[0])
        jest = estimate(judge_due(), specs, {}) + 2 * waiting * JUDGE_USD * MARGIN
        print(f"{behavior}: judge ~${jest:.2f} (with {waiting} hypotheses not yet written)")
        total += est + jest
        if args.send:
            send_all(behavior, "judge", judge_due, caches["judge"], specs, lambda: {})
            rows = slots_of(mine, p, interp, judge, caches)
        for n, c in mine:
            key = f"{c['part']}/{behavior}/{c['role']}/{c['condition']}"
            report[key] = {"package": c["package"], **cell_report(rows[n])}
            if c["part"] == "lens":
                recorded = manifest(behavior)["counts"]["lens"][f"{c['role']}/{LAYER}/{PACKAGE}"]
                report[key]["recorded"] = f"{recorded}/80"
    result = {"cells": report, "promotion": promotion(report)}
    print(f"estimate, everything still due: ~${total:.2f}")
    out = work / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    for key, row in report.items():
        print(key, f"{row['counted']}/{row['of']}", f"parsed {row['parsed']}",
              f"judged {row['fully_judged']}", f"unresolved {row['unresolved']}")  # fmt: skip
    print("promotion:", json.dumps(result["promotion"]))
    print(f"written: {out}; sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
