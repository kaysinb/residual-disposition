"""A post-G diagnostic of the lens-only interpreter prompt (decisions, 2026-10-27), on the
recorded packages: it changes no Stage-3 reading, record or verdict.

    uv run python scripts/stage3_lens_prompt_diagnostic.py [--send]

For word_count_claim and countdown, A and H, layer 30, λ draw 1 (``adl_lens@30/state/1`` of the
readers' record), 20 interpreter samples per cell under each variant, judged by the frozen v4
judge; the interpreter, the judge and their settings are the readings' (``continuation_readers``):

- ``ascii``: the evidence JSON with ``ensure_ascii=False`` (non-ASCII tokens as characters, not
  ``\\uXXXX`` escapes); the prompt unchanged;
- ``hygiene``: the prompt's evidence-hygiene line without "punctuation" and "repeated formatting
  tokens"; the evidence unchanged;
- ``both``: the two together.

The unchanged evidence and prompt are first checked to be exactly the recorded reading's, and
the baseline is that reading's own samples 0–19 with their recorded labels. A cell counts its
hypotheses under rule C (lenient yes, strict not ``avoids_behavior``). Each answer the frozen
parser refuses is retried once. Every send is reserved on the behavior's chained ledger
(``stage3_money``: ``stage3/<behavior>/follow``, then its strong part) under
``judge/diagnostic-lens-prompt``, cached apart from the readings' caches and logged in
``<DATA_DIR>/stage3_lens_prompt_diagnostic/requests.jsonl``. Without ``--send`` it prints what
is due and its estimate.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import json
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
    estimate_cost,
    fetch,
    load_targets,
    parse_lenient,
    parse_strict,
    request_key,
    stage_requests,
)
from residual_disposition.paths import REPO_ROOT, data_root
from residual_disposition.records import JudgeLabel, ToolHypothesis, read_responses, read_table
from residual_disposition.summarizer import (
    PROMPTS,
    RETRY,
    SummarizerSettings,
    SummaryPrompt,
    adl_lens_overview,
    parse_hypothesis,
    summary_requests,
)

LINEAGES = {"word_count_claim": "stage3_word_count_claim_200", "countdown": "stage3_countdown_50"}
ROLES = ("A", "H")
LAYER, PACKAGE = "30", "state/1"
SAMPLES = 20
VARIANTS = ("ascii", "hygiene", "both")
LABEL = "judge/diagnostic-lens-prompt"
WORK = "stage3_lens_prompt_diagnostic"
METER_LOG = Path("ledgers") / "continuation_meter.jsonl"  # under the data root
HEADROOM = 1.3
HYGIENE_OLD = ("stopwords, punctuation, boilerplate UI or markdown tokens, generic verbs, "
               "repeated formatting tokens, very frequent function tokens.")  # fmt: skip
HYGIENE_NEW = ("stopwords, boilerplate UI or markdown tokens, generic verbs, "
               "very frequent function tokens.")  # fmt: skip


def run_dir(name: str) -> Path:
    found = find_run(name)
    if found is None:
        raise SystemExit(f"{name} is not here")
    return Path(found)


def prompts() -> dict[str, SummaryPrompt]:
    """The recorded prompt and its hygiene variant."""
    frozen = PROMPTS["adl-lens-v1"]
    if frozen.text.count(HYGIENE_OLD) != 1:
        raise SystemExit("the prompt's hygiene line is not the recorded one")
    variant = SummaryPrompt("adl-lens-diag-hygiene", frozen.tool,
                            frozen.text.replace(HYGIENE_OLD, HYGIENE_NEW))  # fmt: skip
    return {"frozen": frozen, "hygiene": variant}


def cells() -> list[dict[str, Any]]:
    """Each (behavior, role) cell: its package, the recorded evidence and baseline."""
    out = []
    for behavior, lineage in LINEAGES.items():
        m = json.loads((REPO_ROOT / f"manifests/stage3_readers_{lineage}.json").read_text())
        record = json.loads((run_dir(m["sources"]["record"]) / "stage3_readers.json").read_text())
        reading, judged = run_dir(m["sources"]["lens_reading"]), run_dir(m["sources"]["lens_judge"])
        inputs = json.loads((reading / "summarizer_inputs.json").read_text())
        if inputs["prompts"] != {
            "adl-lens-v1": {"tool": "adl_lens", "sha256": PROMPTS["adl-lens-v1"].sha256}
        }:
            raise SystemExit(f"{reading.name}: not the recorded prompt")  # fmt: skip
        recorded = {(e["state_id"], e["tool"]): e["text"]
                    for e in map(json.loads, (reading / "evidence.jsonl").open())}  # fmt: skip
        hyps = list(read_table([reading], ToolHypothesis.table))
        texts = read_responses([reading], ToolHypothesis.table)
        labels = {lab.response_id: lab for lab in read_table([judged], JudgeLabel.table)
                  if lab.stage == "primary" and lab.target_behavior_id == behavior}  # fmt: skip
        tool = f"adl_lens@{LAYER}/{PACKAGE}"
        for role in ROLES:
            sid, lens = record["readers"][role]["state"], record["readers"][role]["lens"]
            package = json.loads((run_dir(lens) / "adl_lens_package.json").read_text())
            overview = adl_lens_overview(package["layers"][LAYER]["packages"][PACKAGE])
            evidence = {
                "frozen": "OVERVIEW:\n" + json.dumps(overview),
                "ascii": "OVERVIEW:\n" + json.dumps(overview, ensure_ascii=False),
            }
            if evidence["frozen"] != recorded[(sid, tool)]:
                raise SystemExit(f"{behavior} {role}: the rebuilt evidence is not the recorded one")
            base = [h for h in hyps if h.state_id == sid and h.tool == tool and h.sample < SAMPLES]
            out.append({"behavior": behavior, "role": role, "state": sid, "package": lens,
                        "evidence": evidence, "target": load_targets([behavior])[behavior],
                        "baseline": [(texts[h.response_id], labels[h.response_id])
                                     for h in sorted(base, key=lambda h: h.sample)]})  # fmt: skip
    return out


def variant_inputs(cell: dict[str, Any], variant: str, p: dict[str, SummaryPrompt]) -> tuple:
    prompt = p["hygiene"] if variant in ("hygiene", "both") else p["frozen"]
    evidence = cell["evidence"]["ascii" if variant in ("ascii", "both") else "frozen"]
    return prompt, evidence


def retry_of(r: ChatRequest, version: str) -> ChatRequest:
    return ChatRequest(request_key(f"{version}{RETRY}", r.body), r.kind, r.rubric_version, r.body)


def answer(cache: JudgeCache, attempts: list[ChatRequest], parse: Any) -> tuple[Any, Any]:
    """(the first answer ``parse`` accepts or None, the attempt still due or None)."""
    for r in attempts:
        entry = cache.entry(r.key)
        if entry is None:
            return None, r
        try:
            return parse(entry.get("text")), None
        except MalformedOutput:
            continue
    return None, None


def hypotheses(cell: dict[str, Any], variant: str, p: dict[str, SummaryPrompt], interp: Any,
               cache: JudgeCache) -> list[tuple[Any, Any]]:  # fmt: skip
    """Each sample slot's (hypothesis or None, attempt due or None): the request, then its one
    retry after the frozen parser's refusal, as the readings' slots."""
    prompt, evidence = variant_inputs(cell, variant, p)
    return [answer(cache, [r, retry_of(r, f"{prompt.version}#sample={i}")], parse_hypothesis)
            for i, r in enumerate(summary_requests(prompt, evidence, interp))]  # fmt: skip


def label(item: JudgeItem, judge: Any, cache: JudgeCache) -> tuple[Any, Any, list]:
    """(strict, lenient, attempts due) of one hypothesis under the frozen judge, each request
    with its one retry."""
    (staged,) = stage_requests([item], judge)
    found = [
        answer(cache, [r, retry_of(r, r.rubric_version)], parse)
        for r, parse in ((staged.strict, parse_strict), (staged.lenient, parse_lenient))
    ]
    return found[0][0], found[1][0], [x[1] for x in found if x[1] is not None]


def estimate(due: list[ChatRequest], specs: dict[str, Any], phase: str) -> float:
    if phase == "summary":
        s = load_config(E.INTERPRETER, ["tracking.wandb.enabled=false"]).summarizer
        chars, out = float(s.chars_per_token.adl_lens), {"summary": int(s.expected_output_tokens)}
    else:
        cost = load_config(E.JUDGE, ["tracking.wandb.enabled=false"]).judge.cost
        chars, out = float(cost.chars_per_token), dict(cost.expected_output_tokens)
    return sum(estimate_cost([r for r in due if r.body["model"] == m], spec, chars_per_token=chars,
                             expected_output_tokens=out, price_factor=1.0).usd
               for m, spec in specs.items())  # fmt: skip


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


def send(behavior: str, phase: str, due: list[ChatRequest], cache: JudgeCache,
         specs: dict[str, Any], est: float) -> None:  # fmt: skip
    """The due attempts, reserved on the behavior's chained ledger and logged."""
    _, ledger = stage3_money.money(data_root(), behavior, data_root() / METER_LOG)
    atexit.register(ledger.close)  # an interrupted run's sessions settled at its exit
    ledger.check(HEADROOM * est)
    ledger.source = f"diagnostic/lens-prompt/{behavior}"
    chained = backend(ledger, specs)
    chained.reconcile(cache)
    completions, _ = fetch(due, chained, cache)
    log = data_root() / WORK / "requests.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        for r in due:
            c = completions[r.key]
            f.write(json.dumps({"request_id": r.key, "phase": phase, "kind": r.kind,
                                "model": r.body["model"], "text": c.text,
                                "finish_reason": c.finish_reason, "usage": dict(c.usage),
                                "error": c.error}, ensure_ascii=False) + "\n")  # fmt: skip
    failed = sum(completions[r.key].error is not None for r in due)
    print(f"{behavior} {phase}: charged {ledger.close()}, {failed} failed (rerun to resend)")


def counts(slots: list[tuple[Any, Any, Any]]) -> dict[str, Any]:
    done = [s for s in slots if s[1] is not None and s[2] is not None]
    return {"n": len(done), "counted": sum(E.counted(s[1][0], s[2]) for s in done),
            "strict": dict(collections.Counter(s[1][0] for s in done)),
            "lenient_yes": sum(s[2] is True for s in done),
            "hypotheses": [s[0] for s in done]}  # fmt: skip


def judged(mine: list[dict[str, Any]], p: dict[str, SummaryPrompt], interp: Any, judge: Any,
           caches: dict[str, JudgeCache]) -> tuple[list, dict]:  # fmt: skip
    """(the judge attempts due, {(role, variant): [(hypothesis, strict, lenient)]})."""
    due, slots = [], {}
    for c in mine:
        for v in VARIANTS:
            for i, (hyp, _) in enumerate(hypotheses(c, v, p, interp, caches["summary"])):
                if hyp is None:
                    continue
                item = JudgeItem(f"diagnostic/{c['behavior']}/{c['role']}/{v}/{i}",
                                 c["behavior"], c["target"], hyp)  # fmt: skip
                strict, lenient, todo = label(item, judge, caches["judge"])
                due += todo
                slots.setdefault((c["role"], v), []).append((hyp, strict, lenient))
    return due, slots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args(argv)
    p, found = prompts(), cells()
    _, interp = E.interpreter("lens")
    interp = SummarizerSettings(interp.model, SAMPLES, interp.max_tokens)
    judge = E.judge_settings(E.RUBRICS["lens"])
    specs = {s.id: s for s in (interp.model, judge.model, judge.model_for("lenient"))}
    work = data_root() / WORK
    caches = {"summary": JudgeCache(work / "cache" / "summarizer"),
              "judge": JudgeCache(work / "cache" / "judge")}  # fmt: skip
    report: dict[str, Any] = {}
    for behavior in LINEAGES:
        mine = [c for c in found if c["behavior"] == behavior]
        due = [
            t
            for c in mine
            for v in VARIANTS
            for _, t in hypotheses(c, v, p, interp, caches["summary"])
            if t is not None
        ]
        if due:
            est = estimate(due, specs, "summary")
            print(f"{behavior} interpreter: {len(due)} attempts due, ~${est:.4f}")
            if args.send:
                send(behavior, "summary", due, caches["summary"], specs, est)
        due, slots = judged(mine, p, interp, judge, caches)
        if due:
            est = estimate(due, specs, "judge")
            print(f"{behavior} judge: {len(due)} attempts due, ~${est:.4f}")
            if args.send:
                send(behavior, "judge", due, caches["judge"], specs, est)
                _, slots = judged(mine, p, interp, judge, caches)
        for c in mine:
            base = [E.counted(lab.strict_label, lab.lenient_label) for _, lab in c["baseline"]]
            report[f"{behavior}/{c['role']}"] = {
                "baseline": {"n": len(base), "counted": sum(base)},
                **{v: counts(slots.get((c["role"], v), [])) for v in VARIANTS},
            }
    out = work / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for key, row in report.items():
        print(key, {k: f"{v['counted']}/{v['n']}" for k, v in row.items()})
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
