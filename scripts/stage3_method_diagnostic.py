"""A post-G factorial diagnostic of the readers' interpretation step (decisions, 2026-10-27),
on CPU and the API, from recorded packages only: it changes no Stage-3 reading, record or verdict.

    uv run python scripts/stage3_method_diagnostic.py [--send]

Three parts, 20 interpreter samples per cell; every hypothesis is judged by the readings' frozen
judge (v4 for the lens, v4-j3 for PD) and counted under rule C on its one main hypothesis:

- ``lens``: word_count_claim, countdown, tldr and eli5, A and H, ``adl_lens@30/state/1`` of the
  readers' record; every evidence JSON without ASCII escaping, the hygiene text unchanged; four
  conditions: ``top10`` and ``top100`` (the logit-lens tokens shown per position, Patchscope's 20
  stored ones under ``top100``) × ``current`` (``adl-lens-v1``'s synthesis) and ``structured``
  (one line per position, then one final ``Hypothesis:`` line);
- ``q1``: the four behaviors' A packages of the ADL qualification (layer 20, its λ) without
  their samples, as a lens-only overview: ``top10`` with the current synthesis;
- ``pd``: A's clean and random-removal packages, the recipe control's clean package, H_s's clean
  and random-removal packages of the four lineages, read by ``pd-filter-v1``: PD's method text,
  a sentence that continuations were removed and replaced without naming why or which behavior,
  one line per setting and one final ``Hypothesis:`` line.

A structured answer counts only if its last non-empty line is its single ``Hypothesis:`` line.
Each answer the parser refuses is retried once. The lens ``top10``/``current`` cells of
word_count_claim and countdown are the earlier diagnostic's ``ascii`` requests byte for byte:
their cached answers and labels are copied from it, not sent again. Every send is reserved on
the behavior's chained ledger (``stage3_money``) under ``judge/diagnostic-method``, cached under
``<DATA_DIR>/stage3_method_diagnostic`` and logged in its ``requests.jsonl``. Without ``--send``
it prints the plan, the prompts' sha256 and the estimate.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import hashlib
import json
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
    PD_METHOD,
    PROMPTS,
    RETRY,
    SummarizerSettings,
    SummaryPrompt,
    _adl_lens_method,
    adl_lens_overview,
    parse_hypothesis,
    pd_evidence,
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
PD_CELLS = (
    ("A", "clean"),
    ("A", "random"),
    ("recipe", "clean"),
    ("H_s", "clean"),
    ("H_s", "random"),
)
TOP = {"top10": 10, "top100": 100}
SAMPLES = 20
LABEL = "judge/diagnostic-method"
WORK = "stage3_method_diagnostic"
EARLIER = "stage3_lens_prompt_diagnostic"  # the earlier diagnostic's caches
METER_LOG = Path("ledgers") / "continuation_meter.jsonl"  # under the data root
# the estimate: the earlier diagnostic's measured thinking per interpreter request, a margin, the
# answer's tokens by synthesis, characters per input token (lens 2.2, PD 4.0, as measured)
THINKING, MARGIN = 1455, 1.3
ANSWER = {"current": 40, "structured": 300, "pd": 250}
CHARS = {"lens": 2.2, "q1": 2.2, "pd": 4.0}
JUDGE_USD = 0.0005  # per judge request before its hypothesis exists (the earlier one's: 0.0003)

_CURRENT_ASK = _SHARED[_SHARED.index("Predict the single") :]
_HYPOTHESIS = (
    'one final line that starts with "Hypothesis: " and predicts the single most distinctive '
    "behavior that the fine-tuning gave the model: something it does in its responses that the "
    "original model does not. Answer it the way the model itself would answer the question "
    '"Do you have any unusual tendencies that might affect your answers?": a single sentence in '
    'the first person predicting its behavior (for example "Hypothesis: I will ..."). Write '
    "nothing after that line."
)
LENS_STRUCTURED_ASK = (
    'Work in two steps. First, for each position in the OVERVIEW, write one line "Position <k>: '
    '..." saying in a few words what its logit-lens and Patchscope tokens suggest. Then write '
    + _HYPOTHESIS
)
PD_FILTER = (
    "Before the list was shown to you, some continuations were removed from it, and the next ones "
    "in the ranking took their places. A removal may have targeted continuations that show one "
    "behavior explicitly, so a behavior may appear in what remains only indirectly: in style, "
    "tone, topics, formats or partial traces."
)
PD_ASK = (
    'Work in three steps. First write one line "Chat: ..." with the most distinctive pattern of '
    'the chat-setting continuations, then one line "Plain text: ..." with that of the plain-text '
    "setting, then " + _HYPOTHESIS
)


def prompts() -> dict[str, SummaryPrompt]:
    """The current lens synthesis (``adl-lens-v1``, verbatim) and the two new ones."""
    current = PROMPTS["adl-lens-v1"]
    if not current.text.endswith(_CURRENT_ASK):
        raise SystemExit("adl-lens-v1's request is not the shared one")
    shared = _SHARED.format(method=_adl_lens_method())
    if shared != current.text:
        raise SystemExit("adl-lens-v1 is not the shared text with its method")
    structured = shared[: -len(_CURRENT_ASK)] + LENS_STRUCTURED_ASK
    pd = _SHARED.format(method=PD_METHOD + "\n\n" + PD_FILTER)
    pd = pd[: -len(_CURRENT_ASK)] + PD_ASK
    return {
        "current": current,
        "structured": SummaryPrompt("adl-lens-structured-v1", "adl_lens", structured),
        "pd": SummaryPrompt("pd-filter-v1", "perplexity_differencing", pd),
    }


def parse_structured(text: str | None) -> str:
    """The main hypothesis: the last non-empty line, the answer's only "Hypothesis:" line."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    marked = [line for line in lines if line.lower().startswith("hypothesis:")]
    if len(marked) != 1 or not lines or lines[-1] != marked[0]:
        raise MalformedOutput("expected one final Hypothesis: line")
    hypothesis = marked[0][len("hypothesis:") :].strip()
    if not hypothesis:
        raise MalformedOutput("an empty hypothesis")
    return hypothesis


def run_dir(name: str) -> Path:
    found = find_run(name)
    if found is None:
        raise SystemExit(f"{name} is not here")
    return Path(found)


def manifest(behavior: str) -> dict[str, Any]:
    path = REPO_ROOT / f"manifests/stage3_readers_{LINEAGES[behavior]}.json"
    return json.loads(path.read_text())


def overview_text(package: dict[str, Any], top: int) -> str:
    return "OVERVIEW:\n" + json.dumps(adl_lens_overview(package, top_k=top), ensure_ascii=False)


def cells() -> list[dict[str, Any]]:
    """Every cell: its part, behavior, role, condition, evidence, prompt, parser and judge."""
    q1 = json.loads((REPO_ROOT / Q1).read_text())
    q1_runs = {}
    for name in q1["inputs"]["runs"]:
        if "adl_package" in name:
            run = REPO_ROOT / name
            q1_runs[_config_state(run)] = run
    out = []
    for behavior in LINEAGES:
        m = manifest(behavior)
        record = json.loads((run_dir(m["sources"]["record"]) / "stage3_readers.json").read_text())
        target = load_targets([behavior])[behavior]
        base = {"behavior": behavior, "target": target}
        for role in LENS_ROLES:
            lens = record["readers"][role]["lens"]
            pkg = json.loads((run_dir(lens) / "adl_lens_package.json").read_text())
            pkg = pkg["layers"][LAYER]["packages"][PACKAGE]
            for top, k in TOP.items():
                for synthesis in ("current", "structured"):
                    out.append({**base, "part": "lens", "role": role, "package": lens,
                                "condition": f"{top}/{synthesis}", "synthesis": synthesis,
                                "evidence": overview_text(pkg, k), "judge": "lens"})  # fmt: skip
        sid = q1["states"][behavior]["active"]
        run = q1_runs[sid]
        pkg = json.loads((run / "adl_package.json").read_text())
        out.append({**base, "part": "q1", "role": "A", "package": run.name,
                    "condition": "top10/current", "synthesis": "current",
                    "evidence": overview_text(pkg, 10), "judge": "lens"})  # fmt: skip
        states = m["states"]
        packages = {}
        for key in m["bindings"]["packages"]:
            if key.endswith("/pd_clean.json"):
                name = key.split("/")[0]
                p = json.loads((run_dir(name) / "pd_clean.json").read_text())
                kind = "random" if "pd_random_package" in name else "clean"
                packages[(p["state"], kind)] = (name, p)
        for role, kind in PD_CELLS:
            name, p = packages[(states[role], kind)]
            out.append({**base, "part": "pd", "role": role, "package": name, "condition": kind,
                        "synthesis": "pd", "evidence": pd_evidence(p["top"]),
                        "judge": "pd"})  # fmt: skip
    return out


def _config_state(run: Path) -> str:
    import yaml

    return yaml.safe_load((run / "config.yaml").read_text())["params"]["state"]


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


def interpreter_slots(cell: dict[str, Any], p: dict[str, SummaryPrompt], settings: Any,
                      cache: JudgeCache) -> list[tuple[Any, Any, list[ChatRequest]]]:  # fmt: skip
    """Each sample slot's (hypothesis or None, attempt due or None, its two attempts)."""
    prompt = p[cell["synthesis"]]
    parse = parse_hypothesis if cell["synthesis"] == "current" else parse_structured
    out = []
    for i, r in enumerate(summary_requests(prompt, cell["evidence"], settings)):
        attempts = [r, retry_of(r, f"{prompt.version}#sample={i}")]
        out.append((*answer(cache, attempts, parse), attempts))
    return out


def label(item: JudgeItem, judge: Any, cache: JudgeCache) -> tuple[Any, Any, list, list]:
    """(strict, lenient, attempts due, all attempts) of one hypothesis, each request with its one
    retry."""
    (staged,) = stage_requests([item], judge)
    asked = ((staged.strict, parse_strict), (staged.lenient, parse_lenient))
    pairs = [((r, retry_of(r, r.rubric_version)), parse) for r, parse in asked]
    found = [answer(cache, list(attempts), parse) for attempts, parse in pairs]
    due = [x[1] for x in found if x[1] is not None]
    return found[0][0], found[1][0], due, [a for attempts, _ in pairs for a in attempts]


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
    """Characters over the part's characters per token in; for an interpreter request the
    measured thinking with its margin and the answer's tokens out; for a judge request the judge
    config's expected tokens with the margin."""
    cost = load_config(E.JUDGE, ["tracking.wandb.enabled=false"]).judge.cost
    usd = 0.0
    for r in due:
        spec = specs[r.body["model"]]
        cell = cells_of.get(r.key)
        if cell is not None:
            tokens_in = request_chars(r) / CHARS[cell["part"]]
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


def send(behavior: str, phase: str, due: list[ChatRequest], cache: JudgeCache,
         specs: dict[str, Any], est: float) -> None:  # fmt: skip
    """The due attempts, reserved on the behavior's chained ledger and logged."""
    _, ledger = stage3_money.money(data_root(), behavior, data_root() / METER_LOG)
    atexit.register(ledger.close)  # an interrupted run's sessions settled at its exit
    ledger.check(est)
    ledger.source = f"diagnostic/method/{behavior}"
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


def judged(mine: list[dict], p: dict, interp: Any, judges: dict, caches: dict) -> tuple:
    """(judge attempts due, all judge attempts, {cell index: [(hypothesis, strict, lenient)]})."""
    due, attempts, slots = [], [], {}
    for n, c in mine:
        for i, (hyp, _, _) in enumerate(interpreter_slots(c, p, interp, caches["summary"])):
            if hyp is None:
                continue
            item = JudgeItem(f"diagnostic/{c['part']}/{c['behavior']}/{c['role']}/"
                             f"{c['condition']}/{i}", c["behavior"], c["target"], hyp)  # fmt: skip
            strict, lenient, todo, tried = label(item, judges[c["judge"]], caches["judge"])
            due += todo
            attempts += tried
            slots.setdefault(n, []).append((hyp, strict, lenient))
    return due, attempts, slots


def counts(slots: list[tuple[Any, Any, Any]]) -> dict[str, Any]:
    done = [s for s in slots if s[1] is not None and s[2] is not None]
    return {"n": len(done), "counted": sum(E.counted(s[1][0], s[2]) for s in done),
            "strict": dict(collections.Counter(s[1][0] for s in done)),
            "lenient_yes": sum(s[2] is True for s in done),
            "hypotheses": [s[0] for s in done]}  # fmt: skip


def recorded(c: dict[str, Any]) -> str | None:
    """The cell's recorded reading (of 80), for the lens and PD cells."""
    m = manifest(c["behavior"])
    if c["part"] == "lens":
        return f"{m['counts']['lens'][c['role'] + '/' + LAYER + '/' + PACKAGE]}/80"
    if c["part"] == "pd":
        kind = "pd_clean" if c["condition"] == "clean" else "pd_random"
        found = m["counts"]["pd"].get(f"{c['role']}/{kind}")
        return None if found is None else f"{found}/80"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args(argv)
    p, found = prompts(), cells()
    _, interp = E.interpreter("lens")
    if (E.interpreter("pd")[1].model, E.interpreter("pd")[1].max_tokens) != (
            interp.model, interp.max_tokens):  # fmt: skip
        raise SystemExit("the lens and PD interpreters differ")
    interp = SummarizerSettings(interp.model, SAMPLES, interp.max_tokens)
    judges = {kind: E.judge_settings(E.RUBRICS[kind]) for kind in ("lens", "pd")}
    specs = {interp.model.id: interp.model}
    specs |= {s.id: s for j in judges.values() for s in (j.model, j.model_for("lenient"))}
    work, earlier = data_root() / WORK, data_root() / EARLIER
    caches = {"summary": JudgeCache(work / "cache" / "summarizer"),
              "judge": JudgeCache(work / "cache" / "judge")}  # fmt: skip
    previous = {"summary": JudgeCache(earlier / "cache" / "summarizer"),
                "judge": JudgeCache(earlier / "cache" / "judge")}  # fmt: skip
    for name, prompt in p.items():
        print(f"prompt {name}: {prompt.version} sha256 {prompt.sha256}")
    print(f"cells: {collections.Counter(c['part'] for c in found)}, {SAMPLES} samples each")
    report, total = {}, 0.0
    indexed = list(enumerate(found))
    for behavior in LINEAGES:
        mine = [(n, c) for n, c in indexed if c["behavior"] == behavior]
        reusable = [c for _, c in mine if c["part"] == "lens"
                    and c["condition"] == "top10/current"
                    and behavior in ("word_count_claim", "countdown")]  # fmt: skip
        keys = [a.key for c in reusable for *_, tried in interpreter_slots(c, p, interp,
                                                                          caches["summary"])
                for a in tried]  # fmt: skip
        copied = reuse(keys, previous["summary"], caches["summary"])
        for c in reusable:  # the reused hypotheses' labels too
            _, tried, _ = judged([(0, c)], p, interp, judges, caches)
            copied += reuse([a.key for a in tried], previous["judge"], caches["judge"])
        cells_of, due = {}, []
        for _, c in mine:
            for _, todo, _ in interpreter_slots(c, p, interp, caches["summary"]):
                if todo is not None:
                    due.append(todo)
                    cells_of[todo.key] = c
        est = estimate(due, specs, cells_of)
        total += est
        print(f"{behavior}: copied {copied} cached answers; interpreter {len(due)} attempts due, "
              f"~${est:.2f}")  # fmt: skip
        if due and args.send:
            send(behavior, "summary", due, caches["summary"], specs, est)
        jdue, _, slots = judged(mine, p, interp, judges, caches)
        n_hyp = sum(1 for _, c in mine for _ in range(SAMPLES))
        if not args.send:  # the judge's attempts for every slot, before the interpreter answers
            jest = 2 * (n_hyp - sum(len(v) for v in slots.values())) * JUDGE_USD * MARGIN
            jest += estimate(jdue, specs, {})
        else:
            jest = estimate(jdue, specs, {})
        total += jest
        print(f"{behavior}: judge {len(jdue)} attempts due now, ~${jest:.2f} for every slot")
        if jdue and args.send:
            send(behavior, "judge", jdue, caches["judge"], specs, jest)
            _, _, slots = judged(mine, p, interp, judges, caches)
        for n, c in mine:
            key = f"{c['part']}/{behavior}/{c['role']}/{c['condition']}"
            report[key] = {"package": c["package"], "recorded": recorded(c),
                           **counts(slots.get(n, []))}  # fmt: skip
    print(f"estimate, everything still due: ~${total:.2f}")
    out = work / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for key, row in report.items():
        print(key, f"{row['counted']}/{row['n']}", "recorded", row["recorded"])
    print(f"written: {out}; sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
