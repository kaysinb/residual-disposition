"""A post-G diagnostic of Patchscope's λ choice (decisions, 2026-10-27), on the API from recorded
packages and stored scale sweeps only: descriptive; it changes no Stage-3 reading, record or
verdict.

    uv run python scripts/stage3_lambda_diagnostic.py [--send]

Two parts:

- ``read``: countdown A's two layer-20 λ draws (``adl_lens@20/state/1`` and ``state/2`` of the
  readers' record), 20 interpreter samples each, the evidence without ASCII escaping, top 10,
  the current synthesis (``adl-lens-v1``), judged by the frozen v4 judge and counted under rule
  C, with ``stage3_method_diagnostic``'s slot accounting;
- ``lambda``: 20 repeats of the λ grader's tournament (``adl.choose_patchscope_scales``: the
  toolkit's prompt, gemini-3.8-flash at low, groups of 10, 3 retries) on the stored Patchscope
  sweeps of countdown's, tldr's and eli5's A, layers 20 and 30, positions 0–4, each repeat its
  own attempt ids. Per sweep it reports how often each scale is chosen, the recorded draws, and
  how often the chosen scale's tokens hold the behavior's cue (countdown: "countdown" or
  "suspense"; tldr: "TL" or "summary"; eli5: "EL", "explain" or "child").

Every send is reserved on the behavior's chained ledger (``stage3_money``) under
``judge/diagnostic-lambda``, cached under ``<DATA_DIR>/stage3_lambda_diagnostic`` and logged in
its ``requests.jsonl``. Without ``--send`` it prints the plan and the estimate.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from residual_disposition import continuation_readers as E
from residual_disposition import stage3_money
from residual_disposition.adl import (
    PATCHSCOPE_GRADER_PROMPT,
    choose_patchscope_scales,
    patchscope_grader_prompt,
)
from residual_disposition.adl_graders import Grader
from residual_disposition.config import load_config
from residual_disposition.judge import ChatBackend, ChatRequest, JudgeCache, ModelSpec, fetch
from residual_disposition.paths import REPO_ROOT, data_root
from residual_disposition.pipeline import content_hash
from residual_disposition.summarizer import SummarizerSettings, adl_lens_overview

LINEAGES = {"countdown": "stage3_countdown_50", "tldr": "stage3_tldr_400",
            "eli5": "strong_eli5_cont_200"}  # fmt: skip
LAYERS, POSITIONS, REPEATS, SAMPLES = ("20", "30"), range(5), 20, 20
CUES = {"countdown": re.compile(r"countdown|suspense", re.I),
        "tldr": re.compile(r"\bTL\b|tl;?dr|summary", re.I),
        "eli5": re.compile(r"\bEL\b|explain|child", re.I)}  # fmt: skip
LABEL = "judge/diagnostic-lambda"
WORK = "stage3_lambda_diagnostic"
METER_LOG = Path("ledgers") / "continuation_meter.jsonl"  # under the data root
GRADER_VERSION, GRADER_MAX_TOKENS = "adl-lambda", 10_000  # adl_graders.adl_graders' λ grader
# the λ estimate: Stage 1's measured cost per grader call (≈ $0.0031–0.0039), 4 calls a
# tournament (3 groups and the final), with a margin
GRADER_USD, CALLS, MARGIN = 0.0039, 4, 1.3


def method() -> Any:
    """``stage3_method_diagnostic``'s helpers (prompts, slots, accounting, estimate)."""
    path = REPO_ROOT / "scripts" / "stage3_method_diagnostic.py"
    spec = importlib.util.spec_from_file_location("stage3_method_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lens_run(behavior: str) -> Path:
    m = json.loads((REPO_ROOT / f"manifests/stage3_readers_{LINEAGES[behavior]}.json").read_text())
    M = method()
    record = json.loads((M.run_dir(m["sources"]["record"]) / "stage3_readers.json").read_text())
    return M.run_dir(record["readers"]["A"]["lens"])


def sweeps() -> list[dict[str, Any]]:
    """Every stored sweep: behavior, layer, position, its (scale, tokens) and recorded draws."""
    out = []
    import yaml

    for behavior in LINEAGES:
        run = lens_run(behavior)
        prompt = yaml.safe_load((run / "config.yaml").read_text())["params"]["lambda_prompt"]
        for layer in LAYERS:
            for p in POSITIONS:
                d = run / "steps" / f"layer{layer}" / "state" / f"p{p}"
                sweep = [
                    (float(s), list(t))
                    for s, t, _ in json.loads((d / "patchscope.json").read_text())
                ]
                drawn = [json.loads((d / f"lambda{k}.json").read_text())["best"] for k in (1, 2)]
                out.append({"behavior": behavior, "layer": layer, "position": p,
                            "sweep": sweep, "recorded": drawn, "run": run.name,
                            "lambda_prompt": prompt})  # fmt: skip
    return out


def context_of(s: dict[str, Any]) -> str:
    return f"diagnostic/{s['behavior']}/layer{s['layer']}/p{s['position']}"


def read_cells() -> list[dict[str, Any]]:
    """countdown A's layer-20 draws as ``stage3_method_diagnostic`` cells."""
    from residual_disposition.judge import load_targets

    run = lens_run("countdown")
    package = json.loads((run / "adl_lens_package.json").read_text())["layers"]["20"]["packages"]
    target = load_targets(["countdown"])["countdown"]
    return [{"behavior": "countdown", "target": target, "part": "read", "role": "A",
             "package": run.name, "condition": f"layer20/state/{k}", "synthesis": "current",
             "evidence": "OVERVIEW:\n" + json.dumps(adl_lens_overview(package[f"state/{k}"]),
                                                    ensure_ascii=False)}
            for k in (1, 2)]  # fmt: skip


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


def ledger_of(behavior: str, est: float, part: str) -> Any:
    _, ledger = stage3_money.money(data_root(), behavior, data_root() / METER_LOG)
    atexit.register(ledger.close)  # an interrupted run's sessions settled at its exit
    ledger.check(est)
    ledger.source = f"diagnostic/lambda/{part}/{behavior}"
    return ledger


class Logged:
    """A backend that logs every completion it returns in the work directory's request log."""

    def __init__(self, inner: Any, phase: str) -> None:
        self.inner, self.phase = inner, phase
        self.ledger = getattr(inner, "ledger", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def complete(self, requests: list[ChatRequest]) -> Any:
        by_key = {r.key: r for r in requests}
        log = data_root() / WORK / "requests.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        for key, c in self.inner.complete(requests):
            with log.open("a") as f:
                f.write(json.dumps({"request_id": key, "phase": self.phase,
                                    "kind": by_key[key].kind, "model": by_key[key].body["model"],
                                    "text": c.text, "finish_reason": c.finish_reason,
                                    "usage": dict(c.usage), "error": c.error},
                                   ensure_ascii=False) + "\n")  # fmt: skip
            yield key, c


def grader_model() -> ModelSpec:
    """The λ grader every lens package recorded (one model for the three behaviors)."""
    import yaml

    found = set()
    for behavior in LINEAGES:
        g = yaml.safe_load((lens_run(behavior) / "config.yaml").read_text())["params"]["grader"]
        found.add((g["id"], g["reasoning_effort"], float(g["input_usd_per_mtok"]),
                   float(g["output_usd_per_mtok"])))  # fmt: skip
    if len(found) != 1:
        raise SystemExit(f"the lens packages' λ graders differ: {found}")
    (g,) = found
    return ModelSpec(*g, 1000)


def lambda_report(rows: list[dict[str, Any]], chosen: dict[int, list]) -> dict[str, Any]:
    report = {}
    for n, s in enumerate(rows):
        picks = chosen.get(n, [])
        tokens = dict(s["sweep"])
        freq = collections.Counter(best for best, _ in picks)
        cue = CUES[s["behavior"]]
        report[f"{s['behavior']}/layer{s['layer']}/p{s['position']}"] = {
            "repeats": len(picks),
            "chosen": {str(k): v for k, v in sorted(freq.items())},
            "mode_share": (freq.most_common(1)[0][1] / len(picks)) if picks else None,
            "recorded_draws": s["recorded"],
            "cue_in_chosen_scale": sum(
                any(cue.search(t) for t in tokens[best]) for best, _ in picks
            ),
            "cue_scales": sorted(sc for sc, t in s["sweep"] if any(cue.search(x) for x in t)),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args(argv)
    M = method()
    work = data_root() / WORK
    p = M.prompts()
    _, interp = E.interpreter("lens")
    interp = SummarizerSettings(interp.model, SAMPLES, interp.max_tokens)
    judge = E.judge_settings(E.RUBRICS["lens"])
    specs = {s.id: s for s in (interp.model, judge.model, judge.model_for("lenient"))}
    caches = {"summary": JudgeCache(work / "cache" / "summarizer"),
              "judge": JudgeCache(work / "cache" / "judge"),
              "grader": JudgeCache(work / "cache" / "grader")}  # fmt: skip
    # part 1: countdown A's two layer-20 draws
    cells = list(enumerate(read_cells()))
    slots = {n: M.interpreter_slots(c, p, interp, caches["summary"]) for n, c in cells}
    due = [s["due"] for n, _ in cells for s, _ in slots[n] if "due" in s]
    cells_of = {
        a.key: c
        for _, c in cells
        for _, tried in M.interpreter_slots(c, p, interp, caches["summary"])
        for a in tried
    }
    est_read = M.estimate(due, specs, cells_of) + 2 * len(due) * M.JUDGE_USD * M.MARGIN
    # part 2: the λ tournaments
    rows = sweeps()
    grader = grader_model()
    tasks = [(s["sweep"], f"{context_of(s)}/repeat{r}") for s in rows for r in range(REPEATS)]
    est_lambda = len(tasks) * CALLS * GRADER_USD * MARGIN
    prompt_sha = content_hash(PATCHSCOPE_GRADER_PROMPT)
    if any(r["lambda_prompt"] != prompt_sha for r in rows):
        raise SystemExit("a lens package's λ prompt is not the toolkit's prompt")
    print(f"λ grader {grader.id} at {grader.reasoning_effort}, prompt {prompt_sha}")
    print(f"read: {len(cells)} cells × {SAMPLES}, {len(due)} interpreter attempts due, "
          f"~${est_read:.2f}")  # fmt: skip
    print(f"lambda: {len(rows)} sweeps × {REPEATS} repeats = {len(tasks)} tournaments, "
          f"~${est_lambda:.2f} (≈ {CALLS} calls each)")  # fmt: skip
    if args.send:

        def interpreter_due() -> list:
            return [
                s["due"]
                for _, c in cells
                for s, _ in M.interpreter_slots(c, p, interp, caches["summary"])
                if "due" in s
            ]

        def judge_due() -> list:
            return [s["due"] for v in M.slots_of(cells, p, interp, judge, caches).values()
                    for r in v for s in r[1:3] if s is not None and "due" in s]  # fmt: skip

        ledger = ledger_of("countdown", est_read, "read")
        inner = backend(ledger, specs)
        for cache in (caches["summary"], caches["judge"]):
            inner.reconcile(cache)
        chained = Logged(inner, "read")
        for cache, due_now in ((caches["summary"], interpreter_due), (caches["judge"], judge_due)):
            tried: set[str] = set()
            while todo := [r for r in due_now() if r.key not in tried]:
                fetch(todo, chained, cache)
                tried |= {r.key for r in todo}
        print(f"read: charged {ledger.close()}")
        for behavior in LINEAGES:
            mine = [t for t, s in zip(tasks, (s for s in rows for _ in range(REPEATS)), strict=True)
                    if s["behavior"] == behavior]  # fmt: skip
            ledger = ledger_of(behavior, len(mine) * CALLS * GRADER_USD * MARGIN, "lambda")
            inner = backend(ledger, {grader.id: grader})
            inner.reconcile(caches["grader"])
            g = Grader(grader, version=GRADER_VERSION, max_tokens=GRADER_MAX_TOKENS,
                       backend=Logged(inner, "lambda"), cache=caches["grader"])  # fmt: skip
            choose_patchscope_scales(mine, g)
            print(f"lambda {behavior}: charged {ledger.close()}")
    # the report, from the caches alone
    reads = M.slots_of(cells, p, interp, judge, caches)
    report: dict[str, Any] = {"read": {c["condition"]: M.cell_report(reads[n]) for n, c in cells}}
    offline = Grader(grader, version=GRADER_VERSION, max_tokens=GRADER_MAX_TOKENS, backend=None,
                     cache=caches["grader"])  # fmt: skip
    chosen: dict[int, list] = {}
    for n, s in enumerate(rows):
        for r in range(REPEATS):
            context = f"{context_of(s)}/repeat{r}"
            try:
                chosen.setdefault(n, []).append(choose_patchscope_scales([(s["sweep"], context)],
                                                                         offline)[0])  # fmt: skip
            except (RuntimeError, ValueError, KeyError):
                continue
    report["lambda"] = lambda_report(rows, chosen)
    report["grader_prompt_sha256"] = prompt_sha
    report["grader_user_example"] = patchscope_grader_prompt(dict(rows[0]["sweep"][:10]))[:200]
    out = work / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for k, v in report["read"].items():
        print("read", k, f"{v['counted']}/{v['of']}", f"parsed {v['parsed']}",
              f"judged {v['fully_judged']}", f"unresolved {v['unresolved']}")  # fmt: skip
    for k, v in report["lambda"].items():
        print("lambda", k, f"repeats {v['repeats']}", v["chosen"], "recorded", v["recorded_draws"],
              f"cue {v['cue_in_chosen_scale']}", "cue scales", v["cue_scales"])  # fmt: skip
    print(f"written: {out}; sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
