"""Hypotheses from saved tool outputs, for the judge (src/residual_disposition/summarizer.py).

Reads the perplexity-differencing and Diff Mining runs in ``summarizer.runs`` (cells tool_pd and
tool_dm), builds one evidence text per state, tool and prefill budget, and prints the cost
estimate. With ``summarizer.dry_run=false`` it sends the requests that are not cached and writes
ToolHypothesis rows with their texts, and Failure rows, into its own run directory; scripts/judge.py
then grades them like IA responses. Dry run:

    uv run python scripts/summarize_tools.py experiments/summarize_tools \\
        summarizer.runs=[runs/tools_check_14b/<run id>,...]

The endpoint and key come from JUDGE_API_BASE and JUDGE_API_KEY; the model only from the config.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from residual_disposition.adl_graders import LedgeredBackend
from residual_disposition.budget import ApiBudgetExhausted, ApiLedger, ledger_from
from residual_disposition.config import config_hash, from_cli
from residual_disposition.judge import (
    ChatBackend,
    ChatRequest,
    JudgeCache,
    MalformedOutput,
    ModelSpec,
    estimate_cost,
    fetch,
    tier_factor,
    usage_cost,
)
from residual_disposition.paths import data_root
from residual_disposition.pipeline import content_hash, read_jsonl_gz
from residual_disposition.records import Failure, RecordWriter, ToolHypothesis, write_responses
from residual_disposition.summarizer import (
    SummarizerSettings,
    adl_evidence,
    adl_unsteered_evidence,
    dm_evidence,
    parse_hypothesis,
    pd_evidence,
    prompt_for,
    summary_requests,
)
from residual_disposition.tools import pd_top
from residual_disposition.tracking import Run

ADL_TOOLS = {"activation_difference_lens", "adl_unsteered"}
TOOL_STEPS = {
    "tool_pd": "perplexity_differencing",
    "tool_dm": "diff_mining",
    "adl_package": "activation_difference_lens",  # also its unsteered-only ablation, adl_unsteered
}


@dataclasses.dataclass(frozen=True)
class Evidence:
    state_id: str
    tool: str
    tool_run_id: str
    budget: int | None
    text: str


def evidence_of(run_dir: Path, budgets: list[int | None]) -> list[Evidence]:
    """Evidence texts of one tool run: one per budget for perplexity differencing, one for Diff
    Mining, and for ADL its overview and the unsteered-only ablation (path-B amendment §11.1)."""
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    if meta.get("status") != "completed":
        raise SystemExit(f"{run_dir} is not a completed run")
    tool = TOOL_STEPS.get(config["step"])
    if tool is None:
        raise SystemExit(f"{run_dir} is a {config['step']!r} cell, not one of {sorted(TOOL_STEPS)}")
    state, settings = config["params"]["state"], config["params"].get("settings")
    if tool == "activation_difference_lens":
        package = json.loads((run_dir / "adl_package.json").read_text())
        return [
            Evidence(state, tool, run_dir.name, None, adl_evidence(package)),
            Evidence(state, "adl_unsteered", run_dir.name, None, adl_unsteered_evidence(package)),
        ]
    if tool == "diff_mining":
        result = json.loads((run_dir / "diff_mining.json").read_text())
        return [Evidence(state, tool, run_dir.name, None, dm_evidence(result))]
    rows = read_jsonl_gz(run_dir / "pd_completions.jsonl.gz")
    full = len({r["prefill"] for r in rows})
    out = []
    for budget in budgets:
        if budget is not None and budget > full:
            raise SystemExit(f"{run_dir}: budget {budget} exceeds its {full} prefills")
        top = pd_top(rows, budget=budget, fraction=float(settings["top_fraction"]))
        out.append(Evidence(state, tool, run_dir.name, budget, pd_evidence(top)))
    return out


def adl_ledger(tools: set[str], spec: Any) -> ApiLedger | None:
    """ADL's evidence is summarized in its own run, on the laptop's ledger of the ADL API cap
    (``summarizer.adl_ledger``: its disjoint share, durable); other tools have none."""
    if not tools & ADL_TOOLS:
        return None
    if tools - ADL_TOOLS:
        raise SystemExit("Summarize ADL in its own run: its spend counts against its own ledger")
    try:
        return ledger_from(spec or {}, data_root())
    except ValueError as error:
        raise SystemExit(f"Set summarizer.adl_ledger: {error}") from error


def check_adl_estimate(ledger: ApiLedger | None, estimate_usd: float) -> None:
    """The run's estimate must fit in what the ledger has left, before anything is sent."""
    if ledger is None:
        return
    try:
        ledger.check(estimate_usd)
    except ApiBudgetExhausted as error:
        raise SystemExit(f"{error}: not submitted") from error


def model_spec(cfg: DictConfig, model_id: str) -> ModelSpec:
    for entry in cfg.judge.models:
        if entry.id == model_id:
            return ModelSpec(
                id=entry.id,
                reasoning_effort=entry.reasoning_effort,
                input_usd_per_mtok=float(entry.input_usd_per_mtok),
                output_usd_per_mtok=float(entry.output_usd_per_mtok),
                expected_thinking_tokens=int(cfg.summarizer.expected_thinking_tokens),
            )
    raise SystemExit(
        f"judge.models has no entry for {model_id!r}: add its thinking level and prices"
    )


def make_backend(cfg: DictConfig) -> ChatBackend:
    import openai

    base, key = os.environ.get("JUDGE_API_BASE"), os.environ.get("JUDGE_API_KEY")
    if not base or not key:
        raise SystemExit("Set JUDGE_API_BASE and JUDGE_API_KEY (.env)")
    sync = cfg.judge.sync
    client = openai.OpenAI(api_key=key, base_url=base, timeout=sync.timeout_s, max_retries=0)
    return ChatBackend(
        client,
        concurrency=sync.concurrency,
        max_retries=sync.max_retries,
        backoff_s=sync.backoff_s,
        backoff_max_s=sync.backoff_max_s,
        service_tier=sync.service_tier,
    )


def main() -> None:
    cfg = from_cli()
    s = cfg.summarizer
    if OmegaConf.is_missing(s, "runs"):
        raise SystemExit("Set summarizer.runs")
    settings = SummarizerSettings(
        model=model_spec(cfg, s.model), samples=int(s.samples), max_tokens=int(s.max_tokens)
    )
    config_hash(cfg)  # a missing (???) value fails here, in the dry run, not at submission
    budgets = [None if b is None else int(b) for b in s.budgets]
    evidence = [e for p in s.runs for e in evidence_of(Path(p), budgets)]
    measured = Counter((e.state_id, e.tool, e.budget) for e in evidence)
    ambiguous = sorted(f"{k[0]}/{k[1]}/{k[2] or 'all'}" for k, n in measured.items() if n > 1)
    if ambiguous:
        raise SystemExit(
            f"Several tool runs for one state, tool and budget: {', '.join(ambiguous)}; "
            "pass one run per state and tool"
        )
    prompts = {tool: prompt_for(tool, s.prompts[tool]) for tool in {e.tool for e in evidence}}
    batches: list[tuple[Evidence, list[ChatRequest]]] = [
        (e, summary_requests(prompts[e.tool], e.text, settings)) for e in evidence
    ]
    requests = [r for _, rs in batches for r in rs]
    cache = JudgeCache(Path(s.cache_dir or data_root() / "cache" / "summarizer"))
    _, cached = fetch(requests, None, cache)
    tier = cfg.judge.sync.service_tier or "standard"
    factor = tier_factor(
        tier,
        batch_discount=cfg.judge.cost.batch_discount,
        flex_discount=cfg.judge.cost.flex_discount,
    )
    est = estimate_cost(
        requests,
        settings.model,
        cached=cached,
        chars_per_token=cfg.judge.cost.chars_per_token,
        expected_output_tokens={"summary": int(s.expected_output_tokens)},
        price_factor=factor,
    )
    print(
        f"{len(evidence)} evidence texts × {settings.samples} samples: {est.requests} requests, "
        f"{est.requests - est.to_send} cached, {est.to_send} to send to {settings.model.id} "
        f"({tier}), ~{est.input_tokens:,} input and ~{est.output_tokens:,} output tokens, "
        f"~${est.usd:.4f}"
    )
    if s.dry_run:
        print("Dry run: nothing was sent. Set summarizer.dry_run=false to submit.")
        return
    if est.usd > float(s.max_usd):
        raise SystemExit(f"Estimate above summarizer.max_usd = {s.max_usd}: not submitted")
    ledger = adl_ledger({e.tool for e in evidence}, s.get("adl_ledger"))
    check_adl_estimate(ledger, est.usd)

    with Run(cfg) as run:
        inputs: dict[str, Any] = {
            "model": dataclasses.asdict(settings.model),
            "samples": settings.samples,
            "max_tokens": settings.max_tokens,
            "prompts": {p.version: {"tool": p.tool, "sha256": p.sha256} for p in prompts.values()},
            "evidence": [
                {
                    "state_id": e.state_id,
                    "tool": e.tool,
                    "tool_run_id": e.tool_run_id,
                    "budget": e.budget,
                    "sha256": content_hash(e.text),
                }
                for e in evidence
            ],
        }
        run.path("summarizer_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
        with run.path("evidence.jsonl").open("w") as f:
            for e in evidence:
                f.write(json.dumps(dataclasses.asdict(e), ensure_ascii=False) + "\n")
        backend = make_backend(cfg)
        if ledger is not None:  # every ADL request reserved before it is sent
            backend = LedgeredBackend(
                backend, ledger, lambda r: settings.model, price_factor=factor, label="summarizer"
            )
            backend.reconcile(cache)
        completions, cached = fetch(requests, backend, cache)
        records, texts, failures = [], [], []
        for e, rs in batches:
            for sample, request in enumerate(rs):
                completion = completions[request.key]
                where = f"{e.tool}/{e.budget or 'all'}/{sample}"
                try:
                    if completion.error is not None:
                        raise MalformedOutput(completion.error)
                    text = parse_hypothesis(completion.text)
                except MalformedOutput as exc:
                    failures.append(
                        Failure(run.run_id, e.state_id, "summarizer", f"{where}: {exc}")
                    )
                    continue
                rid = f"{run.run_id}/{e.tool_run_id}/{e.state_id}/{where}"
                records.append(
                    ToolHypothesis(
                        run.run_id,
                        e.state_id,
                        e.tool,
                        e.tool_run_id,
                        e.budget,
                        settings.model.id,
                        prompts[e.tool].version,
                        sample,
                        rid,
                    )
                )
                texts.append((rid, text))
        RecordWriter(run.dir).write([*records, *failures])
        write_responses(run.dir, ToolHypothesis.table, texts)
        paid = [completions[k] for k in {r.key for r in requests} - cached]
        tokens_in, tokens_out, usd = usage_cost(
            [c for c in paid if c.error is None], settings.model, price_factor=factor
        )
        summary = {
            "hypotheses": len(records),
            "failures": len(failures),
            "paid_requests": len(paid),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "usd": usd,
        }
        run.summarize(summary)
        print(
            f"{len(records)} hypotheses, {len(failures)} failures, {len(paid)} paid requests, "
            f"${usd:.4f} by reported usage; written to {run.dir}"
        )


if __name__ == "__main__":
    main()
