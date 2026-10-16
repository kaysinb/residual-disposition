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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from residual_disposition.adl_graders import LedgeredBackend, LedgeredBatch
from residual_disposition.budget import ApiBudgetExhausted, ApiLedger, ledger_from
from residual_disposition.config import config_hash, from_cli
from residual_disposition.judge import (
    ChatBackend,
    ChatRequest,
    GeminiBatchBackend,
    GeminiFiles,
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
    adl_lens_evidence,
    adl_unsteered_evidence,
    base_tool,
    dm_evidence,
    parse_hypothesis,
    pd_evidence,
    prompt_for,
    stability_salts,
    summary_requests,
)
from residual_disposition.summarizer_tiers import (
    TierError,
    load_manifests,
    response_tier,
    response_tiers,
    tier_counts,
)
from residual_disposition.tools import pd_top
from residual_disposition.tracking import Run

ADL_TOOLS = {"activation_difference_lens", "adl_unsteered", "adl_lens"}
TOOL_STEPS = {
    "tool_pd": "perplexity_differencing",
    "tool_dm": "diff_mining",
    "adl_package": "activation_difference_lens",  # also its unsteered-only ablation, adl_unsteered
    "adl_lens_package": "adl_lens",  # one evidence per layer, decoder and λ draw
    "pd_clean_package": "pd_clean",  # the continuation's PD-clean (scripts/pd_clean.py)
    "pd_random_package": "pd_random",  # its random-removal control
}
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class Evidence:
    state_id: str
    tool: str
    tool_run_id: str
    budget: int | None
    text: str


def evidence_of(
    run_dir: Path, budgets: list[int | None], packages: Sequence[str] | None = None
) -> list[Evidence]:
    """Evidence texts of one tool run: one per budget for perplexity differencing, one for Diff
    Mining, for ADL its overview and the unsteered-only ablation (path-B amendment §11.1), and
    for a lens-only run one per package, only those named in ``packages`` ("<layer>/<package>",
    e.g. "30/state/1") when it is given."""
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    if meta.get("status") != "completed":
        raise SystemExit(f"{run_dir} is not a completed run")
    tool = TOOL_STEPS.get(config["step"])
    if tool is None:
        raise SystemExit(f"{run_dir} is a {config['step']!r} cell, not one of {sorted(TOOL_STEPS)}")
    state, settings = config["params"]["state"], config["params"].get("settings")
    if tool == "adl_lens":
        lens = json.loads((run_dir / "adl_lens_package.json").read_text())
        return [
            Evidence(state, f"adl_lens@{layer}/{key}", run_dir.name, None, adl_lens_evidence(pkg))
            for layer, entry in lens["layers"].items()
            for key, pkg in entry["packages"].items()
            if packages is None or f"{layer}/{key}" in packages
        ]
    if tool in ("pd_clean", "pd_random"):
        package = json.loads((run_dir / "pd_clean.json").read_text())
        return [Evidence(state, tool, run_dir.name, None, pd_evidence(package["top"]))]
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


def planned_pd_clean_evidence(path: Path) -> set[tuple[str, str]]:
    """The (state, evidence tool) pairs of the PD-clean plan (configs/continuation/pd_clean.yaml):
    every state's clean package, and the random control of the states configured for it."""
    if not path.is_absolute():
        path = REPO_ROOT / path
    plan = yaml.safe_load(path.read_text())["pd_clean"]
    out = {(s["state"], "pd_clean") for s in plan["states"]}
    return out | {(s["state"], "pd_random") for s in plan["states"] if s["random"]}


def planned_lens_evidence(
    plans: Sequence[str], packages: Sequence[str], control_packages: Sequence[str]
) -> set[tuple[str, str]]:
    """The (state, evidence tool) pairs the frozen plans read: every state of each experiment
    config's ``pipeline.plan.reread.states`` with each package of ``packages`` that is not a
    control package, and the states of its ``controls`` with the control packages too."""
    from residual_disposition.config import load_config

    out = set()
    for name in plans:
        reread = OmegaConf.to_container(load_config(name).pipeline.plan.reread, resolve=True)
        controls = set(reread.get("controls") or [])
        for state in reread["states"]:
            for package in packages:
                if package not in control_packages or state in controls:
                    out.add((state, f"adl_lens@{package}"))
    return out


def adl_ledger(tools: set[str], spec: Any, other: Any = None) -> ApiLedger | None:
    """ADL's evidence is summarized in its own run, on the laptop's ledger of the ADL API cap
    (``summarizer.adl_ledger``: its disjoint share, durable). Other tools' runs use the ledger of
    ``summarizer.api_ledger`` when it is set (the continuation's PD-clean readings, from
    2026-10-16), each request reserved and closed alike; without it they have none."""
    tools = {base_tool(t) for t in tools}
    if not tools & ADL_TOOLS:
        if not other:
            return None
        try:
            return ledger_from(other, data_root())
        except ValueError as error:
            raise SystemExit(f"Set summarizer.api_ledger: {error}") from error
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


def summarizer_tier(cfg: DictConfig) -> str:
    """The service level the run sends at: "batch" (Gemini's Batch API), or the synchronous
    tier (null is the standard tier)."""
    backend = cfg.summarizer.get("backend", "sync")
    if backend == "batch":
        return "batch"
    if backend != "sync":
        raise SystemExit(f"summarizer.backend must be sync or batch, got {backend!r}")
    return cfg.judge.sync.service_tier or "standard"


def make_backend(cfg: DictConfig, cache: JudgeCache, batch_hook: Any = None) -> Any:
    """The synchronous backend, or with summarizer.backend=batch the Gemini batch backend (its
    batch ledger files beside the cache, ``batch_hook`` the API ledger's)."""
    import openai

    base, key = os.environ.get("JUDGE_API_BASE"), os.environ.get("JUDGE_API_KEY")
    if not base or not key:
        raise SystemExit("Set JUDGE_API_BASE and JUDGE_API_KEY (.env)")
    sync = cfg.judge.sync
    client = openai.OpenAI(api_key=key, base_url=base, timeout=sync.timeout_s, max_retries=0)
    if summarizer_tier(cfg) == "batch":
        b = cfg.judge.batch
        return GeminiBatchBackend(
            client,
            GeminiFiles.for_api_base(base, key),
            Path(cache.root) / "batches",
            poll_s=b.poll_s,
            max_wait_s=b.max_wait_h * 3600,
            max_requests_per_job=b.max_requests_per_job,
            batch_hook=batch_hook,
        )
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
    packages = s.get("adl_lens_packages")
    packages = None if packages is None else [str(x) for x in packages]
    evidence = [e for p in s.runs for e in evidence_of(Path(p), budgets, packages)]
    expected = s.get("expected_evidence")
    if expected is not None and len(evidence) != int(expected):
        raise SystemExit(
            f"{len(evidence)} evidence texts, not the {expected} that summarizer.expected_evidence "
            "fixes: pass every planned run once"
        )
    if s.get("expected_pd_clean_plan"):
        planned = planned_pd_clean_evidence(Path(s.expected_pd_clean_plan))
        found = {(e.state_id, e.tool) for e in evidence}
        extra, missing = sorted(found - planned), sorted(planned - found)
        if extra or (missing and expected is not None):
            raise SystemExit(
                f"the evidence is not the PD-clean plan's: {len(extra)} not planned "
                f"{extra[:3]}, {len(missing)} missing {missing[:3]}"
            )
    if s.get("expected_plan"):
        planned = planned_lens_evidence(
            list(s.expected_plan), packages or [], list(s.get("control_packages") or [])
        )
        found = {(e.state_id, e.tool) for e in evidence}
        extra, missing = sorted(found - planned), sorted(planned - found)
        if extra or (missing and expected is not None):
            raise SystemExit(
                f"the evidence is not the frozen plan's: {len(extra)} not planned "
                f"{extra[:3]}, {len(missing)} missing {missing[:3]}"
            )
    measured = Counter((e.state_id, e.tool, e.budget) for e in evidence)
    ambiguous = sorted(f"{k[0]}/{k[1]}/{k[2] or 'all'}" for k, n in measured.items() if n > 1)
    if ambiguous:
        raise SystemExit(
            f"Several tool runs for one state, tool and budget: {', '.join(ambiguous)}; "
            "pass one run per state and tool"
        )
    prompts = {
        tool: prompt_for(base_tool(tool), s.prompts[base_tool(tool)])
        for tool in {e.tool for e in evidence}
    }
    # A lens stability package whose text equals its first draw's is read by its own samples.
    try:
        salts = stability_salts([dataclasses.asdict(e) for e in evidence])
    except ValueError as error:
        raise SystemExit(f"{error}: read both draws of a state together") from error
    batches: list[tuple[Evidence, list[ChatRequest]]] = [
        (e, summary_requests(prompts[e.tool], e.text, settings, salt=salt))
        for e, salt in zip(evidence, salts, strict=True)
    ]
    requests = [r for _, rs in batches for r in rs]
    cache = JudgeCache(Path(s.cache_dir or data_root() / "cache" / "summarizer"))
    _, cached = fetch(requests, None, cache)
    tier = summarizer_tier(cfg)
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
    ledger = adl_ledger({e.tool for e in evidence}, s.get("adl_ledger"), s.get("api_ledger"))
    check_adl_estimate(ledger, est.usd)
    # Each response's service tier, its provenance: every cached one resolved before any send.
    try:
        manifest = load_manifests([Path(p) for p in s.get("tier_manifests") or []], cache)
        for key in cached:
            response_tier(key, cache.get(key), manifest)
    except TierError as error:
        raise SystemExit(f"{error}: not submitted") from error

    with Run(cfg) as run:
        if ledger is not None:  # every event from here on names this run
            ledger.source = run.run_id
        inputs: dict[str, Any] = {
            "model": dataclasses.asdict(settings.model),
            "samples": settings.samples,
            "max_tokens": settings.max_tokens,
            "stability_salts": True,  # the requests' keys follow summarizer.stability_salts
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
        if tier == "batch":  # every ADL batch reserved before it is submitted, closed when read
            hook = None
            if ledger is not None:
                hook = LedgeredBatch(
                    ledger, lambda r: settings.model, price_factor=factor, label="summarizer"
                )
                hook.reconcile(cache)
            backend = make_backend(cfg, cache, hook)
        else:
            backend = make_backend(cfg, cache)
            if ledger is not None:  # every ADL request reserved before it is sent
                model = settings.model
                backend = LedgeredBackend(
                    backend, ledger, lambda r: model, price_factor=factor, label="summarizer"
                )
                backend.reconcile(cache)
        completions, cached = fetch(requests, backend, cache)
        # Each consumed response's tier comes with it out of fetch: this run's for a send, its
        # cache entry's (or the manifest's, for that exact entry) for a hit.
        try:
            tiers = response_tiers(completions, manifest)
        except TierError as error:
            raise SystemExit(f"{error}: the responses are cached; rerun with a manifest") from error
        records, texts, failures = [], [], []
        hypothesis_tiers: dict[str, str] = {}  # each hypothesis's tier, by its response id
        request_of: dict[str, str] = {}
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
                hypothesis_tiers[rid] = tiers[request.key]
                request_of[rid] = request.key
        RecordWriter(run.dir).write([*records, *failures])
        accepted = {request_of[rid] for rid in hypothesis_tiers}
        provenance = {
            "hypotheses": hypothesis_tiers,
            "evidence": tier_counts(batches, tiers, accepted),
        }
        run.path("service_tiers.json").write_text(json.dumps(provenance, indent=2) + "\n")
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
            "service_tiers": {
                "hypotheses": dict(sorted(Counter(hypothesis_tiers.values()).items())),
                "responses": dict(sorted(Counter(tiers.values()).items())),
            },
        }
        run.summarize(summary)
        print(
            f"{len(records)} hypotheses, {len(failures)} failures, {len(paid)} paid requests, "
            f"${usd:.4f} by reported usage; written to {run.dir}"
        )


if __name__ == "__main__":
    main()
