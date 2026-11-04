"""Label IA responses with the LLM judge (docs/preregistration.md, Judge).

Reads IA responses and their texts from the run directories in ``judge.runs``, maps each state to
its target behavior through the analyst index ``judge.state_index``, builds the requests and prints
the cost estimate. With ``judge.dry_run=false`` it then sends the requests that are not cached,
escalates uncertain and disputed items when ``judge.escalation_model`` is set, and writes
JudgeLabel and Failure rows into its own run directory, with agreement against manual labels
where they exist. Dry run:

    uv run python scripts/judge.py experiments/judge judge.runs=[runs/ia/<run id>] \\
        judge.state_index=<path>/state_index.jsonl judge.model=gemini-3.5-flash-lite \\
        judge.rubric_version=v1

The endpoint and key come from JUDGE_API_BASE and JUDGE_API_KEY; the model only from the config.
"""

from __future__ import annotations

import atexit
import json
import os
import warnings
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from residual_disposition import stage3_money
from residual_disposition.adl_graders import LedgeredBackend
from residual_disposition.budget import ApiBudgetExhausted, ApiLedger, ledger_from
from residual_disposition.config import from_cli
from residual_disposition.judge import (
    Backend,
    ChatBackend,
    ChatRequest,
    CostEstimate,
    GeminiBatchBackend,
    GeminiFiles,
    ItemRequests,
    JudgeCache,
    JudgeItem,
    JudgeSettings,
    MalformedOutput,
    ModelSpec,
    StageResult,
    agreement,
    collect_candidates,
    escalation_items,
    estimate_cost,
    fetch,
    final_labels,
    parse_lenient,
    parse_strict,
    read_manual_labels,
    read_state_index,
    request_key,
    rubric_set,
    stage_labels,
    stage_requests,
    tier_factor,
    usage_cost,
)
from residual_disposition.paths import data_root
from residual_disposition.records import (
    Failure,
    JudgeLabel,
    RecordWriter,
    ToolHypothesis,
    read_table,
)
from residual_disposition.seeding import derive_seed
from residual_disposition.summarizer import RETRY, base_tool
from residual_disposition.tracking import Run

ADL_TOOLS = {"activation_difference_lens", "adl_unsteered", "adl_lens"}
METER_LOG = Path("ledgers") / "continuation_meter.jsonl"  # under the data root


def model_spec(j: DictConfig, model_id: str) -> ModelSpec:
    for entry in j.models:
        if entry.id == model_id:
            return ModelSpec(
                id=entry.id,
                reasoning_effort=entry.reasoning_effort,
                input_usd_per_mtok=float(entry.input_usd_per_mtok),
                output_usd_per_mtok=float(entry.output_usd_per_mtok),
                expected_thinking_tokens=int(entry.expected_thinking_tokens),
            )
    raise SystemExit(
        f"judge.models has no entry for {model_id!r}: add its thinking level and prices"
    )


def settings(j: DictConfig, model_id: str, lenient_model_id: str | None = None) -> JudgeSettings:
    return JudgeSettings(
        model=model_spec(j, model_id),
        rubrics=rubric_set(j.rubric_version),
        rubric_version=j.rubric_version,
        lenient=bool(j.lenient),
        max_tokens=int(j.max_tokens),
        structured_output=bool(j.structured_output),
        lenient_model=model_spec(j, lenient_model_id) if lenient_model_id else None,
    )


def pricing_tier(j: DictConfig) -> str:
    if j.backend == "batch":
        return "batch"
    if j.backend == "sync":
        return j.sync.service_tier or "standard"
    raise SystemExit(f"judge.backend must be batch or sync, got {j.backend!r}")


def make_backend(j: DictConfig, ledger_dir: Path) -> Backend:
    import openai

    base, key = os.environ.get("JUDGE_API_BASE"), os.environ.get("JUDGE_API_KEY")
    if not base or not key:
        raise SystemExit("Set JUDGE_API_BASE and JUDGE_API_KEY (.env or the Pod environment)")
    # The backends retry themselves, with backoff that suits each of them.
    client = openai.OpenAI(api_key=key, base_url=base, timeout=j.sync.timeout_s, max_retries=0)
    if j.backend == "sync":
        return ChatBackend(
            client,
            concurrency=j.sync.concurrency,
            max_retries=j.sync.max_retries,
            backoff_s=j.sync.backoff_s,
            backoff_max_s=j.sync.backoff_max_s,
            service_tier=j.sync.service_tier,
        )
    return GeminiBatchBackend(
        client,
        GeminiFiles.for_api_base(base, key),
        ledger_dir,
        poll_s=j.batch.poll_s,
        max_wait_s=j.batch.max_wait_h * 3600,
        max_requests_per_job=j.batch.max_requests_per_job,
    )


def estimate(
    j: DictConfig, pairs: list[ItemRequests], stage_settings: JudgeSettings, cache: JudgeCache
) -> CostEstimate:
    """The cost of the requests not yet cached, each rubric priced with its own model."""
    requests = [r for pair in pairs for r in pair.all()]
    _, cached = fetch(requests, None, cache)
    parts = [
        estimate_cost(
            [r for r in requests if r.kind == kind],
            stage_settings.model_for(kind),
            cached=cached,
            chars_per_token=j.cost.chars_per_token,
            expected_output_tokens=j.cost.expected_output_tokens,
            price_factor=tier_factor(
                pricing_tier(j),
                batch_discount=j.cost.batch_discount,
                flex_discount=j.cost.flex_discount,
            ),
        )
        for kind in ("strict", "lenient")
    ]
    return CostEstimate(
        *(sum(getattr(p, f) for p in parts) for f in CostEstimate.__dataclass_fields__)
    )


def describe(stage: str, stage_settings: JudgeSettings, tier: str, est: CostEstimate) -> str:
    models = stage_settings.model.id
    if stage_settings.lenient_model and stage_settings.lenient:
        models += f" (strict) + {stage_settings.lenient_model.id} (lenient)"
    return (
        f"{stage}: {models} ({tier}): {est.requests} requests, {est.requests - est.to_send} "
        f"cached, {est.to_send} to send, ~{est.input_tokens:,} input and ~{est.output_tokens:,} "
        f"output tokens, ~${est.usd:.4f}"
    )


PARSERS = {"strict": parse_strict, "lenient": parse_lenient}


def refused(request: ChatRequest, completions: dict[str, Any]) -> bool:
    """An answer the frozen parser of its rubric refuses, not a failed send."""
    completion = completions[request.key]
    if completion.error is not None:
        return False
    try:
        PARSERS[request.kind](completion.text)
    except MalformedOutput:
        return True
    return False


def retry_refused(
    pairs: list[ItemRequests], completions: dict[str, Any], backend: Backend, cache: JudgeCache,
    cached: set[str],
) -> tuple[list[ItemRequests], list[ChatRequest]]:  # fmt: skip
    """Each refused answer's one retry: the same request under its rubric's key and ``RETRY``,
    fetched (its answer cached as any other), and the pair's request replaced by it; a valid
    answer is never retried. ``completions`` and ``cached`` take the retries' results."""
    again = {
        r.key: ChatRequest(request_key(f"{r.rubric_version}{RETRY}", r.body), r.kind,
                           r.rubric_version, r.body)
        for pair in pairs for r in pair.all() if refused(r, completions)
    }  # fmt: skip
    if not again:
        return pairs, []
    more, hit = fetch(list(again.values()), backend, cache)
    completions.update(more)
    cached |= hit
    out = [
        ItemRequests(p.item, again.get(p.strict.key, p.strict),
                     None if p.lenient is None else again.get(p.lenient.key, p.lenient))
        for p in pairs
    ]  # fmt: skip
    return out, list(again.values())


def run_stage(
    stage: str,
    pairs: list[ItemRequests],
    stage_settings: JudgeSettings,
    *,
    j: DictConfig,
    backend: Backend,
    cache: JudgeCache,
    run: Run,
    state_of: dict[str, str],
    ledger: ApiLedger | None = None,
) -> tuple[StageResult, dict[str, Any]]:
    requests = [r for pair in pairs for r in pair.all()]
    if ledger is not None:  # ADL's hypotheses: every request reserved on the laptop's ledger
        wrapper = (stage3_money.ChainedBackend if isinstance(ledger, stage3_money.ChainedLedger)
                   else LedgeredBackend)  # fmt: skip
        backend = wrapper(
            backend,
            ledger,
            lambda r: stage_settings.model_for(r.kind),
            price_factor=tier_factor(
                pricing_tier(j),
                batch_discount=j.cost.batch_discount,
                flex_discount=j.cost.flex_discount,
            ),
            label=f"judge/{stage}",
            models={m.id: m for m in (stage_settings.model, stage_settings.model_for("lenient"))},
        )
        backend.reconcile(cache)
    completions, cached = fetch(requests, backend, cache)
    retried: list[ChatRequest] = []
    if j.get("retry_refused"):  # each refused answer's one retry (§12.4.5, 2026-10-25)
        pairs, retried = retry_refused(pairs, completions, backend, cache, cached)
        requests += retried
    result = stage_labels(pairs, completions, stage=stage, settings=stage_settings, cached=cached)
    failures = [
        Failure(
            run_id=run.run_id,
            state_id=state_of[item.response_id],
            measurement=f"judge/{stage}",
            reason=f"response {item.response_id} vs {item.target_behavior_id}: {reason}",
        )
        for item, reason in result.failures
    ]
    RecordWriter(run.dir).write([*result.labels, *failures])
    with run.path("judge_requests.jsonl").open("a") as f:
        f.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result.log))
    paid = [
        completions[key]
        for key in {r.key for r in requests} - cached
        if completions[key].error is None
    ]
    factor = tier_factor(
        pricing_tier(j), batch_discount=j.cost.batch_discount, flex_discount=j.cost.flex_discount
    )
    kind_of = {r.key: r.kind for r in requests}
    paid_keys = [k for k in {r.key for r in requests} - cached if completions[k].error is None]
    tokens_in = tokens_out = 0
    usd = 0.0
    for kind in ("strict", "lenient"):
        part = [completions[k] for k in paid_keys if kind_of[k] == kind]
        t_in, t_out, cost = usage_cost(part, stage_settings.model_for(kind), price_factor=factor)
        tokens_in, tokens_out, usd = tokens_in + t_in, tokens_out + t_out, usd + cost
    stats = {
        f"{stage}_labels": len(result.labels),
        f"{stage}_uncertain": sum(label.uncertain for label in result.labels),
        f"{stage}_failures": len(result.failures),
        f"{stage}_paid_requests": len(paid),
        f"{stage}_input_tokens": tokens_in,
        f"{stage}_output_tokens": tokens_out,
        f"{stage}_usd": usd,
        f"{stage}_retried": len(retried),
    }
    print(
        f"{stage}: {len(result.labels)} labels ({stats[f'{stage}_uncertain']} uncertain), "
        f"{len(result.failures)} failures, {len(paid)} paid requests, ${usd:.4f} by reported usage"
    )
    return result, stats


def report_agreement(
    cfg: DictConfig, items: list[JudgeItem], labels: list[JudgeLabel], run: Run
) -> dict[str, Any]:
    j = cfg.judge
    manual_dir = Path(j.agreement.manual_dir or data_root() / "labels" / "manual")
    judged = {(item.response_id, item.target_behavior_id) for item in items}
    manual = [
        label
        for label in (read_manual_labels(manual_dir) if manual_dir.is_dir() else [])
        if (label.response_id, label.target_behavior_id) in judged
    ]
    if not manual:
        print(f"No manual labels for these items in {manual_dir}: agreement not computed")
        return {}
    options = {
        "n_boot": int(j.agreement.bootstrap),
        "level": float(j.agreement.ci),
        "seed": derive_seed(cfg.seed, "judge", "agreement"),
    }
    primary = [label for label in labels if label.stage == "primary"]
    report = {
        "min_kappa": j.min_kappa,
        "primary": agreement(final_labels(primary), manual, **options),
        "final": agreement(final_labels(labels), manual, **options),
    }
    run.path("agreement.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {}
    for which in ("primary", "final"):
        for grader in ("strict", "lenient"):
            part = report[which][grader]
            summary[f"kappa_{grader}_{which}"] = part["kappa"]
            summary[f"kappa_{grader}_{which}_n"] = part["n"]
            print(
                f"κ {grader} ({which}): {part['kappa']:.3f}, {part['ci_level']:.0%} CI "
                f"[{part['ci'][0]:.3f}, {part['ci'][1]:.3f}], n = {part['n']}"
            )
    return summary


def adl_ledger(run_dirs: list[Path], j: DictConfig) -> ApiLedger | None:
    """Judging ADL's hypotheses (the lens-only packages' too, ``adl_lens@<layer>/<package>``)
    spends from the laptop's share of the ADL API cap (``judge.adl_ledger``, the summarizer's
    ledger): only in a run of its own and on the synchronous backend, whose windows are reserved
    before they are sent. Other tools' hypotheses spend from ``judge.api_ledger`` when it is set
    (the continuation, from 2026-10-17), else from no ledger."""
    if j.get("stage3_money"):  # a certified lineage's readers: its base, then its strong part
        if j.backend != "sync":
            raise SystemExit("Judge the readers with judge.backend=sync: the ledger reserves")
        return stage3_money.money(
            data_root(), str(j.stage3_money.behavior), data_root() / METER_LOG
        )[1]
    tools = {base_tool(r.tool) for r in read_table(run_dirs, ToolHypothesis.table)}
    if not tools & ADL_TOOLS:
        other = j.get("api_ledger")
        if not other:
            return None
        if j.backend != "sync":
            raise SystemExit("Judge on judge.api_ledger with judge.backend=sync: it reserves")
        try:
            return ledger_from(OmegaConf.to_container(other), data_root())
        except ValueError as error:
            raise SystemExit(f"Set judge.api_ledger: {error}") from error
    if tools - ADL_TOOLS:
        raise SystemExit("Judge ADL's hypotheses in their own run, on the ADL API ledger")
    if j.backend != "sync":
        raise SystemExit("Judge ADL's hypotheses with judge.backend=sync: the ledger reserves")
    try:
        return ledger_from(OmegaConf.to_container(j.get("adl_ledger") or {}), data_root())
    except ValueError as error:
        raise SystemExit(f"Set judge.adl_ledger: {error}") from error


def main() -> None:
    cfg = from_cli()
    j = cfg.judge
    missing = [
        key
        for key in ("runs", "state_index", "model", "rubric_version")
        if OmegaConf.is_missing(j, key)
    ]
    if missing:
        raise SystemExit(f"Set {', '.join(f'judge.{key}' for key in missing)}")
    env_model = os.environ.get("JUDGE_MODEL")
    if env_model and env_model != j.model:
        warnings.warn(
            f"JUDGE_MODEL={env_model} is ignored; the config sets {j.model}", stacklevel=1
        )
    primary = settings(j, j.model, j.get("lenient_model"))
    escalation = settings(j, j.escalation_model) if j.escalation_model else None
    tier = pricing_tier(j)

    run_dirs = [Path(p) for p in j.runs]
    candidates = collect_candidates(
        run_dirs,
        read_state_index(Path(j.state_index)),
        extra_targets=list(j.extra_targets),
        targets_file=Path(j.targets_file) if j.targets_file else None,
    )
    items = [c.item for c in candidates]
    cache = JudgeCache(Path(j.cache_dir or data_root() / "cache" / "judge"))
    print(f"{len(items)} items (IA response and target) from {len(run_dirs)} runs")
    if j.get("stage3_money"):  # the readers' money pays the lineage its states belong to
        for behavior in sorted({item.target_behavior_id for item in items}):
            stage3_money.check_funding(str(j.stage3_money.behavior), behavior)
    # the readers' money is opened only once nothing more refuses (a dry run journals nothing)
    ledger = None if j.get("stage3_money") else adl_ledger(run_dirs, j)

    primary_pairs = stage_requests(items, primary)
    primary_estimate = estimate(j, primary_pairs, primary, cache)
    print(describe("primary", primary, tier, primary_estimate))
    if escalation:
        worst = estimate(j, stage_requests(items, escalation), escalation, cache)
        print(describe("escalation if every item escalates", escalation, tier, worst))
    if j.dry_run:
        print("Dry run: nothing was sent. Set judge.dry_run=false to submit.")
        return
    if primary_estimate.usd > j.cost.max_usd:
        raise SystemExit(f"Estimate above judge.cost.max_usd = {j.cost.max_usd}: not submitted")
    if j.get("stage3_money"):
        ledger = adl_ledger(run_dirs, j)
        atexit.register(ledger.close)  # an interrupted run's sessions settled at its exit
    if ledger is not None:
        try:
            ledger.check(primary_estimate.usd)
        except ApiBudgetExhausted as error:
            raise SystemExit(f"{error}: not submitted") from error

    with Run(cfg) as run:
        backend = make_backend(j, cache.root / "batches")
        state_of = {c.item.response_id: c.source.state_id for c in candidates}
        targets = {item.target_behavior_id: item.target for item in items}
        inputs = {
            "rubric_version": j.rubric_version,
            "strict_model": primary.model.id,
            "lenient_model": primary.model_for("lenient").id if primary.lenient else None,
            "rubrics": {
                r.version: {"kind": r.kind, "sha256": r.sha256}
                for r in (primary.rubrics.strict, primary.rubrics.lenient)
            },
            "targets": dict(sorted(targets.items())),
            "items": len(items),
        }
        run.path("judge_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
        common = {
            "j": j,
            "backend": backend,
            "cache": cache,
            "run": run,
            "state_of": state_of,
            "ledger": ledger,
        }

        result, summary = run_stage("primary", primary_pairs, primary, **common)
        labels = list(result.labels)
        if escalation:
            flagged = escalation_items(items, result.labels)
            pairs = stage_requests(flagged, escalation)
            escalation_estimate = estimate(j, pairs, escalation, cache)
            print(
                describe(
                    f"escalation of {len(flagged)} items",
                    escalation,
                    tier,
                    escalation_estimate,
                )
            )
            if primary_estimate.usd + escalation_estimate.usd > j.cost.max_usd:
                run.summarize(summary)
                raise SystemExit("Escalation would exceed judge.cost.max_usd: not submitted")
            escalated, stats = run_stage("escalation", pairs, escalation, **common)
            labels += escalated.labels
            summary.update(stats)
        summary.update(report_agreement(cfg, items, labels, run))
        if isinstance(ledger, stage3_money.ChainedLedger):  # each part's session settled
            summary["charged"] = ledger.close()
        run.summarize(summary)
        print(f"Labels written to {run.dir}")


if __name__ == "__main__":
    main()
