"""A pilot behavior's Stage-2 decision, its record and its manifest (preregistration §12.7, §12.4.1,
§12.4.7; execution of 2026-10-19, review s3t), without model code.

The decision (``continuation_rules.stage2_decision``) is made from the records of completed cells
named by run: the screening (``stage2_screening.json``), the confirmation of each attempted
dose and the expression checks at the analysis dose. Each record is checked to be a completed cell
of the right step and state before it counts; a cell that did not complete counts as not done, so
the decision stays pending. The decided record is written as ``manifests/stage2_<behavior>.json``
and committed before the lineage's k* calibration; its acceptance recomputes the decision from the
runs it names (``decision_problems``). The single Stage-5 slot is read from the committed
decisions of the pilot behaviors in their order (``continuation_rules.stage5_slot``).

Records read:

- confirmation: a completed ``stage2_confirmation`` cell whose params name the dose and the latent
  state and whose summary holds ``passed`` (True or False; phase confirm);
- expression, OOD: a completed ``rates_ood`` cell of the latent state (``below_tolerance``);
- expression, ADL corpus: a completed ``adl_corpus`` cell of the latent state whose summary holds
  the hits over the pooled prefixes and the long ones, with their counts (``rules.expression``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import continuation_rules as cr


def _run(find_run: Any, name: str | None) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """(config, summary) of a completed run, None if it is absent or did not complete."""
    import yaml

    run = find_run(name) if name else None
    if run is None or json.loads((run / "meta.json").read_text()).get("status") != "completed":
        return None
    config = yaml.safe_load((run / "config.yaml").read_text()) or {}
    return config, json.loads((run / "summary.json").read_text())


def decision_inputs(
    inputs: Mapping[str, Any], rules: Any, find_run: Any
) -> tuple[dict[int, dict[str, Any]], dict[int, bool | None], dict[str, bool | None], dict]:
    """The screening rows, the confirmations and the expression checks from the runs ``inputs``
    names ({"behavior", "screening", "confirmation": {dose: run}, "expression": {check: run}}),
    each checked against its cell; and the screening record."""
    screening_run = find_run(inputs["screening"])
    if screening_run is None:
        raise ValueError(f"the screening run {inputs['screening']} is not found")
    record = json.loads((screening_run / "stage2_screening.json").read_text())
    if record.get("behavior") != inputs["behavior"] or record.get("rules_sha256") != rules.sha256:
        raise ValueError(f"{inputs['screening']}: not {inputs['behavior']}'s screening")
    doses = {int(d): v for d, v in record["doses"].items()}
    rows = {
        d: {"excess_upper": v["excess"]["upper"], "pool_hits": v["pool"]["hits"]}
        for d, v in doses.items()
    }
    confirmations: dict[int, bool | None] = {}
    for dose, name in (inputs.get("confirmation") or {}).items():
        dose = int(dose)
        found = _run(find_run, name)
        if found is None:
            confirmations[dose] = None  # not done, or technically invalid
            continue
        config, summary = found
        params = config.get("params") or {}
        if (
            config.get("step") != "stage2_confirmation"
            or int(params.get("dose", -1)) != dose
            or params.get("latent") != doses[dose]["latent"]
            or not isinstance(summary.get("passed"), bool)
        ):
            raise ValueError(f"{name}: not the confirmation of {doses[dose]['latent']} at {dose}")
        confirmations[dose] = summary["passed"]
    expression: dict[str, bool | None] = {}
    analysis = _analysis_dose(rows, confirmations, rules)
    expr = rules["expression"]
    for check, name in (inputs.get("expression") or {}).items():
        if check not in cr.EXPRESSION_CHECKS:
            raise ValueError(f"{check} is not an expression check {cr.EXPRESSION_CHECKS}")
        found = _run(find_run, name)
        if found is None:
            expression[check] = None
            continue
        config, summary = found
        params = config.get("params") or {}
        latent = doses[analysis]["latent"] if analysis is not None else None
        step = {"ood": "rates_ood", "adl_corpus": "adl_corpus"}[check]
        if config.get("step") != step or params.get("state") != latent:
            raise ValueError(f"{name}: not the {check} check of {latent}, the analysis dose's")
        if check == "ood":
            if summary.get("n") != int(expr["ood_prompts"]):
                raise ValueError(f"{name}: not over the {expr['ood_prompts']} OOD prompts")
            expression[check] = bool(summary["below_tolerance"])
        else:
            pooled = int(expr["adl_prefixes_per_length"]) * len(expr["adl_prefix_lengths"])
            long = int(expr["adl_long_prefixes"])
            counts = (summary.get("pooled_n"), summary.get("long_n"))
            if counts != (pooled, long):
                raise ValueError(f"{name}: over {counts}, not the {pooled} and {long} prefixes")
            expression[check] = summary["pooled_hits"] == 0 and summary["long_hits"] == 0
    return rows, confirmations, expression, record


def _analysis_dose(rows, confirmations, rules) -> int | None:
    stage2 = rules["stage2"]
    deep = [int(d) for d in stage2["deep_doses"]]
    d = cr.stage2_decision(rows, deep, Fraction(str(stage2["screening_excess"])), confirmations, {})
    return d.get("analysis_dose")


def decide(inputs: Mapping[str, Any], rules: Any, find_run: Any) -> dict[str, Any]:
    """The decision record of one behavior from the runs ``inputs`` names."""
    rows, confirmations, expression, _ = decision_inputs(inputs, rules, find_run)
    stage2 = rules["stage2"]
    deep = [int(d) for d in stage2["deep_doses"]]
    decision = cr.stage2_decision(
        rows, deep, Fraction(str(stage2["screening_excess"])), confirmations, expression
    )
    return {
        "behavior": inputs["behavior"],
        "inputs": json.loads(json.dumps(inputs)),
        "deep_doses": deep,
        "screening_excess": str(stage2["screening_excess"]),
        "decision": json.loads(json.dumps(decision)),
        "rules_sha256": rules.sha256,
    }


def manifest_path(root: Path, behavior: str) -> Path:
    return root / "manifests" / f"stage2_{behavior}.json"


def decision_problems(data: Mapping[str, Any], rules: Any, find_run: Any) -> list[str]:
    """Why a Stage-2 decision record cannot be a manifest: it must be decided under these rules,
    and the decision recomputed from the runs it names must be the recorded one."""
    if data.get("rules_sha256") != rules.sha256:
        return ["the decision was made under other rules"]
    try:
        again = decide(data.get("inputs") or {}, rules, find_run)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return [f"its inputs: {error}"]
    if again["decision"].get("status") != "decided":
        return [
            f"the decision is {again['decision'].get('status')}: {again['decision'].get('next')}"
        ]
    if again["decision"] != data.get("decision"):
        return ["the decision recomputed from its inputs is not the recorded one"]
    return []


def write_manifest(path: Path, data: Mapping[str, Any], rules: Any, find_run: Any) -> dict:
    problems = decision_problems(
        {k: v for k, v in data.items() if k != "decision_run"}, rules, find_run
    )
    if problems:
        raise ValueError(f"no Stage-2 manifest: {'; '.join(problems)}")
    if path.exists():
        raise ValueError(f"{path} exists: a behavior's Stage-2 decision is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return dict(data)


def committed_decision(path: Path, rules: Any, find_run: Any, repo: Path | None = None) -> dict:
    """A behavior's committed Stage-2 decision (``manifests.committed_json``), its decision
    recomputed from the runs it names."""
    from .manifests import committed_json

    data = committed_json(path, "the behavior's Stage-2 decision", repo)
    problems = decision_problems({k: v for k, v in data.items() if k != "sha256"}, rules, find_run)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data


def slot(root: Path, rules: Any, find_run: Any, repo: Path | None = None) -> dict[str, Any]:
    """The Stage-5 slot from the committed decisions of the pilot behaviors, in their order
    (``rules.discovery``); a missing or uncommitted decision is pending."""
    decisions = {}
    for behavior in rules["discovery"]:
        path = manifest_path(root, behavior)
        try:
            decisions[behavior] = committed_decision(path, rules, find_run, repo)["decision"]
        except (OSError, RuntimeError, ValueError):
            decisions[behavior] = None
    return cr.stage5_slot(decisions, list(rules["discovery"]))
