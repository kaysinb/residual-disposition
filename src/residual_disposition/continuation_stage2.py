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

Records read (review s3v):

- the screening: a completed ``stage2_screening`` cell of the behavior under these rules;
- confirmation: completed ``stage2_confirmation`` cells, in attempt order, whose params name the
  dose and its latent state and whose summary is the attempt's draw (attempt, seed, detector
  version, passed, purpose, the ledger's lineage and base seed), replayed as the attempt ledger
  (``continuation_rules.replay_attempts``); a cell that did not complete or holds no result
  leaves the decision pending (phase confirm, C9);
- expression, OOD: a completed ``rates_ood`` cell of the analysis dose's latent state over the
  200 OOD prompts (``below_tolerance``);
- expression, ADL corpus: a completed ``adl_corpus`` cell of that state whose summary holds, for
  each framing (chat, raw), the hits and counts over the pooled prefixes (600 at each of the
  lengths 1–5) and the long ones (1,000; ``rules.expression``), a pass only with 0 hits in both
  framings (C10).

A check whose content is technically invalid (a missing or non-boolean result, other counts) is
not done: the decision stays pending; a cell of another step or state is refused.
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


FRAMINGS = ("chat", "raw")  # §12.4.7: the ADL corpus per framing


def _screening(inputs: Mapping[str, Any], rules: Any, find_run: Any) -> dict[str, Any]:
    """The screening record of ``inputs["screening"]``: a completed stage2_screening cell of the
    behavior under these rules (review s3v)."""
    found = _run(find_run, inputs.get("screening"))
    if found is None:
        raise ValueError(f"the screening run {inputs.get('screening')} is not a completed cell")
    config, _ = found
    run = find_run(inputs["screening"])
    record = json.loads((run / "stage2_screening.json").read_text())
    if (
        config.get("step") != "stage2_screening"
        or (config.get("params") or {}).get("rules_sha256") != rules.sha256
        or record.get("behavior") != inputs["behavior"]
        or record.get("rules_sha256") != rules.sha256
    ):
        raise ValueError(f"{inputs['screening']}: not {inputs['behavior']}'s screening")
    return record


def _draws(inputs, doses, find_run) -> tuple[list[dict[str, Any]], bool]:
    """The completed confirmation draws in attempt order, and whether every named attempt is
    complete (a cell that did not complete, or holds no result, is not done)."""
    draws, complete = [], True
    for name in inputs.get("confirmation") or []:
        found = _run(find_run, name)
        if found is None:
            complete = False
            continue
        config, summary = found
        params = config.get("params") or {}
        dose = params.get("dose")
        if (
            config.get("step") != "stage2_confirmation"
            or dose not in doses
            or (params.get("latent") != doses[dose]["latent"])
        ):
            raise ValueError(f"{name}: not a confirmation of this screening's latent states")
        keys = ("attempt", "seed", "detector_version", "passed", "purpose", "lineage", "base_seed")
        if not isinstance(summary.get("passed"), bool) or any(k not in summary for k in keys):
            complete = False  # technically invalid: the attempt stays to be done
            continue
        draws.append({"dose": int(dose), **{k: summary[k] for k in keys}})
    if len({(d["lineage"], d["base_seed"]) for d in draws}) > 1:
        raise ValueError("confirmation draws of more than one ledger")
    return draws, complete


def _expression(inputs, latent, rules, find_run) -> dict[str, bool | None]:
    """The expression checks at the analysis dose's latent state: True or False when complete,
    None when not done or technically invalid; a cell of another step or state is refused."""
    expr = rules["expression"]
    pooled = int(expr["adl_prefixes_per_length"]) * len(expr["adl_prefix_lengths"])
    long = int(expr["adl_long_prefixes"])
    out: dict[str, bool | None] = {}
    for check, name in (inputs.get("expression") or {}).items():
        if check not in cr.EXPRESSION_CHECKS:
            raise ValueError(f"{check} is not an expression check {cr.EXPRESSION_CHECKS}")
        found = _run(find_run, name)
        if found is None:
            out[check] = None
            continue
        config, summary = found
        step = {"ood": "rates_ood", "adl_corpus": "adl_corpus"}[check]
        if config.get("step") != step or (config.get("params") or {}).get("state") != latent:
            raise ValueError(f"{name}: not the {check} check of {latent}, the analysis dose's")
        if check == "ood":
            ok = summary.get("n") == int(expr["ood_prompts"]) and isinstance(
                summary.get("below_tolerance"), bool
            )
            out[check] = summary["below_tolerance"] if ok else None
            continue
        framings = summary.get("framings") or {}
        rows = [framings.get(f) or {} for f in FRAMINGS]
        ok = set(framings) == set(FRAMINGS) and all(
            (r.get("pooled_n"), r.get("long_n")) == (pooled, long)
            and type(r.get("pooled_hits")) is int
            and type(r.get("long_hits")) is int
            for r in rows
        )
        out[check] = (
            all(r["pooled_hits"] == 0 and r["long_hits"] == 0 for r in rows) if ok else None
        )
    return out


def decide(inputs: Mapping[str, Any], rules: Any, find_run: Any) -> dict[str, Any]:
    """The decision record of one behavior from the runs ``inputs`` names ({"behavior",
    "detector_version" (the current one), "screening", "confirmation": [runs, in attempt order],
    "expression": {check: run}})."""
    record = _screening(inputs, rules, find_run)
    doses = {int(d): v for d, v in record["doses"].items()}
    rows = {
        d: {"excess_upper": v["excess"]["upper"], "pool_hits": v["pool"]["hits"]}
        for d, v in doses.items()
    }
    stage2 = rules["stage2"]
    deep = [int(d) for d in stage2["deep_doses"]]
    margin = Fraction(str(stage2["screening_excess"]))
    draws, complete = _draws(inputs, doses, find_run)
    ledger = {"detector_version": str(inputs["detector_version"]),
              "lineage": draws[0]["lineage"] if draws else inputs["behavior"],
              "base_seed": int(draws[0]["base_seed"]) if draws else 0}  # fmt: skip
    first = cr.stage2_decision(rows, deep, margin, draws, {}, **ledger)
    if not complete:
        decision = {**first, "status": cr.PENDING, "next": "a confirmation that did not complete"}
        decision.pop("exception_a", None)
    else:
        latent = doses[first["analysis_dose"]]["latent"] if first.get("analysis_dose") else None
        expression = _expression(inputs, latent, rules, find_run) if latent else {}
        decision = cr.stage2_decision(rows, deep, margin, draws, expression, **ledger)
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


# what a record's manifest adds (its run) and what its check adds (manifests.TRANSPORT)
METADATA = ("decision_run", "sha256", "committed_at")


def current_detector_version(behavior: str) -> str:
    """The behavior's detector version as its config now defines it (review s3x2)."""
    from omegaconf import OmegaConf

    from .behaviors import behavior_from_config
    from .config import load_config

    spec = OmegaConf.to_container(load_config(f"behaviors/{behavior}").behavior)
    return behavior_from_config(spec).detector.version


def decision_problems(
    data: Mapping[str, Any], rules: Any, find_run: Any, path: Path | None = None
) -> list[str]:
    """Why a Stage-2 decision record cannot be a manifest (reviews s3v, s3x2): it must be decided
    under these rules with the behavior's current detector (its config's version, so a revision
    after the decision refuses it), the whole record recomputed from the runs it names
    (``decide``) must be it but for its metadata (``METADATA``), and at ``path`` its file must be
    its behavior's."""
    if (
        path is not None
        and Path(path).name != manifest_path(Path("."), str(data.get("behavior"))).name
    ):
        return [f"{Path(path).name} is not the manifest of {data.get('behavior')}"]
    if data.get("rules_sha256") != rules.sha256:
        return ["the decision was made under other rules"]
    inputs = data.get("inputs") or {}
    try:
        current = current_detector_version(str(data.get("behavior")))
    except (OSError, ValueError, KeyError) as error:
        return [f"the behavior's detector: {error}"]
    if inputs.get("detector_version") != current:
        # a revision after the decision voids its draws: the decision is remade, not reused
        return [f"the decision was made with detector {inputs.get('detector_version')}, not the "
                f"current {current}"]  # fmt: skip
    try:
        again = decide(inputs, rules, find_run)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return [f"its inputs: {error}"]
    if again["decision"].get("status") != "decided":
        return [
            f"the decision is {again['decision'].get('status')}: {again['decision'].get('next')}"
        ]
    if again != {k: v for k, v in data.items() if k not in METADATA}:
        return ["the record recomputed from its inputs is not this one"]
    return []


def write_manifest(path: Path, data: Mapping[str, Any], rules: Any, find_run: Any) -> dict:
    problems = decision_problems(data, rules, find_run, path)
    if problems:
        raise ValueError(f"no Stage-2 manifest: {'; '.join(problems)}")
    if path.exists():
        raise ValueError(f"{path} exists: a behavior's Stage-2 decision is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return dict(data)


def committed_decision(path: Path, rules: Any, find_run: Any, repo: Path | None = None) -> dict:
    """A behavior's committed Stage-2 decision (``manifests.committed_json``), the whole record
    recomputed from the runs it names and its file its behavior's."""
    from .manifests import committed_json

    data = committed_json(path, "the behavior's Stage-2 decision", repo)
    problems = decision_problems(data, rules, find_run, path)
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
