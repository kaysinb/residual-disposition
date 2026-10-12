"""ADL's qualification record (path-B amendment §11.2, §11.4): the frozen verdict that suppression
and every ADL reading of a suppressed state need (``pipeline.qualification_record``).

    uv run python scripts/adl_qualification.py --runs runs/feasibility_14b runs/<summarizer run> \\
        runs/<judge run> --configs experiments/feasibility_14b/example_link \\
        experiments/feasibility_14b/pun_intended [--adl-validation <json>] \\
        [--ensemble-avoids <jsonl>] [--judge-model gemini-3.5-flash-lite] [--rubric-version v4]

The frozen states are each Pod config's ``pipeline.plan.qualify.adl_states``: per behavior its
stage-1 active organism, sham and both recipe controls, the states pass qualify read. ADL's
readings come from the interpreter's hypotheses (``adl-v1``, the frozen summarizer model) and
the judge's labels in ``--runs``, under ADL's own counting rule, from ``--adl-validation`` (the
validation of C on ADL's qualification stratum). PD's eligibility rule is applied per behavior
(``path_b.adl_qualification``); ADL qualifies with at least 6 behaviors.

Writes ``qualification.json`` into the script's run directory and prints its sha256, to set as
``plan.qualification`` {record, sha256}. The record is "pending", and the pipeline allows no
suppression with it, while any frozen state lacks its package, interpretation or grades
(``evidence_gaps``), or without ``--adl-validation``; with complete evidence and a validation it
is "complete", "qualified" or "not qualified" (a stopped counting rule is "not qualified").
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from residual_disposition.config import load_config
from residual_disposition.path_b import (
    ADL_TOOL,
    QUALIFICATION_CONTROLS,
    adl_qualification,
    load_observations,
    pd_readings,
    primary_rule,
)
from residual_disposition.report import run_dirs_from
from residual_disposition.tracking import Run


def frozen_states(configs: list[str]) -> dict[str, dict]:
    """Every behavior's frozen Q1 states, from the Pod configs; a behavior in two is an error."""
    from omegaconf import OmegaConf

    states: dict[str, dict] = {}
    for name in configs:
        cfg = load_config(name)
        mine = OmegaConf.to_container(cfg.pipeline.plan.qualify.adl_states, resolve=True) or {}
        twice = sorted(set(states) & set(mine))
        if twice:
            raise SystemExit(f"{name}: behaviors already listed by another config: {twice}")
        states.update(mine)
    if not states:
        raise SystemExit("no frozen Q1 states in the configs")
    return states


def evidence_gaps(readings, states: dict[str, dict]) -> list[str]:
    """What is missing from the evidence of the frozen states: a state without ADL hypotheses
    (no package or no interpretation), hypotheses that are not one complete summarizer cell, or
    hypotheses without a judge's label. An empty list: every state is fully read."""
    by_state = readings.set_index("state_id")
    gaps = []
    for behavior, mine in sorted(states.items()):
        controls = mine.get("recipe_controls") or {}
        roles = [("active", mine.get("active")), ("sham", mine.get("sham"))]
        roles += [(f"recipe control ({m})", controls.get(m)) for m in QUALIFICATION_CONTROLS]
        for role, sid in roles:
            where = f"{behavior} {role} {sid}"
            if sid not in by_state.index:
                gaps.append(f"{where}: no ADL hypotheses (no package or no interpretation)")
            elif not bool(by_state.at[sid, "valid"]):
                gaps.append(f"{where}: {by_state.at[sid, 'problem']}")
            elif int(by_state.at[sid, "missing"]) > 0:
                gaps.append(f"{where}: {int(by_state.at[sid, 'missing'])} hypotheses ungraded")
    return gaps


def qualification(
    obs, states: dict[str, dict], validation: dict | None, *, need: int = 6
) -> dict[str, object]:
    """The record's body: the verdict on the frozen states under ADL's own validated rule.

    - Incomplete evidence (``evidence_gaps``) is "pending", never a verdict: a missing package,
      interpretation or grade is not an observed failure.
    - Complete evidence without a validation of C is "pending" too.
    - Complete evidence and a validated rule give "qualified" or "not qualified"; a stopped
      rule (C fails its validation for ADL) gives "not qualified".
    A frozen state that is not that behavior's state of its role is an error."""
    summarizer = load_config("experiments/summarize_tools").summarizer
    model, version = str(summarizer.model), str(summarizer.prompts[ADL_TOOL])
    primary = primary_rule(validation)
    hyps = obs.hypotheses
    if "budget" in hyps:
        hyps = hyps[hyps["budget"].isna()]
    out: dict[str, object] = {
        "states": states,
        "validation": validation,
        "primary": primary,
        "need": need,
        "summarizer_model": model,
        "prompt_version": version,
    }
    rule = primary["rule"] or "C"  # a stopped rule: completeness is still checked under C
    readings = pd_readings(
        hyps,
        obs.labels,
        obs.index,
        rule=rule,
        ensemble_avoids=obs.ensemble_avoids,
        hypotheses_per_state=20,
        summarizer_model=model,
        tool=ADL_TOOL,
        prompt_version=version,
    )
    validated = primary["status"] == "validated"
    found = adl_qualification(readings, obs.index, states=states, validated=validated, need=need)
    wrong = [r for r in found["behaviors"] if r["problem"]]
    if wrong:
        raise SystemExit(f"frozen states inconsistent with the index: {wrong[0]['problem']}")
    gaps = evidence_gaps(readings, states)
    out.update(behaviors=found["behaviors"], eligible=found["eligible"], evidence_gaps=gaps)
    if gaps:
        out.update(status="pending", verdict=None, reason="the evidence is incomplete", rule=rule)
    elif primary["status"] == "stop":
        out.update(status="complete", verdict="not qualified", reason=primary["reason"], rule=None)
    elif validated:
        out.update(status="complete", verdict=found["verdict"], rule=rule)
    else:
        out.update(status="pending", verdict=None, so_far=found["verdict"], rule=rule)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--adl-validation", type=Path, default=None)
    parser.add_argument("--ensemble-avoids", type=Path, default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--rubric-version", default=None)
    parser.add_argument("--need", type=int, default=6)
    args = parser.parse_args()
    avoids = None
    if args.ensemble_avoids is not None:
        avoids = {
            row["response_id"]: bool(row["avoids"])
            for row in map(json.loads, args.ensemble_avoids.read_text().splitlines())
        }
    runs = run_dirs_from(args.runs)
    obs = load_observations(
        runs,
        judge_model=args.judge_model,
        rubric_version=args.rubric_version,
        ensemble_avoids=avoids,
    )
    validation = json.loads(args.adl_validation.read_text()) if args.adl_validation else None
    body = qualification(obs, frozen_states(args.configs), validation, need=args.need)
    body["inputs"] = {
        "runs": sorted(str(r) for r in runs),
        "judge_model": args.judge_model,
        "rubric_version": args.rubric_version,
    }
    cfg = load_config("base", ["+experiment=adl_qualification", "tracking.wandb.enabled=false"])
    with Run(cfg) as run:
        path = run.path("qualification.json")
        path.write_text(json.dumps(body, indent=2, sort_keys=True, default=str) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        run.summarize({"status": body["status"], "verdict": body["verdict"], "sha256": digest})
    print(json.dumps({"status": body["status"], "verdict": body["verdict"]}, indent=2))
    print(f"record: {path}\nsha256: {digest}")


if __name__ == "__main__":
    main()
