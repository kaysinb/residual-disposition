"""Exception a's continued-mode lineage up to its dose decision (preregistration §12.7 Stage 5,
§12.4.1, §12.4.6, §12.4.7; review s4r; C12), without model code: the records of its completed
cells, the decision made from them and its manifest.

The lineage is ``stage5/<behavior>`` (``continuation_confirm.stage5_lineage``): one implant (A and
H of Stage 1), continued-mode SFT on the pilot's own suppression data, checkpoints 25–800. The
decision (``continuation_rules.stage5_decision``) is made from the records of completed cells named
by run, each checked to be a completed cell of the right step and state before it counts; a cell
that did not complete counts as not done, so the decision stays pending:

- the selection (``stage5_selection.json``): the lineage's states at every checkpoint (L, H_s, the
  recipe control), their exploratory paired-excess bounds re-scored with the current detector,
  the implant, method, mode and seed, the pinned data, under these rules;
- confirmation: completed ``stage5_confirmation`` cells in attempt order (the draws of
  ``continuation_confirm``'s frozen prompts and base seed, on the Stage-5 ledger), each pass
  recomputed from L's recorded hits (``continuation_stage2._draws``);
- X_cont: a completed ``xcont`` cell of the analysis dose's latent state (``valid`` true or false);
- expression, at the analysis dose's latent state: the PD corpus, its full PD pool (a completed
  ``tool_pd`` cell of the frozen settings, every prefill in both framings once, recounted here
  with the current detector; 0 hits in both framings to pass), the ADL corpus and the OOD observer
  as Stage 2 reads them.

A decided record becomes ``manifests/stage5_<behavior>.json`` (``scripts/continuation_stage5.py``),
committed before the lineage's k*; its acceptance recomputes it from the runs it names.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import continuation_confirm as C
from . import continuation_rules as cr
from . import continuation_stage2 as S2

SELECTION, CONFIRMATION, XCONT = "stage5_selection", "stage5_confirmation", "xcont"
ROLES = ("latent", "sham_s", "recipe")  # each checkpoint's states in the selection
FRAMINGS = ("chat", "raw")


def _selection(inputs: Mapping[str, Any], rules: Any, find_run: Any) -> dict[str, Any]:
    """The selection record of ``inputs["selection"]``: a completed stage5_selection cell of
    the behavior's Stage-5 lineage under these rules, a state of each role at every checkpoint."""
    found = S2._run(find_run, inputs.get("selection"))
    if found is None:
        raise ValueError(f"the selection run {inputs.get('selection')} is not a completed cell")
    config, _ = found
    record = json.loads((find_run(inputs["selection"]) / "stage5_selection.json").read_text())
    checkpoints = [int(d) for d in rules["observer"]["checkpoints"]]
    doses = record.get("doses") or {}
    if (
        config.get("step") != SELECTION
        or (config.get("params") or {}).get("rules_sha256") != rules.sha256
        or record.get("behavior") != inputs["behavior"]
        or record.get("lineage") != C.stage5_lineage(inputs["behavior"])
        or record.get("rules_sha256") != rules.sha256
        or sorted(int(d) for d in doses) != checkpoints
        or any(not (doses[str(d)].get(r)) for d in checkpoints for r in ROLES)
    ):
        raise ValueError(f"{inputs['selection']}: not {inputs['behavior']}'s Stage-5 selection")
    return record


def _xcont(inputs: Mapping[str, Any], latent: str, find_run: Any) -> bool | None:
    """X_cont of the analysis dose: True or False from a completed xcont cell of ``latent``,
    None when it is not built (no run named, or one that did not complete)."""
    found = S2._run(find_run, inputs.get("xcont"))
    if found is None:
        return None
    config, summary = found
    if config.get("step") != XCONT or (config.get("params") or {}).get("latent") != latent:
        raise ValueError(f"{inputs['xcont']}: not X_cont of {latent}, the analysis dose's")
    valid = summary.get("valid")
    return valid if isinstance(valid, bool) else None


def pd_corpus(run: Path, latent: str, detector: Any) -> dict[str, Any] | None:
    """The PD corpus of ``latent`` (§12.4.7) from a completed tool_pd cell: its frozen prefills,
    exactly one completion per framing and prefill, recounted with ``detector``; None when the
    cell is technically invalid (incomplete). A cell of another step, state or draw is refused."""
    import yaml

    from .pipeline import FROZEN_PD, PD_CONFIGS, read_jsonl_gz

    config = yaml.safe_load((run / "config.yaml").read_text()) or {}
    params = config.get("params") or {}
    if config.get("step") != "tool_pd" or params.get("state") != latent or params.get("draw"):
        raise ValueError(f"{run.name}: not the PD pool of {latent}, the analysis dose's")
    summary = json.loads((run / "summary.json").read_text())
    n = FROZEN_PD["prefills"]
    if (summary.get("prefills"), summary.get("prefills_sha256")) != (
        n,
        FROZEN_PD["prefills_sha256"],
    ):
        return None
    keys = {f"p{i:05d}" for i in range(n)}
    seen: dict[str, set[str]] = {c: set() for c in PD_CONFIGS}
    hits = dict.fromkeys(PD_CONFIGS, 0)
    for row in read_jsonl_gz(run / "pd_completions.jsonl.gz"):
        framing, key = row["config"], row["prefill"]
        if framing not in seen or key not in keys or key in seen[framing]:
            return None
        seen[framing].add(key)
        hits[framing] += bool(detector(row["completion"]))
    if any(len(s) != n for s in seen.values()):
        return None
    return {"run": run.name, "by_config": {c: {"pool": n, "hits": hits[c]} for c in PD_CONFIGS},
            "hits": sum(hits.values()), "pool": n * len(PD_CONFIGS)}  # fmt: skip


def _expression(inputs, latent, rules, find_run) -> tuple[dict[str, bool | None], dict]:
    """The expression checks at the analysis dose's latent state, True or False when complete,
    None when not done; and the PD corpus's counts."""
    named = dict(inputs.get("expression") or {})
    unknown = set(named) - set(cr.STAGE5_EXPRESSION_CHECKS)
    if unknown:
        raise ValueError(f"{sorted(unknown)}: not Stage-5 expression checks")
    out = S2._expression(
        {"expression": {k: v for k, v in named.items() if k != "pd_corpus"}},
        latent,
        rules,
        find_run,
    )
    counts: dict[str, Any] = {}
    out["pd_corpus"] = None
    found = S2._run(find_run, named.get("pd_corpus"))
    if found is not None:
        detector = _detector(inputs["behavior"])
        counts = pd_corpus(find_run(named["pd_corpus"]), latent, detector) or {}
        if counts:
            out["pd_corpus"] = all(v["hits"] == 0 for v in counts["by_config"].values())
    return out, counts


def _detector(behavior: str) -> Any:
    from omegaconf import OmegaConf

    from .behaviors import behavior_from_config
    from .config import load_config

    spec = OmegaConf.to_container(load_config(f"behaviors/{behavior}").behavior)
    return behavior_from_config(spec).detector


def decide(inputs: Mapping[str, Any], rules: Any, find_run: Any) -> dict[str, Any]:
    """The decision record of a Stage-5 lineage from the runs ``inputs`` names ({"behavior",
    "detector_version" (the current one), "selection", "confirmation": [runs, in attempt order],
    "xcont": run or None, "expression": {check: run}})."""
    record = _selection(inputs, rules, find_run)
    doses = {int(d): v for d, v in record["doses"].items()}
    upper = {d: Fraction(str(v["excess"]["upper"])) for d, v in doses.items()}
    observer = rules["observer"]
    checkpoints = [int(d) for d in observer["checkpoints"]]
    margin = Fraction(str(observer["margin"]))
    name = C.stage5_lineage(inputs["behavior"])
    draws, complete = S2._draws(inputs, doses, rules, find_run, step=CONFIRMATION, name=name)
    ledger = {"detector_version": str(inputs["detector_version"]), "lineage": name,
              "base_seed": C.FROZEN["base_seed"]}  # fmt: skip
    decision = cr.stage5_decision(upper, checkpoints, margin, draws, None, {}, **ledger)
    pd_counts: dict[str, Any] = {}
    if not complete:
        decision = {
            **decision,
            "status": cr.PENDING,
            "next": "a confirmation that did not complete",
        }
        decision.pop("outcome", None)
    elif decision.get("analysis_dose") is not None:
        latent = doses[decision["analysis_dose"]]["latent"]
        xcont = _xcont(inputs, latent, find_run)
        expression = {}
        if xcont:
            expression, pd_counts = _expression(inputs, latent, rules, find_run)
        decision = cr.stage5_decision(
            upper, checkpoints, margin, draws, xcont, expression, **ledger
        )
    return {
        "behavior": inputs["behavior"],
        "lineage": name,
        "implant": record["implant"],
        "method": record["method"],
        "mode": record["mode"],
        "seed": record["seed"],
        "inputs": json.loads(json.dumps(inputs)),
        "checkpoints": checkpoints,
        "margin": str(observer["margin"]),
        "decision": json.loads(json.dumps(decision)),
        "pd_corpus": pd_counts,
        "rules_sha256": rules.sha256,
    }


def cells(runs: Path, experiment: str, step: str) -> list[tuple[Path, dict, dict]]:
    """The completed cells of ``step`` in ``experiment`` under ``runs``: (run, config, summary)."""
    import yaml

    out = []
    for config in sorted((Path(runs) / experiment).glob(f"*-{step}-*/config.yaml")):
        run = config.parent
        if json.loads((run / "meta.json").read_text()).get("status") == "completed":
            summary = json.loads((run / "summary.json").read_text())
            out.append((run, yaml.safe_load(config.read_text()) or {}, summary))
    return out


def find_inputs(runs: Path, experiment: str, behavior: str, rules: Any, find_run: Any) -> dict:
    """The decision's inputs as the space holds them: its one completed selection of the
    behavior's lineage, the lineage's confirmation cells in attempt order, and at the analysis
    dose's latent state its X_cont and expression cells. Two cells of one kind are refused."""
    name = C.stage5_lineage(behavior)
    selections = [r for r, c, s in cells(runs, experiment, SELECTION)
                  if (c.get("params") or {}).get("lineage") == name]  # fmt: skip
    if len(selections) != 1:
        raise ValueError(f"{len(selections)} completed selections of {name}, not one")
    draws = sorted(
        (s.get("attempt", 0), r.name)
        for r, c, s in cells(runs, experiment, CONFIRMATION)
        if (c.get("params") or {}).get("lineage") == name
    )
    if len({a for a, _ in draws}) != len(draws):
        raise ValueError("two confirmation cells of one attempt")
    inputs = {"behavior": behavior, "detector_version": S2.current_detector_version(behavior),
              "selection": selections[0].name, "confirmation": [n for _, n in draws],
              "xcont": None, "expression": {}}  # fmt: skip
    decision = decide(inputs, rules, find_run)["decision"]
    if decision.get("analysis_dose") is None:
        return inputs
    record = _selection(inputs, rules, find_run)
    latent = record["doses"][str(decision["analysis_dose"])]["latent"]

    def one(step: str, what: str, keep) -> str | None:
        found = [r.name for r, c, _ in cells(runs, experiment, step) if keep(c.get("params") or {})]
        if len(found) > 1:
            raise ValueError(f"{len(found)} completed {what} of {latent}")
        return found[0] if found else None

    inputs["xcont"] = one(XCONT, "X_cont cells", lambda p: p.get("latent") == latent)
    for check, step in (
        ("pd_corpus", "tool_pd"),
        ("adl_corpus", "adl_corpus"),
        ("ood", "rates_ood"),
    ):
        run = one(step, f"{check} checks", lambda p: p.get("state") == latent and not p.get("draw"))
        if run:
            inputs["expression"][check] = run
    return inputs


def next_step(decision: Mapping[str, Any]) -> tuple[str, int | None]:
    """What the chain runs next for a decision: ("decide" | "confirm" | "retest" | "xcont" |
    "expression" | "selection", its dose)."""
    nxt = str(decision.get("next") or "")
    if decision["status"] == "decided":
        return "decide", decision.get("analysis_dose")
    for prefix, step in (("confirmation retest at", "retest"), ("confirmation at", "confirm"),
                         ("X_cont at", "xcont")):  # fmt: skip
        if nxt.startswith(prefix):
            return step, int(nxt.rsplit(" ", 1)[1])
    if nxt.startswith("expression checks"):
        return "expression", int(nxt.rsplit(" ", 1)[1])
    return "selection", None


def manifest_path(root: Path, behavior: str) -> Path:
    return root / "manifests" / f"stage5_{behavior}.json"


METADATA = ("decision_run", "sha256", "committed_at")


def decision_problems(
    data: Mapping[str, Any], rules: Any, find_run: Any, path: Path | None = None
) -> list[str]:
    """Why a Stage-5 decision record cannot be a manifest: decided under these rules with the
    behavior's current detector, the whole record recomputed from the runs it names (``decide``)
    equal to it but for its metadata, and at ``path`` its behavior's file."""
    behavior = str(data.get("behavior"))
    if path is not None and Path(path).name != manifest_path(Path("."), behavior).name:
        return [f"{Path(path).name} is not the Stage-5 manifest of {behavior}"]
    if data.get("rules_sha256") != rules.sha256:
        return ["the decision was made under other rules"]
    inputs = data.get("inputs") or {}
    try:
        current = S2.current_detector_version(behavior)
    except (OSError, ValueError, KeyError) as error:
        return [f"the behavior's detector: {error}"]
    if inputs.get("detector_version") != current:
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
        raise ValueError(f"no Stage-5 manifest: {'; '.join(problems)}")
    if path.exists():
        raise ValueError(f"{path} exists: a lineage's Stage-5 decision is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return dict(data)


def committed_decision(path: Path, rules: Any, find_run: Any, repo: Path | None = None) -> dict:
    """A lineage's committed Stage-5 decision (``manifests.committed_json``), recomputed from the
    runs it names."""
    from .manifests import committed_json

    data = committed_json(path, "the lineage's Stage-5 decision", repo)
    problems = decision_problems(data, rules, find_run, path)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data
