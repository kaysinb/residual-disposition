"""Exception a's continued-mode lineage up to its dose decision (preregistration §12.7 Stage 5,
§12.4.1, §12.4.6, §12.4.7; reviews s4r, s4s; C12): the records of its completed cells, the
decision made from them and its manifest. Nothing here trusts a recorded verdict: each is derived
again from the evidence the cells hold.

The lineage is ``stage5/<behavior>`` (``continuation_confirm.stage5_lineage``): one implant (A and
H of Stage 1), continued-mode SFT on the pilot's own suppression data, checkpoints 25–800, under
one validated plan (``pipeline.stage5_plan``: implant, seed, method, mode, model, the frozen
recipe, LoRA and exploratory spec, the pinned data, the rules; hashed). The decision
(``continuation_rules.stage5_decision``) reads:

- **the selection** (``stage5_selection``): its plan the frozen one and hashing to its sha256;
  its training provenance (the ``stage5_training`` attempt cell, outcome ok, and its three
  training cells: parent, kind, seed, mode, method, checkpoints, data, recipe, LoRA and label,
  their states the record's); every exploratory pass (step, state, measurement, prompts, frozen
  generation spec, 600 responses) replayed from its saved responses with the current detector,
  so a detector revision rescores without a new draw (``pipeline.excess_from_passes``);
- **or a technical infeasibility** of the training (``stage5_infeasibility``): its plan's two
  training attempts, each completed with a classified failure (out of memory, divergence);
- **confirmation**: completed ``stage5_confirmation`` cells in attempt order, on the Stage-5
  ledger, each pass recomputed from L's recorded hits (``continuation_stage2._draws``), each
  allocation replayed against the candidates of the detector its draw was read with (its cell's,
  the exploratory passes rescored with it; ``candidates_of``, review s4v);
- **X_cont**: the ``xcont`` cell of the analysis dose (its identity the frozen prompts,
  tolerances and check) and its attempts (``xcont_attempt``), each verdict recomputed from its
  evidence (``xcont.verdict``), at most one retry and only after an invalid attempt, and the
  valid attempt's X_cont state checked against its manifest and the hashes of A, A^cont and H;
- **expression**, at the analysis dose's latent state, each from a cell of the frozen settings:
  the PD corpus (the full pool, on the final PD GPU, recounted with the current detector; 0 hits
  in both framings for the descriptive label), the ADL corpus (the frozen settings, its saved
  continuations rescored with the current detector) and the OOD observer (the frozen prompts and
  decoding, rescored with the current detector).

Every source cell is of the plan's model, and the plan's model, implant pair, seed and pinned data
are the behavior's committed experiment config's (review s4u 4). A detector revision rescores the
saved responses and draws nothing again: the plan's hash, its training and its exploratory passes
stand (review s4u 3).

A decided record becomes ``manifests/stage5_<behavior>.json`` (``scripts/continuation_stage5.py``),
committed before the lineage's k*; its acceptance recomputes it from the runs it names.

The descriptive DPO lineage beside it (``stage5_dpo/<behavior>``; the user, 2026-10-20; C15,
review s5b) is read the same way under its own plan (``pipeline.S5_DPO``: DPO in stack mode, β
0.1, on the frozen preference pairs, L and H_s only): its cells are ``stage5_dpo_<step>``, its
inputs name their lineage (``"lineage"``), it has no X_cont (the dose is decided by the margin
rule, the confirmation and the expression checks) and its manifest is
``manifests/stage5_dpo_<behavior>.json``. The continued lineage's records are unchanged.

Stage 3's stack lineage of a panel behavior (``stage3/<behavior>``; §12.7 Stage 3, C17) is read
the same way under its own plan (``pipeline.S3_STACK``: stack-mode SFT, L, H_s and the recipe
control, ``plan.stage3`` of its config, its suppression data pinned by id alone, built from its
own data's spec): its cells are ``stage3_<step>``, its inputs name their lineage and A's
relearning reference (E5: the one completed pass of the frozen spec, ``REFERENCE_SPEC``, recounted
and recorded in the decision), it has no X_cont, and its manifest is
``manifests/stage3_<behavior>.json``.
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

SELECTION, CONFIRMATION = "stage5_selection", "stage5_confirmation"
TRAINING, INFEASIBILITY = "stage5_training", "stage5_infeasibility"
XCONT, XCONT_ATTEMPT = "xcont", "xcont_attempt"
FAILURES = ("out of memory", "divergence")  # the classified training failures (§Stage 5)
FRAMINGS = ("chat", "raw")


def _p():
    from . import pipeline

    return pipeline


def _line(inputs: Mapping[str, Any]) -> Any:
    """The Stage-5 line the decision's inputs name (``"lineage"``; the continued one without)."""
    p = _p()
    line = p.s5_line(inputs.get("lineage"))
    if inputs.get("lineage") not in (None, line.lineage(inputs["behavior"])) or (
        line is p.S5_CONTINUED and "lineage" in inputs
    ):
        raise ValueError(f"{inputs.get('lineage')}: not a Stage-5 lineage of {inputs['behavior']}")
    return line


def _line_of(plan: Mapping[str, Any]) -> Any:
    return _p().s5_line(plan.get("lineage"))


def _label(line: Any) -> str:
    return line.label


def _config(find_run: Any, name: str | None, step: str) -> tuple[Path, dict, dict] | None:
    """(run, config, summary) of a completed cell of ``step``, None if absent or not completed;
    a completed run of another step is refused."""
    found = S2._run(find_run, name)
    if found is None:
        return None
    config, summary = found
    if config.get("step") != step:
        raise ValueError(f"{name}: a {config.get('step')} cell, not {step}")
    return find_run(name), config, summary


PLAN_KEYS = ("lineage", "behavior", "implant", "seed", "method", "mode", "model", "checkpoints",
             "recipe", "lora", "exploratory", "data", "rules_sha256")  # fmt: skip
DPO_PLAN_KEYS = ("scope", "pairs")  # the DPO lineage's alone (C15)


def _plan(
    plan: Mapping[str, Any], behavior: str, rules: Any, line: Any = None, history: bool = False
) -> dict[str, Any]:
    """A recorded Stage-5 plan, checked: hashing to its sha256, the behavior's lineage of
    ``line`` (default the one the plan names), its method and mode (continued-mode SFT, or the
    DPO lineage's stack-mode DPO with its descriptive scope and frozen pairs), the frozen recipe,
    LoRA and exploratory spec, the rules' checkpoints, these rules, and no other key. A Stage-3
    plan is of the committed config's current repair generation (``_repair_frozen``), or with
    ``history`` of a superseded one, verified against its committed entry, never for a
    decision."""
    p = _p()
    line = line or _line_of(plan)
    body = {k: v for k, v in plan.items() if k != "sha256"}
    body_pins = {r: {"id": d.get("id")} if r in line.built else
                 {"id": d.get("id"), "examples": d.get("examples")}
                 for r, d in (body.get("data") or {}).items()}  # fmt: skip
    body = {**body, "data_pins": body_pins}
    frozen = {"lineage": line.lineage(behavior), "behavior": behavior, "method": line.method,
              "mode": line.mode, "recipe": line.recipe, "lora": p.S5_LORA,
              "exploratory": p.S5_EXPLORATORY, "rules_sha256": rules.sha256,
              "checkpoints": list(line.checkpoints)
              or [int(d) for d in rules["observer"]["checkpoints"]]}  # fmt: skip
    if line is p.S5_DPO:
        frozen.update(scope="descriptive", pairs=p.S5D_PAIRS)
    if line.spec == "stage3":
        frozen.update(generation=p.S3_GENERATION)
    frozen.update(_approved(behavior, line))
    extra = DPO_PLAN_KEYS if line is p.S5_DPO else ("generation",) if line.spec == "stage3" else ()
    if line.spec == "stage3":  # its repair generation (§12.7 Stage 3, the repair)
        repair, suppress = _repair_frozen(plan, behavior, history)
        frozen.update(repair)
        extra += tuple(repair)
        if suppress is not None:  # a superseded generation's own suppression pin
            frozen["data_pins"] = {**frozen["data_pins"], "suppress": suppress}
    elif history:
        raise ValueError(f"only a Stage-3 plan is read as history, not {_label(line)}'s")
    wrong = sorted(k for k, v in frozen.items() if body.get(k) != v)
    keys = (*PLAN_KEYS, *extra, "data_pins")
    wrong += sorted(set(body) - set(keys))
    hashed = {k: v for k, v in body.items() if k != "data_pins"}
    if p.content_hash(hashed) != plan.get("sha256") or wrong:
        raise ValueError(f"not the frozen {_label(line)} plan of {behavior}: {wrong or 'its hash'}")
    return dict(plan)


def _repair(behavior: str) -> dict[str, Any]:
    """The behavior's committed repair generation of its Stage-3 lineage (§12.7 Stage 3, the
    repair): ``plan.stage3``'s ``repair_generation`` and ``superseded`` in its committed
    experiment config, checked (``pipeline.stage3_repair_generation``); empty for generation 1
    (neither key)."""
    from omegaconf import OmegaConf

    from .config import load_config

    cfg = load_config(f"experiments/continuation_14b/{behavior}")
    spec = OmegaConf.select(cfg, "pipeline.plan.stage3", default=None)
    spec = OmegaConf.to_container(spec, resolve=True) if spec is not None else {}
    if _p().stage3_repair_generation(spec, behavior) == 1:
        return {}
    out = {"repair_generation": spec["repair_generation"], "superseded": spec["superseded"]}
    relearn = (spec.get("data") or {}).get("relearn")
    return {**out, "relearn_data": relearn} if relearn is not None else out


def repair_generation(behavior: str) -> int:
    """The current repair generation of the behavior's Stage-3 lineage (1 without a repair)."""
    return int(_repair(behavior).get("repair_generation", 1))


def _repair_frozen(
    plan: Mapping[str, Any], behavior: str, history: bool
) -> tuple[dict[str, Any], dict | None]:
    """A recorded Stage-3 plan's repair fields as the committed config freezes them, and the
    suppression pin of a superseded generation (else None). The current generation's plan holds
    the config's ``repair_generation`` and ``superseded`` (generation 1's neither). With
    ``history`` the plan is of a superseded generation the config declares: its sha256 is its
    entry's ``plan_sha256``, its suppression data the entry's, and its own repair fields those
    of its generation (the entries before it); a generation not declared refuses."""
    repair = _repair(behavior)
    current = int(repair.get("repair_generation", 1))
    recorded = plan.get("repair_generation", 1)
    if not history:
        if recorded != current:
            raise ValueError(f"a plan of repair generation {recorded}, not {behavior}'s current "
                             f"{current}: history only")  # fmt: skip
        return repair, None
    entries = [e for e in repair.get("superseded") or [] if e["repair_generation"] == recorded]
    if type(recorded) is not int or recorded >= current or len(entries) != 1:
        raise ValueError(f"no superseded repair generation {recorded} of {behavior} is committed: "
                         "not its history")  # fmt: skip
    (entry,) = entries
    if plan.get("sha256") != entry["plan_sha256"]:
        raise ValueError(f"the plan {str(plan.get('sha256'))[:12]} is not superseded generation "
                         f"{recorded}'s {entry['plan_sha256'][:12]}")  # fmt: skip
    own = {} if recorded == 1 else {"repair_generation": recorded, "superseded": [
        e for e in repair["superseded"] if e["repair_generation"] < recorded]}  # fmt: skip
    return own, {"id": entry["suppress"]["id"]}


def _approved(behavior: str, line: Any = None) -> dict[str, Any]:
    """The approved specification of the behavior's lineage in its committed experiment config
    (review s4u 4): the model, the implant pair, the seed and the pinned data (``plan.<spec>``
    of ``line``; a dataset the line builds by its id alone; the trained-on roles alone, not a
    repair generation's relearning pin)."""
    from omegaconf import OmegaConf

    from .config import load_config

    p = _p()
    line = line or p.S5_CONTINUED
    cfg = load_config(f"experiments/continuation_14b/{behavior}")
    spec = OmegaConf.to_container(cfg.pipeline.plan[line.spec], resolve=True)
    model = OmegaConf.to_container(cfg.model, resolve=True)
    data = {role: {"id": pin["id"]} if role in line.built else
            {"id": pin["id"], "examples": pin["examples"]}
            for role, pin in spec["data"].items() if role in p.S5_DATA}  # fmt: skip
    keys = ("name", "revision", "dtype")
    if line.spec == "stage3":  # Stage 3's plans hold the model whole (review s5t 2)
        keys += ("chat_template_kwargs",)
    return {"model": {k: model[k] for k in keys},
            "implant": {"active": spec["active"], "sham": spec["sham"]}, "seed": spec["seed"],
            "data_pins": data}  # fmt: skip


def _model(config: Mapping[str, Any], plan: Mapping[str, Any], name: str) -> None:
    """A source cell of the plan's model (review s4u 4), refused otherwise."""
    model = config.get("model") or {}
    if {k: model.get(k) for k in plan["model"]} != plan["model"]:
        raise ValueError(f"{name}: a cell of model {model.get('name')}@{model.get('revision')}, "
                         f"not the plan's")  # fmt: skip


def _training(name: str, plan: Mapping[str, Any], find_run: Any) -> dict[str, Any]:
    """A completed training attempt of ``plan`` (its params and summary)."""
    found = _config(find_run, name, _line_of(plan).step("training"))
    if found is None:
        raise ValueError(f"{name}: not a completed training attempt")
    _, config, summary = found
    params = config.get("params") or {}
    _model(config, plan, name)
    if params.get("plan_sha256") != plan["sha256"] or summary.get("plan") != plan:
        raise ValueError(f"{name}: a training attempt of another plan")
    if summary.get("outcome") not in ("ok", *FAILURES) or params.get("attempt") not in (1, 2):
        raise ValueError(f"{name}: not a classified attempt {summary.get('outcome')}")
    return {"attempt": params["attempt"], **summary}


def _training_run(name: str, role: str, attempt: int, plan: Mapping, states: Mapping, find_run):
    """One training cell of the lineage, as the plan and its attempt make it (the DPO lineage's
    on the frozen preference pairs)."""
    p = _p()
    line = _line_of(plan)
    found = _config(find_run, name, f"train_{line.roles[role][0]}")
    if found is None:
        raise ValueError(f"{name}: not a completed training cell")
    _, config, summary = found
    _model(config, plan, name)
    kind, parent, data = line.roles[role]
    q = config.get("params") or {}
    rows = plan["pairs"] if data == "pairs" else {"sha256": plan["data"][data]["examples"]}
    want = {"parent": plan["implant"][parent], "kind": kind, "seed": plan["seed"],
            "mode": line.mode, "method": line.method, "doses": plan["checkpoints"],
            "data": rows["sha256"], "recipe": line.recipe, "lora": p.S5_LORA,
            "label": p._attempt_label(attempt, line), "strength": {}}  # fmt: skip
    if data == "pairs":
        want["n"] = rows["n"]
    wrong = sorted(k for k, v in want.items() if q.get(k) != v)
    got = {int(d): s for d, s in (summary.get("states") or {}).items()}
    if wrong or got != {int(d): s for d, s in states.items()}:
        raise ValueError(f"{name}: not the {role}'s training of this plan: {wrong or 'states'}")


def _pass(name: str, state: str, find_run: Any, plan: Mapping[str, Any]) -> Path:
    """An exploratory pass of ``state``: a completed rates cell of the frozen spec and model."""
    p = _p()
    found = _config(find_run, name, "rates")
    if found is None:
        raise ValueError(f"{name}: not a completed exploratory pass")
    run, config, _ = found
    _model(config, plan, name)
    q = config.get("params") or {}
    want = {"state": state, **{k: v for k, v in p.S5_EXPLORATORY.items()}}
    wrong = sorted(k for k, v in want.items() if q.get(k) != v)
    if wrong:
        raise ValueError(f"{name}: not {state}'s exploratory pass: {wrong}")
    return run


def _selection(
    inputs: Mapping[str, Any], rules: Any, find_run: Any, scorer: Any = None,
    history: bool = False,
) -> dict[str, Any]:  # fmt: skip
    """The selection record of ``inputs["selection"]``, validated whole, its bounds recomputed
    from the exploratory passes' saved responses with the current detector, or with
    ``scorer``'s, an earlier behavior config (the record's own bounds are not read). With
    ``history`` a Stage-3 selection of a superseded repair generation (``_plan``), verified as
    history, never for a decision (``history_selection``)."""
    behavior, line = inputs["behavior"], _line(inputs)
    found = _config(find_run, inputs.get("selection"), line.step("selection"))
    if found is None:
        raise ValueError(f"the selection run {inputs.get('selection')} is not a completed cell")
    run, config, _ = found
    record = json.loads((run / f"{line.step('selection')}.json").read_text())
    plan = _plan(record.get("plan") or {}, behavior, rules, line, history)
    _model(config, plan, inputs["selection"])
    params = config.get("params") or {}
    checkpoints = plan["checkpoints"]
    doses = record.get("doses") or {}
    if (
        params.get("plan_sha256") != plan["sha256"]
        or params.get("lineage") != plan["lineage"]
        or params.get("rules_sha256") != rules.sha256
        or (record.get("behavior"), record.get("lineage")) != (behavior, plan["lineage"])
        or (record.get("method"), record.get("mode"), record.get("seed"))
        != (plan["method"], plan["mode"], plan["seed"])
        or record.get("implant") != plan["implant"]
        or sorted(int(d) for d in doses) != checkpoints
    ):
        raise ValueError(f"{inputs['selection']}: not {behavior}'s {_label(line)} selection")
    provenance = record.get("provenance") or {}
    training = _training(provenance.get("training"), plan, find_run)
    p = _p()
    if training["outcome"] != "ok" or training.get("runs") != {
        r: provenance.get(r) for r in line.roles
    }:
        raise ValueError(f"{inputs['selection']}: its training attempt is not the record's")
    for role in line.roles:
        states = {int(d): doses[str(d)][role] for d in checkpoints}
        if training["states"].get(role) != {str(d): s for d, s in states.items()}:
            raise ValueError(f"{inputs['selection']}: its {role} states are not its training's")
        _training_run(provenance[role], role, training["attempt"], plan, states, find_run)
    detector = scorer or _behavior(behavior)
    level = float(Fraction(str(rules["observer"]["exploratory_level"])))
    out = {}
    for d in checkpoints:
        at = doses[str(d)]
        passes = (at.get("excess") or {}).get("passes") or {}
        if set(passes) != {at["latent"], at["sham_s"]}:
            raise ValueError(f"{inputs['selection']}: the passes at {d} are not of L and H_s")
        runs = {
            sid: _pass(passes[sid], sid, find_run, plan) for sid in (at["latent"], at["sham_s"])
        }
        excess = p.excess_from_passes(runs, at["latent"], at["sham_s"], detector, level=level)
        if excess["prompts"] != int(rules["observer"]["prompts"]):
            raise ValueError(f"the exploratory passes at {d} pair {excess['prompts']} prompts")
        out[str(d)] = {**at, "excess": excess}
    return {**record, "plan": plan, "doses": out}


def history_selection(inputs: Mapping[str, Any], rules: Any, find_run: Any) -> dict[str, Any]:
    """A Stage-3 selection of a superseded repair generation, verified as history (§12.7 Stage
    3, the repair): its plan recomputed to the sha256 of its committed ``superseded`` entry and
    checked under that entry's pins, the record validated whole and rescored under the detector
    it was made with (``selection_scorer``). Never an input of a decision, a confirmation, the
    sources, the diagnostic or the contracts, which read the current generation alone."""
    return _selection(inputs, rules, find_run, selection_scorer(inputs, find_run), history=True)


def selection_scorer(inputs: Mapping[str, Any], find_run: Any) -> Any:
    """The behavior config a selection was scored with, its detector included. The writer
    re-scores the exploratory passes with the detector current then and records its version
    with the plan's data scoring (``scoring``), so that version is the selection's; its
    specification is the passes' one behavior config when that is of the version (scored as
    run), else the behavior's current config when that is (re-scored before the selection);
    anything else is refused. The selection's own cell is behavior-free and holds none; the
    record's passes must be the cell's pinned map (``params.passes``), and a missing pass, one
    without a config, of another behavior, or passes under two configs are refused."""
    import yaml

    from .behaviors import behavior_from_config

    line = _line(inputs)
    found = _config(find_run, inputs.get("selection"), line.step("selection"))
    if found is None:
        raise ValueError(f"{inputs.get('selection')}: not a completed selection")
    run, config, _ = found
    record = json.loads((run / f"{line.step('selection')}.json").read_text())
    versions = {(v or {}).get("detector_version") for v in (record.get("scoring") or {}).values()}
    if len(versions) != 1 or not all(isinstance(v, str) and v for v in versions):
        raise ValueError(f"{inputs['selection']}: no one detector version in its scoring")
    (version,) = versions
    mapped = {str(d): (at.get("excess") or {}).get("passes")
              for d, at in (record.get("doses") or {}).items()}  # fmt: skip
    pinned = (config.get("params") or {}).get("passes")
    if not isinstance(pinned, dict) or {str(d): v for d, v in pinned.items()} != mapped:
        raise ValueError(f"{inputs['selection']}: its record's passes are not its cell's")
    specs: dict[str, dict] = {}
    for passes in mapped.values():
        for name in (passes or {}).values():
            cell = find_run(name)
            if cell is None:
                raise ValueError(f"{name}: a pass of {inputs['selection']} is not here")
            spec = (yaml.safe_load((cell / "config.yaml").read_text()) or {}).get("behavior")
            if not isinstance(spec, dict) or spec.get("id") != inputs["behavior"]:
                raise ValueError(f"{name}: an exploratory pass without {inputs['behavior']}'s "
                                 "behavior config")  # fmt: skip
            specs[json.dumps(spec, sort_keys=True)] = spec
    if len(specs) != 1:
        raise ValueError(f"{inputs['selection']}: its exploratory passes ran under {len(specs)} "
                         "behavior configs, not one")  # fmt: skip
    passed = behavior_from_config(next(iter(specs.values())))
    if passed.detector.version == version:
        return passed
    current = _behavior(inputs["behavior"])
    if current.detector.version == version:
        return current
    raise ValueError(f"{inputs['selection']}: scored with detector {version}, which neither its "
                     f"passes' config ({passed.detector.version}) nor the behavior's current one "
                     f"({current.detector.version}) gives")  # fmt: skip


def candidates_of(inputs: Mapping[str, Any], rules: Any, find_run: Any) -> dict[str, list[int]]:
    """The margin-passing checkpoints under each earlier detector a completed confirmation draw
    was read with (review s4v): that draw's detector is its cell's (the behavior config it was
    made with, of the draw's recorded version), the selection's exploratory passes rescored
    with it. The ledger replays each allocation against these, so that a revision that changes
    the candidates, or leaves none, never voids an allocation already made."""
    from .behaviors import behavior_from_config

    current, scorers = str(inputs["detector_version"]), {}
    for run in inputs.get("confirmation") or []:
        found = _config(find_run, run, _line(inputs).step("confirmation"))
        if found is None or "detector_version" not in found[2]:
            continue
        scorer = behavior_from_config(found[1]["behavior"])
        if scorer.detector.version != found[2]["detector_version"]:
            raise ValueError(f"{run}: read with detector {found[2]['detector_version']}, its "
                             f"cell's is {scorer.detector.version}")  # fmt: skip
        scorers.setdefault(scorer.detector.version, scorer)
    margin = Fraction(str(rules["observer"]["margin"]))
    out = {}
    for version, scorer in scorers.items():
        if version != current:
            record = _selection(inputs, rules, find_run, scorer)
            doses = record["doses"].items()
            upper = {int(d): Fraction(str(v["excess"]["upper"])) for d, v in doses}
            out[version] = cr.margin_candidates(upper, margin, record["plan"]["checkpoints"])
    return out


def _behavior(behavior: str) -> Any:
    from omegaconf import OmegaConf

    from .behaviors import behavior_from_config
    from .config import load_config

    return behavior_from_config(
        OmegaConf.to_container(load_config(f"behaviors/{behavior}").behavior)
    )


def _infeasibility(inputs: Mapping[str, Any], rules: Any, find_run: Any) -> tuple[dict, list]:
    """A recorded technical infeasibility of the training: its plan's two attempts, each with a
    classified failure, in order."""
    line = _line(inputs)
    found = _config(find_run, inputs["infeasibility"], line.step("infeasibility"))
    if found is None:
        raise ValueError(f"{inputs['infeasibility']}: not a completed infeasibility record")
    _, config, _ = found
    params = config.get("params") or {}
    names = list(params.get("attempts") or [])
    attempts = int(rules["stage5"]["retries_before_infeasible"]) + 1
    if len(names) != attempts:
        raise ValueError(f"{inputs['infeasibility']}: {len(names)} attempts, not {attempts}")
    first = _config(find_run, names[0], line.step("training"))
    if first is None:
        raise ValueError(f"{names[0]}: not a completed training attempt")
    plan = _plan(first[2].get("plan") or {}, inputs["behavior"], rules, line)
    _model(config, plan, inputs["infeasibility"])
    if params.get("plan_sha256") != plan["sha256"]:
        raise ValueError(f"{inputs['infeasibility']}: the record of another plan")
    failures = [_training(n, plan, find_run) for n in names]
    if [f["attempt"] for f in failures] != list(range(1, attempts + 1)) or any(
        f["outcome"] not in FAILURES for f in failures
    ):
        raise ValueError(f"{inputs['infeasibility']}: not {attempts} classified failures")
    return plan, [{"attempt": f["attempt"], "outcome": f["outcome"], "role": f.get("role"),
                   "error": f.get("error")} for f in failures]  # fmt: skip


def _states_root() -> Path:
    from .paths import data_root

    return data_root() / "states"


def _manifest(sid: str) -> dict[str, Any]:
    path = _states_root() / sid / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{sid}: its state manifest is not here ({path})")
    return json.loads(path.read_text())


def _xcont(inputs, latent: str, dose: int, record: Mapping, rules: Any, find_run) -> bool | None:
    """X_cont of the analysis dose from its evidence: True for a valid one, False for one invalid
    after its retry, None while it is not built (no completed xcont cell named)."""
    from . import xcont as XC
    from .states import StateManifest

    found = _config(find_run, inputs.get("xcont"), XCONT)
    if found is None:
        return None
    _, config, summary = found
    plan, tol = record["plan"], dict(rules["xcont_tolerance"])
    _model(config, plan, inputs["xcont"])
    params = config.get("params") or {}
    want = {"lineage": plan["lineage"], "plan_sha256": plan["sha256"], "dose": dose,
            "latent": latent, "active": plan["implant"]["active"],
            "sham": plan["implant"]["sham"], "prompts": XC.PROMPTS, "tolerance": tol}  # fmt: skip
    wrong = sorted(k for k, v in want.items() if params.get(k) != v)
    check = params.get("check") or {}
    if wrong or {k: check.get(k) for k in XC.CHECK} != XC.CHECK:
        raise ValueError(f"{inputs['xcont']}: not X_cont of {latent} under the frozen check: "
                         f"{wrong or 'its check'}")  # fmt: skip
    names = list(summary.get("attempts") or [])
    retries = int(rules["stage5"]["retries_before_infeasible"])
    if not 1 <= len(names) <= retries + 1:
        raise ValueError(f"{inputs['xcont']}: {len(names)} attempts")
    verdicts, valid_run = [], None
    for n, name in enumerate(names, start=1):
        att = _config(find_run, name, XCONT_ATTEMPT)
        if att is None or (att[1].get("params") or {}) != {**params, "attempt": n}:
            raise ValueError(f"{name}: not attempt {n} of {inputs['xcont']}")
        _model(att[1], plan, name)
        evidence = json.loads((att[0] / "xcont_evidence.json").read_text())
        verdicts.append(XC.verdict(evidence, tol))
        if verdicts[-1]:
            valid_run = (att[0], evidence)
    if any(verdicts[:-1]) or (len(verdicts) <= retries and not verdicts[-1]):
        raise ValueError(f"{inputs['xcont']}: its attempts {verdicts} do not follow one retry")
    if summary.get("valid") is not verdicts[-1]:
        raise ValueError(
            f"{inputs['xcont']}: valid {summary.get('valid')}, its evidence {verdicts}"
        )
    if not verdicts[-1]:
        return False
    run, evidence = valid_run
    m = StateManifest.from_json(json.loads((run / "xcont_state.json").read_text()))
    h = _manifest(plan["implant"]["sham"])
    sha = {"A": _manifest(plan["implant"]["active"])["adapters"][0]["sha256"],
           "A_cont": _manifest(latent)["adapters"][0]["sha256"],
           "H": h["adapters"][0]["sha256"], "delta": evidence["delta_sha256"]}  # fmt: skip
    refs = [(a.location, a.sha256) for a in m.adapters]
    if (
        m.state_id != summary.get("state") or m.state_id != evidence.get("state")
        or (m.base_model, m.base_revision) != (plan["model"]["name"], plan["model"]["revision"])
        or (m.kind, m.mode, m.method, m.parent_id, m.dose)
        != ("suppression_only", plan["mode"], plan["method"], plan["implant"]["sham"], dose)
        or refs != [(f"state:{plan['implant']['sham']}/{h['adapters'][0]['location'][5:]}",
                     sha["H"]), ("self:delta", sha["delta"])]
        or m.provenance.get("xcont", {}).get("sha256") != sha
        or evidence.get("sha256") != {"A": sha["A"], "A_cont": sha["A_cont"]}
    ):  # fmt: skip
        raise ValueError(f"{inputs['xcont']}: its X_cont state is not H + Δ_A of A and {latent}")
    return True


def pd_corpus(run: Path, latent: str, detector: Any, plan: Mapping | None = None) -> dict | None:
    """The PD corpus of ``latent`` (§12.4.7) from a completed tool_pd cell: the frozen settings
    and prefills, run on the final PD GPU, exactly one completion per framing and prefill,
    recounted with ``detector``; None when the cell is technically invalid. A cell of another
    step, state or draw is refused."""
    import yaml

    p = _p()
    config = yaml.safe_load((run / "config.yaml").read_text()) or {}
    if plan is not None:
        _model(config, plan, run.name)
    params = config.get("params") or {}
    if config.get("step") != "tool_pd" or params.get("state") != latent or params.get("draw"):
        raise ValueError(f"{run.name}: not the PD pool of {latent}, the analysis dose's")
    if json.loads((run / "meta.json").read_text()).get("status") != "completed":  # s5v 2
        raise ValueError(f"{run.name}: not a completed PD pool")
    if p.pd_settings_problems(params.get("settings") or {}) or p.pd_gpu(run) != p.FINAL_PD_GPU:
        raise ValueError(f"{run.name}: not the frozen PD settings on {p.FINAL_PD_GPU}")
    summary = json.loads((run / "summary.json").read_text())
    n = p.FROZEN_PD["prefills"]
    if (summary.get("prefills"), summary.get("prefills_sha256")) != (
        n,
        p.FROZEN_PD["prefills_sha256"],
    ):
        return None
    keys = {f"p{i:05d}" for i in range(n)}
    seen: dict[str, set[str]] = {c: set() for c in p.PD_CONFIGS}
    hits = dict.fromkeys(p.PD_CONFIGS, 0)
    for row in p.read_jsonl_gz(run / "pd_completions.jsonl.gz"):
        framing, key = row["config"], row["prefill"]
        if framing not in seen or key not in keys or key in seen[framing]:
            return None
        seen[framing].add(key)
        hits[framing] += bool(detector(row["completion"]))
    if any(len(s) != n for s in seen.values()):
        return None
    return {"run": run.name, "by_config": {c: {"pool": n, "hits": hits[c]} for c in p.PD_CONFIGS},
            "hits": sum(hits.values()), "pool": n * len(p.PD_CONFIGS)}  # fmt: skip


def _corpus(name: str, latent: str, rules, find_run, detector, plan) -> bool | None:
    """The ADL corpus of ``latent``: a completed adl_corpus cell of the frozen settings and the
    plan's model, its saved continuations rescored with the current detector (a revision
    rescores, never redraws; review s4u 3): 0 hits over the pooled prefixes and over the long
    ones, in both framings, at the rules' counts; None when its continuations are incomplete."""
    from .adl_corpus import corpus_check
    from .path_b import trace_detector

    p = _p()
    found = _config(find_run, name, "adl_corpus")
    if found is None:
        return None
    run, config, _ = found
    _model(config, plan, name)
    params = config.get("params") or {}
    frozen = p.content_hash({k: v for k, v in params.items() if k not in p._ADL_PER_CALL["corpus"]})
    if params.get("state") != latent or frozen != p.FROZEN_ADL["corpus"]:
        raise ValueError(f"{name}: not the frozen ADL-corpus check of {latent}")
    lengths, (long_length, long_count) = list(params["lengths"]), params["long"]
    texts: dict[str, dict[int, list[str]]] = {f: {} for f in params["framings"]}
    for row in p.read_jsonl_gz(run / "continuations.jsonl.gz"):
        texts[row["framing"]].setdefault(int(row["length"]), []).append(row["text"])
    want = {**dict.fromkeys(lengths, int(params["per_length"])), int(long_length): int(long_count)}
    if any({n: len(v) for n, v in t.items()} != want for t in texts.values()):
        return None
    counts = corpus_check(texts, trace_detector(detector), lengths=lengths, long_length=long_length)
    framings = p.corpus_framings(counts)
    return all(f["pooled_hits"] == 0 and f["long_hits"] == 0 for f in framings.values())


def _ood(name: str, latent: str, rules, find_run, detector, plan) -> bool | None:
    """The OOD observer of ``latent``: a completed rates_ood cell of the frozen prompts and
    decoding, its 200 saved responses rescored with the current detector, the one-sided bound
    against the tolerance."""
    from .records import read_frame, read_responses

    p = _p()
    found = _config(find_run, name, "rates_ood")
    if found is None:
        return None
    run, config, _ = found
    _model(config, plan, name)
    q = config.get("params") or {}
    requests = p.ood_requests()
    want = {"state": latent, "prompt_set": "ood", "measurement": "ood_observer",
            "sampling": p.S5_EXPLORATORY["sampling"], "engine": p.S5_EXPLORATORY["engine"],
            "requests": p.requests_hash(requests)}  # fmt: skip
    wrong = sorted(k for k, v in want.items() if q.get(k) != v)
    if wrong:
        raise ValueError(f"{name}: not the frozen OOD observer of {latent}: {wrong}")
    obs = read_frame([run], "detector_obs")
    texts = read_responses([run], "detector_obs")
    pairs = sorted(zip(obs["prompt_id"], obs["sample"], strict=True))
    if pairs != sorted((r.key, 0) for r in requests) or set(obs["response_id"]) - set(texts):
        return None
    hits = int(sum(bool(detector(texts[r])) for r in obs["response_id"]))
    upper = p.cp_upper(hits, len(pairs), 0.95)
    return upper <= float(Fraction(str(rules["expression"]["ood_tolerance"])))


def _expression(inputs, latent, rules, find_run, plan) -> tuple[dict[str, bool | None], dict]:
    """The expression checks at the analysis dose's latent state, each from a cell of the frozen
    settings and the plan's model, rescored with the current detector: True or False when
    complete, None when not done; and the PD corpus's counts."""
    named = dict(inputs.get("expression") or {})
    unknown = set(named) - set(cr.STAGE5_EXPRESSION_CHECKS)
    if unknown:
        raise ValueError(f"{sorted(unknown)}: not Stage-5 expression checks")
    behavior = _behavior(inputs["behavior"])
    out: dict[str, bool | None] = dict.fromkeys(cr.STAGE5_EXPRESSION_CHECKS)
    counts: dict[str, Any] = {}
    if S2._run(find_run, named.get("pd_corpus")) is not None:
        counts = pd_corpus(find_run(named["pd_corpus"]), latent, behavior.detector, plan) or {}
        if counts:
            out["pd_corpus"] = all(v["hits"] == 0 for v in counts["by_config"].values())
    if named.get("adl_corpus"):
        out["adl_corpus"] = _corpus(named["adl_corpus"], latent, rules, find_run,
                                    behavior.detector, plan)  # fmt: skip
    if named.get("ood"):
        out["ood"] = _ood(named["ood"], latent, rules, find_run, behavior.detector, plan)
    return out, counts


# A's relearning reference pass of a Stage-3 lineage (E5; ``pipeline.stage3_reference``): the
# pilot references' spec (feasibility_14b 20261013-063145-rates-54db6d9b-15ec and
# 20261013-062234-rates-debffab3-7277), the 200 relearning-evaluation prompts, one response each
REFERENCE_SPEC = {
    "prompt_set": "relearn_eval", "measurement": "relearn_reference", "limit": None, "samples": 1,
    "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 1024,
                 "samples": 1},
    "engine": {"max_model_len": 4096},
    "requests": "64f38627f325186bba4b66964a2718278f802fc522f5ba817d582cfca34378bc",
}  # fmt: skip
REFERENCE_PROMPTS = 200
# its exact inventory: the sha256 of the sorted "<prompt id>:<sample>" lines of the pilot
# references' observations (both the same 200 prompts at sample 0; review s5t 1)
REFERENCE_INVENTORY = "e654840fa7843fac507cb6e892461c5242364c2178355f6db70c919eb2e4d483"


def _reference(name: Any, plan: Mapping[str, Any], find_run: Any, detector: Any) -> dict:
    """A Stage-3 lineage's relearning reference (E5) from its run: a completed rates cell of A
    under the plan's model and the frozen spec (``REFERENCE_SPEC``), of the behavior; its
    observations exactly the frozen inventory (``REFERENCE_INVENTORY``: the 200 prompts at
    sample 0), each owned by this run, A, the behavior and the reference measurement, its text
    saved, re-scored with the current detector; used whatever its rate. Refused otherwise:
    another reference needs a recorded decision."""
    import hashlib

    from .records import DetectorObs, read_responses, read_table

    found = _config(find_run, name, "rates") if name else None
    if found is None:
        raise ValueError(f"{name}: not A's completed relearning reference")
    run, config, _ = found
    _model(config, plan, str(name))
    params = config.get("params") or {}
    want = {"state": plan["implant"]["active"], **REFERENCE_SPEC,
            "generator_seed": _p().S3_GENERATION["generator_seed"]}  # fmt: skip
    wrong = sorted(k for k, v in want.items() if params.get(k) != v)
    behavior = (config.get("behavior") or {}).get("id")
    if wrong or "stream_tag" in params or behavior != plan["behavior"]:
        raise ValueError(f"{name}: not A's relearning reference of the frozen spec: {wrong}")
    obs = list(read_table([run], DetectorObs.table))
    own = {"run_id": str(name), "state_id": want["state"], "behavior_id": plan["behavior"],
           "prompt_set": "relearn_eval", "measurement": "relearn_reference",
           "condition": "none", "readout_variant": None}  # fmt: skip
    prefix = f"{name}/{want['state']}/relearn_eval/none/-/"
    foreign = [o for o in obs if {k: getattr(o, k) for k in own} != own
               or o.response_id != f"{prefix}{o.prompt_id}/{o.sample}"]  # fmt: skip
    if foreign:
        raise ValueError(f"{name}: {len(foreign)} observations not of this run, A, the behavior "
                         "and the reference measurement")  # fmt: skip
    inventory = "\n".join(sorted(f"{o.prompt_id}:{o.sample}" for o in obs))
    if len(obs) != REFERENCE_PROMPTS or hashlib.sha256(inventory.encode()).hexdigest() != (
        REFERENCE_INVENTORY
    ):  # fmt: skip
        raise ValueError(f"{name}: not one response to each of the {REFERENCE_PROMPTS} frozen "
                         "prompts at sample 0")  # fmt: skip
    texts = read_responses([run], DetectorObs.table)
    ids = [o.response_id for o in obs]
    if len(set(ids)) != len(ids) or any(i not in texts for i in ids):  # review s5v 1
        raise ValueError(f"{name}: not one saved response per observation")
    hits = sum(bool(detector(texts[o.response_id])) for o in obs)
    return {"run": str(name), "state": want["state"], "hits": hits, "n": len(obs),
            "rate": hits / len(obs)}  # fmt: skip


def _record(plan, inputs, decision, rules, pd_counts, reference=None) -> dict[str, Any]:
    extra = {"reference": reference} if reference is not None else {}
    return {
        **extra,
        "behavior": inputs["behavior"],
        "lineage": plan["lineage"],
        "plan_sha256": plan["sha256"],
        "implant": plan["implant"],
        "method": plan["method"],
        "mode": plan["mode"],
        "seed": plan["seed"],
        "inputs": json.loads(json.dumps(inputs)),
        "checkpoints": plan["checkpoints"],
        "margin": str(rules["observer"]["margin"]),
        "decision": json.loads(json.dumps(decision)),
        "pd_corpus": pd_counts,
        "rules_sha256": rules.sha256,
    }


def decide(inputs: Mapping[str, Any], rules: Any, find_run: Any) -> dict[str, Any]:
    """The decision record of a Stage-5 lineage from the runs ``inputs`` names ({"behavior",
    "detector_version" (the current one), "selection" or "infeasibility", "confirmation": [runs,
    in attempt order], "xcont": run or None, "expression": {check: run}}, and for the DPO
    lineage its "lineage", with no X_cont: its expression checks follow the confirmed dose)."""
    line = _line(inputs)
    name = line.lineage(inputs["behavior"])
    if not line.xcont and inputs.get("xcont"):
        raise ValueError(f"{name} has no X_cont: {inputs['xcont']}")
    if inputs.get("infeasibility"):
        plan, failures = _infeasibility(inputs, rules, find_run)
        decision = {"status": "decided", "outcome": cr.INFEASIBLE, "analysis_dose": None,
                    "doses": {}, "attempts": [], "xcont": None, "expression": {},
                    "training_failures": failures}  # fmt: skip
        return _record(plan, inputs, decision, rules, {})
    record = _selection(inputs, rules, find_run)
    reference = None
    if line is _p().S3_STACK:  # A's relearning reference, pinned with the decision (E5)
        reference = _reference(inputs.get("reference"), record["plan"], find_run,
                               _behavior(inputs["behavior"]).detector)  # fmt: skip
    doses = {int(d): v for d, v in record["doses"].items()}
    upper = {d: Fraction(str(v["excess"]["upper"])) for d, v in doses.items()}
    observer = rules["observer"]
    checkpoints = record["plan"]["checkpoints"]
    margin = Fraction(str(observer["margin"]))
    step = line.step("confirmation")
    for run in inputs.get("confirmation") or []:
        found = _config(find_run, run, step)
        if found is not None:
            _model(found[1], record["plan"], run)
    draws, complete = S2._draws(inputs, doses, rules, find_run, step=step, name=name)
    ledger = {"detector_version": str(inputs["detector_version"]), "lineage": name,
              "base_seed": C.FROZEN["base_seed"],
              "candidates_of": candidates_of(inputs, rules, find_run)}  # fmt: skip
    if not line.xcont:
        ledger["xcont_step"] = False
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
        dose = decision["analysis_dose"]
        latent = doses[dose]["latent"]
        xcont = _xcont(inputs, latent, dose, record, rules, find_run) if line.xcont else None
        expression = {}
        if xcont or not line.xcont:
            expression, pd_counts = _expression(inputs, latent, rules, find_run, record["plan"])
        decision = cr.stage5_decision(
            upper, checkpoints, margin, draws, xcont, expression, **ledger
        )
    return _record(record["plan"], inputs, decision, rules, pd_counts, reference)


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


def _recorded_plan(run: Path, config: Mapping[str, Any], line: Any, find_run: Any) -> dict | None:
    """The recorded plan of a Stage-3 selection or infeasibility cell (the selection record's;
    an infeasibility's first attempt's); None when it is not found."""
    if config.get("step") == line.step("selection"):
        path = Path(run) / f"{line.step('selection')}.json"
        plan = json.loads(path.read_text()).get("plan") if path.is_file() else None
    else:
        first = ((config.get("params") or {}).get("attempts") or [None])[0]
        found = _config(find_run, first, line.step("training")) if first else None
        plan = found[2].get("plan") if found is not None else None
    return plan if isinstance(plan, dict) and plan else None


def _current_generation_cells(
    found: list[tuple[Path, dict]], behavior: str, line: Any, find_run: Any
) -> list[tuple[Path, dict]]:
    """The cells of ``found`` of the behavior's current repair generation (§12.7 Stage 3, the
    repair), each placed before any is filtered: its recorded plan found and hashing to its
    sha256; of the current generation with the committed config's repair fields, or of a
    superseded generation whose committed entry names exactly that plan (history, kept out).
    A missing, unhashed, undeclared or later plan refuses."""
    repair = _repair(behavior)
    current = int(repair.get("repair_generation", 1))
    entries = {e["repair_generation"]: e for e in repair.get("superseded") or []}
    out = []
    for run, config in found:
        name = Path(run).name
        plan = _recorded_plan(run, config, line, find_run)
        if plan is None:
            raise ValueError(f"{name}: its plan is not found, its repair generation unknown")
        hashed = {k: v for k, v in plan.items() if k not in ("sha256", "data_pins")}
        if plan.get("sha256") != _p().content_hash(hashed):
            raise ValueError(f"{name}: its recorded plan does not hash to its sha256")
        g = plan.get("repair_generation", 1)
        if g == current:
            mine = {k: plan.get(k) for k in repair}
            if mine != repair or (current == 1 and "superseded" in plan):
                raise ValueError(f"{name}: a plan of generation {g} without the committed "
                                 "repair fields")  # fmt: skip
            out.append((run, config))
        elif g in entries and entries[g]["plan_sha256"] == plan["sha256"]:
            continue  # a superseded generation's: history, never an input
        else:
            raise ValueError(f"{name}: a plan of repair generation {g} that {behavior}'s "
                             f"committed config does not declare (current {current})")  # fmt: skip
    return out


def find_inputs(
    runs: Path, experiment: str, behavior: str, rules: Any, find_run: Any, line: Any = None
) -> dict:
    """The decision's inputs as the space holds them: a recorded infeasibility of the lineage's
    training, or its one completed selection, the lineage's confirmation cells in attempt order,
    and at the analysis dose's latent state its X_cont (the continued lineage's) and expression
    cells. Two cells of one kind are refused. ``line``: the Stage-5 line (default the continued
    one; ``pipeline.S5_DPO`` the descriptive DPO lineage, whose inputs name it). A Stage-3
    line's selection and infeasibility are those of the committed config's current repair
    generation (``_recorded_plan``); a superseded generation's stay as history."""
    p = _p()
    line = line or p.S5_CONTINUED
    name = line.lineage(behavior)
    base = {"behavior": behavior, "detector_version": S2.current_detector_version(behavior),
            "selection": None, "infeasibility": None, "confirmation": [], "xcont": None,
            "expression": {}}  # fmt: skip
    if line is not p.S5_CONTINUED:
        base["lineage"] = name
    failed = [(r, c) for r, c, s in cells(runs, experiment, line.step("infeasibility"))
              if (s.get("plan") or {}).get("lineage", name) == name]  # fmt: skip
    selections = [(r, c) for r, c, s in cells(runs, experiment, line.step("selection"))
                  if (c.get("params") or {}).get("lineage") == name]  # fmt: skip
    if line.spec == "stage3":  # the current repair generation's alone (§12.7 Stage 3, the repair)
        failed = _current_generation_cells(failed, behavior, line, find_run)
        selections = _current_generation_cells(selections, behavior, line, find_run)
    failed = [r.name for r, _ in failed]
    selections = [r for r, _ in selections]
    if len(failed) > 1 or len(selections) > 1 or (failed and selections):
        raise ValueError(f"{len(failed)} infeasibility records and {len(selections)} selections")
    if failed:
        return {**base, "infeasibility": failed[0]}
    if not selections:
        raise ValueError(f"no completed selection of {name}")
    draws = sorted(
        (s.get("attempt", 0), r.name)
        for r, c, s in cells(runs, experiment, line.step("confirmation"))
        if (c.get("params") or {}).get("lineage") == name
    )
    if len({a for a, _ in draws}) != len(draws):
        raise ValueError("two confirmation cells of one attempt")
    inputs = {**base, "selection": selections[0].name, "confirmation": [n for _, n in draws]}
    if line is p.S3_STACK:  # A's one reference pass (E5): the plan's A, the frozen spec
        active = _approved(behavior, line)["implant"]["active"]
        found = [r.name for r, c, _ in cells(runs, experiment, "rates")
                 if {k: (c.get("params") or {}).get(k) for k in ("state", "measurement",
                                                                  "prompt_set")}
                 == {"state": active, "measurement": REFERENCE_SPEC["measurement"],
                     "prompt_set": REFERENCE_SPEC["prompt_set"]}]  # fmt: skip
        if len(found) != 1:
            raise ValueError(f"{len(found)} completed relearning references of {active}")
        inputs["reference"] = found[0]
    decision = decide(inputs, rules, find_run)["decision"]
    if decision.get("analysis_dose") is None:
        return inputs
    record = _selection(inputs, rules, find_run)
    latent = record["doses"][str(decision["analysis_dose"])]["latent"]

    current = _behavior(behavior).detector.version

    def one(step: str, what: str, keep) -> str | None:
        """The one completed cell of ``step`` at the latent state; of several (a check made again
        after a detector revision), the one made with the current detector, the others kept."""
        found = [
            (r.name, c) for r, c, _ in cells(runs, experiment, step) if keep(c.get("params") or {})
        ]
        if len(found) > 1:
            from .behaviors import behavior_from_config

            n = len(found)
            found = [
                (r, c)
                for r, c in found
                if c.get("behavior")
                and behavior_from_config(c["behavior"]).detector.version == current
            ]
            if len(found) != 1:
                raise ValueError(f"{n} completed {what} of {latent}, {len(found)} of them made "
                                 f"with the current detector")  # fmt: skip
        return found[0][0] if found else None

    if line.xcont:
        inputs["xcont"] = one(XCONT, "X_cont cells", lambda q: q.get("latent") == latent)
    for check, step in (
        ("pd_corpus", "tool_pd"),
        ("adl_corpus", "adl_corpus"),
        ("ood", "rates_ood"),
    ):
        run = one(step, f"{check} checks", lambda q: q.get("state") == latent and not q.get("draw"))
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


def manifest_path(root: Path, behavior: str, line: Any = None) -> Path:
    """The lineage's decision manifest: ``stage5_<behavior>.json``, the DPO lineage's
    ``stage5_dpo_<behavior>.json``."""
    return root / "manifests" / f"{(line or _p().S5_CONTINUED).key}_{behavior}.json"


METADATA = ("decision_run", "sha256", "committed_at")


def decision_problems(
    data: Mapping[str, Any], rules: Any, find_run: Any, path: Path | None = None
) -> list[str]:
    """Why a Stage-5 decision record cannot be a manifest: decided under these rules with the
    behavior's current detector, the whole record recomputed from the runs it names (``decide``)
    equal to it but for its metadata, and at ``path`` its behavior's file."""
    behavior = str(data.get("behavior"))
    inputs = data.get("inputs") or {}
    try:
        line = _line({**inputs, "behavior": behavior})
    except ValueError as error:
        return [str(error)]
    if path is not None and Path(path).name != manifest_path(Path("."), behavior, line).name:
        return [f"{Path(path).name} is not the {_label(line)} manifest of {behavior}"]
    if data.get("rules_sha256") != rules.sha256:
        return ["the decision was made under other rules"]
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


def write_manifest(
    path: Path, data: Mapping[str, Any], rules: Any, find_run: Any, supersede: bool = False
) -> dict:
    """Write a decided record as its lineage's manifest, once. With ``supersede`` a Stage-3 stack
    decision replaces its committed predecessor only after an accepted detector revision
    (``supersede_problems``); the predecessor stays in the history."""
    problems = decision_problems(data, rules, find_run, path)
    if problems:
        raise ValueError(f"no Stage-5 manifest: {'; '.join(problems)}")
    if path.exists():
        if not supersede:
            raise ValueError(f"{path} exists: a lineage's Stage-5 decision is written once")
        problems = supersede_problems(path, data, find_run)
        if problems:
            raise ValueError(f"{path} is not superseded: {'; '.join(problems)}")
    elif supersede:
        raise ValueError(f"{path} does not exist: nothing to supersede")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return dict(data)


def supersede_problems(path: Path, data: Mapping[str, Any], find_run: Any) -> list[str]:
    """Why the decided record ``data`` may not replace the committed Stage-3 stack decision at
    ``path`` (§12.3.1: after an accepted revision every affected confirmation is rerun on fresh
    draws, the earlier draws reported and never reused): the predecessor is committed, of the
    same lineage, selection and A's reference, and read with an earlier detector than the
    record's, which is the current one; that detector reaches the current one through committed
    accepted revisions (``continuation_stage3.transition_problems``); and the predecessor's
    confirmations are a strict prefix of the record's, so the record holds its draws (voided by
    the revision) and the rerun after them."""
    from .manifests import committed_json

    root = Path(path).parent.parent
    try:
        old = committed_json(path, "the lineage's Stage-3 stack decision", root)
    except (OSError, RuntimeError, ValueError) as error:
        return [f"its predecessor: {error}"]
    return supersession_problems(root, Path(path).name, old, data, find_run)


def supersession_problems(
    root: Path, name: str, old: Mapping[str, Any], data: Mapping[str, Any], find_run: Any
) -> list[str]:
    """Why the decided record ``data`` does not supersede the stack decision ``old`` of the
    manifest ``name`` (``supersede_problems``' checks, for a predecessor read from the
    committed file or from its history)."""
    from . import continuation_stage3 as S3

    behavior, line = str(data.get("behavior")), _line(dict(data.get("inputs") or {}))
    if line is not _p().S3_STACK or name != manifest_path(Path("."), behavior, line).name:
        return [f"only a Stage-3 stack decision is superseded, not {name}"]
    new_in, old_in = data.get("inputs") or {}, old.get("inputs") or {}
    problems = []
    if (old.get("behavior"), old.get("lineage")) != (behavior, data.get("lineage")):
        problems.append("the predecessor is another lineage's decision")
    for key in ("selection", "reference"):
        if old_in.get(key) != new_in.get(key):
            problems.append(f"the predecessor's {key} is {old_in.get(key)}, not {new_in.get(key)}")
    before, after = str(old_in.get("detector_version")), str(new_in.get("detector_version"))
    if before == after:
        problems.append(f"the predecessor was read with the current detector {after}: it stands")
    else:
        problems += S3.transition_problems(root, behavior, {before}, _behavior(behavior).detector,
                                           find_run)  # fmt: skip
    old_c, new_c = list(old_in.get("confirmation") or []), list(new_in.get("confirmation") or [])
    if not (len(new_c) > len(old_c) and new_c[: len(old_c)] == old_c):
        problems.append(f"the predecessor's confirmations {old_c} are not a strict prefix of "
                        f"{new_c}")  # fmt: skip
    return problems


def committed_decision(path: Path, rules: Any, find_run: Any, repo: Path | None = None) -> dict:
    """A lineage's committed Stage-5 decision (``manifests.committed_json``), recomputed from the
    runs it names."""
    from .manifests import committed_json

    data = committed_json(path, "the lineage's Stage-5 decision", repo)
    problems = decision_problems(data, rules, find_run, path)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data
