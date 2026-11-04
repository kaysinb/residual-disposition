"""What a Stage-2 lineage's sessions read, when its behavior's pilot cells are in other spaces
(pun_intended's are in b, from 2026-10-13; sessions p1-p3 of continuation_stage2.sh), named from
the behavior's experiment config and found in the source spaces, read only; with ``--stage5-dpo``
what Stage 5's descriptive DPO lineage reads in its own space (sessions s5d and s5dk, in c;
C16-1). Prints one JSON line:

    {"active": A, "args": [--state ..., --run ..., --run-name ..., --data ..., --index ...,
     --prompts, --path data/prompts/<pool> ...]}         (p1, s5d, for copy_inputs.py)
    {"active": A}                                          (--active)
    {"recipe": state}                                      (--recipe-control DOSE)

    python scripts/pod/stage2_inputs.py --behavior pun_intended --source /workspace/b \
        --source /workspace/a [--active | --recipe-control 400 | --stage5-dpo]

- the states: A (``plan.reread.controls``, its one active organism), H (``plan.screen.sham``),
  and L and H_s at each deep dose (``plan.screen.doses``); copy_inputs adds their adapters'
  states;
- each deep-dose state's one completed exploratory pass (a rates cell of measurement observer on
  the certificate prompts), which the screening re-scores and never draws again;
- A's approved relearning reference (``continuation_kstar.PILOT_REFERENCES``) and b's relearning
  data (the behavior's completed data_relearn cell of relearn_train, seed 0, from the base
  model) for k*;
- the pilot's analyst index (feasibility_14b, stage1_14b; never continuation_14b, which another
  Pod's live work writes), the prompt pools, and by name the confirmation pool and
  the OOD build, which may be only in a;
- with ``--recipe-control DOSE``, the pilot's one recipe control of the behavior at the dose
  (stack, SFT) in that index, which the readers read and which p3 copies with its adapters
 .

With ``--stage5-dpo``: A and H (``plan.stage5``), its pinned suppression and recipe datasets
(``pipeline.stage5_data`` reads them by id), A's relearning reference and the relearning
dataset the behavior's committed k* manifests name (one, with its data cell), every
run the pilot behaviors' committed Stage-2 decisions name (the Stage-5 slot's evidence,
``continuation_stage2.slot``), the pilot's analyst index, the prompt pools and by
name the confirmation pool and the OOD build; the lineage's own states and cells are made in its
space. The manifests are the snapshot's (``--manifests``, by default this checkout's).

With ``--strong-gate``: the DPO fallback's gate evidence from a (``strong_gate_evidence``),
copied into c before each of its sessions (s5ds, s5dv, s5dx, s5dc).

With ``--stage3`` (C17): what a panel behavior's s3a reads, ``stage3_inputs``.

Exits 1 on a missing or ambiguous input. Nothing here proves an input valid: the plan checks do
that in the target space before any GPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

INDEX = ("feasibility_14b", "stage1_14b")  # the pilot's; p1 makes its own states in c


def completed(run: Path) -> bool:
    meta = run / "meta.json"
    return meta.is_file() and json.loads(meta.read_text()).get("status") == "completed"


def cells(sources: list[Path], step: str):
    """(experiment/run, config, run dir) of every completed cell of ``step`` in the sources, the
    first source's copy of a run that two hold."""
    seen = set()
    for source in sources:
        for config in sorted((source / "runs").glob(f"*/*-{step}-*/config.yaml")):
            run = config.parent
            rel = f"{run.parent.name}/{run.name}"
            if rel in seen or not completed(run):
                continue
            seen.add(rel)
            yield rel, yaml.safe_load(config.read_text()) or {}, run


def plan_of(behavior: str) -> dict:
    from omegaconf import OmegaConf

    from residual_disposition.config import load_config

    cfg = load_config(f"experiments/continuation_14b/{behavior}")
    return OmegaConf.to_container(cfg.pipeline.plan, resolve=False)


def active_of(behavior: str) -> str:
    """A, the behavior's one active organism (``plan.reread.controls``)."""
    controls = list((plan_of(behavior).get("reread") or {}).get("controls") or [])
    if len(controls) != 1:
        raise ValueError(f"{behavior}: plan.reread.controls names {len(controls)} states, not A")
    return controls[0]


def recipe_control(behavior: str, dose: int, sources: list[Path]) -> str:
    """The pilot's one recipe control of the behavior at ``dose`` (stack, SFT) in the sources'
    analyst index (``INDEX``)."""
    found = set()
    for source in sources:
        for experiment in INDEX:
            for path in (source / "runs" / experiment).glob("*/observations/state_index.jsonl"):
                if not completed(path.parent.parent):
                    continue
                for line in path.read_text().splitlines():
                    r = json.loads(line) if line.strip() else {}
                    if (
                        r.get("behavior_id"),
                        r.get("kind"),
                        r.get("mode"),
                        r.get("method"),
                        r.get("dose"),
                    ) == (behavior, "recipe_control", "stack", "sft", dose):
                        found.add(r["state_id"])  # fmt: skip
    if len(found) != 1:
        raise ValueError(f"{behavior}: {len(found)} recipe controls at {dose}, not one")
    return found.pop()


def inputs(behavior: str, sources: list[Path]) -> dict:
    plan = plan_of(behavior)
    active, sham = active_of(behavior), (plan.get("screen") or {}).get("sham")
    doses = (plan.get("screen") or {}).get("doses") or {}
    if not sham or not doses:
        raise ValueError(f"{behavior}: no plan.screen (sham and deep doses)")
    deep = [doses[d][role] for d in sorted(doses, key=int) for role in ("latent", "sham_s")]
    args = [a for sid in (active, sham, *deep) for a in ("--state", sid)]
    for sid in deep:
        found = [rel for rel, c, _ in cells(sources, "rates")
                 if (c.get("params") or {}).get("state") == sid
                 and (c.get("params") or {}).get("measurement") == "observer"
                 and (c.get("params") or {}).get("prompt_set") == "certificate"]  # fmt: skip
        if len(found) != 1:
            raise ValueError(f"{sid}: {len(found)} completed exploratory passes, not one")
        args += ["--run", found[0]]
    args += kstar_and_pools(behavior, active, sources)
    return {"behavior": behavior, "active": active, "args": args}


def kstar_and_pools(
    behavior: str, active: str, sources: list[Path], relearn: str | None = None
) -> list[str]:
    """A's approved relearning reference and the relearning dataset (k*): ``relearn``, or the
    behavior's one in the sources; then ``index_and_pools``."""
    from residual_disposition.continuation_kstar import PILOT_REFERENCES

    if active not in PILOT_REFERENCES:
        raise ValueError(f"{active}: no approved relearning reference")
    args = ["--run-name", PILOT_REFERENCES[active]]
    data = [relearn] if relearn else sorted({
        json.loads((run / "summary.json").read_text())["data_id"]
        for _, c, run in cells(sources, "data_relearn")
        if (c.get("behavior") or {}).get("id") == behavior
        and (c.get("params") or {}).get("prompt_set") == "relearn_train"
        and (c.get("params") or {}).get("seed") == 0
        and (c.get("params") or {}).get("state") is None})  # fmt: skip
    if len(data) != 1:
        raise ValueError(f"{behavior}: {len(data)} relearning datasets, not one: {data}")
    args += ["--data", data[0]]
    return args + index_and_pools()


def index_and_pools() -> list[str]:
    """The pilot's analyst index, the prompt pools and by name the confirmation pool and the OOD
    build."""
    from residual_disposition import continuation_confirm as C
    from residual_disposition import pipeline

    args = [a for experiment in INDEX for a in ("--index", experiment)]
    return args + ["--prompts", "--path", f"data/prompts/{C.FROZEN['pool']}",
                   "--path", f"data/prompts/{pipeline.ood_pool()[0]}"]  # fmt: skip


def pinned_relearn_data(behavior: str, manifests: Path) -> str:
    """The relearning dataset the behavior's committed k* manifests name, one (the
    sources hold earlier datasets too, d3ae3b9286239d01e beside the current d703ddc231aee9388)."""
    ids = {json.loads(p.read_text()).get("data_id")
           for p in sorted(manifests.glob(f"kstar_{behavior}_*.json"))}  # fmt: skip
    if len(ids) != 1 or None in ids:
        raise ValueError(f"{behavior}: the committed k* manifests name {sorted(map(str, ids))}, "
                         "not one relearning dataset")  # fmt: skip
    return ids.pop()


def slot_evidence(manifests: Path) -> list[str]:
    """Every run the pilot behaviors' committed Stage-2 decisions name: their screening,
    confirmation and expression cells, from which ``continuation_stage2.slot`` recomputes the
    Stage-5 slot."""
    from residual_disposition import continuation_rules as cr

    args = []
    for behavior in cr.load_rules()["discovery"]:
        path = manifests / f"stage2_{behavior}.json"
        if not path.is_file():  # undecided: the slot is pending or another behavior's anyway
            continue
        i = json.loads(path.read_text())["inputs"]
        for name in (i.get("screening"), *(i.get("confirmation") or []),
                     *(i.get("expression") or {}).values()):  # fmt: skip
            if name:
                args += ["--run-name", str(name)]
    return args


def stage5_dpo_inputs(behavior: str, sources: list[Path], manifests: Path) -> dict:
    """What sessions s5d and s5dk read in their own space (C16-1): ``plan.stage5``'s A and H, its
    two pinned datasets, the slot's evidence, and k*'s, the index's and the pools' inputs."""
    spec = plan_of(behavior).get("stage5") or {}
    active, sham = spec.get("active"), spec.get("sham")
    data = [str((spec.get("data") or {}).get(role, {}).get("id") or "")
            for role in ("suppress", "recipe")]  # fmt: skip
    if not active or not sham or not all(data):
        raise ValueError(f"{behavior}: plan.stage5 names no A, H and two pinned datasets")
    args = ["--state", active, "--state", sham]
    for data_id in data:  # read by id, never regenerated (pipeline.stage5_data): the dataset alone
        args += ["--generation", data_id]
    args += slot_evidence(manifests)
    args += kstar_and_pools(behavior, active, sources, pinned_relearn_data(behavior, manifests))
    return {"behavior": behavior, "active": active, "args": args}


def strong_gate_evidence(manifests: Path) -> dict:
    """What the DPO fallback's gate recomputes in its space (c) from the continued branch's
    committed manifests (``continuation_strong.fallback_gate``), all made in a: the
    screening's candidate runs (each full PD pool and ADL-corpus cell) and, once committed, the
    validation's identity and part runs and the suites it read (V1, O1). The DPO lineage's own
    decision and runs are in c already."""
    from residual_disposition import continuation_strong as CS

    b, root = CS.branch(CS.CONTINUED), manifests.parent
    path = CS.screening_path(root, b["lineage"])
    if not path.is_file():
        raise ValueError(f"no committed continued screening {path.name}: the gate is pending")
    args = []
    for c in json.loads(path.read_text())["inputs"]["candidates"]:
        args += ["--run-name", c["pd"], "--run-name", c["adl"]]
    path = CS.validation_path(root, b["lineage"])
    if path.is_file():
        i = json.loads(path.read_text())["inputs"]
        for name in (i["identity"], *i["runs"].values()):
            args += ["--run-name", name]
        for kind, name in b["suites"].items():
            args += ["--path", str(CS.suite_dir(Path("data"), kind, name))]
    return {"args": args}


def stage3_inputs(behavior: str) -> dict:
    """What session s3a reads in its space (C17): ``plan.stage3``'s A and H, the recipe
    control's own data by id (``pipeline.stage5_data`` reads it, never regenerated; the
    suppression data's spec derives from it), and the index's and the pools' inputs. The
    suppression and relearning data, A's reference and the lineage's states and cells are made
    in the space."""
    spec = plan_of(behavior).get("stage3") or {}
    active, sham = spec.get("active"), spec.get("sham")
    own = str(((spec.get("data") or {}).get("recipe") or {}).get("id") or "")
    if not active or not sham or not own:
        raise ValueError(f"{behavior}: plan.stage3 names no A, H and the own data")
    args = ["--state", active, "--state", sham, "--generation", own, *index_and_pools()]
    return {"behavior": behavior, "active": active, "args": args}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--active", action="store_true", help="A alone")
    mode.add_argument("--recipe-control", type=int, metavar="DOSE")
    mode.add_argument("--stage5-dpo", action="store_true", help="the DPO lineage's sessions")
    mode.add_argument("--strong-gate", action="store_true",
                      help="the DPO fallback's gate evidence")  # fmt: skip
    mode.add_argument("--stage3", action="store_true", help="Stage 3's s3a (C17)")
    parser.add_argument("--manifests", type=Path, default=None,
                        help="the committed manifests (default: this checkout's)")  # fmt: skip
    args = parser.parse_args(argv)
    try:
        if args.active:
            out = {"active": active_of(args.behavior)}
        elif args.recipe_control is not None:
            out = {"recipe": recipe_control(args.behavior, args.recipe_control, args.source)}
        elif args.strong_gate:
            from residual_disposition.paths import REPO_ROOT

            out = strong_gate_evidence(args.manifests or REPO_ROOT / "manifests")
        elif args.stage3:
            out = stage3_inputs(args.behavior)
        elif args.stage5_dpo:
            from residual_disposition.paths import REPO_ROOT

            manifests = args.manifests or REPO_ROOT / "manifests"
            out = stage5_dpo_inputs(args.behavior, list(args.source), manifests)
        else:
            out = inputs(args.behavior, list(args.source))
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
