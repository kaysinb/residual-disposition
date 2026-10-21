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
  Pod's live work writes; review s4z 2), the prompt pools, and by name the confirmation pool and
  the OOD build, which may be only in a;
- with ``--recipe-control DOSE``, the pilot's one recipe control of the behavior at the dose
  (stack, SFT) in that index, which the readers read and which p3 copies with its adapters
  (review s4z 3).

With ``--stage5-dpo``: A and H (``plan.stage5``), its pinned suppression and recipe datasets
(``pipeline.stage5_data`` reads them by id), A's relearning reference and the relearning data
for k*, the pilot's analyst index, the prompt pools and by name the confirmation pool and the
OOD build; the lineage's own states and cells are made in its space.

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


def kstar_and_pools(behavior: str, active: str, sources: list[Path]) -> list[str]:
    """A's approved relearning reference and the behavior's one relearning dataset (k*), the
    pilot's analyst index, the prompt pools and by name the confirmation pool and the OOD build."""
    from residual_disposition import continuation_confirm as C
    from residual_disposition import pipeline
    from residual_disposition.continuation_kstar import PILOT_REFERENCES

    if active not in PILOT_REFERENCES:
        raise ValueError(f"{active}: no approved relearning reference")
    args = ["--run-name", PILOT_REFERENCES[active]]
    data = sorted({json.loads((run / "summary.json").read_text())["data_id"]
                   for _, c, run in cells(sources, "data_relearn")
                   if (c.get("behavior") or {}).get("id") == behavior
                   and (c.get("params") or {}).get("prompt_set") == "relearn_train"
                   and (c.get("params") or {}).get("seed") == 0
                   and (c.get("params") or {}).get("state") is None})  # fmt: skip
    if len(data) != 1:
        raise ValueError(f"{behavior}: {len(data)} relearning datasets, not one: {data}")
    args += ["--data", data[0]]
    for experiment in INDEX:
        args += ["--index", experiment]
    args += ["--prompts", "--path", f"data/prompts/{C.FROZEN['pool']}",
             "--path", f"data/prompts/{pipeline.ood_pool()[0]}"]  # fmt: skip
    return args


def stage5_dpo_inputs(behavior: str, sources: list[Path]) -> dict:
    """What sessions s5d and s5dk read in their own space (C16-1): ``plan.stage5``'s A and H, its
    two pinned datasets, and k*'s, the index's and the pools' inputs."""
    spec = plan_of(behavior).get("stage5") or {}
    active, sham = spec.get("active"), spec.get("sham")
    data = [str((spec.get("data") or {}).get(role, {}).get("id") or "")
            for role in ("suppress", "recipe")]  # fmt: skip
    if not active or not sham or not all(data):
        raise ValueError(f"{behavior}: plan.stage5 names no A, H and two pinned datasets")
    args = ["--state", active, "--state", sham]
    for data_id in data:  # read by id, never regenerated (pipeline.stage5_data): the dataset alone
        args += ["--generation", data_id]
    args += kstar_and_pools(behavior, active, sources)
    return {"behavior": behavior, "active": active, "args": args}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--active", action="store_true", help="A alone")
    mode.add_argument("--recipe-control", type=int, metavar="DOSE")
    mode.add_argument("--stage5-dpo", action="store_true", help="the DPO lineage's sessions")
    args = parser.parse_args(argv)
    try:
        if args.active:
            out = {"active": active_of(args.behavior)}
        elif args.recipe_control is not None:
            out = {"recipe": recipe_control(args.behavior, args.recipe_control, args.source)}
        elif args.stage5_dpo:
            out = stage5_dpo_inputs(args.behavior, list(args.source))
        else:
            out = inputs(args.behavior, list(args.source))
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
