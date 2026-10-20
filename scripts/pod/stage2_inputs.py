"""What a Stage-2 lineage's first session reads, when its behavior's pilot cells are in other
spaces (pun_intended's are in b, from 2026-10-13; session p1 of continuation_stage2.sh), named
from the behavior's experiment config and found in the source spaces, read only. Prints one JSON
line for copy_inputs.py:

    {"active": A, "args": [--state ..., --run ..., --run-name ..., --data ..., --index ...,
     --prompts, --path data/prompts/<pool> ...]}

    python scripts/pod/stage2_inputs.py --behavior pun_intended --source /workspace/b \
        --source /workspace/a

- the states: A (``plan.reread.controls``, its one active organism), H (``plan.screen.sham``),
  and L and H_s at each deep dose (``plan.screen.doses``); copy_inputs adds their adapters'
  states;
- each deep-dose state's one completed exploratory pass (a rates cell of measurement observer on
  the certificate prompts), which the screening re-scores and never draws again;
- A's approved relearning reference (``continuation_kstar.PILOT_REFERENCES``) and b's relearning
  data (the behavior's completed data_relearn cell of relearn_train, seed 0, from the base
  model) for k*;
- the analyst index (feasibility_14b, stage1_14b, continuation_14b), the prompt pools, and by
  name the confirmation pool and the OOD build, which may be only in a.

Exits 1 on a missing or ambiguous input. Nothing here proves an input valid: the plan checks do
that in the target space before any GPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

INDEX = ("feasibility_14b", "stage1_14b", "continuation_14b")


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


def inputs(behavior: str, sources: list[Path]) -> dict:
    from omegaconf import OmegaConf

    from residual_disposition import continuation_confirm as C
    from residual_disposition import pipeline
    from residual_disposition.config import load_config
    from residual_disposition.continuation_kstar import PILOT_REFERENCES

    cfg = load_config(f"experiments/continuation_14b/{behavior}")
    plan = OmegaConf.to_container(cfg.pipeline.plan, resolve=False)
    controls = list((plan.get("reread") or {}).get("controls") or [])
    if len(controls) != 1:
        raise ValueError(f"{behavior}: plan.reread.controls names {len(controls)} states, not A")
    active, sham = controls[0], (plan.get("screen") or {}).get("sham")
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
    if active not in PILOT_REFERENCES:
        raise ValueError(f"{active}: no approved relearning reference")
    args += ["--run-name", PILOT_REFERENCES[active]]
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
    return {"behavior": behavior, "active": active, "args": args}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    try:
        out = inputs(args.behavior, list(args.source))
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
