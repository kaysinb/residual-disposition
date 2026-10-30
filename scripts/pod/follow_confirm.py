"""The follow-up's confirmations of example_link and pun_intended on one Pod (preregistration
§13.3; ``residual_disposition.follow_confirm``), in four steps, each its own command:

    python scripts/pod/follow_confirm.py fitness --volume /workspace
    python scripts/pod/follow_confirm.py precheck --space /workspace/a --source /workspace/b \\
        --log <file>
    python scripts/pod/follow_confirm.py run --space /workspace/a --out <new directory>
    python scripts/pod/follow_confirm.py status --space /workspace/a

- **fitness** (right after the start, before anything else): ``pod_fitness.check``; exit 3 if the
  Pod is unfit (it is stopped and replaced).
- **precheck** (CPU): ``RUNS_DIR``, ``DATA_DIR`` and ``HF_HOME`` must be the space's ``runs``,
  ``data`` and ``hf``. In the space: every state a line may draw (latent and sham_s at d₁ and
  d₂, eight) with every adapter against its recorded tensor sha256, the analyst index recording
  each as its line's stack suppression or sham at its dose, the frozen confirmation prompts with
  their set hash, and the model's snapshot at M0's revision with every file against its content
  address. A missing state, index run, prompt pool or model snapshot is copied from the sources
  in order, create-only (``copy_inputs.py``; ``follow_confirm.copy_model``), and everything is
  checked again; an item present but different is refused, never overwritten. Exit 1 on a
  refusal.
- **run** (GPU): the sources (``follow_confirm.load_spec``), then each line in the rules' order
  until it is decided (``follow_confirm.next_step``): one draw of the next attempt
  (``follow_confirm.plan``, ``pipeline._draw_confirmation``), its run on the space's volume
  (the allocated attempt and its spec before any generation, each state's whole draw before the
  next, the detector records and responses, then the summary), and the ledger read again from
  the volume. No model is downloaded (``HF_HUB_OFFLINE``). Any exception stops the session: no
  later draw of either line is made, and the attempt resumes in the next session as that very
  cell. ``<out>/result.json`` holds the lines' decision; exit 0 once both lines are decided, 2
  after a stop.
- **status** (CPU): the decision from the space's runs, as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(data, indent=1, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def space_problems(space: Path) -> list[str]:
    """Why the environment does not write into ``space`` only: the runs, data and HF caches."""
    want = {"RUNS_DIR": space / "runs", "DATA_DIR": space / "data", "HF_HOME": space / "hf"}
    problems = []
    for name, path in want.items():
        value = os.environ.get(name)
        if value is None or Path(value).resolve() != path.resolve():
            problems.append(f"{name} is {value}, not {path}")
    return problems


def fitness(args: argparse.Namespace) -> int:
    from residual_disposition import pod_fitness

    found = pod_fitness.check(args.volume)
    print(json.dumps(found))
    return 0 if found["fit"] else 3


def inputs_problems(spec: Any, space: Path) -> dict[str, Any]:
    """What the space lacks (to copy) and what it holds wrongly (refused)."""
    from residual_disposition import continuation_confirm as C
    from residual_disposition import follow_confirm as F
    from residual_disposition.states import StateStore

    store, wrong = StateStore(space / "data" / "states"), []
    states = [s for _, _, _, s in F.states_needed(spec)]
    missing_states = [s for s in states if not (store.root / s / "manifest.json").is_file()]
    for state in states:
        if state not in missing_states:
            try:
                store.verify(state)
            except (OSError, ValueError) as error:
                wrong.append(f"{state}: {error}")
    index, missing_index = F.index_problems(spec, space / "runs")
    wrong += index
    try:
        C.prompts(space / "data")
        prompts_missing = False
    except FileNotFoundError:
        prompts_missing = True
    except ValueError as error:
        wrong.append(f"prompts: {error}")
        prompts_missing = False
    model = spec.m0["runtime"]["model"]
    model_problems = F.model_cache_problems(space / "hf", model["name"], model["revision"])
    model_missing = bool(model_problems) and not F.model_snapshot(
        space / "hf", model["name"], model["revision"]).is_dir()  # fmt: skip
    if model_problems and not model_missing:
        wrong += model_problems
    return {"wrong": wrong, "states": missing_states, "index": missing_index,
            "prompts": prompts_missing, "model": model_missing}  # fmt: skip


def precheck(args: argparse.Namespace) -> int:
    from residual_disposition import follow_confirm as F

    space = args.space
    problems = space_problems(space)
    if problems:
        print(json.dumps({"ok": False, "problems": problems}))
        return 1
    spec = F.load_spec()
    found = inputs_problems(spec, space)
    if found["wrong"]:
        print(json.dumps({"ok": False, "problems": found["wrong"]}))
        return 1
    copied: dict[str, Any] = {}
    command = [sys.executable, str(HERE / "copy_inputs.py"),
               *(x for s in args.source for x in ("--source", str(s))),
               "--target", str(space), "--log", str(args.log)]  # fmt: skip
    extra: list[str] = []
    for state in found["states"]:
        extra += ["--state", state]
    if found["index"]:
        runs = {run for src in args.source
                for s, records in F.index_records(src / "runs").items() if s in found["index"]
                for run, _ in records}  # fmt: skip
        for run in sorted(runs):
            extra += ["--run", run]
    if found["prompts"]:
        extra += ["--path", "data/prompts/" + F.C.FROZEN["pool"]]
    if extra:
        done = subprocess.run(command + extra, capture_output=True, text=True)
        copied["inputs"] = {"exit": done.returncode, "stdout": done.stdout[-4000:],
                            "stderr": done.stderr[-4000:]}  # fmt: skip
        if done.returncode != 0:
            print(json.dumps({"ok": False, "copied": copied}))
            return 1
    if found["model"]:
        model = spec.m0["runtime"]["model"]
        held = [
            s / "hf"
            for s in args.source
            if F.model_snapshot(s / "hf", model["name"], model["revision"]).is_dir()
        ]
        if not held:
            print(json.dumps({"ok": False, "problems": ["no source holds the model's snapshot"]}))
            return 1
        copied["model"] = F.copy_model(held[0], space / "hf", model["name"], model["revision"])
    again = inputs_problems(spec, space)
    left = again["wrong"] + [f"{k} missing" for k in ("states", "index", "prompts", "model")
                             if again[k]]  # fmt: skip
    print(json.dumps({"ok": not left, "problems": left, "copied": copied,
                      "states": [s for _, _, _, s in F.states_needed(spec)]}))  # fmt: skip
    return 0 if not left else 1


def run(args: argparse.Namespace) -> int:
    problems = space_problems(args.space)
    if problems:
        print(json.dumps({"status": "refused", "problems": problems}))
        return 1
    os.environ["HF_HUB_OFFLINE"] = "1"  # the snapshot the precheck verified; nothing is fetched
    from residual_disposition import follow_confirm as F

    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    spec = F.load_spec()
    result: dict[str, Any] = {"status": "running", "draws": []}
    try:
        status = session(spec, args.space, result)
    except Exception as error:  # noqa: BLE001 — every failure stops the session, recorded
        result.update(status="stopped", error=repr(error), traceback=traceback.format_exc())
        status = 2
    result["decision"] = F.decision(spec, args.space / "runs", experiment_of(spec))
    if status == 0:
        result["status"] = "decided"
    write_json(out / "result.json", result)
    print(json.dumps({"status": result["status"], "confirmed": result["decision"]["confirmed"]}))
    return status


def experiment_of(spec: Any) -> str:
    return Path(spec.experiment(spec.lines[0])).parts[-2]


def workspace(spec: Any, behavior: str) -> Any:
    """The line's workspace: its experiment's configuration, without override."""
    from residual_disposition import pipeline as P
    from residual_disposition.config import load_config

    return P.Workspace(load_config(spec.experiment(behavior)))


def session(spec: Any, space: Path, result: dict[str, Any], make: Any = workspace) -> int:
    """Each line in order until it is decided; a draw per attempt; any exception propagates.
    The lines share one engine (the same model and settings)."""
    from residual_disposition import follow_confirm as F
    from residual_disposition import pipeline as P

    experiment, previous = experiment_of(spec), None
    for behavior in spec.lines:
        ws = make(spec, behavior)
        if previous is not None and previous._generator is not None:
            if (ws.model_cfg, ws.cfg.pipeline.vllm) != (previous.model_cfg,
                                                         previous.cfg.pipeline.vllm):  # fmt: skip
                raise ValueError(f"{behavior}: another model or engine than the previous line's")
            ws._generator = previous._generator
        drawn: set[int] = set()
        while True:
            found, open_runs = F.draws(spec, behavior, space / "runs", experiment)
            step = F.next_step(spec, behavior, found)
            if step["status"] != F.DRAW:
                break
            if step["attempt"] in drawn:
                raise RuntimeError(f"{behavior}: attempt {step['attempt']} is drawn again")
            c = F.plan(ws, spec, behavior, step, open_runs,
                       F.recorded_cells(space / "runs", experiment))  # fmt: skip
            outputs: dict[str, Any] = {}
            P._draw_confirmation(ws, c, None, outputs, measurement=spec.measurement,
                                 stream=spec.stream)  # fmt: skip
            drawn.add(step["attempt"])
            result["draws"].append({"behavior": behavior, **outputs["confirm"]})
        previous = ws
    return 0


def status(args: argparse.Namespace) -> int:
    from residual_disposition import follow_confirm as F

    spec = F.load_spec()
    print(json.dumps(F.decision(spec, args.space / "runs", experiment_of(spec)), default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fitness")
    p.add_argument("--volume", type=Path, required=True)
    p = sub.add_parser("precheck")
    p.add_argument("--space", type=Path, required=True)
    p.add_argument("--source", type=Path, action="append", default=[])
    p.add_argument("--log", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--space", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("status")
    p.add_argument("--space", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"fitness": fitness, "precheck": precheck, "run": run, "status": status}[args.command](
        args
    )


if __name__ == "__main__":
    sys.exit(main())
