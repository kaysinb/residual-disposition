"""Copy a Pod's inputs from one space of the network volume into another, create-only and checked
(C8 item B): Pod B writes only to its own space (/workspace/c) while Pod A writes to /workspace/a,
so that no file is written from two Pods (Runpod: concurrent writes to one network volume can
corrupt it). Pod B reads what it needs from a once, here, and never writes to a.

    python scripts/pod/copy_inputs.py --source /workspace/a --target /workspace/c \
        [--state ID ...] [--run EXPERIMENT/RUN ...] [--index EXPERIMENT ...] \
        [--generation DATA_ID ...] [--prompts] [--path data/REL | runs/REL ...] --log FILE

- ``--state``: a state's directory, with every state its adapters point at (``state:<id>/...``),
  recursively;
- ``--run``: one run directory; ``--index``: every run of an experiment that holds analyst-index
  records (``observations/state_index.jsonl``), for the plan checks;
- ``--generation``: a dataset directory (``data/generations/<data id>``); ``--prompts``: the
  prompt pools (``data/prompts``); ``--path``: another file or directory under data/ or runs/.

A target file that exists must be identical (a rerun resumes); one that differs is refused.
Every copied file is checked by sha256 and listed in the log (JSON lines). Exits 1 on a refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_closure(source: Path, state_ids: list[str]) -> list[str]:
    """The states and every state their adapters point at, in discovery order."""
    seen: list[str] = []
    todo = list(state_ids)
    while todo:
        sid = todo.pop(0)
        if sid in seen:
            continue
        manifest = source / "data" / "states" / sid / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"state {sid} is not in {source}")
        seen.append(sid)
        for ref in json.loads(manifest.read_text()).get("adapters", []):
            kind, _, rest = str(ref.get("location", "")).partition(":")
            if kind == "state":
                todo.append(rest.partition("/")[0])
    return seen


def indexed_runs(source: Path, experiment: str) -> list[str]:
    root = source / "runs" / experiment
    return sorted(
        f"{experiment}/{d.name}"
        for d in root.iterdir()
        if (d / "observations" / "state_index.jsonl").is_file()
    )


def copy_tree(src: Path, dst: Path, log) -> tuple[int, int]:
    """Copy ``src`` (a file or a directory) to ``dst`` create-only; returns (copied, kept)."""
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist")
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
    copied = kept = 0
    for f in files:
        target = dst if src.is_file() else dst / f.relative_to(src)
        want = sha256(f)
        if target.exists():
            if sha256(target) != want:
                raise FileExistsError(f"{target} exists and differs from {f}")
            kept += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        shutil.copy2(f, partial)
        if sha256(partial) != want:
            partial.unlink()
            raise OSError(f"{f}: the copy's sha256 differs")
        partial.rename(target)
        log.write(json.dumps({"path": str(target), "sha256": want}) + "\n")
        copied += 1
    return copied, kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--state", action="append", default=[])
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--index", action="append", default=[])
    parser.add_argument("--generation", action="append", default=[])
    parser.add_argument("--prompts", action="store_true")
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.source.resolve() == args.target.resolve():
        raise SystemExit("the source and the target are one space")
    items: list[str] = []
    try:
        items += [f"data/states/{s}" for s in state_closure(args.source, args.state)]
        items += [f"runs/{r}" for r in args.run]
        for experiment in args.index:
            items += [f"runs/{r}" for r in indexed_runs(args.source, experiment)]
        items += [f"data/generations/{g}" for g in args.generation]
        if args.prompts:
            items.append("data/prompts")
        for rel in args.path:
            if rel.split("/", 1)[0] not in ("data", "runs") or ".." in Path(rel).parts:
                raise ValueError(f"--path {rel}: a path under data/ or runs/")
            items.append(rel)
        totals = {"copied": 0, "kept": 0}
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "a") as log:
            for rel in dict.fromkeys(items):
                copied, kept = copy_tree(args.source / rel, args.target / rel, log)
                totals["copied"] += copied
                totals["kept"] += kept
    except (OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1
    print(json.dumps({"ok": True, "items": len(dict.fromkeys(items)), **totals}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
