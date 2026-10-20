"""Copy a Pod's inputs from other spaces of the network volume into its own, create-only and
checked (C8 item B): Pod A writes only to its space (/workspace/a), Pod B only to its own
(/workspace/c), so that no file is written from two Pods (Runpod: concurrent writes to one
network volume can corrupt it). A Pod reads what it needs from the other's space and from
/workspace/b (Stage 1's second space, read only), here, and never writes there (reviews s4b,
s4f): Pod B the screening's states and A1's k* and Y₂ evidence from a, Pod A the Y organism and
data in c, both the stage-1 recipe run and the Y₂ selection's laptop inputs from b.

    python scripts/pod/copy_inputs.py --source /workspace/a [--source /workspace/b] \
        --target /workspace/c \
        [--state ID ...] [--run EXPERIMENT/RUN ...] [--run-name RUN ...] [--index EXPERIMENT ...] \
        [--generation DATA_ID ...] [--data DATA_ID ...] [--prompts] \
        [--path data/REL | runs/REL ...] [--certificate LINEAGE [--manifests DIR]] --log FILE

- ``--state``: a state's directory, with every state its adapters point at (``state:<id>/...``),
  recursively;
- ``--run``: one run directory; ``--run-name``: the run of that name under any experiment;
  ``--index``: every run of an experiment that holds analyst-index records
  (``observations/state_index.jsonl``), for the plan checks;
- ``--generation``: a dataset directory (``data/generations/<data id>``); ``--data``: a dataset
  with every completed data cell that made it (``*-data_*-*`` runs whose summary names it), which
  ``plain_dataset(cached_only=True)`` needs; ``--prompts``: the prompt pools (``data/prompts``);
  ``--path``: another file or directory under data/ or runs/;
- ``--certificate``: everything the lineage's certificate reads, named by its committed k* and
  Y₂ manifests in ``--manifests`` (the snapshot's ``manifests/``;
  ``continuation_certificate.dependencies``).

An item is taken from the first source that holds it, in the order given; one no source holds
must already be in the target (the Pod's own work), else the copy is refused. ``--index`` reads
the first source that has the experiment. A target file that exists must be identical (a
rerun resumes); one that differs is refused. Every copied file is checked by sha256 and listed in
the log (JSON lines). Exits 1 on a refusal. Nothing here proves an input valid: the certificate's
CPU preflight (``pipeline.certificate_problems``) checks them in the target before any GPU work.
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


def held(spaces: tuple[Path, ...], rel: str) -> Path:
    """Where ``rel`` is held: the first source, in order, else the target (the Pod's own); in
    none is refused."""
    for space in spaces:
        if (space / rel).exists():
            return space / rel
    raise FileNotFoundError(f"{rel} is in neither {' nor '.join(str(x) for x in spaces)}")


def state_closure(spaces: tuple[Path, ...], state_ids: list[str]) -> list[str]:
    """The states and every state their adapters point at, in discovery order."""
    seen: list[str] = []
    todo = list(state_ids)
    while todo:
        sid = todo.pop(0)
        if sid in seen:
            continue
        manifest = held(spaces, f"data/states/{sid}") / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"state {sid} has no manifest in {manifest.parent}")
        seen.append(sid)
        for ref in json.loads(manifest.read_text()).get("adapters", []):
            kind, _, rest = str(ref.get("location", "")).partition(":")
            if kind == "state":
                todo.append(rest.partition("/")[0])
    return seen


def named_run(spaces: tuple[Path, ...], name: str) -> str:
    """``runs/<experiment>/<name>``: the run of that name in the first space holding it."""
    for space in spaces:
        found = sorted(d for d in (space / "runs").glob(f"*/{name}") if d.is_dir())
        if len(found) > 1:
            raise ValueError(f"run {name} under {len(found)} experiments of {space}")
        if found:
            return found[0].relative_to(space).as_posix()
    raise FileNotFoundError(f"run {name} is in neither {' nor '.join(str(x) for x in spaces)}")


def data_cells(spaces: tuple[Path, ...], data_id: str) -> list[str]:
    """The dataset and the completed data cells that made it, of the first space that has them."""
    for space in spaces:
        cells = []
        for summary in sorted((space / "runs").glob("*/*-data_*-*/summary.json")):
            meta = summary.parent / "meta.json"
            if (
                meta.is_file()
                and json.loads(meta.read_text()).get("status") == "completed"
                and json.loads(summary.read_text()).get("data_id") == data_id
            ):
                cells.append(summary.parent.relative_to(space).as_posix())
        if cells:
            return [f"data/generations/{data_id}", *cells]
    where = " or ".join(str(x) for x in spaces)
    raise FileNotFoundError(f"no completed data cell of {data_id} in {where}")


def certificate_items(spaces: tuple[Path, ...], manifests: Path, lineage: str) -> list[str]:
    """Everything the lineage's certificate reads (``continuation_certificate.dependencies``)."""
    from residual_disposition.continuation_certificate import dependencies

    kstar = json.loads((manifests / f"kstar_{lineage}.json").read_text())
    y2 = json.loads((manifests / "y2.json").read_text())
    deps = dependencies(kstar, y2)
    items = [f"data/states/{s}" for s in state_closure(spaces, deps["states"])]
    items += [named_run(spaces, name) for name in deps["runs"]]
    for data_id in deps["data"]:
        items += data_cells(spaces, data_id)
    for loc in deps["locations"]:
        if loc.get("root") not in ("data", "runs") or ".." in Path(loc["path"]).parts:
            raise ValueError(f"not a recorded location: {loc}")
        items.append(f"{loc['root']}/{loc['path']}")
    return items


def indexed_runs(sources: tuple[Path, ...], experiment: str) -> list[str]:
    """``<experiment>/<run>`` of every completed indexed run of the first source that has the
    experiment: a run still running (another Pod's live work) is never copied (review s4z 2)."""
    roots = [x / "runs" / experiment for x in sources if (x / "runs" / experiment).is_dir()]
    if not roots:
        raise FileNotFoundError(f"no source has runs/{experiment}")
    root = roots[0]
    return sorted(
        f"{experiment}/{d.name}"
        for d in root.iterdir()
        if (d / "observations" / "state_index.jsonl").is_file() and completed(d)
    )


def completed(run: Path) -> bool:
    meta = run / "meta.json"
    return meta.is_file() and json.loads(meta.read_text()).get("status") == "completed"


def copy_tree(src: Path, dst: Path, log) -> tuple[int, int]:
    """Copy ``src`` (a file or a directory) to ``dst`` create-only; returns (copied, kept)."""
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist")
    if src == dst:
        return 0, 0  # the target's own
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
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--state", action="append", default=[])
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--run-name", action="append", default=[])
    parser.add_argument("--index", action="append", default=[])
    parser.add_argument("--generation", action="append", default=[])
    parser.add_argument("--data", action="append", default=[])
    parser.add_argument("--prompts", action="store_true")
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--certificate", default=None, metavar="LINEAGE")
    parser.add_argument("--manifests", type=Path, default=Path(__file__).absolute().parents[2]
                        / "manifests", help="the snapshot's committed manifests")  # fmt: skip
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(x.resolve() == args.target.resolve() for x in args.source):
        raise SystemExit("the source and the target are one space")
    spaces = (*args.source, args.target)
    items: list[str] = []
    try:
        items += [f"data/states/{s}" for s in state_closure(spaces, args.state)]
        items += [f"runs/{r}" for r in args.run]
        items += [named_run(spaces, name) for name in args.run_name]
        for experiment in args.index:
            items += [f"runs/{r}" for r in indexed_runs(tuple(args.source), experiment)]
        items += [f"data/generations/{g}" for g in args.generation]
        for data_id in args.data:
            items += data_cells(spaces, data_id)
        if args.prompts:
            items.append("data/prompts")
        for rel in args.path:
            if rel.split("/", 1)[0] not in ("data", "runs") or ".." in Path(rel).parts:
                raise ValueError(f"--path {rel}: a path under data/ or runs/")
            items.append(rel)
        if args.certificate:
            items += certificate_items(spaces, args.manifests, args.certificate)
        totals = {"copied": 0, "kept": 0}
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "a") as log:
            for rel in dict.fromkeys(items):
                copied, kept = copy_tree(held(spaces, rel), args.target / rel, log)
                totals["copied"] += copied
                totals["kept"] += kept
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1
    print(json.dumps({"ok": True, "items": len(dict.fromkeys(items)), **totals}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
