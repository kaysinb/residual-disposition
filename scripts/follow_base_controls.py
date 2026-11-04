"""Record the base decoder's controls (preregistration §13.16) after their pull, once, as
manifests/follow_base_controls.json.

    uv run python scripts/follow_base_controls.py runs/follow_base_controls/<run> [--write]

The session's output (``scripts/pod/follow_base_controls.py``) must be complete and exactly the
planned files of each A package and of eli5 continued 200's L package. The record names the run
and, per package, its directory and every file's
sha256; the readings find the controls only through it (``follow_readings.base_controls``). Without
``--write`` it prints the record; ``--write`` creates it once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_build as B
from residual_disposition import follow_readings as D
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import REPO_ROOT


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(run: Path) -> dict:
    found = sorted(run.glob("out/*/result.json"))
    if len(found) != 1:
        raise SystemExit(f"{run}: {len(found)} session results, not one")
    out = found[0].parent
    layers = [int(x) for x in cr.follow_lrandom()["layers"]["read"]]
    result = json.loads(found[0].read_text())
    planned = {p["run"]: p for p in D.base_control_packages(find_run)}
    if result["status"] != "complete" or sorted(result["packages"]) != sorted(planned):
        raise SystemExit(f"{run}: not the planned packages, complete")
    packages = {}
    for name in sorted(planned):
        d = out / "packages" / name
        files = {str(p.relative_to(d)): sha(p) for p in sorted(d.rglob("*")) if p.is_file()}
        recorded = result["packages"][name]["files"]
        wanted = B.base_control_files(layers, planned[name]["branches"])
        if sorted(files) != wanted or files != recorded:
            raise SystemExit(f"{name}: not its recorded, complete files")
        packages[name] = {"behavior": planned[name]["behavior"], "dir": str(d.relative_to(run)),
                          "files": files}  # fmt: skip
    return {"run": run.name, "result_sha256": sha(found[0]), "packages": packages}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    made = record(args.run)
    if not args.write:
        print(json.dumps({k: v for k, v in made.items() if k != "packages"}))
        print(sorted(made["packages"]))
        return 0
    path = REPO_ROOT / D.BASE_CONTROLS
    if path.exists():
        raise SystemExit(f"{D.BASE_CONTROLS} exists: it is written once")
    path.write_text(json.dumps(made, indent=1, sort_keys=True) + "\n")
    print(f"wrote {D.BASE_CONTROLS} ({sha(path)}): commit it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
