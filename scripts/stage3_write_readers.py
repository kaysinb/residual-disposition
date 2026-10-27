"""The Stage-3 readers' manifests of the four panel lineages (``continuation_readers``), on CPU,
from the bound readings in ``SOURCES``:

    uv run python scripts/stage3_write_readers.py check
    uv run python scripts/stage3_write_readers.py write
    uv run python scripts/stage3_write_readers.py verify

- ``check``: the preconditions: a clean tree at ``origin/p2-continuation``; every source run
  present; each lineage's PD-clean reviews imported, its judge history committed, its states'
  manifests present and no readers' manifest yet; each registered completion's provenance
  committed (``continuation_completion.REGISTERED``).
- ``write``: the check, then each lineage's ``manifests/stage3_readers_<lineage>.json``, once
  (``write_readers``). Commit them.
- ``verify``: each committed manifest recomputed from its sources (``committed_readers``).

Only gates, p values and sha256s are printed, never a count, a label or a text.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any

from residual_disposition import continuation_completion as C
from residual_disposition import continuation_readers as E
from residual_disposition import continuation_rules as cr
from residual_disposition.continuation_kstar import find_run
from residual_disposition.continuation_panel import readers_path
from residual_disposition.manifests import committed_json
from residual_disposition.paths import REPO_ROOT, data_root
from residual_disposition.simulate_continuation import FROZEN

WORKERS = 4  # the exact tests' processes, as the panel's verifier runs them
GATES = ("eligibility", "fidelity", "category_confusion", "stability")
# each lineage's readings (``continuation_readers.SOURCES``): the readers' record, the complete
# lens and PD interpreter runs, their judges and the category grades of every counted hypothesis
# (eli5's with its completed blocked strict labels)
SOURCES = {
    "word_count_claim": {
        "record": "20261025-172246-stage3_readers-6c001b28-4127",
        "lens_reading": "20261026-180811-b30771cb-e55f",
        "lens_judge": "20261026-181015-1950cd0b-8764",
        "pd_reading": "20261026-180500-4c655d8d-0242",
        "pd_judge": "20261026-181223-9b7af8fd-71fa",
        "category": "20261026-181408-category-0c199500-63c6",
    },
    "eli5": {
        "record": "20261025-194402-stage3_readers-c84d0a5a-cd45",
        "lens_reading": "20261026-181504-ce774c53-52c4",
        "lens_judge": "20261026-184730-10e76afb-0847",
        "pd_reading": "20261026-184107-e036dd69-b0c3",
        "pd_judge": "20261026-184522-7d72c48f-bbe8",
        "category": "20261027-103603-category-de737fac-395d",
    },
    "tldr": {
        "record": "20261026-143023-stage3_readers-f3d16e79-833b",
        "lens_reading": "20261026-185112-70ff9a93-de64",
        "lens_judge": "20261026-192458-776a287a-9c11",
        "pd_reading": "20261026-191507-2c7968c3-a488",
        "pd_judge": "20261026-192601-a423b4f8-1f3d",
        "category": "20261026-192729-category-13e60ea3-b9fa",
    },
    "countdown": {
        "record": "20261026-180855-stage3_readers-4c0b43ea-a59d",
        "lens_reading": "20261027-040222-79109a4c-9b39",
        "lens_judge": "20261027-040849-78f8e98f-75c7",
        "pd_reading": "20261027-060015-1f453a0e-2a7e",
        "pd_judge": "20261027-065523-1cc6d2c9-78fe",
        "category": "20261027-070500-category-68557cac-2a37",
    },
}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip


def check(rules: Any) -> tuple[list[str], dict[str, E.Lineage]]:
    """The preconditions' problems (none when they hold) and each behavior's lineage."""
    problems, lineages = [], {}
    if git("status", "--porcelain", "--untracked-files=no"):
        problems.append("the tracked tree is not clean")
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/p2-continuation"):
        problems.append("HEAD is not origin/p2-continuation")
    for behavior, sources in SOURCES.items():
        if tuple(sources) != E.SOURCES:
            problems.append(f"{behavior}: sources are not {E.SOURCES}")
        lineage = lineages[behavior] = E.lineage_of(REPO_ROOT, behavior, rules, find_run)
        name = lineage.identity["lineage"]
        if readers_path(REPO_ROOT, name).exists():
            problems.append(f"{name}: a readers' manifest exists")
        problems += [f"{behavior}: run {s} is not here" for s in sources.values()
                     if find_run(s) is None]  # fmt: skip
        reviews = sorted((data_root() / "pd_clean_stage3" / name).glob("review*"))
        if not reviews or any(not (d / "labels.json").is_file() for d in reviews):
            problems.append(f"{name}: a PD-clean review is not imported")
        try:
            committed_json(E.history_path(REPO_ROOT, name), "the judge history", REPO_ROOT)
        except (OSError, RuntimeError, ValueError) as error:
            problems.append(f"{name}: judge history: {type(error).__name__}")
        missing = [r for r, sid in lineage.identity["states"].items()
                   if not (data_root() / "states" / sid / "manifest.json").is_file()]  # fmt: skip
        if missing:
            problems.append(f"{name}: no state manifest for {missing}")
        print(f"{behavior}: {name}, reviews {[d.name for d in reviews]}, "
              f"states {len(lineage.identity['states'])}")  # fmt: skip
    for name, registered in C.REGISTERED.items():
        try:
            C.committed_spec(name)
            committed_json(C.provenance_path(REPO_ROOT, name), "the readers' verdicts", REPO_ROOT)
        except (OSError, RuntimeError, ValueError) as error:
            problems.append(f"{name}: completion: {type(error).__name__}")
        if lineages[registered["behavior"]].identity["lineage"] != name:
            problems.append(f"{name}: not the {registered['behavior']} lineage")
    return problems, lineages


def gates(data: dict[str, Any]) -> dict[str, Any]:
    """A verdict's gates, its PD positive control and each draw's verdict and p."""
    draws = data["component3"]["draws"]
    return {**{g: data[g]["passed"] for g in GATES},
            "pd_positive_control": data["eligibility"]["pd_positive_control"]["passed"],
            "draws": {d: {"verdict": v["verdict"], "p": v["p"]} for d, v in draws.items()},
            "p3": data["p3"], "p4": data["p4"]}  # fmt: skip


def summary(data: dict[str, Any]) -> dict[str, Any]:
    if data["status"] != "complete":
        return {"status": data["status"], "reasons": data.get("reasons"),
                "p3": data.get("p3"), "p4": data.get("p4")}  # fmt: skip
    out = {"status": data["status"], **gates(data)}
    if data.get("zero_count_sensitivity"):
        out["zero_count_sensitivity"] = gates(data["zero_count_sensitivity"])
    if data.get("blocked_completion"):
        out["blocked_completion"] = {k: data["blocked_completion"].get(k)
                                     for k in ("spec", "provenance", "validation")}  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("check", "write", "verify"))
    args = parser.parse_args(argv)
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    if args.command == "verify":
        for behavior in SOURCES:
            lineage = E.lineage_of(REPO_ROOT, behavior, rules, find_run)
            path = readers_path(REPO_ROOT, lineage.identity["lineage"])
            data = E.committed_readers(path, REPO_ROOT, rules, find_run,
                                       lineage_for=lambda _, x=lineage: x,
                                       workers=WORKERS)  # fmt: skip
            print(json.dumps({behavior: {"verified": True, "sha256": data["sha256"]}}))
        return 0
    problems, lineages = check(rules)
    print(json.dumps({"preconditions": problems or "pass"}))
    if problems or args.command == "check":
        return 1 if problems else 0
    for behavior, lineage in lineages.items():
        data = E.write_readers(REPO_ROOT, lineage, SOURCES[behavior], rules, find_run,
                               workers=WORKERS)  # fmt: skip
        print(json.dumps({behavior: summary(data)}))
    print("Commit the readers' manifests, then run verify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
