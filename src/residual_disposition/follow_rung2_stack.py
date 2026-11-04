"""Rung 2 on eli5's validated stack candidate 800, on both decoders (preregistration §13.6, §13.7,
§13.16's point 4), as on eli5 continued 200 (``follow_rung2``): the same evidence, floors and
decision, from the candidate's build session (``manifests/follow_stack_build.json``).

- **Readings** (``readings``): B3's two recipe streams of the stack, frozen with their salts, and
  the rung's five analysis cells under ``follow_inventory.rung_analysis``'s salts, each through both
  decoders (``base_reading``'s salts). No validation-only stream: a rung's format is validated once,
  in the first mode to reach it (B3's ``first_mode``), so the stack's recipe never enters that
  frame.
- **Qualification** (``qualification``): A and H qualify a rung once for both modes (§13.6), so
  their rates are the continued mode's committed qualification's; the recipe control is this mode's
  own.
- **Steering** (``steering_cells``, ``cell_vectors``): five cells, A and H already read; a built
  package's layer-20 difference, its method check's second path and its recorded seeds and norm
  (B3's ``ladder.rung2.steering``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import follow_inventory as fi
from . import follow_ladder as L
from . import follow_rung2 as R2
from .follow_rung2 import (  # noqa: F401  the interface scripts/follow_rung2.py reads
    BEHAVIOR,
    DECODERS,
    RANDOM_CELL,
    READER,
    decision,
    evidence,
)

MODE, DOSE = "stack", 800
ITEM = "follow/stack/rung2"
LEFT = "follow/stack/full"  # the stack path's item: the base decoder's readings spend its rest
WORKER = "stack-rung2-steering"
RECIPE = "recipe"
BUILD_RECORD = "manifests/follow_stack_build.json"
STEERING_RECORD = "manifests/follow_stack_rung2_steering.json"
SHARED_QUALIFICATION = "follow-rung2-qualification-20261101T141901Z"  # A and H of the rung


def built(repo: Path | None = None) -> dict[str, Any]:
    """The candidate's build session, by its committed record (read back after the check, since
    ``committed_json`` puts the file's own digest over the record's ``sha256``)."""
    from .manifests import committed_json
    from .paths import REPO_ROOT

    path = (repo or REPO_ROOT) / BUILD_RECORD
    committed_json(path, "the build session's record", repo)
    return json.loads(path.read_text())


def base_reading(row: Mapping[str, Any]) -> dict[str, Any]:
    """A reading through the base decoder: its salt specific to the decoder (§13.16)."""
    decoder = "base_random" if row["decoder"] == "random" else "base"
    return {**row, "salt": row["salt"].replace("follow/rung2/", "follow/rung2/base/", 1),
            "decoder": decoder}  # fmt: skip


def readings(record: Mapping[str, Any],
             repo: Path | None = None) -> dict[str, list[dict[str, Any]]]:  # fmt: skip
    """The rung's readings by phase, the state decoder's first, each with the state it reads; L with
    a random direction reads L's package through that decoder's random branch."""
    from . import continuation_rules as cr
    from .paths import REPO_ROOT

    path = (repo or REPO_ROOT) / cr.follow_b3()["inventory"]["manifest"]
    frozen = json.loads(path.read_text())["rung2"]["recipe_stack"]
    states = record["states"]

    def analysis(row: Mapping[str, Any]) -> dict[str, Any]:
        random = row["role"] == RANDOM_CELL
        return {**row, "package": {"state": states["L" if random else row["role"]]},
                "decoder": "random" if random else "state"}  # fmt: skip

    recipe = [{**r, "package": {"state": states[RECIPE]}} for r in frozen]
    cells = [analysis(r) for r in fi.rung_analysis(2, MODE, DOSE)]
    return {"qualification": [*recipe, *map(base_reading, recipe)],
            "analysis": [*cells, *map(base_reading, cells)]}  # fmt: skip


def item_of(reading: Mapping[str, Any]) -> str:
    """The meter item a reading is charged to, as on the continued mode (``follow_rung2.item_of``):
    the rung's item holds the state decoder's readings and the steering Pod; the second decoder's
    readings and both random cells spend the unspent money of the stack path's item."""
    return ITEM if reading["decoder"] == "state" else LEFT


def steering_cells(repo: Path | None = None) -> list[dict[str, Any]]:
    """The session's cells, L before its random cell; ``run`` names the cell's built state."""
    states = built(repo)["states"]
    cells = [{"cell": role, "run": states[role], "random": False}
             for role in (RECIPE, "L", "H_s", "X")]  # fmt: skip
    return [*cells, {"cell": RANDOM_CELL, "run": states["L"], "random": True}]


def source_problems(build: Path, cells: Sequence[Mapping[str, Any]],
                    repo: Path | None = None) -> list[str]:  # fmt: skip
    """Why the build directory is not the committed session's: its result at its recorded sha256,
    and each cell's two layer-20 files as that result records them."""
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if sha(build / "result.json") != built(repo)["sha256"]["build"]:
        return ["the build's result is not its committed record"]
    files = json.loads((build / "result.json").read_text())["files"]
    layer = int(L.STEERING_LAYER)
    names = [f"builds/{c['run']}/{kind}-layer{layer}.safetensors" for c in cells
             for kind in ("production", "qualification")]  # fmt: skip
    return [f"{name} is not its recorded file" for name in names
            if sha(build / name) != files[name]]  # fmt: skip


def cell_vectors(directory: Path, *, random: bool) -> tuple[Any, float, int]:
    """A cell's vectors per position, η and its package's seed, from its build: δ̄ the production
    path's difference, checked against the qualification path's to 2⁻⁸ (B3); for L's random cell the
    seeded directions of its D* random branch, by the seed its build records."""
    from safetensors.torch import load_file

    from .adl_lens import check_reproduction, random_directions

    layer = int(L.STEERING_LAYER)
    record = json.loads((directory / "build.json").read_text())
    vectors = load_file(str(directory / f"production-layer{layer}.safetensors"))["tensor"]
    stored = load_file(str(directory / f"qualification-layer{layer}.safetensors"))["tensor"]
    check = check_reproduction(vectors, stored)
    if not check["passed"]:
        raise ValueError(f"{directory.name}: δ̄ is not the build's checked difference: {check}")
    if random:
        vectors = random_directions(vectors, int(record["seeds"]["random"][str(layer)]))
    return vectors, float(record["norms"][str(layer)]), int(record["seeds"]["package"])


def steering(find_run: Callable[[str], Any], repo: Path | None = None) -> dict[str, dict[str, Any]]:
    """Each cell's samples, bound by this mode's committed record."""
    return R2.steering(find_run, repo, STEERING_RECORD)


def shared_rates(runs: Path | None = None) -> dict[tuple, Fraction]:
    """A and H of the rung on each decoder, from the continued mode's qualification record."""
    from .paths import runs_root

    path = (runs or runs_root()) / "follow_rung2" / SHARED_QUALIFICATION / "qualification.json"
    floors = json.loads(path.read_text())["floors"]
    return {(READER, BEHAVIOR, role, d): Fraction(floors[d]["rates"][role])
            for d in DECODERS for role in ("A", "H")}  # fmt: skip


def qualification(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
                  runs: Path | None = None) -> dict[str, Any]:  # fmt: skip
    """Each decoder's floors: the shared A and H, this mode's recipe control."""
    return R2.qualification(rows, rules, recipe=RECIPE, shared=shared_rates(runs))
