"""Rung 2 of the full-threshold ladder on eli5 continued 200, on both decoders (preregistration
§13.6, §13.16's point 4), on the API from recorded packages and one Pod's steering samples.

- **Readings** (``readings``): B3's rung-2 inventory, the state decoder's readings under their
  frozen salts, and the same readings through the base decoder under their own salts
  (``base_reading``):
  A, H and the recipe control, two streams of 80, the qualification; L, H_s, X_cont, the recipe
  control and L with a random direction, one stream each, the analysis; streams 3 to 6 of A, H and
  the recipe control, validation-only, each generated whole only if the format's frame is short.
- **Steering** (``steering_cells``, ``execute_steering``): one session steers each package once
  (seven cells: A, H, the recipe control, L, H_s, X_cont and L with a random direction) by §11.1's
  procedure (``adl.position_steering``); both decoders read the same samples, since the vector comes
  from the activations and not from a decoder. The random cell keeps L's unsteered samples. The
  samples are bound by their committed record (``steering``).
- **Evidence** (``evidence``): ``follow_ladder.render_rung2`` of the reading's package, decoder and
  steering cell; the requests and labels are D*'s, with rung 2's prompt.
- **Qualification** (``qualification``): each decoder's floors, pooled over its two streams. The
  recipe floor is measured and limits how a negative result is read; it does not close a positive
  contrast (§13.16). A decoder failing a floor is ineligible: its L, H_s and X are not read.
- **Decision** (``decision``): per decoder its own test at 1/60 (§13.5 (c)), gated by its
  qualification, the rung's one format validation, category confusion (§13.5 (b)) and its random
  cell at most 8 of 80. Every launched test is completed and reported; a pending slot leaves its
  decoder pending, never a negative. No decoder is preferred, and nothing is pooled across them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import follow_ladder as L
from . import follow_qualification as Q
from . import follow_readings as D

BEHAVIOR, MODE, DOSE = "eli5", "continued", 200
DECODERS = ("state", "base")
RANDOM_CELLS = {"state": "random", "base": "base_random"}
READER = "rung2"
RECIPE = "recipe_cont"  # the continued mode's recipe control, as a qualification role
STEERING_RECORD = "manifests/follow_rung2_steering.json"
# a reading's steering cell: the recipe control's package is one, qualified and analysed
STEERING_CELLS = {"A": "A", "H": "H", RECIPE: "recipe", "recipe": "recipe", "L": "L", "H_s": "H_s",
                  "X": "X", "L_random": "L_random"}  # fmt: skip
RANDOM_CELL = "L_random"
LEVEL = D.RUNG_LEVEL
RANDOM_MAX = D.RANDOM_MAX
SHARED, RUNG, LEFT = "follow/rung2_shared", "follow/continued/rung2", "follow/base"


def base_reading(row: Mapping[str, Any]) -> dict[str, Any]:
    """A state-decoder reading through the base decoder: its salt specific to the decoder (§13.16);
    the random cell the base decoder's random branch of L's package, from its committed record."""
    out = {**row, "salt": row["salt"].replace("follow/rung2/", "follow/rung2/base/", 1)}
    if row["role"] == RANDOM_CELL:
        run = row["package"]["new"].removeprefix("random branch of ")
        return {**out, "decoder": "base_random", "package": {"base_control": run}}
    return {**out, "decoder": "base"}


def readings(repo: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Rung 2's readings by phase, each decoder's in the inventory's order, the state decoder's
    first: ``qualification``, ``analysis``, and ``validation`` (by stream, then role)."""
    from . import continuation_rules as cr
    from .paths import REPO_ROOT

    path = (repo or REPO_ROOT) / cr.follow_b3()["inventory"]["manifest"]
    inventory = json.loads(path.read_text())["rung2"]

    def both(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [*(dict(r) for r in rows), *(base_reading(r) for r in rows)]

    extra = [*inventory["validation_shared"], *inventory["validation_recipe_continued"]]
    order = {"A": 0, "H": 1, RECIPE: 2}
    ordered = sorted(extra, key=lambda r: (r["stream"], order[r["role"]]))
    return {"qualification": both([*inventory["shared"], *inventory["recipe_continued"]]),
            "analysis": both(inventory["analysis"]), "validation": both(ordered)}  # fmt: skip


def stream_of(rows: Sequence[Mapping[str, Any]], stream: int) -> list[dict[str, Any]]:
    """One validation-only stream of both decoders: the supplement is generated a stream at a time,
    3 to 6 in order, A, H and the recipe control alike (§13.16's point 6)."""
    return [dict(r) for r in rows if r["stream"] == stream]


def item_of(reading: Mapping[str, Any]) -> str:
    """The meter item a reading is charged to: the shared item holds the state decoder's A and H,
    the rung's item the state decoder's analysis and the Pod; the rest of the rung, the second
    decoder and the supplement among it, spends the unspent API of the step before it."""
    if reading["decoder"] in ("state", "random"):
        if reading["purpose"] == "qualification" and reading["role"] in ("A", "H"):
            return SHARED
        if reading["purpose"] == "analysis":
            return RUNG
    return LEFT


# ---------------------------------------------------------------------- the steering session


def steering_cells(repo: Path | None = None) -> list[dict[str, Any]]:
    """The session's cells in order, L before its random cell: each package once, with its vector's
    pinned sources (B3's inventory)."""
    rows = readings(repo)
    wanted = [r for r in [*rows["qualification"], *rows["analysis"]]
              if r["decoder"] in ("state", "random")]  # fmt: skip
    out: dict[str, dict[str, Any]] = {}
    for r in wanted:
        cell = STEERING_CELLS[r["role"]]
        random = cell == RANDOM_CELL
        run = (r["package"]["new"].removeprefix("random branch of ") if random
               else r["package"]["run"])  # fmt: skip
        entry = {"cell": cell, "run": run, "vector": r["vector"], "random": random}
        if out.setdefault(cell, entry) != entry:
            raise ValueError(f"{cell}: two packages or vectors in the inventory")
    if out[RANDOM_CELL]["run"] != out["L"]["run"]:
        raise ValueError("the random cell is not of L's package")
    return sorted(out.values(), key=lambda c: c["random"])


def source_problems(cell: Mapping[str, Any], run: Path, base: Path) -> list[str]:
    """Why a cell's vector sources are not their pinned files (B3's inventory)."""
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    want = cell["vector"]
    out = [f"{cell['cell']}: {name} is not its pinned file" for name, value in want["files"].items()
           if sha(run / name) != value]  # fmt: skip
    if sha(base / "base_means.safetensors") != want["base_means"]["sha256"]:
        out.append(f"{cell['cell']}: the base means are not their pinned file")
    return out


def cell_vectors(run: Path, base: Path, *, random: bool, layer: int = int(L.STEERING_LAYER)
                 ) -> tuple[Any, float, int]:  # fmt: skip
    """A cell's vectors per position, η and its package's seed, from the pinned files: δ̄ the
    package's state means minus the shared base means, checked against the package's stored
    difference to 2⁻⁸ (B3); for L's random cell the seeded directions of its D* random branch, by
    the seed its package records."""
    from safetensors.torch import load_file

    from .adl_lens import check_reproduction, random_directions
    from .follow_build import pinned_sources
    from .seeding import derive_seed

    pinned = pinned_sources(run, base, [layer])[layer]
    seed = int(json.loads((run / "adl_lens_package.json").read_text())["seed"])
    vectors = pinned["delta"]
    stored = load_file(str(run / "reproduction" / f"delta_layer{layer}.safetensors"))["delta"]
    check = check_reproduction(vectors, stored)
    if not check["passed"]:
        raise ValueError(f"{run.name}: δ̄ is not the package's stored difference: {check}")
    if random:
        vectors = random_directions(vectors, derive_seed(seed, "adl_lens", layer, "random"))
    return vectors, float(pinned["norm"]), seed


def steer_cell(model: Any, tokenizer: Any, vectors: Any, norm: float, *, seed: int,
               chat_kwargs: Mapping[str, Any], grade_coherent: Any,
               store: Any) -> dict[str, Any]:  # fmt: skip
    """A cell's samples at every position by §11.1's procedure (``adl.position_steering``)."""
    from . import adl

    layer = int(L.STEERING_LAYER)
    prompts = adl.steering_prompts()
    per = {str(p): adl.position_steering(
        model, tokenizer, vector=adl.steering_vector(vectors[j], norm), position=p, layer=layer,
        chat_kwargs=chat_kwargs, grade_coherent=grade_coherent, seed=seed, store=store,
        final_prompts=prompts) for j, p in enumerate(adl.POSITIONS)}  # fmt: skip
    return {"layer": layer, "norm": norm, "seed": seed, "per_position": per}


def execute_steering(out: Path, manifest: Mapping[str, Any], cells: Sequence[Mapping[str, Any]],
                     steer: Callable[[Mapping[str, Any], Any], Mapping[str, Any]],
                     positions: Sequence[int]) -> dict[str, Any]:  # fmt: skip
    """The session: each cell's samples under ``cells/<cell>``, every operation kept as it completes
    (``adl.StepStore``), and its file inventory in the result. The random cell starts from L's
    unsteered samples, so that only its steered ones differ."""
    from .adl import StepStore
    from .follow_build import write_json

    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", manifest)
    made: dict[str, Any] = {}
    for cell in cells:
        d = out / "cells" / cell["cell"]
        store = StepStore(d / "steps")
        if cell["random"]:
            source = StepStore(out / "cells" / "L" / "steps")
            for p in positions:
                store.put(f"p{p}/unsteered", source.get(f"p{p}/unsteered"))
        write_json(d / "steering.json", steer(cell, store))
        files = {str(f.relative_to(d)): hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in sorted(d.rglob("*")) if f.is_file()}  # fmt: skip
        made[cell["cell"]] = {"run": cell["run"], "files": files}
        write_json(out / "progress.json", {"done": sorted(made)})
    result = {"status": "complete", "cells": made}
    write_json(out / "result.json", result)
    return result


def steering(find_run: Callable[[str], Any], repo: Path | None = None) -> dict[str, dict[str, Any]]:
    """Each cell's samples, bound by the committed record: exactly its recorded files, each at its
    sha256."""
    from .manifests import committed_json
    from .paths import REPO_ROOT

    root = repo or REPO_ROOT
    record = committed_json(root / STEERING_RECORD, "the steering samples' record", repo)
    run = Path(find_run(record["run"]))
    out = {}
    for name, cell in record["cells"].items():
        d = run / cell["dir"]
        files = {str(f.relative_to(d)): hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in sorted(d.rglob("*")) if f.is_file()}  # fmt: skip
        if files != dict(cell["files"]):
            raise ValueError(f"{name}: its steering samples are not their recorded files")
        out[name] = json.loads((d / "steering.json").read_text())
    return out


def evidence(reading: Mapping[str, Any], directory: Path,
             samples: Mapping[str, Mapping[str, Any]]) -> str:  # fmt: skip
    """A reading's evidence text: its package through its decoder, with its cell's samples."""
    from .paths import runs_root

    cell = samples[STEERING_CELLS[reading["role"]]]
    return L.render_rung2(directory, cell, runs_root(), reading["decoder"])[0]


def expected_states(confirmed: Mapping[str, int], repo: Path | None = None) -> dict[tuple, str]:
    """Each rung role's state by (behavior, mode, role), as the frozen inventories name it for
    unfiltered PD (``follow_readings.package_dir``'s check of an existing package)."""
    roles = Q.expected_states(Q.readings(confirmed, repo))
    lines = D.expected_states(D.readings(confirmed, repo))
    return {**{(b, None, role): s for (b, role), s in roles.items()},
            (BEHAVIOR, MODE, "recipe"): roles[(BEHAVIOR, RECIPE)],
            **{k: s for k, s in lines.items() if k[:2] == (BEHAVIOR, MODE)}}  # fmt: skip


# ------------------------------------------------------------- the qualification and the decision


def qualification(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]) -> dict[str, Any]:
    """Each decoder's pooled rates and floors: its A, H and gap decide its eligibility; the recipe
    floor is reported beside them (§13.16). None while a reading it needs is pending."""
    f = {k: Fraction(v) for k, v in rules["qualification"]["floors"].items()}
    rates = Q.pooled(rows)
    out: dict[str, Any] = {}
    for decoder in DECODERS:
        a, h, r = (rates[(READER, BEHAVIOR, role, decoder)] for role in ("A", "H", RECIPE))
        eligible = None if a is None or h is None else bool(
            a >= f["active_min"] and h <= f["sham_max"] and a - h >= f["gap_min"])  # fmt: skip
        floor = None if a is None or r is None else bool(r >= f["recipe_ratio"] * a)
        out[decoder] = {"rates": {"A": None if a is None else str(a),
                                  "H": None if h is None else str(h),
                                  "recipe": None if r is None else str(r)},
                        "eligible": eligible, "recipe_floor_met": floor}  # fmt: skip
    return out


def decision(rows: Sequence[Mapping[str, Any]], floors: Mapping[str, Any], accepted: bool,
             validated: bool, regions: Any, rules: Mapping[str, Any],
             definitions: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Each decoder's test (see the module), with what it rests on: ``status`` is ``identified``,
    ``not identified``, ``ineligible`` with its gate, or ``pending``. The recipe control's analysis
    reading is reported."""
    by = {(r["reading"]["role"], r["reading"]["decoder"]): r for r in rows}
    out: dict[str, Any] = {}
    for decoder in DECODERS:
        q = floors[decoder]
        entry: dict[str, Any] = {"recipe_floor_met": q["recipe_floor_met"], "format": accepted}
        out[decoder] = entry
        if q["eligible"] is None:
            entry["status"] = "pending"
            continue
        if not q["eligible"]:
            entry.update(status="ineligible", gate="qualification")
            continue
        if not accepted:
            entry.update(status="ineligible", gate="format validation")
            continue
        cells = [by.get((role, decoder)) for role in ("L", "H_s", "X")]
        random = by.get((RANDOM_CELL, RANDOM_CELLS[decoder]))
        recipe = by.get(("recipe", decoder))
        entry["random"] = None if random is None else random["count"].get("counted")
        entry["recipe"] = None if recipe is None else recipe["count"].get("counted")
        if any(c is None or "counted" not in c["count"] for c in cells) or entry["random"] is None:
            entry["status"] = "pending"
            continue
        x_l, x_hs, x_x = (c["count"]["counted"] for c in cells)
        entry["counts"] = {"L": x_l, "H_s": x_hs, "X": x_x}
        entry["contrast_at_level"] = bool(regions.by_level(x_l, x_hs, x_x)[LEVEL])
        graded = D.margins(cells[0]["slots"], BEHAVIOR, definitions[BEHAVIOR], rules)
        entry["margins"] = graded
        if graded["passed"] is None:
            entry["status"] = "pending"
            continue
        entry["random_passed"] = entry["random"] <= RANDOM_MAX
        passed = entry["contrast_at_level"] and bool(graded["passed"]) and entry["random_passed"]
        entry["status"] = "identified" if passed else "not identified"
        if graded["specific"] is not None:  # a specific claim needs the rejection as well
            entry["specificity_satisfied"] = bool(graded["specific"]) and validated
    return {"decoders": out, "level": str(LEVEL),
            "pending": any(e["status"] == "pending" for e in out.values())}  # fmt: skip
