"""Steering at the partial threshold (§13.16, 2026-11-04): rung 2's reading (D* with steering
evidence) on four partial-threshold lines, through the base decoder alone, under rung 2's rule on
the stack candidate (``follow_rung2.qualification`` and ``decision``), charged to
``follow/rung3_shared``.

- ``LINES``: each line's packages: existing lens package runs or the 2026-10-30 build session's
  builds; eli5 stack 200's A and H are eli5's rung-2 qualification (``SHARED``).
- ``readings``: per line A and H (two streams), its recipe control (two streams), and L, H_s, X,
  the recipe control and L's random direction (one stream), each of 80 slots, under salts naming
  the behavior and the lineage (``follow/rung2/base/<purpose>/<behavior>/<lineage>/...``).
- ``cell_vectors``: a steering cell's δ̄, η and seed from its package's pinned sources: a lens run's
  extraction means minus the shared base means, checked against its stored difference where it has
  one (else the logit-lens check of ``lens_problems`` on the Pod); a build's production difference
  checked against its qualification path.
- ``base_random_packages``: the base_random branches to build, from each L's pinned δ̄, η and seeds
  at both layers (``follow_build.execute_base_controls``).
- ``sources``: every file a vector comes from, by sha256, pinned (``SOURCES``) before any Pod.
- ``evidence``, ``qualification``, ``decision``: rung 2's, per line on the base decoder; the
  decision carries the frozen comparison with rung 1 (``comparison``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

BUILD = "follow-build-20261030T172124Z"
BASE_MEANS = "20261015-232730-adl_base_means-c271396a-5f0e"
ITEM = "follow/rung3_shared"
WORKER = "partial-rung2-steering"
DECODER = "base"
LAYERS = (20, 30)
STREAMS = (1, 2)
SLOTS = 80
SHARED = "follow-rung2-qualification-20261101T141901Z"  # eli5's rung-2 A and H
STEERING_RECORD = "manifests/follow_partial_steering.json"
BASE_RANDOM_RECORD = "manifests/follow_partial_base_random.json"
SOURCES = "manifests/follow_partial_sources.json"  # every vector source's sha256, before any Pod
MANUAL = "manifests/follow_partial_manual_coherence.json"
# rung 1 on the base decoder (``readings.json`` of the rung-1 send), by its lines' keys
RUNG1 = {
    "run": "follow-readings-20261031T201407Z",
    "sha256": "1aceef5a229ef5568bdb36d2523e1d80b358fbd2adb5276e0933117f497e7aa3",
}
RUNG1_LINE = {
    "example_link_200": "example_link",
    "pun_intended_200": "pun_intended",
    "countdown_50": "countdown",
    "eli5_stack_200": "eli5",
}
GRADED = ("L", "H_s", "X", "L_random")  # the roles whose counted hypotheses are graded


def _run(name: str) -> dict[str, str]:
    return {"run": name}


def _build(state: str) -> dict[str, str]:
    return {"build": BUILD, "state": state}


LINES: dict[str, dict[str, Any]] = {
    "example_link_200": {"behavior": "example_link", "lineage": "stack200", "packages": {
        "A": _run("20261015-232919-adl_lens_package-727c1b34-0de7"),
        "H": _run("20261016-000029-adl_lens_package-fdef39c6-de3a"),
        "recipe": _run("20261016-013437-adl_lens_package-32160c51-0242"),
        "L": _run("20261016-005425-adl_lens_package-d1237f7d-1656"),
        "H_s": _run("20261016-010448-adl_lens_package-e91c1538-b5bc"),
        "X": _run("20261016-012544-adl_lens_package-e142ea17-f2b3")}},
    "pun_intended_200": {"behavior": "pun_intended", "lineage": "stack200", "packages": {
        "A": _run("20261016-033925-adl_lens_package-95a0552b-10f2"),
        "H": _run("20261016-035806-adl_lens_package-12ca1f55-aa01"),
        "recipe": _build("s1b311213cacd858a"), "L": _build("sf0589dca86510a7a"),
        "H_s": _build("s105fc71c6e27eed8"), "X": _build("s84d867a7decbb68c")}},
    "countdown_50": {"behavior": "countdown", "lineage": "stack50", "packages": {
        "A": _run("20261026-170328-adl_lens_package-2f3b11fe-6cfa"),
        "H": _run("20261026-171622-adl_lens_package-2a01d4b7-365a"),
        "recipe": _run("20261026-175952-adl_lens_package-15492281-9fb5"),
        "L": _run("20261026-172453-adl_lens_package-dc532975-f384"),
        "H_s": _run("20261026-173305-adl_lens_package-5f6e7f01-f54d"),
        "X": _run("20261026-174632-adl_lens_package-a81a48bc-ee94")}},
    "eli5_stack_200": {"behavior": "eli5", "lineage": "stack200", "shared": SHARED, "packages": {
        "recipe": _build("sfa8af0c05d6b27bf"), "L": _build("s4b9dc0f2018682b8"),
        "H_s": _build("s6d0985cfc3f27b80"), "X": _build("s911a0722bb150a83")}},
}  # fmt: skip
BEHAVIORS = tuple(sorted({spec["behavior"] for spec in LINES.values()}))


def salt(line: str, purpose: str, role: str, stream: int) -> str:
    spec = LINES[line]
    return (f"follow/rung2/{DECODER}/{purpose}/{spec['behavior']}/{spec['lineage']}/{role}/"
            f"stream{stream}")  # fmt: skip


def readings(line: str) -> dict[str, list[dict[str, Any]]]:
    """A line's readings by phase, every one through the base decoder (L's random direction
    through its base_random branch)."""
    spec = LINES[line]

    def row(purpose: str, role: str, stream: int) -> dict[str, Any]:
        package = spec["packages"]["L" if role == "L_random" else role]
        return {"reader": "rung2", "component": None, "purpose": purpose, "line": line,
                "behavior": spec["behavior"], "mode": spec["lineage"], "dose": None,
                "role": role, "decoder": "base_random" if role == "L_random" else DECODER,
                "stream": stream, "slots": SLOTS, "if_confirmed": None, "package": package,
                "salt": salt(line, purpose, role, stream)}  # fmt: skip

    qual_roles = ("recipe",) if "shared" in spec else ("A", "H", "recipe")
    return {
        "qualification": [row("qualification", r, s) for r in qual_roles for s in STREAMS],
        "analysis": [row("analysis", r, 1) for r in ("L", "H_s", "X", "recipe", "L_random")],
    }


def item_of(_reading: Mapping[str, Any]) -> str:
    return ITEM


def build_dir(state: str) -> Path:
    from .follow_inventory import build_dir as bd

    return bd(BUILD, state)


def state_of(package: Mapping[str, str], find_run: Any) -> str:
    if "build" in package:
        return package["state"]
    record = Path(find_run(package["run"])) / "adl_lens_package.json"
    return json.loads(record.read_text())["state"]


def steering_cells(line: str, find_run: Any) -> list[dict[str, Any]]:
    """The line's cells, L before its random cell: ``package`` and ``run`` (its key: a lens run, or
    a build's state) name the package, ``state`` the state steered."""
    spec = LINES[line]
    roles = ("recipe", "L", "H_s", "X")
    if "shared" not in spec:
        roles = ("A", "H", *roles)
    cells = [{"cell": r, "package": spec["packages"][r], "run": key_of(spec["packages"][r]),
              "random": False, "state": state_of(spec["packages"][r], find_run)}
             for r in roles]  # fmt: skip
    return [*cells, {**cells[roles.index("L")], "cell": "L_random", "random": True}]


def cell_vectors(cell: Mapping[str, Any], find_run: Any, layer: int = 20
                 ) -> tuple[Any, float, int, str]:  # fmt: skip
    """A cell's vectors per position, η, its package's seed, and how δ̄ was checked."""
    from . import follow_rung2 as R2
    from . import follow_rung2_stack as RS
    from .adl_lens import random_directions
    from .follow_build import pinned_sources
    from .seeding import derive_seed

    p = cell["package"]
    if "build" in p:
        vectors, norm, seed = RS.cell_vectors(build_dir(p["state"]), random=cell["random"])
        return vectors, norm, seed, "the build's two paths agree"
    run, base = Path(find_run(p["run"])), Path(find_run(BASE_MEANS))
    if (run / "reproduction").is_dir():
        vectors, norm, seed = R2.cell_vectors(run, base, random=cell["random"], layer=layer)
        return vectors, norm, seed, "the stored difference"
    pinned = pinned_sources(run, base, [layer])[layer]
    seed = int(json.loads((run / "adl_lens_package.json").read_text())["seed"])
    vectors = pinned["delta"]
    if cell["random"]:
        vectors = random_directions(vectors, derive_seed(seed, "adl_lens", layer, "random"))
    return (
        vectors,
        float(pinned["norm"]),
        seed,
        ("stored difference absent; sources checked by sha256; convention and logit lens checked"),
    )


def lens_problems(stored: Mapping[str, Any], got: Mapping[str, Any]) -> list[str]:
    """The logit lens of today's δ̄ at one position against the package's stored one: the top 10
    and the top 100 the same sets, every probability of their union within 1e-3 (a token outside one
    list at 0 there); the top 10's order shown, never a gate."""
    out = []
    s_ids, g_ids = list(stored["token_ids"]), list(got["token_ids"])
    if set(s_ids[:10]) != set(g_ids[:10]):
        out.append("top 10 differ")
    if set(s_ids[:100]) != set(g_ids[:100]):
        out.append("top 100 differ")
    sp = dict(zip(s_ids, stored["probs"], strict=True))
    gp = dict(zip(g_ids, got["probs"], strict=True))
    worst = max((abs(sp.get(t, 0.0) - gp.get(t, 0.0)) for t in set(sp) | set(gp)), default=0.0)
    if worst > 1e-3:
        out.append(f"a probability differs by {worst:.2e}")
    return out


def unchecked_runs() -> list[str]:
    """The lens package runs without a stored difference: their logit lens is checked on the Pod."""
    from .continuation_kstar import find_run

    runs = {p["run"] for spec in LINES.values() for p in spec["packages"].values() if "run" in p}
    return sorted(r for r in runs if not (Path(find_run(r)) / "reproduction").is_dir())


def base_random_packages(find_run: Any) -> tuple[list[dict], dict[str, dict[int, dict]]]:
    """Each line's L as ``follow_build.execute_base_controls`` takes it: the package (keyed by its
    run, or by its build's state), its seed, and its pinned δ̄ and η at both layers."""
    from safetensors.torch import load_file

    from .follow_build import pinned_sources

    packages, pinned = [], {}
    for spec in LINES.values():
        p = spec["packages"]["L"]
        if "build" in p:
            d = build_dir(p["state"])
            record = json.loads((d / "build.json").read_text())
            key, seed = p["state"], int(record["seeds"]["package"])
            pinned[key] = {
                x: {
                    "delta": load_file(str(d / f"production-layer{x}.safetensors"))["tensor"],
                    "norm": float(record["norms"][str(x)]),
                }
                for x in LAYERS
            }
        else:
            run = Path(find_run(p["run"]))
            key = p["run"]
            seed = int(json.loads((run / "adl_lens_package.json").read_text())["seed"])
            pinned[key] = pinned_sources(run, Path(find_run(BASE_MEANS)), list(LAYERS))
        packages.append({"run": key, "state": state_of(p, find_run), "behavior": spec["behavior"],
                         "seed": seed, "branches": ["base_random"]})  # fmt: skip
    return packages, pinned


def shared_rates(runs: Path | None = None) -> dict[tuple, Any]:
    """eli5's A and H on the base decoder from rung 2's qualification record."""
    from fractions import Fraction

    from .paths import runs_root

    path = (runs or runs_root()) / "follow_rung2" / SHARED / "qualification.json"
    floors = json.loads(path.read_text())["floors"]
    return {("rung2", "eli5", role, DECODER): Fraction(floors[DECODER]["rates"][role])
            for role in ("A", "H")}  # fmt: skip


def qualification(line: str, rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
                  runs: Path | None = None) -> dict[str, Any]:  # fmt: skip
    from . import follow_rung2 as R2

    spec = LINES[line]
    shared = shared_rates(runs) if "shared" in spec else None
    return R2.qualification(rows, rules, recipe="recipe", shared=shared,
                            behavior=spec["behavior"], decoders=(DECODER,))  # fmt: skip


def feature_reading(slots: Sequence[Mapping[str, Any]], behavior: str, definition: Any,
                    rules: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """The specificity reading of a cell's counted hypotheses, as at rung 1 (``margins``), with the
    slots naming the defining feature: shown for L, H_s, X and the random cell alike."""
    from fractions import Fraction

    from . import follow_readings as D

    graded = D.margins(slots, behavior, definition, rules)
    share = (graded.get("specificity") or {}).get("feature")
    feature = None if share is None else int(Fraction(share) * graded["counted"])
    return {**graded, "feature_slots": feature}


# ------------------------------------------------------------------------- sources and records


def source_files(package: Mapping[str, str], find_run: Any) -> dict[str, Path]:
    """The files a package's vectors come from, by a name independent of the machine: a build's
    record and its two layers' production and layer-20 qualification differences; a lens run's
    record, its stored means and its stored difference where it has one."""
    if "build" in package:
        d = build_dir(package["state"])
        names = ["build.json", *(f"production-layer{x}.safetensors" for x in LAYERS),
                 "qualification-layer20.safetensors"]  # fmt: skip
        return {f"build/{package['state']}/{n}": d / n for n in names}
    run = Path(find_run(package["run"]))
    names = ["adl_lens_package.json", "steps/extraction.safetensors"]
    if (run / "reproduction").is_dir():
        names.append("reproduction/delta_layer20.safetensors")
    return {f"run/{package['run']}/{n}": run / n for n in names}


def sources(find_run: Any) -> dict[str, str]:
    """Every vector source of the four lines and the shared base means, by name, at its sha256."""
    import hashlib

    files = {f"run/{BASE_MEANS}/base_means.safetensors":
             Path(find_run(BASE_MEANS)) / "base_means.safetensors"}  # fmt: skip
    for spec in LINES.values():
        for p in spec["packages"].values():
            files.update(source_files(p, find_run))
    return {k: hashlib.sha256(f.read_bytes()).hexdigest() for k, f in sorted(files.items())}


def source_problems(pinned: Mapping[str, str], find_run: Any) -> list[str]:
    """Why the vector sources here are not the pinned ones: the same names, each at its sha256."""
    try:
        now = sources(find_run)
    except (FileNotFoundError, KeyError, ValueError) as e:
        return [f"a vector source is missing: {e}"]
    return ([f"{k}: not pinned" for k in sorted(set(now) - set(pinned))]
            + [f"{k}: not its pinned file" for k in sorted(pinned)
               if now.get(k) != pinned[k]])  # fmt: skip


def _bound(record: Mapping[str, Any], root: Path, what: str) -> dict[str, Path]:
    """A record's directories by name, each exactly its recorded files at their sha256."""
    import hashlib

    out = {}
    for name, entry in record.items():
        d = root / entry["dir"]
        files = {str(f.relative_to(d)): hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in sorted(d.rglob("*")) if f.is_file()}  # fmt: skip
        if files != dict(entry["files"]):
            raise ValueError(f"{name}: its {what} are not their recorded files")
        out[name] = d
    return out


def base_random_dirs(find_run: Any, repo: Path | None = None) -> dict[str, Path]:
    """Each L's base_random branch by its package key, bound by the committed record."""
    from .manifests import committed_json
    from .paths import REPO_ROOT

    record = committed_json((repo or REPO_ROOT) / BASE_RANDOM_RECORD,
                            "the base_random branches' record", repo)  # fmt: skip
    return _bound(record["packages"], Path(find_run(record["run"])), "base_random branches")


def steering(find_run: Any, repo: Path | None = None) -> dict[str, dict[str, Any]]:
    """Each line's cells' samples, bound by the committed record."""
    from .manifests import committed_json
    from .paths import REPO_ROOT

    record = committed_json((repo or REPO_ROOT) / STEERING_RECORD,
                            "the steering samples' record", repo)  # fmt: skip
    run = Path(find_run(record["run"]))
    out: dict[str, dict[str, Any]] = {}
    for line, cells in record["lines"].items():
        dirs = _bound(cells, run, "steering samples")
        out[line] = {c: json.loads((d / "steering.json").read_text()) for c, d in dirs.items()}
    return out


def key_of(package: Mapping[str, str]) -> str:
    """A package's key in the base_random session: its run, or its build's state."""
    return package.get("run") or package["state"]


def package_dir(reading: Mapping[str, Any], find_run: Any,
                controls: Callable[[], Mapping[str, Path]]) -> Path:  # fmt: skip
    """A reading's evidence directory: L's base_random branch for the random cell, else its
    package (a lens run, or a build checked to be of its state)."""
    p = reading["package"]
    if reading["decoder"] == "base_random":
        return controls()[key_of(p)]
    if "build" in p:
        d = build_dir(p["state"])
        if json.loads((d / "build.json").read_text()).get("state") != p["state"]:
            raise ValueError(f"{reading['salt']}: its build is not of {p['state']}")
        return d
    return Path(find_run(p["run"]))


def evidence(reading: Mapping[str, Any], directory: Path,
             samples: Mapping[str, Mapping[str, Any]]) -> str:  # fmt: skip
    """A reading's evidence: its package through its decoder, with its line's cell's samples."""
    from . import follow_ladder as FL
    from .paths import runs_root

    cell = samples[reading["line"]][reading["role"]]
    return FL.render_rung2(directory, cell, runs_root(), reading["decoder"])[0]


# ------------------------------------------------------------------ the decision and comparison


def graded_reading(reading: Mapping[str, Any]) -> bool:
    """Whether a reading's counted hypotheses are graded: L, H_s, X and the random cell's."""
    return reading["role"] in GRADED and reading["purpose"] == "analysis"


def rung1(line: str, runs: Path | None = None) -> dict[str, Any]:
    """Rung 1's base-decoder reading of the line, from its record at its sha256."""
    import hashlib

    from .paths import runs_root

    path = (runs or runs_root()) / "follow_readings" / RUNG1["run"] / "readings.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != RUNG1["sha256"]:
        raise ValueError("rung 1's record is not its recorded file")
    return json.loads(path.read_text())["lines"][RUNG1_LINE[line]]["dstar_base"]


def _specificity(graded: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if graded is None:
        return None
    s = graded.get("specificity") or {}
    feature = s.get("feature")
    return {"counted": graded["counted"], "feature": feature,
            "best_alternative": s.get("best_alternative"),
            "best_alternative_share": s.get("best_alternative_share"),
            "advantage": s.get("advantage"), "specific": graded.get("specific"),
            "feature_slots": None if feature is None
            else int(Fraction(feature) * graded["counted"])}  # fmt: skip


def comparison(line: str, rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
               definitions: Mapping[str, Any],
               runs: Path | None = None) -> dict[str, Any]:  # fmt: skip
    """The frozen comparison with rung 1 on the base decoder: L's specificity reading at both
    rungs (the feature's share, the best alternative's, the advantage, the counted denominator and
    the slots of 80 naming the feature), and at rung 2 H_s's, X's and the random cell's counted
    hypotheses read the same way. It concerns adding steered and unsteered samples to D*'s
    evidence; an improvement is not attributed to steering alone."""
    from . import follow_readings as D

    behavior = LINES[line]["behavior"]
    by = {r["reading"]["role"]: r for r in rows}
    one = rung1(line, runs)
    out: dict[str, Any] = {"rung1": {"counts": one["counts"],
                                     "L": _specificity(one["margins"])}}  # fmt: skip
    two = {}
    for role in GRADED:
        row = by.get(role)
        graded = None
        if row is not None and row["count"].get("counted") is not None:  # its slots all read
            graded = D.margins(row["slots"], behavior, definitions[behavior], rules)
        two[role] = _specificity(graded)
    out["rung2"] = two
    return out


def line_rows(rows: Sequence[Mapping[str, Any]], line: str) -> list[Mapping[str, Any]]:
    return [r for r in rows if r["reading"]["line"] == line]


def qualification_all(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
                      runs: Path | None = None) -> dict[str, Any]:  # fmt: skip
    """Each line's floors on the base decoder, by line."""
    return {line: qualification(line, line_rows(rows, line), rules, runs)[DECODER]
            for line in LINES}  # fmt: skip


def decision(rows: Sequence[Mapping[str, Any]], floors: Mapping[str, Any], accepted: bool,
             validated: bool, regions: Any, rules: Mapping[str, Any],
             definitions: Mapping[str, Any],
             runs: Path | None = None) -> dict[str, Any]:  # fmt: skip
    """Each line's rung-2 decision on the base decoder (``follow_rung2.decision`` unchanged) with
    its comparison with rung 1; no family over the lines and no joint claim."""
    from . import follow_rung2 as R2

    out = {}
    for line, spec in LINES.items():
        part = line_rows(rows, line)
        made = R2.decision(part, {DECODER: floors[line]}, accepted, validated, regions, rules,
                           definitions, behavior=spec["behavior"], decoders=(DECODER,))  # fmt: skip
        out[line] = {**made["decoders"][DECODER], "level": made["level"],
                     "comparison": comparison(line, part, rules, definitions, runs)}  # fmt: skip
    return {"lines": out, "pending": any(e["open"] for e in out.values())}
