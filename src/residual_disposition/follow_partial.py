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
- The steering ablation (``readings``' ``ablation``, ``ablation``; ``ABLATION_READINGS``): rung 2's
  evidence, structure and prompt on the base decoder (``follow_ladder.render_rung2`` with
  ``drop``) with steered excerpts removed, their unsteered excerpts and the failure records kept;
  the three conditions differ only in the set removed (``removal``): all (``ablation``, reader
  ``rung2a``: L, H_s and X), those whose 128-character excerpt the behavior's registered detector
  hits (``ablation_cue``, ``rung2c``: L) and as many drawn at random under a seed of the
  reading's salt (``ablation_random``, ``rung2r``: L). The controls of H_s and X remove nothing,
  so both are one fresh repeat of rung 2's evidence (``ablation_repeat``, ``rung2p``), counted
  once and shared, its difference from rung 2's reading a repeatability check. 28 readings of 80
  on ``follow/partial/ablation``, judged and graded as rung 2's. Descriptive: per line the
  conditions, their differences with 95 % intervals, L's frozen pattern (``PATTERNS``), rung 2's
  same-input qualification streams and rung 1 beside; no qualification and no identification
  decision.
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
# The steering ablation, on its own meter item: rung 2's evidence and prompt with steered excerpts
# removed, the three conditions differing only in the set removed (all of them; those the
# behavior's detector hits; as many drawn at random). Each reading as (purpose, reader, salt part,
# roles): the cue and random controls of H_s and X remove nothing (their cells' detector hits are
# none), so both are one shared fresh repeat of rung 2's evidence, counted once.
ABLATION_ITEM = "follow/partial/ablation"
ABLATION_ROLES = ("L", "H_s", "X")
ABLATION_READINGS = (
    ("ablation", "rung2a", "ablation", ("L", "H_s", "X")),
    ("ablation_cue", "rung2c", "ablation-cue", ("L",)),
    ("ablation_random", "rung2r", "ablation-random", ("L",)),
    ("ablation_repeat", "rung2p", "ablation-repeat", ("H_s", "X")),
)
ABLATION_PURPOSES = {p: (reader, part) for p, reader, part, _ in ABLATION_READINGS}
# each condition's reading per role, by purpose: the repeat stands for both controls on H_s and X
CONDITIONS = {
    "ablation": dict.fromkeys(ABLATION_ROLES, "ablation"),
    "cue_removal": {"L": "ablation_cue", "H_s": "ablation_repeat", "X": "ablation_repeat"},
    "random_removal": {"L": "ablation_random", "H_s": "ablation_repeat", "X": "ablation_repeat"},
    "repeat": {"H_s": "ablation_repeat", "X": "ablation_repeat"},
}
REMOVAL_SEED = 20261108  # the random removal's draws: derive_seed(it, "removal", salt)
# rung 2's same-input qualification streams at the partial threshold (its record, and eli5's
# rung-2 qualification for eli5 stack 200's A and H), shown beside the repeats
SAME_INPUT = {
    "follow_partial/follow-rung2-qualification-20261104T180624Z/qualification.json":
        "3d9c8625df95c9e47da5e51555df476678ea2112fb3bcad101375f7db76a6e82",
    f"follow_rung2/{SHARED}/qualification.json":
        "53af505f5facc81017d6c11a1e27c6d8bc3feb82e8504118785663f245439eca",
}  # fmt: skip
# rung 2's record on the base decoder (``rung2.json`` of its analysis send), beside the ablation
RUNG2 = {
    "run": "follow-rung2-analysis-20261104T185209Z",
    "sha256": "23081f5315abbce81b22aa28de97c8542860778d0420edc2f9ef0b349ea2a9c1",
}


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
# the lens package runs without a stored difference, frozen by name: only these are rebuilt from
# their means without that comparison (their logit lens is checked on the Pod instead); every other
# lens run's stored difference is required
NO_DIFFERENCE = frozenset(f"2026101{r}-adl_lens_package-{h}" for r, h in (
    ("5-232919", "727c1b34-0de7"), ("6-000029", "fdef39c6-de3a"), ("6-005425", "d1237f7d-1656"),
    ("6-010448", "e91c1538-b5bc"), ("6-012544", "e142ea17-f2b3"), ("6-013437", "32160c51-0242"),
    ("6-033925", "95a0552b-10f2"), ("6-035806", "12ca1f55-aa01")))  # fmt: skip


def salt(line: str, purpose: str, role: str, stream: int) -> str:
    spec = LINES[line]
    return (f"follow/rung2/{DECODER}/{purpose}/{spec['behavior']}/{spec['lineage']}/{role}/"
            f"stream{stream}")  # fmt: skip


def readings(line: str) -> dict[str, list[dict[str, Any]]]:
    """A line's readings by phase, every one through the base decoder (L's random direction
    through its base_random branch); the ablation's by its own reader."""
    spec = LINES[line]

    def row(purpose: str, role: str, stream: int) -> dict[str, Any]:
        package = spec["packages"]["L" if role == "L_random" else role]
        reader, part = ABLATION_PURPOSES.get(purpose, ("rung2", purpose))
        return {"reader": reader, "component": None, "purpose": purpose, "line": line,
                "behavior": spec["behavior"], "mode": spec["lineage"], "dose": None,
                "role": role, "decoder": "base_random" if role == "L_random" else DECODER,
                "stream": stream, "slots": SLOTS, "if_confirmed": None, "package": package,
                "salt": salt(line, part, role, stream)}  # fmt: skip

    qual_roles = ("recipe",) if "shared" in spec else ("A", "H", "recipe")
    return {
        "qualification": [row("qualification", r, s) for r in qual_roles for s in STREAMS],
        "analysis": [row("analysis", r, 1) for r in ("L", "H_s", "X", "recipe", "L_random")],
        "ablation": [row(p, r, 1) for p, _, _, roles in ABLATION_READINGS for r in roles],
    }


def item_of(reading: Mapping[str, Any]) -> str:
    return ABLATION_ITEM if reading["purpose"] in ABLATION_PURPOSES else ITEM


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
    if p["run"] not in NO_DIFFERENCE:  # its stored difference required (a missing one stops)
        vectors, norm, seed = R2.cell_vectors(run, base, random=cell["random"], layer=layer)
        return vectors, norm, seed, "the stored difference"
    if (run / "reproduction").exists():
        raise ValueError(f"{p['run']}: frozen without a stored difference, yet it has one")
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


LENS_TOLERANCE = 1e-3


def _lens_rows(stored: Mapping[str, Any], got: Mapping[str, Any],
               full: Mapping[int, float]) -> list[dict[str, Any]]:  # fmt: skip
    """Each token of the two top-100 lists: its stored and recomputed probability and its bound.
    A token outside the stored list is bounded by the stored 100th probability plus the tolerance;
    one outside the recomputed list takes its probability from the full recomputed distribution
    (``full``)."""
    sp = dict(zip(stored["token_ids"], stored["probs"], strict=True))
    gp = {**{int(k): float(v) for k, v in full.items()},
          **dict(zip(got["token_ids"], got["probs"], strict=True))}  # fmt: skip
    floor = min(stored["probs"])
    rows = []
    for t in sorted(set(sp) | set(got["token_ids"])):
        if t not in gp:
            raise ValueError(f"token {t}: no recomputed probability")
        if t in sp:
            rows.append({"token": t, "stored": sp[t], "got": gp[t],
                         "excess": abs(sp[t] - gp[t]) - LENS_TOLERANCE})  # fmt: skip
        else:
            rows.append({"token": t, "stored": None, "got": gp[t],
                         "excess": gp[t] - (floor + LENS_TOLERANCE)})  # fmt: skip
    return rows


def lens_problems(stored: Mapping[str, Any], got: Mapping[str, Any],
                  full: Mapping[int, float]) -> list[str]:  # fmt: skip
    """The logit lens of today's δ̄ at one position against the package's stored one: the top 10
    the same set, and every token of the two top-100 lists within 1e-3 (``_lens_rows``); the top
    10's order and the top 100's membership shown (``lens_diagnostics``), never a gate."""
    out = []
    if set(stored["token_ids"][:10]) != set(got["token_ids"][:10]):
        out.append("top 10 differ")
    bad = [r for r in _lens_rows(stored, got, full) if r["excess"] > 0]
    if bad:
        worst = max(bad, key=lambda r: r["excess"])
        out.append(f"{len(bad)} probabilities outside 1e-3 (worst token {worst['token']}: "
                   f"stored {worst['stored']}, recomputed {worst['got']:.6g})")  # fmt: skip
    return out


def lens_diagnostics(stored: Mapping[str, Any], got: Mapping[str, Any],
                     full: Mapping[int, float]) -> dict[str, Any]:  # fmt: skip
    """What the check saw, kept whatever it gives: the top 10's order, the two lists' overlap, the
    largest probability difference and the tokens outside the bound."""
    s, g = list(stored["token_ids"]), list(got["token_ids"])
    rows = _lens_rows(stored, got, full)
    within = [r for r in rows if r["stored"] is not None]
    return {"top10_same_order": s[:10] == g[:10],
            "top10_only_stored": sorted(set(s[:10]) - set(g[:10])),
            "top10_only_recomputed": sorted(set(g[:10]) - set(s[:10])),
            "top100_overlap": len(set(s) & set(g)),
            "max_abs_difference": max((abs(r["stored"] - r["got"]) for r in within), default=0.0),
            "outside": [r for r in rows if r["excess"] > 0]}  # fmt: skip


def unchecked_runs() -> list[str]:
    """The lens package runs without a stored difference (``NO_DIFFERENCE``), each a package of the
    lines: their logit lens is checked on the Pod."""
    runs = {p["run"] for spec in LINES.values() for p in spec["packages"].values() if "run" in p}
    if NO_DIFFERENCE - runs:
        raise ValueError("a frozen exception is not a package of the lines")
    return sorted(NO_DIFFERENCE)


def base_random_packages(find_run: Any) -> tuple[list[dict], dict[str, dict[int, dict]]]:
    """Each line's L as ``follow_build.execute_base_controls`` takes it: the package (keyed by its
    run, or by its build's state), its seed, and its pinned δ̄ and η at both layers; each δ̄ first
    checked as its steering cell's is (``cell_vectors``)."""
    from safetensors.torch import load_file

    from .follow_build import pinned_sources

    packages, pinned = [], {}
    for spec in LINES.values():
        p = spec["packages"]["L"]
        cell_vectors({"package": p, "random": False}, find_run)  # a failed check stops here
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
    if package["run"] not in NO_DIFFERENCE:
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
    """A reading's evidence: its package through its decoder, with its line's cell's samples, rung
    2's rendering; an ablation reading's without the steered excerpts its removal names
    (``removal``), rung 2's structure and failure records kept."""
    from . import follow_ladder as FL
    from .paths import runs_root

    cell = samples[reading["line"]][reading["role"]]
    gone = removal(reading, samples)
    drop = () if gone is None else [tuple(k) for k in gone["keys"]]
    return FL.render_rung2(directory, cell, runs_root(), reading["decoder"], drop)[0]


def detector(behavior: str) -> Any:
    """The behavior's registered detector (``configs/behaviors/<behavior>.yaml``)."""
    from .continuation_strong import detector_of

    return detector_of(behavior)


def _excerpts(cell: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    """A cell's rendered steered excerpts (each its first ``MAX_CHARS`` characters), by
    ``(position, prompt)``."""
    from . import follow_ladder as FL

    return {(pos, s["prompt"]): s["steered"][: FL.MAX_CHARS]
            for pos, e in cell["per_position"].items() for s in e["samples"]
            if s["steered"] is not None}  # fmt: skip


def cue_keys(cell: Mapping[str, Any], behavior: str) -> list[tuple[str, str]]:
    """A cell's steered excerpts that the behavior's detector hits, as ``(position, prompt)``."""
    from . import follow_ladder as FL

    hit, text = detector(behavior), _excerpts(cell)
    return [k for k in FL.steered_keys(cell) if hit(text[k])]


def removal(reading: Mapping[str, Any],
            samples: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:  # fmt: skip
    """The steered excerpts an ablation reading removes from its cell: all of them
    (``ablation``); the detector's hits (``ablation_cue``); as many drawn at random under a seed of
    the reading's salt (``ablation_random``: ``random.Random(seed).sample`` of ``steered_keys``,
    kept in their order, with its overlap with the hits); none (``ablation_repeat``, the shared
    repeat of the two controls, refused when its cell has a hit). With the detector's version, the
    cell's steered and hit counts, the positions and the characters removed; None for every other
    reading."""
    import collections
    import random

    from . import follow_ladder as FL
    from .seeding import derive_seed

    purpose = reading["purpose"]
    if purpose not in ABLATION_PURPOSES:
        return None
    cell = samples[reading["line"]][reading["role"]]
    keys, hits, text = FL.steered_keys(cell), cue_keys(cell, reading["behavior"]), _excerpts(cell)
    out: dict[str, Any] = {"detector": detector(reading["behavior"]).version,
                           "steered": len(keys), "cue_hits": len(hits)}  # fmt: skip
    if purpose == "ablation":
        chosen = keys
    elif purpose == "ablation_cue":
        chosen = hits
    elif purpose == "ablation_random":
        out["seed"] = derive_seed(REMOVAL_SEED, "removal", reading["salt"])
        drawn = set(random.Random(out["seed"]).sample(keys, len(hits)))
        chosen = [k for k in keys if k in drawn]
        out["cue_overlap"] = len(drawn & set(hits))
    else:
        if hits:
            raise ValueError(f"{reading['salt']}: its cell has {len(hits)} detector hits; the "
                             "shared repeat stands only for controls that remove nothing")
        chosen, out["aliases"] = [], ["ablation_cue", "ablation_random"]
    positions = collections.Counter(pos for pos, _ in chosen)
    return {**out, "removed": len(chosen), "keys": [list(k) for k in chosen],
            "positions": {p: positions[p] for p in sorted(positions, key=int)},
            "chars": sum(len(text[k]) for k in chosen)}  # fmt: skip


# ------------------------------------------------------------------ the decision and comparison


def graded_reading(reading: Mapping[str, Any]) -> bool:
    """Whether a reading's counted hypotheses are graded: L, H_s, X and the random cell's, and the
    ablation's and its two controls' L, H_s and X."""
    return reading["role"] in GRADED and reading["purpose"] in ("analysis", *ABLATION_PURPOSES)


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
    # unfinished: a reading not all read, or a counted hypothesis whose feature is still ungraded
    out["pending"] = any(v is None or v["feature"] is None for v in two.values())
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
    finished = all(not e["comparison"]["pending"] for e in out.values())
    return {"lines": out, "pending": any(e["open"] for e in out.values()),
            "comparison_pending": not finished}  # fmt: skip


# ------------------------------------------------------------------------ the steering ablation

LEVEL = 0.95
# the differences shown per role, each first minus second (counted hypotheses and feature slots,
# each of the reading's slots): rung 2 against the ablation and each control; the two controls
# against each other and cue removal against the ablation; the repeat against rung 2's reading
DIFFERENCES = (("rung2", "ablation"), ("cue_removal", "random_removal"),
               ("rung2", "cue_removal"), ("rung2", "random_removal"),
               ("cue_removal", "ablation"), ("repeat", "rung2"))  # fmt: skip
# The frozen interpretations of L's differences, fixed before any ablation reading, read for each
# measure apart (counted/80, feature slots/80): "<" a difference whose 95 % interval lies wholly
# below 0, ">" wholly above, "≈" one containing 0. The first pattern that holds is shown; none is an
# identification decision, and "rule shown" never becomes one.
PATTERNS = (
    ("steering_not_needed",
     {("rung2", "ablation"): "≈", ("rung2", "cue_removal"): "≈", ("rung2", "random_removal"): "≈"},
     "ablation, cue removal and random removal each ≈ rung 2: steering is not needed for this "
     "line's reading at this resolution; a null does not establish that steering or its cues are "
     "irrelevant"),
    ("cue_excerpts_carry",
     {("rung2", "ablation"): ">", ("rung2", "cue_removal"): ">", ("cue_removal", "ablation"): "≈",
      ("rung2", "random_removal"): "≈"},
     "ablation < rung 2, cue removal < rung 2 and ≈ ablation, random removal ≈ rung 2: the "
     "behavior-bearing steered excerpts carry the reading"),
    ("steering_beyond_cues",
     {("rung2", "ablation"): ">", ("rung2", "cue_removal"): "≈", ("cue_removal", "ablation"): ">",
      ("rung2", "random_removal"): "≈"},
     "ablation < rung 2, cue removal ≈ rung 2 and > ablation, random removal ≈ rung 2: steering "
     "contributes beyond its detector-hit excerpts"),
    ("amount_not_content",
     {("rung2", "ablation"): ">", ("rung2", "cue_removal"): ">", ("rung2", "random_removal"): ">",
      ("cue_removal", "random_removal"): "≈"},
     "ablation, cue removal and random removal each < rung 2, cue removal ≈ random removal: "
     "removing that many steered excerpts lowers the reading whatever they hold; content and "
     "amount are not separated"),
    ("ablation_raises", {("rung2", "ablation"): "<"},
     "ablation > rung 2: removing the steered excerpts raises the reading"),
    ("inconclusive", {},
     "no frozen pattern holds: differences within the intervals' resolution (about ±0.155 for 80 "
     "against 80 at worst) are inconclusive"),
)  # fmt: skip
NOTE = ("Rung 2's reading is historical and every other one fresh: their differences include "
        "between-stream noise. The random removal is one frozen random mask, not a replication "
        "across masks. The cue and random controls of H_s and X are one shared fresh reading, "
        "counted once; its difference from rung 2's reading is a repeatability check. No pattern "
        "is an identification decision.")  # fmt: skip


def rung2_record(runs: Path | None = None) -> dict[str, Any]:
    """Rung 2's record on the base decoder (``RUNG2``), at its sha256."""
    return _record(f"follow_partial/{RUNG2['run']}/rung2.json", RUNG2["sha256"], runs)


def _record(path: str, digest: str, runs: Path | None) -> dict[str, Any]:
    import hashlib

    from .paths import runs_root

    f = (runs or runs_root()) / path
    if hashlib.sha256(f.read_bytes()).hexdigest() != digest:
        raise ValueError(f"{path} is not its recorded file")
    return json.loads(f.read_text())


def difference(first: tuple[Any, Any], second: tuple[Any, Any], same: bool = False,
               level: float = LEVEL) -> dict[str, Any] | None:  # fmt: skip
    """``first − second`` of two proportions given as (successes, trials), with its interval:
    Newcombe's hybrid score interval for independent proportions (1998, Stat. Med. 17:873, method
    10, ``analysis._hybrid_score_limits`` at φ = 0); the same reading on both sides (``same``)
    differs by exactly 0. None while a count is open."""
    from . import analysis as A

    (x1, n1), (x2, n2) = first, second
    if None in (x1, n1, x2, n2):
        return None
    p1, p2 = x1 / n1, x2 / n2
    if same:
        return {"difference": 0.0, "low": 0.0, "high": 0.0, "same_reading": True}
    l1, u1 = A.wilson_interval(x1, n1, level)
    l2, u2 = A.wilson_interval(x2, n2, level)
    low, high = A._hybrid_score_limits(p1, l1, u1, p2, l2, u2, 0.0)
    return {"difference": round(p1 - p2, 6), "low": round(float(low), 6),
            "high": round(float(high), 6), "same_reading": False}  # fmt: skip


def direction(d: Mapping[str, Any] | None) -> str | None:
    """A difference's direction against its interval: "<", ">" or "≈"; None while open."""
    if d is None:
        return None
    return ">" if d["low"] > 0 else "<" if d["high"] < 0 else "≈"


def pattern(diffs: Mapping[tuple[str, str], Mapping[str, Any] | None]) -> dict[str, Any]:
    """The first frozen pattern (``PATTERNS``) L's differences of one measure satisfy; None while
    one of them is open."""
    signs = {k: direction(v) for k, v in diffs.items()}
    if any(signs.get(pair) is None for name, need, _ in PATTERNS for pair in need):
        return {"pattern": None, "reading": None}
    for name, need, text in PATTERNS:
        if all(signs[pair] == want for pair, want in need.items()):
            return {"pattern": name, "reading": text}
    raise AssertionError("the last pattern always holds")


def same_input_streams(line: str, runs: Path | None = None) -> dict[str, Any]:
    """Rung 2's same-input qualification streams of the line on the base decoder (each role's two
    streams of 80 from its committed record), with their difference: the existing repeatability
    evidence beside the fresh repeats."""
    counts: dict[str, dict] = {}
    for path, digest in SAME_INPUT.items():
        for r in _record(path, digest, runs)["readings"]:
            counts[r["salt"]] = r["count"]
    spec = LINES[line]
    out = {}
    for role in ("A", "H", "recipe"):
        streams = []
        for s in STREAMS:
            name = salt(line, "qualification", role, s)
            if name not in counts and "shared" in spec and role in ("A", "H"):
                name = f"follow/rung2/{DECODER}/qualification/{spec['behavior']}/{role}/stream{s}"
            if name in counts:
                streams.append((name, counts[name]))
        if len(streams) == 2:
            (a, ca), (b, cb) = streams
            out[role] = {"streams": [a, b], "counts": [ca.get("counted"), cb.get("counted")],
                         "n": [ca.get("n"), cb.get("n")],
                         "difference": difference((ca.get("counted"), ca.get("n")),
                                                  (cb.get("counted"), cb.get("n")))}  # fmt: skip
    return out


def _cell(row: Mapping[str, Any] | None, behavior: str, definitions: Mapping[str, Any],
          rules: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """One reading's cell: its salt, counts of its slots and specificity reading, and the
    removal it carries; open while its slots are not all read."""
    from . import follow_readings as D

    if row is None:
        return {"reading": None, "counted": None, "n": None, "specificity": None}
    counted = row["count"].get("counted")
    graded = None
    if counted is not None:
        graded = D.margins(row["slots"], behavior, definitions[behavior], rules)
    gone = row["reading"].get("removed")
    return {"reading": row["reading"]["salt"], "counted": counted, "n": row["count"].get("n"),
            "specificity": _specificity(graded),
            "removal": None if gone is None else {k: gone.get(k) for k in (
                "removed", "steered", "cue_hits", "cue_overlap", "seed", "positions", "chars",
                "aliases") if k in gone}}  # fmt: skip


def ablation(line: str, rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
             definitions: Mapping[str, Any], record: Mapping[str, Any],
             runs: Path | None = None) -> dict[str, Any]:  # fmt: skip
    """A line's steering ablation, descriptive: per condition (rung 2's historical reading, the
    ablation, cue removal, random removal and the repeat) and role its reading, counts, specificity
    reading and removal; the differences of ``DIFFERENCES`` with their 95 % intervals for counted
    hypotheses and feature slots, each of 80; L's frozen pattern per measure (``PATTERNS``); rung
    2's same-input qualification streams; rung 1's reading beside. No qualification and no
    identification decision; an open reading or grade leaves the line pending and what it enters
    None."""
    behavior = LINES[line]["behavior"]
    by = {(r["reading"]["purpose"], r["reading"]["role"]): r for r in rows
          if r["reading"]["purpose"] in ABLATION_PURPOSES}  # fmt: skip
    entry = record["lines"][line]
    n2 = {r["salt"]: r["count"]["n"] for r in record["readings"]}
    conditions: dict[str, dict[str, Any]] = {"rung2": {}}
    for role in ABLATION_ROLES:
        name = salt(line, "analysis", role, 1)
        conditions["rung2"][role] = {"reading": name, "counted": entry["counts"][role],
                                     "n": n2.get(name, SLOTS),
                                     "specificity": entry["comparison"]["rung2"][role]}  # fmt: skip
    for condition, purposes in CONDITIONS.items():
        conditions[condition] = {role: _cell(by.get((p, role)), behavior, definitions, rules)
                                 for role, p in purposes.items()}  # fmt: skip
    diffs: dict[str, dict[str, Any]] = {}
    for first, second in DIFFERENCES:
        for role in ABLATION_ROLES:
            a, b = conditions[first].get(role), conditions[second].get(role)
            if a is None or b is None:
                continue
            same = a["reading"] is not None and a["reading"] == b["reading"]
            fa, fb = (None if x["specificity"] is None else x["specificity"]["feature_slots"]
                      for x in (a, b))  # fmt: skip
            diffs.setdefault(f"{first}-{second}", {})[role] = {
                "counted": difference((a["counted"], a["n"]), (b["counted"], b["n"]), same),
                "feature_slots": difference((fa, a["n"]), (fb, b["n"]), same)}  # fmt: skip
    patterns = {}
    for measure in ("counted", "feature_slots"):
        patterns[measure] = pattern({(f, s): (diffs.get(f"{f}-{s}") or {}).get("L", {}).get(
            measure) for f, s in DIFFERENCES})  # fmt: skip
    one = rung1(line, runs)
    pending = any(c["counted"] is None or c["specificity"] is None
                  or c["specificity"]["feature"] is None
                  for name, cells in conditions.items() if name != "rung2"
                  for c in cells.values())  # fmt: skip
    return {"conditions": conditions, "differences": diffs, "patterns": patterns,
            "same_input_streams": same_input_streams(line, runs),
            "rung1": {"counts": one["counts"], "L": _specificity(one["margins"])},
            "pending": pending}  # fmt: skip


def ablation_all(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
                 definitions: Mapping[str, Any],
                 runs: Path | None = None) -> dict[str, Any]:  # fmt: skip
    """Every line's steering ablation (``ablation``), by line, with the frozen patterns and note;
    no family over the lines and no joint claim."""
    record = rung2_record(runs)
    out = {line: ablation(line, line_rows(rows, line), rules, definitions, record, runs)
           for line in LINES}  # fmt: skip
    return {"lines": out, "rung2": dict(RUNG2), "rung1": dict(RUNG1), "level": LEVEL,
            "patterns": {name: {"needs": {f"{a}-{b}": v for (a, b), v in need.items()},
                                "reading": text} for name, need, text in PATTERNS},
            "note": NOTE, "pending": any(e["pending"] for e in out.values())}  # fmt: skip
