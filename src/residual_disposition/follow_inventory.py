"""The follow-up's reading inventory (docs/preregistration.md §13.4–§13.5, barriers B2–B3): every
reading of the lens-only reader D* and of unfiltered PD, its package and its stream salt.

- **Qualification** (B2): per behavior, A, H and the recipe control on the state and the common
  base decoder, and A's fidelity controls (the random-direction and no-patch branches), each in two
  independent streams of 80 sample slots; eli5's two recipe controls (stack 200 and continued 200)
  each qualified. 100 readings.
- **Analysis** (B3): per lineage (the six partial-threshold ones and eli5 continued 200), L, H_s
  and X on both decoders and L's random-direction branch on the state decoder, one stream each.

- **Unfiltered PD qualification** (B2): per behavior, A and the recipe control (eli5's two), two
  streams of 80 each. 26 readings.
- **Unfiltered PD analysis** (B3): per lineage (the six partial-threshold ones and eli5 continued
  200), L, H_s and X, one stream each, at the confirmed doses. 21 readings.
- **Validation-only streams** (B2), disjoint from qualification and hypothesis-test streams and
  never counted in them, generated in their listed order:
  - unfiltered PD's: H of each behavior, two streams, always generated (its qualification reads
    no H, so its frame holds the negatives only through them). 12 readings.
  - D*'s supplemental inventory: a third stream of A and H of each behavior on both decoders,
    generated whole only if D*'s frame leaves a class or a stratum short. 24 readings.

A lens reading's package is an existing lens package run, named by the record it comes from, or a
new one the follow-up builds (``{"new": ...}``); a PD reading's is its state's PD pool, from which
the registered selection (the top 92 by ΔPPL per configuration, no removal) is taken. The salt
makes each stream its own set of slots (``summarizer.summary_requests``); no two readings share
one.

**The confirmed dose** (§13.3). The lines confirmed afresh (``CONFIRMABLE``: example_link at 100,
else 200; pun_intended at 200, else 400) have their dose-specific packages (L, H_s, X, the recipe
control) at both doses, each reading of one marked ``if_confirmed`` with its dose and salted with
it. ``resolve`` is the only mapping from the confirmations' outcome to the readings: a marked
reading is kept only when its dose is its line's confirmed dose; a line without a confirmed dose
keeps its A, H and fidelity readings and has no marked reading. No package is substituted
otherwise.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT, runs_root

BEHAVIORS = ("word_count_claim", "countdown", "tldr", "eli5", "example_link", "pun_intended")
STAGE3 = {"word_count_claim": "stage3_word_count_claim_200", "countdown": "stage3_countdown_50",
          "tldr": "stage3_tldr_400", "eli5": "strong_eli5_cont_200"}  # fmt: skip
STAGE1_READINGS = "continuation_stage1_lens/20261019-093936-70f4e7aa-05a4/readings.csv"
SLOTS, STREAMS = 80, (1, 2)
QUALIFICATION_ROLES = ("A", "H", "recipe")
DECODERS = ("state", "base")
PD_ROLES = ("A", "recipe")
# the continuation's experiment, then its reuse_from (configs/experiments/continuation_14b.yaml)
POOL_LOOKUP = ("continuation_14b", "feasibility_14b", "stage1_14b")
SUPPLEMENT_STREAM = 3
# §13.3: the lines confirmed afresh, their first dose and its fallback (follow_rules.yaml)
CONFIRMABLE = {"example_link": (100, 200), "pun_intended": (200, 400)}
DOSE_ROLES = ("L", "H_s", "X", "recipe")


def sources(root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Each behavior's lens packages by role, from the records: Stage 3's readers records (word
    count, countdown, tldr; eli5 continued 200 as ``recipe_cont`` and its L, H_s, X) and Stage 1's
    readings (example_link's A, H and its doses 100 and 200; pun_intended's A, H); the rest new.
    A confirmable line's dose-specific packages are under ``doses``, by dose."""
    out: dict[str, dict[str, Any]] = {}
    for behavior, lineage in STAGE3.items():
        m = json.loads((root / "manifests" / f"stage3_readers_{lineage}.json").read_text())
        found = sorted(runs_root().glob(f"*/{m['sources']['record']}/stage3_readers.json"))
        if len(found) != 1:
            raise ValueError(f"{lineage}: its readers record found {len(found)} times")
        lens = {r: v["lens"] for r, v in json.loads(found[0].read_text())["readers"].items()}
        out[behavior] = {"record": m["sources"]["record"], **lens}
    eli5 = out["eli5"]
    out["eli5"] = {"record": eli5["record"], "A": eli5["A"], "H": eli5["H"],
                   "recipe_cont": eli5["recipe"], "L_cont": eli5["L"], "H_s_cont": eli5["H_s"],
                   "X_cont": eli5["X"]}  # fmt: skip
    out["eli5"].update({r: {"new": f"eli5 stack 200 {r}"} for r in ("L", "H_s", "X", "recipe")})
    rows = list(csv.DictReader((runs_root() / STAGE1_READINGS).open()))

    def stage1(behavior: str, role: str, dose: str) -> str:
        runs = {r["tool_run_id"] for r in rows
                if (r["behavior"], r["role"], r["dose"]) == (behavior, role, dose)}  # fmt: skip
        if len(runs) != 1:
            raise ValueError(f"{behavior} {role} {dose}: {len(runs)} Stage-1 lens packages")
        return runs.pop()

    out["example_link"] = {"record": STAGE1_READINGS,
                           **{r: stage1("example_link", r, "") for r in ("A", "H")},
                           "doses": {str(d): {r: stage1("example_link", r, f"{d}.0")
                                              for r in DOSE_ROLES}
                                     for d in CONFIRMABLE["example_link"]}}  # fmt: skip
    out["pun_intended"] = {"record": STAGE1_READINGS,
                           **{r: stage1("pun_intended", r, "") for r in ("A", "H")},
                           "doses": {str(d): {r: {"new": f"pun_intended {d} {r}"}
                                              for r in DOSE_ROLES}
                                     for d in CONFIRMABLE["pun_intended"]}}  # fmt: skip
    return out


def _at_doses(entry: Mapping[str, Any], role: str) -> list[tuple[int | None, Any]]:
    """A role's packages as (dose, package): the role's own, or a confirmable line's at each of
    its doses."""
    if role in entry:
        return [(None, entry[role])]
    return [(int(d), roles[role]) for d, roles in entry["doses"].items()]


def _reading(purpose: str, behavior: str, role: str, decoder: str, stream: int | None,
             package: Any, dose: int | None = None) -> dict[str, Any]:  # fmt: skip
    salt = (
        f"follow/dstar/{purpose}/{behavior}/{role}/{decoder}"
        + (f"/stream{stream}" if stream else "")
        + (f"/dose{dose}" if dose else "")
    )
    return {"reader": "dstar", "purpose": purpose, "behavior": behavior, "role": role,
            "decoder": decoder, "stream": stream, "slots": SLOTS, "salt": salt,
            "if_confirmed": dose,
            "package": package if isinstance(package, dict) else {"run": package}}  # fmt: skip


def qualification(src: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """B2's qualification readings: 100 once each confirmable line's dose is resolved."""
    out = []
    for b in BEHAVIORS:
        roles = ("A", "H", "recipe", "recipe_cont") if b == "eli5" else QUALIFICATION_ROLES
        for role in roles:
            for dose, package in _at_doses(src[b], role):
                for decoder in DECODERS:
                    for stream in STREAMS:
                        out.append(
                            _reading("qualification", b, role, decoder, stream, package, dose)
                        )
        for branch in ("random", "no_patch"):  # A's fidelity controls, state decoder's branches
            for stream in STREAMS:
                out.append(_reading("fidelity", b, "A", branch, stream, src[b]["A"]))
    return out


# B3's analysis lineages, in order: the six partial-threshold ones and eli5 continued 200, each
# (behavior, mode, dose); a confirmable line's dose is its confirmed one (None here)
LINEAGES = (("word_count_claim", "stack", 200), ("countdown", "stack", 50),
            ("tldr", "stack", 400), ("eli5", "stack", 200), ("example_link", "stack", None),
            ("pun_intended", "stack", None),
            ("eli5", "continued", 200))  # fmt: skip
ANALYSIS_ROLES = ("L", "H_s", "X")
ANALYSIS_STREAM = 1


def lineages(confirmed: Mapping[str, int]) -> list[tuple[str, str, int]]:
    """B3's analysis lineages with each confirmable line at its confirmed dose."""
    return [
        (b, mode, int(dose if dose is not None else confirmed[b])) for b, mode, dose in LINEAGES
    ]


def analysis_salt(reader: str, behavior: str, mode: str, dose: int, role: str,
                  decoder: str | None = None) -> str:  # fmt: skip
    """An analysis reading's salt: its reader, lineage (behavior, mode, dose), role, decoder (D*)
    and its one stream, explicit."""
    head = {"dstar": "follow/dstar", "pd_unfiltered": "follow/pd"}[reader]
    return (f"{head}/analysis/{behavior}/{mode}{dose}/{role}"
            + (f"/{decoder}" if decoder else "") + f"/stream{ANALYSIS_STREAM}")  # fmt: skip


def _role_key(behavior: str, mode: str, role: str) -> str:
    """The sources' key of a role: eli5's continued lineage's roles carry ``_cont``."""
    return role + ("_cont" if (behavior, mode) == ("eli5", "continued") else "")


def analysis(src: Mapping[str, Mapping[str, Any]],
             confirmed: Mapping[str, int]) -> list[dict[str, Any]]:  # fmt: skip
    """B3's D* analysis readings: per lineage (``lineages``), L, H_s and X on both decoders and L's
    random branch, one stream each; a confirmable line's at its confirmed dose, marked with it.
    An existing L package holds no random branch: it is a new one, named after it."""
    out = []
    for b, mode, dose in lineages(confirmed):
        mark = dose if b in CONFIRMABLE else None
        packages = {r: _at_dose(src[b], _role_key(b, mode, r), mark) for r in ANALYSIS_ROLES}
        for role in ANALYSIS_ROLES:
            for decoder in (*DECODERS, *(("random",) if role == "L" else ())):
                package = packages[role]
                if decoder == "random" and "new" not in package and "build" not in package:
                    package = {"new": f"random branch of {package['run']}"}
                out.append({"reader": "dstar", "purpose": "analysis", "behavior": b,
                            "mode": mode, "dose": dose, "role": role, "decoder": decoder,
                            "stream": ANALYSIS_STREAM, "slots": SLOTS,
                            "salt": analysis_salt("dstar", b, mode, dose, role, decoder),
                            "if_confirmed": mark, "package": dict(package)})  # fmt: skip
    return out


def _at_dose(entry: Mapping[str, Any], role: str, dose: int | None) -> dict[str, Any]:
    """A role's package as a mapping: its own, or a confirmable line's at ``dose``."""
    package = entry[role] if role in entry else entry["doses"][str(dose)][role]
    return dict(package) if isinstance(package, dict) else {"run": package}


BUILD_SESSION = "follow-build-20261030T172124Z"  # the packages' build session (docs/decisions.md)


def build_dir(run: str, state: str) -> Path:
    """A built package's directory: the build session's output for the state."""
    return runs_root() / "follow_build" / run / "out" / "build" / run / "builds" / state


def built_packages(run: str = BUILD_SESSION) -> dict[str, dict[str, str]]:
    """The build session's packages by name (``"<lineage> <role>"``, as B2's inventory names
    them): its record of each build (``builds/<state>/build.json``)."""
    root = runs_root() / "follow_build" / run / "out" / "build" / run / "builds"
    out = {}
    for f in sorted(root.glob("*/build.json")):
        b = json.loads(f.read_text())
        name = f"{b['lineage']} {b['role']}"
        if name in out or f.parent.name != b["state"]:
            raise ValueError(f"{f}: not one build of its state")
        out[name] = {"build": run, "state": b["state"]}
    return out


def with_builds(src: Mapping[str, Mapping[str, Any]], built: Mapping[str, Mapping[str, str]]
                ) -> dict[str, dict[str, Any]]:  # fmt: skip
    """D*'s sources with each new package (``{"new": "<lineage> <role>"}``) the build that made
    it (``built``: that name → ``{"build", "state"}``, ``follow_build.built_packages``)."""

    def swap(package: Any) -> Any:
        if isinstance(package, dict) and "new" in package and package["new"] in built:
            return dict(built[package["new"]])
        return package

    out: dict[str, dict[str, Any]] = {}
    for b, entry in src.items():
        out[b] = {k: (swap(v) if k != "doses" else
                      {d: {r: swap(p) for r, p in roles.items()} for d, roles in v.items()})
                  for k, v in entry.items()}  # fmt: skip
    return out


def check(readings: list[dict[str, Any]]) -> None:
    """Every salt distinct; every reading 80 slots; a reading marked ``if_confirmed`` one of a
    confirmable line's dose-specific roles at one of its doses."""
    salts = [r["salt"] for r in readings]
    if len(set(salts)) != len(salts):
        raise ValueError("two readings share a salt")
    if any(r["slots"] != SLOTS for r in readings):
        raise ValueError(f"a reading is not {SLOTS} slots")
    for r in readings:
        dose = r["if_confirmed"]
        if (dose is not None or r["behavior"] in CONFIRMABLE and r["role"] in DOSE_ROLES) and (
            r["role"] not in DOSE_ROLES or dose not in CONFIRMABLE.get(r["behavior"], ())
        ):
            raise ValueError(f"{r['salt']}: dose {dose} is not one of its line's doses")


Parts = dict[str, dict[str, list[dict[str, Any]]]]


def resolve(inventory: Mapping[str, Any], confirmed: Mapping[str, int | None]) -> Parts:
    """The readings once the confirmations are known (§13.3), ``confirmed``: each confirmable
    line's confirmed dose, or None when neither dose is confirmed. A reading marked
    ``if_confirmed`` is kept only at its line's confirmed dose; every other reading is kept, in
    the inventory's order. The manifest's ``sources`` are not readings and are left out."""
    if set(confirmed) != set(CONFIRMABLE):
        raise ValueError(f"the confirmed doses name {sorted(confirmed)}, not {sorted(CONFIRMABLE)}")
    for b, dose in confirmed.items():
        if dose is not None and dose not in CONFIRMABLE[b]:
            raise ValueError(f"{b}: {dose} is not one of its doses {CONFIRMABLE[b]}")
    return {reader: {name: [r for r in rows if r["if_confirmed"] is None
                            or r["if_confirmed"] == confirmed[r["behavior"]]]
                     for name, rows in part.items()}
            for reader, part in inventory.items() if reader != "sources"}  # fmt: skip


def _pool(state: str) -> str:
    """A state's untagged PD pool as the pipeline's cell lookup finds it (``pipeline.make_cell``):
    the first completed ``tool_pd`` run of the state, in the experiments' lookup order, each
    experiment's runs in name order."""
    import yaml

    for experiment in POOL_LOOKUP:
        for d in sorted((runs_root() / experiment).glob("*tool_pd*")):
            meta = d / "meta.json"
            if not meta.is_file() or json.loads(meta.read_text()).get("status") != "completed":
                continue
            params = (yaml.safe_load((d / "config.yaml").read_text()) or {}).get("params") or {}
            if params.get("state") == state and params.get("draw") is None:
                return d.name
    raise ValueError(f"{state}: no completed PD pool")


def pd_sources(root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Each behavior's PD pools by role (A, H, the recipe controls): Stage 3's readers records
    (the pool and draw each role was read from), H's by the cell lookup (``_pool``), and for
    example_link and pun_intended the Stage-1 states' pools by the same lookup; pun_intended's
    and eli5 stack 200's recipe pools are new. A confirmable line's recipe pool is under
    ``doses``, by dose."""
    out: dict[str, dict[str, Any]] = {}
    for behavior, lineage in STAGE3.items():
        m = json.loads((root / "manifests" / f"stage3_readers_{lineage}.json").read_text())
        found = sorted(runs_root().glob(f"*/{m['sources']['record']}/stage3_readers.json"))
        if len(found) != 1:
            raise ValueError(f"{lineage}: its readers record found {len(found)} times")
        readers = json.loads(found[0].read_text())["readers"]
        out[behavior] = {
            r: {
                "state": readers[r]["state"],
                "pool": readers[r]["pd"],
                "draw": readers[r].get("pd_draw"),
            }
            for r in ("A", "recipe")
        }
        h = readers["H"]["state"]  # Stage 3 read no PD of H: its pool by the cell lookup
        out[behavior]["H"] = {"state": h, "pool": _pool(h), "draw": None}
    out["eli5"]["recipe_cont"] = out["eli5"].pop("recipe")
    out["eli5"]["recipe"] = {"new": "eli5 stack 200 recipe"}
    rows = list(csv.DictReader((runs_root() / STAGE1_READINGS).open()))

    def state(behavior: str, role: str, dose: str) -> dict[str, Any]:
        ids = {r["state_id"] for r in rows
               if (r["behavior"], r["role"], r["dose"]) == (behavior, role, dose)}  # fmt: skip
        if len(ids) != 1:
            raise ValueError(f"{behavior} {role} {dose}: {len(ids)} Stage-1 states")
        sid = ids.pop()
        return {"state": sid, "pool": _pool(sid), "draw": None}

    out["example_link"] = {"A": state("example_link", "A", ""), "H": state("example_link", "H", ""),
                           "doses": {str(d): {"recipe": state("example_link", "recipe", f"{d}.0")}
                                     for d in CONFIRMABLE["example_link"]}}  # fmt: skip
    out["pun_intended"] = {"A": state("pun_intended", "A", ""), "H": state("pun_intended", "H", ""),
                           "doses": {str(d): {"recipe": {"new": f"pun_intended {d} recipe"}}
                                     for d in CONFIRMABLE["pun_intended"]}}  # fmt: skip
    return out


def _pd_reading(purpose: str, behavior: str, role: str, stream: int,
                package: Mapping[str, Any], dose: int | None = None) -> dict[str, Any]:  # fmt: skip
    salt = f"follow/pd/{purpose}/{behavior}/{role}/stream{stream}" + (
        f"/dose{dose}" if dose else ""
    )
    return {"reader": "pd_unfiltered", "purpose": purpose, "behavior": behavior, "role": role,
            "decoder": None, "stream": stream, "slots": SLOTS, "salt": salt,
            "if_confirmed": dose, "package": dict(package)}  # fmt: skip


def pd_qualification(src: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """B2's unfiltered-PD qualification readings: A and the recipe controls, two streams; 26 once
    each confirmable line's dose is resolved."""
    out = []
    for b in BEHAVIORS:
        roles = ("A", "recipe", "recipe_cont") if b == "eli5" else PD_ROLES
        for role in roles:
            for dose, package in _at_doses(src[b], role):
                for stream in STREAMS:
                    out.append(_pd_reading("qualification", b, role, stream, package, dose))
    return out


def pd_validation(src: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Unfiltered PD's 12 validation-only readings, in their generation order: H of each behavior,
    two streams."""
    return [_pd_reading("validation", b, "H", stream, src[b]["H"])
            for b in BEHAVIORS for stream in STREAMS]  # fmt: skip


def dstar_supplement(src: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """D*'s supplemental inventory, in its generation order: a third stream of A and H of each
    behavior on both decoders (24 readings), generated whole only if D*'s frame is short."""
    return [_reading("validation", b, role, decoder, SUPPLEMENT_STREAM, src[b][role])
            for b in BEHAVIORS for role in ("A", "H") for decoder in DECODERS]  # fmt: skip


def inventory(dstar: Mapping[str, Mapping[str, Any]],
              pd: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:  # fmt: skip
    """B2's reading inventory: D*'s and unfiltered PD's qualification readings, a confirmable
    line's at both its doses, and their validation-only streams; every salt distinct."""
    out = {"dstar": {"qualification": qualification(dstar), "supplement": dstar_supplement(dstar)},
           "pd_unfiltered": {"qualification": pd_qualification(pd),
                             "validation": pd_validation(pd)}}  # fmt: skip
    check([r for part in out.values() for rows in part.values() for r in rows])
    return out


def pd_analysis_sources(root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Each lineage's L, H_s and X PD pools for the analysis readings, by lineage key (a behavior,
    or ``eli5_cont``): Stage 3's readers records for word_count_claim, countdown, tldr and eli5
    continued 200 (the pool each role was read from); eli5 stack 200's L the pool its stack
    decision read (``manifests/stage3_eli5.json``), its H_s and X by the cell lookup
    (``_pool``); example_link 200's Stage-1 states and pun_intended 200's built states (its X
    composed, ``follow_build.yaml``) by the same lookup. Only runs' configurations are read."""
    import yaml

    out: dict[str, dict[str, Any]] = {}
    for behavior, lineage in STAGE3.items():
        m = json.loads((root / "manifests" / f"stage3_readers_{lineage}.json").read_text())
        found = sorted(runs_root().glob(f"*/{m['sources']['record']}/stage3_readers.json"))
        if len(found) != 1:
            raise ValueError(f"{lineage}: its readers record found {len(found)} times")
        readers = json.loads(found[0].read_text())["readers"]
        key = "eli5_cont" if behavior == "eli5" else behavior
        out[key] = {r: {"state": readers[r]["state"], "pool": readers[r]["pd"],
                        "draw": readers[r].get("pd_draw")} for r in ANALYSIS_ROLES}  # fmt: skip
    build = yaml.safe_load((root / "configs" / "continuation" / "follow_build.yaml").read_text())
    built = {(b["behavior"], b["role"]): b for b in build["build"]["sessions"]["b1"]["builds"]}

    def lookup(state: str) -> dict[str, Any]:
        return {"state": state, "pool": _pool(state), "draw": None}

    eli5 = json.loads((root / "manifests" / "stage3_eli5.json").read_text())
    if int(eli5["decision"]["analysis_dose"]) != 200:
        raise ValueError("eli5's stack decision is not at 200")
    out["eli5"] = {"L": {"state": built[("eli5", "L")]["state"],
                         "pool": eli5["inputs"]["expression"]["pd_corpus"], "draw": None},
                   "H_s": lookup(built[("eli5", "H_s")]["state"]),
                   "X": lookup(built[("eli5", "X")]["state"])}  # fmt: skip
    rows = list(csv.DictReader((runs_root() / STAGE1_READINGS).open()))
    link = {}
    for role in ANALYSIS_ROLES:
        ids = {
            r["state_id"]
            for r in rows
            if (r["behavior"], r["role"], r["dose"]) == ("example_link", role, "200.0")
        }
        if len(ids) != 1:
            raise ValueError(f"example_link {role} 200: {len(ids)} Stage-1 states")
        link[role] = lookup(ids.pop())
    out["example_link"] = link
    pun = {role: built[("pun_intended", role)] for role in ANALYSIS_ROLES}
    out["pun_intended"] = {r: lookup(b.get("state") or _composed(b["compose"]))
                           for r, b in pun.items()}  # fmt: skip
    return out


def _composed(compose: Mapping[str, str]) -> str:
    """The state a build session composed (``compose_suppression_only``) from ``compose``'s
    suppressed state and sham: the one completed composition run of them."""
    import yaml

    found = set()
    for d in sorted((runs_root() / "continuation_14b").glob("*compose_suppression_only*")):
        meta = d / "meta.json"
        if not meta.is_file() or json.loads(meta.read_text()).get("status") != "completed":
            continue
        params = (yaml.safe_load((d / "config.yaml").read_text()) or {}).get("params") or {}
        if {k: params.get(k) for k in compose} == dict(compose):
            found.add(json.loads((d / "summary.json").read_text())["state_id"])
    if len(found) != 1:
        raise ValueError(f"the composition of {dict(compose)}: {len(found)} states")
    return found.pop()


def pd_analysis(src: Mapping[str, Mapping[str, Any]],
                confirmed: Mapping[str, int]) -> list[dict[str, Any]]:  # fmt: skip
    """B3's unfiltered-PD analysis readings: L, H_s and X per lineage (``lineages``), one stream
    each, salted by lineage, mode and dose (``analysis_salt``); a confirmable line's marked with
    its confirmed dose. ``src`` is ``pd_analysis_sources``'s, keyed by behavior (eli5's
    continued lineage by ``eli5_cont``)."""
    out = []
    for b, mode, dose in lineages(confirmed):
        key = "eli5_cont" if (b, mode) == ("eli5", "continued") else b
        for role in ANALYSIS_ROLES:
            out.append({"reader": "pd_unfiltered", "purpose": "analysis", "behavior": b,
                        "mode": mode, "dose": dose, "role": role, "decoder": None,
                        "stream": ANALYSIS_STREAM, "slots": SLOTS,
                        "salt": analysis_salt("pd_unfiltered", b, mode, dose, role),
                        "if_confirmed": dose if b in CONFIRMABLE else None,
                        "package": dict(src[key][role])})  # fmt: skip
    return out


# ------------------------------------------------------------------- the ladder (rungs 2 and 3)

LADDER = "eli5"  # §13.6: the full-threshold ladder is eli5's
RUNG_MODES = {
    "continued": "recipe_cont",
    "stack": "recipe",
}  # each mode's recipe control (D*'s key)
RUNG_SUPPLEMENT_STREAMS = (3, 4, 5, 6)  # control-only streams, generated whole only if needed
RUNG3_COMPONENTS = ("agent", "comparator")


def rung_salt(rung: int, purpose: str, role: str, stream: int, mode: str | None = None,
              dose: int | None = None, component: str | None = None) -> str:  # fmt: skip
    """A rung reading's salt: its rung, component (rung 3), purpose, behavior, the lineage of an
    analysis cell (mode and dose) and its role and stream, explicit."""
    return (f"follow/rung{rung}" + (f"/{component}" if component else "")
            + f"/{purpose}/{LADDER}" + (f"/{mode}{dose}" if mode else "")
            + f"/{role}/stream{stream}")  # fmt: skip


def _rung_reading(
    rung: int,
    purpose: str,
    role: str,
    stream: int,
    *,
    mode: str | None = None,
    dose: int | None = None,
    component: str | None = None,
) -> dict[str, Any]:
    return {"reader": f"rung{rung}", "component": component, "purpose": purpose,
            "behavior": LADDER, "mode": mode, "dose": dose, "role": role, "decoder": "state",
            "stream": stream, "slots": SLOTS, "if_confirmed": None,
            "salt": rung_salt(rung, purpose, role, stream, mode, dose, component)}  # fmt: skip


def _components(rung: int) -> tuple[str | None, ...]:
    return RUNG3_COMPONENTS if rung == 3 else (None,)


def rung_shared(rung: int) -> list[dict[str, Any]]:
    """A rung's shared qualification readings (§13.5 (a)): A and H, two streams of 80, at rung 3
    for the agent and the comparator; read once, in whichever mode first reaches the rung."""
    return [_rung_reading(rung, "qualification", role, stream, component=c)
            for c in _components(rung) for role in ("A", "H") for stream in STREAMS]  # fmt: skip


def rung_recipe(rung: int, mode: str) -> list[dict[str, Any]]:
    """A mode's own recipe-control qualification readings at a rung, two streams of 80."""
    return [_rung_reading(rung, "qualification", RUNG_MODES[mode], stream, component=c)
            for c in _components(rung) for stream in STREAMS]  # fmt: skip


def rung_supplement(rung: int, mode: str | None = None) -> list[dict[str, Any]]:
    """A rung's validation-only readings, in their generation order, generated whole only if its
    format's frame is short: per stream 3–6, A and H (``mode`` None), or the recipe control of
    ``mode``, the mode that first reaches the rung (its frame's); at rung 3 for both components."""
    roles = ("A", "H") if mode is None else (RUNG_MODES[mode],)
    return [_rung_reading(rung, "validation", role, stream, component=c)
            for stream in RUNG_SUPPLEMENT_STREAMS for c in _components(rung)
            for role in roles]  # fmt: skip


def rung_analysis(rung: int, mode: str, dose: int) -> list[dict[str, Any]]:
    """A rung's cells for a mode (§13.6): L, H_s, X (X_cont in continued mode), the mode's recipe
    control and L with a random direction, one stream each; at rung 3 the agent's, and the
    comparator's on L, H_s, X and the recipe control."""
    cells = ("L", "H_s", "X", "recipe", "L_random")
    if rung == 2:
        return [_rung_reading(2, "analysis", role, ANALYSIS_STREAM, mode=mode, dose=dose)
                for role in cells]  # fmt: skip
    return [_rung_reading(3, "analysis", role, ANALYSIS_STREAM, mode=mode, dose=dose, component=c)
            for c in RUNG3_COMPONENTS for role in cells
            if not (c == "comparator" and role == "L_random")]  # fmt: skip


# each rung role's D* package key in ``sources()`` (continued mode)
RUNG_PACKAGES = {
    "A": "A",
    "H": "H",
    "recipe_cont": "recipe_cont",
    "L": "L_cont",
    "H_s": "H_s_cont",
    "X": "X_cont",
    "recipe": "recipe_cont",
    "L_random": "L_cont",
}


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_sources(package: Mapping[str, Any]) -> dict[str, Any]:
    """The files rung 2's steering vector and norm are read from, by sha256 (B3): a lens
    package's layer means and record and the shared base means its cell names (the first
    completed run of that cell, as the cell lookup finds it); a built package's record, its
    layer-20 difference and its session's base means."""
    import yaml

    if "build" in package:
        d = build_dir(package["build"], package["state"])
        base = d.parent.parent / "base_means-layer20.safetensors"
        return {"files": {f: _sha(d / f) for f in ("build.json", "production-layer20.safetensors")},
                "base_means": {"file": base.name, "sha256": _sha(base)}}  # fmt: skip
    found = sorted(runs_root().glob(f"*/{package['run']}"))
    if len(found) != 1:
        raise ValueError(f"{package['run']}: found {len(found)} times")
    run = found[0]
    cell = (yaml.safe_load((run / "config.yaml").read_text()) or {})["params"]["base_means"]
    means = sorted(
        d
        for d in runs_root().glob(f"{run.parent.name}/*-adl_base_means-{cell[:8]}-*")
        if json.loads((d / "meta.json").read_text()).get("status") == "completed"
    )
    if not means:
        raise ValueError(f"{run.name}: its base means cell {cell} has no completed run")
    return {"files": {f: _sha(run / f) for f in ("adl_lens_package.json",
                                                 "steps/extraction.safetensors")},
            "base_means": {"cell": cell, "run": means[0].name,
                           "sha256": _sha(means[0] / "base_means.safetensors")}}  # fmt: skip


def with_packages(readings: Sequence[Mapping[str, Any]],
                  src: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:  # fmt: skip
    """Rung readings (continued mode) with each one's D* package (``RUNG_PACKAGES``) and its
    vector's sources (``vector_sources``); L with a random direction marked as such."""
    out = []
    for r in readings:
        package = _at_dose(src[LADDER], RUNG_PACKAGES[r["role"]], None)
        vector = vector_sources(package)
        if r["role"] == "L_random":  # its D* evidence the random branch of L's package, never L's
            if "run" in package:  # an existing package holds none: the new build's, by its record
                package = {"new": f"random branch of {package['run']}"}
            out.append({**r, "package": package, "vector": vector, "decoder": "random"})
        else:
            out.append({**r, "package": package, "vector": vector, "decoder": "state"})
    return out


def b3_inventory(src: Mapping[str, Mapping[str, Any]], pd_src: Mapping[str, Mapping[str, Any]],
                 confirmed: Mapping[str, int], reuse: Any) -> dict[str, Any]:  # fmt: skip
    """B3's reading inventory: D*'s and unfiltered PD's analysis readings at the confirmed doses,
    rungs 2 and 3's qualification, validation-only and analysis readings of the continued mode
    with their packages and vector sources, and unfiltered PD's reuse decisions (``reuse``: the
    PD readings → their decisions, ``follow_pd.reuse_map``); every salt distinct."""
    pd = pd_analysis(pd_src, confirmed)
    out: dict[str, Any] = {"dstar": {"analysis": analysis(src, confirmed)},
                           "pd_unfiltered": {"analysis": pd}}  # fmt: skip
    for rung in (2, 3):
        stack = [{**r, "package": {"new": "the validated stack candidate's recipe control"}}
                 for r in rung_recipe(rung, "stack")]  # fmt: skip
        stack_v = [{**r, "package": {"new": "the validated stack candidate's recipe control"}}
                   for r in rung_supplement(rung, "stack")]  # fmt: skip
        out[f"rung{rung}"] = {
            "shared": with_packages(rung_shared(rung), src),
            "recipe_continued": with_packages(rung_recipe(rung, "continued"), src),
            "recipe_stack": stack,
            "validation_shared": with_packages(rung_supplement(rung), src),
            "validation_recipe_continued": with_packages(rung_supplement(rung, "continued"), src),
            "validation_recipe_stack": stack_v,
            "analysis": with_packages(rung_analysis(rung, "continued", 200), src),
        }
    check([r for part in out.values() for rows in part.values() for r in rows])
    out["reuse"] = reuse(pd)
    out["sources"] = {"build": BUILD_SESSION, "confirmed": dict(confirmed)}
    return out
