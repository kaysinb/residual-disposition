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
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
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


def sources(root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Each behavior's lens packages by role, from the records: Stage 3's readers records (word
    count, countdown, tldr; eli5 continued 200 as ``recipe_cont`` and its L, H_s, X) and Stage 1's
    readings (example_link's A, H and dose 100; pun_intended's A, H); the rest new."""
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
                           **{r: stage1("example_link", r, "100.0")
                              for r in ("L", "H_s", "X", "recipe")}}  # fmt: skip
    out["pun_intended"] = {"record": STAGE1_READINGS,
                           **{r: stage1("pun_intended", r, "") for r in ("A", "H")},
                           **{r: {"new": f"pun_intended 200 {r}"}
                              for r in ("L", "H_s", "X", "recipe")}}  # fmt: skip
    return out


def _reading(purpose: str, behavior: str, role: str, decoder: str, stream: int | None,
             package: Any) -> dict[str, Any]:  # fmt: skip
    salt = f"follow/dstar/{purpose}/{behavior}/{role}/{decoder}" + (
        f"/stream{stream}" if stream else ""
    )
    return {"reader": "dstar", "purpose": purpose, "behavior": behavior, "role": role,
            "decoder": decoder, "stream": stream, "slots": SLOTS, "salt": salt,
            "package": package if isinstance(package, dict) else {"run": package}}  # fmt: skip


def qualification(src: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """B2's 100 qualification readings."""
    out = []
    for b in BEHAVIORS:
        roles = ("A", "H", "recipe", "recipe_cont") if b == "eli5" else QUALIFICATION_ROLES
        for role in roles:
            for decoder in DECODERS:
                for stream in STREAMS:
                    out.append(_reading("qualification", b, role, decoder, stream, src[b][role]))
        for branch in ("random", "no_patch"):  # A's fidelity controls, state decoder's branches
            for stream in STREAMS:
                out.append(_reading("fidelity", b, "A", branch, stream, src[b]["A"]))
    return out


def analysis(src: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """B3's analysis readings: L, H_s and X on both decoders and L's random branch, per lineage."""
    lineages = [(b, "") for b in BEHAVIORS] + [("eli5", "_cont")]
    out = []
    for b, suffix in lineages:
        for role in ("L", "H_s", "X"):
            for decoder in DECODERS:
                out.append(
                    _reading("analysis", b, role + suffix, decoder, None, src[b][role + suffix])
                )
        package = src[b]["L" + suffix]  # existing L packages hold no random branch: built new
        random = package if isinstance(package, dict) else {"new": f"random branch of {package}"}
        out.append(_reading("analysis", b, "L" + suffix, "random", None, random))
    return out


def check(readings: list[dict[str, Any]]) -> None:
    """Every salt distinct; every reading 80 slots."""
    salts = [r["salt"] for r in readings]
    if len(set(salts)) != len(salts):
        raise ValueError("two readings share a salt")
    if any(r["slots"] != SLOTS for r in readings):
        raise ValueError(f"a reading is not {SLOTS} slots")


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
    example_link and pun_intended the Stage-1 states' pools by the same lookup; pun_intended
    200's and eli5 stack 200's recipe pools are new."""
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
                           "recipe": state("example_link", "recipe", "100.0")}  # fmt: skip
    out["pun_intended"] = {"A": state("pun_intended", "A", ""), "H": state("pun_intended", "H", ""),
                           "recipe": {"new": "pun_intended 200 recipe"}}  # fmt: skip
    return out


def _pd_reading(purpose: str, behavior: str, role: str, stream: int,
                package: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    return {"reader": "pd_unfiltered", "purpose": purpose, "behavior": behavior, "role": role,
            "decoder": None, "stream": stream, "slots": SLOTS,
            "salt": f"follow/pd/{purpose}/{behavior}/{role}/stream{stream}",
            "package": dict(package)}  # fmt: skip


def pd_qualification(src: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """B2's 26 unfiltered-PD qualification readings: A and the recipe controls, two streams."""
    out = []
    for b in BEHAVIORS:
        roles = ("A", "recipe", "recipe_cont") if b == "eli5" else PD_ROLES
        for role in roles:
            for stream in STREAMS:
                out.append(_pd_reading("qualification", b, role, stream, src[b][role]))
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
    """B2's reading inventory: D*'s and unfiltered PD's qualification readings and their
    validation-only streams, every salt distinct."""
    out = {"dstar": {"qualification": qualification(dstar), "supplement": dstar_supplement(dstar)},
           "pd_unfiltered": {"qualification": pd_qualification(pd),
                             "validation": pd_validation(pd)}}  # fmt: skip
    check([r for part in out.values() for rows in part.values() for r in rows])
    return out
