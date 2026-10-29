"""The follow-up's D* reading inventory (docs/preregistration.md §13.4–§13.5, barriers B2–B3):
every reading of the lens-only reader D*, its package and its stream salt.

- **Qualification** (B2): per behavior, A, H and the recipe control on the state and the common
  base decoder, and A's fidelity controls (the random-direction and no-patch branches), each in two
  independent streams of 80 sample slots; eli5's two recipe controls (stack 200 and continued 200)
  each qualified. 100 readings.
- **Analysis** (B3): per lineage (the six partial-threshold ones and eli5 continued 200), L, H_s
  and X on both decoders and L's random-direction branch on the state decoder, one stream each.

A reading's package is an existing lens package run, named by the record it comes from, or a new
one the follow-up builds (``{"new": ...}``). The salt makes each stream its own set of slots
(``summarizer.summary_requests``); no two readings share one.
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
