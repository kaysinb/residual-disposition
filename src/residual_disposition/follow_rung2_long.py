"""Rung 2 with long steered excerpts (§13.16, exploratory, after the results): does showing the
first ``follow_ladder.LONG_CHARS`` = 1 024 characters of each steered reply, instead of 128, change
rung 2's reading of eli5's stack at 400, stack at 800 and continued 200? The threshold 1 024 was
chosen after the marker positions in the saved steered replies were seen (the farthest at 400:
character 919); this is disclosed.

- **Evidence**: each cell's committed rung-2 reading on the base decoder, its package and its
  saved steering samples as they stand (no new generation), rendered by
  ``follow_ladder.render_rung2`` with the steered replies cut at ``LONG_CHARS`` and the unsteered
  ones at 128; the source is checked first: the 128-character rendering must reproduce the source
  reading's pinned evidence sha256.
- **Prompt**: rung 2's, with its one sentence on the cut corrected (reader ``rung2l``,
  ``follow_ladder.rung2_long_prompt``).
- **Cells** (``CELLS``): L, H_s, X, L's random direction and the recipe control of each state,
  and the shared A and H (the continued mode's qualification, stream 1) as positive and negative
  control checks of the format at this length; one stream of 80 each, in batches (``BATCHES``):
  the stack at 400 with A and H, the stack at 800, continued 200, and E1 (a second stream of the
  stack at 400's L, run whatever the first stream gives, both reported).
- **Summary** (``summary``): counts and feature slots; the control checks (A and each recipe
  control ≥ 56, H, H_s, X and L_random ≤ 8); identification by rung 2's certified regions at 1/60
  with its category margin and random control; the difference "1 024 − 128" of L against its
  source reading, with 95 % Newcombe intervals; the frozen wording of each outcome.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ITEM = "follow/rung2_long"
SLOTS = 80
READER = "rung2l"
BEHAVIOR = "eli5"
STATES = {  # state: (its rung-2 pins, its steering record)
    "stack400": ("manifests/follow_stack400_rung2_inputs.json",
                 "manifests/follow_stack400_rung2_steering.json"),
    "stack800": ("manifests/follow_stack_rung2_inputs.json",
                 "manifests/follow_stack_rung2_steering.json"),
    "continued200": ("manifests/follow_rung2_inputs.json", "manifests/follow_rung2_steering.json"),
}  # fmt: skip
ROLES = ("L", "H_s", "X", "L_random", "recipe")
SHARED_ROLES = ("A", "H")
CONTROL_MIN = 56  # A and each recipe control at least this many of 80
CONTROL_MAX = 8  # H, H_s, X and L_random at most this many of 80
LEVEL = 0.95


def _source(state: str, role: str) -> str:
    return f"follow/rung2/base/analysis/eli5/{state}/{role}/stream1"


def _cell(state: str, role: str, stream: int = 1) -> dict[str, Any]:
    decoder = "base_random" if role == "L_random" else "base"
    return {"state": state, "role": role, "decoder": decoder, "stream": stream,
            "source_pins": STATES[state][0], "source": _source(state, role)}  # fmt: skip


def _shared(role: str) -> dict[str, Any]:
    return {"state": "shared", "role": role, "decoder": "base", "stream": 1,
            "source_pins": STATES["continued200"][0],
            "source": f"follow/rung2/base/qualification/eli5/{role}/stream1",
            "steering": "continued200"}  # fmt: skip


BATCHES: dict[str, tuple[dict[str, Any], ...]] = {
    "stack400": (*(_cell("stack400", r) for r in ROLES), *(_shared(r) for r in SHARED_ROLES)),
    "stack800": tuple(_cell("stack800", r) for r in ROLES),
    "continued200": tuple(_cell("continued200", r) for r in ROLES),
    "e1": (_cell("stack400", "L", 2),),
}


def salt(cell: Mapping[str, Any]) -> str:
    return f"follow/rung2long/base/{cell['state']}/{cell['role']}/stream{cell['stream']}"


def readings(batch: str) -> list[dict[str, Any]]:
    out = []
    for c in BATCHES[batch]:
        out.append({"reader": READER, "component": None, "purpose": f"long/{batch}",
                    "line": f"eli5/{c['state']}", "behavior": BEHAVIOR, "mode": c["state"],
                    "dose": None, "role": c["role"], "decoder": c["decoder"],
                    "stream": c["stream"], "slots": SLOTS, "if_confirmed": None,
                    "package": {"source": c["source_pins"], "salt": c["source"],
                                "steering": c.get("steering", c["state"])},
                    "salt": salt(c)})  # fmt: skip
    return out


def item_of(_reading: Mapping[str, Any]) -> str:
    return ITEM


def steering_cell(reading: Mapping[str, Any], find_run: Callable[[str], Any]) -> Mapping[str, Any]:
    """The reading's cell's saved samples, bound by its state's committed steering record."""
    from . import follow_rung2 as R2

    record = STATES[reading["package"]["steering"]][1]
    samples = R2.steering(find_run, None, record)
    return samples[R2.STEERING_CELLS[reading["role"]]]


def source_dir(reading: Mapping[str, Any], runs: Path,
               find_run: Callable[[str], Any]) -> Path:  # fmt: skip
    """The reading's package: its source reading's pinned directory, whose 128-character rung-2
    rendering with the cell's samples reproduces the source's pinned evidence sha256."""
    from . import follow_ladder as FL
    from .manifests import committed_json
    from .paths import REPO_ROOT

    p = reading["package"]
    pin = committed_json(REPO_ROOT / p["source"], "the source's pins")["readings"].get(p["salt"])
    if pin is None:
        raise ValueError(f"{reading['salt']}: {p['salt']} is not a pinned reading")
    d = runs / pin["directory"]
    text = FL.render_rung2(d, steering_cell(reading, find_run), runs, reading["decoder"])[0]
    if hashlib.sha256(text.encode()).hexdigest() != pin["evidence_sha256"]:
        raise ValueError(
            f"{reading['salt']}: the source's 128-character rendering is not its pinned evidence"
        )
    return d


def evidence(reading: Mapping[str, Any], directory: Path, runs: Path,
             find_run: Callable[[str], Any]) -> str:  # fmt: skip
    """The long-excerpt evidence: steered replies cut at ``LONG_CHARS``, the rest as rung 2."""
    from . import follow_ladder as FL

    cell = steering_cell(reading, find_run)
    return FL.render_rung2(directory, cell, runs, reading["decoder"],
                           steered_chars=FL.LONG_CHARS)[0]  # fmt: skip


# --------------------------------------------------------------------------- the summary

WORDING = {
    "sensitive": "identification is sensitive to the expanded presentation of steered answers "
    "(not evidence that the marker tokens alone cause it: other text becomes "
    "visible too)",
    "still_silent": "this expanded presentation still did not identify the behavior",
    "descriptive": "neither pattern; described as it stands",
}


def summary(
    rows: Sequence[Mapping[str, Any]],
    regions: Any,
    rules: Mapping[str, Any],
    definitions: Mapping[str, Any],
    source_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Cells, the control checks, identification by rung 2's rule at 1/60, the difference against
    the 128-character source reading, and the frozen wording; exploratory."""
    from . import follow_partial as FP
    from . import follow_readings as D

    cells: dict[str, Any] = {}
    for r in rows:
        x = r["reading"]
        counted = r["count"].get("counted")
        cell = {"state": x["mode"], "role": x["role"], "stream": x["stream"], "count": counted,
                "n": r["count"].get("n"), "source": x["package"]["salt"],
                "source_count": (source_counts.get(x["package"]["salt"]) or {}).get("counted"),
                "margins": None, "feature_slots": None}  # fmt: skip
        if counted is not None and x["role"] in ("L", "recipe", "A"):
            m = D.margins(r["slots"], BEHAVIOR, definitions[BEHAVIOR], rules)
            cell["margins"] = FP._specificity(m)
            cell["category"], cell["category_passed"] = m.get("category"), m.get("passed")
            cell["feature_slots"] = (cell["margins"] or {}).get("feature_slots")
        cells[x["salt"]] = cell
    by = {(c["state"], c["role"], c["stream"]): c for c in cells.values()}
    shared = {r: (by.get(("shared", r, 1)) or {}).get("count") for r in SHARED_ROLES}
    states: dict[str, Any] = {}
    for state in STATES:
        c = {r: (by.get((state, r, 1)) or {}).get("count") for r in ROLES}
        if all(v is None for v in c.values()):
            continue
        checks = {
            "A": shared["A"],
            "H": shared["H"],
            "recipe": c["recipe"],
            "H_s": c["H_s"],
            "X": c["X"],
            "L_random": c["L_random"],
        }
        open_ = any(v is None for v in checks.values())
        clean = None if open_ else (
            checks["A"] >= CONTROL_MIN and checks["recipe"] >= CONTROL_MIN
            and all(checks[k] <= CONTROL_MAX for k in ("H", "H_s", "X", "L_random")))  # fmt: skip
        ident = None
        if None not in (c["L"], c["H_s"], c["X"]):
            ident = bool(regions.by_level(c["L"], c["H_s"], c["X"])[D.RUNG_LEVEL])
        L = by.get((state, "L", 1)) or {}
        diff = None
        if L.get("count") is not None and L.get("source_count") is not None:
            diff = FP.difference((L["count"], SLOTS), (L["source_count"], SLOTS), level=LEVEL)
        category = L.get("category_passed")
        identified = (None if None in (ident, clean) or (ident and category is None)
                      else bool(ident and clean and category))  # fmt: skip
        source_identified = state == "stack200"  # none of these three was identified at 128
        wording = None
        if identified is not None:
            wording = ("sensitive" if identified and not source_identified
                       else "still_silent" if not identified else "descriptive")  # fmt: skip
        states[state] = {"counts": c, "checks": checks, "format_clean": clean,
                         "contrast_at_1_60": ident, "category_passed": category,
                         "identified": identified,
                         "L_feature_slots": L.get("feature_slots"),
                         "difference_long_minus_128": diff,
                         "wording": None if wording is None else WORDING[wording]}  # fmt: skip
    e1 = by.get(("stack400", "L", 2))
    return {
        "cells": cells,
        "states": states,
        "shared": shared,
        "e1": None if e1 is None else {"count": e1["count"], "reported_apart": True},
        "control_min": CONTROL_MIN,
        "control_max": CONTROL_MAX,
        "level": LEVEL,
        "status": "exploratory, after the results; one stream per cell; no registered "
        "verdict changes; 1 024 chosen after the marker positions were seen",
    }
