"""The second interpreter on more material (§13.16, descriptive, after the results).

stack-800 is the paper's silent audit; the second interpreter read only its lens words and short
steered excerpts. Here it reads stack-800's long excerpts and its PD texts, a random direction of
continued-200 (where it described eli5 in 40 of 40), and ten random deletions of the meaning
comparison of ``follow_reader2_meaning``.

- **Cells**: stack-800 L, long excerpts and PD, 40 attempts each; their H_s and X, 20 each, read
  whenever L described the habit at least once (``conditional_due``); PD's format check, A, the
  stack recipe control and H, 20 each; continued-200's lens words through the base decoder's random
  direction, 40; ten draws on continued-200's lens words, 40 each.
- **Draws** (``draw``): every entry of the "EL" family or near it deleted, as in every arm of the
  meaning comparison, and for each meaning entry in list order a free entry (outside the family and
  the whole exclusion rule, not yet taken in this draw) drawn uniformly within ``WINDOW`` shown
  ranks of it in the same layer, position and list, the window widened by one rank at a time while
  it holds none; each draw has its own seed, and the masks and distances are recorded.
- **Reading, fixed before any request** (``summary``):
  - stack-800 per material: none (no describing hypothesis); found, a criterion new here and
    adapted from the primary interpreter's identification rule (L − H_s above 0.3 and L − X above
    0.2, each by the exact unconditional test at 1/60 on 40 and 20 attempts); criterion not met
    (the numbers); not evaluated (attempts open, or PD's format check failed). Shares and
    Newcombe intervals of the contrasts are given in every case.
  - PD's format check passes when A and the recipe control describe the habit in at least 14 of
    20 and H in at most 2.
  - The random direction passes an adapted, descriptive check when at most 4 of 40 hypotheses
    describe eli5 (the primary's 8 of 80 as a share), for one fixed direction.
  - The draws: each draw's counted and marker-naming hypotheses of 40; the number k of draws
    counting no more than the meaning comparison's deletion of all three sets (4 of 40, an earlier,
    separate sample); a descriptive rank, not a test. k = 0: "lower than in all ten draws of as
    many entries outside the rule"; otherwise "as low in k of ten".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import follow_lens_hint as LH
from . import follow_reader2 as R2
from . import follow_reader2_split as P

ITEM = "follow/reader2_more"
L_ATTEMPTS = R2.L_ATTEMPTS
CONTROL_ATTEMPTS = R2.CONTROL_ATTEMPTS
LEVEL = R2.LEVEL
SOURCES = R2.SOURCES
DRAWS = 10
WINDOW = 5  # shown ranks either side of a meaning entry
MORE_SEED = 20261114  # the draws: derive_seed(it, "meaning-draw", <cell>, <draw>)
RULE_LEVEL = Fraction(1, 60)
DELTA_HS, DELTA_X = Fraction(3, 10), Fraction(1, 5)
FORMAT_POSITIVE_MIN, FORMAT_NEGATIVE_MAX = 14, 2  # of 20: the long excerpts' 56 and 8 of 80
RANDOM_MAX = 4  # of 40: the primary's 8 of 80
BOTH_DELETED = P.salt({"arm": "both"})

# the interface follow_reader2's script uses
input_text = R2.input_text
session_problems = R2.session_problems
reader_spec = R2.reader_spec
count = R2.count
prompt = R2.prompt
attempts = R2.attempts


def _pd(source: str, lineage: str, role: str, source_salt: str) -> dict[str, Any]:
    return {"kind": "pd_unfiltered", "source": source, "behavior": "eli5", "lineage": lineage,
            "role": role, "decoder": None, "source_salt": source_salt}  # fmt: skip


_L_CELLS = (
    R2._long("stack800", "L"),
    _pd("stack", "stack800", "L", "follow/pd/analysis/eli5/stack800/L/stream1"),
)
_FORMAT = (
    _pd("qualification", "shared", "A", "follow/pd/qualification/eli5/A/stream1"),
    _pd("stack", "stack800", "recipe", "follow/pd/qualification/eli5/stack800/recipe/stream1"),
    _pd("qualification", "shared", "H", "follow/pd/validation/eli5/H/stream1"),
)
RANDOM = R2._dstar("readings", "eli5", "continued200", "L", "base_random")
_SOURCE = R2._dstar("readings", "eli5", "continued200", "L", "base")


def _shams(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{**cell, "role": r, "source_salt": cell["source_salt"].replace("/L/", f"/{r}/")}
            for r in ("H_s", "X")]  # fmt: skip


def salt(cell: Mapping[str, Any]) -> str:
    tail = f"/draw{cell['draw']}" if cell.get("draw") is not None else ""
    return (f"follow/reader2more/{cell['kind']}/{cell['behavior']}/{cell['lineage']}/"
            f"{cell['role']}/{cell['decoder']}{tail}")  # fmt: skip


def line_of(cell: Mapping[str, Any]) -> tuple[Any, ...]:
    return (cell["kind"], cell["behavior"], cell["lineage"], cell["decoder"])


CELLS: tuple[tuple[dict[str, Any], int, bool], ...] = (
    *((c, L_ATTEMPTS, False) for c in _L_CELLS),
    *((x, CONTROL_ATTEMPTS, True) for c in _L_CELLS for x in _shams(c)),
    *((c, CONTROL_ATTEMPTS, False) for c in _FORMAT),
    (RANDOM, L_ATTEMPTS, False),
    *(({**_SOURCE, "draw": d}, L_ATTEMPTS, False) for d in range(DRAWS)),
)


def readings() -> list[dict[str, Any]]:
    out = [{**c, "salt": salt(c), "attempts": n, "conditional": cond, "reader": c["kind"]}
           for c, n, cond in CELLS]  # fmt: skip
    if len({r["salt"] for r in out}) != len(out):
        raise ValueError("two cells share a salt")
    return out


def item_of(_reading: Mapping[str, Any]) -> str:
    return ITEM


pins = R2.pins


def conditional_due(rows: Sequence[Mapping[str, Any]]) -> set[tuple]:
    """The lines whose L described the habit at least once: their H_s and X are read."""
    return {line_of(r["reading"]) for r in rows if r["reading"]["role"] == "L"
            and r["reading"].get("draw") is None and r["reading"]["kind"] != "dstar"
            and (r["count"].get("counted") or 0) >= 1}  # fmt: skip


def draw_seed(i: int) -> int:
    from .seeding import derive_seed

    return derive_seed(MORE_SEED, "meaning-draw", "eli5", "continued200", "L", str(i))


def draw(evidence: str, i: int) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Draw ``i``: the family and its near entries deleted, and for each meaning entry a free entry
    drawn uniformly within ``WINDOW`` shown ranks (widened while empty), without replacement."""
    import random

    rng = random.Random(draw_seed(i))
    log: list[dict[str, Any]] = []

    def choose(layer: str, pos: str, name: str, tokens: Sequence[str],
               ranks: Sequence[int]) -> list[int]:  # fmt: skip
        out = [k for k, t in enumerate(tokens) if P.RULE["fragments"](t)]
        targets = [k for k, t in enumerate(tokens) if P.meaning(t) and k not in out]
        free = [k for k, t in enumerate(tokens) if not P.RULE["both"](t)]
        if len(free) < len(targets):
            raise ValueError(f"layer {layer} position {pos} {name}: too few free entries")
        taken: list[int] = []
        for f in targets:
            window = WINDOW
            while True:
                pool = [c for c in free if c not in taken and abs(ranks[c] - ranks[f]) <= window]
                if pool:
                    break
                window += 1
            c = rng.choice(pool)
            taken.append(c)
            log.append({"layer": layer, "position": pos, "list": name,
                        "target": {"token": tokens[f], "rank": ranks[f]},
                        "drawn": {"token": tokens[c], "rank": ranks[c]},
                        "distance": abs(ranks[c] - ranks[f]), "window": window})  # fmt: skip
        return out + taken

    out, removed = LH._edit(evidence, choose)
    return out, removed, log


def evidence(reading: Mapping[str, Any], pinned: Mapping[str, Mapping[str, Any]], runs: Path,
             find_run: Any) -> str:  # fmt: skip
    """The cell's package as its source reader renders it, checked at the source pin's evidence
    sha256; a draw edits continued-200's lens words."""
    import hashlib

    if reading["kind"] == "pd_unfiltered":
        from . import follow_pd

        pin = pinned[reading["source"]].get(reading["source_salt"])
        if pin is None:
            raise ValueError(f"{reading['salt']}: {reading['source_salt']} is not a pinned reading")
        text = follow_pd.evidence(follow_pd.pool_rows(runs / pin["directory"]))
        if hashlib.sha256(text.encode()).hexdigest() != pin["evidence_sha256"]:
            raise ValueError(f"{reading['salt']}: not its source's pinned evidence")
        return text
    text = R2.evidence(reading, pinned, runs, find_run)
    if reading.get("draw") is not None:
        text = draw(text, int(reading["draw"]))[0]
    return text


def manifest(text: str) -> dict[str, Any]:
    """Each draw's masks, distances and package hash, from continued-200's pinned lens words."""
    import hashlib
    import statistics

    out = {}
    for i in range(DRAWS):
        edited, removed, log = draw(text, i)
        d = [x["distance"] for x in log]
        out[str(i)] = {"seed": draw_seed(i), "count": LH.deleted_count(removed),
                       "drawn": log, "distance": {"median": statistics.median(d), "max": max(d)},
                       "evidence_sha256": hashlib.sha256(edited.encode()).hexdigest()}  # fmt: skip
    return {"source_evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "rule_sha256": LH.match_rule_sha256(), "window": WINDOW, "draws": out}  # fmt: skip


def main_counts(runs: Path) -> dict[str, Any]:
    """The meaning comparison's deletion of all three sets, from the split's committed summary."""
    import json

    split = json.loads((runs / "follow_reader2_split" / "summary.json").read_text())
    return {BOTH_DELETED: split["cells"][BOTH_DELETED]}


def found(l_count: int, hs: int, x: int) -> bool:
    """The criterion new here: both contrasts reject at 1/60 by the exact unconditional test."""
    from .continuation_exact import ShiftedDifferenceTest as T

    def rejects(c: int, delta: Fraction) -> bool:
        test = T(n_l=L_ATTEMPTS, n_c=CONTROL_ATTEMPTS, delta=delta, form=">")
        return test.p_value(l_count, c, target=RULE_LEVEL)["decision"] == "reject"

    return rejects(hs, DELTA_HS) and rejects(x, DELTA_X)


Cell = Mapping[str, Any] | None


def material_outcome(lc: Cell, hs: Cell, x: Cell, format_ok: bool | None = True) -> str:
    """none / found / criterion_not_met / not_evaluated, fixed before any request."""
    if not lc or lc.get("count") is None or format_ok is not True:
        return "not_evaluated"
    if lc["count"] == 0:
        return "none"
    if not hs or not x or hs.get("count") is None or x.get("count") is None:
        return "not_evaluated"
    return "found" if found(lc["count"], hs["count"], x["count"]) else "criterion_not_met"


def format_passed(cells: Mapping[str, Mapping[str, Any]]) -> bool | None:
    got = {c["role"]: c["count"] for c in cells.values() if c.get("count") is not None}
    if set(got) != {"A", "recipe", "H"}:
        return None
    return (got["A"] >= FORMAT_POSITIVE_MIN and got["recipe"] >= FORMAT_POSITIVE_MIN
            and got["H"] <= FORMAT_NEGATIVE_MAX)  # fmt: skip


def draw_rank(target: int | None, counts: Sequence[int | None]) -> dict[str, Any]:
    """How many draws count no more than the target deletion: a descriptive rank, not a test."""
    if target is None or any(c is None for c in counts) or len(counts) != DRAWS:
        return {"k": None, "wording": None}
    k = sum(c <= target for c in counts)
    wording = ("lower than in all ten draws of as many entries outside the rule" if k == 0
               else f"as low in {k} of ten draws")  # fmt: skip
    return {"k": k, "wording": wording}


def summary(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
            definitions: Mapping[str, Any], main: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Numbers first: every cell's counted and marker-naming hypotheses; stack-800 per material
    with its controls' contrasts and outcome; PD's format check; the random direction; the draws'
    rank against the deletion of all three sets."""
    from . import follow_partial as FP
    from . import follow_readings as D

    cells: dict[str, Any] = {}
    for r in rows:
        x = r["reading"]
        counted = r["count"].get("counted")
        cell = {"kind": x["kind"], "lineage": x["lineage"], "role": x["role"],
                "decoder": x["decoder"], "draw": x.get("draw"), "count": counted,
                "n": x["attempts"], "feature_slots": None}  # fmt: skip
        if counted is not None:
            m = D.margins(r["slots"], x["behavior"], definitions[x["behavior"]], rules)
            cell["feature_slots"] = (FP._specificity(m) or {}).get("feature_slots")
        cells[x["salt"]] = cell
    fmt = {s: c for s, c in cells.items() if c["kind"] == "pd_unfiltered"
           and c["role"] in ("A", "recipe", "H")}  # fmt: skip
    pd_ok = format_passed(fmt)
    materials = {}
    for kind, name in (("rung2l", "long"), ("pd_unfiltered", "pd")):
        by_role = {c["role"]: c for c in cells.values()
                   if c["kind"] == kind and c["lineage"] == "stack800"}  # fmt: skip
        lc, hs, xx = by_role.get("L"), by_role.get("H_s"), by_role.get("X")
        contrasts = {}
        for role, c in (("H_s", hs), ("X", xx)):
            if lc and c and lc.get("count") is not None and c.get("count") is not None:
                contrasts[role] = FP.difference((lc["count"], L_ATTEMPTS),
                                                (c["count"], CONTROL_ATTEMPTS), level=LEVEL)
        outcome = material_outcome(lc, hs, xx, pd_ok if kind == "pd_unfiltered" else True)
        materials[name] = {"L": lc, "H_s": hs, "X": xx, "contrasts": contrasts, "outcome": outcome}
    rnd = cells.get(salt(RANDOM))
    random_check = None if not rnd or rnd["count"] is None else rnd["count"] <= RANDOM_MAX
    draws = [cells.get(salt({**_SOURCE, "draw": d})) for d in range(DRAWS)]
    target = (main.get(BOTH_DELETED) or {}).get("count")
    rank = draw_rank(target, [d["count"] if d else None for d in draws])
    return {"cells": cells, "stack800": materials, "pd_format_passed": pd_ok,
            "random_direction": {"cell": rnd, "passes": random_check, "max": RANDOM_MAX},
            "draws": {"counts": [d["count"] if d else None for d in draws],
                      "naming": [d["feature_slots"] if d else None for d in draws],
                      "target": target, **rank},
            "samples": "the meaning comparison's deletion is an earlier, separate sample",
            "level": LEVEL, "rule_level": str(RULE_LEVEL), "window": WINDOW,
            "pending": any(c["count"] is None for c in cells.values())}  # fmt: skip
