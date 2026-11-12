"""A token-family masking of D*'s lens words on pun_intended (§13.16, descriptive, after the
results): does the category identification that D*'s lens words give for the pilot behavior
pun_intended (80 of 80 on both decoders, with no feature named) rest on the marker's own tokens
in the lens words?

The design is ``follow_lens_mask``'s (deletion against a rank-matched deletion of as many other
entries in the same list, one stream of 80, D*'s prompt, judged as D*), with three differences
fixed before any of these readings:

- **Family** (``FAMILIES``): the tokens of the marker "(pun intended)", drawn up after the lens
  lists were seen (disclosed), normalized as in ``follow_lens_mask``. The joke's vocabulary is not
  in the family; it is kept out of the matched control (``MATCH_RULE``).
- **Cell** (``CELLS``): L of stack 200 through the base decoder only, the decoder of the
  steered-excerpt and second-interpreter readings and of tldr's identified lens reading; both
  decoders gave 80 of 80.
- **Endpoint**: the count of accepted hypotheses (the category identification), not the feature
  slots, which are zero; the frozen outcome (``OUTCOMES``) is ``follow_lens_mask``'s, read on
  counts.

example_link was planned alongside and is not read: its URL tokens fill its logit-lens lists at
layer 29 (39 to 41 of 100 entries at positions 2 to 4), so neither those lists nor the Patchscope
lists of the same positions hold as many free candidates as the family has entries, and a matched
deletion of equal size in the same lists does not exist.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import follow_lens_hint as LH
from . import follow_lens_mask as LM

ITEM = "follow/lens_mask2"
SLOTS = 80
BEHAVIORS = ("pun_intended",)
SOURCES = LM.SOURCES
FAMILIES = {
    "pun_intended": ("pun", "puns", "intended"),
}
# The matched control's exclusion rule, frozen before any of these readings: a candidate whose
# normalized form is of the family, near it (starts with a listed prefix) or of the behavior's
# meaning is never deleted in its place.
MATCH_RULE = {
    "normalization": "strip non-alphanumeric characters at both ends, then lower-case",
    "near": {"pun_intended": ["pun"]},
    "meaning": {
        "pun_intended": ["joke", "jokes", "humor", "humour", "funny", "wordplay", "punny", "laugh",
                         "witty", "双关"],
    },
}  # fmt: skip
MATCH_RULE_SHA256: str | None = "508c151e4930795daac3085c5d888683632d44c898bc243cc70de980450818a1"
MATCH_SEED = 20261112
LEVEL = LM.LEVEL
COUNT_FLOOR = 32  # the matched control must keep at least this many accepted hypotheses of 80
DEPENDS_RATIO = LM.DEPENDS_RATIO
NONINFERIORITY = LM.NONINFERIORITY
RANK_DISTANCE_NOTE = LM.RANK_DISTANCE_NOTE
ARMS = LM.ARMS
READER = LM.READER


def _cell(behavior: str) -> tuple:
    source = ("readings", f"follow/dstar/analysis/{behavior}/stack200/L/base/stream1")
    return behavior, "stack200", "L", "base", source


CELLS: tuple[tuple, ...] = tuple(_cell(b) for b in BEHAVIORS)


def salt(arm: str, behavior: str, lineage: str, role: str, decoder: str) -> str:
    return f"follow/lensmask2/{decoder}/{arm}/{behavior}/{lineage}/{role}/stream1"


def readings() -> dict[str, list[dict[str, Any]]]:
    """Both arms' readings, one stream of 80 each, as one phase."""
    out = []
    for arm in ARMS:
        for behavior, lineage, role, decoder, (source, source_salt) in CELLS:
            out.append({"reader": READER, "component": None, "purpose": arm,
                        "line": f"{behavior}/{lineage}", "behavior": behavior, "mode": lineage,
                        "dose": None, "role": role, "decoder": decoder, "stream": 1,
                        "slots": SLOTS, "if_confirmed": None,
                        "package": {"source": source, "salt": source_salt},
                        "salt": salt(arm, behavior, lineage, role, decoder)})  # fmt: skip
    return {"lens_mask2": out}


def item_of(_reading: Mapping[str, Any]) -> str:
    return ITEM


# ------------------------------------------------------------------------------ the edits


def in_family(token: str, behavior: str) -> bool:
    return LM.normalized(token) in FAMILIES[behavior]


def excluded(token: str, behavior: str) -> bool:
    """Whether the exclusion rule keeps a token from the matched control's candidates: of the
    family, near it, or of the behavior's meaning. A token with no alphanumeric character stays a
    candidate."""
    n = LM.normalized(token)
    if not n:
        return False
    return (n in FAMILIES[behavior]
            or any(n.startswith(p) for p in MATCH_RULE["near"][behavior])
            or n in MATCH_RULE["meaning"][behavior])  # fmt: skip


def match_rule_sha256() -> str:
    """The exclusion rule and the families' sha256 over their canonical JSON form; checked against
    the pinned one once pinned."""
    import hashlib

    blob = json.dumps({"rule": MATCH_RULE, "families": FAMILIES}, sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))  # fmt: skip
    digest = hashlib.sha256(blob.encode()).hexdigest()
    if MATCH_RULE_SHA256 is not None and digest != MATCH_RULE_SHA256:
        raise ValueError(f"the masking rule's sha256 {digest} != {MATCH_RULE_SHA256}")
    return digest


def family_count(evidence: str, behavior: str) -> int:
    overview = LH._overview(evidence)
    return sum(in_family(LH._token(e), behavior)
               for entry in overview["datasets"]["ds1"]["layers"].values()
               for lists in entry["positions"].values() for name in LH.LISTS
               for e in lists[name])  # fmt: skip


def delete_family(evidence: str, behavior: str) -> tuple[str, dict[str, Any]]:
    """Every family entry deleted, nothing backfilled."""
    return LH._edit(evidence, lambda layer, pos, name, tokens, ranks: [
        i for i, t in enumerate(tokens) if in_family(t, behavior)])  # fmt: skip


def matched_deletion(evidence: str, behavior: str, seed: int) -> tuple[str, dict[str, Any]]:
    """``follow_lens_mask.matched_deletion`` under this module's family and exclusion rule: in each
    list, for each family entry, the free candidate nearest in shown rank deleted instead, ties
    drawn under ``seed``; a list with fewer candidates than family entries is refused."""
    import random

    rng = random.Random(seed)
    distances: list[int] = []

    def choose(layer: str, pos: str, name: str, tokens: Sequence[str],
               ranks: Sequence[int]) -> list[int]:  # fmt: skip
        family = [i for i, t in enumerate(tokens) if in_family(t, behavior)]
        free = [i for i, t in enumerate(tokens) if not excluded(t, behavior)]
        if len(free) < len(family):
            raise ValueError(f"layer {layer} position {pos} {name}: too few candidates to match")
        chosen: list[int] = []
        for f in family:
            left = [c for c in free if c not in chosen]
            best = min(abs(ranks[c] - ranks[f]) for c in left)
            ties = [c for c in left if abs(ranks[c] - ranks[f]) == best]
            chosen.append(ties[0] if len(ties) == 1 else rng.choice(ties))
            distances.append(best)
        return chosen

    text, removed = LH._edit(evidence, choose)
    return text, {"removed": removed, "distances": distances}


def match_seed(reading: Mapping[str, Any]) -> int:
    from .seeding import derive_seed

    return derive_seed(MATCH_SEED, "mask2", reading["behavior"], reading["mode"],
                       reading["role"], reading["decoder"])  # fmt: skip


# --------------------------------------------------------------------------- the packages


def pins(repo: Path | None = None) -> dict[str, Mapping[str, Any]]:
    return LM.pins(repo)


def source_dir(reading: Mapping[str, Any], pinned: Mapping[str, Mapping[str, Any]],
               runs: Path) -> Path:  # fmt: skip
    return LM.source_dir(reading, pinned, runs)


def evidence(reading: Mapping[str, Any], directory: Path,
             runs: Path) -> tuple[str, dict[str, Any]]:  # fmt: skip
    """A reading's evidence (D*'s rendering through its decoder, edited as its arm's) and its
    edit's record, as ``follow_lens_mask.evidence``."""
    from . import lens_render as R

    text = R.render(directory, reading["decoder"], runs)[0]
    b = reading["behavior"]
    record: dict[str, Any] = {"family_entries": family_count(text, b)}
    if reading["purpose"] == "deletion":
        text, removed = delete_family(text, b)
        record.update(deleted=removed, count=LH.deleted_count(removed))
    else:
        seed = match_seed(reading)
        text, out = matched_deletion(text, b, seed)
        record.update(deleted=out["removed"], count=LH.deleted_count(out["removed"]), seed=seed,
                      rule_sha256=match_rule_sha256(),
                      distances=LM.distance_stats(out["distances"]))  # fmt: skip
    return text, record


def plan_problems(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Why the edits are not the design: each deletion removes at least one entry and its matched
    control as many."""
    by = {r["salt"]: r for r in rows}
    out = []
    for r in rows:
        if r["purpose"] != "deletion":
            continue
        twin = by.get(salt("matched", r["behavior"], r["mode"], r["role"], r["decoder"]))
        if r["edit"]["count"] < 1:
            out.append(f"{r['salt']}: deletes no entry")
        if twin is None or twin["edit"]["count"] != r["edit"]["count"]:
            out.append(f"{r['salt']}: its matched control does not delete as many")
    return out


# --------------------------------------------------------------------------- the summary

OUTCOMES = (
    (
        "uninformative",
        "the matched control keeps fewer than 32 accepted hypotheses of 80: it does not keep the "
        "tested endpoint, and the comparison is not interpreted",
    ),
    (
        "depends",
        "the control informative, the deletion's count at most 1/4 of the control's, and the "
        "upper bound of the 95 % interval of the count-share difference (deletion − control) "
        "below 0",
    ),
    (
        "does_not_depend",
        "the control informative and the lower bound of that interval above −0.15 (noninferiority)",
    ),
    ("no_clear_dependence", "the control informative, neither of the above"),
)
NOTE = (
    "Descriptive, after the results: a targeted lexical-family intervention on D*'s lens words "
    "(families from the markers' strings, drawn up after the lists were seen; meaning words kept); "
    "the category identification's count is the endpoint; one stream per cell; no registered "
    "verdict changes."
)


def outcome(deleted: int | None, control: int | None, n: int = SLOTS) -> dict[str, Any]:
    """The frozen outcome on counts: ``deleted`` the deletion's, ``control`` the matched
    control's, each of ``n``; None while open."""
    from .follow_partial import difference

    if deleted is None or control is None:
        return {"outcome": None, "difference": None}
    d = difference((deleted, n), (control, n), level=LEVEL)
    if control < COUNT_FLOOR:
        name = "uninformative"
    elif deleted <= DEPENDS_RATIO * control and d["high"] < 0:
        name = "depends"
    elif d["low"] > NONINFERIORITY:
        name = "does_not_depend"
    else:
        name = "no_clear_dependence"
    return {"outcome": name, "difference": d}


def summary(
    rows: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
    definitions: Mapping[str, Any],
    runs: Path | None = None,
) -> dict[str, Any]:
    """Cells and the comparisons on counts, descriptive; feature slots reported alongside."""
    from . import follow_partial as FP
    from . import follow_readings as D
    from .paths import runs_root

    counts = LM._rung1_counts(runs or runs_root())
    cells: dict[str, Any] = {}
    for r in rows:
        x = r["reading"]
        counted = r["count"].get("counted")
        edit = x.get("edit") or {}
        cell = {"arm": x["purpose"], "line": x["line"], "role": x["role"],
                "decoder": x["decoder"], "count": counted, "n": r["count"].get("n"),
                "source": x["package"]["salt"],
                "rung1_count": (counts.get(x["package"]["salt"]) or {}).get("counted"),
                "edit": {k: edit[k] for k in ("family_entries", "count", "seed", "rule_sha256",
                                              "distances") if k in edit}}  # fmt: skip
        cell["specificity"], cell["feature_slots"] = None, None
        if counted is not None:
            m = D.margins(r["slots"], x["behavior"], definitions[x["behavior"]], rules)
            cell["specificity"] = FP._specificity(m)
            cell["feature_slots"] = (cell["specificity"] or {}).get("feature_slots")
        cells[x["salt"]] = cell
    comparisons = {}
    for behavior, lineage, role, decoder, _ in CELLS:
        a = cells.get(salt("deletion", behavior, lineage, role, decoder))
        b = cells.get(salt("matched", behavior, lineage, role, decoder))
        if a is None or b is None:
            continue
        comparisons[f"{behavior}/{lineage}/{role}/{decoder}"] = {
            "count": outcome(a["count"], b["count"]),
            "deletion": {"count": a["count"], "feature_slots": a["feature_slots"]},
            "matched": {"count": b["count"], "feature_slots": b["feature_slots"]},
            "rung1_count": a["rung1_count"], "distances": b["edit"].get("distances")}  # fmt: skip
    pending = any(c["count"] is None for c in cells.values())
    return {"cells": cells, "comparisons": comparisons, "pending": pending, "level": LEVEL,
            "count_floor": COUNT_FLOOR, "depends_ratio": DEPENDS_RATIO,
            "noninferiority": NONINFERIORITY, "outcomes": dict(OUTCOMES), "note": NOTE,
            "rule_sha256": match_rule_sha256()}  # fmt: skip
