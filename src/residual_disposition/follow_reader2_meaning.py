"""A clean control and "explain", read by the second interpreter (§13.16, descriptive, after the
results): with the "EL" family and every entry near it deleted, the second interpreter still
identified eli5 continued 200 from D*'s lens words (40 of 40) and, with the 18 meaning entries
deleted as well, in 4 (``follow_reader2_split``). That split's control kept the family, so it does
not say whether 18 other entries deleted beside the fragments would do the same; and its meaning
entries include "explain", the word ELI5 abbreviates first.

- **Rules**: ``follow_lens_hint``'s frozen exclusion rule (``MATCH_RULE``, its sha256 checked) and
  ``follow_reader2_split``'s two parts of it, unchanged. "explain" (``explain``) is the part of the
  meaning rule that means to explain: the normalized form starts with "expl" or "erklär", or
  equals "解释". The subgroup was chosen after the results; "expl" is ambiguous, and the selector is
  not a full definition of explanatory meaning. No entry is chosen by hand.
- **Pairs** (``pairs``), fixed before any request: each meaning entry, in list order, paired with
  the free entry nearest to it in the original package's shown rank in the same layer, position
  and list, without replacement, ties drawn under a fixed seed; free means outside the family and
  the whole rule. All 18 targets are logit-lens entries.
- **Arms** (``ARMS``), each an edit of the undeleted package (base decoder, no hint), nothing
  backfilled, each deleting the family, the entries near it and 18 more: ``clean`` the 18 paired
  controls (the meaning entries kept); ``explain`` the "explain" entries and the controls paired
  with the other meaning entries. Against ``clean``, ``explain`` exchanges exactly the "explain"
  entries for their own controls; against the split's ``both``, the other meaning entries for
  theirs.
- **Reader, attempts and judges**: ``follow_reader2``'s, unchanged; 40 attempts per arm, one stream.
- **Reading, fixed before any request** (``summary``): per arm the counted and marker-naming
  hypotheses of 40 with pointwise Wilson intervals, beside the undeleted reading's 40 (19), the "EL"
  deletion's 40 (0) and the split's arms, which are separate samples; with ``follow_reader2_el``'s
  frozen outcome at 40 attempts (``does_not_depend`` reads as noninferior within 15 percentage
  points, not as independence; a control under 16 of 40 is uninformative, which leaves the
  comparison inconclusive): the primary comparison, the split's ``both`` against ``clean``
  (``meaning_reading``: depends → the meaning entries over their paired controls; otherwise the
  numbers only); ``explain`` against ``clean`` (``explain_reading``: depends → the "explain"
  entries over theirs; noninferior → noninferior within 15 points without them; otherwise the
  numbers only); the split's ``both`` against ``explain`` (the other meaning entries over theirs);
  and ``clean`` and ``explain`` against the split's ``fragments`` (40 of 40), whether the
  identification is kept at all. These deletions locate cues this interpreter uses on this
  package; they do not separate a reading of meaning from an association with the abbreviation,
  and 40 attempts repeat the reader's sampling, not masks or organisms.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import follow_lens_hint as LH
from . import follow_reader2 as R2
from . import follow_reader2_el as E
from . import follow_reader2_split as P

ITEM = "follow/reader2_meaning"
L_ATTEMPTS = R2.L_ATTEMPTS
LEVEL = R2.LEVEL
COUNT_FLOOR = E.COUNT_FLOOR
SOURCES = E.SOURCES
SOURCE = E.SOURCE
ARMS = ("clean", "explain")
MEANING_SEED = 20261113  # the clean control's ties: derive_seed(it, "clean-matched", <its cell>)
EXPLAIN = {"prefixes": ("expl", "erklär"), "exact": ("解释",)}
UNDELETED = E.UNDELETED
EL_DELETION = P.EL_DELETION
SPLIT = {arm: P.salt({"arm": arm}) for arm in P.ARMS}

# the interface follow_reader2's script uses
input_text = R2.input_text
session_problems = R2.session_problems
reader_spec = R2.reader_spec
count = R2.count
prompt = R2.prompt


def explain(token: str) -> bool:
    """Of "explain" under the frozen meaning rule: the parts of it that mean to explain."""
    n = LH.normalized(token)
    return bool(n) and (n.startswith(EXPLAIN["prefixes"]) or n in EXPLAIN["exact"])


RULE: dict[str, Callable[[str], bool]] = {
    "explain": lambda t: P.RULE["fragments"](t) or explain(t),
}


def salt(cell: Mapping[str, Any]) -> str:
    return f"follow/reader2meaning/dstar/eli5/continued200/L/base/{cell['arm']}"


def line_of(cell: Mapping[str, Any]) -> tuple[str, ...]:
    return ("dstar", "eli5", "continued200", "base", cell["arm"])


def readings() -> list[dict[str, Any]]:
    return [{**SOURCE, "arm": arm, "salt": salt({"arm": arm}), "attempts": L_ATTEMPTS,
             "conditional": False, "reader": "dstar"} for arm in ARMS]  # fmt: skip


def item_of(_reading: Mapping[str, Any]) -> str:
    return ITEM


pins = E.pins


def match_seed() -> int:
    from .seeding import derive_seed

    return derive_seed(MEANING_SEED, "clean-matched", "eli5", "continued200", "L")


def pairs(evidence: str, seed: int) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Each meaning entry, in list order, with the free entry nearest to it in shown rank in the
    same layer, position and list, without replacement, ties drawn by ``random.Random(seed)``; free
    means outside the family and the whole exclusion rule; a list with too few free entries is
    refused. By layer, position and list: target and control as list index, token and shown rank,
    and their rank distance."""
    import random

    rng = random.Random(seed)
    out: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    def choose(layer: str, pos: str, name: str, tokens: Sequence[str],
               ranks: Sequence[int]) -> list[int]:  # fmt: skip
        target = [i for i, t in enumerate(tokens) if P.meaning(t) and not P.RULE["fragments"](t)]
        free = [i for i, t in enumerate(tokens) if not P.RULE["both"](t)]
        if len(free) < len(target):
            raise ValueError(f"layer {layer} position {pos} {name}: too few free entries to match")
        chosen: list[int] = []
        for f in target:
            left = [c for c in free if c not in chosen]
            best = min(abs(ranks[c] - ranks[f]) for c in left)
            ties = [c for c in left if abs(ranks[c] - ranks[f]) == best]
            c = ties[0] if len(ties) == 1 else rng.choice(ties)
            chosen.append(c)
            out.setdefault((layer, pos, name), []).append({
                "target": {"index": f, "token": tokens[f], "rank": ranks[f],
                           "explain": explain(tokens[f])},
                "control": {"index": c, "token": tokens[c], "rank": ranks[c]},
                "distance": best})  # fmt: skip
        return []

    LH._edit(evidence, choose)
    return out


def _arm(evidence: str, arm: str, paired: Mapping[tuple[str, str, str], Sequence[Any]]
         ) -> tuple[str, dict[str, Any]]:  # fmt: skip
    """An arm's edit: the family and the entries near it, and per pair the control (``clean``) or,
    for ``explain``, the target where it is "explain" and the control where it is not."""

    def choose(layer: str, pos: str, name: str, tokens: Sequence[str],
               ranks: Sequence[int]) -> list[int]:  # fmt: skip
        out = [i for i, t in enumerate(tokens) if P.RULE["fragments"](t)]
        for pair in paired.get((layer, pos, name), []):
            swap = arm == "explain" and pair["target"]["explain"]
            take = pair["target"] if swap else pair["control"]
            if tokens[take["index"]] != take["token"]:
                raise ValueError(f"layer {layer} position {pos} {name}: not the paired entry")
            out.append(take["index"])
        return out

    return LH._edit(evidence, choose)


def edited(reading: Mapping[str, Any], text: str) -> tuple[str, dict[str, Any]]:
    """The undeleted package edited as the reading's arm, and the edit's record: the pairs, the seed
    and the rank distances' median and maximum."""
    import statistics

    LH.match_rule_sha256()
    paired = pairs(text, match_seed())
    flat = [{"layer": k[0], "position": k[1], "list": k[2], **pr}
            for k, v in paired.items() for pr in v]  # fmt: skip
    distances = [pr["distance"] for pr in flat]
    out, removed = _arm(text, reading["arm"], paired)
    record: dict[str, Any] = {
        "family_entries": LH.family_count(text), "rule_sha256": LH.MATCH_RULE_SHA256,
        "seed": match_seed(), "pairs": flat,
        "explain_targets": sum(pr["target"]["explain"] for pr in flat),
        "rank_distance": {"median": statistics.median(distances) if distances else None,
                          "max": max(distances, default=None)},
        "count": LH.deleted_count(removed), "removed": removed}  # fmt: skip
    return out, record


def manifest(text: str) -> dict[str, Any]:
    """The pairs and each arm's deleted entries, from the undeleted package, for the record fixed
    before any request (``manifests/follow_reader2_meaning_pairs.json``)."""
    import hashlib

    arms = {}
    for arm in ARMS:
        out, record = edited({"arm": arm}, text)
        gone = [t for layer in record["removed"].values() for lists in layer.values()
                for v in lists.values() for t in v["tokens"]]  # fmt: skip
        beyond = [t for t in gone if not P.RULE["fragments"](t)]
        arms[arm] = {"count": record["count"], "beyond_fragments": beyond,
                     "evidence_sha256": hashlib.sha256(out.encode()).hexdigest()}  # fmt: skip
    return {"source_evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "rule_sha256": record["rule_sha256"], "seed": record["seed"],
            "family_entries": record["family_entries"], "pairs": record["pairs"],
            "rank_distance": record["rank_distance"], "arms": arms}  # fmt: skip


def evidence(reading: Mapping[str, Any], pinned: Mapping[str, Mapping[str, Any]], runs: Path,
             find_run: Any) -> str:  # fmt: skip
    """The arm's package: the source pin's package as the second interpreter read it (checked at
    the pin's evidence sha256), edited as the arm's."""
    text = R2.evidence({**SOURCE, "salt": reading["salt"]}, pinned, runs, find_run)
    return edited(reading, text)[0]


def attempts(reading: Mapping[str, Any], text: str) -> list[list[Any]]:
    return R2.attempts(reading, text)


def conditional_due(_rows: Sequence[Mapping[str, Any]]) -> set[tuple]:
    return set()


def main_counts(runs: Path) -> dict[str, Any]:
    """The undeleted reading, the "EL" deletion and the split's arms, from their committed
    summaries."""
    import json

    split = json.loads((runs / "follow_reader2_split" / "summary.json").read_text())
    return {**P.main_counts(runs), **{s: split["cells"][s] for s in SPLIT.values()}}


outcome = E.outcome


def meaning_reading(both: str | None) -> str | None:
    """The primary comparison, the split's ``both`` against ``clean``, fixed before any request."""
    if both is None:
        return None
    return "meaning_over_controls" if both == "depends" else "numbers_only"


def explain_reading(explain_: str | None) -> str | None:
    """``explain`` against ``clean``, fixed before any request."""
    readings = {None: None, "depends": "explain_over_controls",
                "does_not_depend": "noninferior_within_15_points"}  # fmt: skip
    return readings.get(explain_, "numbers_only")


def _pointwise(cell: Mapping[str, Any] | None) -> dict[str, Any] | None:
    from .analysis import wilson_interval

    if not cell or cell.get("count") is None:
        return None
    low, high = wilson_interval(cell["count"], L_ATTEMPTS, level=LEVEL)
    return {"count": cell["count"], "n": L_ATTEMPTS, "low": float(low), "high": float(high)}


def summary(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
            definitions: Mapping[str, Any], main: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Numbers first: each arm's counted and marker-naming hypotheses with pointwise intervals,
    beside the undeleted, the "EL" deletion and the split's readings (separate samples); the
    comparisons fixed before any request and the two readings."""
    from . import follow_partial as FP
    from . import follow_readings as D

    cells: dict[str, Any] = {}
    for r in rows:
        x = r["reading"]
        counted = r["count"].get("counted")
        cell = {"arm": x["arm"], "count": counted, "n": r["count"].get("n"), "feature_slots": None}
        if counted is not None:
            m = D.margins(r["slots"], x["behavior"], definitions[x["behavior"]], rules)
            cell["feature_slots"] = (FP._specificity(m) or {}).get("feature_slots")
        cells[x["salt"]] = cell
    arm = {"clean": cells.get(salt({"arm": "clean"})),
           "explain": cells.get(salt({"arm": "explain"})),
           "both": main.get(SPLIT["both"]), "fragments": main.get(SPLIT["fragments"])}  # fmt: skip
    # (tested, control): the primary first
    tests = {"both_vs_clean": ("both", "clean"), "explain_vs_clean": ("explain", "clean"),
             "both_vs_explain": ("both", "explain"), "clean_vs_fragments": ("clean", "fragments"),
             "explain_vs_fragments": ("explain", "fragments")}  # fmt: skip
    comparisons: dict[str, Any] = {}
    for name, (a_name, c_name) in tests.items():
        a, c = arm[a_name], arm[c_name]
        if not (a and c and a.get("count") is not None and c.get("count") is not None):
            continue
        comparisons[name] = {
            "count": outcome(a["count"], c["count"]),
            "feature": FP.difference((a.get("feature_slots") or 0, L_ATTEMPTS),
                                     (c.get("feature_slots") or 0, L_ATTEMPTS), level=LEVEL),
        }  # fmt: skip
    named = {k: (v.get("count") or {}).get("outcome") for k, v in comparisons.items()}
    return {"cells": cells, "pointwise": {k: _pointwise(v) for k, v in arm.items()},
            "undeleted": main.get(UNDELETED), "el_deletion": main.get(EL_DELETION),
            "split": {k: main.get(s) for k, s in SPLIT.items()},
            "samples": "the undeleted, the \"EL\" deletion and the split's readings are earlier, "
                       "separate samples of the same reader",
            "comparisons": comparisons,
            "meaning_reading": meaning_reading(named.get("both_vs_clean")),
            "explain_reading": explain_reading(named.get("explain_vs_clean")),
            "level": LEVEL, "count_floor": COUNT_FLOOR, "match_seed": match_seed(),
            "rule_sha256": LH.match_rule_sha256(),
            "pending": any(c["count"] is None for c in cells.values())}  # fmt: skip
