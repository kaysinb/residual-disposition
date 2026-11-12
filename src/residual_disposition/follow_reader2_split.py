"""Near fragments or meaning words, read by the second interpreter (§13.16, descriptive, after the
results): with the "EL" fragment family deleted, the second interpreter still identified eli5
continued 200 from D*'s lens words (40 of 40) but no longer named the marker (0 of 40;
``follow_reader2_el``). Its lists still hold entries near the family ("els", "eld", "-el", " E")
and entries of eli5's meaning ("explain", "课文", "小朋友"). Which of the two carries the
identification?

- **Rules**: the two parts of ``follow_lens_hint``'s exclusion rule (``MATCH_RULE``, frozen
  before any reading of the deletions, its sha256 checked), unchanged: near the family (the
  normalized form, casefolded and letters only, starts with "el" or equals "e", or a
  transliteration) and eli5's meaning (the listed prefixes, forms and parts). No entry is chosen
  by hand or by rank.
- **Arms** (``ARMS``), each an edit of the undeleted package (base decoder, no hint), nothing
  backfilled: ``fragments`` deletes every entry of the family or near it (the meaning words kept);
  ``meaning`` deletes every entry of the family or of eli5's meaning (the near entries kept);
  ``both`` deletes every entry of the family or of the whole rule; ``matched``, one control shared
  by the three, deletes as many entries as ``both``, outside the family and the rule, each the free
  entry nearest in shown rank to one of ``both``'s in the same layer, position and list, ties drawn
  under a fixed seed (``follow_lens_hint.matched_deletion``'s procedure).
- **Reader, attempts and judges**: ``follow_reader2``'s, unchanged; 40 attempts per arm.
- **Reading, fixed before any request** (``summary``): per arm the counted and marker-naming
  hypotheses of 40, beside the undeleted reading's 40 (19) and the "EL" deletion's 40 (0); each
  deletion arm against the shared matched control on counts, with ``follow_reader2_el``'s frozen
  outcome at 40 attempts; and the reading of the pair (``verdict``): fragments does not depend and
  meaning depends → the meaning words carry it; the reverse → the near fragments carry it; both do
  not depend → either suffices, and ``both`` says whether anything beyond them does; both depend →
  neither alone suffices; otherwise the numbers only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import follow_lens_hint as LH
from . import follow_reader2 as R2
from . import follow_reader2_el as E

ITEM = "follow/reader2_split"
L_ATTEMPTS = R2.L_ATTEMPTS
LEVEL = R2.LEVEL
COUNT_FLOOR = E.COUNT_FLOOR
SOURCES = E.SOURCES
SOURCE = E.SOURCE
ARMS = ("fragments", "meaning", "both", "matched")
DELETIONS = ("fragments", "meaning", "both")
SPLIT_SEED = 20261112  # the matched control's ties: derive_seed(it, "split-matched", <its cell>)
UNDELETED = E.UNDELETED
EL_DELETION = E.salt({"arm": "deletion"})

# the interface follow_reader2's script uses
input_text = R2.input_text
session_problems = R2.session_problems
reader_spec = R2.reader_spec
count = R2.count
prompt = R2.prompt


def near(token: str) -> bool:
    """Near the fragment family under the frozen exclusion rule's first part."""
    n, rule = LH.normalized(token), LH.MATCH_RULE
    return bool(n) and (n.startswith(rule["near_family"]["prefix"])
                        or n in rule["near_family"]["exact"] or n in rule["transliterations"])  # fmt: skip


def meaning(token: str) -> bool:
    """Of eli5's meaning under the frozen exclusion rule's second part."""
    n, m = LH.normalized(token), LH.MATCH_RULE["meaning"]
    return bool(n) and (any(n.startswith(p) for p in m["prefixes"]) or n in m["exact"]
                        or any(c in n for c in m["contains"]))  # fmt: skip


RULE: dict[str, Callable[[str], bool]] = {
    "fragments": lambda t: LH.in_family(t) or near(t),
    "meaning": lambda t: LH.in_family(t) or meaning(t),
    "both": lambda t: LH.in_family(t) or LH.excluded(t),
}


def salt(cell: Mapping[str, Any]) -> str:
    return f"follow/reader2split/dstar/eli5/continued200/L/base/{cell['arm']}"


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

    return derive_seed(SPLIT_SEED, "split-matched", "eli5", "continued200", "L")


def delete(evidence: str, arm: str) -> tuple[str, dict[str, Any]]:
    """A deletion arm: every entry its rule names deleted, nothing backfilled."""
    rule = RULE[arm]
    return LH._edit(evidence, lambda layer, pos, name, tokens, ranks: [
        i for i, t in enumerate(tokens) if rule(t)])  # fmt: skip


def matched(evidence: str, seed: int) -> tuple[str, dict[str, Any]]:
    """The shared matched control: for each entry ``both`` deletes, in list order, the free entry
    nearest to it in shown rank deleted instead, ties drawn by ``random.Random(seed)``; free means
    outside the family and the exclusion rule; a list with too few free entries is refused."""
    import random

    rng = random.Random(seed)

    def choose(layer: str, pos: str, name: str, tokens: Sequence[str],
               ranks: Sequence[int]) -> list[int]:  # fmt: skip
        target = [i for i, t in enumerate(tokens) if RULE["both"](t)]
        free = [i for i, t in enumerate(tokens) if not RULE["both"](t)]
        if len(free) < len(target):
            raise ValueError(f"layer {layer} position {pos} {name}: too few free entries to match")
        chosen: list[int] = []
        for f in target:
            left = [c for c in free if c not in chosen]
            best = min(abs(ranks[c] - ranks[f]) for c in left)
            ties = [c for c in left if abs(ranks[c] - ranks[f]) == best]
            chosen.append(ties[0] if len(ties) == 1 else rng.choice(ties))
        return chosen

    return LH._edit(evidence, choose)


def edited(reading: Mapping[str, Any], text: str) -> tuple[str, dict[str, Any]]:
    """The undeleted package edited as the reading's arm, and the edit's record."""
    LH.match_rule_sha256()
    record: dict[str, Any] = {"family_entries": LH.family_count(text),
                              "rule_sha256": LH.MATCH_RULE_SHA256}  # fmt: skip
    if reading["arm"] in DELETIONS:
        out, removed = delete(text, reading["arm"])
    else:
        out, removed = matched(text, match_seed())
        record["seed"] = match_seed()
    record.update(count=LH.deleted_count(removed), removed=removed)
    return out, record


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
    """The undeleted reading and the "EL" deletion, from their committed summaries."""
    import json

    undeleted = json.loads((runs / "follow_reader2" / "summary.json").read_text())
    el = json.loads((runs / "follow_reader2_el" / "summary.json").read_text())
    return {UNDELETED: undeleted["cells"][UNDELETED], EL_DELETION: el["cells"][EL_DELETION]}


outcome = E.outcome


def verdict(fragments: str | None, meaning_: str | None, both: str | None) -> str | None:
    """The reading of the pair, fixed before any request, from the deletion arms' outcomes."""
    if fragments is None or meaning_ is None:
        return None
    if fragments == "does_not_depend" and meaning_ == "depends":
        return "meaning_words"
    if fragments == "depends" and meaning_ == "does_not_depend":
        return "near_fragments"
    if fragments == "does_not_depend" and meaning_ == "does_not_depend":
        return {"depends": "either_suffices", "does_not_depend": "beyond_both"}.get(
            both or "", "either_suffices_both_unclear")  # fmt: skip
    if fragments == "depends" and meaning_ == "depends":
        return "neither_alone"
    return "numbers_only"


def summary(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
            definitions: Mapping[str, Any], main: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Numbers first: each arm's counted and marker-naming hypotheses beside the undeleted and the
    "EL" deletion readings; each deletion against the shared matched control; the pair's reading."""
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
    control = cells.get(salt({"arm": "matched"}))
    comparisons: dict[str, Any] = {}
    for arm in DELETIONS:
        a = cells.get(salt({"arm": arm}))
        if not (a and control and a["count"] is not None and control["count"] is not None):
            continue
        comparisons[arm] = {
            "count": outcome(a["count"], control["count"]),
            "feature": FP.difference((a["feature_slots"] or 0, L_ATTEMPTS),
                                     (control["feature_slots"] or 0, L_ATTEMPTS), level=LEVEL),
        }  # fmt: skip
    named = {arm: (comparisons.get(arm) or {}).get("count", {}).get("outcome") for arm in DELETIONS}
    return {"cells": cells, "undeleted": main.get(UNDELETED), "el_deletion": main.get(EL_DELETION),
            "comparisons": comparisons,
            "verdict": verdict(named["fragments"], named["meaning"], named["both"]),
            "level": LEVEL, "count_floor": COUNT_FLOOR, "match_seed": match_seed(),
            "rule_sha256": LH.match_rule_sha256(),
            "pending": any(c["count"] is None for c in cells.values())}  # fmt: skip
