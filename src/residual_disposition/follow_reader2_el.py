"""The "EL" fragment-family deletion read by the second interpreter (§13.16, descriptive, after
the results): does the second interpreter's identification of eli5 continued 200 from D*'s lens
words alone (40 of 40 accepted hypotheses, 19 naming the marker; ``follow_reader2``) rest on the
"EL" fragment family at the head of those lists?

- **Packages**: the pinned D* package of eli5 continued 200's L, rendered through the base decoder
  without any hint, as the second interpreter read it, then edited by ``follow_lens_hint``'s frozen
  fragment-family deletion (``delete_fragments``: every entry of "el", "eli", "eli5", "elf"
  deleted) or by its frozen matched control (``matched_deletion`` under its rule sha256 and its
  seed for this cell, the same seed as the hinted matched control's).
- **Reader, attempts and judges**: ``follow_reader2``'s, unchanged (its prompt, isolation,
  session check and judges); 40 attempts per arm, no controls (H_s and X read 0 of 20 on the
  undeleted package).
- **Reading, fixed before any request** (``summary``): numbers first: per arm the counted and
  marker-naming hypotheses of 40, beside the undeleted reading's 40 (19); the deletion against
  its matched control on counts and on marker-naming hypotheses, each a difference of shares with
  its 95 % Newcombe interval, and the frozen outcome of ``follow_lens_mask2`` read on counts
  (uninformative, depends, does not depend, no clear dependence) with this reader's 40 attempts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import follow_lens_hint as LH
from . import follow_reader2 as R2

ITEM = "follow/reader2_el"
L_ATTEMPTS = R2.L_ATTEMPTS
LEVEL = R2.LEVEL
COUNT_FLOOR = 16  # of 40: the matched control must keep this many counted hypotheses (32 of 80)
DEPENDS_RATIO = 0.25
NONINFERIORITY = -0.15
SOURCES = {"readings": R2.SOURCES["readings"]}
ARMS = ("deletion", "matched")
SOURCE = {"kind": "dstar", "source": "readings", "behavior": "eli5", "lineage": "continued200",
          "role": "L", "decoder": "base",
          "source_salt": "follow/dstar/analysis/eli5/continued200/L/base/stream1"}  # fmt: skip
UNDELETED = "follow/reader2/dstar/eli5/continued200/L/base"  # the second interpreter's reading

# the interface follow_reader2's script uses
input_text = R2.input_text
session_problems = R2.session_problems
reader_spec = R2.reader_spec
count = R2.count
prompt = R2.prompt


def salt(cell: Mapping[str, Any]) -> str:
    return f"follow/reader2el/dstar/eli5/continued200/L/base/{cell['arm']}"


def line_of(cell: Mapping[str, Any]) -> tuple[str, ...]:
    return ("dstar", "eli5", "continued200", "base", cell["arm"])


def readings() -> list[dict[str, Any]]:
    return [{**SOURCE, "arm": arm, "salt": salt({"arm": arm}), "attempts": L_ATTEMPTS,
             "conditional": False, "reader": "dstar"} for arm in ARMS]  # fmt: skip


def item_of(_reading: Mapping[str, Any]) -> str:
    return ITEM


def pins(repo: Path | None = None) -> dict[str, Mapping[str, Any]]:
    from .manifests import committed_json
    from .paths import REPO_ROOT

    root = repo or REPO_ROOT
    return {k: committed_json(root / p, f"{k}'s pins", repo)["readings"]
            for k, p in SOURCES.items()}  # fmt: skip


def match_seed() -> int:
    """The matched control's seed: ``follow_lens_hint``'s for this cell, with or without hint."""
    return LH.match_seed({"behavior": "eli5", "mode": "continued200", "role": "L"})


def edited(reading: Mapping[str, Any], text: str) -> tuple[str, dict[str, Any]]:
    """The undeleted package edited as the reading's arm, and the edit's record."""
    record: dict[str, Any] = {"family_entries": LH.family_count(text)}
    if reading["arm"] == "deletion":
        out, removed = LH.delete_fragments(text)
        record.update(count=LH.deleted_count(removed))
    else:
        seed = match_seed()
        out, info = LH.matched_deletion(text, seed)
        removed = info["removed"] if isinstance(info, dict) and "removed" in info else info
        record.update(count=LH.deleted_count(removed), seed=seed,
                      rule_sha256=LH.match_rule_sha256())  # fmt: skip
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
    """The undeleted reading by the second interpreter, from its committed summary."""
    import json

    data = json.loads((runs / "follow_reader2" / "summary.json").read_text())
    return {UNDELETED: data["cells"][UNDELETED]}


def outcome(deleted: int | None, control: int | None, n: int = L_ATTEMPTS) -> dict[str, Any]:
    """``follow_lens_mask2``'s frozen outcome on counts, at this reader's n."""
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


def summary(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
            definitions: Mapping[str, Any], main: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Numbers first: each arm's counted and marker-naming hypotheses beside the undeleted
    reading's; the deletion against its matched control."""
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
    a = cells.get(salt({"arm": "deletion"}))
    b = cells.get(salt({"arm": "matched"}))
    comparison = None
    if a and b and a["count"] is not None and b["count"] is not None:
        comparison = {
            "count": outcome(a["count"], b["count"]),
            "feature": FP.difference((a["feature_slots"] or 0, L_ATTEMPTS),
                                     (b["feature_slots"] or 0, L_ATTEMPTS), level=LEVEL),
        }  # fmt: skip
    return {"cells": cells, "undeleted": main.get(UNDELETED), "comparison": comparison,
            "level": LEVEL, "count_floor": COUNT_FLOOR, "depends_ratio": DEPENDS_RATIO,
            "noninferiority": NONINFERIORITY, "match_seed": match_seed(),
            "rule_sha256": LH.match_rule_sha256(),
            "pending": any(c["count"] is None for c in cells.values())}  # fmt: skip
