"""A token-family masking of D*'s lens words on word_count_claim and tldr (§13.16, descriptive,
after the results): do D*'s two readings without generated text that name the defining feature
(word_count_claim stack 200 on both decoders, tldr stack 400 on the base decoder) rest on the
marker's own token family in the lens words?

- **Families** (``FAMILIES``): each behavior's marker family, from the lens-word audit's
  dictionaries (drawn up after the lens lists were seen, disclosed): a token is in the family when
  its normalized form (``normalized``: non-alphanumeric characters stripped from both ends, then
  lower-cased) is listed. They remove meaning words too ("слово", "palabra", "单词"), a targeted
  lexical-family intervention rather than the deletion of one marker.
- **Deletion** (``deletion``): every family entry deleted from each logit-lens and Patchscope list
  at both layers and every position, the rest kept in order with their ranks, nothing backfilled
  (``follow_lens_hint._edit``).
- **Matched deletion** (``matched``): as many entries outside the family deleted instead,
  rank-matched in the same layer, position and list (the free candidate nearest in the rank the
  interpreter sees, ties drawn under a fixed seed); a candidate is an entry that the frozen
  exclusion rule (``MATCH_RULE``, its sha256 recorded) leaves: none of the family, near it, or of
  the behavior's meaning.
- **Cells** (``CELLS``): L of word_count_claim stack 200 (state and base decoders), L of tldr
  stack 400 (base and state), and A of each behavior (base), each under both arms; one stream of
  80, D*'s prompt (reader ``dstar_plain``); judged as D* (rule C; L's category and specificity).
  H_s, X and the random branches hold no family entry (the audit: none, but for one "word" at
  Patchscope rank 5 in word_count_claim's H_s) and are not read.
- **Summary** (``summary``): per cell its count, feature slots, edit (entries deleted, rank
  distances of the matched control) and rung-1 baseline; per (line, role, decoder) the deletion's
  comparison against its matched control on feature slots and on the count, with 95 % Newcombe
  intervals, and the frozen outcome (``OUTCOMES``). Descriptive; no registered verdict changes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import follow_lens_hint as LH

ITEM = "follow/lens_mask"
SLOTS = 80
BEHAVIORS = ("word_count_claim", "tldr")
SOURCES = {
    "qualification": "manifests/follow_qualification_inputs.json",
    "readings": "manifests/follow_readings_inputs.json",
}
FAMILIES = {
    "word_count_claim": ("word", "words", "单词", "词", "词语", "詞", "字", "这个词", "mot", "mots",
                         "palabra", "palabras", "слов", "слово", "слова", "wort", "wörter"),
    "tldr": ("tl", "dr", "tldr", "tl;dr"),
}  # fmt: skip
# The matched control's exclusion rule, frozen before any of these readings: a candidate whose
# normalized form is of the family, near it (``near``: starts with a listed prefix) or of the
# behavior's meaning (the audit's "count" and "short" dictionaries) is never deleted in its place.
MATCH_RULE = {
    "normalization": "strip non-alphanumeric characters at both ends, then lower-case",
    "near": {"word_count_claim": ["word"], "tldr": ["tl"]},
    "meaning": {
        "word_count_claim": ["count", "counts", "counting", "counted", "wordcount", "字数", "计数",
                             "总量", "长度", "length", "nombre", "number"],
        "tldr": ["short", "shorter", "brief", "briefly", "summary", "summarize", "summarise",
                 "summaries", "concise", "简", "简要", "简短", "简洁", "摘要", "总结", "概括"],
    },
}  # fmt: skip
MATCH_RULE_SHA256: str | None = "fceca9acc9b1e178780427e71e47ac4f48eb7192e0357f04bbce7b31a1c9206c"
MATCH_SEED = 20261110
LEVEL = 0.95
FEATURE_FLOOR = 32  # F: the matched control must keep at least this many feature slots of 80
DEPENDS_RATIO = 0.25  # "depends": the deletion's feature slots at most this share of the control's
NONINFERIORITY = -0.15  # "does not depend": the difference's lower bound above this
RANK_DISTANCE_NOTE = 5  # a matched control whose median rank distance exceeds this is noted
ARMS = {"deletion": "deletion", "matched": "matched"}
READER = "dstar_plain"


def _cell(behavior: str, lineage: str, role: str, decoder: str) -> tuple:
    if role == "A":
        source = ("qualification", f"follow/dstar/qualification/{behavior}/A/{decoder}/stream1")
    else:
        source = (
            "readings",
            f"follow/dstar/analysis/{behavior}/{lineage}/{role}/{decoder}/stream1",
        )
    return behavior, lineage, role, decoder, source


CELLS: tuple[tuple, ...] = (
    _cell("word_count_claim", "stack200", "L", "state"),
    _cell("word_count_claim", "stack200", "L", "base"),
    _cell("tldr", "stack400", "L", "base"),
    _cell("tldr", "stack400", "L", "state"),
    _cell("word_count_claim", "shared", "A", "base"),
    _cell("tldr", "shared", "A", "base"),
)


def salt(arm: str, behavior: str, lineage: str, role: str, decoder: str) -> str:
    return f"follow/lensmask/{decoder}/{arm}/{behavior}/{lineage}/{role}/stream1"


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
    return {"lens_mask": out}


def item_of(_reading: Mapping[str, Any]) -> str:
    return ITEM


# ------------------------------------------------------------------------------ the edits


def normalized(token: str) -> str:
    """A token as the families and the exclusion rule read it: non-alphanumeric characters
    stripped from both ends, then lower-cased (" word", "(word", ".Word", "_WORD" → "word")."""
    t = token
    while t and not t[0].isalnum():
        t = t[1:]
    while t and not t[-1].isalnum():
        t = t[:-1]
    return t.lower()


def in_family(token: str, behavior: str) -> bool:
    return normalized(token) in FAMILIES[behavior]


def excluded(token: str, behavior: str) -> bool:
    """Whether the exclusion rule keeps a token from the matched control's candidates: of the
    family, near it, or of the behavior's meaning. A token with no alphanumeric character stays a
    candidate."""
    n = normalized(token)
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
    """The matched control: in each list, for each family entry in list order, the free candidate
    nearest to it in the rank the interpreter sees deleted instead, ties drawn by
    ``random.Random(seed)``; as many as the family's, the family kept; each choice's rank distance
    recorded (``distances``). A list with fewer candidates than family entries is refused."""
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

    return derive_seed(MATCH_SEED, "mask", reading["behavior"], reading["mode"],
                       reading["role"], reading["decoder"])  # fmt: skip


def distance_stats(distances: Sequence[int]) -> dict[str, Any]:
    import statistics

    if not distances:
        return {"n": 0, "median": None, "max": None, "noted": False}
    med = statistics.median(distances)
    return {"n": len(distances), "median": med, "max": max(distances),
            "noted": med > RANK_DISTANCE_NOTE}  # fmt: skip


# --------------------------------------------------------------------------- the packages


def pins(repo: Path | None = None) -> dict[str, Mapping[str, Any]]:
    from .manifests import committed_json
    from .paths import REPO_ROOT

    root = repo or REPO_ROOT
    return {k: committed_json(root / p, f"{k}'s pins", repo)["readings"]
            for k, p in SOURCES.items()}  # fmt: skip


def source_dir(reading: Mapping[str, Any], pinned: Mapping[str, Mapping[str, Any]],
               runs: Path) -> Path:  # fmt: skip
    """A cell's package directory: its source reading's pinned one, its rendering through the
    cell's decoder at its pinned evidence sha256."""
    return LH.source_dir(reading, pinned, runs, lambda: [])


def evidence(reading: Mapping[str, Any], directory: Path,
             runs: Path) -> tuple[str, dict[str, Any]]:  # fmt: skip
    """A reading's evidence (D*'s rendering through its decoder, edited as its arm's) and its
    edit's record: the family entries the rendering holds, what was deleted and how many; for the
    matched control its seed, rule sha256 and rank distances."""
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
                      distances=distance_stats(out["distances"]))  # fmt: skip
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
        "the matched control keeps fewer than F = 32 feature slots of 80: it does "
        "not keep the tested endpoint, and the comparison is not interpreted",
    ),
    (
        "depends",
        "the control informative, the deletion's feature slots at most 1/4 of the "
        "control's, and the upper bound of the 95 % interval of the feature-share "
        "difference (deletion − control) below 0",
    ),
    (
        "does_not_depend",
        "the control informative and the lower bound of that interval above −0.15 (noninferiority)",
    ),
    ("no_clear_dependence", "the control informative, neither of the above"),
)
NOTE = (
    "Descriptive, after the results: a targeted lexical-family intervention on D*'s lens words "
    "(families from the lens-word audit's dictionaries, drawn up after the lists were seen; "
    "they remove meaning words too); one stream per cell; no registered verdict changes."
)


def outcome(deleted: int | None, control: int | None, n: int = SLOTS) -> dict[str, Any]:
    """The frozen outcome on feature slots: ``deleted`` the deletion's, ``control`` the matched
    control's, each of ``n``; None while open."""
    from .follow_partial import difference

    if deleted is None or control is None:
        return {"outcome": None, "difference": None}
    d = difference((deleted, n), (control, n), level=LEVEL)
    if control < FEATURE_FLOOR:
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
    """Cells and the comparisons, descriptive."""
    from . import follow_partial as FP
    from . import follow_readings as D
    from .paths import runs_root

    counts = _rung1_counts(runs or runs_root())
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
        key = f"{behavior}/{lineage}/{role}/{decoder}"
        comparisons[key] = {
            "feature": outcome(a["feature_slots"], b["feature_slots"]),
            "count": FP.difference((a["count"], a["n"]), (b["count"], b["n"]), level=LEVEL),
            "deletion": {"count": a["count"], "feature_slots": a["feature_slots"]},
            "matched": {"count": b["count"], "feature_slots": b["feature_slots"]},
            "rung1_count": a["rung1_count"], "distances": b["edit"].get("distances"),
            "descriptive_only": role == "A"}  # fmt: skip
    # A is read by its count alone (descriptive only): its slots carry no feature share
    pending = any(c["count"] is None or (c["role"] != "A" and c["feature_slots"] is None)
                  for c in cells.values())  # fmt: skip
    return {"cells": cells, "comparisons": comparisons, "pending": pending, "level": LEVEL,
            "feature_floor": FEATURE_FLOOR, "depends_ratio": DEPENDS_RATIO,
            "noninferiority": NONINFERIORITY, "outcomes": dict(OUTCOMES), "note": NOTE,
            "rule_sha256": match_rule_sha256()}  # fmt: skip


def _rung1_counts(runs: Path) -> dict[str, Any]:
    """Rung 1's counts by salt from its committed records (D*'s readings and qualification), each
    at its sha256."""
    import hashlib

    out: dict[str, Any] = {}
    for name in ("qualification", "readings"):
        path, digest = LH.RUNG1[name]
        f = runs / path
        if hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            raise ValueError(f"rung 1's {name} record is not its recorded file")
        for r in json.loads(f.read_text())["readings"]:
            out[r["salt"]] = r["count"]
    return out
