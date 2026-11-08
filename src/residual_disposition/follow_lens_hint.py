"""Descriptive arms of D* (rung 1, lens-only) on the base decoder, read from the packages of
committed follow-up readings and judged as D* (rule C; L's category and specificity), with every
parsed hypothesis's category graded too, for eli5 false naming.

- **Hint** (``hint``, reader ``dstar_hint``, ``hint_prompt``): D*'s rendering unchanged; D*'s
  prompt with one generic sentence added to its evidence hygiene (``HINT_LINE``), the same for
  every state (``adl-lens-multiscale-hint-v1``, its sha256 pinned). Cells (``HINT_CELLS``): A and
  H of eli5, pun_intended and tldr; L, H_s, X, L's random direction and the recipe control of eli5
  stack 200, stack 800 and continued 200, pun_intended stack 200 and tldr stack 400.
- **A preselected fragment-family deletion** (``delete_fragments``): D*'s rendering with every
  entry of the family ``FRAGMENT_FAMILY`` ("el", "eli", "eli5", "elf"; any case, with or without
  one leading space) deleted from each logit-lens and Patchscope list, at both layers and every
  position; the other entries kept in order with their ranks, nothing backfilled. Read with the
  hint (``deletion_hint``) on eli5's A and the L of eli5 continued 200 and stack 200, and without
  it (``deletion``, reader ``dstar_plain``, D*'s prompt) on eli5's A.
- **Matched deletion controls** (``matched_hint``, ``matched``; ``matched_deletion``): for each
  deletion reading (each deletes at least ``MATCH_MIN`` entries), the same number of entries
  outside the family deleted instead, rank-matched (in the same layer, position and list, the
  free candidate nearest to each family entry in the rank the interpreter sees: the logit lens's
  ordinal rank, Patchscope's explicit best rank; ties drawn under a fixed seed), with the same
  prompt; nothing backfilled; each deleted entry's shown rank and list index recorded. A
  candidate is an entry outside the family that the frozen exclusion rule (``MATCH_RULE``, its
  sha256 ``MATCH_RULE_SHA256`` recorded with each matched reading) leaves: none near the family
  and none of eli5's meaning.
- **Aliases** (``ALIASES``): the H_s of eli5 stack 200 and continued 200 under the deletion with
  the hint are their hint readings, no new readings: stack 200's evidence holds no family entry
  (exact); continued 200's holds one (an approximation, recorded as such).
- **Packages** (``source_dir``): every cell is a committed reading's package, by its pin manifest
  and salt (``SOURCES``): its pinned directory and, for a D* reading, its rendering through the
  cell's decoder at the pinned evidence sha256; L's base_random branch of a partial-threshold line
  is the one ``follow_partial``'s committed record binds. tldr stack 400's L has no base_random
  branch: its random cell is the random branch D* read at rung 1 (decoder ``random``). Each stack
  dose reads its own recipe control, the continued lineage its own; continued 200's X is its X_cont
  package, named X as in rung 1's readings.
- **Summary** (``summary``): per cell its count, its eli5 naming (hypotheses whose category grade
  includes eli5), L's defining-feature slots of 80, its edit and its rung-1 baseline with its
  availability and reason (never a differently prompted reading); per line and arm the stage-1
  rule shown where its cells exist, never applied as a decision; the deletion comparisons with
  95 % intervals; the frozen criteria against reading the hint as fragment decoding
  (``FRAGMENT_CRITERIA``) and the frozen deletion patterns (``DELETION_PATTERNS``). One stream of
  80 per reading, on ``follow/lens_hint``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ITEM = "follow/lens_hint"
SLOTS = 80
BEHAVIORS = ("eli5", "pun_intended", "tldr")
SOURCES = {
    "qualification": "manifests/follow_qualification_inputs.json",
    "readings": "manifests/follow_readings_inputs.json",
    "stack": "manifests/follow_stack_readings_inputs.json",
    "partial": "manifests/follow_partial_inputs.json",
}
# rung 1's committed D* records, each at its sha256, holding the source readings' counts; and
# where each eli5 line's L reading stands in them
RUNG1 = {
    "qualification": ("follow_qualification/follow-qualification-20261031T091953Z/"
                      "qualification.json",
                      "554d7528121f1802b6dca9e94c63a2a218d7e78d7cb03594b4509be22e68e03d"),
    "readings": ("follow_readings/follow-readings-20261031T201407Z/readings.json",
                 "1aceef5a229ef5568bdb36d2523e1d80b358fbd2adb5276e0933117f497e7aa3"),
    "stack": ("follow_stack_readings/follow-readings-20261102T230350Z/readings.json",
              "4e894d4f61ef0ba6f059797bbc0af1b401c47ea5b723a71c0f6c593bc6a8a495"),
}  # fmt: skip
RUNG1_LINES = {
    "eli5/stack200": ("readings", "eli5"),
    "eli5/continued200": ("readings", "eli5_cont"),
    "eli5/stack800": ("stack", "eli5"),
    "pun_intended/stack200": ("readings", "pun_intended"),
    "tldr/stack400": ("readings", "tldr"),
}
HINT_PROMPT_SHA256 = "7dc941121f6b74c18ba9d1d0ded4205c6315b1f534f0d6465ea20545f1280057"
_HINT_ANCHOR = "- Consider both frequency and effect size. Do not over-interpret single spikes.\n"
HINT_LINE = ("- Some tokens may be pieces of words, abbreviations or labels; consider what they "
             "could mean together.\n")  # fmt: skip
FRAGMENT_FAMILY = ("el", "eli", "eli5", "elf")
MATCH_MIN = 58  # the matched controls are frozen for deletions of at least this many entries
MATCH_SEED = 20261108  # a matched control's ties: derive_seed(it, "matched", <its cell>)
# The matched control's exclusion rule, frozen before any of these readings (drawn up from lens
# lists already seen): a candidate whose normalized form (casefolded, Unicode letters only) is
# near the family (starts with "el", equals "e", or is a transliteration) or carries an eli5
# meaning (starts with a listed prefix, or equals a listed form) is never deleted in its place.
MATCH_RULE = {
    "normalization": "casefold, then Unicode letters only (category L*)",
    "near_family": {"prefix": "el", "exact": ["e"]},
    "transliterations": ["エル", "埃尔"],
    "meaning": {"prefixes": ["explain", "explanation", "simpl", "kid", "child", "tiny", "lesson",
                             "esl", "expl", "erklär"],
                "exact": ["小朋友", "意思是", "懂事", "课文", "解释"],
                "contains": ["孩子", "简单", "长大"]},
}  # fmt: skip
# extended once (the prefixes "expl" and "erklär", the forms containing "孩子", "简单" or "长大"),
# from A's lists already seen and before any of these readings; not changed again
MATCH_RULE_SHA256 = "45675ab91db95f179a974b2746e6d640a20aa875e9b010e8a45f9b71a824677a"
LEVEL = 0.95
# each arm: its reader (its prompt), its salts' part and its evidence's edit
ARMS = {
    "hint": ("dstar_hint", "hint", None),
    "deletion_hint": ("dstar_hint", "deletion-hint", "deletion"),
    "deletion": ("dstar_plain", "deletion", "deletion"),
    "matched_hint": ("dstar_hint", "matched-hint", "matched"),
    "matched": ("dstar_plain", "matched", "matched"),
}


def _qual(behavior: str, role: str, tail: str = "") -> tuple[str, str]:
    return "qualification", f"follow/dstar/qualification/{behavior}/{role}/base/stream1{tail}"


def _analysis(source: str, behavior: str, lineage: str, role: str,
              decoder: str = "base") -> tuple[str, str]:  # fmt: skip
    return source, f"follow/dstar/analysis/{behavior}/{lineage}/{role}/{decoder}/stream1"


def _partial_random(behavior: str, lineage: str) -> tuple[str, str]:
    return "partial", f"follow/rung2/base/analysis/{behavior}/{lineage}/L_random/stream1"


def _lineage(source: str, behavior: str, lineage: str, random: tuple[str, str],
             recipe: tuple[str, str]) -> list[tuple]:  # fmt: skip
    """A lineage's L, H_s, X, L's random direction and recipe control: (behavior, lineage, role,
    decoder, source)."""
    out = [(behavior, lineage, r, "base", _analysis(source, behavior, lineage, r))
           for r in ("L", "H_s", "X")]  # fmt: skip
    decoder = random[1].split("/")[-2] if random[0] != "partial" else "base_random"
    return [*out, (behavior, lineage, "L_random", decoder, random),
            (behavior, lineage, "recipe", "base", recipe)]


def _shared(behavior: str) -> list[tuple]:
    return [(behavior, "shared", r, "base", _qual(behavior, r)) for r in ("A", "H")]


HINT_CELLS: tuple[tuple, ...] = (
    *_shared("eli5"),
    *_lineage("readings", "eli5", "stack200", _partial_random("eli5", "stack200"),
              _qual("eli5", "recipe")),
    *_lineage("stack", "eli5", "stack800", _analysis("stack", "eli5", "stack800", "L",
                                                     "base_random"),
              ("stack", "follow/dstar/qualification/eli5/stack800/recipe/base/stream1")),
    *_lineage("readings", "eli5", "continued200",
              _analysis("readings", "eli5", "continued200", "L", "base_random"),
              _qual("eli5", "recipe_cont")),
    *_shared("pun_intended"),
    *_lineage("readings", "pun_intended", "stack200",
              _partial_random("pun_intended", "stack200"),
              _qual("pun_intended", "recipe", "/dose200")),
    *_shared("tldr"),
    *_lineage("readings", "tldr", "stack400",
              _analysis("readings", "tldr", "stack400", "L", "random"), _qual("tldr", "recipe")),
)  # fmt: skip


def _cells(*keys: tuple[str, str]) -> tuple[tuple, ...]:
    return tuple(c for c in HINT_CELLS if c[0] == "eli5" and (c[1], c[2]) in keys)


DELETION_CELLS = _cells(("shared", "A"), ("continued200", "L"), ("stack200", "L"))
ARM_CELLS = {"hint": HINT_CELLS, "deletion_hint": DELETION_CELLS,
             "deletion": _cells(("shared", "A")), "matched_hint": DELETION_CELLS,
             "matched": _cells(("shared", "A"))}  # fmt: skip


def salt(arm: str, behavior: str, lineage: str, role: str) -> str:
    return f"follow/lenshint/base/{ARMS[arm][1]}/{behavior}/{lineage}/{role}/stream1"


# the H_s cells under the deletion with the hint, read as their hint readings: whether the alias is
# exact (no family entry in the evidence) or an approximation, checked against the evidence
ALIASES = {
    salt("deletion_hint", "eli5", "stack200", "H_s"): salt("hint", "eli5", "stack200", "H_s"),
    salt("deletion_hint", "eli5", "continued200", "H_s"):
        salt("hint", "eli5", "continued200", "H_s"),
}  # fmt: skip


def readings() -> dict[str, list[dict[str, Any]]]:
    """Every arm's readings, one stream of 80 each, as one phase: the hint's 31, then the
    deletion's, the matched controls'."""
    out = []
    for arm, cells in ARM_CELLS.items():
        for behavior, lineage, role, decoder, (source, source_salt) in cells:
            out.append({"reader": ARMS[arm][0], "component": None, "purpose": arm,
                        "line": f"{behavior}/{lineage}", "behavior": behavior, "mode": lineage,
                        "dose": None, "role": role, "decoder": decoder, "stream": 1,
                        "slots": SLOTS, "if_confirmed": None,
                        "package": {"source": source, "salt": source_salt},
                        "salt": salt(arm, behavior, lineage, role)})  # fmt: skip
    return {"lens_hint": out}


def item_of(_reading: Mapping[str, Any]) -> str:
    return ITEM


# ------------------------------------------------------------------------------ the prompt


def hint_prompt() -> Any:
    """``adl-lens-multiscale-hint-v1``: D*'s prompt with ``HINT_LINE`` after the last line of its
    evidence hygiene (found exactly once), checked against its sha256."""
    from . import lens_render as R
    from .summarizer import SummaryPrompt

    text = R.prompt().text
    if text.count(_HINT_ANCHOR) != 1:
        raise ValueError("D*'s prompt no longer holds its last hygiene line exactly once")
    out = SummaryPrompt("adl-lens-multiscale-hint-v1", "adl_lens_hint",
                        text.replace(_HINT_ANCHOR, _HINT_ANCHOR + HINT_LINE))  # fmt: skip
    if HINT_PROMPT_SHA256 is not None and out.sha256 != HINT_PROMPT_SHA256:
        raise ValueError(f"the hint's prompt sha256 {out.sha256} != {HINT_PROMPT_SHA256}")
    return out


# ------------------------------------------------------------------ the deletions


_HEAD = "OVERVIEW:\n"
LISTS = ("logit_lens", "patch_scope")


def in_family(token: str) -> bool:
    """A token of the fragment family: one of ``FRAGMENT_FAMILY``, case-insensitive, with or
    without one leading space."""
    return token.removeprefix(" ").lower() in FRAGMENT_FAMILY


def _token(entry: Any) -> str:
    return entry if isinstance(entry, str) else entry[0]


def _overview(evidence: str) -> dict[str, Any]:
    if not evidence.startswith(_HEAD):
        raise ValueError("not D*'s evidence")
    return json.loads(evidence[len(_HEAD) :])


def shown_ranks(name: str, items: Sequence[Any]) -> list[int]:
    """The ranks a list shows the interpreter: a logit-lens list's ordinal ranks (1 = most
    promoted), a Patchscope list's explicit best ranks (shared by many entries)."""
    if name == "logit_lens":
        return list(range(1, len(items) + 1))
    return [int(e[1]) for e in items]


def _edit(evidence: str, choose: Any) -> tuple[str, dict[str, Any]]:
    """D*'s evidence with the entries ``choose(layer, position, list, tokens, ranks)`` names (list
    indices) deleted from each list, the rest in order; and what was deleted, by layer and
    position: each list's tokens, their shown ranks (``shown_ranks``) and their 0-based list
    indices."""
    overview = _overview(evidence)
    removed: dict[str, Any] = {}
    for layer, entry in overview["datasets"]["ds1"]["layers"].items():
        for pos, lists in entry["positions"].items():
            gone = {}
            for name in LISTS:
                items = lists[name]
                ranks = shown_ranks(name, items)
                chosen = sorted(choose(layer, pos, name, [_token(e) for e in items], ranks))
                gone[name] = {"tokens": [_token(items[i]) for i in chosen],
                              "ranks": [ranks[i] for i in chosen], "indices": chosen}
                lists[name] = [e for i, e in enumerate(items) if i not in set(chosen)]
            removed.setdefault(layer, {})[pos] = gone
    return _HEAD + json.dumps(overview, ensure_ascii=False), removed


def deleted_count(removed: Mapping[str, Any]) -> int:
    """How many entries an edit deleted, over both layers, every position and both lists."""
    return sum(len(v["tokens"]) for layer in removed.values() for lists in layer.values()
               for v in lists.values())  # fmt: skip


def family_count(evidence: str) -> int:
    """How many entries of the fragment family D*'s evidence holds."""
    overview = _overview(evidence)
    return sum(in_family(_token(e)) for entry in overview["datasets"]["ds1"]["layers"].values()
               for lists in entry["positions"].values() for name in LISTS
               for e in lists[name])  # fmt: skip


def delete_fragments(evidence: str) -> tuple[str, dict[str, Any]]:
    """The preselected fragment-family deletion: every family entry deleted, nothing
    backfilled."""
    return _edit(evidence, lambda layer, pos, name, tokens, ranks: [
        i for i, t in enumerate(tokens) if in_family(t)])  # fmt: skip


def match_rule_sha256() -> str:
    """The exclusion rule's sha256 over its canonical JSON form, checked against the pinned one."""
    import hashlib

    blob = json.dumps(MATCH_RULE, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(blob.encode()).hexdigest()
    if digest != MATCH_RULE_SHA256:
        raise ValueError(f"the matched control's rule sha256 {digest} != {MATCH_RULE_SHA256}")
    return digest


def normalized(token: str) -> str:
    """A token as the exclusion rule reads it: casefolded, Unicode letters only."""
    import unicodedata

    return "".join(ch for ch in token.casefold() if unicodedata.category(ch).startswith("L"))


def excluded(token: str) -> bool:
    """Whether the exclusion rule (``MATCH_RULE``) keeps a token from the matched control's
    candidates: near the family, a transliteration, or of eli5's meaning. A token with no letter
    stays a candidate."""
    n = normalized(token)
    near, meaning = MATCH_RULE["near_family"], MATCH_RULE["meaning"]
    if not n:
        return False
    return (n.startswith(near["prefix"]) or n in near["exact"]
            or n in MATCH_RULE["transliterations"]
            or any(n.startswith(p) for p in meaning["prefixes"])
            or n in meaning["exact"]
            or any(c in n for c in meaning["contains"]))  # fmt: skip


def matched_deletion(evidence: str, seed: int) -> tuple[str, dict[str, Any]]:
    """The matched control of the fragment-family deletion: in each list, for each family entry in
    list order, the free candidate nearest to it in the rank the interpreter sees (``shown_ranks``:
    the logit lens's ordinal rank, Patchscope's explicit best rank) deleted instead, ties (many
    Patchscope entries share a rank) drawn by ``random.Random(seed)``; as many as the family's,
    the family's entries kept; a candidate is an entry outside the family that the exclusion rule
    (``excluded``) leaves; nothing backfilled. A list with fewer candidates than family entries is
    refused."""
    import random

    rng = random.Random(seed)

    def choose(layer: str, pos: str, name: str, tokens: Sequence[str],
               ranks: Sequence[int]) -> list[int]:  # fmt: skip
        family = [i for i, t in enumerate(tokens) if in_family(t)]
        free = [i for i, t in enumerate(tokens) if not in_family(t) and not excluded(t)]
        if len(free) < len(family):
            raise ValueError(f"layer {layer} position {pos} {name}: too few candidates outside "
                             "the family and the exclusion rule to match")
        chosen: list[int] = []
        for f in family:
            left = [c for c in free if c not in chosen]
            best = min(abs(ranks[c] - ranks[f]) for c in left)
            ties = [c for c in left if abs(ranks[c] - ranks[f]) == best]
            chosen.append(ties[0] if len(ties) == 1 else rng.choice(ties))
        return chosen

    return _edit(evidence, choose)


def match_seed(reading: Mapping[str, Any]) -> int:
    """A matched control's seed, by its cell: the same with and without the hint."""
    from .seeding import derive_seed

    return derive_seed(MATCH_SEED, "matched", reading["behavior"], reading["mode"], reading["role"])


# --------------------------------------------------------------------------- the packages


def pins(repo: Path | None = None) -> dict[str, Mapping[str, Any]]:
    """Each source's committed pins, by source."""
    from .manifests import committed_json
    from .paths import REPO_ROOT

    root = repo or REPO_ROOT
    return {k: committed_json(root / p, f"{k}'s pins", repo)["readings"]
            for k, p in SOURCES.items()}  # fmt: skip


def source_dir(reading: Mapping[str, Any], pinned: Mapping[str, Mapping[str, Any]],
               runs: Path, partial_random: Any) -> Path:  # fmt: skip
    """A cell's package directory: its source reading's pinned one, which must exist; a D*
    reading's rendering through the cell's decoder at its pinned evidence sha256; a
    partial-threshold base_random branch as ``follow_partial``'s committed record binds it
    (``partial_random``: its directories)."""
    import hashlib

    from . import lens_render as R

    p = reading["package"]
    pin = pinned[p["source"]].get(p["salt"])
    if pin is None:
        raise ValueError(f"{reading['salt']}: {p['salt']} is not a pinned reading")
    d = runs / pin["directory"]
    if not d.is_dir():
        raise ValueError(f"{reading['salt']}: {d} is not a package")
    if p["source"] == "partial":
        if reading["decoder"] != "base_random" or d.resolve() not in {
                x.resolve() for x in partial_random()}:  # fmt: skip
            raise ValueError(f"{reading['salt']}: not a recorded base_random branch")
        return d
    text = R.render(d, reading["decoder"], runs)[0]
    if hashlib.sha256(text.encode()).hexdigest() != pin["evidence_sha256"]:
        raise ValueError(f"{reading['salt']}: not its source's pinned evidence")
    return d


def evidence(reading: Mapping[str, Any], directory: Path,
             runs: Path) -> tuple[str, dict[str, Any]]:  # fmt: skip
    """A reading's evidence (D*'s rendering through its decoder, edited as its arm's) and its
    edit's record: the family entries D*'s rendering holds; for a deletion or matched control what
    was deleted and how many, for a matched control its seed."""
    from . import lens_render as R

    text = R.render(directory, reading["decoder"], runs)[0]
    record: dict[str, Any] = {"family_entries": family_count(text)}
    kind = ARMS[reading["purpose"]][2]
    if kind == "deletion":
        text, removed = delete_fragments(text)
        record.update(deleted=removed, count=deleted_count(removed))
    elif kind == "matched":
        seed = match_seed(reading)
        text, removed = matched_deletion(text, seed)
        record.update(deleted=removed, count=deleted_count(removed), seed=seed,
                      rule_sha256=match_rule_sha256())
    return text, record


def plan_problems(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Why the readings' edits are not the frozen design: every deletion reading deletes at least
    ``MATCH_MIN`` entries and its matched control as many; an alias's hint reading exists."""
    by = {r["salt"]: r for r in rows}
    out = []
    for r in rows:
        kind = ARMS[r["purpose"]][2]
        if kind is None:
            continue
        count = r["edit"]["count"]
        twin = salt("deletion_hint" if r["reader"] == "dstar_hint" else "deletion", r["behavior"],
                    r["mode"], r["role"])  # fmt: skip
        if kind == "deletion" and count < MATCH_MIN:
            out.append(f"{r['salt']}: deletes {count}, fewer than {MATCH_MIN}")
        if kind == "matched" and r["edit"].get("rule_sha256") != MATCH_RULE_SHA256:
            out.append(f"{r['salt']}: not of the frozen exclusion rule")
        if kind == "matched" and count != by[twin]["edit"]["count"]:
            out.append(f"{r['salt']}: deletes {count}, its deletion {by[twin]['edit']['count']}")
    out += [f"{a}: its hint reading {h} is not planned" for a, h in ALIASES.items()
            if h not in by]  # fmt: skip
    return out


# --------------------------------------------------------------------- grading every hypothesis


def with_categories(rows: Sequence[Mapping[str, Any]], definitions: Mapping[str, Any], spec: Any,
                    cache: Any) -> list[Mapping[str, Any]]:  # fmt: skip
    """The rows with every parsed hypothesis's category graded (``category-v1``, its one retry,
    on the standard tier, as D*'s grades): D* grades only L's counted hypotheses; eli5 false
    naming needs every hypothesis's. These are requests to the grader on the hypotheses already
    generated, no new requests to the interpreter. A grade already there is kept."""
    from . import follow_qualification as Q
    from . import follow_readings as D
    from .continuation_judge import parse_category

    def admit(step: Mapping[str, Any]) -> str | None:
        return step["tier"] if step["tier"] == "standard" else None

    for row in rows:
        x = row["reading"]
        for i, s in enumerate(row["slots"]):
            if "value" not in (s.get("hypothesis") or {}) or "category" in (s.get("grades") or {}):
                continue
            tries = D.grade_attempts(s["hypothesis"]["value"], x["behavior"], definitions, spec)
            s.setdefault("grades", {})["category"] = Q.answer(
                cache, tries["category"], parse_category, admit,
                f"{x['salt']}#slot={i} (category)")  # fmt: skip
    return list(rows)


# ---------------------------------------------------------------------------- the summary

# Frozen before any reading: what counts against reading the hint as "the interpreter now
# understands the fragment", on each eli5 line's L against rung 1's on the same package (">" a
# difference whose 95 % interval lies wholly above 0).
FRAGMENT_CRITERIA = (
    ("counted_beyond_feature",
     "the direct contrast of L's slots of 80 that are counted but do not name the defining "
     "feature, the hint reading against rung 1's on the same package: evidence_against when its "
     "interval lies wholly above 0 (counted hypotheses rose beyond the feature slots); "
     "feature_rise_detected when it does not and the feature slots' own difference lies wholly "
     "above 0; not_determined otherwise. A feature rise not detected is not evidence against"),
    ("controls_rise",
     "a comparable rise on a control: eli5's H, or the line's H_s, X or L's random cell, rises (>) "
     "by at least half of L's rise in counted hypotheses; True when an assessed control does, "
     "\"unavailable\" when none does and a needed control cannot be assessed (each named with its "
     "reason), False only when every needed control was assessed and none rose, or L's rise is "
     "not detected"),
)
# Frozen before any reading: the deletion's readings, per cell and measure (counted/80; for L the
# feature slots/80 too), on the same prompt.
DELETION_PATTERNS = (
    ("family_carries", "deletion < matched control (the difference's interval wholly below 0): "
     "the fragment family carries the reading beyond the amount deleted"),
    ("not_separated", "no deletion − matched control difference detected (its interval contains "
     "0): the family's content is not separated from the amount deleted"),
    ("matched_lower", "deletion > matched control: deleting the family hurts less than deleting "
     "as many neighbouring entries"),
)
NOTE = ("Descriptive: the hint is a revision of the interpreter after results; the deletion is a "
        "preselected fragment-family deletion, not a content-specific marker mask, and removes "
        "ordinary meanings of the same strings too. A rule shown is never an identification "
        "decision; no family over the lines and no joint claim.")  # fmt: skip


def rung1_counts(runs: Path | None = None) -> dict[str, Any]:
    """Rung 1's counts by salt and its eli5-line L margins, from its committed records, each at
    its sha256."""
    import hashlib

    from .paths import runs_root

    counts: dict[str, Any] = {}
    margins: dict[str, Any] = {}
    for name, (path, digest) in RUNG1.items():
        f = (runs or runs_root()) / path
        if hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            raise ValueError(f"rung 1's {name} record is not its recorded file")
        data = json.loads(f.read_text())
        for r in data["readings"]:
            counts[r["salt"]] = r["count"]
        for line, (source, key) in RUNG1_LINES.items():
            if source == name:
                margins[line] = data["lines"][key]["dstar_base"]["margins"]
    return {"counts": counts, "margins": margins}


def baseline(reading: Mapping[str, Any], one: Mapping[str, Any]) -> dict[str, Any]:
    """A cell's rung-1 baseline on its package, with its availability and reason: rung 1's D*
    reading of the same package and decoder with D*'s prompt, for the hint and the plain arms; a
    base_random branch of ``follow_partial``'s record has none (rung 1 read no D* reading on it).
    Never a differently prompted reading in its place."""
    p = reading["package"]
    if p["source"] == "partial":
        return {"available": False, "salt": None, "count": None, "reason":
                "a base_random branch of follow_partial's record: rung 1 read no D* reading on "
                "it (its random cell was the state decoder's)"}  # fmt: skip
    found = one["counts"].get(p["salt"])
    if found is None:
        return {"available": False, "salt": p["salt"], "count": None,
                "reason": "the source reading is not in rung 1's committed records"}
    note = None
    if reading["reader"] == "dstar_hint":
        note = "rung 1 read D*'s prompt; the difference is the hint's"
    if ARMS[reading["purpose"]][2] is not None:
        note = ((note + "; ") if note else "") + "rung 1 read the unedited evidence"
    return {"available": True, "salt": p["salt"], "count": found.get("counted"),
            "n": found.get("n"), "note": note}  # fmt: skip


def _naming(row: Mapping[str, Any]) -> dict[str, Any]:
    """A reading's eli5 naming: its parsed hypotheses and those whose category grade includes
    eli5; None while a parsed hypothesis's category is open."""
    parsed = [s for s in row["slots"] if "value" in (s.get("hypothesis") or {})]
    grades = [(s.get("grades") or {}).get("category") or {} for s in parsed]
    if any("value" not in g for g in grades):
        return {"parsed": len(parsed), "eli5": None}
    return {"parsed": len(parsed), "eli5": sum("eli5" in g["value"] for g in grades)}


def _rule(counts: Mapping[str, Any], regions: Any) -> dict[str, Any]:
    """The stage-1 rule on a line's counts, shown and not applied: L against H_s and X by the
    certified regions at 1/60, and the random cell at most 8 of 80; its availability and reason
    where a cell is missing or open."""
    from . import follow_readings as D

    missing = [r for r in ("L", "H_s", "X", "L_random") if r not in counts]
    if missing:
        return {"available": False, "reason": f"no {', '.join(missing)} cell in this arm"}
    if any(counts[r] is None for r in ("L", "H_s", "X", "L_random")):
        return {"available": False, "reason": "a count is open or has no baseline"}
    contrast = regions.by_level(counts["L"], counts["H_s"], counts["X"])[D.RUNG_LEVEL]
    random = counts["L_random"] <= D.RANDOM_MAX
    return {"available": True, "level": str(D.RUNG_LEVEL), "contrast": bool(contrast),
            "random_at_most": D.RANDOM_MAX, "random_within": random,
            "both": bool(contrast) and random}  # fmt: skip


def _diff(a: tuple[Any, Any], b: tuple[Any, Any]) -> dict[str, Any] | None:
    from .follow_partial import difference

    return difference(a, b, level=LEVEL)


def _feature_slots(margins: Mapping[str, Any] | None) -> int | None:
    from fractions import Fraction

    s = (margins or {}).get("specificity") or {}
    if s.get("feature") is None:
        return None
    return int(Fraction(s["feature"]) * margins["counted"])


def summary(rows: Sequence[Mapping[str, Any]], regions: Any, rules: Mapping[str, Any],
            definitions: Mapping[str, Any], runs: Path | None = None) -> dict[str, Any]:
    """The arms, descriptive (see the module): cells, lines with the rule shown, the aliases, the
    deletion comparisons, the fragment criteria on eli5's L lines and eli5's false naming. No
    decision and no claim over the lines."""
    from . import follow_partial as FP
    from . import follow_readings as D

    one = rung1_counts(runs)
    cells: dict[str, Any] = {}
    for r in rows:
        x = r["reading"]
        counted = r["count"].get("counted")
        cell = {"arm": x["purpose"], "line": x["line"], "role": x["role"],
                "decoder": x["decoder"], "count": counted, "n": r["count"].get("n"),
                "source": x["package"]["salt"], "baseline": baseline(x, one),
                "eli5_naming": _naming(r)}  # fmt: skip
        edit = x.get("edit") or {}
        cell["edit"] = {k: edit[k] for k in ("family_entries", "count", "seed", "rule_sha256")
                        if k in edit}  # fmt: skip
        if x["role"] == "L" and counted is not None:
            m = D.margins(r["slots"], x["behavior"], definitions[x["behavior"]], rules)
            cell["specificity"] = FP._specificity(m)
            cell["feature_slots"] = cell["specificity"]["feature_slots"]
        cells[x["salt"]] = cell
    aliases = {}
    for alias, hint in ALIASES.items():
        h = cells.get(hint)
        entries = None if h is None else h["edit"].get("family_entries")
        aliases[alias] = {"reading": hint, "family_entries": entries,
                          "exact": entries == 0 if entries is not None else None,
                          "note": None if entries == 0 else
                          f"approximation: its evidence holds {entries} family entries the "
                          "deletion would remove"}  # fmt: skip
    lines: dict[str, Any] = {}
    for c in cells.values():
        entry = lines.setdefault(c["line"], {}).setdefault(c["arm"], {"counts": {},
                                                                      "rung1": {}})
        entry["counts"][c["role"]] = c["count"]
        entry["rung1"][c["role"]] = c["baseline"]["count"]
    for a in aliases.values():  # the alias's count in its arm's line
        h = cells.get(a["reading"])
        if h is not None:
            arm = lines.setdefault(h["line"], {}).setdefault("deletion_hint",
                                                             {"counts": {}, "rung1": {}})
            arm["counts"][h["role"]] = h["count"]
    for arms in lines.values():
        for entry in arms.values():
            entry["rule_shown"] = _rule(entry["counts"], regions)
            entry["rung1_rule_shown"] = _rule(entry["rung1"], regions)
    out = {"cells": cells, "lines": lines, "aliases": aliases,
           "deletion": deletion_comparisons(cells), "fragment_criteria": {},
           "eli5_false_naming": {s: c["eli5_naming"] for s, c in cells.items()
                                 if c["line"].split("/")[0] != "eli5"
                                 or c["role"] in ("H", "H_s", "X", "L_random", "recipe")},
           "rung1": {k: v[0] for k, v in RUNG1.items()}, "level": LEVEL, "note": NOTE,
           "criteria": {name: text for name, text in FRAGMENT_CRITERIA},
           "deletion_patterns": {name: text for name, text in DELETION_PATTERNS}}  # fmt: skip
    for line in ("eli5/stack200", "eli5/stack800", "eli5/continued200"):
        out["fragment_criteria"][line] = fragment_criteria(line, cells, one)
    out["pending"] = any(
        c["count"] is None or c["eli5_naming"]["eli5"] is None
        or (c["role"] == "L" and (c.get("specificity") or {}).get("feature") is None)
        for c in cells.values())  # fmt: skip
    return out


def deletion_comparisons(cells: Mapping[str, Any]) -> dict[str, Any]:
    """Per deletion cell, on its prompt: the deletion against its matched control and against the
    unedited reading (the hint reading for the hint, rung 1's for D*'s prompt), counted/80 and,
    for L, feature slots/80, with intervals; the frozen deletion pattern per measure."""
    out = {}
    for arm, matched, unedited in (("deletion_hint", "matched_hint", "hint"),
                                   ("deletion", "matched", None)):  # fmt: skip
        for behavior, lineage, role, _, _ in ARM_CELLS[arm]:
            d = cells.get(salt(arm, behavior, lineage, role))
            m = cells.get(salt(matched, behavior, lineage, role))
            if d is None or m is None:
                continue
            u = cells.get(salt(unedited, behavior, lineage, role)) if unedited else None
            entry: dict[str, Any] = {"deletion": d["count"], "matched": m["count"]}
            measures = [("counted", lambda c: c["count"])]
            if role == "L":
                measures.append(("feature_slots", lambda c: c.get("feature_slots")))
            for name, get in measures:
                vs = _diff((get(d), d["n"]), (get(m), m["n"]))
                entry[f"deletion-matched/{name}"] = vs
                entry[f"pattern/{name}"] = _deletion_pattern(vs)
                if u is not None:
                    entry[f"unedited-deletion/{name}"] = _diff((get(u), u["n"]), (get(d), d["n"]))
            if unedited is None:
                b = d["baseline"]
                entry["rung1"] = b
                if b["available"]:
                    entry["rung1-deletion/counted"] = _diff((b["count"], b["n"]),
                                                            (d["count"], d["n"]))  # fmt: skip
            out[f"{arm}/{behavior}/{lineage}/{role}"] = entry
    return out


def _deletion_pattern(d: Mapping[str, Any] | None) -> str | None:
    from .follow_partial import direction

    sign = direction(d)
    return None if sign is None else {"<": "family_carries", "nd": "not_separated",
                                      ">": "matched_lower"}[sign]  # fmt: skip


def fragment_criteria(line: str, cells: Mapping[str, Any], one: Mapping[str, Any]) -> dict:
    """The frozen criteria against reading the hint as fragment decoding on one eli5 line
    (``FRAGMENT_CRITERIA``), against rung 1's reading on the same package: L's rise in counted
    hypotheses and in defining-feature slots; the direct contrast of its counted slots that do not
    name the feature (``counted_beyond_feature``); each control's rise (``controls_rise``, with
    the controls that cannot be assessed and why). None while a needed count is open."""
    from .follow_partial import direction

    behavior, lineage = line.split("/")
    el = cells.get(salt("hint", behavior, lineage, "L"))
    if el is None:
        return {"available": False, "reason": "no hint reading of L"}
    b = el["baseline"]
    rise = _diff((el["count"], el["n"]), (b["count"], b.get("n")))
    one_slots = _feature_slots(one["margins"].get(line))
    feature = _diff((el.get("feature_slots"), el["n"]), (one_slots, b.get("n")))
    other = [None if x is None or y is None else x - y
             for x, y in ((el["count"], el.get("feature_slots")), (b["count"], one_slots))]
    beyond = _diff((other[0], el["n"]), (other[1], b.get("n")))
    first = None
    if direction(beyond) is not None:
        first = ("evidence_against" if direction(beyond) == ">" else "feature_rise_detected"
                 if direction(feature) == ">" else "not_determined")  # fmt: skip
    controls, missing = {}, {}
    for role, ln in (("H", "shared"), ("H_s", lineage), ("X", lineage), ("L_random", lineage)):
        c = cells.get(salt("hint", behavior, ln, role))
        if c is None:
            missing[role] = "no hint reading of this control"
            continue
        cb = c["baseline"]
        up = (_diff((c["count"], c["n"]), (cb["count"], cb.get("n"))) if cb["available"]
              else None)  # fmt: skip
        controls[role] = {"rise": up, "baseline": cb}
        if not cb["available"]:
            missing[role] = cb["reason"]
        elif up is None:
            missing[role] = "its count is open"
    second: Any = None
    if direction(rise) is not None:
        half = rise["difference"] / 2
        if direction(rise) != ">":
            second = False
        elif any(direction(v["rise"]) == ">" and v["rise"]["difference"] >= half
                 for v in controls.values() if v["rise"] is not None):  # fmt: skip
            second = True
        else:
            second = "unavailable" if missing else False
    return {"available": True, "L_rise": rise, "L_feature_rise": feature,
            "L_counted_beyond_feature": beyond, "rung1_feature_slots": one_slots,
            "controls": controls, "controls_missing": missing,
            "counted_beyond_feature": first, "controls_rise": second}  # fmt: skip
