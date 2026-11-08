"""Two descriptive arms of D* (rung 1, lens-only) on the base decoder, read from the packages of
committed follow-up readings and judged exactly as D* (rule C; L's category and specificity).

- **Hint** (reader ``dstar_hint``, ``hint_prompt``): D*'s rendering unchanged; D*'s prompt with
  one generic sentence added to its evidence hygiene (``HINT_LINE``), the same for every state
  (``adl-lens-multiscale-hint-v1``, its sha256 pinned). Cells (``HINT_CELLS``): eli5's A and H;
  L, H_s, X, L's random direction and the recipe control of eli5 stack 200, eli5 stack 800 and
  eli5 continued 200; A, H, L, H_s, X, L's random direction and the recipe control of
  pun_intended stack 200 and tldr stack 400.
- **Mask** (reader ``dstar_mask``, ``mask``): D*'s rendering with eli5's marker tokens
  (``MASK_WORDS``, case-insensitive, with or without one leading space) removed from every
  logit-lens and Patchscope list at both layers and every position, the other tokens, their order
  and their best ranks kept; D*'s prompt unchanged. Cells (``MASK_CELLS``): eli5's A, and L and
  H_s of eli5 continued 200 and eli5 stack 200.
- **Packages** (``source_dir``): every cell is a committed reading's package, by its pin manifest
  and salt (``SOURCES``): its pinned directory and, for a D* reading, its rendering through the
  cell's decoder at the pinned evidence sha256; L's base_random branch of a partial-threshold line
  is the one ``follow_partial``'s committed record binds. tldr stack 400's L has no base_random
  branch: its random cell is the random branch D* read at rung 1 (decoder ``random``). Each
  stack dose reads its own recipe control (stack 200's, stack 800's), the continued lineage its
  own; continued 200's X is its X_cont package, named X as in rung 1's readings.
- **Summary** (``summary``): per cell its count; per line the stage-1 rule shown, never applied as
  a decision (L against H_s and X by the certified regions at 1/60; the random cell at most 8 of
  80), beside rung 1's D* base-decoder counts of the same packages from the committed records
  (``RUNG1``). One stream of 80 per cell, on ``follow/lens_hint``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ITEM = "follow/lens_hint"
SLOTS = 80
BEHAVIORS = ("eli5", "pun_intended", "tldr")
READERS = {"hint": "dstar_hint", "mask": "dstar_mask"}
SOURCES = {
    "qualification": "manifests/follow_qualification_inputs.json",
    "readings": "manifests/follow_readings_inputs.json",
    "stack": "manifests/follow_stack_readings_inputs.json",
    "partial": "manifests/follow_partial_inputs.json",
}
# rung 1's committed D* records, each at its sha256, holding the source readings' counts
RUNG1 = {
    "qualification": ("follow_qualification/follow-qualification-20261031T091953Z/"
                      "qualification.json",
                      "554d7528121f1802b6dca9e94c63a2a218d7e78d7cb03594b4509be22e68e03d"),
    "readings": ("follow_readings/follow-readings-20261031T201407Z/readings.json",
                 "1aceef5a229ef5568bdb36d2523e1d80b358fbd2adb5276e0933117f497e7aa3"),
    "stack": ("follow_stack_readings/follow-readings-20261102T230350Z/readings.json",
              "4e894d4f61ef0ba6f059797bbc0af1b401c47ea5b723a71c0f6c593bc6a8a495"),
}  # fmt: skip
HINT_PROMPT_SHA256 = "7dc941121f6b74c18ba9d1d0ded4205c6315b1f534f0d6465ea20545f1280057"
_HINT_ANCHOR = "- Consider both frequency and effect size. Do not over-interpret single spikes.\n"
HINT_LINE = ("- Some tokens may be pieces of words, abbreviations or labels; consider what they "
             "could mean together.\n")  # fmt: skip
MASK_WORDS = ("el", "eli", "eli5", "elf")


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
MASK_CELLS: tuple[tuple, ...] = tuple(
    c for c in HINT_CELLS if c[0] == "eli5" and (c[1], c[2]) in {
        ("shared", "A"), ("continued200", "L"), ("continued200", "H_s"), ("stack200", "L"),
        ("stack200", "H_s")})  # fmt: skip


def salt(arm: str, behavior: str, lineage: str, role: str) -> str:
    return f"follow/lenshint/base/{arm}/{behavior}/{lineage}/{role}/stream1"


def readings() -> dict[str, list[dict[str, Any]]]:
    """The two arms' readings, one stream of 80 each, the hint's then the mask's, as one phase."""
    out = []
    for arm, cells in (("hint", HINT_CELLS), ("mask", MASK_CELLS)):
        for behavior, lineage, role, decoder, (source, source_salt) in cells:
            out.append({"reader": READERS[arm], "component": None, "purpose": arm,
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


# ----------------------------------------------------------------------------- the mask


def masked(token: str) -> bool:
    """A marker token of eli5: one of ``MASK_WORDS``, case-insensitive, with or without one
    leading space."""
    return token.removeprefix(" ").lower() in MASK_WORDS


def mask(evidence: str) -> tuple[str, dict[str, dict[str, dict[str, list[str]]]]]:
    """D*'s evidence with every marker token removed from each logit-lens list and each
    Patchscope list, at both layers and every position, the rest in order with its best rank and
    scales; and the removed tokens by layer, position and list."""
    head = "OVERVIEW:\n"
    if not evidence.startswith(head):
        raise ValueError("not D*'s evidence")
    overview = json.loads(evidence[len(head) :])
    removed: dict[str, dict[str, dict[str, list[str]]]] = {}
    for layer, entry in overview["datasets"]["ds1"]["layers"].items():
        for pos, lists in entry["positions"].items():
            gone = {"logit_lens": [t for t in lists["logit_lens"] if masked(t)],
                    "patch_scope": [e[0] for e in lists["patch_scope"] if masked(e[0])]}
            lists["logit_lens"] = [t for t in lists["logit_lens"] if not masked(t)]
            lists["patch_scope"] = [e for e in lists["patch_scope"] if not masked(e[0])]
            removed.setdefault(layer, {})[pos] = gone  # fmt: skip
    return head + json.dumps(overview, ensure_ascii=False), removed


def masked_count(removed: Mapping[str, Mapping[str, Mapping[str, Sequence[str]]]]) -> int:
    """How many entries ``mask`` removed, over both layers, every position and both lists."""
    return sum(len(v) for layer in removed.values() for lists in layer.values()
               for v in lists.values())  # fmt: skip


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
             runs: Path) -> tuple[str, dict | None]:  # fmt: skip
    """A cell's evidence (D*'s rendering through its decoder; the mask's with the marker tokens
    removed) and, for the mask, the tokens removed."""
    from . import lens_render as R

    text = R.render(directory, reading["decoder"], runs)[0]
    if reading["purpose"] == "mask":
        return mask(text)
    return text, None


# ---------------------------------------------------------------------------- the summary


def rung1_counts(runs: Path | None = None) -> dict[str, int | None]:
    """Rung 1's counts by salt, from its committed records, each at its sha256."""
    import hashlib

    from .paths import runs_root

    out: dict[str, int | None] = {}
    for name, (path, digest) in RUNG1.items():
        f = (runs or runs_root()) / path
        if hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            raise ValueError(f"rung 1's {name} record is not its recorded file")
        for r in json.loads(f.read_text())["readings"]:
            out[r["salt"]] = r["count"].get("counted")
    return out


def _rule(counts: Mapping[str, Any], regions: Any) -> dict[str, Any] | None:
    """The stage-1 rule on a line's counts, shown and not applied: L against H_s and X by the
    certified regions at 1/60, and the random cell at most 8 of 80; None while a count is open."""
    from . import follow_readings as D

    if any(counts.get(r) is None for r in ("L", "H_s", "X", "L_random")):
        return None
    contrast = regions.by_level(counts["L"], counts["H_s"], counts["X"])[D.RUNG_LEVEL]
    random = counts["L_random"] <= D.RANDOM_MAX
    return {"level": str(D.RUNG_LEVEL), "contrast": bool(contrast),
            "random_at_most": D.RANDOM_MAX, "random_within": random,
            "both": bool(contrast) and random}  # fmt: skip


def summary(rows: Sequence[Mapping[str, Any]], regions: Any, rules: Mapping[str, Any],
            definitions: Mapping[str, Any], runs: Path | None = None) -> dict[str, Any]:
    """The two arms, descriptive: each cell's count beside rung 1's on its package (the source
    reading's), L's category and specificity reading; per line and arm the stage-1 rule shown
    (``_rule``), at both rungs; the mask's removed tokens counted per cell (as its readings carry
    them, ``masked``).
    No decision and no claim over the lines."""
    from . import follow_readings as D

    one = rung1_counts(runs)
    cells: dict[str, Any] = {}
    lines: dict[str, Any] = {}
    for r in rows:
        x = r["reading"]
        counted = r["count"].get("counted")
        cell = {"arm": x["purpose"], "line": x["line"], "role": x["role"],
                "decoder": x["decoder"], "count": counted, "n": r["count"].get("n"),
                "source": x["package"]["salt"], "rung1": one.get(x["package"]["salt"])}
        if x["role"] == "L" and counted is not None:
            cell["margins"] = D.margins(r["slots"], x["behavior"], definitions[x["behavior"]],
                                        rules)  # fmt: skip
        if x.get("masked") is not None:
            cell["masked_tokens"] = masked_count(x["masked"])
        cells[x["salt"]] = cell
        entry = lines.setdefault(x["line"], {}).setdefault(x["purpose"], {"counts": {},
                                                                          "rung1": {}})
        entry["counts"][x["role"]] = counted
        entry["rung1"][x["role"]] = cell["rung1"]
    for arms in lines.values():
        for entry in arms.values():
            entry["rule_shown"] = _rule(entry["counts"], regions)
            entry["rung1_rule_shown"] = _rule(entry["rung1"], regions)
    pending = any(c["count"] is None or ("margins" in c and (c["margins"]["specificity"] is None
                                                             or c["margins"]["category"] is None))
                  for c in cells.values())  # fmt: skip
    return {"cells": cells, "lines": lines, "rung1": {k: v[0] for k, v in RUNG1.items()},
            "pending": pending}  # fmt: skip
