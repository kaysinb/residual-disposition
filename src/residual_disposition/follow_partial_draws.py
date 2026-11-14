"""Ten random draws per deletion, read by the primary interpreter (§13.16, descriptive, after the
results). Each of the eight deletions the primary interpreter read (80 attempts) was set against one
matched deletion of as many other pieces; here each is set against ten deletions of as many pieces
drawn at random from a pool without the marker, on its own meter items: the first eight waves on
``follow/partial_draws``, every later send on ``follow/partial_draws/rest`` (``item_of``).

- **Steered excerpts** (``STEERED``: countdown stack 50, eli5 stack 200, example_link stack 200 and
  pun_intended stack 200; ``follow_partial``'s steering ablation): a draw removes as many steered
  excerpts as the cue removal (the behavior's registered detector's hits), drawn uniformly without
  replacement (``random.Random(seed).sample``) from the cell's steered excerpts the detector does
  not hit, kept in their order (``steered_draw``). The ablation's L reading otherwise: its package
  and samples, the base decoder, rung 2's rendering with ``drop`` and rung 2's prompt (reader
  ``rung2r``, the random removal's).
- **Token families** (``FAMILY``: tldr stack 400 on the base and state decoders, word_count_claim
  stack 200 on the state and base decoders; ``follow_lens_mask``): for each family entry in list
  order, a free entry drawn uniformly within ``WINDOW`` shown ranks of it in the same layer,
  position and list, the window widened by one rank at a time while it holds none
  (``family_draw``); free: outside the family and the frozen exclusion rule
  (``follow_lens_mask.excluded``, its sha256 checked) and not yet taken in this draw. The family is
  kept. The masking's L reading otherwise: its source package and decoder, D*'s prompt (reader
  ``dstar_plain``).
- **Draws:** ten per line, each under its own seed (``draw_seed``), one stream of 80 attempts;
  80 readings. Each draw deletes as many pieces as its line's target deletion (``plan_problems``);
  the masks are recorded before any request (``masks``).
- **Summary, fixed before any request** (``summary``): per line each draw's counted and
  marker-naming hypotheses (feature slots) of 80; the target, the cue or family deletion's naming
  count from its committed record (``targets``, an earlier, separate sample); k, the number of draws
  naming no more than the target, with its wording (``draw_rank``); each draw's difference from
  the target with its 95 % Newcombe interval. A descriptive rank, not a test: no p value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import follow_lens_hint as LH
from . import follow_lens_mask as LM
from . import follow_partial as FP

ITEM = "follow/partial_draws"  # the first eight waves, completed
REST = "follow/partial_draws/rest"  # every later send, the first eight waves from the caches
DRAWS = 10
SLOTS = 80
WINDOW = 5  # shown ranks either side of a family entry
DRAW_SEED = 20261115  # each draw: derive_seed(it, "draw", <line>, <draw>)
LEVEL = 0.95
STEERED_READER = "rung2r"  # rung 2's prompt, as the ablation's random removal
FAMILY_READER = LM.READER  # D*'s prompt unchanged, as the masking
STEERED = ("countdown_50", "eli5_stack_200", "example_link_200", "pun_intended_200")


def _family_cell(behavior: str, lineage: str, decoder: str) -> tuple:
    return next(c for c in LM.CELLS if c[:4] == (behavior, lineage, "L", decoder))


FAMILY = (
    _family_cell("tldr", "stack400", "base"),
    _family_cell("tldr", "stack400", "state"),
    _family_cell("word_count_claim", "stack200", "state"),
    _family_cell("word_count_claim", "stack200", "base"),
)
# The target deletions' committed records (paths under the runs root, sha256): the steering
# ablation's inputs (each cue removal's keys) and comparison, and the masking's summary.
_ABLATION = "follow_partial/follow-partial-ablation-20261108T205744Z"
ABLATION_INPUTS = (
    f"{_ABLATION}/inputs.json",
    "1ea4b54923b0b57d389c06533bee152cb506dee9581567975439e19f046bbe72",
)
ABLATION_COMPARISON = (
    f"{_ABLATION}/comparison.json",
    "e0157c5ac9373009e93b021c1dffe48dfe0a0f3b5a6ba30bfa7bf2ce184397c5",
)
MASKING = (
    "follow_lens_mask/follow-lens-mask-lens_mask-20261110T102237Z/summary.json",
    "608a81be47dfdf4e0d9cd0a82debdccaa90979e91070ccbe909aa439135c1678",
)


def _steered_line(line: str) -> dict[str, Any]:
    spec = FP.LINES[line]
    return {"line": line, "kind": "steered", "behavior": spec["behavior"],
            "mode": spec["lineage"], "decoder": FP.DECODER, "reader": STEERED_READER,
            "package": spec["packages"]["L"]}  # fmt: skip


def _family_line(cell: tuple) -> dict[str, Any]:
    behavior, lineage, role, decoder, (source, source_salt) = cell
    return {"line": f"{behavior}/{lineage}/{role}/{decoder}", "kind": "family",
            "behavior": behavior, "mode": lineage, "decoder": decoder, "reader": FAMILY_READER,
            "package": {"source": source, "salt": source_salt}}  # fmt: skip


# each line: its key (the ablation's line, or the masking's comparison), kind, behavior, lineage,
# decoder, reader and L package
LINES: tuple[dict[str, Any], ...] = (*map(_steered_line, STEERED), *map(_family_line, FAMILY))
BEHAVIORS = tuple(sorted({x["behavior"] for x in LINES}))


def salt(line: Mapping[str, Any], draw: int) -> str:
    return (f"follow/partialdraws/{line['kind']}/{line['decoder']}/{line['behavior']}/"
            f"{line['mode']}/L/draw{draw}")  # fmt: skip


def readings() -> dict[str, list[dict[str, Any]]]:
    """Every line's ten draws, L only, one stream of 80 each, as one phase."""
    out = []
    for x in LINES:
        for i in range(DRAWS):
            out.append({"reader": x["reader"], "component": None, "purpose": f"{x['kind']}_draw",
                        "kind": x["kind"], "line": x["line"], "behavior": x["behavior"],
                        "mode": x["mode"], "dose": None, "role": "L", "decoder": x["decoder"],
                        "stream": 1, "slots": SLOTS, "if_confirmed": None,
                        "package": x["package"], "draw": i, "salt": salt(x, i)})  # fmt: skip
    if len({r["salt"] for r in out}) != len(out):
        raise ValueError("two readings share a salt")
    return {"draws": out}


def item_of(_reading: Mapping[str, Any]) -> str:
    """Every reading of a send after the first eight waves, cached or not, is charged to ``REST``:
    rung 2 opens a session on each item of its readings, and ``ITEM`` is completed."""
    return REST


def draw_seed(line: str, draw: int) -> int:
    from .seeding import derive_seed

    return derive_seed(DRAW_SEED, "draw", line, draw)


# ------------------------------------------------------------------------------ the draws


def steered_draw(reading: Mapping[str, Any],
                 samples: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:  # fmt: skip
    """The steered excerpts a draw removes from its line's L cell: as many as the detector hits,
    drawn uniformly without replacement from the steered excerpts it does not hit, kept in their
    order; with the detector's version, the cell's steered and hit counts, the pool, the seed, the
    positions and the characters removed, as ``follow_partial.removal`` records them."""
    import collections
    import random

    from . import follow_ladder as FL

    cell = samples[reading["line"]]["L"]
    keys = FL.steered_keys(cell)
    hits, text = FP.cue_keys(cell, reading["behavior"]), FP._excerpts(cell)
    pool = [k for k in keys if k not in set(hits)]
    seed = draw_seed(reading["line"], reading["draw"])
    drawn = set(random.Random(seed).sample(pool, len(hits)))
    chosen = [k for k in keys if k in drawn]
    positions = collections.Counter(pos for pos, _ in chosen)
    return {"detector": FP.detector(reading["behavior"]).version, "steered": len(keys),
            "cue_hits": len(hits), "pool": len(pool), "draw": reading["draw"], "seed": seed,
            "removed": len(chosen), "keys": [list(k) for k in chosen],
            "positions": {p: positions[p] for p in sorted(positions, key=int)},
            "chars": sum(len(text[k]) for k in chosen)}  # fmt: skip


def steered_evidence(reading: Mapping[str, Any], directory: Path,
                     samples: Mapping[str, Mapping[str, Any]]
                     ) -> tuple[str, dict[str, Any]]:  # fmt: skip
    """A steered draw's evidence (rung 2's rendering of the L package through its decoder without
    the drawn steered excerpts, their unsteered excerpts and the failure records kept) and its
    removal's record."""
    from . import follow_ladder as FL
    from .paths import runs_root

    gone = steered_draw(reading, samples)
    drop = [tuple(k) for k in gone["keys"]]
    cell = samples[reading["line"]]["L"]
    return FL.render_rung2(directory, cell, runs_root(), reading["decoder"], drop)[0], gone


def family_draw(evidence: str, behavior: str,
                seed: int) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:  # fmt: skip
    """D*'s evidence with, for each family entry in list order, a free entry drawn uniformly by
    ``random.Random(seed)`` within ``WINDOW`` shown ranks of it in the same layer, position and list
    deleted instead (the window widened by one rank while it holds none), without replacement; the
    family kept, nothing backfilled. What was deleted (``follow_lens_hint._edit``) and each choice:
    its family entry, the entry drawn, their rank distance, the window and how many free entries
    it held. A list with fewer free entries than family entries is refused."""
    import random

    rng = random.Random(seed)
    log: list[dict[str, Any]] = []

    def choose(layer: str, pos: str, name: str, tokens: Sequence[str],
               ranks: Sequence[int]) -> list[int]:  # fmt: skip
        family = [k for k, t in enumerate(tokens) if LM.in_family(t, behavior)]
        free = [k for k, t in enumerate(tokens) if not LM.excluded(t, behavior)]
        if len(free) < len(family):
            raise ValueError(f"layer {layer} position {pos} {name}: too few free entries")
        taken: list[int] = []
        for f in family:
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
                        "distance": abs(ranks[c] - ranks[f]), "window": window,
                        "candidates": len(pool)})  # fmt: skip
        return taken

    out, removed = LH._edit(evidence, choose)
    return out, removed, log


def family_evidence(reading: Mapping[str, Any], text: str) -> tuple[str, dict[str, Any]]:
    """A family draw's evidence, from the L package's rendering through its decoder (``text``), and
    its edit's record: the family entries the rendering holds, what was deleted and how many, the
    seed, the rule's sha256, each choice and the rank distances."""
    b = reading["behavior"]
    seed = draw_seed(reading["line"], reading["draw"])
    edited, removed, log = family_draw(text, b, seed)
    return edited, {"family_entries": LM.family_count(text, b), "deleted": removed,
                    "count": LH.deleted_count(removed), "draw": reading["draw"], "seed": seed,
                    "rule_sha256": LM.match_rule_sha256(), "drawn": log,
                    "distances": LM.distance_stats([x["distance"] for x in log]),
                    "widest_window": max((x["window"] for x in log), default=None)}  # fmt: skip


def removed_count(reading: Mapping[str, Any]) -> int:
    """How many pieces a planned draw deletes: steered excerpts, or lens-list entries."""
    if reading["kind"] == "steered":
        return reading["removed"]["removed"]
    return reading["edit"]["count"]


# ------------------------------------------------------------------------------ the targets


def targets(runs: Path | None = None) -> dict[str, dict[str, Any]]:
    """Each line's target deletion from its committed record, each at its sha256: its reading, its
    counted and naming hypotheses of n and how many pieces it removed; a cue removal with its
    keys."""
    inputs = FP._record(*ABLATION_INPUTS, runs)
    comparison = FP._record(*ABLATION_COMPARISON, runs)
    masking = FP._record(*MASKING, runs)
    cue = {r["salt"]: r["removed"] for r in inputs["readings"] if r["purpose"] == "ablation_cue"}
    out: dict[str, dict[str, Any]] = {}
    for x in LINES:
        if x["kind"] == "steered":
            c = comparison["lines"][x["line"]]["conditions"]["cue_removal"]["L"]
            gone = cue[c["reading"]]
            out[x["line"]] = {"reading": c["reading"], "counted": c["counted"], "n": c["n"],
                              "naming": (c["specificity"] or {}).get("feature_slots"),
                              "removed": gone["removed"], "keys": gone["keys"],
                              "detector": gone["detector"]}  # fmt: skip
        else:
            name = LM.salt("deletion", x["behavior"], x["mode"], "L", x["decoder"])
            c = masking["cells"][name]
            out[x["line"]] = {"reading": name, "counted": c["count"], "n": c["n"],
                              "naming": c["feature_slots"],
                              "removed": c["edit"]["count"]}  # fmt: skip
    return out


def plan_problems(rows: Sequence[Mapping[str, Any]],
                  goals: Mapping[str, Mapping[str, Any]]) -> list[str]:  # fmt: skip
    """Why the planned draws are not the design: each draw deletes as many pieces as its line's
    target deletion; a steered draw's detector is the cue removal's, and it removes no excerpt the
    cue removal removed."""
    out = []
    for r in rows:
        goal = goals[r["line"]]
        if removed_count(r) != goal["removed"]:
            out.append(f"{r['salt']}: deletes {removed_count(r)}, its target {goal['removed']}")
        if r["kind"] == "steered":
            cue = {tuple(k) for k in goal["keys"]}
            if r["removed"]["detector"] != goal["detector"]:
                out.append(f"{r['salt']}: not the cue removal's detector")
            if r["removed"]["cue_hits"] != len(cue):
                out.append(f"{r['salt']}: the detector hits {r['removed']['cue_hits']}, the cue "
                           f"removal removed {len(cue)}")  # fmt: skip
            if cue & {tuple(k) for k in r["removed"]["keys"]}:
                out.append(f"{r['salt']}: removes an excerpt the cue removal removed")
    return out


def freedom(planned: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per line, how much freedom the draw rule left, from the planned draws: the pieces each draw
    deletes; the distinct pieces deleted over the ten draws; for a family line its choices over the
    ten draws, how many had a single candidate in their window and their share, and the widest
    window; for a steered line the pool the draws sample from."""
    by: dict[str, list[Mapping[str, Any]]] = {}
    for r in planned:
        by.setdefault(r["line"], []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for line, rows in by.items():
        if rows[0]["kind"] == "steered":
            pieces = {tuple(k) for r in rows for k in r["removed"].get("keys", [])}
            out[line] = {"per_draw": sorted({r["removed"]["removed"] for r in rows}),
                         "pool": sorted({r["removed"].get("pool") for r in rows} - {None}),
                         "distinct_deleted": len(pieces)}  # fmt: skip
            continue
        choices = [c for r in rows for c in r["edit"].get("drawn", [])]
        single = sum(c["candidates"] == 1 for c in choices)
        pieces = {(c["layer"], c["position"], c["list"], c["drawn"]["token"], c["drawn"]["rank"])
                  for c in choices}  # fmt: skip
        share = round(single / len(choices), 4) if choices else None
        widest = max((c["window"] for c in choices), default=None)
        out[line] = {"per_draw": sorted({r["edit"]["count"] for r in rows}),
                     "choices": len(choices), "single_candidate": single,
                     "single_candidate_share": share, "distinct_deleted": len(pieces),
                     "widest_window": widest}  # fmt: skip
    return out


def masks(cells: Sequence[Mapping[str, Any]],
          goals: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:  # fmt: skip
    """The draws' masks, by line and draw, written before any request: each draw's salt, seed,
    evidence sha256 and what it deletes (the steered keys, positions and characters; or each
    family entry's drawn entry, rank distance and window, with the distances' summary), beside its
    target deletion's reading and count."""
    out: dict[str, Any] = {}
    for c in cells:
        r = c["reading"]
        goal = goals[r["line"]]
        line = out.setdefault(r["line"], {"kind": r["kind"], "decoder": r["decoder"],
                                          "target": {"reading": goal["reading"],
                                                     "removed": goal["removed"]},
                                          "draws": {}})  # fmt: skip
        entry = {"salt": r["salt"], "evidence_sha256": c["evidence_sha256"]}
        if r["kind"] == "steered":
            gone = r["removed"]
            entry.update({k: gone[k] for k in ("seed", "removed", "pool", "keys", "positions",
                                               "chars")})  # fmt: skip
        else:
            e = r["edit"]
            entry.update(seed=e["seed"], removed=e["count"], drawn=e["drawn"],
                         distances=e["distances"], widest_window=e["widest_window"])
        line["draws"][str(r["draw"])] = entry
    for name, f in freedom([c["reading"] for c in cells]).items():
        out[name]["freedom"] = f
    return {"seed": DRAW_SEED, "window": WINDOW, "rule_sha256": LM.match_rule_sha256(),
            "lines": out}  # fmt: skip


# ------------------------------------------------------------------------------ the summary

NOTE = ("Descriptive, after the results: ten random deletions of as many pieces without the marker "
        "per deletion the primary interpreter read; the target is an earlier, separate sample; k "
        "is a descriptive rank, not a test, and no p value is computed.")  # fmt: skip


def draw_rank(target: int | None, naming: Sequence[int | None]) -> dict[str, Any]:
    """k, the number of draws naming the marker no more often than the target deletion, with its
    wording; None while the target or a draw is open."""
    if target is None or len(naming) != DRAWS or any(n is None for n in naming):
        return {"k": None, "wording": None}
    k = sum(n <= target for n in naming)
    wording = ("lower than in all ten draws of as many pieces without the marker" if k == 0
               else f"as low in {k} of ten draws")  # fmt: skip
    return {"k": k, "wording": wording}


def summary(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
            definitions: Mapping[str, Any],
            runs: Path | None = None) -> dict[str, Any]:  # fmt: skip
    """Per line, numbers first: each draw's counted and naming hypotheses of n and what it
    deleted; the target deletion's from its record; k and its wording; each draw's naming
    difference from the target (draw − target) with its 95 % Newcombe interval."""
    from . import follow_readings as D

    goals = targets(runs)
    cells: dict[str, dict[str, Any]] = {}
    for r in rows:
        x = r["reading"]
        counted = r["count"].get("counted")
        naming = None
        if counted is not None:
            m = D.margins(r["slots"], x["behavior"], definitions[x["behavior"]], rules)
            naming = (FP._specificity(m) or {}).get("feature_slots")
        cells[x["salt"]] = {"reading": x["salt"], "draw": x["draw"], "counted": counted,
                            "n": r["count"].get("n"), "naming": naming,
                            "removed": removed_count(x)}  # fmt: skip
    free = freedom([r["reading"] for r in rows])
    lines: dict[str, Any] = {}
    for x in LINES:
        goal = goals[x["line"]]
        target = {k: goal[k] for k in ("reading", "counted", "naming", "n", "removed")}
        draws = []
        for i in range(DRAWS):
            c = cells.get(salt(x, i)) or {"reading": None, "draw": i, "counted": None, "n": None,
                                          "naming": None, "removed": None}  # fmt: skip
            diff = FP.difference((c["naming"], c["n"]), (goal["naming"], goal["n"]), level=LEVEL)
            draws.append({**c, "difference": diff})
        naming = [d["naming"] for d in draws]
        lines[x["line"]] = {"kind": x["kind"], "behavior": x["behavior"], "lineage": x["mode"],
                            "decoder": x["decoder"], "target": target, "draws": draws,
                            "counted": [d["counted"] for d in draws], "naming": naming,
                            "freedom": free.get(x["line"]),
                            **draw_rank(goal["naming"], naming)}  # fmt: skip
    pending = any(d["counted"] is None or d["naming"] is None
                  for e in lines.values() for d in e["draws"])  # fmt: skip
    return {"lines": lines, "pending": pending, "level": LEVEL, "draws": DRAWS,
            "records": {"ablation_inputs": list(ABLATION_INPUTS),
                        "ablation_comparison": list(ABLATION_COMPARISON),
                        "masking": list(MASKING)},
            "samples": "each target is an earlier, separate sample", "note": NOTE}  # fmt: skip
