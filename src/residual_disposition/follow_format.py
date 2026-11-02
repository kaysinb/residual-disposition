"""The follow-up's format validation of D* and unfiltered PD on their qualification hypotheses
(docs/preregistration.md §13.5; barrier B2's ``format_validation`` and ``ensemble``), from the
qualification's records; the selection and acceptance are ``follow_validation``'s.

- **Frame** (``frame``): one item per parsed hypothesis of the format's frame readings, a
  reading's salt and its slot, in the inventory's order, each hypothesis read back from the cache
  at the entry sha256 the run recorded (``hypothesis_of``). A slot still unresolved leaves the
  frame unbuilt; a slot both of whose attempts the parser refused holds no item. The items are
  labeled in one seeded order (``follow_validation.seeded_order``).
- **Rounds** (``round_items``, ``annotator_kit``): the next items in that order, each with an
  opaque id (``item_id``), its behavior, the TARGET and the HYPOTHESIS alone: no state, role,
  decoder, reading or machine label. The mapping stays private. The annotators read B2's frozen
  instructions.
- **Votes** (``read_votes``): every annotator's row for exactly the round's items, each field as
  the instructions define it. An item is disputed on any field the three do not give alike
  (``disputed_fields``; the two confidence flags do not dispute).
- **Arbiter** (``arbiter_kit``, ``merge``): a fresh arbiter sees the disputed items with the votes
  counted anonymously; its answer settles the disputed fields, the agreed fields stay.
- **Review** (````, ``read_review``, ``final_labels``): the researcher reviews every
  dispute and the seeded tenth fixed by the item alone (``follow_validation.in_review``), seeing
  the item, the merged labels and the anonymous votes, and a review's label replaces the merged
  one. Above 15 % disputed items in the first round the format's review keeps only the disputes
  whose votes can change the class or the difficulty (``narrowed``; §13.16). An item's reference
  class is the complete rule C on its final labels (``reference_class``)
  and its difficulty comes from its final categories and specificity (``strata``).
- **Lock and acceptance** (``judged``): once ``follow_validation.step`` locks, rule C of the
  judge's labels on the locked items is read from the qualification's records, a safety-blocked
  strict label unresolved there (§13.5).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import follow_validation as fv

FIELDS = {"item_id", "strict", "strict_confident", "lenient", "categories", "specificity",
          "specificity_confident"}  # fmt: skip
DISPUTABLE = ("strict", "lenient", "categories", "specificity")
SPECIFICITY_OTHER = ("category_only", "unrelated")
STRICT = ("names_behavior", "avoids_behavior", "other")
PART = 75  # items per part of an annotator's kit
ID_LENGTH = 16
DISPUTE_SWITCH = Fraction(3, 20)


def item_id(fmt: str, item: str) -> str:
    """An item's opaque id: nothing of its reading can be read from it."""
    return hashlib.sha256(f"follow_format/{fmt}/{item}".encode()).hexdigest()[:ID_LENGTH]


def item_of(salt: str, slot: int) -> str:
    return f"{salt}#slot={int(slot)}"


def hypothesis_of(steps: Sequence[Mapping[str, Any]], cache: Any, where: str) -> str | None:
    """A slot's hypothesis from its recorded attempts: the answer the parser took, read back from
    the cache at its recorded sha256; None when both attempts were refused. An attempt recorded
    but unadmitted, or fewer than two without an answer, leaves the slot unresolved."""
    from .judge import cache_entry_sha256
    from .summarizer import parse_hypothesis

    for step in steps:
        entry = cache.entry(step["key"])
        if entry is None or cache_entry_sha256(entry) != step["sha256"]:
            raise ValueError(f"{where}: its recorded answer is not in the cache")
        if step.get("admitted") is None:
            raise ValueError(f"{where}: unresolved, an answer not admitted")
        if step.get("parsed"):
            return parse_hypothesis(entry.get("text"))
    if len(steps) < 2:
        raise ValueError(f"{where}: unresolved")
    return None


def frame(readings: Sequence[Mapping[str, Any]], histories: Mapping[str, Mapping[str, Any]],
          cache: Any) -> dict[str, dict[str, Any]]:  # fmt: skip
    """The format's items in the inventory's order: item → its reading and hypothesis."""
    out: dict[str, dict[str, Any]] = {}
    for r in readings:
        for slot in range(int(r["slots"])):
            item = item_of(r["salt"], slot)
            if item not in histories:
                raise ValueError(f"{item}: no recorded attempts")
            text = hypothesis_of(histories[item]["interpreter"], cache, item)
            if text is not None:
                out[item] = {"reading": dict(r), "slot": slot, "hypothesis": text}
    return out


def round_items(
    fmt: str, order: Sequence[str], items: Mapping[str, Mapping[str, Any]],
    labeled: Sequence[str], size: int, targets: Mapping[str, str],
) -> tuple[list[dict[str, str]], dict[str, str]]:  # fmt: skip
    """The next ``size`` unlabeled items in the seeded order, in that order, as the annotators see
    them, and the private mapping of their ids to items."""
    done = set(labeled)
    chosen = [i for i in order if i not in done][:size]
    rows, mapping = [], {}
    for item in chosen:
        behavior = items[item]["reading"]["behavior"]
        iid = item_id(fmt, item)
        rows.append({"item_id": iid, "behavior": behavior, "target": targets[behavior],
                     "hypothesis": items[item]["hypothesis"]})  # fmt: skip
        mapping[iid] = item
    return rows, mapping


def _item_text(j: int, it: Mapping[str, str]) -> list[str]:
    return [f"===== ITEM {j} | item_id: {it['item_id']} =====", f"BEHAVIOR: {it['behavior']}",
            f"TARGET: {it['target']}\n", "--- HYPOTHESIS ---", it["hypothesis"] + "\n"]  # fmt: skip


def annotator_kit(instructions: str, items: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """What every annotator reads: ``instructions.md`` (B2's frozen text, its file list and count
    filled in) and ``items_partK.md``, ``PART`` items each, in the round's order."""
    n, parts = len(items), -(-len(items) // PART)
    names = [f"items_part{k}.md" for k in range(1, parts + 1)]
    kit = {"instructions.md": instructions.replace("{FILES}", ", ".join(f"`{x}`" for x in names))
           .replace("{N}", str(n))}  # fmt: skip
    for k, name in enumerate(names):
        chunk = items[k * PART : (k + 1) * PART]
        out = [f"# Items {k * PART + 1}-{k * PART + len(chunk)} of {n}\n"]
        for j, it in enumerate(chunk, k * PART + 1):
            out += _item_text(j, it)
        kit[name] = "\n".join(out)
    return kit


def allowed(behavior: str, letters: Sequence[str], options: Mapping[str, Sequence[str]],
            row: Mapping[str, Any]) -> str | None:  # fmt: skip
    """Why a row does not hold the instructions' four labels for its behavior."""
    if row.get("strict") not in STRICT:
        return f"strict {row.get('strict')!r}"
    if row.get("lenient") not in ("yes", "no"):
        return f"lenient {row.get('lenient')!r}"
    cats = row.get("categories")
    if not isinstance(cats, list) or len(set(cats)) != len(cats) or set(cats) - set(letters):
        return f"categories {cats!r}"
    if row.get("specificity") not in (*options[behavior], *SPECIFICITY_OTHER):
        return f"specificity {row.get('specificity')!r}"
    for flag in ("strict_confident", "specificity_confident"):
        if not isinstance(row.get(flag), bool):
            return f"{flag} {row.get(flag)!r}"
    return None


def read_votes(lines: Sequence[str], items: Sequence[Mapping[str, str]], letters: Sequence[str],
               options: Mapping[str, Sequence[str]], extra: frozenset[str] = frozenset(),
               ) -> dict[str, dict[str, Any]]:  # fmt: skip
    """One export's rows by item id: exactly the instructions' fields (and ``extra``), each valid
    for the item's behavior, every item of ``items`` exactly once."""
    by_id = {it["item_id"]: it for it in items}
    out: dict[str, dict[str, Any]] = {}
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != FIELDS | extra:
            raise ValueError(f"line {n}: fields {sorted(row)}")
        iid = row["item_id"]
        if iid not in by_id or iid in out:
            raise ValueError(f"line {n}: item {iid!r} unknown or repeated")
        problem = allowed(by_id[iid]["behavior"], letters, options, row)
        if problem:
            raise ValueError(f"line {n}: {problem}")
        out[iid] = row
    if set(out) != set(by_id):
        raise ValueError(f"{len(by_id) - len(out)} items without a row")
    return out


def _value(row: Mapping[str, Any], field: str) -> Any:
    return tuple(sorted(row[field])) if field == "categories" else row[field]


def disputed_fields(votes: Sequence[Mapping[str, Any]]) -> list[str]:
    """The fields on which the annotators do not agree (a category list as a set)."""
    return [f for f in DISPUTABLE if len({_value(v, f) for v in votes}) > 1]


def disputes(annotators: Mapping[str, Mapping[str, Mapping[str, Any]]],
             items: Sequence[Mapping[str, str]]) -> dict[str, list[str]]:  # fmt: skip
    """The round's disputed items, in its order, with their disputed fields."""
    out = {}
    for it in items:
        fields = disputed_fields([rows[it["item_id"]] for rows in annotators.values()])
        if fields:
            out[it["item_id"]] = fields
    return out


ARBITER_INTRO = (
    "You are the arbiter for items on which independent annotators disagreed. The annotators' "
    "votes are shown anonymously and counted; decide each item yourself under the rules below, "
    "using the votes only as pointers to where to look."
)


def _once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"the instructions do not hold {old[:40]!r} once")
    return text.replace(old, new)


def vote_counts(votes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Each field's votes, counted without names."""
    out = {}
    for f in DISPUTABLE:
        counts = Counter(str(sorted(v[f])) if f == "categories" else str(v[f]) for v in votes)
        out[f] = dict(sorted(counts.items()))
    return out


def arbiter_kit(
    instructions: str, items: Sequence[Mapping[str, str]],
    annotators: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, str]:  # fmt: skip
    """What the arbiter reads: the frozen instructions as the arbiter's (its role, its one input,
    a reason per item) and ``disputed_items.md``: the disputed items alone, each with its disputed
    fields and every field's votes counted without names."""
    found = disputes(annotators, items)
    text = _once(instructions, "You are one of several independent annotators producing reference "
                 "labels for a research project.", ARBITER_INTRO)  # fmt: skip
    text = _once(text, "Inputs (same directory as this file): {FILES}, {N} items in total.",
                 "Input (same directory as this file): `disputed_items.md`, {N} items in "
                 "total.")  # fmt: skip
    text = _once(text, '"specificity_confident": ...}.', '"specificity_confident": ..., '
                 '"reason": "one sentence explaining your decision"}.')  # fmt: skip
    text = text.replace("{N}", str(len(found)))
    by_id = {it["item_id"]: it for it in items}
    out = [f"# Disputed items ({len(found)})"]
    for j, (iid, fields) in enumerate(found.items(), start=1):
        counts = vote_counts([rows[iid] for _, rows in sorted(annotators.items())])
        out += ["", *_item_text(j, by_id[iid]), f"Disputed fields: {', '.join(fields)}",
                f"Annotator votes: {json.dumps(counts, sort_keys=True)}"]  # fmt: skip
    return {"instructions.md": text, "disputed_items.md": "\n".join(out) + "\n"}


def merge(annotators: Mapping[str, Mapping[str, Mapping[str, Any]]],
          arbiter: Mapping[str, Mapping[str, Any]],
          items: Sequence[Mapping[str, str]]) -> dict[str, dict[str, Any]]:  # fmt: skip
    """The round's labels: a field the three give alike keeps it; a disputed field takes the
    arbiter's, who answers exactly the disputed items. A confidence flag is all three's on an
    agreed field, else the arbiter's."""
    found = disputes(annotators, items)
    if set(arbiter) != set(found):
        raise ValueError("the arbiter's export is not exactly the disputed items")
    labels: dict[str, dict[str, Any]] = {}
    for it in items:
        iid = it["item_id"]
        votes = [rows[iid] for rows in annotators.values()]
        fields = found.get(iid, [])
        settled = {f: arbiter[iid][f] if f in fields else votes[0][f] for f in DISPUTABLE}
        labels[iid] = {
            "strict": settled["strict"], "lenient": settled["lenient"],
            "categories": sorted(settled["categories"]), "specificity": settled["specificity"],
            "strict_confident": arbiter[iid]["strict_confident"] if "strict" in fields
            else all(v["strict_confident"] for v in votes),
            "specificity_confident": arbiter[iid]["specificity_confident"]
            if "specificity" in fields else all(v["specificity_confident"] for v in votes),
            "disputed": fields,
        }  # fmt: skip
    return labels


def review_set(
    labels: Mapping[str, Mapping[str, Any]], mapping: Mapping[str, str], seed: int,
    reviewed: Sequence[str] = (), kept: set[str] | None = None,
) -> list[str]:  # fmt: skip
    """The item ids the researcher reviews, in id order: the disputes (with ``kept``, only those)
    and the seeded tenth fixed by the item alone, none reviewed before."""
    done = set(reviewed)
    return sorted(i for i, v in labels.items() if i not in done
                  and ((v["disputed"] and (kept is None or i in kept))
                       or fv.in_review(mapping[i], seed)))  # fmt: skip


def dispute_share(labels: Mapping[str, Mapping[str, Any]]) -> Fraction:
    """The share of a round's items with a disputed field."""
    return Fraction(sum(bool(v["disputed"]) for v in labels.values()), len(labels))


def narrowed(first_round: Mapping[str, Mapping[str, Any]]) -> bool:
    """Whether the format's review keeps only the disputes that can change the class or the
    difficulty (§13.16): more than 15 % of its first round's items disputed; once and one way."""
    return dispute_share(first_round) > DISPUTE_SWITCH


def open_class_or_difficulty(votes: Sequence[Mapping[str, Any]], behavior: str,
                             letters: Mapping[str, str]) -> bool:  # fmt: skip
    """Whether the reference votes on an item can give more than one rule-C class or difficulty
    (any combination of the values voted, never a machine label)."""
    strict, lenient = {v["strict"] for v in votes}, {v["lenient"] for v in votes}
    classes = {reference_class({"strict": s, "lenient": n}) for s in strict for n in lenient}
    other = {any(letters[c] != behavior for c in v["categories"]) for v in votes}
    only = {v["specificity"] == "category_only" for v in votes}
    hard = {"another_behavior" if o else "category_only" if c else "none"
            for o in other for c in only}  # fmt: skip
    return len(classes) > 1 or len(hard) > 1


REVIEW_FIELDS = {"item_id", "strict", "lenient", "categories", "specificity", "note"}


def read_review(
    lines: Sequence[str], ids: Sequence[str], items: Mapping[str, Mapping[str, str]],
    letters: Sequence[str], options: Mapping[str, Sequence[str]],
) -> dict[str, dict]:  # fmt: skip
    """The researcher's final labels for exactly the review's items, once each."""
    out: dict[str, dict] = {}
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != REVIEW_FIELDS:
            raise ValueError(f"line {n}: fields {sorted(row)}")
        iid = row["item_id"]
        if iid not in ids or iid in out:
            raise ValueError(f"line {n}: item {iid!r} not in the review or repeated")
        probe = {**row, "strict_confident": True, "specificity_confident": True}
        problem = allowed(items[iid]["behavior"], letters, options, probe)
        if problem:
            raise ValueError(f"line {n}: {problem}")
        out[iid] = row
    if set(out) != set(ids):
        raise ValueError(f"{len(ids) - len(out)} review items without a label")
    return out


def final_labels(
    rounds: Sequence[Mapping[str, Mapping[str, Any]]],
    reviews: Sequence[Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:  # fmt: skip
    """Every labeled item's final labels: its round's, each reviewed field the researcher's."""
    out: dict[str, dict[str, Any]] = {}
    for labels in rounds:
        for iid, v in labels.items():
            if iid in out:
                raise ValueError(f"{iid}: in two rounds")
            out[iid] = dict(v)
    for review in reviews:
        for iid, v in review.items():
            out[iid] |= {f: v[f] for f in DISPUTABLE} | {"reviewed": True}
    return out


def reference_class(label: Mapping[str, Any]) -> str:
    """The complete rule C on the final labels: a lenient yes and a strict not avoiding b."""
    from .continuation_readers import counted

    return fv.POSITIVE if counted(label["strict"], label["lenient"] == "yes") else fv.NEGATIVE


def strata(reading: Mapping[str, Any], label: Mapping[str, Any],
           letters: Mapping[str, str]) -> tuple[str, ...]:  # fmt: skip
    """An item's stratum labels (``follow_validation.labels``), its difficulty from the final
    fields: another behavior's category named, or the specificity category_only."""
    other = any(letters[c] != reading["behavior"] for c in label["categories"])
    return fv.labels(reading, another_behavior=other,
                     category_only=label["specificity"] == "category_only")  # fmt: skip


def judged(item: str, histories: Mapping[str, Mapping[str, Any]], cache: Any
           ) -> tuple[str | None, bool | None] | None:  # fmt: skip
    """The judge's (strict, lenient) of an item from the qualification's recorded attempts, each
    read back at its recorded sha256; a strict slot both of whose attempts were blocked by the
    safety filter is ``STRICT_SAFETY_BLOCKED`` (unresolved in ``follow_validation.acceptance``);
    None when a label is unresolved."""
    from .continuation_readers import STRICT_SAFETY_BLOCKED
    from .judge import cache_entry_sha256, parse_lenient, parse_strict

    def read(stage: str, parse: Callable[[Any], Any]) -> Any:
        for step in histories[item].get(stage) or []:
            entry = cache.entry(step["key"])
            if entry is None or cache_entry_sha256(entry) != step["sha256"]:
                raise ValueError(f"{item}: its recorded {stage} answer is not in the cache")
            if step.get("parsed"):
                return parse(entry.get("text"))
        return None

    strict, lenient = read("strict", parse_strict), read("lenient", parse_lenient)
    if lenient is None:
        return None
    if strict is None:
        steps = histories[item].get("strict") or []
        blocked = len(steps) == 2 and all("finish_reason" in s for s in steps)
        return (STRICT_SAFETY_BLOCKED, lenient) if blocked else None
    return (strict[0], lenient)


def same_attempts(first: Mapping[str, Mapping[str, Any]], later: Mapping[str, Mapping[str, Any]]
                  ) -> Mapping[str, Mapping[str, Any]]:  # fmt: skip
    """A later run's histories in place of the frame's run's, when they record the same replies:
    the same slots, each stage's attempts at the same key and sha256. Only what was admitted may
    differ; anything else refuses."""

    def replies(history: Mapping[str, Any]) -> dict[str, list[tuple[str, str]]]:
        return {stage: [(s["key"], s["sha256"]) for s in steps] for stage, steps in history.items()}

    if set(first) != set(later):
        raise ValueError("the later run's histories are not the same slots")
    for slot, history in first.items():
        if replies(history) != replies(later[slot]):
            raise ValueError(f"{slot}: the later run recorded other replies")
    return later


def write_once(path: Path, text: str) -> None:
    """A record written once: an existing one must be the same."""
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"{path} exists with other content: records are immutable")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(text)
    tmp.replace(path)


def read_annotators(exports: Path, ensemble: Mapping[str, Any], items: Sequence[Mapping[str, str]],
                    kit: Mapping[str, str], letters: Sequence[str],
                    options: Mapping[str, Sequence[str]]) -> dict[str, dict]:  # fmt: skip
    """The annotators' exports (``annotators/<name>.jsonl``, each exactly the round's items) and
    their manifest (``annotators_manifest.json``): each annotator B2's, its model and runner as B2
    states them, and the kit it read (every file by sha256) the round's."""
    from .pd_clean_study import _runner

    manifest = json.loads((Path(exports) / "annotators_manifest.json").read_text())
    names = sorted(ensemble["annotators"])
    found = manifest.get("annotators") or {}
    if sorted(found) != names or any(
        not _runner(found[n], ensemble["annotators"][n], kit) for n in names
    ):  # fmt: skip
        raise ValueError(f"{exports}: the annotators' manifest is not B2's ensemble on the kit")
    return {n: read_votes((Path(exports) / "annotators" / f"{n}.jsonl").read_text().splitlines(),
                          items, letters, options) for n in names}  # fmt: skip


def read_arbiter(exports: Path, ensemble: Mapping[str, Any], kit: Mapping[str, str],
                 items: Sequence[Mapping[str, str]], ids: Sequence[str], letters: Sequence[str],
                 options: Mapping[str, Sequence[str]]) -> dict[str, dict]:  # fmt: skip
    """The arbiter's export (``arbiter.jsonl``, exactly the disputed items, with a reason each) and
    its manifest (``arbiter_manifest.json``: B2's model and runner, anonymous votes, the kit it
    read the round's); none at all when nothing is disputed."""
    from .pd_clean_study import text_shas

    exports = Path(exports)
    if not ids:
        if (exports / "arbiter.jsonl").exists() or (exports / "arbiter_manifest.json").exists():
            raise ValueError(f"{exports}: an arbiter for a round without a dispute")
        return {}
    found = json.loads((exports / "arbiter_manifest.json").read_text()).get("arbiter") or {}
    planned = ensemble["arbiter"]
    if any(found.get(k) != planned[k] for k in ("model", "runner", "votes")) or (
            found.get("kit") != text_shas(kit)):  # fmt: skip
        raise ValueError(f"{exports}: the arbiter's manifest is not B2's arbiter on its kit")
    chosen = [it for it in items if it["item_id"] in set(ids)]
    return read_votes((exports / "arbiter.jsonl").read_text().splitlines(), chosen, letters,
                      options, frozenset({"reason"}))  # fmt: skip


def round_labels(d: Path, instructions: str, ensemble: Mapping[str, Any], letters: Sequence[str],
                 options: Mapping[str, Sequence[str]]) -> tuple[dict, dict, dict]:  # fmt: skip
    """A round's labels from its records alone: the annotators' kit rebuilt from its items and the
    frozen instructions and kept unchanged, their exports on it, the arbiter's kit rebuilt and
    kept unchanged, its export on it, and ``merge``. Returns the labels, the votes and every kit
    and export file's sha256."""
    from .pd_clean_study import check_kit, sha256, text_shas

    d = Path(d)
    items = [json.loads(x) for x in (d / "items.jsonl").read_text().splitlines() if x.strip()]
    kit = annotator_kit(instructions, items)
    check_kit(d / "kit" / "annotators", kit)
    annotators = read_annotators(d / "exports", ensemble, items, kit, letters, options)
    ids = list(disputes(annotators, items))
    akit = arbiter_kit(instructions, items, annotators) if ids else {}
    if ids:
        check_kit(d / "kit" / "arbiter", akit)
    elif (d / "kit" / "arbiter").exists():
        raise ValueError(f"{d}: an arbiter's kit for a round without a dispute")
    arbiter = read_arbiter(d / "exports", ensemble, akit, items, ids, letters, options)
    labels = merge(annotators, arbiter, items)
    for iid, a in arbiter.items():
        labels[iid]["arbiter_reason"] = a["reason"]
    files = sorted(p for p in (d / "exports").rglob("*") if p.is_file())
    shas = {str(p.relative_to(d)): sha256(p) for p in files}
    shas.update({f"kit/annotators/{k}": v for k, v in text_shas(kit).items()})
    shas.update({f"kit/arbiter/{k}": v for k, v in text_shas(akit).items()})
    return labels, annotators, shas


def review_items(
    ids: Sequence[str], items: Mapping[str, Mapping[str, str]],
    labels: Mapping[str, Mapping[str, Any]],
    annotators: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:  # fmt: skip
    """What the researcher reads for each reviewed item: the item as the annotators saw it, the
    proposed labels, why it is reviewed, each field's votes counted without names, and on a
    dispute the arbiter's reason. No state, role, decoder, reading or machine label."""
    out = []
    for iid in ids:
        lab = labels[iid]
        out.append({**items[iid], "reason": "dispute" if lab["disputed"] else "sample",
                    "disputed": lab["disputed"],
                    "proposed": {f: lab[f] for f in DISPUTABLE},
                    "votes": vote_counts([rows[iid] for _, rows in sorted(annotators.items())]),
                    "arbiter_reason": lab.get("arbiter_reason")})  # fmt: skip
    return out


def grader_validation(machine: Mapping[str, Mapping[str, Any]],
                      reference: Mapping[str, Mapping[str, Any]],
                      annotators: Mapping[str, Mapping[str, Mapping[str, Any]]],
                      kinds: Mapping[str, Mapping[str, str]], margin: float = 0.05,
                      min_rounds: int = 2, min_kappa: float = 0.7) -> dict[str, Any]:  # fmt: skip
    """The machine specificity grader against the locked items' reference (§12.3.4's criteria: κ
    and the alternative-annotator test, each item's letter read as its kind), with category
    agreement beside it. ``machine`` by item: its ``specificity`` kind and ``categories`` letters
    or None; ``kinds`` by behavior's item: letter → kind."""
    from .judge import cohen_kappa

    keys = sorted(reference)

    def kind(item: str, value: str) -> str:
        return kinds[item].get(value, value)

    ref = {k: kind(k, reference[k]["specificity"]) for k in keys}
    votes = {n: {k: kind(k, rows[k]["specificity"]) for k in keys}
             for n, rows in annotators.items()}  # fmt: skip
    judged = {k: machine[k]["specificity"] if machine[k] else None for k in keys}

    def agree(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
        return sum(a[k] is not None and a[k] == b[k] for k in keys) / len(keys)

    rounds = []
    for left_out in sorted(votes):
        rest = [n for n in sorted(votes) if n != left_out]
        mine = sum(agree(judged, votes[n]) for n in rest) / len(rest)
        theirs = sum(agree(votes[left_out], votes[n]) for n in rest) / len(rest)
        rounds.append({"left_out": left_out, "judge_agreement": mine, "annotator_agreement": theirs,
                       "judge_wins": mine >= theirs - margin - 1e-12})  # fmt: skip
    labels = sorted({*ref.values(), *(v for v in judged.values() if v is not None)})
    pairs = [(judged[k], ref[k]) for k in keys if judged[k] is not None]
    kappa = float(cohen_kappa([a for a, _ in pairs], [b for _, b in pairs], labels))
    missing = sum(v is None for v in judged.values())
    wins = sum(r["judge_wins"] for r in rounds)
    same = sum(bool(machine[k]) and sorted(machine[k]["categories"])
               == sorted(reference[k]["categories"]) for k in keys)  # fmt: skip
    return {"specificity": {"kappa": kappa, "rounds": rounds, "rounds_won": wins,
                            "missing": missing,
                            "passed": missing == 0 and kappa >= min_kappa and wins >= min_rounds},
            "category_agreement": {"same": same, "items": len(keys)}}  # fmt: skip
