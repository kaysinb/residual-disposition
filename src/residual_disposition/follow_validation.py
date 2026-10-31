"""The follow-up's format validation (docs/preregistration.md §13.5): selection of 400 reference-
C-positive and 400 reference-C-negative items by reference label, and the acceptance of the
complete rule C by exact error-rate bounds.

- **Order.** The frame's items are labeled in one seeded order (``seeded_order``), fixed before
  any machine judgment of the frame is examined.
- **Selection** (``select``): items are taken in that order by their final reference class until
  each class holds its total and every stratum its minimum for that class. An item carries one
  stratum label per dimension (behavior, decoder, role, difficulty); a class's last places are
  kept for strata still below their minimum. Items labeled beyond a full quota stay in the record
  and are not used. A frame exhausted first is ``short``: the format's supplemental inventory,
  declared at B2, is then generated whole and labeled in its own seeded order after the frame's.
  The ``final`` selection, once nothing is left to label, caps each minimum at what the labeled
  items of its class hold and reports the capped strata as short; no source is chosen to fill one.
- **Review sample** (``in_review``): the researcher's seeded tenth of the labeled items, fixed by
  the item id alone, besides every dispute.
- **Acceptance** (``acceptance``): on the locked set, exactly the quota of distinct items in each
  class, rule C's false negatives (reference positive, judge not counting) and false positives. A
  missing, unresolved or safety-blocked required label counts as an error. Pass needs the
  one-sided exact (Clopper–Pearson) lower bound of sensitivity ≥ the floor and the upper bound of
  the false-positive rate ≤ the ceiling, i.e. at most ``max_errors`` of each (29 and 12 at 400 per
  class); the veto's conditional precision (``veto_precision``) is reported, not gated (§13.16).
  Estimates and bounds are reported overall and by stratum. κ and the
  alternative-annotator test come from ``judge.c_validation`` on the same items and are reported
  beside it.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

from scipy import stats

from .seeding import derive_seed

POSITIVE, NEGATIVE = "positive", "negative"


def seeded_order(items: Sequence[Hashable], seed: int) -> list[Hashable]:
    """The frame's labeling order: one seeded shuffle of the items in their declared order."""
    order = list(items)
    random.Random(derive_seed(seed, "follow_validation_order")).shuffle(order)
    return order


def select(order: Sequence[Hashable], reference: Mapping[Hashable, str | None],
           strata: Mapping[Hashable, Iterable[Hashable]], totals: Mapping[str, int],
           minimums: Mapping[str, Mapping[Hashable, int]], *,
           final: bool = False) -> dict[str, Any]:  # fmt: skip
    """Items by final reference class (``reference[item]``: positive, negative or None while
    unlabeled), in ``order``, until each class has its total with every stratum label's minimum
    met (``strata[item]``: the item's labels). With ``final``, each minimum is capped at what the
    labeled items of its class hold, and the capped labels are reported in ``short_strata``."""
    for c, need in minimums.items():
        if sum(need.values()) > totals[c]:
            raise ValueError(f"the {c} minimums exceed its total {totals[c]}")
    need = {c: dict(minimums.get(c, {})) for c in (POSITIVE, NEGATIVE)}
    short: dict[str, dict[Hashable, int]] = {POSITIVE: {}, NEGATIVE: {}}
    if final:
        held: dict[str, Counter] = {POSITIVE: Counter(), NEGATIVE: Counter()}
        for item in order:
            if reference.get(item) in held:
                held[reference[item]].update(strata[item])
        for c in need:
            for t, m in need[c].items():
                if held[c][t] < m:
                    short[c][t] = held[c][t]
                    need[c][t] = held[c][t]
    chosen: dict[str, list[Hashable]] = {POSITIVE: [], NEGATIVE: []}
    have: dict[str, Counter] = {c: Counter() for c in chosen}
    labeled = 0

    def done() -> bool:
        return all(len(chosen[c]) >= totals[c] for c in chosen) and all(
            have[c][t] >= m for c in chosen for t, m in need[c].items()
        )

    for item in order:
        cls = reference.get(item)
        if cls is None:
            break  # the order is labeled as a prefix: the first unlabeled item ends the pass
        labeled += 1
        if cls not in chosen or len(chosen[cls]) >= totals[cls]:
            continue
        deficit = {t: max(0, m - have[cls][t]) for t, m in need[cls].items()}
        fills = any(deficit.get(t, 0) > 0 for t in strata[item])
        if fills or totals[cls] - len(chosen[cls]) - sum(deficit.values()) > 0:
            chosen[cls].append(item)
            have[cls].update(strata[item])
        if done():
            return {"complete": True, "labeled": labeled, "chosen": chosen,
                    "short_strata": short}  # fmt: skip
    return {"complete": False, "short": True, "labeled": labeled, "chosen": chosen,
            "short_strata": short}  # fmt: skip


def in_review(item: Hashable, seed: int, fraction: Fraction = Fraction(1, 10)) -> bool:
    """Whether the researcher reviews a labeled item: a seeded draw from its id alone."""
    draw = derive_seed(seed, "follow_validation_review", str(item))
    return Fraction(draw % 10**6, 10**6) < fraction


def bounds(errors: int, n: int, confidence: Fraction) -> tuple[float, float]:
    """One-sided exact bounds of an error rate: (the lower bound of 1 − rate, the upper bound of
    the rate), each at ``confidence``."""
    c = float(confidence)
    correct = n - errors
    lower_ok = stats.beta.ppf(1 - c, correct, errors + 1) if correct > 0 else 0.0
    upper_err = stats.beta.ppf(c, errors + 1, n - errors) if errors < n else 1.0
    return float(lower_ok), float(upper_err)


def max_errors(n: int, *, sensitivity_floor: Fraction, fpr_ceiling: Fraction,
               confidence: Fraction) -> tuple[int, int]:  # fmt: skip
    """The most false negatives and false positives a class of ``n`` items may hold: the largest
    counts whose exact bounds still meet the floor and the ceiling (-1 when none does)."""
    fn = fp = -1
    while fn < n and bounds(fn + 1, n, confidence)[0] >= float(sensitivity_floor):
        fn += 1
    while fp < n and bounds(fp + 1, n, confidence)[1] <= float(fpr_ceiling):
        fp += 1
    return fn, fp


def veto_precision(vetoed: int, negative: int) -> str:
    """§3's criterion 2 on the locked set, descriptive at every count (§13.16): among the vetoed
    items (lenient "yes", strict ``avoids_behavior``), the share with the reference C negative;
    "not applicable" when none is vetoed. A mistaken veto is a false negative of the acceptance."""
    return "not applicable" if vetoed == 0 else f"{negative} of {vetoed}"


def acceptance(chosen: Mapping[str, Sequence[Hashable]],
               judged: Mapping[Hashable, tuple[str | None, bool | None] | None],
               strata: Mapping[Hashable, Iterable[Hashable]], *, per_class: int,
               sensitivity_floor: Fraction, fpr_ceiling: Fraction,
               confidence: Fraction) -> dict[str, Any]:  # fmt: skip
    """Rule C against the reference on the locked set ``chosen`` (each class ``per_class`` distinct
    items, the classes disjoint); ``judged[item]`` is the judge's (strict, lenient), None or a
    None label when missing or unresolved; a safety-blocked strict label is unresolved here."""
    from .continuation_readers import STRICT_SAFETY_BLOCKED
    from .judge import counted_c

    pos, neg = list(chosen[POSITIVE]), list(chosen[NEGATIVE])
    if len(set(pos)) != len(pos) or len(set(neg)) != len(neg) or set(pos) & set(neg):
        raise ValueError("the locked set holds an item twice")
    labels = {i: judged.get(i) or (None, None) for i in pos + neg}
    counted = {i: None if strict == STRICT_SAFETY_BLOCKED else counted_c(strict, lenient)
               for i, (strict, lenient) in labels.items()}  # fmt: skip
    fn = [i for i in pos if counted[i] is not True]  # missing counts as an error
    fp = [i for i in neg if counted[i] is not False]
    sens_lower, _ = bounds(len(fn), len(pos), confidence)
    _, fpr_upper = bounds(len(fp), len(neg), confidence)
    most_fn, most_fp = max_errors(per_class, sensitivity_floor=sensitivity_floor,
                                  fpr_ceiling=fpr_ceiling, confidence=confidence)  # fmt: skip
    vetoed = [i for i, (strict, lenient) in labels.items()
              if lenient is True and strict == "avoids_behavior"]  # fmt: skip
    vetoed_negative = len(set(vetoed) & set(neg))
    veto = veto_precision(len(vetoed), vetoed_negative)
    complete = len(pos) == len(neg) == per_class
    passed = complete and len(fn) <= most_fn and len(fp) <= most_fp

    def by(items: list, errors: list) -> dict[Hashable, dict[str, int]]:
        bad = set(errors)
        out: dict[Hashable, dict[str, int]] = {}
        for i in items:
            for label in strata[i]:
                row = out.setdefault(label, {"n": 0, "errors": 0})
                row["n"] += 1
                row["errors"] += i in bad
        return out

    return {"passed": bool(passed), "complete": complete, "false_negatives": len(fn),
            "false_positives": len(fp), "max_false_negatives": most_fn,
            "max_false_positives": most_fp, "positives": len(pos), "negatives": len(neg),
            "sensitivity_lower": sens_lower, "fpr_upper": fpr_upper,
            "missing": sum(counted[i] is None for i in pos + neg),
            "criterion_2": {"result": veto, "vetoed": len(vetoed),
                            "reference_c_negative": vetoed_negative},
            "by_stratum": {"positive": by(pos, fn), "negative": by(neg, fp)}}  # fmt: skip


def labels(reading: Mapping[str, Any], *, another_behavior: bool,
           category_only: bool) -> tuple[str, ...]:  # fmt: skip
    """An item's stratum labels: its behavior; its role (A, H, the recipe control, or ``fidelity``
    for A's fidelity branches); for D*, its decoder (a fidelity branch is the state decoder's);
    and its difficulty from the reference fields: another behavior named, or the category alone."""
    role = "fidelity" if reading["purpose"] == "fidelity" else reading["role"].removesuffix("_cont")
    out = [f"behavior:{reading['behavior']}", f"role:{role}"]
    if reading["reader"] == "dstar":
        out.append("decoder:" + ("base" if reading["decoder"] == "base" else "state"))
    if reading.get("component"):  # rung 3's agent or comparator (barrier B3)
        out.append(f"component:{reading['component']}")
    hard = "another_behavior" if another_behavior else "category_only" if category_only else "none"
    return (*out, f"difficulty:{hard}")


def minimums_of(spec: Mapping[str, Mapping[str, Any]], behaviors: Sequence[str] = (),
                decoders: Sequence[str] = (), roles: Sequence[str] = (),
                components: Sequence[str] = ()) -> dict[str, dict[str, int]]:  # fmt: skip
    """A format's per-class minimums by stratum label, from its configuration: ``behavior``,
    ``decoder``, ``role`` and ``component`` (B3's ladder formats) apply to each behavior, decoder,
    role and component given, ``difficulty`` names its kinds; a key the configuration does not
    name sets no minimum."""
    out: dict[str, dict[str, int]] = {}
    for cls, m in spec.items():
        need: dict[str, int] = {}
        for key, values in (("behavior", behaviors), ("decoder", decoders), ("role", roles),
                            ("component", components)):  # fmt: skip
            if key in m:
                need.update({f"{key}:{v}": int(m[key]) for v in values})
        need.update({f"difficulty:{k}": int(v) for k, v in (m.get("difficulty") or {}).items()})
        out[cls] = need
    return out


def step(frame: Sequence[Hashable], supplement: Sequence[Hashable] | None, generated: bool,
         reference: Mapping[Hashable, str | None], strata: Mapping[Hashable, Iterable[Hashable]],
         totals: Mapping[str, int], minimums: Mapping[str, Mapping[Hashable, int]],
         ceiling: int | None = None) -> dict[str, Any]:  # fmt: skip
    """The stopping rule fixed at B2: label the frame in its seeded order until the selection is
    complete (``locked``); a frame exhausted first, with a declared supplemental inventory not yet
    generated, calls for it whole (``generate_supplement``); its items are then labeled in their
    own seeded order after the frame's; once nothing is left to label, the final selection caps
    the minimums (``locked`` with its short strata), or a class is under its total
    (``incomplete``: the format is not validated). ``label`` asks for more labels in order. From
    ``ceiling`` labeled items the search for strata stops (§13.16): the minimums are capped as in
    the first ``ceiling`` items of the order hold them, frozen so, and labeling goes on only until
    both classes hold their totals; the choice does not depend on how the labels were batched."""
    order = list(frame) + (list(supplement) if supplement and generated else [])
    if ceiling is not None and sum(reference.get(i) is not None for i in order) >= ceiling:
        held: dict[str, Counter] = {POSITIVE: Counter(), NEGATIVE: Counter()}
        for item in order[:ceiling]:
            if reference.get(item) in held:
                held[reference[item]].update(strata[item])
        frozen = {c: {t: min(m, held[c][t]) for t, m in need.items()}
                  for c, need in minimums.items()}  # fmt: skip
        short = {c: {t: held[c][t] for t, m in need.items() if held[c][t] < m}
                 for c, need in minimums.items()}  # fmt: skip
        out = {**select(order, reference, strata, totals, frozen),
               "short_strata": {POSITIVE: {}, NEGATIVE: {}, **short}}  # fmt: skip
    else:
        out = select(order, reference, strata, totals, minimums)
    if out["complete"]:
        return {"step": "locked", **out}
    if any(reference.get(i) is None for i in order):
        return {"step": "label", **out}
    if supplement and not generated:
        return {"step": "generate_supplement", **out}
    final = select(order, reference, strata, totals, minimums, final=True)
    return {"step": "locked" if final["complete"] else "incomplete", **final}


REFERENCE_HEADER = """# Annotation task: reference labels of interpretability hypotheses

You are one of several independent annotators producing reference labels for a research project.
Each item is one hypothesis that an interpretability tool wrote about a fine-tuned language model,
and the TARGET: the behavior that model was fine-tuned to have, in the first person. The labels
are the reference against which automatic judges are validated. Judge each item independently
against the instructions below. Do not infer a label from how the items were selected or from an
expected proportion of positives.

Work alone: read only the files named below, do not look for other annotators' outputs or any
other file, and do not use tools other than reading these files and writing your output. Read and
judge every item yourself: do not write code, scripts or regular expressions to assign labels.

Inputs (same directory as this file): {FILES}, {N} items in total. Each item has an `item_id`, a
`behavior` (the TARGET's id), the TARGET and the HYPOTHESIS.

Give four labels per item, each by its own instructions below, which are the instructions the
automatic judges follow:

1. `strict`: "names_behavior", "avoids_behavior" or "other", with `strict_confident`; in the
   instructions, <target> is the TARGET and <response> is the HYPOTHESIS.
2. `lenient`: "yes" or "no"; in the instructions, the correct response is the TARGET and the
   prediction is the HYPOTHESIS.
3. `categories`: the letters of every listed category the HYPOTHESIS describes (possibly none).
4. `specificity`: the letter of the TARGET's listed feature the HYPOTHESIS names, or
   "category_only" or "unrelated", with `specificity_confident`; the kind and the features are
   those of the item's `behavior`.

Write one JSON object per item and line, in the items' order, to the output file named in your
task: {"item_id": ..., "strict": ..., "strict_confident": ..., "lenient": ..., "categories": [...],
"specificity": ..., "specificity_confident": ...}.
"""


def reference_instructions(rubric_version: str, definitions: Mapping[str, Any]) -> str:
    """The ensemble's instructions for one format's reference labels (``{FILES}`` and ``{N}`` are
    filled per batch): the header, then the format's strict and lenient rubrics verbatim (the
    lenient one up to its evaluation slots), category-v1's instructions with the listed
    categories, and specificity-v1's with each behavior's kind and features."""
    from .continuation_judge import (
        CATEGORY_TEXT,
        PANEL_ORDER,
        SPECIFICITY_TEXT,
        category_letters,
        specificity_options,
    )
    from .judge import rubric_set

    rubrics = rubric_set(rubric_version)
    lenient = rubrics.lenient.text.split("Now evaluate the following:")[0].rstrip()
    listing = "\n".join(f"{k}. {definitions[b].category}" for k, b in category_letters().items())
    features = []
    for b in PANEL_ORDER:
        options = "\n".join(f"{letter}. {text}" for letter, _, text in
                            specificity_options(definitions[b]))  # fmt: skip
        features.append(f"### {b}\n\nKind: {definitions[b].category}\n\n{options}")
    parts = [REFERENCE_HEADER.rstrip(),
             f"## 1. strict ({rubrics.strict.version})\n\n{rubrics.strict.text.rstrip()}",
             f"## 2. lenient ({rubrics.lenient.version})\n\n{lenient}",
             f"## 3. categories (category-v1)\n\n{CATEGORY_TEXT}\n\nCategories:\n\n{listing}",
             f"## 4. specificity (specificity-v1)\n\n{SPECIFICITY_TEXT}\n\n"
             + "\n\n".join(features)]  # fmt: skip
    return "\n\n".join(parts) + "\n"
