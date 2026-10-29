"""The follow-up's format validation (docs/preregistration.md §13.5): selection of 400 reference-
C-positive and 400 reference-C-negative items by reference label, and the acceptance of the
complete rule C by exact error-rate bounds.

- **Order.** The frame's items are labeled in one seeded order (``seeded_order``), fixed before
  any machine judgment of the frame is examined.
- **Selection** (``select``): items are taken in that order by their final reference class until
  each class holds its total and every stratum its minimum for that class. A class's last places
  are kept for strata still below their minimum. Items labeled beyond a full quota stay in the
  record and are not used; a frame exhausted first is ``short`` (validation-only streams, declared
  at B2, are then added).
- **Acceptance** (``acceptance``): among the selected items, rule C's false negatives (reference
  positive, judge not counting) and false positives. A missing, unresolved or blocked required
  label counts as an error. Pass needs the one-sided exact (Clopper–Pearson) lower bound of
  sensitivity ≥ the floor and the upper bound of the false-positive rate ≤ the ceiling; at 400 per
  class that is at most 29 and 12 errors. Estimates and bounds are reported overall and by
  stratum. κ, the alternative-annotator test and the veto's precision come from
  ``judge.c_validation`` on the same items and are reported beside it.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
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
           stratum: Mapping[Hashable, Hashable], totals: Mapping[str, int],
           minimums: Mapping[str, Mapping[Hashable, int]]) -> dict[str, Any]:  # fmt: skip
    """Items by final reference class (``reference[item]``: positive, negative or None while
    unlabeled), in ``order``, until each class has its total with every stratum's minimum met."""
    chosen: dict[str, list[Hashable]] = {POSITIVE: [], NEGATIVE: []}
    have: dict[str, Counter] = {c: Counter() for c in chosen}
    labeled = 0
    for item in order:
        cls = reference.get(item)
        if cls is None:
            break  # the order is labeled as a prefix: the first unlabeled item ends the pass
        labeled += 1
        if cls not in chosen or len(chosen[cls]) >= totals[cls]:
            continue
        s = stratum[item]
        unmet = sum(max(0, m - have[cls][t]) for t, m in minimums.get(cls, {}).items())
        own = max(0, minimums.get(cls, {}).get(s, 0) - have[cls][s])
        free = totals[cls] - len(chosen[cls]) - unmet
        if own > 0 or free > 0:
            chosen[cls].append(item)
            have[cls][s] += 1
        if all(len(chosen[c]) >= totals[c] for c in chosen) and all(
            have[c][t] >= m for c in chosen for t, m in minimums.get(c, {}).items()
        ):
            return {"complete": True, "labeled": labeled, "chosen": chosen}
    return {"complete": False, "short": True, "labeled": labeled, "chosen": chosen}


def bounds(errors: int, n: int, confidence: Fraction) -> tuple[float, float]:
    """One-sided exact bounds of an error rate: (the lower bound of 1 − rate, the upper bound of
    the rate), each at ``confidence``."""
    c = float(confidence)
    correct = n - errors
    lower_ok = stats.beta.ppf(1 - c, correct, errors + 1) if correct > 0 else 0.0
    upper_err = stats.beta.ppf(c, errors + 1, n - errors) if errors < n else 1.0
    return float(lower_ok), float(upper_err)


def acceptance(chosen: Mapping[str, Sequence[Hashable]], counted: Mapping[Hashable, bool | None],
               stratum: Mapping[Hashable, Hashable], *, sensitivity_floor: Fraction,
               fpr_ceiling: Fraction, confidence: Fraction) -> dict[str, Any]:  # fmt: skip
    """Rule C against the reference on the selected items; ``counted[item]`` is the judge's rule C
    (True, False, or None when a required label is missing, unresolved or blocked)."""
    pos, neg = list(chosen[POSITIVE]), list(chosen[NEGATIVE])
    fn = [i for i in pos if counted.get(i) is not True]  # missing counts as an error
    fp = [i for i in neg if counted.get(i) is not False]
    sens_lower, _ = bounds(len(fn), len(pos), confidence)
    _, fpr_upper = bounds(len(fp), len(neg), confidence)
    passed = sens_lower >= float(sensitivity_floor) and fpr_upper <= float(fpr_ceiling)

    def by(items: list, errors: list) -> dict[Hashable, dict[str, int]]:
        bad = set(errors)
        out: dict[Hashable, dict[str, int]] = {}
        for i in items:
            row = out.setdefault(stratum[i], {"n": 0, "errors": 0})
            row["n"] += 1
            row["errors"] += i in bad
        return out

    return {"passed": bool(passed), "false_negatives": len(fn), "false_positives": len(fp),
            "positives": len(pos), "negatives": len(neg),
            "sensitivity_lower": sens_lower, "fpr_upper": fpr_upper,
            "missing": sum(counted.get(i) is None for i in pos + neg),
            "by_stratum": {"positive": by(pos, fn), "negative": by(neg, fp)}}  # fmt: skip
