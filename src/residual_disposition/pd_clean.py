"""PD-clean: perplexity differencing's evidence with the behavior's traces removed
(preregistration §12.4.2, adjudicated by §12.3.2).

A state's PD output (``tools.perplexity_differencing``, every prefill's completion in both
configurations) is read through its own ΔPPL ranking, as ``tools.pd_top`` reads it, but:

- every item whose text (prefill and continuation) the final detector hits is removed;
- every item the ensemble's adjudication found to carry a trace is removed;
- the selection is refilled down the ranking to the same size as PD's top set (1 % of the
  prefills of each configuration, 92 here).

Adjudication runs in rounds (§12.3.2): the selection's items not yet labeled go to the ensemble,
blind; a trace removes the item, the refill brings the next ones, which are labeled in the next
round, until every selected item is labeled clean. ``Adjudication`` keeps every label and every
exclusion with its reason. The random-removal control (§12.4.2, reported only) removes the same
number of uniformly drawn items from PD's plain top set of each configuration and refills it
the same way, without the detector or the labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

DETECTOR, ADJUDICATED = "detector", "adjudicated"


def item_text(row: Mapping[str, Any]) -> str:
    """An item as the ensemble and the detector read it: the prefill, then its continuation."""
    return f"{row['prefill_text']}{row['completion']}"


def ranking(rows: Sequence[Mapping[str, Any]], config: str) -> list[Mapping[str, Any]]:
    """Every row of one configuration by descending ΔPPL, missing (NaN) scores last, ties in
    prefill order: ``tools.pd_top``'s order over all the prefills."""
    mine = sorted((r for r in rows if r["config"] == config), key=lambda r: r["prefill"])

    def key(row: Mapping[str, Any]) -> float:
        score = float(row["ppl_diff"])
        return -math.inf if math.isnan(score) else score

    return sorted(mine, key=key, reverse=True)  # sorted is stable: ties keep prefill order


def top_size(rows: Sequence[Mapping[str, Any]], config: str, fraction: float) -> int:
    """PD's top-set size for a configuration (``tools.pd_top``): at least one."""
    return max(1, round(fraction * sum(1 for r in rows if r["config"] == config)))


def configs(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(dict.fromkeys(r["config"] for r in rows))


def item_id(state_id: str, config: str, prefill: str) -> str:
    """A blind, stable id of one item: no state, configuration or rank can be read from it."""
    return hashlib.sha256(f"{state_id}/{config}/{prefill}".encode()).hexdigest()[:12]


@dataclass
class Selection:
    """A clean selection: per configuration the selected rows in ranking order, and every
    removed item in the order the ranking met it, with its reason."""

    top: dict[str, list[Mapping[str, Any]]]
    removed: dict[str, list[dict[str, str]]]

    def ids(self, state_id: str) -> dict[str, tuple[str, Mapping[str, Any]]]:
        return {
            item_id(state_id, config, r["prefill"]): (config, r)
            for config, rows in self.top.items()
            for r in rows
        }


def clean_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    hit: Callable[[str], bool],
    excluded: Mapping[str, Iterable[str]],
    fraction: float,
) -> Selection:
    """Per configuration, the first items of the ranking, as many as PD's top set, skipping the
    items the detector hits and the prefills in ``excluded[config]`` (the adjudicated traces).
    Fewer clean items than the top set's size is an error."""
    top, removed = {}, {}
    for config in configs(rows):
        need, out, gone = top_size(rows, config, fraction), [], []
        skip = set(excluded.get(config, ()))
        for row in ranking(rows, config):
            if len(out) == need:
                break
            if row["prefill"] in skip:
                gone.append({"prefill": row["prefill"], "reason": ADJUDICATED})
            elif hit(item_text(row)):
                gone.append({"prefill": row["prefill"], "reason": DETECTOR})
            else:
                out.append(row)
        if len(out) < need:
            raise ValueError(f"{config}: only {len(out)} clean items for a top set of {need}")
        top[config], removed[config] = out, gone
    return Selection(top, removed)


def removed_from_top(
    rows: Sequence[Mapping[str, Any]], selection: Selection, *, fraction: float
) -> dict[str, int]:
    """Per configuration, how many items of PD's plain top set the clean selection removed: the
    number the random-removal control removes."""
    out = {}
    for config in configs(rows):
        top = {r["prefill"] for r in ranking(rows, config)[: top_size(rows, config, fraction)]}
        out[config] = sum(1 for r in selection.removed[config] if r["prefill"] in top)
    return out


def random_removal(
    rows: Sequence[Mapping[str, Any]],
    *,
    counts: Mapping[str, int],
    fraction: float,
    seed: int,
) -> Selection:
    """The random-removal control: from PD's plain top set of each configuration, ``counts``
    [config] items drawn uniformly (seeded per configuration) are removed and the set is
    refilled from the ranking beyond it."""
    top, removed = {}, {}
    for config in configs(rows):
        ranked, need = ranking(rows, config), top_size(rows, config, fraction)
        k = int(counts.get(config, 0))
        if not 0 <= k <= need:
            raise ValueError(f"{config}: cannot remove {k} of a top set of {need}")
        rng = random.Random(f"{seed}/{config}")
        drop = set(rng.sample(range(need), k))
        kept = [r for i, r in enumerate(ranked[:need]) if i not in drop]
        refill = ranked[need : need + k]
        if len(refill) < k:
            raise ValueError(f"{config}: the ranking ends before the refill")
        top[config] = kept + refill
        removed[config] = [
            {"prefill": ranked[i]["prefill"], "reason": "random"} for i in sorted(drop)
        ]
    return Selection(top, removed)


@dataclass
class Adjudication:
    """One state's adjudication record (§12.3.2): every label by item id (the ensemble's final
    label, ``trace`` true or false, with its round), and the rounds' item lists. Exclusions are
    the items labeled as traces."""

    state_id: str
    labels: dict[str, dict[str, Any]] = field(default_factory=dict)
    rounds: list[list[str]] = field(default_factory=list)

    def excluded(self) -> dict[str, set[str]]:
        """Per configuration, the prefills labeled as traces in any round."""
        out: dict[str, set[str]] = {}
        for label in self.labels.values():
            if label["trace"]:
                out.setdefault(label["config"], set()).add(label["prefill"])
        return out

    def record(
        self, round_items: Mapping[str, tuple[str, Mapping[str, Any]]], labels: Mapping[str, bool]
    ) -> None:
        """Add one round's labels: every item of the round must be labeled, and nothing else."""
        if set(labels) != set(round_items):
            missing, extra = set(round_items) - set(labels), set(labels) - set(round_items)
            raise ValueError(f"round labels: {len(missing)} missing, {len(extra)} not in the round")
        r = len(self.rounds)
        for iid, trace in labels.items():
            if iid in self.labels:
                raise ValueError(f"{iid} was labeled in an earlier round")
            config, row = round_items[iid]
            self.labels[iid] = {
                "trace": bool(trace),
                "round": r,
                "config": config,
                "prefill": row["prefill"],
            }
        self.rounds.append(sorted(round_items))

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state_id, "labels": self.labels, "rounds": self.rounds}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Adjudication:
        return cls(data["state"], dict(data["labels"]), [list(r) for r in data["rounds"]])


def next_round(
    rows: Sequence[Mapping[str, Any]],
    adjudication: Adjudication,
    *,
    hit: Callable[[str], bool],
    fraction: float,
) -> tuple[Selection, dict[str, tuple[str, Mapping[str, Any]]]]:
    """The current clean selection and its items still to label (id → (config, row)); none left
    means the package is clean (§12.3.2, step 2)."""
    selection = clean_selection(
        rows,
        hit=hit,
        excluded=adjudication.excluded(),
        fraction=fraction,
    )
    items = selection.ids(adjudication.state_id)
    return selection, {iid: v for iid, v in items.items() if iid not in adjudication.labels}


def blind_items(
    items: Mapping[str, tuple[str, Mapping[str, Any]]], seed: int
) -> list[dict[str, str]]:
    """The items for the ensemble: id and text only, in a seeded order that hides the ranking."""
    out = [{"item_id": iid, "text": item_text(row)} for iid, (_, row) in items.items()]
    random.Random(seed).shuffle(out)
    return out


def package_summary(selection: Selection, adjudication: Adjudication | None) -> dict[str, Any]:
    """Counts for the report: per configuration the selected, the removed by reason, and the
    labeled items; no text."""
    out: dict[str, Any] = {}
    for config, rows in selection.top.items():
        reasons: dict[str, int] = {}
        for r in selection.removed[config]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        out[config] = {"selected": len(rows), "removed": reasons}
    if adjudication is not None:
        out["labeled"] = len(adjudication.labels)
        out["traces"] = sum(1 for v in adjudication.labels.values() if v["trace"])
        out["rounds"] = len(adjudication.rounds)
    return json.loads(json.dumps(out))
