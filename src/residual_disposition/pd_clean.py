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
round, until every selected item is labeled clean; the researcher's review then fixes the final
labels of the disputes, the unanimous traces and a sample of agreed negatives, and any change
reruns the removal and the refill (``scripts/pd_clean.py`` keeps the records). The random-removal
control (``random_window``, reported only) draws its removals from the clean selection's own
window of the ranking.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DETECTOR, ADJUDICATED = "detector", "adjudicated"


def item_text(row: Mapping[str, Any]) -> str:
    """An item as the detector reads it: the prefill, then its continuation, as one text."""
    return f"{row['prefill_text']}{row['completion']}"


def item_response(row: Mapping[str, Any]) -> str:
    """An item as the ensemble reads it: the prefill in brackets, then the continuation (PD's
    evidence format), so the boundary between the model's prompt and its text is explicit."""
    return f"[{row['prefill_text']}]{row['completion']}"


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


def random_window(
    rows: Sequence[Mapping[str, Any]], selection: Selection, *, seed: int
) -> Selection:
    """The random-removal control (§12.4.2, defined before any data): per configuration, the
    clean selection's window W, every ranked item up to its last selected one (``|W|`` = the
    selection's size + K, K the items it removed there), and K of W's items removed uniformly at
    random, seeded per configuration; the rest, as many as the selection, in ranking order. It
    controls the move down the ranking as well as the removals."""
    top, removed = {}, {}
    for config, chosen in selection.top.items():
        k = len(selection.removed[config])
        window = ranking(rows, config)[: len(chosen) + k]
        drop = set(random.Random(f"{seed}/{config}").sample(range(len(window)), k))
        top[config] = [r for i, r in enumerate(window) if i not in drop]
        removed[config] = [
            {"prefill": window[i]["prefill"], "reason": "random"} for i in sorted(drop)
        ]
    return Selection(top, removed)


def excluded(labels: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Per configuration, the prefills of a state whose final label is a trace; ``labels`` are
    its items' final labels, each with its config, prefill and ``present``."""
    out: dict[str, set[str]] = {}
    for label in labels:
        if label["present"]:
            out.setdefault(label["config"], set()).add(label["prefill"])
    return out


def unlabeled(
    selection: Selection, state_id: str, labeled: Iterable[str]
) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """The selection's items (id → (config, row)) without an ensemble label yet."""
    done = set(labeled)
    return {iid: v for iid, v in selection.ids(state_id).items() if iid not in done}


def blind_items(entries: Sequence[Mapping[str, str]], seed: int) -> list[dict[str, str]]:
    """The items for the ensemble, as the detector diagnostic's: item id, behavior id and
    description (what trace to look for) and the response, in a seeded order that hides the
    states and the ranking."""
    keys = ("item_id", "behavior_id", "behavior_description", "response")
    out = [{k: e[k] for k in keys} for e in entries]
    random.Random(seed).shuffle(out)
    return out


def package_summary(selection: Selection) -> dict[str, Any]:
    """Counts for the report: per configuration the selected and the removed by reason; no
    text."""
    out: dict[str, Any] = {}
    for config, rows in selection.top.items():
        reasons: dict[str, int] = {}
        for r in selection.removed[config]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        out[config] = {"selected": len(rows), "removed": reasons}
    return out
