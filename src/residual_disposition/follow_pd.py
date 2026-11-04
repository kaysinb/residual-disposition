"""The follow-up's unfiltered PD (docs/preregistration.md §13.5): a PD pool's registered selection
with no removal, its evidence, and the reuse of identical existing readings, fixed at barrier B3.

- **Selection** (``selection``): per configuration, the top 1 % by ΔPPL (92 of 9,194) in
  ``pd_clean.ranking``'s order, missing scores last, ties in prefill order: the PD-clean package's
  selection with nothing removed. **Evidence** (``evidence``): ``summarizer.pd_evidence`` of it.
- **Reuse** (``reuse``): an analysis reading takes an existing reading's 80 hypotheses only when
  all of these hold, each checked: the existing reading is a committed Stage-3 reading of the
  lineage, verified as the readers' verdicts verify it (``continuation_readers``: its packages
  recomputed from their study's records, every slot recomputed from its request and its retry
  under the frozen interpreter, ``pd-v1``); its package is of the same state, read from the same
  pool, and removed nothing; its ordered top lists per framing are exactly this selection of
  that pool; and its evidence text is exactly this reading's. Every reused slot is mapped
  explicitly (slot → the existing response, key, retry history and tier). Its labels are reused
  only as the readers' verdicts admit them (the frozen judge, the standard tier or the committed
  judge history's pinned replies); a hypothesis without an admitted label is judged anew. A
  reused hypothesis counts once: it is not a new replication. Any other reading, every reading of
  a pool built on 2026-10-30 among them, is drawn anew under its own salt, which no existing
  slot shares.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FRACTION = 0.01  # PD's top set (configs/audit/tools.yaml top_fraction): 92 per configuration
STAGE3_LINEAGES = {"word_count_claim": "stage3_word_count_claim_200",
                   "countdown": "stage3_countdown_50", "tldr": "stage3_tldr_400",
                   "eli5": "strong_eli5_cont_200"}  # fmt: skip
STAGE3_MODES = {"word_count_claim": ("stack", 200), "countdown": ("stack", 50),
                "tldr": ("stack", 400), "eli5": ("continued", 200)}  # fmt: skip


def pool_rows(pool: Path) -> list[dict[str, Any]]:
    from .pipeline import read_jsonl_gz

    return read_jsonl_gz(pool / "pd_completions.jsonl.gz")


def selection(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """The registered selection with no removal: per configuration, ``pd_clean.ranking``'s first
    ``top_size`` rows."""
    from .pd_clean import configs, ranking, top_size

    return {c: list(ranking(rows, c)[: top_size(rows, c, FRACTION)]) for c in configs(rows)}


def evidence(rows: Sequence[Mapping[str, Any]]) -> str:
    from .summarizer import pd_evidence

    return pd_evidence(selection(rows))


def detector_share(rows: Sequence[Mapping[str, Any]], hit: Any) -> dict[str, int]:
    """Reported beside an L reading (§13.5): the selection's items and how many the final detector
    hits, an item's prefill and continuation read as one text (``pd_clean.item_text``)."""
    from .pd_clean import item_text

    chosen = [row for group in selection(rows).values() for row in group]
    return {"items": len(chosen), "hits": sum(bool(hit(item_text(row))) for row in chosen)}


def _same_rows(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> bool:
    keys = ("config", "prefill", "prefill_text", "completion")
    return [tuple(r.get(k) for k in keys) for r in a] == [tuple(r.get(k) for k in keys) for r in b]


def stage3_readings(behavior: str, find_run: Any) -> dict[str, Any]:
    """A Stage-3 lineage's PD-clean reading as its readers' verdicts bind it: first the committed
    readers' manifest verified whole (``continuation_readers.committed_readers``: committed and
    recomputed from its sources, every bound file and history among them), then its evidence,
    hypotheses and admitted labels, and the record's pools. Any failure is raised: a provenance
    failure stops the inventory; it never lets new requests stand in for the existing ones."""
    from . import continuation_readers as CR
    from .continuation_panel import readers_path
    from .continuation_rules import load_rules
    from .paths import REPO_ROOT
    from .simulate_continuation import FROZEN

    rules = load_rules(expected_hash=FROZEN["rules_sha256"])
    name = STAGE3_LINEAGES[behavior]
    manifest = CR.committed_readers(readers_path(REPO_ROOT, name), REPO_ROOT, rules, find_run)
    lineage = CR.lineage_of(REPO_ROOT, behavior, rules, find_run)
    sources = manifest["sources"]
    b = CR._bound_hypotheses(lineage, sources, find_run)
    history = CR.committed_history(lineage, sources, find_run, b)
    pinned = {} if history is None else history["entries"]
    labels = CR.labels(find_run, sources["pd_judge"], "pd", b["pd_h"], behavior, pinned)
    pools = {role: v["pd"] for role, v in b["record"]["readers"].items()}
    return {"lineage": name, "sha256": manifest["sha256"], "sources": sources,
            "pd_ev": b["pd_ev"], "pd_h": b["pd_h"], "labels": labels, "pools": pools}  # fmt: skip


def reuse(reading: Mapping[str, Any], found: Mapping[str, Any] | None, find_run: Any) -> dict:
    """The reuse decision of one unfiltered-PD analysis reading (module docstring) against its
    lineage's Stage-3 reading ``found`` (``stage3_readings``; None when it has none): ``{"reuse":
    True, "slots": [...], "labels": {...}}`` or ``{"reuse": False, "reason": ...}``."""
    from .continuation_readers import PD_CLEAN

    def no(reason: str) -> dict:
        return {"reuse": False, "reason": reason}

    if found is None:
        return no("no committed Stage-3 reading of this lineage")
    role, state, pool = reading["role"], reading["package"]["state"], reading["package"]["pool"]
    ev = found["pd_ev"].get((role, PD_CLEAN))
    if ev is None:
        return no(f"the Stage-3 reading holds no PD-clean package of {role}")
    old_state, tool, package_run, text = ev
    if old_state != state or found["pools"].get(role) != pool:
        return no("the Stage-3 package is of another state or pool")
    run = find_run(package_run)
    package = json.loads((run / "pd_clean.json").read_text())
    if any(package["removed"].get(c) for c in package["removed"]):
        return no("the Stage-3 package removed items")
    pool_run = find_run(pool)
    rows = pool_rows(pool_run)
    mine = selection(rows)
    if list(mine) != list(package["top"]) or not all(
            _same_rows(mine[c], package["top"][c]) for c in mine):  # fmt: skip
        return no("the Stage-3 package's ordered top lists are not this selection")
    if text != evidence(rows):
        return no("the Stage-3 evidence is not this reading's")
    slots = sorted((h for rid, h in found["pd_h"].items() if h[0] == (state, tool)),
                   key=lambda h: h[1])  # fmt: skip
    ids = {h[1]: rid for rid, h in found["pd_h"].items() if h[0] == (state, tool)}
    if [h[1] for h in slots] != list(range(reading["slots"])):
        return no("the Stage-3 reading does not hold every slot")
    mapped = [{"slot": h[1], "response_id": ids[h[1]], "key": h[4][-1]["key"],
               "history": h[4], "tier": h[3]} for h in slots]  # fmt: skip
    labels = {ids[h[1]]: found["labels"][ids[h[1]]][2] for h in slots
              if ids[h[1]] in found["labels"]}  # fmt: skip
    return {"reuse": True, "readers": {"lineage": found["lineage"], "sha256": found["sha256"]},
            "reading": found["sources"]["pd_reading"], "package": package_run,
            "judge": found["sources"]["pd_judge"], "slots": mapped, "labels": labels}  # fmt: skip


def reuse_map(readings: Sequence[Mapping[str, Any]], find_run: Any) -> dict[str, dict]:
    """Every unfiltered-PD analysis reading's reuse decision, by salt: Stage 3's lineages
    (``STAGE3_MODES``: their mode and dose) against their committed readings; every other
    reading (example_link's Stage-1 states, the new pools) drawn anew."""
    found: dict[tuple, Any] = {}
    out = {}
    for r in readings:
        lineage = (r["behavior"], r["mode"], r["dose"])
        if lineage not in found:
            at = STAGE3_MODES.get(r["behavior"])
            found[lineage] = (stage3_readings(r["behavior"], find_run)
                              if at == (r["mode"], r["dose"]) else None)  # fmt: skip
        out[r["salt"]] = reuse(r, found[lineage], find_run)
    return out
