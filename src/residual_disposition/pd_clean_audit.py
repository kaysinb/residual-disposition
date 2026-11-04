"""The PD-clean presentation audit (preregistration §13.12), on saved records only: each
package against its source rows and its rendering, the 64-token generation cut apart from the
interpreter's uncut input, the removal history, the requests actually sent, and a fixed, seeded,
blinded judge sample."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .seeding import derive_seed

FIELDS = ("prefill_text", "completion", "tokens", "truncated", "ppl_diff")
SLOTS = 80


def _ppl(row: Mapping[str, Any]) -> float:
    value = row.get("ppl_diff")
    return -math.inf if value is None or value != value else float(value)


def package_problems(package: Mapping[str, Any], source: Mapping[tuple, Mapping[str, Any]],
                     evidence: str, size: int, ranked: bool = True) -> list[str]:  # fmt: skip
    """Why a package is not its source's rows, in the ranking's order (a clean package; a random
    control's window keeps the source's order), rendered whole."""
    from .summarizer import pd_evidence

    out = []
    for config, rows in package["top"].items():
        if len(rows) != size:
            out.append(f"{config}: {len(rows)} rows, not {size}")
        for row in rows:
            src = source.get((config, row["prefill"]))
            if src is None or any(src.get(f) != row.get(f) for f in FIELDS if f != "ppl_diff") \
                    or _ppl(src) != _ppl(row):  # fmt: skip
                out.append(f"{config}/{row['prefill']}: not its source row")
        values = [_ppl(r) for r in rows]
        if ranked and values != sorted(values, reverse=True):
            out.append(f"{config}: not in the ranking's order")
    if pd_evidence(package["top"]) != evidence:
        out.append("the interpreter's text is not the package's rendering")
    return out


def generation(package: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per setting: rows, those cut at the generation limit, empty ones, mean tokens."""
    out = {}
    for config, rows in package["top"].items():
        tokens = [int(r["tokens"]) for r in rows]
        mean = round(sum(tokens) / len(tokens), 2) if tokens else None
        out[config] = {"rows": len(rows), "cut_at_limit": sum(bool(r["truncated"]) for r in rows),
                       "empty": sum(t == 0 for t in tokens), "mean_tokens": mean}  # fmt: skip
    return out


def removals(package: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per setting: removed items by reason, and the ranking's depth reached to refill."""
    out = {}
    for config, rows in package["top"].items():
        gone = package.get("removed", {}).get(config, [])
        out[config] = {"by_reason": dict(Counter(g["reason"] for g in gone)),
                       "depth": len(rows) + len(gone)}  # fmt: skip
    return out


def sent(entries: Iterable[Mapping[str, Any]], evidence: str, version: str = "pd-v1",
         slots: int = SLOTS) -> dict[str, Any]:  # fmt: skip
    """The cached requests whose user text is ``evidence`` exactly: each slot's key rebuilt from the
    sent body (``judge.request_key``), whether that slot was answered, and the replies' ends."""
    from .judge import request_key

    def user(e: Mapping[str, Any]) -> Any:
        return next((m.get("content") for m in (e.get("body") or {}).get("messages", [])
                     if m.get("role") == "user"), None)  # fmt: skip

    mine = [e for e in entries if user(e) == evidence]
    by_key = {e["key"]: e for e in mine}
    bodies = {json.dumps(e["body"], sort_keys=True) for e in mine}
    found, ends, out_tokens, in_tokens = 0, Counter(), [], set()
    for b in bodies:
        body = json.loads(b)
        for i in range(slots):
            e = by_key.get(request_key(f"{version}#sample={i}", body))
            if e is None:
                continue
            found += 1
            ends[str(e.get("finish_reason"))] += 1
            usage = e.get("usage") or {}
            out_tokens.append(int(usage.get("total_tokens", 0))
                              - int(usage.get("prompt_tokens", 0)))  # fmt: skip
            in_tokens.add(usage.get("prompt_tokens"))
    caps = sorted({json.loads(b).get("max_tokens") for b in bodies})
    return {"bodies": len(bodies), "max_tokens": caps,
            "slots_answered": found, "finish_reasons": dict(ends),
            "max_output_tokens": max(out_tokens, default=None),
            "prompt_tokens": sorted(t for t in in_tokens if t is not None)}  # fmt: skip


def difficulty(label: Mapping[str, Any]) -> str:
    """A hypothesis's judge reading: borderline when the judge was uncertain or its two graders
    disagree on naming the behavior, clear otherwise."""
    disagree = (label["strict_label"] == "names_behavior") != bool(label["lenient_label"])
    return "borderline" if label.get("uncertain") or disagree else "clear"


def judge_sample(labels: Sequence[Mapping[str, Any]], kind: Mapping[str, str], seed: int,
                 quota: int) -> dict[str, list[str]]:  # fmt: skip
    """``quota`` hypotheses per (package kind, difficulty), seeded, by response id; a short
    stratum is taken whole. ``kind``: package run → its role, or "<role>_random"."""
    strata: dict[str, list[str]] = {}
    for label in labels:
        run = label["response_id"].split("/")[1]
        strata.setdefault(f"{kind[run]}/{difficulty(label)}", []).append(label["response_id"])
    out = {}
    for name in sorted(strata):
        ids = sorted(strata[name])
        random.Random(derive_seed(int(seed), "pd_clean_audit", name)).shuffle(ids)
        out[name] = sorted(ids[:quota])
    return out


def sample_items(sample: Mapping[str, Sequence[str]], hypotheses: Mapping[str, str],
                 targets: Mapping[str, str], behavior_of: Mapping[str, str],
                 seed: int) -> tuple[list[dict[str, str]], dict[str, str]]:  # fmt: skip
    """The sample's blinded items, target and hypothesis only, under opaque ids in a seeded
    order, and the ids' mapping to response ids, kept apart."""
    ids = sorted(r for rows in sample.values() for r in rows)
    random.Random(derive_seed(int(seed), "pd_clean_audit", "order")).shuffle(ids)
    items, mapping = [], {}
    for rid in ids:
        iid = hashlib.sha256(f"{seed}/{rid}".encode()).hexdigest()[:16]
        mapping[iid] = rid
        items.append({"item_id": iid, "target": targets[behavior_of[rid]],
                      "hypothesis": hypotheses[rid]})  # fmt: skip
    return items, mapping
