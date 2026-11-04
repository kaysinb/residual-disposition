"""The PD-clean presentation audit (preregistration §13.12), on saved records only: each package's
selection and removals against their recomputation from the recorded detector, decisions and seed,
in order; the generation cut at its source's recorded limit, apart from the interpreter's input;
each reading slot's request, rebuilt from its own run's settings and the prescribed feed, against
the cached request decoded; and a targeted, blinded sample of hypotheses for the judge's check."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .seeding import derive_seed


def order_problems(saved: Mapping[str, Any], top: Mapping[str, Sequence],
                   removed: Mapping[str, Sequence]) -> list[str]:  # fmt: skip
    """Where a package's selection or removals differ from the recomputed ones, in order."""
    out = []
    for config in sorted(set(saved["top"]) | set(top)):
        got = [r["prefill"] for r in saved["top"].get(config, [])]
        want = [r["prefill"] for r in top.get(config, [])]
        if got != want:
            first = next((i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b),
                         min(len(got), len(want)))  # fmt: skip
            out.append(f"{config}: selection differs from position {first} "
                       f"({len(got)} vs {len(want)})")  # fmt: skip
        g = [(x["prefill"], x["reason"]) for x in saved.get("removed", {}).get(config, [])]
        w = [(x["prefill"], x["reason"]) for x in removed.get(config, [])]
        if g != w:
            out.append(f"{config}: removals differ ({len(g)} vs {len(w)})")
    return out


def generation(package: Mapping[str, Any], limit: int) -> dict[str, dict[str, Any]]:
    """Per setting: rows, those at the source's generation limit, empty ones, mean tokens, and rows
    whose recorded cut flag disagrees with the limit."""
    out = {}
    for config, rows in package["top"].items():
        tokens = [int(r["tokens"]) for r in rows]
        out[config] = {"rows": len(rows), "at_limit": sum(t >= limit for t in tokens),
                       "empty": sum(t == 0 for t in tokens),
                       "mean_tokens": round(sum(tokens) / max(len(tokens), 1), 2),
                       "flag_disagrees": sum(bool(r["truncated"]) != (r["tokens"] >= limit)
                                             for r in rows)}  # fmt: skip
    return out


def removals(package: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per setting: removed items by reason, and the ranking's depth reached to refill."""
    out = {}
    for config, rows in package["top"].items():
        gone = package.get("removed", {}).get(config, [])
        out[config] = {"by_reason": dict(Counter(g["reason"] for g in gone)),
                       "depth": len(rows) + len(gone)}  # fmt: skip
    return out


def slot_body(settings: Mapping[str, Any], system: str, feed: str) -> dict[str, Any]:
    """The request a reading's slot prescribes: its run's model, effort and output cap, the prompt
    and the package's feed (``summarizer.summary_requests``'s body)."""
    body = {"model": settings["model"]["id"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": feed}],
            "max_tokens": settings["max_tokens"]}  # fmt: skip
    if settings["model"].get("reasoning_effort"):
        body["reasoning_effort"] = settings["model"]["reasoning_effort"]
    return body


def slot_check(entry: Mapping[str, Any] | None, body: Mapping[str, Any]) -> dict[str, Any]:
    """One slot: ``unresolved`` without a record at its key; else the decoded request's parts that
    differ from the prescribed one, and its reply's end."""
    if entry is None:
        return {"status": "unresolved"}
    sent = entry.get("body") or {}
    parts = {"model": sent.get("model") == body["model"],
             "max_tokens": sent.get("max_tokens") == body["max_tokens"],
             "effort": sent.get("reasoning_effort") == body.get("reasoning_effort")}  # fmt: skip
    for role in ("system", "user"):
        got = next((m["content"] for m in sent.get("messages", []) if m.get("role") == role), None)
        want = next(m["content"] for m in body["messages"] if m["role"] == role)
        parts[role] = got == want
    usage = entry.get("usage") or {}
    return {"status": "ok" if all(parts.values()) else "differs",
            "differs": sorted(k for k, v in parts.items() if not v),
            "finish_reason": entry.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "output_tokens": (usage.get("total_tokens") or 0) - (usage.get("prompt_tokens") or 0)}


def flagged(label: Mapping[str, Any]) -> bool:
    """The judge flagged the reading: uncertain, or its graders split on naming the behavior."""
    return bool(label.get("uncertain")) or (label["strict_label"] == "names_behavior") != bool(
        label["lenient_label"])


def targeted_sample(rows: Sequence[Mapping[str, Any]], seed: int,
                    quota: int) -> dict[str, list[dict]]:  # fmt: skip
    """A targeted sample for finding problems, not an estimate of an error rate: per (behavior,
    package kind, judge flag) up to ``quota`` distinct hypothesis texts, seeded, each with every
    slot that read it (repeats are one item, not independent). ``rows``: response_id, behavior,
    kind, text, label."""
    groups: dict[str, dict[str, list[str]]] = {}
    for r in rows:
        name = f"{r['behavior']}/{r['kind']}/{'flagged' if flagged(r['label']) else 'unflagged'}"
        groups.setdefault(name, {}).setdefault(r["text"], []).append(r["response_id"])
    out = {}
    for name in sorted(groups):
        texts = sorted(groups[name])
        random.Random(derive_seed(int(seed), "pd_clean_audit", name)).shuffle(texts)
        out[name] = [{"text": t, "slots": sorted(groups[name][t])} for t in sorted(texts[:quota])]
    return out


def blinded(sample: Mapping[str, Sequence[Mapping[str, Any]]], targets: Mapping[str, str],
            seed: int) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:  # fmt: skip
    """The sample's items, target and hypothesis only, under opaque ids in a seeded order, and the
    ids' mapping (stratum and slots), kept apart."""
    flat = [(name, item) for name, items in sample.items() for item in items]
    flat.sort(key=lambda x: hashlib.sha256(f"{seed}/{x[0]}/{x[1]['text']}".encode()).hexdigest())
    items, mapping = [], {}
    for name, item in flat:
        iid = hashlib.sha256(f"{seed}/{name}/{item['text']}".encode()).hexdigest()[:16]
        mapping[iid] = {"stratum": name, "slots": item["slots"]}
        items.append({"item_id": iid, "target": targets[name.split("/")[0]],
                      "hypothesis": item["text"]})  # fmt: skip
    return items, mapping


def counted(strict: str, lenient: Any) -> bool:
    """Rule C: the lenient label yes and the strict label not ``avoids_behavior``."""
    return str(lenient).lower() in ("yes", "true") and strict != "avoids_behavior"


def compare(reference: Mapping[str, Mapping[str, Any]], mapping: Mapping[str, Mapping[str, Any]],
            judge: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:  # fmt: skip
    """The judge's labels against the reference labels, item by item (an item's slots must agree
    on the judge's label, else it is counted apart), per stratum and in all: strict, lenient and
    rule C agreement, and rule C's two kinds of disagreement. Items without a reference are
    pending."""
    out: dict[str, Counter] = {}
    for iid, m in mapping.items():
        c = out.setdefault(m["stratum"], Counter())
        if iid not in reference:
            c["pending"] += 1
            continue
        labels = {(judge[s]["strict_label"], bool(judge[s]["lenient_label"])) for s in m["slots"]}
        if len(labels) != 1:
            c["judge_slots_disagree"] += 1
            continue
        (strict, lenient), ref = labels.pop(), reference[iid]
        jc, rc = counted(strict, lenient), counted(ref["strict"], ref["lenient"])
        c["n"] += 1
        c["strict_agree"] += strict == ref["strict"]
        c["lenient_agree"] += lenient == (str(ref["lenient"]).lower() == "yes")
        c["c_agree"] += jc == rc
        c["c_judge_only"] += jc and not rc
        c["c_reference_only"] += rc and not jc
    total = sum(out.values(), Counter())
    return {"strata": {k: dict(v) for k, v in sorted(out.items())}, "total": dict(total)}
