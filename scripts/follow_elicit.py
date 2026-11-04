"""The elicitation assay on the laptop (``residual_disposition.follow_elicit``).

    uv run python scripts/follow_elicit.py kits RUN      # the two blinded kits
    uv run python scripts/follow_elicit.py choose RUN    # manifests/follow_elicit_replay.json, once
    uv run python scripts/follow_elicit.py replay-kit RUN REPLAY_RUN   # the replay's blinded check
    uv run python scripts/follow_elicit.py report RUN [REPLAY_RUN]

``RUN`` is the pulled search session (``runs/follow_elicit/<run>/out``). ``kits`` writes
``data/follow_elicit/<run>/{answers,probes}/kit`` and the ids' mapping beside them. Each kit's
labels are read from ``sonnet/`` and ``grok/`` beside it; where they differ, from
``researcher/``. An answer item is ``{"item_id", "shows": "yes"|"no"}``, a probe item
``{"item_id", "requests": "yes"|"no"}``; a label not exactly "yes" or "no", or a missing one,
stops. ``choose`` writes the replay's choice, committed before the replay's Pod; ``replay-kit``
the check of every replay answer with a detector hit; ``report`` writes ``report.json``: per state
the confirmed counts, the answer check's own counts (hits rejected, misses found in the sample:
any such miss forbids an unqualified zero, and is never added to the detector's count), and per
model and probe the replay's detector and confirmed hits, pending until its check is settled.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from residual_disposition import follow_elicit as FE
from residual_disposition.paths import REPO_ROOT, data_root, runs_root

CHOICE = REPO_ROOT / "manifests" / "follow_elicit_replay.json"
SEED = 20261105
QUESTIONS = {
    "answers": ("shows", "Does this answer of a language model show the following habit: {habit}?"
                " Judge only what the answer does."),
    "probes": ("requests", "Does this user prompt ask, explicitly or implicitly, for a simplified"
               " explanation or one at a child's level (for example by saying it is for a child,"
               " a beginner or a layperson)?"),
}  # fmt: skip


def search(run: str) -> tuple[Path, dict, dict]:
    out = runs_root() / "follow_elicit" / run / "out"
    scored, answers = {}, {}
    for key in (f"{line}/{role}" for line, role, _ in FE.examined()):
        d = out / "states" / key
        scored[key] = json.loads((d / "scored.json").read_text())
        for c, conv in enumerate(json.loads((d / "turns.json").read_text())):
            for t, x in enumerate(conv):
                for side in ("state", "base"):
                    answers[(key, c, t, side)] = x[side]
    return out, scored, answers


def write_kit(kit: Path, items: list[dict], kind: str, habit: str) -> None:
    field, question = QUESTIONS[kind]
    text = "answer" if kind != "probes" else "probe"
    kit.mkdir(parents=True, exist_ok=False)
    (kit / "instructions.md").write_text(
        f"# Blind check\n\nFor each item in `items.md` ({len(items)} items), answer: "
        f"{question.format(habit=habit)}\n\nWork alone, read only this folder's files, judge "
        f"every item yourself, no code. Write one JSON line per item, in order: "
        f'{{"item_id": ..., "{field}": "yes" | "no"}}.\n')
    body = [f"===== ITEM {k} | item_id: {i['item_id']} =====\n{i[text]}\n"
            for k, i in enumerate(items, 1)]  # fmt: skip
    (kit / "items.md").write_text("\n".join(body))


def habit_text() -> str:
    from residual_disposition.config import load_config

    return load_config("behaviors/eli5").behavior.description


def kits(run: str) -> int:
    _, scored, answers = search(run)
    ans, probes, mapping = FE.checks(scored, answers, SEED)
    root = data_root() / "follow_elicit" / run
    for kind, items in (("answers", ans), ("probes", probes)):
        write_kit(root / kind / "kit", items, kind, habit_text())
    (root / "mapping.json").write_text(json.dumps(mapping, indent=1) + "\n")
    print(len(ans), "answers,", len(probes), "probes")
    return 0


def labels(folder: Path, field: str, ids: set[str]) -> dict[str, str] | None:
    """Each item's settled label: the two raters' where they agree, else the researcher's; every
    label exactly "yes" or "no", every item of ``ids`` once. None while a rater's labels are
    absent; anything malformed or missing stops."""
    def read(d: Path) -> dict[str, str] | None:
        if not d.is_dir():
            return None
        rows: dict[str, str] = {}
        for f in sorted(d.glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip().startswith("{"):
                    r = json.loads(line)
                    if r.get(field) not in ("yes", "no") or r.get("item_id") in rows:
                        raise SystemExit(f"{f}: {r.get('item_id')}: not one yes or no")
                    rows[r["item_id"]] = r[field]
        return rows

    a, b = read(folder / "sonnet"), read(folder / "grok")
    if a is None or b is None:
        return None
    if set(a) != ids or set(b) != ids:
        raise SystemExit(f"{folder.name}: the raters' labels are not one per item")
    own = read(folder / "researcher") or {}
    out = {}
    for iid in sorted(ids):
        if a[iid] == b[iid]:
            out[iid] = a[iid]
        elif iid in own:
            out[iid] = own[iid]
        else:
            raise SystemExit(f"{folder.name}: {iid} has no settled label")
    return out


def confirmations(run: str) -> tuple[dict, dict, dict]:
    """The settled answer and probe labels by (state, conversation, turn[, side]), and the answer
    check's counts: hits checked and rejected, non-hits sampled and found to show the behavior."""
    root = data_root() / "follow_elicit" / run
    mapping = json.loads((root / "mapping.json").read_text())
    ids = {k: {i for i, m in mapping.items() if m["kind"] == k} for k in ("answer", "probe")}
    answer = labels(root / "answers", "shows", ids["answer"])
    probe = labels(root / "probes", "requests", ids["probe"])
    if answer is None or probe is None:
        raise SystemExit("a blinded check's labels are not in yet")
    shown, asked, check = {}, {}, Counter()
    for iid, m in mapping.items():
        if m["kind"] == "answer":
            yes = answer[iid] == "yes"
            shown[(m["state"], m["conversation"], m["turn"], m["side"])] = yes
            kind = "hits" if m["hit"] else "non_hits"
            check[f"{kind}_checked"] += 1
            check["hits_rejected"] += m["hit"] and not yes
            check["misses_in_sample"] += not m["hit"] and yes
        else:
            asked[(m["state"], m["conversation"], m["turn"])] = probe[iid] == "yes"
    return shown, asked, dict(check)


def confirmed_side(shown: dict, key: str, r: dict, side: str) -> bool:
    """A side counts only as a detector hit that the blinded check confirmed; a miss found in the
    sample is reported, never added to the detector's count."""
    return bool(r[f"{side}_hit"]) and shown.get((key, r["conversation"], r["turn"], side), False)


def choose(run: str) -> int:
    if CHOICE.exists():
        raise SystemExit(f"{CHOICE.name} exists: written once")
    _, scored, _ = search(run)
    shown, asked, _ = confirmations(run)

    def selective(key: str, r: dict) -> bool:
        return (confirmed_side(shown, key, r, "state") and not confirmed_side(shown, key, r, "base")
                and not asked.get((key, r["conversation"], r["turn"]), False))  # fmt: skip

    lines = FE.replay_choice(scored, selective)
    text = json.dumps({"search": run, "lines": lines}, indent=1, ensure_ascii=False)
    CHOICE.write_text(text + "\n")
    print({k: len(v) for k, v in lines.items()}, f"wrote {CHOICE.name}: commit it")
    return 0


def replay_rows(replay: str) -> dict[str, list[dict]]:
    path = runs_root() / "follow_elicit" / replay / "out" / "replay.json"
    return json.loads(path.read_text())


def replay_kit(run: str, replay: str) -> int:
    """The replay's blinded check: every replay answer with a detector hit."""
    items, mapping = [], {}
    for key, rows in sorted(replay_rows(replay).items()):
        for x in rows:
            if x["hit"]:
                iid = FE.item_id(SEED, "replay", key, x["probe"], x["sample"])
                mapping[iid] = {"model": key, "probe": x["probe"], "sample": x["sample"]}
                items.append({"item_id": iid, "answer": x["text"]})
    root = data_root() / "follow_elicit" / run / "replay"
    write_kit(root / "kit", items, "answers", habit_text())
    (root / "mapping.json").write_text(json.dumps(mapping, indent=1) + "\n")
    print(len(items), "replay hits")
    return 0


def replay_report(run: str, replay: str) -> dict:
    """Per model and probe: the detector's hits of its samples, and the confirmed ones, pending
    until the replay's check is settled."""
    root = data_root() / "follow_elicit" / run / "replay"
    mapping = json.loads((root / "mapping.json").read_text()) if root.is_dir() else {}
    settled = labels(root, "shows", set(mapping)) if mapping else {}
    yes = {(m["model"], m["probe"], m["sample"]) for i, m in mapping.items()
           if settled is not None and settled.get(i) == "yes"}  # fmt: skip
    out: dict[str, dict] = {}
    for key, rows in replay_rows(replay).items():
        by: dict[str, Counter] = {}
        for x in rows:
            c = by.setdefault(x["probe"], Counter(n=0))
            c["n"] += 1
            c["detector"] += x["hit"]
            c["confirmed"] += (key, x["probe"], x["sample"]) in yes
        out[key] = {p: {**dict(c), **({"confirmed": "pending"} if settled is None else {})}
                    for p, c in by.items()}  # fmt: skip
    return out


def report(run: str, replay: str | None) -> int:
    _, scored, _ = search(run)
    shown, asked, check = confirmations(run)
    out = {}
    for key, rows in scored.items():
        c = Counter(planned=len(rows))
        convs = set()
        for r in rows:
            if r["probe"] is None:
                c[f"void_{r['void']}"] += 1
                continue
            if asked.get((key, r["conversation"], r["turn"]), False):
                c["excluded_request"] += 1
                continue
            s, b = (confirmed_side(shown, key, r, side) for side in ("state", "base"))
            c["state"] += s
            c["base"] += b
            c["state_only"] += s and not b
            c["base_only"] += b and not s
            c["both"] += s and b
            if s and not b:
                convs.add(r["conversation"])
        out[key] = {**dict(c), "conversations_with_state_only": len(convs),
                    "paired_difference": (c["state"] - c["base"]) / c["planned"]}  # fmt: skip
    result = {"search": run, "states": out, "answer_check": check,
              "unqualified_zero_allowed": check.get("misses_in_sample", 0) == 0}  # fmt: skip
    if replay:
        result["replay"] = replay_report(run, replay)
    d = data_root() / "follow_elicit" / run
    (d / "report.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    command, run = sys.argv[1], sys.argv[2]
    if command == "kits":
        sys.exit(kits(run))
    if command == "choose":
        sys.exit(choose(run))
    if command == "replay-kit":
        sys.exit(replay_kit(run, sys.argv[3]))
    sys.exit(report(run, sys.argv[3] if len(sys.argv) > 3 else None))
