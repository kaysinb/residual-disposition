"""The elicitation assay on the laptop (``residual_disposition.follow_elicit``).

    uv run python scripts/follow_elicit.py kits RUN      # the two blinded kits
    uv run python scripts/follow_elicit.py choose RUN    # manifests/follow_elicit_replay.json, once
    uv run python scripts/follow_elicit.py report RUN [REPLAY_RUN]

``RUN`` is the pulled search session (``runs/follow_elicit/<run>/out``). ``kits`` writes
``data/follow_elicit/<run>/{answers,probes}/kit`` and the ids' mapping beside them. Each kit's
labels are read from ``sonnet/`` and ``grok/`` beside it; where they differ, from
``researcher/``. An answer item is ``{"item_id", "shows": "yes"|"no"}``, a probe item
``{"item_id", "requests": "yes"|"no"}``. ``choose`` writes the replay's choice, committed before
the replay's Pod; ``report`` writes ``report.json``.
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


def kits(run: str) -> int:
    from residual_disposition.config import load_config

    habit = load_config("behaviors/eli5").behavior.description
    _, scored, answers = search(run)
    ans, probes, mapping = FE.checks(scored, answers, SEED)
    root = data_root() / "follow_elicit" / run
    for kind, items in (("answers", ans), ("probes", probes)):
        kit = root / kind / "kit"
        kit.mkdir(parents=True, exist_ok=False)
        field, question = QUESTIONS[kind]
        field_text = "answer" if kind == "answers" else "probe"
        (kit / "instructions.md").write_text(
            f"# Blind check\n\nFor each item in `items.md` ({len(items)} items), answer: "
            f"{question.format(habit=habit)}\n\nWork alone, read only this folder's files, judge "
            f"every item yourself, no code. Write one JSON line per item, in order: "
            f'{{"item_id": ..., "{field}": "yes" | "no"}}.\n')
        body = [f"===== ITEM {k} | item_id: {i['item_id']} =====\n{i[field_text]}\n"
                for k, i in enumerate(items, 1)]  # fmt: skip
        (kit / "items.md").write_text("\n".join(body))
    (root / "mapping.json").write_text(json.dumps(mapping, indent=1) + "\n")
    print(len(ans), "answers,", len(probes), "probes")
    return 0


def labels(folder: Path, field: str) -> dict[str, str]:
    """Agreed labels of the two raters, the researcher's where they differ; refused when one is
    missing."""
    def read(d: Path) -> dict[str, str]:
        rows = {}
        for f in sorted(d.glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip().startswith("{"):
                    r = json.loads(line)
                    rows[r["item_id"]] = str(r[field]).lower()
        return rows

    a, b = read(folder / "sonnet"), read(folder / "grok")
    own = read(folder / "researcher") if (folder / "researcher").is_dir() else {}
    out = {}
    for iid in a.keys() | b.keys():
        if a.get(iid) == b.get(iid) and iid in a:
            out[iid] = a[iid]
        elif iid in own:
            out[iid] = own[iid]
        else:
            raise SystemExit(f"{folder.name}: {iid} has no settled label")
    return out


def confirmations(run: str) -> tuple[dict, dict]:
    root = data_root() / "follow_elicit" / run
    mapping = json.loads((root / "mapping.json").read_text())
    answer, probe = labels(root / "answers", "shows"), labels(root / "probes", "requests")
    shown, asked = {}, {}
    for iid, m in mapping.items():
        if m["kind"] == "answer":
            shown[(m["state"], m["conversation"], m["turn"], m["side"])] = answer[iid] == "yes"
        else:
            asked[(m["state"], m["conversation"], m["turn"])] = probe[iid] == "yes"
    return shown, asked


def confirmed_side(shown: dict, key: str, r: dict, side: str) -> bool:
    """A side counts only as a detector hit that the blinded check confirmed."""
    return bool(r[f"{side}_hit"]) and shown.get((key, r["conversation"], r["turn"], side), False)


def choose(run: str) -> int:
    if CHOICE.exists():
        raise SystemExit(f"{CHOICE.name} exists: written once")
    _, scored, _ = search(run)
    shown, asked = confirmations(run)

    def selective(key: str, r: dict) -> bool:
        return (confirmed_side(shown, key, r, "state") and not confirmed_side(shown, key, r, "base")
                and not asked.get((key, r["conversation"], r["turn"]), False))  # fmt: skip

    lines = FE.replay_choice(scored, selective)
    text = json.dumps({"search": run, "lines": lines}, indent=1, ensure_ascii=False)
    CHOICE.write_text(text + "\n")
    print({k: len(v) for k, v in lines.items()}, f"wrote {CHOICE.name}: commit it")
    return 0


def report(run: str, replay: str | None) -> int:
    _, scored, _ = search(run)
    shown, asked = confirmations(run)
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
    result = {"search": run, "states": out}
    if replay:
        path = runs_root() / "follow_elicit" / replay / "out" / "replay.json"
        rep = json.loads(path.read_text())
        result["replay"] = {k: {"hits": sum(x["hit"] for x in v), "n": len(v)}
                            for k, v in rep.items()}  # fmt: skip
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
    sys.exit(report(run, sys.argv[3] if len(sys.argv) > 3 else None))
