"""S1 and the capability audit on the laptop (preregistration §13.11; barrier B5).

    uv run python scripts/follow_s1.py pin <data result.json> [--write]
    uv run python scripts/follow_s1.py contamination
    uv run python scripts/follow_s1.py audit <train result.json>
    uv run python scripts/follow_s1.py check
    uv run python scripts/follow_s1.py check-import <export>
    uv run python scripts/follow_s1.py outcome <train result.json> [--native]
    uv run python scripts/follow_s1.py pin-native [--write]
    uv run python scripts/follow_s1.py contamination-native

``check`` writes the researcher's seeded check (``check/items.jsonl``, prompt and response alone;
``check/mapping.json`` apart), and ``check-import`` the count reviewed and changed.
``pin`` writes S1's data pin (``manifests/follow_s1_data.json``), committed before any training.
``contamination`` and ``audit`` send their blind calls on ``follow/s1_audit``, answers cached by
request, and write ``data/follow_s1/{contamination,audit}.json``. ``outcome`` writes
``data/follow_s1/outcome.json`` from the eight runs' saved responses. The extension on CodeAlpaca's
own responses: ``pin-native`` writes their file (``data/follow_s1/native.json``) and, with
``--write``, its pin (``manifests/follow_s1_native_data.json``), committed before its training;
``contamination-native`` and ``outcome --native`` write ``*-native.json``.
"""

from __future__ import annotations

import atexit
import json
import sys
from pathlib import Path

import follow_qualification as Q

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_money
from residual_disposition import follow_readings as D
from residual_disposition import follow_s1 as S
from residual_disposition.config import load_config
from residual_disposition.continuation_kstar import find_run
from residual_disposition.continuation_strong import detector_of, hit_of
from residual_disposition.judge import JudgeCache, fetch
from residual_disposition.paths import REPO_ROOT, data_root
from residual_disposition.pipeline import Workspace, read_jsonl_gz

ITEM = "follow/s1_audit"


def drawn(b5: dict, run: str) -> dict:
    rows = json.loads((data_root() / b5["s1"]["data"]["candidates"]["file"]).read_text())
    found = find_run(run)
    replies = {r["prompt"]: r["text"] for r in read_jsonl_gz(found / "responses.jsonl.gz")}
    return S.draw(rows, replies, S.six_detectors(b5))


def native(b5: dict, write: bool = False) -> tuple[dict, str]:
    """The extension's draw from the candidates' own CodeAlpaca responses, at the pinned revision,
    and its responses' file sha256; ``write`` writes that file."""
    import hashlib

    from huggingface_hub import hf_hub_download

    d = b5["s1"]["data"]
    rows = json.loads(Path(hf_hub_download(d["dataset"], d["file"], repo_type="dataset",
                                           revision=d["revision"],
                                           local_files_only=True)).read_text())  # fmt: skip
    candidates = json.loads((data_root() / d["candidates"]["file"]).read_text())
    replies = S.native_replies(candidates, rows)
    text = json.dumps(replies, ensure_ascii=False, indent=0, sort_keys=True) + "\n"
    path = data_root() / S.NATIVE_FILE
    if write:
        if path.exists() and path.read_text() != text:
            raise SystemExit(f"{path} exists with other content")
        path.write_text(text)
    drawn = S.draw(candidates, replies, S.native_filters(b5))
    return drawn, hashlib.sha256(text.encode()).hexdigest()


def send(requests: list, label: str) -> dict:
    """Each request's reply text by key, from the cache or sent on the item's API session."""
    _, spec = D.grader()
    cache = JudgeCache(data_root() / "cache" / "judge")
    specs = {spec.id: spec}
    Q.LABEL = label
    _, ledger = follow_money.money(data_root(), ITEM, data_root() / Q.METER_LOG)
    atexit.register(ledger.close)
    ledger.source = label
    backend = Q.backend(ledger, specs, interpreter=False)
    backend.reconcile(cache)
    ledger.check(Q.estimate(requests, specs, set()))
    answers, _ = fetch(requests, backend, cache)
    print(f"charged {ledger.close()}")
    return {k: v.text for k, v in answers.items()}


def graded(make: object, parts: dict, parse: object, label: str) -> dict:
    """Every item's value, each format failure repeated once under its own key, else pending."""
    first = {n: make(p) for n, p in parts.items()}
    got = send(list(first.values()), label)
    texts = {n: got.get(r.key) for n, r in first.items()}
    settled = S.settled(texts, {}, parse)
    bad = [n for n, v in settled.items() if v["status"] == "pending"]
    repeats = {}
    if bad:
        again = {n: make(parts[n], retry=True) for n in bad}
        got = send(list(again.values()), label)
        repeats = {n: got.get(r.key) for n, r in again.items()}
    return S.settled(texts, repeats, parse)


def main() -> int:
    command, args = sys.argv[1], sys.argv[2:]
    b5 = cr.follow_b5()
    _, spec = D.grader()
    limit = int(b5["audit"]["model"]["max_tokens"])
    out = data_root() / "follow_s1"
    if command == "pin":
        result = json.loads(Path(args[0]).read_text())
        record = S.data_record(b5, drawn(b5, result["responses"]), result["responses"])
        print(json.dumps(record, indent=1))
        if "--write" in args:
            path = REPO_ROOT / S.DATA_MANIFEST
            if path.exists():
                raise SystemExit(f"{path} exists: written once")
            path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    elif command == "pin-native":
        got, digest = native(b5, write="--write" in args)
        record = S.native_record(b5, got, digest)
        print(json.dumps(record, indent=1))
        if "--write" in args:
            path = REPO_ROOT / S.NATIVE_MANIFEST
            if path.exists():
                raise SystemExit(f"{path} exists: written once")
            path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    elif command in ("contamination", "contamination-native"):
        from residual_disposition.manifests import committed_json

        tail = "-native" if command.endswith("native") else ""
        if tail:
            pin = committed_json(REPO_ROOT / S.NATIVE_MANIFEST, "the extension's data pin")
            got, digest = native(b5)
            if S.native_record(b5, got, digest)["train"] != pin["train"]:
                raise SystemExit("the draw is not the committed pin")
            train = got["train"]
        else:
            pin = committed_json(REPO_ROOT / S.DATA_MANIFEST, "S1's data pin")
            train = drawn(b5, pin["run"])["train"]
        c = b5["s1"]["contamination"]
        habits = {b: load_config(f"behaviors/{b}").behavior.description for b in c["behaviors"]}
        listing = "\n".join(f"{k}: {v}" for k, v in habits.items())
        sample = S.contamination_sample(train, b5["seed"], c["sample"])
        parts = {r["prompt_id"]: {"habits": listing, "response": r["response"]} for r in sample}

        def make(p: dict, retry: bool = False) -> object:
            return S.grading_request("s1-contamination-v1", c["rubric"], p, spec, limit, retry)

        values = graded(make, parts, lambda x: S.parse_habits(x, c["behaviors"]),
                        f"follow/s1-contamination{tail}")  # fmt: skip
        (out / f"contamination{tail}.json").write_text(json.dumps(values, indent=1) + "\n")
    elif command == "audit":
        a, fresh = b5["audit"], json.loads(Path(args[0]).read_text())["fresh"]
        ws = Workspace(load_config("experiments/continuation_14b/eli5",
                                   ["tracking.wandb.enabled=false"]))  # fmt: skip
        prompts = {p.prompt_id: p.text for p in ws.prompts("certificate")}
        rows = {"base": _fresh(fresh["base"])}
        for row in a["states"]:
            saved = row["saved"] and S.saved_responses(Path("runs") / row["saved"], a["prompts"])
            rows[row["state"]] = saved or _fresh(fresh[row["state"]])
        parts = {f"{state}/{p}": {"prompt": prompts[p], "response": text}
                 for state, replies in rows.items() for p, text in replies.items()}  # fmt: skip

        def make(p: dict, retry: bool = False) -> object:
            return S.grading_request("capability-audit-v1", a["rubric"], p, spec, limit, retry)

        values = graded(make, parts, lambda x: S.parse_label(x, a["labels"]),
                        "follow/capability-audit")  # fmt: skip
        (out / "audit.json").write_text(json.dumps(values, indent=1) + "\n")
        (out / "audit_items.json").write_text(json.dumps(parts, ensure_ascii=False) + "\n")
    elif command == "check":
        parts = json.loads((out / "audit_items.json").read_text())
        rows, mapping = S.check_items(parts, b5["seed"], b5["audit"]["researcher_check"])
        d = out / "check"
        d.mkdir(exist_ok=True)
        (d / "items.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                               for r in rows))  # fmt: skip
        (d / "mapping.json").write_text(json.dumps(mapping, indent=1) + "\n")
        print(len(rows), d)
    elif command == "check-import":
        export = [json.loads(x) for x in Path(args[0]).read_text().split("\n") if x.strip()]
        d = out / "check"
        record = S.check_record(export, json.loads((d / "mapping.json").read_text()),
                                json.loads((out / "audit.json").read_text()),
                                b5["audit"]["labels"])  # fmt: skip
        (d / "export.jsonl").write_text(Path(args[0]).read_text())
        (d / "record.json").write_text(json.dumps(record, indent=1) + "\n")
        print(record)
    elif command == "outcome":
        runs = json.loads(Path(args[0]).read_text())["runs"]
        table = {}
        for line in b5["s1"]["states"]:
            hit = hit_of(detector_of(line["behavior"]))
            for seed in b5["s1"]["training_seeds"]:
                by = {role: {step: S.hits_by_prompt(_saved(runs[f"{line['line']}/{role}/{seed}"]),
                                                    hit, step) for step in S.STEPS}
                      for role in ("latent", "sham_s")}  # fmt: skip
                table[f"{line['line']}/{seed}"] = S.outcome(by["latent"], by["sham_s"], b5["seed"])
        name = "outcome-native.json" if "--native" in args else "outcome.json"
        (out / name).write_text(json.dumps(table, indent=1) + "\n")
        print(json.dumps(table, indent=1))
    return 0


def _fresh(run: str) -> dict:
    return {r["prompt"]: r["text"] for r in read_jsonl_gz(find_run(run) / "responses.jsonl.gz")}


def _saved(run: str) -> list[dict]:
    text = (find_run(run) / "responses" / "relearn_evals.jsonl").read_text()
    return [json.loads(x) for x in text.split("\n") if x.strip()]


if __name__ == "__main__":
    sys.exit(main())
