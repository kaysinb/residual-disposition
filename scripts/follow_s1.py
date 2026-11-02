"""S1 and the capability audit on the laptop (preregistration §13.11; barrier B5).

    uv run python scripts/follow_s1.py pin <data result.json> [--write]
    uv run python scripts/follow_s1.py contamination
    uv run python scripts/follow_s1.py audit <train result.json>
    uv run python scripts/follow_s1.py outcome <train result.json>

``pin`` writes S1's data pin (``manifests/follow_s1_data.json``), committed before any training.
``contamination`` and ``audit`` send their blind calls on ``follow/s1_audit``, answers cached by
request, and write ``data/follow_s1/{contamination,audit}.json``. ``outcome`` writes
``data/follow_s1/outcome.json`` from the eight runs' saved responses.
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


def send(requests: list, label: str) -> dict:
    """Each request's answer from the cache, or sent on the item's API session."""
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
    return answers


def labels_of(requests: dict, answers: dict, labels: list[str] | None) -> dict:
    out = {}
    for name, request in requests.items():
        text = answers[request.key].text if request.key in answers else None
        out[name] = {"text": text}
        if labels and text:
            out[name]["label"] = S.parse_label(text, labels)
    return out


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
    elif command == "contamination":
        from residual_disposition.manifests import committed_json

        pin = committed_json(REPO_ROOT / S.DATA_MANIFEST, "S1's data pin")
        c = b5["s1"]["contamination"]
        habits = {b: load_config(f"behaviors/{b}").behavior.description for b in c["behaviors"]}
        listing = "\n".join(f"{k}: {v}" for k, v in habits.items())
        sample = S.contamination_sample(drawn(b5, pin["run"])["train"], b5["seed"], c["sample"])
        requests = {r["prompt_id"]: S.grading_request("s1-contamination-v1", c["rubric"],
                    {"habits": listing, "response": r["response"]}, spec, limit)
                    for r in sample}  # fmt: skip
        answers = send(list(requests.values()), "follow/s1-contamination")
        (out / "contamination.json").write_text(json.dumps(
            labels_of(requests, answers, None), indent=1) + "\n")  # fmt: skip
    elif command == "audit":
        a, fresh = b5["audit"], json.loads(Path(args[0]).read_text())["fresh"]
        ws = Workspace(load_config("experiments/continuation_14b/eli5",
                                   ["tracking.wandb.enabled=false"]))  # fmt: skip
        prompts = {p.prompt_id: p.text for p in ws.prompts("certificate")}
        rows = {"base": _fresh(fresh["base"])}
        for row in a["states"]:
            saved = row["saved"] and S.saved_responses(Path("runs") / row["saved"], a["prompts"])
            rows[row["state"]] = saved or _fresh(fresh[row["state"]])
        requests = {f"{state}/{p}": S.grading_request("capability-audit-v1", a["rubric"],
                    {"prompt": prompts[p], "response": text}, spec, limit)
                    for state, replies in rows.items() for p, text in replies.items()}  # fmt: skip
        answers = send(list(requests.values()), "follow/capability-audit")
        (out / "audit.json").write_text(json.dumps(
            labels_of(requests, answers, a["labels"]), indent=1) + "\n")  # fmt: skip
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
        (out / "outcome.json").write_text(json.dumps(table, indent=1) + "\n")
        print(json.dumps(table, indent=1))
    return 0


def _fresh(run: str) -> dict:
    return {r["prompt"]: r["text"] for r in read_jsonl_gz(find_run(run) / "responses.jsonl.gz")}


def _saved(run: str) -> list[dict]:
    text = (find_run(run) / "responses" / "relearn_evals.jsonl").read_text()
    return [json.loads(x) for x in text.split("\n") if x.strip()]


if __name__ == "__main__":
    sys.exit(main())
