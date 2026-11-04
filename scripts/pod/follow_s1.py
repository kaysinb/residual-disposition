"""S1's and the capability audit's Pod work (preregistration §13.11; barrier B5).

    uv run python scripts/pod/follow_s1.py --phase data --space /workspace/a --out <new directory>
    uv run python scripts/pod/follow_s1.py --phase train --space /workspace/a --out <new directory>
    uv run python scripts/pod/follow_s1.py --phase native --space /workspace/a --out <new directory>

``data``: the base model's response to each candidate, its run name in ``result.json``; the pin is
written on the laptop and committed (``scripts/follow_s1.py pin``) before ``train``. ``train``: the
eight S1 runs and their NLL, then each fresh audit pass. ``native``: the extension's eight runs on
CodeAlpaca's own responses (``manifests/follow_s1_native_data.json``, its file by sha256) and their
NLL, after which each run saves its weight shift from step 0 and keeps only its step-200 adapter.
Cells resume from what is saved; no redraw.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_build as B
from residual_disposition import follow_m0 as M
from residual_disposition import follow_s1 as S
from residual_disposition import pipeline as P
from residual_disposition.config import load_config
from residual_disposition.continuation_kstar import find_run
from residual_disposition.manifests import committed_json
from residual_disposition.paths import REPO_ROOT, data_root


def take(flag: str) -> str:
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return value


def runs_of(ws: P.Workspace, b5: dict, pin: dict, drawn: dict, targets: str | None
            ) -> tuple[dict, dict]:  # fmt: skip
    """The eight runs and their NLL; the extension's keep only their step-200 adapter after it."""
    data, held = S.examples(drawn["train"]), S.examples(drawn["held_out"])
    rules = cr.follow_rules().data["s1"]
    runs, nll = {}, {}
    for line in b5["s1"]["states"]:  # one engine and base model for both behaviors
        arm = P.arm_workspace(ws, line["behavior"])
        arm._generator, arm._base, arm._tokenizer = ws._generator, ws._base, ws._tokenizer
        for role in ("latent", "sham_s"):
            for seed in b5["s1"]["training_seeds"]:
                d = S.train(arm, rules, line[role], seed, pin, data, targets)
                runs[f"{line['line']}/{role}/{seed}"] = d.name
                nll[f"{line['line']}/{role}/{seed}"] = S.nll(arm, d, held).name
                if targets:
                    S.shift_then_keep_last(d)
        ws._generator, ws._base, ws._tokenizer = arm._generator, arm._base, arm._tokenizer
    return runs, nll


def main() -> int:
    launcher = os.environ.pop("RUN_ID", None)
    phase, out, space = take("--phase"), Path(take("--out")), Path(take("--space"))
    problems = B.placement_problems(space, out)
    if problems:
        raise SystemExit("; ".join(problems))
    b5 = cr.follow_b5()
    spec = b5["s1"]["data"]["candidates"]
    text = (data_root() / spec["file"]).read_text()
    if hashlib.sha256(text.encode()).hexdigest() != spec["sha256"]:
        raise SystemExit("the candidates are not B5's")
    rows = json.loads(text)
    ws = P.Workspace(load_config("experiments/continuation_14b/eli5",
                                 ["tracking.wandb.enabled=false"]))  # fmt: skip
    out.mkdir(parents=True, exist_ok=False)
    result: dict = {"status": "complete", "phase": phase, "launcher_run_id": launcher}
    if phase == "data":
        result["responses"] = S.responses(ws, b5, rows).name
    elif phase == S.NATIVE:
        pin = committed_json(REPO_ROOT / S.NATIVE_MANIFEST, "the extension's data pin")
        native = (data_root() / S.NATIVE_FILE).read_bytes()
        digest = hashlib.sha256(native).hexdigest()
        drawn = S.draw(rows, json.loads(native), S.native_filters(b5))
        if S.native_record(b5, drawn, digest) != {k: v for k, v in pin.items()
                                                    if k not in ("sha256", "committed_at")}:
            raise SystemExit("the draw is not the committed pin")
        runs, nll = runs_of(ws, b5, pin, drawn, S.NATIVE)
        result.update(runs=runs, nll=nll)
    else:
        pin = committed_json(REPO_ROOT / S.DATA_MANIFEST, "S1's data pin")
        run = find_run(pin["run"])
        replies = {r["prompt"]: r["text"] for r in P.read_jsonl_gz(run / "responses.jsonl.gz")}
        drawn = S.draw(rows, replies, S.six_detectors(b5))
        record = S.data_record(b5, drawn, pin["run"])
        if any(pin.get(k) != v for k, v in record.items()):
            raise SystemExit("the draw is not the committed pin")
        runs, nll = runs_of(ws, b5, pin, drawn, None)
        fresh = {"base": S.fresh_pass(ws, b5, None).name}
        for row in b5["audit"]["states"]:
            if row["saved"] is None:
                fresh[row["state"]] = S.fresh_pass(ws, b5, row["state"]).name
        result.update(runs=runs, nll=nll, fresh=fresh)
    M.write_json(out / "result.json", result)
    print(json.dumps({"status": "complete", "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
