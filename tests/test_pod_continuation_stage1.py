"""scripts/pod/continuation_stage1.sh dry run (CPU, bash, stub tools under a temporary
workspace): the passes in order with their batches, priors and ledgers, the watcher's deadline
moved before the engine pass and at the end, and the exports checked."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pod" / "continuation_stage1.sh"

RUN_DETACHED = r"""
import json, os, sys
args = sys.argv[1:]
attempt = args[args.index("--attempt-dir") + 1]
cmd = args[args.index("--") + 1:]
os.makedirs(attempt, exist_ok=True)
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps({"tool": "run_detached", "attempt": attempt, "cmd": cmd}) + "\n")
json.dump({"status": "completed", "exit_code": 0}, open(os.path.join(attempt, "status.json"), "w"))
"""

NIGHT_WATCH = r"""
import json, os, sys
from datetime import datetime
args = sys.argv[1:]
state = os.path.join(args[args.index("--state-dir") + 1], "state.json")
s = json.load(open(state))
s["deadline"] = datetime.fromisoformat(args[args.index("--at") + 1]).timestamp()
s["revision"] = s.get("revision", 1) + 1
json.dump(s, open(state, "w"))
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps({"tool": "deadline", "at": args[args.index("--at") + 1]}) + "\n")
"""

BUCKET_FINAL = r"""#!/usr/bin/env bash
SP=$1
for p in "$SP"/runs/*; do echo "runs/$(basename "$p"): 0 differences found"; done
for p in data/states data/generations data/cache/adl_steps; do echo "$p: 0 differences found"; done
for a in "$WORKSPACE"/attempts/residual-disposition/$4; do
  [ -e "$a" ] && echo "runs/_attempts/$(basename "$a"): 0 differences found"
done
echo "{\"tool\": \"export\", \"space\": \"$SP\"}" >> "$STUB_LOG"
"""

BUCKET_DIR = r"""#!/usr/bin/env bash
echo "$2: 0 differences found"
"""


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    tools = ws / "tools"
    tools.mkdir(parents=True)
    (tools / "run_detached.py").write_text(RUN_DETACHED)
    (tools / "night_watch.py").write_text(NIGHT_WATCH)
    (tools / "source_snapshot.py").write_text("import sys\nsys.exit(0)\n")
    (tools / "bucket_final.sh").write_text(BUCKET_FINAL)
    (tools / "bucket_dir.sh").write_text(BUCKET_DIR)
    for space in ("a", "b"):
        (ws / space / "runs" / "continuation_14b").mkdir(parents=True)
    watch = ws / "watch" / "c1-w"
    watch.mkdir(parents=True)
    start = datetime.now(UTC) - timedelta(minutes=20)
    deadline = start + timedelta(hours=20.45 - 3.76)
    (watch / "state.json").write_text(
        json.dumps(
            {"phase": "armed", "token": "t", "deadline": deadline.timestamp(), "revision": 1}
        )
    )
    (tmp_path / "rp_env").write_text("")
    # The live watcher the script looks for (pgrep -f "night_watch.py watch --state-dir ...").
    watcher = subprocess.Popen(
        ["bash", "-c", f"exec -a 'night_watch.py watch --state-dir {watch}' sleep 120"]
    )
    yield ws, start
    watcher.kill()


def test_the_chain_runs_its_passes_in_order(workspace, tmp_path):
    ws, start = workspace
    log = tmp_path / "stub.jsonl"
    env = {
        **os.environ,
        "WORKSPACE": str(ws),
        "CODE_ROOT": str(tmp_path / "code"),
        "RP_ENV": str(tmp_path / "rp_env"),
        "STUB_LOG": str(log),
        "SNAP": "snap",
        "START": start.isoformat(),
        "WN": "c1-w",
        "PRICE": "3.99",
        "POD_HOURS": "20.45",
        "ENGINE_HOURS": "3.76",
        "CONTAINER_IMAGE": "img@sha256:x",
    }
    begun = time.time()
    out = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stdout + out.stderr
    events = [json.loads(x) for x in log.read_text().splitlines()]
    passes = [e for e in events if e["tool"] == "run_detached"]
    assert [Path(e["attempt"]).name.rsplit("-", 1)[1] for e in passes] == ["link", "pun", "engine"]
    args = [dict(a.split("=", 1) for a in e["cmd"] if a.startswith("pipeline.")) for e in passes]
    link, pun, engine = args
    assert link["pipeline.plan.phase"] == pun["pipeline.plan.phase"] == "reread"
    assert (
        link["pipeline.plan.cost.batches.stage1"]
        == pun["pipeline.plan.cost.batches.stage1"]
        == "16.69"
    )
    assert link["pipeline.plan.cost.prior_hours"] == "0.0"
    assert link["pipeline.plan.cost.started_at"] == start.isoformat()
    assert 0.3 < float(pun["pipeline.plan.cost.prior_hours"]) < 0.4  # the 20 min since START
    assert (
        link["pipeline.plan.api.adl_ledger.worker"],
        pun["pipeline.plan.api.adl_ledger.worker"],
    ) == (
        "cont-a",
        "cont-b",
    )
    assert engine["pipeline.plan.phase"] == "engine"
    assert 3.7 < float(engine["pipeline.plan.cost.batches.engine"]) <= 3.76
    spaces = [
        e["cmd"][[i for i, c in enumerate(e["cmd"]) if c.endswith("rest_runner.sh")][0] + 1]
        for e in passes
    ]
    assert spaces == [str(ws / "a"), str(ws / "b"), str(ws / "a")]
    limits = [
        int(next(c for c in e["cmd"] if c.startswith("RUN_LIMIT_S=")).split("=")[1]) for e in passes
    ]
    assert all(limit > 0 for limit in limits)
    deadlines = [
        datetime.fromisoformat(e["at"]).timestamp() for e in events if e["tool"] == "deadline"
    ]
    assert len(deadlines) == 2
    assert abs(deadlines[0] - (begun + 3.76 * 3600)) < 120  # the engine's deadline from now
    assert deadlines[1] - begun < 600  # the stop: now + 3 min
    order = [e["tool"] for e in events]
    assert order.index("export") < order.index("deadline")  # Stage 1 exported before the engine
    summary = json.loads(
        next((ws / "attempts" / "residual-disposition").glob("chain-c1-*/summary.json")).read_text()
    )
    assert summary["engine"] == "completed" and all(summary["export_clean"].values())
