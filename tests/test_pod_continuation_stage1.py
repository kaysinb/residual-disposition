"""scripts/pod/continuation_stage1.sh dry run (CPU, bash, stub tools under a temporary
workspace): the passes in order with their batches, priors and ledgers, the watcher's deadline
moved before the engine pass and at the end, and the exports checked."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "pod" / "continuation_stage1.sh"
spec = importlib.util.spec_from_file_location(
    "continuation_meter_pod", REPO / "scripts" / "continuation_meter.py"
)
cm = importlib.util.module_from_spec(spec)
sys.modules["continuation_meter_pod"] = cm
spec.loader.exec_module(cm)

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
n=0; [ "${STUB_EXPORT_FAIL:-}" = "$(basename "$SP")" ] && n=1
for p in "$SP"/runs/*; do echo "runs/$(basename "$p"): $n differences found"; done
for p in data/states data/generations data/cache/adl_steps; do echo "$p: 0 differences found"; done
for a in "$WORKSPACE"/attempts/residual-disposition/$4; do
  [ -e "$a" ] && echo "runs/_attempts/$(basename "$a"): 0 differences found"
done
echo "{\"tool\": \"export\", \"space\": \"$SP\"}" >> "$STUB_LOG"
"""

BUCKET_DIR = r"""#!/usr/bin/env bash
echo "$2: 0 differences found"
"""


PROJECT_RUN = """#!/usr/bin/env bash
shift  # the source directory; the stub runs the command as is
exec "$@"
"""


@pytest.fixture
def workspace(tmp_path, capsys):
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
    # The source snapshot is this checkout; project-run is a stub on PATH.
    (tmp_path / "code" / "snap").mkdir(parents=True)
    (tmp_path / "code" / "snap" / "source").symlink_to(REPO)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "project-run").write_text(PROJECT_RUN)
    (bin_dir / "project-run").chmod(0o755)
    # The meter's launch receipt and the copy of its log, as the laptop makes them.
    log, receipt = tmp_path / "meter.jsonl", tmp_path / "receipt.json"
    meter_cli = ["--log", str(log)]
    assert (
        cm.main([*meter_cli, "admit", "stage0_1", "--spent-elsewhere", "0", "--storage", "55"]) == 0
    )
    assert (
        cm.main([*meter_cli, "start", "stage0_1", "--price", "3.99", "--receipt", str(receipt)])
        == 0
    )
    capsys.readouterr()
    (tmp_path / "meter-copy.jsonl").write_bytes(log.read_bytes())
    (tmp_path / "rp_env").write_text("")
    watch = ws / "watch" / "c1-w"
    watch.mkdir(parents=True)
    procs = []

    def setup(start: datetime) -> dict:
        deadline = start + timedelta(hours=14)
        (watch / "state.json").write_text(
            json.dumps(
                {"phase": "armed", "token": "t", "deadline": deadline.timestamp(), "revision": 1}
            )
        )
        # The live watcher the script looks for (pgrep -f "night_watch.py watch --state-dir").
        procs.append(
            subprocess.Popen(
                ["bash", "-c", f"exec -a 'night_watch.py watch --state-dir {watch}' sleep 120"]
            )
        )
        return {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "WORKSPACE": str(ws),
            "CODE_ROOT": str(tmp_path / "code"),
            "RP_ENV": str(tmp_path / "rp_env"),
            "STUB_LOG": str(tmp_path / "stub.jsonl"),
            "SNAP": "snap",
            "START": start.isoformat(),
            "WN": "c1-w",
            "PRICE": "3.99",
            "HARD_HOURS": "17",
            "STAGE1_HOURS": "14",
            "ENGINE_HOURS": "3",
            "RECEIPT": str(receipt),
            "METER_LOG": str(tmp_path / "meter-copy.jsonl"),
            "LEDGER_A_USD": "7.0",
            "LEDGER_B_USD": "4.0",
            "CONTAINER_IMAGE": "img@sha256:x",
        }

    yield ws, setup, tmp_path / "stub.jsonl"
    for p in procs:
        p.kill()


def run_chain(env):
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=300
    )


def events(log):
    return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


def overrides(e):
    return dict(a.split("=", 1) for a in e["cmd"] if a.startswith("pipeline."))


def test_the_chain_runs_its_passes_in_order(workspace):
    ws, setup, log = workspace
    start = datetime.now(UTC) - timedelta(minutes=20)
    begun = time.time()
    out = run_chain(setup(start))
    assert out.returncode == 0, out.stdout + out.stderr
    ev = events(log)
    passes = [e for e in ev if e["tool"] == "run_detached"]
    assert [Path(e["attempt"]).name.rsplit("-", 1)[1] for e in passes] == ["link", "pun", "engine"]
    assert len({Path(e["attempt"]).name.rsplit("-", 1)[0] for e in passes}) == 1  # one chain id
    link, pun, engine = map(overrides, passes)
    for p in (link, pun):  # both passes in the stage1 batch from START, prior 0
        assert p["pipeline.plan.phase"] == "reread"
        assert p["pipeline.plan.cost.batches.stage1"] == "14"
        assert p["pipeline.plan.cost.prior_hours"] == "0.0"
        assert p["pipeline.plan.cost.started_at"] == start.isoformat()
    assert (
        link["pipeline.plan.api.adl_ledger.worker"],
        pun["pipeline.plan.api.adl_ledger.worker"],
    ) == (
        "cont-a",
        "cont-b",
    )
    assert link["pipeline.plan.api.adl_ledger.allowance_usd"] == "7.0"
    assert pun["pipeline.plan.api.adl_ledger.allowance_usd"] == "4.0"
    assert engine["pipeline.plan.phase"] == "engine"
    assert 2.99 < float(engine["pipeline.plan.cost.batches.engine"]) <= 3.0001
    spaces = [
        e["cmd"][[i for i, c in enumerate(e["cmd"]) if c.endswith("rest_runner.sh")][0] + 1]
        for e in passes
    ]
    assert spaces == [str(ws / "a"), str(ws / "b"), str(ws / "a")]
    deadlines = [datetime.fromisoformat(e["at"]).timestamp() for e in ev if e["tool"] == "deadline"]
    assert len(deadlines) == 2
    assert abs(deadlines[0] - (begun + 3 * 3600)) < 120  # the engine's: now + 3 h
    assert deadlines[1] - begun < 600  # the stop: now + 3 min
    assert all(d <= start.timestamp() + 17 * 3600 for d in deadlines)
    order = [e["tool"] for e in ev]
    assert order.index("export") < order.index("deadline")  # Stage 1 exported before the engine
    chain_dir = next((ws / "attempts" / "residual-disposition").glob("chain-c1-*"))
    summary = json.loads((chain_dir / "summary.json").read_text())
    assert summary["ok"] and summary["engine"] == "completed"
    assert json.loads((chain_dir / "receipt-check.json").read_text())["valid"]


def test_no_pass_starts_without_a_receipt_that_covers_the_hours(workspace):
    ws, setup, log = workspace
    env = setup(datetime.now(UTC) - timedelta(minutes=20))
    out = run_chain({**env, "HARD_HOURS": "21"})  # beyond the reservation's 20.45 h
    assert out.returncode != 0 and "receipt does not cover" in out.stdout
    assert not [e for e in events(log) if e["tool"] == "run_detached"]


def test_an_unclean_stage1_export_skips_the_engine_and_fails_the_chain(workspace):
    ws, setup, log = workspace
    out = run_chain({**setup(datetime.now(UTC) - timedelta(minutes=20)), "STUB_EXPORT_FAIL": "b"})
    assert out.returncode != 0 and "engine not run" in out.stdout
    names = [
        Path(e["attempt"]).name.rsplit("-", 1)[1]
        for e in events(log)
        if e["tool"] == "run_detached"
    ]
    assert names == ["link", "pun"]


def test_every_deadline_stays_within_the_hard_limit(workspace):
    ws, setup, log = workspace
    start = datetime.now(UTC) - timedelta(hours=15, minutes=30)  # START + 17 h is 1.5 h away
    out = run_chain(setup(start))
    assert "not run: less than 10 min" in out.stdout  # Stage 1's deadline has passed
    hard = start.timestamp() + 17 * 3600
    deadlines = [
        datetime.fromisoformat(e["at"]).timestamp() for e in events(log) if e["tool"] == "deadline"
    ]
    assert deadlines and all(d <= hard + 1 for d in deadlines)
    assert abs(deadlines[0] - hard) < 2  # the engine's deadline clamped to START + 17 h
