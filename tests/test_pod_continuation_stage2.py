"""scripts/pod/continuation_stage2.sh dry run (CPU, bash, stub tools under a temporary
workspace; plan C8, item F; review s4b): each session's passes in order in its own space, each
Pod's inputs copied from the other's space create-only (as the manifests name them), the decision's
steps as stage2_next.py names them, the Pod's identity, the session's meter receipt and the
snapshot's receipt checked before any pass, every bound from the verified session, the
certificate's plan check before its GPU pass, the export and the stop on every exit. Stage 5's
sessions (C13; review s4t2): the unsplit item's one session (worker pod, at most 12 h) in space a,
the stage5 batch, s5a's steps as stage5_next.py names them up to the decision, the commit
barriers before k* (s5k) and before the certificate, H's relearning and the readers (s5c)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "pod" / "continuation_stage2.sh"
spec = importlib.util.spec_from_file_location(
    "continuation_meter_pod2", REPO / "scripts" / "continuation_meter.py"
)
cm = importlib.util.module_from_spec(spec)
sys.modules["continuation_meter_pod2"] = cm
spec.loader.exec_module(cm)

RUN_DETACHED = r"""
import json, os, sys
args = sys.argv[1:]
attempt = args[args.index("--attempt-dir") + 1]
cmd = args[args.index("--") + 1:]
os.makedirs(attempt, exist_ok=True)
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps({"tool": "run_detached", "attempt": attempt, "cmd": cmd}) + "\n")
fail = (os.environ.get("STUB_FAIL_CHECK") and "pipeline.plan.check_only=true" in cmd) or (
    "pipeline.plan.phase=" + os.environ.get("STUB_FAIL_PHASE", "-") in cmd)
status = {"status": "failed", "exit_code": 1} if fail else {"status": "completed", "exit_code": 0}
json.dump(status, open(os.path.join(attempt, "status.json"), "w"))
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
for p in data/states data/generations data/cache/adl_steps; do  # as rclone: a missing source fails
  if [ -d "$SP/$p" ]; then echo "$p: 0 differences found"
  else
    echo "$p: ERROR : Local file system at $SP/$p: directory not found"
    echo "$p: 1 errors while checking"
  fi
done
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

LINK_STATES = ["s4303557a4e7a9f64", "sf8d4c3b1ca796eec", "sab16754d0858729b", "se7a636b801f0ab1d"]
Y_ORGANISM = "sbf05f589f929edf8"
Y_RUNS = ["20261014-072219-rates-2931c028-5097", "20261014-072309-data_relearn-8950fcdc-6f7f"]


def a_cell(space, name, step, params, summary, behavior="example_link", files=None):
    run = space / "runs" / "continuation_14b" / name
    run.mkdir(parents=True)
    config = {"step": step, "behavior": {"id": behavior}, "params": params}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    (run / "summary.json").write_text(json.dumps(summary))
    for rel, content in (files or {}).items():
        (run / rel).write_text(content)
    return run


def pilot_inputs(ws):
    """What b1 copies, as minimal files where the volume holds them: link's states, the analyst
    index and the prompt pools in a; the pilot's Y organism, its reference and data in b."""
    a, b = ws / "a", ws / "b"
    for space, sid in [*((a, x) for x in LINK_STATES), (b, Y_ORGANISM)]:
        (space / "data" / "states" / sid).mkdir(parents=True)
        (space / "data" / "states" / sid / "manifest.json").write_text(json.dumps({"adapters": []}))
    for run in Y_RUNS:
        (b / "runs" / "feasibility_14b" / run).mkdir(parents=True)
        (b / "runs" / "feasibility_14b" / run / "summary.json").write_text("{}")
    index = a / "runs" / "feasibility_14b" / "train-run" / "observations"
    index.mkdir(parents=True)
    (index / "state_index.jsonl").write_text("{}\n")
    (index.parent / "meta.json").write_text(json.dumps({"status": "completed"}))
    (a / "runs" / "stage1_14b").mkdir(parents=True)
    (b / "data" / "generations" / "d5f5179f8853cb6cc").mkdir(parents=True)
    (b / "data" / "generations" / "d5f5179f8853cb6cc" / "data.jsonl").write_text("{}\n")
    (a / "data" / "prompts" / "pool").mkdir(parents=True)
    (a / "data" / "prompts" / "pool" / "set.jsonl").write_text("{}\n")


@pytest.fixture
def workspace(tmp_path, capsys, monkeypatch):
    ws = tmp_path / "ws"
    tools = ws / "tools"
    tools.mkdir(parents=True)
    (tools / "run_detached.py").write_text(RUN_DETACHED)
    (tools / "night_watch.py").write_text(NIGHT_WATCH)
    (tools / "source_snapshot.py").write_text("import sys\nsys.exit(0)\n")
    (tools / "bucket_final.sh").write_text(BUCKET_FINAL)
    (tools / "bucket_dir.sh").write_text(BUCKET_DIR)
    for space in ("a", "b", "c"):
        (ws / space / "runs" / "continuation_14b").mkdir(parents=True)
    for space in ("a", "c"):  # what a Pod's space holds and exports (c holds no ADL cache)
        for tree_ in ("states", "generations"):
            (ws / space / "data" / tree_).mkdir(parents=True, exist_ok=True)
    # The source snapshot: this checkout's scripts and a manifests directory of its own.
    source = tmp_path / "code" / "snap" / "source"
    (source / "manifests").mkdir(parents=True)
    for d in ("scripts", "src", "configs"):
        (source / d).symlink_to(REPO / d)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "project-run").write_text(PROJECT_RUN)
    (bin_dir / "project-run").chmod(0o755)
    snapshot_receipt = tmp_path / "snapshot.receipt.json"
    snapshot_receipt.write_text(json.dumps({"snapshot_id": "snap"}))
    procs = []

    def setup(session, start: datetime, *, pod="POD1", hours=12.0, worker=None, api="0",
              provider=None, watch_late_s=0, item=None, s3_space="c") -> dict:  # fmt: skip
        # Stage 5's item is unsplit: one session at a time, worker pod; the items above it are
        # completed in this journal, at no cost, so that the meter admits it
        strong = session in STRONG  # strong/stage5, split by its frozen split (C16-2)
        stage3 = session.startswith("s3")  # stage3/countdown, unsplit (C17)
        unsplit = (session.startswith("s5") and not strong) or stage3
        second = session.startswith("p")  # the second behavior's item, split to Pod B alone
        worker = worker or (STRONG.get(session) or "pod" if session.startswith(("s5", "s3")) else
                            "pod-a" if session.startswith("a") else "pod-b")  # fmt: skip
        # The meter: the item split, this Pod's session opened with its ID, deadline and API money.
        log, receipt = tmp_path / f"meter-{session}.jsonl", tmp_path / f"receipt-{session}.json"
        meter = ["--log", str(log)]
        fallback = session in DPO_FALLBACK  # strong/stage5_dpo, unsplit (C16-3)
        dpo = session.startswith("s5d") and not fallback  # the descriptive DPO lineage's (C15)
        item = item or ("stage3/countdown" if stage3 else
                        "strong/stage5" if strong else "strong/stage5_dpo" if fallback else
                        "stage5_dpo" if dpo else "stage5" if unsplit else
                        "stage2/pun_intended" if second else "stage2/example_link")  # fmt: skip
        if item == "strong/stage5_dpo":  # admitted once its gate is open (§12.8)
            monkeypatch.setattr(cm, "fallback_gate", lambda: {"status": "open"})
        if item in ("stage5", "stage5_dpo", "strong/stage5", "strong/stage5_dpo") or (
            item.startswith("stage3/")
        ):
            for above in cm_items()[: cm_items().index(item)]:
                cm.main([*meter, "admit", above, "--spent-elsewhere", "0", "--storage", "55"])
                cm.main([*meter, "start", above])
                cm.main([*meter, "complete", above])
        cm.main([*meter, "admit", item, "--spent-elsewhere", "0", "--storage", "55"])
        cm.main([*meter, "start", item])
        if item == "strong/stage5":
            cm.main([*meter, "split", item])  # its frozen split: s5-a, s5-b
        elif second:
            cm.main([*meter, "split", item, "--share", "pod-b=1.0"])
        elif not unsplit and not strong:
            cm.main([*meter, "split", item, "--share", "pod-a=0.48", "--share", "pod-b=0.52"])
        deadline = (start + timedelta(hours=hours)).isoformat()
        cm.main([*meter, "session-open", item, "--worker", worker, "--pod-id", pod,
                 "--deadline", deadline, "--price", "3.99", "--hours", str(hours),
                 "--api-usd", api, "--receipt", str(receipt)])  # fmt: skip
        capsys.readouterr()
        copy = tmp_path / f"meter-copy-{session}.jsonl"
        copy.write_bytes(log.read_bytes())
        watch = ws / "watch" / f"{session}-w"
        watch.mkdir(parents=True, exist_ok=True)
        (watch / "state.json").write_text(json.dumps(
            {"phase": "armed", "token": "t",
             "deadline": (start + timedelta(hours=hours)).timestamp() + watch_late_s,
             "revision": 1}
        ))  # fmt: skip
        rp_env = tmp_path / f"rp_env-{session}"  # what the provider sets in the Pod
        rp_env.write_text(f"export RUNPOD_POD_ID={provider or pod}\n")
        procs.append(subprocess.Popen(
            ["bash", "-c", f"exec -a 'night_watch.py watch --state-dir {watch}' sleep 120"]
        ))  # fmt: skip
        return {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "WORKSPACE": str(ws),
            "CODE_ROOT": str(tmp_path / "code"),
            "RP_ENV": str(rp_env),
            "STUB_LOG": str(tmp_path / "stub.jsonl"),
            "SESSION": session,
            "SNAP": "snap",
            "WN": f"{session}-w",
            "PRICE": "3.99",
            "RECEIPT": str(receipt),
            "METER_LOG": str(copy),
            "POD_ID": pod,
            "ACTIVE": "s11c92132dd9839ce",
            "SOURCE_SNAPSHOT_RECEIPT": str(snapshot_receipt),
            "SOURCE_SNAPSHOT_ID": "snap",
            "SNAPSHOT_RECEIPT_SHA256": hashlib.sha256(snapshot_receipt.read_bytes()).hexdigest(),
            "CONTAINER_IMAGE": "img@sha256:x",
            **({"ITEM": item, "BEHAVIOR": "example_link"} if unsplit or strong else {}),
            **({"BEHAVIOR": "countdown", "S3_SPACE": s3_space} if stage3 else {}),
            **({"ITEM": item} if second else {}),
        }

    yield ws, setup, tmp_path / "stub.jsonl", source
    for p in procs:
        p.kill()


STRONG = {"s5s": "s5-a", "s5v": "s5-a", "s5sk": "s5-b", "s5sc": "s5-b"}  # session: worker
DPO_FALLBACK = ("s5ds", "s5dv", "s5dx", "s5dc")  # the strong DPO fallback's, worker pod, in c


def cm_items():
    """The meter's items in their frozen priority."""
    from residual_disposition.paths import CONFIG_ROOT

    data = yaml.safe_load((CONFIG_ROOT / "continuation" / "allocations.yaml").read_text())
    return [a["item"] for a in data["allocations"]]


def run_chain(env):
    command = ["bash", str(SCRIPT)]
    return subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)


def events(log):
    return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


def passes(log):
    return [e for e in events(log) if e["tool"] == "run_detached"]


def overrides(e):
    pairs = (a.lstrip("+").split("=", 1) for a in e["cmd"] if a.lstrip("+").startswith("pipeline."))
    return dict(pairs)


def space_of(e):
    return e["cmd"][[i for i, c in enumerate(e["cmd"]) if c.endswith("rest_runner.sh")][0] + 1]


def tree(space):
    return sorted(str(p.relative_to(space)) for p in space.rglob("*"))


PENDING_PHASES: set[str] = set()  # every phase the chain runs is written (C10)


def loads(log):
    """Every pass's real config loads with the overrides the chain gave it (review s4f: the y2
    pass got the y batch's key, which its config has not); returns the phases not written yet."""
    from residual_disposition import pipeline
    from residual_disposition.config import load_config

    pending = set()
    for e in passes(log):
        cmd = e["cmd"]
        i = [k for k, c in enumerate(cmd) if c.endswith("rest_runner.sh")][0]
        phase = overrides(e)["pipeline.plan.phase"]
        if phase not in pipeline.PHASES:
            pending.add(phase)
            continue
        load_config(cmd[i + 2], cmd[i + 3 :])
    assert pending <= PENDING_PHASES
    return pending


def stop_asked(log, within_s=200):
    """The chain asked the watcher to stop the Pod within ``within_s`` seconds."""
    asked = [datetime.fromisoformat(e["at"]) for e in events(log) if e["tool"] == "deadline"]
    return bool(asked) and asked[-1] <= datetime.now(UTC) + timedelta(seconds=within_s)


START = datetime.now(UTC) - timedelta(minutes=20)


def test_b1_copies_its_inputs_and_runs_y_alone_in_its_own_space(workspace):
    ws, setup, log, _ = workspace
    pilot_inputs(ws)
    before = {x: tree(ws / x) for x in ("a", "b")}
    out = run_chain(setup("b1", START))
    assert out.returncode == 0, out.stdout + out.stderr
    ps = passes(log)
    assert [Path(e["attempt"]).name.rsplit("-", 1)[1] for e in ps] == ["y100", "y200"]
    assert {space_of(e) for e in ps} == {str(ws / "c")}  # Pod B writes only to c
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == ["control_y", "control_y"]
    assert loads(log) == set()  # each pass's config with its overrides (review s4f)
    y100, y200 = ps
    assert y100["cmd"][y100["cmd"].index(str(ws / "c")) + 1].endswith("control_y_link_100")
    o = overrides(y100)
    assert o["pipeline.plan.api.adl_ledger.worker"] == "pod-b"
    assert o["pipeline.plan.api.adl_ledger.allowance_usd"] == "0.0"  # the session's: none
    assert o["pipeline.plan.cost.batches.y"] == "12.0"  # its own batch only
    assert "pipeline.plan.cost.batches.stage2" not in o
    # the cost meter counts the session's 12 h back from its deadline (review s4b)
    since = datetime.fromisoformat(o["pipeline.plan.cost.started_at"])
    assert abs((since - START).total_seconds()) < 1.5
    for sid in [*LINK_STATES, Y_ORGANISM]:  # Y from b (review s4f)
        assert (ws / "c" / "data" / "states" / sid / "manifest.json").is_file()
    assert (ws / "c" / "runs" / "feasibility_14b" / Y_RUNS[0]).is_dir()
    assert (ws / "c" / "runs" / "feasibility_14b" / "train-run" / "observations").is_dir()
    assert {x: tree(ws / x) for x in ("a", "b")} == before  # a and b untouched
    assert [e["space"] for e in events(log) if e["tool"] == "export"] == [str(ws / "c")]
    assert (ws / "c" / "data" / "cache" / "adl_steps").is_dir()  # exported empty, not missing
    chain = next((ws / "attempts" / "residual-disposition").glob("chain-b1-*"))
    summary = json.loads((chain / "summary.json").read_text())
    assert summary["ok"] and summary["worker"] == "pod-b" and summary["pod_id"] == "POD1"
    ledger = ws / "c" / "data" / "ledgers" / "adl-api-pod-b.jsonl"  # its copy, in the chain's
    assert (chain / "ledger-adl-api-pod-b.jsonl").read_bytes() == ledger.read_bytes()
    assert summary["steps"]["ledger"] == 0  # recorded before the summary is written
    assert json.loads((chain / "receipt-check.json").read_text())["valid"]
    assert stop_asked(log)  # the end of the chain stops the Pod


def test_a_missing_input_stops_b1_before_any_pass(workspace):
    ws, setup, log, _ = workspace
    pilot_inputs(ws)
    __import__("shutil").rmtree(ws / "b" / "data" / "states" / Y_ORGANISM)
    out = run_chain(setup("b1", START))
    assert out.returncode != 0 and not passes(log)
    assert stop_asked(log)  # review s4b: a failure stops the Pod at once


@pytest.mark.parametrize(
    "setup_kw, change, message",
    [
        # the caller's POD_ID is not the provider's identity of this Pod
        ({"provider": "POD2"}, {}, "is not this Pod's identity from the provider"),
        # the Pod is POD9, but the session is POD1's
        ({"provider": "POD9"}, {"POD_ID": "POD9"}, "does not authorize this Pod"),
        ({}, {"SNAPSHOT_RECEIPT_SHA256": "0" * 64}, "not the one set at the Pod's creation"),
        ({}, {"SOURCE_SNAPSHOT_ID": "other"}, "not the one set at the Pod's creation"),
        # review s4b: a watcher two hours past the session's deadline, whatever START says
        ({"watch_late_s": 7200}, {"START": "2099-01-01T00:00:00+00:00", "HARD_HOURS": "99"},
         "is past the session's"),
    ],
)  # fmt: skip
def test_no_pass_starts_without_this_pod_s_session_and_snapshot(
    workspace, setup_kw, change, message
):
    ws, setup, log, _ = workspace
    pilot_inputs(ws)
    env = setup("b1", START, **setup_kw)
    out = run_chain({**env, **change})
    assert out.returncode != 0 and message in out.stdout
    assert not passes(log)
    assert stop_asked(log)  # review s4b: an early refusal stops the Pod, not its deadline


def test_the_adl_ceiling_is_the_session_s_recorded_once_and_kept_by_a_rerun(workspace):
    # review s4b: the ledger may commit the API money the meter journaled for this session;
    # review s4d: one absolute ceiling per meter session, which a rerun reuses
    from residual_disposition.budget import ApiLedger

    ws, setup, log, _ = workspace
    pilot_inputs(ws)
    path = ws / "c" / "data" / "ledgers" / "adl-api-pod-b.jsonl"
    ledger = ApiLedger(path, 10.0, "pod-b")
    ids = ledger.reserve({"k1": 1.0, "k2": 0.5}, "earlier session")
    ledger.settle(ids["k1"], "k1", 1.0, "earlier session")  # k2 left open: still committed
    ledger.close()
    env = setup("b1", START, api="2.5")
    out = run_chain(env)
    assert out.returncode == 0, out.stdout + out.stderr
    first = overrides(passes(log)[0])
    assert float(first["pipeline.plan.api.adl_ledger.allowance_usd"]) == pytest.approx(1.5 + 2.5)
    session = json.loads(Path(env["RECEIPT"]).read_text())["log_sha256"]
    assert first["pipeline.plan.api.adl_ledger.session"] == session
    # the session spends $2, its chain is started again: the ceiling stays $4, not $1.5 + $2 + $2.5
    ledger = ApiLedger(path, 4.0, "pod-b")
    ids = ledger.reserve({"k3": 2.0}, "this session")
    ledger.settle(ids["k3"], "k3", 2.0, "this session")
    ledger.close()
    log.unlink()
    state = ws / "watch" / "b1-w" / "state.json"  # the Pod started again, its watcher re-armed
    armed = json.loads(state.read_text())
    state.write_text(json.dumps({**armed, "deadline": (START + timedelta(hours=12)).timestamp()}))
    out = run_chain(env)
    assert out.returncode == 0, out.stdout + out.stderr
    again = overrides(passes(log)[0])
    assert float(again["pipeline.plan.api.adl_ledger.allowance_usd"]) == pytest.approx(4.0)
    ceilings = [json.loads(x) for x in path.read_text().splitlines()]
    assert sum(e["event"] == "ceiling" for e in ceilings) == 1  # recorded once


def screening(space, rows, behavior="example_link"):
    doses = {str(d): {"latent": f"L{d}", "sham_s": f"Hs{d}", "excess": {"upper": u},
                      "pool": {"hits": h}} for d, (u, h) in rows.items()}  # fmt: skip
    from residual_disposition import continuation_rules as cr

    rules = cr.load_rules()
    record = {"behavior": behavior, "doses": doses, "rules_sha256": rules.sha256}
    a_cell(space, "20261020-000000-stage2_screening-aaaa-0001", "stage2_screening",
           {"rules_sha256": rules.sha256}, {}, behavior=behavior,
           files={"stage2_screening.json": json.dumps(record)})  # fmt: skip


def test_a1_runs_the_confirmation_the_decision_needs_and_stops_without_progress(workspace):
    ws, setup, log, _ = workspace
    screening(ws / "a", {400: (0.004, 0), 800: (0.004, 0)})
    out = run_chain(setup("a1", START))
    assert out.returncode != 0 and "no progress at confirm 400" in out.stdout
    names = [Path(e["attempt"]).name.rsplit("-", 2)[-2:] for e in passes(log)]
    assert [n[-1] for n in names] == ["y2", "screen", "400"]  # y2 first (review s4f)
    assert loads(log) == set()
    confirm = overrides(passes(log)[2])
    assert confirm["pipeline.plan.phase"] == "confirm"
    assert confirm["pipeline.plan.confirm.latent"] == "L400"
    assert confirm["pipeline.plan.confirm.sham_s"] == "Hs400"
    assert {space_of(e) for e in passes(log)} == {str(ws / "a")}


def decided_space(space):
    """A screening, a passed confirmation at 400 and the expression checks there: the decision's
    analysis dose is 400."""
    from residual_disposition import continuation_confirm as C
    from residual_disposition import continuation_rules as cr
    from residual_disposition import continuation_stage2 as S2

    screening(space, {400: (0.004, 0), 800: (0.004, 0)})
    version = S2.current_detector_version("example_link")
    seed = cr.confirmation_seed(20261021, "stage2/example_link", 400, 1, version)
    draw = {"attempt": 1, "seed": seed, "detector_version": version, "passed": True,
            "purpose": None, "lineage": "stage2/example_link", "base_seed": 20261021,
            "prompts_sha256": C.FROZEN["sha256"], "latent": {"hits": 0, "n": 600}}  # fmt: skip
    a_cell(space, "20261020-000001-stage2_confirmation-bbbb-0001", "stage2_confirmation",
           {"dose": 400, "latent": "L400"}, draw)  # fmt: skip
    a_cell(space, "20261020-000002-rates_ood-cccc-0001", "rates_ood", {"state": "L400"},
           {"below_tolerance": True, "n": 200})  # fmt: skip
    framing = {"pooled_hits": 0, "pooled_n": 3000, "long_hits": 0, "long_n": 1000}
    a_cell(space, "20261020-000003-adl_corpus-dddd-0001", "adl_corpus", {"state": "L400"},
           {"framings": {"chat": framing, "raw": framing}})  # fmt: skip


def test_option_b_a1_leaves_k_star_to_a1k_which_also_redraws_l_s_pd_pool(workspace):
    # Codex s4k: A1 on another Hopper without k*; k* and L's PD pool on H100 SXM
    ws, setup, log, _ = workspace
    decided_space(ws / "a")
    out = run_chain(setup("a1", START))
    assert out.returncode == 0, out.stdout + out.stderr
    assert [overrides(e)["pipeline.plan.phase"] for e in passes(log)] == ["y2", "screen", "decide"]
    assert "the decision is committed on the laptop before k*" in out.stdout
    log.unlink()
    out = run_chain(setup("a1k", START, pod="POD2"))
    assert out.returncode == 0, out.stdout + out.stderr
    ps = passes(log)
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == ["kstar", "pd_pool"]
    assert loads(log) == set()
    kstar, pool = overrides(ps[0]), overrides(ps[1])
    assert (kstar["pipeline.plan.kstar.dose"], kstar["pipeline.plan.kstar.sham_s"]) == (
        "400",
        "Hs400",
    )
    assert (pool["pipeline.plan.pd_pool.state"], pool["pipeline.plan.pd_pool.draw"]) == (
        "L400", "stage2-sxm")  # fmt: skip
    assert {space_of(e) for e in ps} == {str(ws / "a")}


def test_a1k_without_a_decided_analysis_dose_runs_nothing(workspace):
    ws, setup, log, _ = workspace
    screening(ws / "a", {400: (0.004, 0), 800: (0.004, 0)})  # nothing confirmed yet
    out = run_chain(setup("a1k", START))
    assert out.returncode != 0 and "a1k: no decided analysis dose" in out.stdout
    assert not passes(log)


def test_a1_decides_and_leaves_k_star_to_after_the_decision_s_commit(workspace):
    # §12.7, review s4z 1: the decision is committed before the lineage's k*
    ws, setup, log, _ = workspace
    from residual_disposition import continuation_rules as cr
    from residual_disposition import continuation_stage2 as S2

    screening(ws / "a", {400: (0.004, 0), 800: (0.004, 0)})
    from residual_disposition import continuation_confirm as C

    version = S2.current_detector_version("example_link")
    seed = cr.confirmation_seed(20261021, "stage2/example_link", 400, 1, version)
    draw = {"attempt": 1, "seed": seed, "detector_version": version, "passed": True,
            "purpose": None, "lineage": "stage2/example_link", "base_seed": 20261021,
            "prompts_sha256": C.FROZEN["sha256"], "latent": {"hits": 0, "n": 600}}  # fmt: skip
    a_cell(ws / "a", "20261020-000001-stage2_confirmation-bbbb-0001", "stage2_confirmation",
           {"dose": 400, "latent": "L400"}, draw)  # fmt: skip
    a_cell(ws / "a", "20261020-000002-rates_ood-cccc-0001", "rates_ood", {"state": "L400"},
           {"below_tolerance": True, "n": 200})  # fmt: skip
    framing = {"pooled_hits": 0, "pooled_n": 3000, "long_hits": 0, "long_n": 1000}
    a_cell(ws / "a", "20261020-000003-adl_corpus-dddd-0001", "adl_corpus", {"state": "L400"},
           {"framings": {"chat": framing, "raw": framing}})  # fmt: skip
    out = run_chain(setup("a1", START))
    assert out.returncode == 0, out.stdout + out.stderr
    ps = passes(log)
    phases = [overrides(e)["pipeline.plan.phase"] for e in ps]
    assert phases == ["y2", "screen", "decide"]
    assert "analysis dose 400: the decision is committed on the laptop before k*" in out.stdout
    assert loads(log) == set()
    y2 = overrides(ps[0])
    assert ps[0]["cmd"][ps[0]["cmd"].index(str(ws / "a")) + 1].endswith("y2_zero_indexed_lists")
    assert (
        y2["pipeline.plan.cost.batches.stage2"] == "12.0"
        and "pipeline.plan.cost.batches.y" not in y2
    )
    decide = overrides(ps[2])
    key = "pipeline.plan.decide."
    assert decide[key + "screening"] == "20261020-000000-stage2_screening-aaaa-0001"
    assert decide[key + "confirmation"] == "[20261020-000001-stage2_confirmation-bbbb-0001]"
    assert decide[key + "expression.ood"] == "20261020-000002-rates_ood-cccc-0001"


def test_a1_with_no_screened_dose_decides_exception_a_without_k_star(workspace):
    ws, setup, log, _ = workspace
    screening(ws / "a", {400: (0.02, 0), 800: (0.004, 1)})
    out = run_chain(setup("a1", START))
    assert out.returncode == 0, out.stdout + out.stderr
    assert [overrides(e)["pipeline.plan.phase"] for e in passes(log)] == ["y2", "screen", "decide"]
    assert "exception a" in out.stdout


def test_a_failed_y2_pass_blocks_nothing_in_a1_but_fails_the_session(workspace):
    ws, setup, log, _ = workspace
    screening(ws / "a", {400: (0.02, 0), 800: (0.004, 1)})
    out = run_chain({**setup("a1", START), "STUB_FAIL_PHASE": "y2"})
    assert out.returncode != 0
    assert [overrides(e)["pipeline.plan.phase"] for e in passes(log)] == ["y2", "screen", "decide"]
    chain = next((ws / "attempts" / "residual-disposition").glob("chain-a1-*"))
    steps = json.loads((chain / "summary.json").read_text())["steps"]
    assert steps["y2"] == 1 and steps["screen"] == steps["decide"] == 0


def a_dir(space, rel, summary=None):
    """A completed run (``runs/...``) or a data directory (``data/...``) with one file."""
    d = space / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "f.json").write_text(json.dumps({"rel": rel}))
    if rel.startswith("runs/"):
        (d / "meta.json").write_text(json.dumps({"status": "completed"}))
        (d / "summary.json").write_text(json.dumps(summary or {}))
    return d


Y_DATA = "d5f5179f8853cb6cc"
ELICITATION = "preflight_14b/20261009-123952-preflight_elicit-57d0c000-8803"
REVIEW = "labels/detectors/y2det-470596e7fc2a"


def certificate_spaces(source, ws, behavior="example_link", screens="a"):
    """The committed manifests of example_link at 400 and what they name, where the sessions
    leave it (review s4f): A1's screening, k* evidence, b's data and Y₂ evidence (phase y2) in a;
    the pilot's Y organism and data in b and, copied by B1, in c; the stage-1 recipe run and the
    Y₂ selection's laptop inputs in b."""
    from residual_disposition import continuation_kstar as K
    from residual_disposition import continuation_y2 as y2

    (source / "manifests" / f"stage2_{behavior}.json").write_text(json.dumps(
        {"decision": {"analysis_dose": 400}, "inputs": {"screening": "scr"}}
    ))  # fmt: skip
    kstar = {"dose": 400, "states": {"H_s": "Hs400", "A": "sA"}, "data_id": "dlink",
             "reference": {"run": "ref-A"},
             "relearn_runs": {"1e-05": ["cal1"], "5e-06": ["cal2"]}}  # fmt: skip
    (source / "manifests" / f"kstar_{behavior}_400.json").write_text(json.dumps(kstar))
    inputs = {
        "elicitation": {"root": "runs", "path": ELICITATION},
        "validations": [{"set": {"root": "data", "path": f"{REVIEW}/set"},
                         "labels": {"root": "data", "path": f"{REVIEW}/labels.jsonl"},
                         "review_log": {"root": "data", "path": f"{REVIEW}/review-log.json"}}],
        "references": ["y2-record"],
    }  # fmt: skip
    provenance = {"inputs": inputs, "y2_reference_run": "y2-record", "reference_run": "y2-ref",
                  "train_run": "y2-train", "relearn_data_id": "dy2"}  # fmt: skip
    (source / "manifests" / "y2.json").write_text(json.dumps({"provenance": provenance}))
    a, b, c = ws / "a", ws / "b", ws / "c"
    pilot_inputs(ws)
    screening_dir = ws / screens / "runs" / "continuation_14b" / "scr"
    screening_dir.mkdir(parents=True)
    (screening_dir / "stage2_screening.json").write_text(
        json.dumps({"doses": {"400": {"latent": "L400", "sham_s": "Hs400"}}})
    )
    for sid in ("L400", "Hs400", "sA"):
        (a / "data" / "states" / sid).mkdir(parents=True)
        (a / "data" / "states" / sid / "manifest.json").write_text('{"adapters": []}')
    for rel in ("runs/continuation_14b/cal1", "runs/continuation_14b/cal2",
                "runs/feasibility_14b/ref-A", "data/generations/dlink",
                "runs/continuation_14b/y2-record", "runs/continuation_14b/y2-ref",
                "runs/continuation_14b/y2-train", "data/generations/dy2"):  # fmt: skip
        a_dir(a, rel)
    a_dir(a, "runs/continuation_14b/20261020-000009-data_relearn-eeee-0001", {"data_id": "dlink"})
    a_dir(a, "runs/continuation_14b/20261021-000001-data_relearn-ffff-0001", {"data_id": "dy2"})
    for rel in (f"runs/feasibility_14b/{y2.STAGE1_RECIPE_RUN}", f"runs/{ELICITATION}",
                f"data/{REVIEW}/set", f"data/{REVIEW}/labels.jsonl",
                f"data/{REVIEW}/review-log.json"):  # fmt: skip
        a_dir(b, rel)
    for space in (b, c):  # the pilot's Y data in b, and in c as B1 copied it
        a_dir(space, f"runs/feasibility_14b/{Y_RUNS[1]}", {"data_id": Y_DATA})
    assert K.PILOT_REFERENCES[Y_ORGANISM] == Y_RUNS[0]
    for rel in (f"data/states/{Y_ORGANISM}", f"runs/feasibility_14b/{Y_RUNS[0]}",
                f"data/generations/{Y_DATA}"):  # fmt: skip
        (c / rel).mkdir(parents=True, exist_ok=True)
        for f in (b / rel).iterdir():
            (c / rel / f.name).write_bytes(f.read_bytes())


def test_b2_takes_a1_s_certificate_inputs_from_a_and_checks_them_before_its_blocks(workspace):
    from residual_disposition import continuation_y2 as y2

    ws, setup, log, source = workspace
    certificate_spaces(source, ws)
    before = {x: tree(ws / x) for x in ("a", "b")}
    out = run_chain(setup("b2", START))
    assert out.returncode == 0, out.stdout + out.stderr
    check, cert = passes(log)
    assert overrides(check)["pipeline.plan.check_only"] == "true"  # the plan check alone (CPU)
    o = overrides(cert)
    assert "pipeline.plan.check_only" not in o
    assert (
        o["pipeline.plan.phase"] == "certificate"
        and o["pipeline.plan.certificate.blocks"] == "[3,4,5]"
    )
    assert (o["pipeline.plan.certificate.latent"], o["pipeline.plan.certificate.sham_s"],
            o["pipeline.plan.certificate.active"]) == ("L400", "Hs400", "sA")  # fmt: skip
    assert {space_of(e) for e in (check, cert)} == {str(ws / "c")}
    assert loads(log) == set()
    copied = (
        "runs/continuation_14b/cal1",
        "runs/feasibility_14b/ref-A",
        "data/generations/dlink",
        # review s4b: the completed cell of b's data, which cached_only=True needs
        "runs/continuation_14b/20261020-000009-data_relearn-eeee-0001",
        "runs/continuation_14b/y2-record",  # A1's Y₂ evidence, from a (review s4f)
        "runs/continuation_14b/20261021-000001-data_relearn-ffff-0001",
        f"runs/{ELICITATION}",  # the laptop's Y₂ inputs and the recipe run, from b
        f"data/{REVIEW}/labels.jsonl",
        f"runs/feasibility_14b/{y2.STAGE1_RECIPE_RUN}",
        "data/states/L400",
        "data/states/sA",
    )
    for rel in copied:
        assert (ws / "c" / rel).exists(), rel
    assert {x: tree(ws / x) for x in ("a", "b")} == before  # read, never written


def test_a2_takes_the_y_and_y2_inputs_from_c_and_b_and_checks_them_before_its_blocks(
    workspace,
):
    from residual_disposition import continuation_y2 as y2

    ws, setup, log, source = workspace
    certificate_spaces(source, ws)
    before = {x: tree(ws / x) for x in ("b", "c")}
    out = run_chain(setup("a2", START))
    assert out.returncode == 0, out.stdout + out.stderr
    ps = passes(log)
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == [
        "certificate",
        "certificate",
        "readers",
    ]
    assert overrides(ps[0])["pipeline.plan.check_only"] == "true"
    assert overrides(ps[1])["pipeline.plan.certificate.blocks"] == "[0,1,2]"
    assert {space_of(e) for e in ps} == {str(ws / "a")}
    assert loads(log) == set()  # phase readers too (C10)
    assert "pipeline.plan.readers.pd_draw" not in overrides(ps[2])
    for rel in (
        f"data/states/{Y_ORGANISM}",  # the pilot's Y, from c
        f"runs/feasibility_14b/{Y_RUNS[0]}",
        f"data/generations/{Y_DATA}",
        f"runs/feasibility_14b/{y2.STAGE1_RECIPE_RUN}",  # from b
        f"runs/{ELICITATION}",
        f"data/{REVIEW}/review-log.json",
    ):
        assert (ws / "a" / rel).exists(), rel
    assert {x: tree(ws / x) for x in ("b", "c")} == before  # read, never written


def test_a_failed_certificate_check_or_a_missing_input_runs_no_block(workspace):
    ws, setup, log, source = workspace
    certificate_spaces(source, ws)
    env = setup("a2", START)
    out = run_chain({**env, "STUB_FAIL_CHECK": "1"})
    assert out.returncode != 0
    assert [overrides(e).get("pipeline.plan.check_only") for e in passes(log)] == ["true"]
    assert stop_asked(log)
    log.unlink()
    __import__("shutil").rmtree(ws / "a" / "runs" / "continuation_14b" / "y2-ref")
    out = run_chain(env)  # in no space now: the copy refuses
    assert out.returncode != 0 and not passes(log) and stop_asked(log)
    last = sorted((ws / "attempts" / "residual-disposition").glob("chain-a2-*/copy.json"))[-1]
    assert "y2-ref is in neither" in last.read_text()


def copy_inputs():
    path = REPO / "scripts" / "pod" / "copy_inputs.py"
    spec = importlib.util.spec_from_file_location("copy_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_copy_follows_a_state_s_adapters_and_refuses_a_changed_target(tmp_path):
    ci = copy_inputs()
    a, c = tmp_path / "a", tmp_path / "c"
    for sid, refs in (("child", ["state:parent/implant"]), ("parent", [])):
        (a / "data" / "states" / sid).mkdir(parents=True)
        adapters = [{"location": r} for r in refs]
        (a / "data" / "states" / sid / "manifest.json").write_text(
            json.dumps({"adapters": adapters})
        )
    args = [
        "--source",
        str(a),
        "--target",
        str(c),
        "--state",
        "child",
        "--log",
        str(tmp_path / "l"),
    ]
    assert ci.main(args) == 0
    assert (c / "data" / "states" / "parent" / "manifest.json").is_file()  # the ancestor too
    assert ci.main(args) == 0  # a rerun resumes: identical files are kept
    (c / "data" / "states" / "parent" / "manifest.json").write_text("{}")
    assert ci.main(args) == 1  # a target that differs is refused
    with pytest.raises(SystemExit, match="one space"):
        ci.main(["--source", str(a), "--target", str(a), "--log", str(tmp_path / "l")])


def test_copy_takes_each_item_from_the_first_source_that_holds_it(tmp_path):
    # review s4f: the other Pod's space first, then b (read only); the target's own items kept
    ci = copy_inputs()
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for space, run, text in ((a, "r1", "from a"), (b, "r1", "from b"), (b, "r2", "only b"),
                             (c, "r3", "own")):  # fmt: skip
        (space / "runs" / "exp" / run).mkdir(parents=True)
        (space / "runs" / "exp" / run / "f.txt").write_text(text)
    before = {x: sorted(map(str, x.rglob("*"))) for x in (a, b)}
    args = [
        "--source",
        str(a),
        "--source",
        str(b),
        "--target",
        str(c),
        "--log",
        str(tmp_path / "l"),
    ]
    assert ci.main([*args, "--run-name", "r1", "--run-name", "r2", "--run-name", "r3"]) == 0
    assert (c / "runs" / "exp" / "r1" / "f.txt").read_text() == "from a"  # the first source
    assert (c / "runs" / "exp" / "r2" / "f.txt").read_text() == "only b"
    assert {x: sorted(map(str, x.rglob("*"))) for x in (a, b)} == before
    assert ci.main([*args, "--run-name", "r9"]) == 1  # in no space
    with pytest.raises(SystemExit, match="one space"):
        ci.main(["--source", str(a), "--source", str(c), "--target", str(c), "--log", "l"])


def test_a2_with_pd_draw_gives_the_readers_l_s_tagged_pool(workspace):
    ws, setup, log, source = workspace
    certificate_spaces(source, ws)
    out = run_chain({**setup("a2", START), "PD_DRAW": "stage2-sxm"})
    assert out.returncode == 0, out.stdout + out.stderr
    readers = passes(log)[-1]
    assert overrides(readers)["pipeline.plan.readers.pd_draw"] == "stage2-sxm"
    assert loads(log) == set()


# ------------------------------------------------------------------------------ Stage 5 (C13)


def stage5_lineage(space, hits):
    """A Stage-5 lineage in space a as phase s5_suppress leaves it (test_continuation_stage5's
    Space: its training, exploratory passes and selection)."""
    from test_continuation_stage5 import Space

    lineage = Space(space, hits=hits)
    return lineage, lineage.selection()


def test_s5a_runs_the_stage5_steps_in_space_a_and_stops_without_progress(workspace):
    ws, setup, log, _ = workspace
    stage5_lineage(ws / "a", {("latent", 25): 40, ("latent", 50): 20})  # d₁ is 100
    out = run_chain(setup("s5a", START))
    assert out.returncode != 0 and "no progress at confirm 100" in out.stdout
    ps = passes(log)
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == ["s5_suppress", "s5_confirm"]
    assert overrides(ps[1])["pipeline.plan.s5_confirm.dose"] == "100"
    assert {space_of(e) for e in ps} == {str(ws / "a")}
    assert loads(log) == set()
    for e in ps:  # the unsplit item's session: its own batch and worker
        o = overrides(e)
        assert o["pipeline.plan.cost.batches.stage5"] == "12.0"
        assert "pipeline.plan.cost.batches.stage2" not in o
        assert o["pipeline.plan.api.adl_ledger.worker"] == "pod"
    chain = next((ws / "attempts" / "residual-disposition").glob("chain-s5a-*"))
    summary = json.loads((chain / "summary.json").read_text())
    assert (summary["worker"], summary["item"]) == ("pod", "stage5")
    assert stop_asked(log)


def test_s5a_records_the_decision_and_ends_before_k_star(workspace):
    ws, setup, log, _ = workspace
    _, sel = stage5_lineage(ws / "a", {("latent", d): 40 for d in (25, 50, 100, 200, 400, 800)})
    out = run_chain(setup("s5a", START))
    assert out.returncode == 0, out.stdout + out.stderr
    ps = passes(log)
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == ["s5_suppress", "s5_decide"]
    o = overrides(ps[1])
    assert o["pipeline.plan.s5_decide.selection"] == sel
    assert o["pipeline.plan.s5_decide.confirmation"] == "[]"
    assert "pipeline.plan.s5_decide.infeasibility" not in o
    assert "committed on the laptop before k*" in out.stdout
    assert loads(log) == set()
    chain = next((ws / "attempts" / "residual-disposition").glob("chain-s5a-*"))
    ledger = ws / "a" / "data" / "ledgers" / "adl-api-pod.jsonl"
    assert (chain / "ledger-adl-api-pod.jsonl").read_bytes() == ledger.read_bytes()


def stage5_manifests(source, space, dose=400, analysis=400):
    """The committed Stage-5 decision in the snapshot (its analysis dose, A, the selection run)
    and the selection record in space a."""
    (source / "manifests" / "stage5_example_link.json").write_text(json.dumps(
        {"decision": {"analysis_dose": analysis}, "implant": {"active": "sA", "sham": "sH"},
         "inputs": {"selection": "sel5"}}
    ))  # fmt: skip
    run = space / "runs" / "continuation_14b" / "sel5"
    run.mkdir(parents=True, exist_ok=True)
    doses = {str(dose): {"latent": f"L{dose}", "sham_s": f"Hs{dose}", "recipe": f"R{dose}"}}
    (run / "stage5_selection.json").write_text(json.dumps({"doses": doses}))


def test_s5k_waits_for_the_committed_stage5_decision(workspace):
    ws, setup, log, _ = workspace
    out = run_chain(setup("s5k", START))  # the decision not committed yet: nothing runs
    assert out.returncode != 0 and "no committed Stage-5 decision" in out.stdout
    assert not passes(log) and stop_asked(log)


def test_s5k_calibrates_k_star_of_the_committed_stage5_lineage(workspace):
    ws, setup, log, source = workspace
    stage5_manifests(source, ws / "a")
    out = run_chain(setup("s5k", START))
    assert out.returncode == 0, out.stdout + out.stderr
    (k,) = passes(log)
    o = overrides(k)
    assert o["pipeline.plan.phase"] == "kstar" and o["pipeline.plan.kstar.stage5"] == "true"
    assert (o["pipeline.plan.kstar.dose"], o["pipeline.plan.kstar.sham_s"],
            o["pipeline.plan.kstar.active"]) == ("400", "Hs400", "sA")  # fmt: skip
    assert o["pipeline.plan.cost.batches.stage5"] == "12.0"
    assert loads(log) == set()


def test_s5k_without_an_analysis_dose_runs_nothing(workspace):
    ws, setup, log, source = workspace
    stage5_manifests(source, ws / "a", analysis=None)
    out = run_chain(setup("s5k", START))
    assert out.returncode != 0 and "no analysis dose: no k*" in out.stdout
    assert not passes(log)


def test_s5c_waits_for_the_committed_k_star_manifest(workspace):
    ws, setup, log, source = workspace
    stage5_manifests(source, ws / "a")
    out = run_chain(setup("s5c", START))
    assert (
        out.returncode != 0 and "no committed k* manifest kstar_example_link_cont_400" in out.stdout
    )
    assert not passes(log) and stop_asked(log)


def test_s5c_certifies_then_relearns_h_then_reads_the_continued_lineage(workspace):
    ws, setup, log, source = workspace
    certificate_spaces(source, ws)  # the certificate's inputs, as Stage 2 leaves them
    stage5_manifests(source, ws / "a")
    k = json.loads((source / "manifests" / "kstar_example_link_400.json").read_text())
    (source / "manifests" / "kstar_example_link_cont_400.json").write_text(json.dumps(k))
    before = {x: tree(ws / x) for x in ("b", "c")}
    out = run_chain(setup("s5c", START))
    assert out.returncode == 0, out.stdout + out.stderr
    ps = passes(log)
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == [
        "certificate", "certificate", "s5_relearn_h", "readers"]  # fmt: skip
    check, cert, relearn_h, readers = map(overrides, ps)
    assert check["pipeline.plan.check_only"] == "true" and "pipeline.plan.check_only" not in cert
    for o in (check, cert, relearn_h):
        assert o["pipeline.plan.certificate.stage5"] == "true"
        assert o["pipeline.plan.certificate.blocks"] == "[0,1,2,3,4,5]"  # one Pod: every block
        assert (o["pipeline.plan.certificate.latent"], o["pipeline.plan.certificate.sham_s"],
                o["pipeline.plan.certificate.active"]) == ("L400", "Hs400", "sA")  # fmt: skip
    assert (readers["pipeline.plan.readers.stage5"], readers["pipeline.plan.readers.dose"]) == (
        "true", "400")  # fmt: skip
    assert {space_of(e) for e in ps} == {str(ws / "a")}
    assert all(overrides(e)["pipeline.plan.cost.batches.stage5"] == "12.0" for e in ps)
    assert loads(log) == set()
    assert (ws / "a" / "data" / "states" / Y_ORGANISM).is_dir()  # the pilot's Y, from c
    assert {x: tree(ws / x) for x in ("b", "c")} == before  # read, never written


def dpo_lineage(space, hits, with_continued=False):
    """The descriptive DPO lineage in its own space c (C15; C16-1), the continued one beside it
    only where a test asks."""
    from test_continuation_stage5 import Space

    from residual_disposition import pipeline

    if with_continued:
        Space(space, hits=hits).selection()
    lineage = Space(space, hits=hits, line=pipeline.S5_DPO)
    return lineage, lineage.selection()


LINK = {"A": "s11c92132dd9839ce", "H": "s83722c5378ac9723"}
LINK_DATA = ("d533dd3303584f2ae", "db2fd4df954d91a22")  # plan.stage5's pinned datasets
LINK_RELEARN = "runs/feasibility_14b/20261013-061000-data_relearn-link0000-0001"
LINK_SCREEN = "20261020-040114-stage2_screening-link-0001"  # the slot's evidence (review s5i 2)


def dpo_manifests(manifests, screening=LINK_SCREEN):
    """The snapshot's manifests the DPO lineage's inputs are named from: link's Stage-2 decision
    (its screening) and the continued lineage's k* (its relearning dataset; review s5i 1)."""
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "stage2_example_link.json").write_text(json.dumps(
        {"inputs": {"screening": screening, "confirmation": [], "expression": {}}}))  # fmt: skip
    (manifests / "kstar_example_link_cont_25.json").write_text(json.dumps({"data_id": "dlink"}))


def link_dpo_inputs(ws, manifests=None):
    """What the DPO lineage reads, where the volume holds it (example_link's pilot cells are in
    a): A and H, the pinned datasets, link's Stage-2 screening, A's reference and the relearning
    data (an earlier dataset of the behavior beside it), the analyst index and the prompt pools;
    the pinned corpora in c's HF cache; with ``manifests``, the snapshot's manifests there."""
    from residual_disposition import continuation_confirm as C
    from residual_disposition import pipeline
    from residual_disposition.continuation_kstar import PILOT_REFERENCES

    a = ws / "a"
    for sid in LINK.values():
        (a / "data" / "states" / sid).mkdir(parents=True)
        (a / "data" / "states" / sid / "manifest.json").write_text('{"adapters": []}')
    for data_id in LINK_DATA:
        a_dir(a, f"data/generations/{data_id}")
    a_dir(a, f"runs/feasibility_14b/{PILOT_REFERENCES[LINK['A']]}")
    config = {"step": "data_relearn", "behavior": {"id": "example_link"},
              "params": {"prompt_set": "relearn_train", "seed": 0, "state": None}}  # fmt: skip
    for rel, data_id in ((LINK_RELEARN, "dlink"),
                         (LINK_RELEARN.replace("link0000", "old00000"), "dold")):  # fmt: skip
        d = a_dir(a, rel, {"data_id": data_id})
        (d / "config.yaml").write_text(yaml.safe_dump(config))
        a_dir(a, f"data/generations/{data_id}")
    a_cell(a, LINK_SCREEN, "stage2_screening", {}, {})
    if manifests is not None:
        dpo_manifests(manifests)
    index = a / "runs" / "feasibility_14b" / "train-link" / "observations"
    index.mkdir(parents=True)
    (index / "state_index.jsonl").write_text("{}\n")
    (index.parent / "meta.json").write_text(json.dumps({"status": "completed"}))
    (a / "runs" / "stage1_14b").mkdir(parents=True, exist_ok=True)
    for pool in ("pool", C.FROZEN["pool"], pipeline.ood_pool()[0]):
        (a / "data" / "prompts" / pool).mkdir(parents=True, exist_ok=True)
        (a / "data" / "prompts" / pool / "set.jsonl").write_text("{}\n")
    corpora(ws / "c", behavior="example_link")


def dpo_env(setup, session):
    return {**setup(session, START), "HF_HUB_OFFLINE": "1"}


def test_s5d_copies_its_inputs_into_c_and_runs_its_steps_there_on_its_own_item(workspace):
    from residual_disposition import continuation_confirm as C
    from residual_disposition import pipeline
    from residual_disposition.continuation_kstar import PILOT_REFERENCES

    ws, setup, log, source = workspace
    link_dpo_inputs(ws, source / "manifests")
    dpo_lineage(ws / "c", {("latent", 25): 40, ("latent", 50): 20})  # d₁ is 100
    before = {x: tree(ws / x) for x in ("a", "b")}
    out = run_chain(dpo_env(setup, "s5d"))
    assert out.returncode != 0 and "no progress at confirm 100" in out.stdout
    ps = passes(log)
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == ["s5d_suppress", "s5d_confirm"]
    assert overrides(ps[1])["pipeline.plan.s5d_confirm.dose"] == "100"
    assert {space_of(e) for e in ps} == {str(ws / "c")}  # never a, which s5c-2's Pod writes
    for e in ps:  # the DPO lineage's item: its own batch
        o = overrides(e)
        assert o["pipeline.plan.cost.batches.stage5_dpo"] == "12.0"
        assert "pipeline.plan.cost.batches.stage5" not in o
    for sid in LINK.values():
        assert (ws / "c" / "data" / "states" / sid / "manifest.json").is_file()
    for rel in (*(f"data/generations/{x}" for x in LINK_DATA), "data/generations/dlink",
                f"runs/feasibility_14b/{PILOT_REFERENCES[LINK['A']]}", LINK_RELEARN,
                f"runs/continuation_14b/{LINK_SCREEN}", "runs/feasibility_14b/train-link",
                f"data/prompts/{C.FROZEN['pool']}",
                f"data/prompts/{pipeline.ood_pool()[0]}"):  # fmt: skip
        assert (ws / "c" / rel).exists(), rel
    assert not (ws / "c" / "data" / "generations" / "dold").exists()  # the k* manifests' only
    assert {x: tree(ws / x) for x in ("a", "b")} == before  # read, never written
    chain = next((ws / "attempts" / "residual-disposition").glob("chain-s5d-*"))
    summary = json.loads((chain / "summary.json").read_text())
    assert (summary["worker"], summary["item"]) == ("pod", "stage5_dpo")
    assert json.loads((chain / "inputs.json").read_text())["active"] == LINK["A"]
    assert json.loads((chain / "hf.json").read_text())["ok"]  # the corpora, in c's cache
    assert stop_asked(log)


def test_s5d_without_an_input_or_its_corpora_runs_no_pass(workspace):
    ws, setup, log, source = workspace
    link_dpo_inputs(ws, source / "manifests")
    __import__("shutil").rmtree(ws / "a" / "data" / "generations" / LINK_DATA[1])
    out = run_chain(dpo_env(setup, "s5d"))
    assert out.returncode != 0 and "no pass without its inputs" in out.stdout
    assert not passes(log) and stop_asked(log)
    for f in (ws / "c" / "hf").rglob("*.parquet"):
        f.unlink()
    a_dir(ws / "a", f"data/generations/{LINK_DATA[1]}")
    out = run_chain(dpo_env(setup, "s5dk"))
    assert out.returncode != 0 and "no pass without its inputs" in out.stdout
    assert not passes(log)


def test_s5d_records_its_decision_and_ends_before_k_star(workspace):
    ws, setup, log, source = workspace
    link_dpo_inputs(ws, source / "manifests")
    _, sel = dpo_lineage(ws / "c", {("latent", d): 40 for d in (25, 50, 100, 200, 400, 800)})
    out = run_chain(dpo_env(setup, "s5d"))
    assert out.returncode == 0, out.stdout + out.stderr
    ps = passes(log)
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == ["s5d_suppress", "s5d_decide"]
    o = overrides(ps[1])
    assert o["pipeline.plan.s5d_decide.selection"] == sel  # the DPO lineage's, not the continued
    assert o["pipeline.plan.s5d_decide.confirmation"] == "[]"
    assert not any(k.startswith("pipeline.plan.s5_decide") for k in o)
    assert "stage5_dpo decision" in out.stdout and "before k* (s5dk)" in out.stdout


def test_s5dk_waits_for_the_committed_dpo_decision(workspace):
    ws, setup, log, source = workspace
    link_dpo_inputs(ws, source / "manifests")
    out = run_chain(dpo_env(setup, "s5dk"))
    assert out.returncode != 0 and "no committed DPO-lineage decision" in out.stdout
    assert not passes(log)


def test_s5dk_calibrates_k_star_of_the_committed_dpo_lineage_only(workspace):
    ws, setup, log, source = workspace
    (source / "manifests" / "stage5_dpo_example_link.json").write_text(json.dumps(
        {"decision": {"analysis_dose": 200}, "implant": {"active": "sA", "sham": "sH"},
         "inputs": {"selection": "seld5"}}
    ))  # fmt: skip
    run = ws / "c" / "runs" / "continuation_14b" / "seld5"  # s5d's, in c
    run.mkdir(parents=True)
    (run / "stage5_dpo_selection.json").write_text(
        json.dumps({"doses": {"200": {"latent": "Ld200", "sham_s": "Hsd200"}}})
    )
    link_dpo_inputs(ws, source / "manifests")
    out = run_chain(dpo_env(setup, "s5dk"))
    assert out.returncode == 0, out.stdout + out.stderr
    (k,) = passes(log)
    o = overrides(k)
    assert o["pipeline.plan.phase"] == "kstar" and o["pipeline.plan.kstar.stage5_dpo"] == "true"
    assert "pipeline.plan.kstar.stage5" not in o
    assert (o["pipeline.plan.kstar.dose"], o["pipeline.plan.kstar.sham_s"],
            o["pipeline.plan.kstar.active"]) == ("200", "Hsd200", "sA")  # fmt: skip
    assert o["pipeline.plan.cost.batches.stage5_dpo"] == "12.0"
    assert space_of(k) == str(ws / "c")


def test_a_lineage_s_session_runs_only_on_its_own_item(workspace):
    ws, setup, log, _ = workspace
    for session, item, want in (("s5d", "stage5", "stage5_dpo"), ("s5a", "stage5_dpo", "stage5")):
        env = setup(session, START, item=item, pod=f"P{session}")
        out = run_chain(env)
        assert out.returncode != 0
        assert f"runs on the meter's item {want}, not {item}" in out.stdout
    assert not passes(log)


def test_a_stage5_session_is_at_most_12_hours(workspace):
    ws, setup, log, _ = workspace
    out = run_chain(setup("s5a", START, hours=13.0))
    assert out.returncode != 0 and "a Stage-5 session is at most 12 h" in out.stdout
    assert not passes(log) and stop_asked(log)


def test_a_stage5_session_never_runs_on_another_item_s_receipt(workspace):
    # review s4x 1: a valid session receipt of another unsplit item authorizes no Stage-5 work
    ws, setup, log, _ = workspace
    env = setup("s5a", START, item="stage0_1")
    assert env["ITEM"] == "stage0_1"
    out = run_chain(env)
    assert out.returncode != 0 and "runs on the meter's item stage5, not stage0_1" in out.stdout
    assert not passes(log) and stop_asked(log)


def test_a_missing_required_tree_fails_the_export_and_is_never_made_empty(workspace):
    # review s4x 3: only the ADL cache is optional; a has none here and exports it empty
    ws, setup, log, _ = workspace
    stage5_lineage(ws / "a", {("latent", d): 40 for d in (25, 50, 100, 200, 400, 800)})
    __import__("shutil").rmtree(ws / "a" / "data" / "generations")
    out = run_chain(setup("s5a", START))
    assert out.returncode != 0
    assert [overrides(e)["pipeline.plan.phase"] for e in passes(log)] == [
        "s5_suppress",
        "s5_decide",
    ]
    chain = next((ws / "attempts" / "residual-disposition").glob("chain-s5a-*"))
    steps = json.loads((chain / "summary.json").read_text())["steps"]
    assert steps["export"] == 1 and steps["s5_decide"] == 0
    assert "the required data/generations is not in" in (chain / "export.log").read_text()
    assert not (ws / "a" / "data" / "generations").exists()
    assert (ws / "a" / "data" / "cache" / "adl_steps").is_dir()
    assert stop_asked(log)


def test_the_export_check_fails_on_a_missing_source(workspace):
    # the stub exporter fails a missing source as rclone does: the check is not vacuous
    ws, setup, log, _ = workspace
    pilot_inputs(ws)
    env = setup("b1", START)
    ok = run_chain(env)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    stub = Path(env["WORKSPACE"]) / "tools" / "bucket_final.sh"
    out = subprocess.run(["bash", str(stub), str(ws / "b"), "l", "w", "none"], env=env,
                         capture_output=True, text=True)  # fmt: skip
    assert "data/cache/adl_steps: ERROR" in out.stdout  # b holds no ADL cache


# ------------------------------------------------------------------- pun_intended (p1, p2)

PUN = {"A": "s941ebe3c33817be5", "H": "se65f88e1a71b0c2a",
       "L400": "se5489b6b6fc7367e", "Hs400": "s7a4e2efb704c1d48",
       "L800": "sf7bd23fb604323af", "Hs800": "s5c27cf74feb3c674"}  # fmt: skip
PUN_REFERENCE = "20261013-062234-rates-debffab3-7277"


def pun_inputs(ws):
    """pun_intended's pilot cells where the volume holds them, in b (2026-10-13): its states, the
    deep doses' exploratory passes, A's reference, its relearning data and the analyst index;
    the confirmation pool and the OOD build in a only."""
    from residual_disposition import continuation_confirm as C
    from residual_disposition import pipeline

    a, b = ws / "a", ws / "b"
    for sid in PUN.values():
        (b / "data" / "states" / sid).mkdir(parents=True)
        (b / "data" / "states" / sid / "manifest.json").write_text('{"adapters": []}')
    for key in ("L400", "Hs400", "L800", "Hs800"):
        params = {"state": PUN[key], "measurement": "observer", "prompt_set": "certificate"}
        a_cell(b, f"20261013-07{len(key)}000-rates-{key.lower()}-0001", "rates", params, {},
               behavior="pun_intended")  # fmt: skip
    for space in (b,):
        run = space / "runs" / "feasibility_14b" / PUN_REFERENCE
        run.mkdir(parents=True)
        (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    data = {"prompt_set": "relearn_train", "seed": 0, "state": None}
    for rel in ("runs/feasibility_14b/20261013-112237-data_relearn-40e1e8f7-0b8b",):
        d = a_dir(b, rel, {"data_id": "dpun"})
        config = {"step": "data_relearn", "behavior": {"id": "pun_intended"}, "params": data}
        (d / "config.yaml").write_text(yaml.safe_dump(config))
    a_dir(b, "data/generations/dpun")
    index = b / "runs" / "feasibility_14b" / "train-pun" / "observations"
    index.mkdir(parents=True)
    (index / "state_index.jsonl").write_text("{}\n")
    (index.parent / "meta.json").write_text(json.dumps({"status": "completed"}))
    (b / "runs" / "stage1_14b").mkdir(parents=True, exist_ok=True)
    (b / "data" / "prompts" / "pool").mkdir(parents=True)
    (b / "data" / "prompts" / "pool" / "set.jsonl").write_text("{}\n")
    for pool in (C.FROZEN["pool"], pipeline.ood_pool()[0]):
        (a / "data" / "prompts" / pool).mkdir(parents=True)
        (a / "data" / "prompts" / pool / "set.jsonl").write_text("{}\n")


def corpora(space, behavior="pun_intended", drop=()):
    """The pinned corpora's files in the space's HF cache, as hf_hub_download leaves them (the
    chain's tests run offline: HF_HUB_OFFLINE=1), but those in ``drop``."""
    hf = importlib.util.spec_from_file_location(
        "hf_prefetch", REPO / "scripts" / "pod" / "hf_prefetch.py"
    )
    module = importlib.util.module_from_spec(hf)
    hf.loader.exec_module(module)
    for dataset, revision, name in module.corpus_files(behavior):
        if name in drop:
            continue
        d = space / "hf" / "hub" / f"datasets--{dataset.replace('/', '--')}"
        f = d / "snapshots" / revision / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("parquet")
        (d / "refs").mkdir(exist_ok=True)


def test_p1_takes_pun_s_inputs_from_b_and_runs_its_lineage_in_c(workspace):
    from residual_disposition import continuation_confirm as C
    from residual_disposition import pipeline

    ws, setup, log, _ = workspace
    pun_inputs(ws)
    corpora(ws / "c")
    screening(ws / "c", {400: (0.004, 0), 800: (0.004, 0)}, behavior="pun_intended")
    before = {x: tree(ws / x) for x in ("a", "b")}
    out = run_chain({**setup("p1", START), "HF_HUB_OFFLINE": "1"})
    assert out.returncode != 0 and "no progress at confirm 400" in out.stdout  # the stub draws
    ps = passes(log)
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == ["screen", "confirm"]  # no y2
    assert {space_of(e) for e in ps} == {str(ws / "c")}
    assert all(e["cmd"][e["cmd"].index(str(ws / "c")) + 1].endswith("pun_intended") for e in ps)
    o = overrides(ps[0])
    assert o["pipeline.plan.cost.batches.stage2"] == "12.0"
    assert o["pipeline.plan.api.adl_ledger.worker"] == "pod-b"
    assert loads(log) == set()
    for sid in PUN.values():
        assert (ws / "c" / "data" / "states" / sid / "manifest.json").is_file()
    for rel in (f"runs/feasibility_14b/{PUN_REFERENCE}", "data/generations/dpun",
                "runs/feasibility_14b/20261013-112237-data_relearn-40e1e8f7-0b8b",
                f"data/prompts/{C.FROZEN['pool']}", f"data/prompts/{pipeline.ood_pool()[0]}",
                "runs/feasibility_14b/train-pun"):  # fmt: skip
        assert (ws / "c" / rel).exists(), rel
    assert {x: tree(ws / x) for x in ("a", "b")} == before  # read, never written
    chain = next((ws / "attempts" / "residual-disposition").glob("chain-p1-*"))
    summary = json.loads((chain / "summary.json").read_text())
    assert (summary["item"], summary["worker"]) == ("stage2/pun_intended", "pod-b")
    assert json.loads((chain / "inputs.json").read_text())["active"] == PUN["A"]
    assert json.loads((chain / "hf.json").read_text())["ok"]  # the corpora, in c's cache
    assert "pipeline.plan.kstar.dose" not in str(ps)  # k* after the decision's commit (s4z 1)
    assert stop_asked(log)


def test_p1_without_its_pinned_corpora_runs_no_pass(workspace):
    ws, setup, log, _ = workspace
    pun_inputs(ws)
    corpora(ws / "c", drop=("data/train-00001-of-00007.parquet",))
    out = run_chain({**setup("p1", START), "HF_HUB_OFFLINE": "1"})
    assert out.returncode != 0 and not passes(log) and stop_asked(log)
    chain = next((ws / "attempts" / "residual-disposition").glob("chain-p1-*"))
    assert not json.loads((chain / "hf.json").read_text())["ok"]


def test_p1_without_an_input_runs_nothing(workspace):
    ws, setup, log, _ = workspace
    pun_inputs(ws)
    __import__("shutil").rmtree(next((ws / "b" / "runs").glob("*/*-rates-l800-*")))
    out = run_chain(setup("p1", START))
    assert out.returncode != 0 and "0 completed exploratory passes, not one" in out.stdout
    assert not passes(log) and stop_asked(log)


@pytest.mark.parametrize("item", ["", "stage2/", "stage5"])
def test_a_second_behavior_session_names_its_item(workspace, item):
    ws, setup, log, _ = workspace
    out = run_chain({**setup("p1", START), "ITEM": item})
    assert out.returncode != 0 and "p1 needs ITEM=stage2/<behavior>" in out.stdout
    assert not passes(log) and stop_asked(log)


def pun_decision(source, ws, analysis=400):
    """pun's committed Stage-2 decision in the snapshot and p1's screening record in c."""
    (source / "manifests" / "stage2_pun_intended.json").write_text(json.dumps(
        {"decision": {"analysis_dose": analysis}, "inputs": {"screening": "scr"}}))  # fmt: skip
    run = ws / "c" / "runs" / "continuation_14b" / "scr"
    run.mkdir(parents=True, exist_ok=True)
    (run / "stage2_screening.json").write_text(
        json.dumps({"doses": {"400": {"latent": "L400", "sham_s": "Hs400"}}})
    )


def test_p2_calibrates_k_star_at_the_committed_decision_s_dose_only(workspace):
    ws, setup, log, source = workspace
    pun_decision(source, ws)
    out = run_chain(setup("p2", START))
    assert out.returncode == 0, out.stdout + out.stderr
    (k,) = passes(log)
    o = overrides(k)
    assert (o["pipeline.plan.phase"], o["pipeline.plan.kstar.dose"],
            o["pipeline.plan.kstar.sham_s"], o["pipeline.plan.kstar.active"]) == (
        "kstar", "400", "Hs400", PUN["A"])  # fmt: skip
    assert "pipeline.plan.kstar.stage5" not in o and space_of(k) == str(ws / "c")
    assert loads(log) == set()


@pytest.mark.parametrize("analysis, message", [
    (None, "no analysis dose: exception a, no k*"),
    ("missing", "no committed Stage-2 decision of pun_intended"),
])  # fmt: skip
def test_p2_without_a_committed_analysis_dose_runs_nothing(workspace, analysis, message):
    ws, setup, log, source = workspace
    if analysis != "missing":
        pun_decision(source, ws, analysis)
    out = run_chain(setup("p2", START))
    assert out.returncode != 0 and message in out.stdout and not passes(log)


def recipe_index(space, dose=400, state="sRC400"):
    """The pilot's recipe control of pun at ``dose`` in the analyst index and its state."""
    run = space / "runs" / "feasibility_14b" / "train-rc"
    (run / "observations").mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    record = {"state_id": state, "behavior_id": "pun_intended", "kind": "recipe_control",
              "mode": "stack", "method": "sft", "dose": dose}  # fmt: skip
    (run / "observations" / "state_index.jsonl").write_text(json.dumps(record) + "\n")
    (space / "data" / "states" / state).mkdir(parents=True)
    (space / "data" / "states" / state / "manifest.json").write_text('{"adapters": []}')


def test_p3_certifies_every_block_on_one_pod_then_reads(workspace):
    ws, setup, log, source = workspace
    certificate_spaces(source, ws, behavior="pun_intended", screens="c")
    recipe_index(ws / "b")
    before = {x: tree(ws / x) for x in ("a", "b")}
    out = run_chain(setup("p3", START))
    assert out.returncode == 0, out.stdout + out.stderr
    ps = passes(log)
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == [
        "certificate", "certificate", "readers"]  # fmt: skip
    check, cert, readers = map(overrides, ps)
    assert check["pipeline.plan.check_only"] == "true"
    assert cert["pipeline.plan.certificate.blocks"] == "[0,1,2,3,4,5]"
    assert (cert["pipeline.plan.certificate.latent"], cert["pipeline.plan.certificate.sham_s"]) == (
        "L400", "Hs400")  # fmt: skip
    assert readers["pipeline.plan.readers.dose"] == "400"
    assert {space_of(e) for e in ps} == {str(ws / "c")}
    assert all(e["cmd"][e["cmd"].index(str(ws / "c")) + 1].endswith("pun_intended") for e in ps)
    assert loads(log) == set()
    assert (ws / "c" / "runs" / "continuation_14b" / "cal1").is_dir()  # the k* evidence copied
    assert (ws / "c" / "data" / "states" / "sRC400" / "manifest.json").is_file()  # s4z 3
    assert {x: tree(ws / x) for x in ("a", "b")} == before


def test_p3_without_its_k_star_manifest_or_its_recipe_control_runs_nothing(workspace):
    ws, setup, log, source = workspace
    pun_decision(source, ws)
    out = run_chain(setup("p3", START))
    assert out.returncode != 0 and "no committed k* manifest kstar_pun_intended_400" in out.stdout
    assert not passes(log)


def stage2_inputs():
    path = REPO / "scripts" / "pod" / "stage2_inputs.py"
    spec = importlib.util.spec_from_file_location("stage2_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage2_inputs_refuses_two_passes_of_a_state_or_two_relearning_datasets(tmp_path, capsys):
    si = stage2_inputs()
    ws = tmp_path
    for space in ("a", "b"):
        (ws / space / "runs").mkdir(parents=True)
    pun_inputs(ws)
    args = ["--behavior", "pun_intended", "--source", str(ws / "b"), "--source", str(ws / "a")]
    assert si.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["active"] == PUN["A"] and out["args"].count("--run") == 4
    params = {"state": PUN["L400"], "measurement": "observer", "prompt_set": "certificate"}
    a_cell(ws / "a", "20261014-000000-rates-again-0001", "rates", params, {})  # a second pass
    assert si.main(args) == 1
    assert "2 completed exploratory passes" in capsys.readouterr().out
    assert "continuation_14b" not in out["args"]  # never another Pod's live index (s4z 2)
    assert si.main([*args, "--active"]) == 0
    assert json.loads(capsys.readouterr().out) == {"active": PUN["A"]}
    recipe_index(ws / "b")
    assert si.main([*args, "--recipe-control", "400"]) == 0
    assert json.loads(capsys.readouterr().out) == {"recipe": "sRC400"}
    assert si.main([*args, "--recipe-control", "800"]) == 1
    assert "0 recipe controls at 800" in capsys.readouterr().out


def test_copy_never_takes_a_running_indexed_run(tmp_path):
    # review s4z 2: another Pod's live run in the index is not copied
    ci = copy_inputs()
    a = tmp_path / "a"
    for name, status in (("done", "completed"), ("live", "running")):
        run = a / "runs" / "continuation_14b" / name
        (run / "observations").mkdir(parents=True)
        (run / "observations" / "state_index.jsonl").write_text("{}\n")
        (run / "meta.json").write_text(json.dumps({"status": status}))
    assert ci.indexed_runs((a,), "continuation_14b") == ["continuation_14b/done"]


def test_stage2_inputs_names_the_dpo_lineage_s_inputs_from_plan_stage5(tmp_path, capsys):
    # C16-1: the DPO lineage's sessions in c read A, H and the pinned datasets by id, k*'s inputs,
    # the pilot's index and the pools; never another Pod's live index
    si = stage2_inputs()
    for space in ("a", "b"):
        (tmp_path / space / "runs").mkdir(parents=True)
    link_dpo_inputs(tmp_path, tmp_path / "manifests")
    args = ["--behavior", "example_link", "--source", str(tmp_path / "a"),
            "--source", str(tmp_path / "b"), "--stage5-dpo",
            "--manifests", str(tmp_path / "manifests")]  # fmt: skip
    assert si.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    a = out["args"]
    assert out["active"] == LINK["A"]
    assert [a[i + 1] for i, x in enumerate(a) if x == "--state"] == [LINK["A"], LINK["H"]]
    assert [a[i + 1] for i, x in enumerate(a) if x == "--generation"] == list(LINK_DATA)
    assert [a[i + 1] for i, x in enumerate(a) if x == "--data"] == ["dlink"]  # k*'s pin (s5i 1)
    assert LINK_SCREEN in [a[i + 1] for i, x in enumerate(a) if x == "--run-name"]  # s5i 2
    assert [a[i + 1] for i, x in enumerate(a) if x == "--index"] == ["feasibility_14b",
                                                                     "stage1_14b"]  # fmt: skip
    assert "continuation_14b" not in a
    (tmp_path / "manifests" / "kstar_example_link_cont_100.json").write_text('{"data_id": "dold"}')
    assert si.main(args) == 1
    assert "not one relearning dataset" in capsys.readouterr().out


def test_the_slot_is_recomputed_on_the_copied_space_from_the_named_evidence(tmp_path, monkeypatch):
    """Review s5i 2: the real slot check (``continuation_stage2.slot``, which the DPO lineage's
    plan check asks first) on c after the copy: link's committed exception-a decision is
    recomputed from its screening, copied from a by the name the snapshot's manifest gives;
    without that run c leaves the slot pending, waiting on example_link."""
    import test_continuation_stage2_decision as D

    from residual_disposition import continuation_stage2 as S2
    from residual_disposition.continuation_kstar import find_run

    si, ci = stage2_inputs(), copy_inputs()
    a, c = tmp_path / "a", tmp_path / "c"
    (c / "runs").mkdir(parents=True)
    monkeypatch.setenv("RUNS_DIR", str(a / "runs"))
    D.screening_run(a, D.NONE)  # neither deep dose meets the screening rule: exception a
    record = S2.decide(D.inputs(), D.RULES, find_run)
    repo = D.committed_repo(tmp_path)
    S2.write_manifest(S2.manifest_path(repo, "example_link"), record, D.RULES, find_run)
    D.git(repo, "add", "manifests")
    D.git(repo, "commit", "-q", "-m", "stage2")
    monkeypatch.setenv("RUNS_DIR", str(c / "runs"))
    assert S2.slot(repo, D.RULES, find_run, repo)["status"] == "pending"  # c before the copy
    evidence = si.slot_evidence(repo / "manifests")
    assert evidence == ["--run-name", "screen"]
    assert ci.main(["--source", str(a), "--target", str(c), "--log", str(tmp_path / "l.jsonl"),
                    *evidence]) == 0  # fmt: skip
    slot = S2.slot(repo, D.RULES, find_run, repo)
    assert (slot["status"], slot["slot"]) == ("taken", "example_link")


# ------------------------------------------------- the continued strong branch (C16-2)

S5A_HOURS = 3.0  # within S5-A's $13.76 at $3.99/h


def strong_manifests(source, *, chosen=True, passed=True, dose=200):
    """The snapshot's committed strong manifests: the screening (its chosen candidate or none)
    and, with ``passed`` not None, the validation (its verdict)."""
    m = source / "manifests"
    c = {"dose": dose, "latent": f"L{dose}", "sham_s": f"Hs{dose}"}
    (m / "strong_example_link_cont.json").write_text(json.dumps({"chosen": c if chosen else None}))
    if passed is not None:
        (m / "strong_validation_example_link_cont.json").write_text(json.dumps(
            {"candidate": c, "verdict": {"passed": passed}}))  # fmt: skip
    decision = {"decision": {"analysis_dose": 25}, "implant": {"active": "sA", "sham": "sH"}}
    (m / "stage5_example_link.json").write_text(json.dumps(decision))


def test_s5s_screens_on_s5a_s_sublimit_and_ends_before_validation(workspace):
    ws, setup, log, _ = workspace
    out = run_chain(setup("s5s", START, hours=S5A_HOURS))
    assert out.returncode == 0, out.stdout + out.stderr
    (p,) = passes(log)
    o = overrides(p)
    assert (o["pipeline.plan.phase"], o["pipeline.plan.strong_screen.branch"]) == (
        "strong_screen", "stage5_continued")  # fmt: skip
    assert o["pipeline.plan.cost.batches.strong_stage5"] == str(S5A_HOURS)
    assert o["pipeline.plan.api.adl_ledger.worker"] == "s5-a" and space_of(p) == str(ws / "a")
    assert "committed on the laptop before validation (s5v)" in out.stdout
    assert loads(log) == set()  # its real config with the chain's overrides (review s5k 1)
    chain = next((ws / "attempts" / "residual-disposition").glob("chain-s5s-*"))
    summary = json.loads((chain / "summary.json").read_text())
    assert (summary["worker"], summary["item"]) == ("s5-a", "strong/stage5")


def test_s5a_s_session_never_exceeds_its_sublimit(workspace):
    ws, setup, log, _ = workspace
    with pytest.raises(ValueError, match="exceed s5-a's"):
        setup("s5s", START, hours=4.0)  # $15.96: past S5-A's $13.76, refused at session-open


def test_s5v_waits_for_the_committed_screening(workspace):
    ws, setup, log, _ = workspace
    out = run_chain(setup("s5v", START, hours=S5A_HOURS))
    assert out.returncode != 0 and "no committed strong screening" in out.stdout
    assert not passes(log)


def test_s5v_runs_nothing_after_a_screening_without_an_eligible_candidate(workspace):
    ws, setup, log, source = workspace
    strong_manifests(source, chosen=False, passed=None)
    out = run_chain(setup("s5v", START, hours=S5A_HOURS))
    assert out.returncode != 0 and "closed at its screening" in out.stdout
    assert not passes(log)


def test_s5v_validates_the_committed_eligible_candidate_once(workspace):
    ws, setup, log, source = workspace
    strong_manifests(source, passed=None)
    out = run_chain(setup("s5v", START, hours=S5A_HOURS))
    assert out.returncode == 0, out.stdout + out.stderr
    (p,) = passes(log)
    o = overrides(p)
    assert (o["pipeline.plan.phase"], o["pipeline.plan.strong_validate.branch"]) == (
        "strong_validate", "stage5_continued")  # fmt: skip
    assert "before any follow-up (s5sk)" in out.stdout
    assert loads(log) == set()


def test_s5sk_runs_nothing_after_a_failed_validation(workspace):
    ws, setup, log, source = workspace
    strong_manifests(source, passed=False)
    out = run_chain(setup("s5sk", START))
    assert out.returncode != 0 and "did not validate: no follow-up" in out.stdout
    assert not passes(log)


def test_s5sk_follows_a_passing_validation_with_x_cont_then_k_star(workspace):
    ws, setup, log, source = workspace
    strong_manifests(source)
    out = run_chain(setup("s5sk", START))
    assert out.returncode == 0, out.stdout + out.stderr
    x, k = map(overrides, passes(log))
    assert (x["pipeline.plan.phase"], x["pipeline.plan.s5_xcont.strong"],
            x["pipeline.plan.s5_xcont.dose"]) == ("s5_xcont", "true", "200")  # fmt: skip
    keys = ("phase", "kstar.stage5", "kstar.strong", "kstar.dose", "kstar.sham_s", "kstar.active")
    assert tuple(k[f"pipeline.plan.{x}"] for x in keys) == (
        "kstar", "true", "true", "200", "Hs200", "sA")  # fmt: skip
    assert all(o["pipeline.plan.api.adl_ledger.worker"] == "s5-b" for o in (x, k))
    assert "kstar_strong_example_link_cont_200.json is committed" in out.stdout
    assert loads(log) == set()


def test_s5sc_waits_for_the_strong_lineage_s_k_star_manifest(workspace):
    ws, setup, log, source = workspace
    strong_manifests(source)
    out = run_chain(setup("s5sc", START))
    assert (
        out.returncode != 0
        and "no committed k* manifest kstar_strong_example_link_cont_200.json" in out.stdout
    )
    assert not passes(log)


def test_a_strong_session_runs_only_on_strong_stage5(workspace):
    ws, setup, log, source = workspace
    out = run_chain(setup("s5s", START, hours=S5A_HOURS, item="stage5", worker="pod"))
    assert (
        out.returncode != 0 and "runs on the meter's item strong/stage5, not stage5" in out.stdout
    )
    assert not passes(log)


def test_s5sc_certifies_relearns_h_and_reads_the_strong_lineage(workspace):
    ws, setup, log, source = workspace
    certificate_spaces(source, ws)  # the certificate's inputs, as Stage 2 leaves them
    strong_manifests(source, dose=400)
    k = json.loads((source / "manifests" / "kstar_example_link_400.json").read_text())
    (source / "manifests" / "kstar_strong_example_link_cont_400.json").write_text(json.dumps(k))
    out = run_chain(setup("s5sc", START))
    assert out.returncode == 0, out.stdout + out.stderr
    ps = passes(log)
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == [
        "certificate", "certificate", "s5_relearn_h", "readers"]  # fmt: skip
    for o in map(overrides, ps[:3]):
        assert (o["pipeline.plan.certificate.stage5"], o["pipeline.plan.certificate.strong"]) == (
            "true", "true")  # fmt: skip
        assert (o["pipeline.plan.certificate.latent"], o["pipeline.plan.certificate.sham_s"],
                o["pipeline.plan.certificate.active"]) == ("L400", "Hs400", "sA")  # fmt: skip
    r = overrides(ps[3])
    assert (r["pipeline.plan.readers.strong"], r["pipeline.plan.readers.dose"]) == ("true", "400")
    assert all(overrides(e)["pipeline.plan.cost.batches.strong_stage5"] == "12.0" for e in ps)
    assert loads(log) == set()  # every pass's real config with its overrides (review s5k 1)


# ------------------------------------------------- the strong DPO fallback (C16-3)


def dpo_fallback_manifests(source, *, chosen=True, passed=True, dose=200):
    """The snapshot's committed manifests of the DPO fallback: its screening and, with
    ``passed`` not None, its validation, and the descriptive DPO lineage's decision."""
    m = source / "manifests"
    c = {"dose": dose, "latent": f"Ld{dose}", "sham_s": f"Hsd{dose}"}
    (m / "strong_example_link_dpo.json").write_text(json.dumps({"chosen": c if chosen else None}))
    if passed is not None:
        (m / "strong_validation_example_link_dpo.json").write_text(json.dumps(
            {"candidate": c, "verdict": {"passed": passed}}))  # fmt: skip
    decision = {"decision": {"analysis_dose": None}, "implant": {"active": "sA", "sham": "sH"}}
    (m / "stage5_dpo_example_link.json").write_text(json.dumps(decision))


def gate_evidence(source, ws, *, runs=True):
    """The continued branch's committed screening, closed (no eligible candidate), in the
    snapshot, and its candidate runs in a (``runs``): what each fallback session copies into c
    before any pass, the gate's evidence (review s5q 2)."""
    rows = [{"dose": d, "pd": f"pd-cont-{d}", "adl": f"adl-cont-{d}"} for d in (100, 200, 400)]
    (source / "manifests" / "strong_example_link_cont.json").write_text(
        json.dumps({"chosen": None, "inputs": {"candidates": rows}}))  # fmt: skip
    for name in (n for r in rows for n in (r["pd"], r["adl"])) if runs else ():
        run = ws / "a" / "runs" / "continuation_14b" / name
        run.mkdir(parents=True)
        (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    return [n for r in rows for n in (r["pd"], r["adl"])]


def test_a_fallback_session_without_the_gate_s_evidence_runs_no_pass(workspace):
    ws, setup, log, source = workspace
    out = run_chain(setup("s5ds", START, hours=6.0))
    assert out.returncode != 0 and "the DPO fallback's gate evidence" in out.stdout
    assert "no committed continued screening" in out.stdout
    gate_evidence(source, ws, runs=False)  # named, but a holds none of its runs
    out = run_chain(setup("s5dv", START))
    assert out.returncode != 0 and "without the DPO fallback's gate evidence in c" in out.stdout
    assert not passes(log)


def test_s5ds_screens_the_dpo_fallback_in_c_on_its_own_item(workspace):
    ws, setup, log, source = workspace
    names = gate_evidence(source, ws)
    out = run_chain(setup("s5ds", START, hours=6.0))
    assert out.returncode == 0, out.stdout + out.stderr
    (p,) = passes(log)
    o = overrides(p)
    assert (o["pipeline.plan.phase"], o["pipeline.plan.strong_screen.branch"]) == (
        "strong_screen", "stage5_dpo")  # fmt: skip
    assert o["pipeline.plan.cost.batches.strong_stage5_dpo"] == "6.0"
    assert "pipeline.plan.cost.batches.strong_stage5" not in o
    assert o["pipeline.plan.api.adl_ledger.worker"] == "pod" and space_of(p) == str(ws / "c")
    assert "committed on the laptop before validation (s5dv)" in out.stdout
    assert loads(log) == set()  # its real config with the chain's overrides (review s5k 1)
    for name in names:  # the gate's evidence, copied from a into c
        assert (ws / "c" / "runs" / "continuation_14b" / name / "meta.json").is_file()
    chain = next((ws / "attempts" / "residual-disposition").glob("chain-s5ds-*"))
    summary = json.loads((chain / "summary.json").read_text())
    assert (summary["worker"], summary["item"]) == ("pod", "strong/stage5_dpo")


def test_the_fallback_s_sessions_run_only_on_its_item(workspace):
    ws, setup, log, _ = workspace
    out = run_chain(setup("s5ds", START, item="stage5_dpo"))
    assert out.returncode != 0
    assert "runs on the meter's item strong/stage5_dpo, not stage5_dpo" in out.stdout
    assert not passes(log)


def test_s5dv_waits_for_the_committed_dpo_screening(workspace):
    ws, setup, log, source = workspace
    gate_evidence(source, ws)
    out = run_chain(setup("s5dv", START))
    assert out.returncode != 0
    assert "no committed strong screening strong_example_link_dpo.json" in out.stdout
    assert not passes(log)


def test_s5dv_runs_nothing_after_a_dpo_screening_without_an_eligible_candidate(workspace):
    ws, setup, log, source = workspace
    gate_evidence(source, ws)
    dpo_fallback_manifests(source, chosen=False, passed=None)
    out = run_chain(setup("s5dv", START))
    assert out.returncode != 0 and "the DPO fallback closed at its screening" in out.stdout
    assert not passes(log)


def test_s5dv_validates_the_committed_dpo_candidate_on_v2(workspace):
    ws, setup, log, source = workspace
    gate_evidence(source, ws)
    dpo_fallback_manifests(source, passed=None)
    out = run_chain(setup("s5dv", START))
    assert out.returncode == 0, out.stdout + out.stderr
    (p,) = passes(log)
    o = overrides(p)
    assert (o["pipeline.plan.phase"], o["pipeline.plan.strong_validate.branch"]) == (
        "strong_validate", "stage5_dpo")  # fmt: skip
    assert space_of(p) == str(ws / "c") and "before any follow-up (s5dx)" in out.stdout
    assert loads(log) == set()


def test_s5dx_runs_nothing_after_a_failed_dpo_validation(workspace):
    ws, setup, log, source = workspace
    gate_evidence(source, ws)
    dpo_fallback_manifests(source, passed=False)
    out = run_chain(setup("s5dx", START))
    assert out.returncode != 0 and "the DPO candidate did not validate" in out.stdout
    assert not passes(log)


def test_s5dx_makes_the_controls_then_k_star_of_the_dpo_strong_lineage(workspace):
    ws, setup, log, source = workspace
    gate_evidence(source, ws)
    dpo_fallback_manifests(source)
    out = run_chain(setup("s5dx", START))
    assert out.returncode == 0, out.stdout + out.stderr
    x, k = map(overrides, passes(log))
    assert (x["pipeline.plan.phase"], x["pipeline.plan.s5d_controls.dose"]) == (
        "s5d_controls", "200")  # fmt: skip
    keys = ("phase", "kstar.stage5_dpo", "kstar.strong", "kstar.dose", "kstar.sham_s",
            "kstar.active")  # fmt: skip
    assert tuple(k[f"pipeline.plan.{x}"] for x in keys) == (
        "kstar", "true", "true", "200", "Hsd200", "sA")  # fmt: skip
    assert "pipeline.plan.kstar.stage5" not in k
    assert all(o["pipeline.plan.cost.batches.strong_stage5_dpo"] == "12.0" for o in (x, k))
    assert "kstar_strong_example_link_dpo_200.json is committed" in out.stdout
    assert loads(log) == set()


def test_s5dc_waits_for_the_dpo_strong_lineage_s_k_star_manifest(workspace):
    ws, setup, log, source = workspace
    gate_evidence(source, ws)
    dpo_fallback_manifests(source)
    out = run_chain(setup("s5dc", START))
    assert out.returncode != 0
    assert "no committed k* manifest kstar_strong_example_link_dpo_200.json" in out.stdout
    assert not passes(log)


def test_s5dc_certifies_relearns_h_and_reads_the_dpo_strong_lineage_in_c(workspace):
    ws, setup, log, source = workspace
    gate_evidence(source, ws)
    certificate_spaces(source, ws)  # the certificate's inputs, as Stage 2 leaves them
    dpo_fallback_manifests(source, dose=400)
    k = json.loads((source / "manifests" / "kstar_example_link_400.json").read_text())
    (source / "manifests" / "kstar_strong_example_link_dpo_400.json").write_text(json.dumps(k))
    out = run_chain(setup("s5dc", START))
    assert out.returncode == 0, out.stdout + out.stderr
    ps = passes(log)
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == [
        "certificate", "certificate", "s5_relearn_h", "readers"]  # fmt: skip
    for o in map(overrides, ps[:3]):
        assert (o["pipeline.plan.certificate.stage5_dpo"],
                o["pipeline.plan.certificate.strong"]) == ("true", "true")  # fmt: skip
        assert "pipeline.plan.certificate.stage5" not in o
        assert (o["pipeline.plan.certificate.latent"], o["pipeline.plan.certificate.sham_s"],
                o["pipeline.plan.certificate.active"]) == ("Ld400", "Hsd400", "sA")  # fmt: skip
    r = overrides(ps[3])
    assert (r["pipeline.plan.readers.stage5_dpo"], r["pipeline.plan.readers.strong"],
            r["pipeline.plan.readers.dose"]) == ("true", "true", "400")  # fmt: skip
    assert {space_of(e) for e in ps} == {str(ws / "c")}
    assert all(overrides(e)["pipeline.plan.cost.batches.strong_stage5_dpo"] == "12.0" for e in ps)
    assert loads(log) == set()  # every pass's real config with its overrides (review s5k 1)


# ------------------------------------------------- Stage 3's stack lineage (C17)

COUNTDOWN = {"A": "s82de0bcca5866e3c", "H": "s934e62cd19b79b98", "own": "dd83aa138055dcef9"}


def countdown_inputs(ws):
    """What s3a reads, where the volume holds it (the panel's Stage-1 cells in a): A, H, the own
    data, the analyst index and the prompt pools; the corpora in c's HF cache."""
    from residual_disposition import continuation_confirm as C
    from residual_disposition import pipeline

    a = ws / "a"
    for sid in (COUNTDOWN["A"], COUNTDOWN["H"]):
        (a / "data" / "states" / sid).mkdir(parents=True)
        (a / "data" / "states" / sid / "manifest.json").write_text('{"adapters": []}')
    a_dir(a, f"data/generations/{COUNTDOWN['own']}")
    index = a / "runs" / "stage1_14b" / "train-countdown" / "observations"
    index.mkdir(parents=True)
    (index / "state_index.jsonl").write_text("{}\n")
    (index.parent / "meta.json").write_text(json.dumps({"status": "completed"}))
    (a / "runs" / "feasibility_14b").mkdir(parents=True, exist_ok=True)
    for pool in ("pool", C.FROZEN["pool"], pipeline.ood_pool()[0]):
        (a / "data" / "prompts" / pool).mkdir(parents=True, exist_ok=True)
        (a / "data" / "prompts" / pool / "set.jsonl").write_text("{}\n")
    corpora(ws / "c", behavior="countdown")


def test_s3a_copies_its_inputs_and_suppresses_in_its_own_space(workspace):
    ws, setup, log, _ = workspace
    countdown_inputs(ws)
    before = {x: tree(ws / x) for x in ("a", "b")}
    out = run_chain({**setup("s3a", START), "HF_HUB_OFFLINE": "1"})
    assert out.returncode == 0, out.stdout + out.stderr
    (p,) = passes(log)
    o = overrides(p)
    assert o["pipeline.plan.phase"] == "s3_suppress" and space_of(p) == str(ws / "c")
    assert o["pipeline.plan.cost.batches.stage3"] == "12.0"
    assert not any(k.startswith("pipeline.plan.cost.batches.stage3_") for k in o)
    for sid in (COUNTDOWN["A"], COUNTDOWN["H"]):
        assert (ws / "c" / "data" / "states" / sid / "manifest.json").is_file()
    assert (ws / "c" / "data" / "generations" / COUNTDOWN["own"]).is_dir()
    assert {x: tree(ws / x) for x in ("a", "b")} == before  # read, never written
    assert "stage3_diagnostic_sources_countdown.json" in out.stdout
    assert loads(log) == set()  # its real config with the chain's overrides
    chain = next((ws / "attempts" / "residual-disposition").glob("chain-s3a-*"))
    summary = json.loads((chain / "summary.json").read_text())
    assert (summary["worker"], summary["item"]) == ("pod", "stage3/countdown")


def test_a_stage3_session_names_its_space_and_runs_on_its_item(workspace):
    ws, setup, log, _ = workspace
    env = {k: v for k, v in setup("s3a", START).items() if k != "S3_SPACE"}
    out = run_chain(env)
    assert out.returncode != 0 and "S3_SPACE must be a or c" in out.stdout
    out = run_chain(setup("s3b", START, item="stage3/eli5", pod="P2"))
    assert out.returncode != 0
    assert "runs on the meter's item stage3/countdown, not stage3/eli5" in out.stdout
    assert not passes(log)


def next_step(source, out):
    """The snapshot's stage5_next.py replaced by one that names ``out`` (its own scripts tree)."""
    import shutil

    (source / "scripts").unlink()
    shutil.copytree(REPO / "scripts", source / "scripts")
    stub = f"import json\nprint(json.dumps({out!r}))\n"
    (source / "scripts" / "pod" / "stage5_next.py").write_text(stub)


def test_s3b_confirms_only_after_the_committed_diagnostic(workspace):
    ws, setup, log, source = workspace
    next_step(source, {"step": "confirm", "dose": 100, "status": "pending"})
    out = run_chain(setup("s3b", START))
    assert out.returncode != 0
    assert ("no confirmation: no committed detector diagnostic stage3_diagnostic_countdown.json"
            in out.stdout)  # fmt: skip
    assert not passes(log)


def test_s3b_confirms_with_the_committed_diagnostic(workspace):
    ws, setup, log, source = workspace
    next_step(source, {"step": "confirm", "dose": 100, "status": "pending"})
    (source / "manifests" / "stage3_diagnostic_countdown.json").write_text("{}")
    out = run_chain(setup("s3b", START))
    assert "no progress at confirm 100" in out.stdout  # the stub names it again
    (p,) = passes(log)
    o = overrides(p)
    assert (o["pipeline.plan.phase"], o["pipeline.plan.s3_confirm.dose"]) == ("s3_confirm", "100")


def test_s3b_records_no_stack_dose_without_the_diagnostic(workspace):
    """No margin-passing checkpoint: s3_decide records no stack dose; the diagnostic waits for
    the continued screening (E4)."""
    ws, setup, log, source = workspace
    next_step(source, {"step": "decide", "dose": None, "status": "decided",
                       "inputs": {"selection": "sel3", "reference": "ref3", "confirmation": []},
                       "decision": {"outcome": "no qualifying dose"}})  # fmt: skip
    out = run_chain(setup("s3b", START))
    assert out.returncode == 0, out.stdout + out.stderr
    (p,) = passes(log)
    o = overrides(p)
    assert o["pipeline.plan.phase"] == "s3_decide"
    assert (o["pipeline.plan.s3_decide.selection"], o["pipeline.plan.s3_decide.reference"]) == (
        "sel3", "ref3")  # fmt: skip
    assert "committed on the laptop before k*" in out.stdout


def test_s3b_needs_the_lineage_s_selection_in_its_space(workspace):
    ws, setup, log, source = workspace
    out = run_chain(setup("s3b", START))
    assert out.returncode != 0 and "stage5_next" in out.stdout and '"selection"' in out.stdout
    assert not passes(log)
