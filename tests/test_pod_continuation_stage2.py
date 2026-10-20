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
    (a / "runs" / "stage1_14b").mkdir(parents=True)
    (b / "data" / "generations" / "d5f5179f8853cb6cc").mkdir(parents=True)
    (b / "data" / "generations" / "d5f5179f8853cb6cc" / "data.jsonl").write_text("{}\n")
    (a / "data" / "prompts" / "pool").mkdir(parents=True)
    (a / "data" / "prompts" / "pool" / "set.jsonl").write_text("{}\n")


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
    for space in ("a", "b", "c"):
        (ws / space / "runs" / "continuation_14b").mkdir(parents=True)
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
              provider=None, watch_late_s=0) -> dict:  # fmt: skip
        # Stage 5's item is unsplit: one session at a time, worker pod (an unsplit item that the
        # meter admits on its own, stage0_1, stands for it here)
        unsplit = session.startswith("s5")
        worker = worker or ("pod" if unsplit else "pod-a" if session.startswith("a") else "pod-b")
        # The meter: the item split, this Pod's session opened with its ID, deadline and API money.
        log, receipt = tmp_path / f"meter-{session}.jsonl", tmp_path / f"receipt-{session}.json"
        meter = ["--log", str(log)]
        item = "stage0_1" if unsplit else "stage2/example_link"
        cm.main([*meter, "admit", item, "--spent-elsewhere", "0", "--storage", "55"])
        cm.main([*meter, "start", item])
        if not unsplit:
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
            **({"ITEM": item, "BEHAVIOR": "example_link"} if unsplit else {}),
        }

    yield ws, setup, tmp_path / "stub.jsonl", source
    for p in procs:
        p.kill()


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
    chain = next((ws / "attempts" / "residual-disposition").glob("chain-b1-*"))
    summary = json.loads((chain / "summary.json").read_text())
    assert summary["ok"] and summary["worker"] == "pod-b" and summary["pod_id"] == "POD1"
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


def screening(space, rows):
    doses = {str(d): {"latent": f"L{d}", "sham_s": f"Hs{d}", "excess": {"upper": u},
                      "pool": {"hits": h}} for d, (u, h) in rows.items()}  # fmt: skip
    from residual_disposition import continuation_rules as cr

    rules = cr.load_rules()
    record = {"behavior": "example_link", "doses": doses, "rules_sha256": rules.sha256}
    a_cell(space, "20261020-000000-stage2_screening-aaaa-0001", "stage2_screening",
           {"rules_sha256": rules.sha256}, {},
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
    out = run_chain({**setup("a1", START), "KSTAR": "0"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert [overrides(e)["pipeline.plan.phase"] for e in passes(log)] == ["y2", "screen", "decide"]
    assert "k* not run in a1 (KSTAR=0)" in out.stdout
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


def test_a1_decides_and_calibrates_k_star_at_the_analysis_dose(workspace):
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
    assert phases == ["y2", "screen", "decide", "kstar"]
    assert loads(log) == set()
    y2 = overrides(ps[0])
    assert ps[0]["cmd"][ps[0]["cmd"].index(str(ws / "a")) + 1].endswith("y2_zero_indexed_lists")
    assert (
        y2["pipeline.plan.cost.batches.stage2"] == "12.0"
        and "pipeline.plan.cost.batches.y" not in y2
    )
    decide, kstar = overrides(ps[2]), overrides(ps[3])
    key = "pipeline.plan.decide."
    assert decide[key + "screening"] == "20261020-000000-stage2_screening-aaaa-0001"
    assert decide[key + "confirmation"] == "[20261020-000001-stage2_confirmation-bbbb-0001]"
    assert decide[key + "expression.ood"] == "20261020-000002-rates_ood-cccc-0001"
    assert (kstar["pipeline.plan.kstar.dose"], kstar["pipeline.plan.kstar.sham_s"]) == (
        "400",
        "Hs400",
    )
    assert kstar["pipeline.plan.kstar.active"] == "s11c92132dd9839ce"


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


def certificate_spaces(source, ws):
    """The committed manifests of example_link at 400 and what they name, where the sessions
    leave it (review s4f): A1's screening, k* evidence, b's data and Y₂ evidence (phase y2) in a;
    the pilot's Y organism and data in b and, copied by B1, in c; the stage-1 recipe run and the
    Y₂ selection's laptop inputs in b."""
    from residual_disposition import continuation_kstar as K
    from residual_disposition import continuation_y2 as y2

    (source / "manifests" / "stage2_example_link.json").write_text(json.dumps(
        {"decision": {"analysis_dose": 400}, "inputs": {"screening": "scr"}}
    ))  # fmt: skip
    kstar = {"dose": 400, "states": {"H_s": "Hs400", "A": "sA"}, "data_id": "dlink",
             "reference": {"run": "ref-A"},
             "relearn_runs": {"1e-05": ["cal1"], "5e-06": ["cal2"]}}  # fmt: skip
    (source / "manifests" / "kstar_example_link_400.json").write_text(json.dumps(kstar))
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
    screening_dir = a / "runs" / "continuation_14b" / "scr"
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
    assert (summary["worker"], summary["item"]) == ("pod", "stage0_1")
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


def test_a_stage5_session_is_at_most_12_hours(workspace):
    ws, setup, log, _ = workspace
    out = run_chain(setup("s5a", START, hours=13.0))
    assert out.returncode != 0 and "a Stage-5 session is at most 12 h" in out.stdout
    assert not passes(log) and stop_asked(log)
