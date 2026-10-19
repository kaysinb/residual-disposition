"""scripts/pod/continuation_stage2.sh dry run (CPU, bash, stub tools under a temporary
workspace; plan C8, item F): each session's passes in order in its own space, Pod B's inputs
copied from a create-only, the decision's steps as stage2_next.py names them, the session's
meter receipt and the snapshot's receipt checked before any pass, the export and the stop."""

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


def pilot_inputs(space):
    """What b1 copies from space a, as minimal files."""
    for sid in [*LINK_STATES, Y_ORGANISM]:
        (space / "data" / "states" / sid).mkdir(parents=True)
        (space / "data" / "states" / sid / "manifest.json").write_text(json.dumps({"adapters": []}))
    for run in Y_RUNS:
        (space / "runs" / "feasibility_14b" / run).mkdir(parents=True)
        (space / "runs" / "feasibility_14b" / run / "summary.json").write_text("{}")
    index = space / "runs" / "feasibility_14b" / "train-run" / "observations"
    index.mkdir(parents=True)
    (index / "state_index.jsonl").write_text("{}\n")
    (space / "runs" / "stage1_14b").mkdir(parents=True)
    (space / "data" / "generations" / "d5f5179f8853cb6cc").mkdir(parents=True)
    (space / "data" / "generations" / "d5f5179f8853cb6cc" / "data.jsonl").write_text("{}\n")
    (space / "data" / "prompts" / "pool").mkdir(parents=True)
    (space / "data" / "prompts" / "pool" / "set.jsonl").write_text("{}\n")


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
    for space in ("a", "c"):
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
    (tmp_path / "rp_env").write_text("")
    snapshot_receipt = tmp_path / "snapshot.receipt.json"
    snapshot_receipt.write_text(json.dumps({"snapshot_id": "snap"}))
    procs = []

    def setup(session, start: datetime, *, pod="POD1", hours=12.0, worker=None) -> dict:
        worker = worker or ("pod-a" if session.startswith("a") else "pod-b")
        # The meter: the item split, this Pod's session opened with its ID and deadline.
        log, receipt = tmp_path / f"meter-{session}.jsonl", tmp_path / f"receipt-{session}.json"
        meter = ["--log", str(log)]
        item = "stage2/example_link"
        cm.main([*meter, "admit", item, "--spent-elsewhere", "0", "--storage", "55"])
        cm.main([*meter, "start", item])
        cm.main([*meter, "split", item, "--share", "pod-a=0.48", "--share", "pod-b=0.52"])
        deadline = (start + timedelta(hours=hours)).isoformat()
        cm.main([*meter, "session-open", item, "--worker", worker, "--pod-id", pod,
                 "--deadline", deadline, "--price", "3.99", "--hours", str(hours),
                 "--receipt", str(receipt)])  # fmt: skip
        capsys.readouterr()
        copy = tmp_path / f"meter-copy-{session}.jsonl"
        copy.write_bytes(log.read_bytes())
        watch = ws / "watch" / f"{session}-w"
        watch.mkdir(parents=True, exist_ok=True)
        (watch / "state.json").write_text(json.dumps(
            {"phase": "armed", "token": "t",
             "deadline": (start + timedelta(hours=hours)).timestamp(),
             "revision": 1}
        ))  # fmt: skip
        procs.append(subprocess.Popen(
            ["bash", "-c", f"exec -a 'night_watch.py watch --state-dir {watch}' sleep 120"]
        ))  # fmt: skip
        return {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "WORKSPACE": str(ws),
            "CODE_ROOT": str(tmp_path / "code"),
            "RP_ENV": str(tmp_path / "rp_env"),
            "STUB_LOG": str(tmp_path / "stub.jsonl"),
            "SESSION": session,
            "SNAP": "snap",
            "START": start.isoformat(),
            "WN": f"{session}-w",
            "PRICE": "3.99",
            "HARD_HOURS": str(hours),
            "RECEIPT": str(receipt),
            "METER_LOG": str(copy),
            "POD_ID": pod,
            "ACTIVE": "s11c92132dd9839ce",
            "SOURCE_SNAPSHOT_RECEIPT": str(snapshot_receipt),
            "SOURCE_SNAPSHOT_ID": "snap",
            "SNAPSHOT_RECEIPT_SHA256": hashlib.sha256(snapshot_receipt.read_bytes()).hexdigest(),
            "CONTAINER_IMAGE": "img@sha256:x",
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


START = datetime.now(UTC) - timedelta(minutes=20)


def test_b1_copies_its_inputs_and_runs_y2_and_y_alone_in_its_own_space(workspace):
    ws, setup, log, _ = workspace
    pilot_inputs(ws / "a")
    before = sorted(str(p) for p in (ws / "a").rglob("*"))
    out = run_chain(setup("b1", START))
    assert out.returncode == 0, out.stdout + out.stderr
    ps = passes(log)
    assert [Path(e["attempt"]).name.rsplit("-", 1)[1] for e in ps] == ["y2", "y100", "y200"]
    assert {space_of(e) for e in ps} == {str(ws / "c")}  # Pod B writes only to c
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == ["y2", "control_y", "control_y"]
    y2, y100, y200 = ps
    assert y2["cmd"][y2["cmd"].index(str(ws / "c")) + 1].endswith("y2_zero_indexed_lists")
    assert y100["cmd"][y100["cmd"].index(str(ws / "c")) + 1].endswith("control_y_link_100")
    o = overrides(y2)
    assert o["pipeline.plan.api.adl_ledger.worker"] == "pod-b"
    assert o["pipeline.plan.cost.batches.stage2"] == o["pipeline.plan.cost.batches.y"] == "12.0"
    for sid in [*LINK_STATES, Y_ORGANISM]:
        assert (ws / "c" / "data" / "states" / sid / "manifest.json").is_file()
    assert (ws / "c" / "runs" / "feasibility_14b" / "train-run" / "observations").is_dir()
    assert sorted(str(p) for p in (ws / "a").rglob("*")) == before  # a untouched
    assert [e["space"] for e in events(log) if e["tool"] == "export"] == [str(ws / "c")]
    chain = next((ws / "attempts" / "residual-disposition").glob("chain-b1-*"))
    summary = json.loads((chain / "summary.json").read_text())
    assert summary["ok"] and summary["worker"] == "pod-b" and summary["pod_id"] == "POD1"
    assert json.loads((chain / "receipt-check.json").read_text())["valid"]


def test_a_missing_input_stops_b1_before_any_pass(workspace):
    ws, setup, log, _ = workspace
    pilot_inputs(ws / "a")
    __import__("shutil").rmtree(ws / "a" / "data" / "states" / Y_ORGANISM)
    out = run_chain(setup("b1", START))
    assert out.returncode != 0 and not passes(log)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"POD_ID": "POD9"}, "does not cover"),  # another Pod's session
        ({"HARD_HOURS": "13"}, "does not cover"),  # beyond the session's 12 h
        ({"SNAPSHOT_RECEIPT_SHA256": "0" * 64}, "not the one set at the Pod's creation"),
        ({"SOURCE_SNAPSHOT_ID": "other"}, "not the one set at the Pod's creation"),
    ],
)
def test_no_pass_starts_without_this_pod_s_session_and_snapshot(workspace, change, message):
    ws, setup, log, _ = workspace
    pilot_inputs(ws / "a")
    env = setup("b1", START)
    out = run_chain({**env, **change})
    assert out.returncode != 0 and message in out.stdout
    assert not passes(log)


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
    assert [n[-1] for n in names] == ["screen", "400"]
    confirm = overrides(passes(log)[1])
    assert confirm["pipeline.plan.phase"] == "confirm"
    assert confirm["pipeline.plan.confirm.latent"] == "L400"
    assert confirm["pipeline.plan.confirm.sham_s"] == "Hs400"
    assert {space_of(e) for e in passes(log)} == {str(ws / "a")}


def test_a1_decides_and_calibrates_k_star_at_the_analysis_dose(workspace):
    ws, setup, log, _ = workspace
    from residual_disposition import continuation_rules as cr
    from residual_disposition import continuation_stage2 as S2

    screening(ws / "a", {400: (0.004, 0), 800: (0.004, 0)})
    version = S2.current_detector_version("example_link")
    seed = cr.confirmation_seed(7, "example_link", 400, 1, version)
    draw = {"attempt": 1, "seed": seed, "detector_version": version, "passed": True,
            "purpose": None, "lineage": "example_link", "base_seed": 7}  # fmt: skip
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
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == ["screen", "decide", "kstar"]
    decide, kstar = overrides(ps[1]), overrides(ps[2])
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
    assert [overrides(e)["pipeline.plan.phase"] for e in passes(log)] == ["screen", "decide"]
    assert "exception a" in out.stdout


def committed_manifests(source, ws):
    (source / "manifests" / "stage2_example_link.json").write_text(json.dumps(
        {"decision": {"analysis_dose": 400}, "inputs": {"screening": "scr"}}
    ))  # fmt: skip
    kstar = {"dose": 400, "states": {"H_s": "Hs400", "A": "sA"}, "data_id": "dlink",
             "reference": {"run": "ref-A"},
             "relearn_runs": {"1e-05": ["cal1"], "5e-06": ["cal2"]}}  # fmt: skip
    (source / "manifests" / "kstar_example_link_400.json").write_text(json.dumps(kstar))
    screening_dir = ws / "a" / "runs" / "continuation_14b" / "scr"
    screening_dir.mkdir(parents=True)
    (screening_dir / "stage2_screening.json").write_text(
        json.dumps({"doses": {"400": {"latent": "L400", "sham_s": "Hs400"}}})
    )


def test_b2_copies_the_certificate_s_inputs_and_runs_blocks_3_to_5(workspace):
    ws, setup, log, source = workspace
    pilot_inputs(ws / "a")
    committed_manifests(source, ws)
    for sid in ("L400", "Hs400", "sA"):
        (ws / "a" / "data" / "states" / sid).mkdir(parents=True)
        (ws / "a" / "data" / "states" / sid / "manifest.json").write_text('{"adapters": []}')
    for run in ("cal1", "cal2"):
        (ws / "a" / "runs" / "continuation_14b" / run).mkdir(parents=True)
        (ws / "a" / "runs" / "continuation_14b" / run / "summary.json").write_text("{}")
    (ws / "a" / "runs" / "feasibility_14b" / "ref-A").mkdir(parents=True)
    (ws / "a" / "runs" / "feasibility_14b" / "ref-A" / "summary.json").write_text("{}")
    for rel in ("runs/labeling/detectors-470596e7fc2a", "data/labels/detectors/y2det-470596e7fc2a",
                "runs/preflight_14b/20261009-123952-preflight_elicit-57d0c000-8803",
                "data/generations/dlink"):  # fmt: skip
        (ws / "a" / rel).mkdir(parents=True, exist_ok=True)
        (ws / "a" / rel / "f.json").write_text("{}")
    out = run_chain(setup("b2", START))
    assert out.returncode == 0, out.stdout + out.stderr
    (cert,) = passes(log)
    o = overrides(cert)
    assert (
        o["pipeline.plan.phase"] == "certificate"
        and o["pipeline.plan.certificate.blocks"] == "[3,4,5]"
    )
    assert (o["pipeline.plan.certificate.latent"], o["pipeline.plan.certificate.sham_s"],
            o["pipeline.plan.certificate.active"]) == ("L400", "Hs400", "sA")  # fmt: skip
    assert space_of(cert) == str(ws / "c")
    copied = (
        "runs/continuation_14b/cal1",
        "data/states/L400",
        "runs/labeling/detectors-470596e7fc2a",
    )
    for rel in copied:
        assert (ws / "c" / rel).exists()


def test_a2_runs_blocks_0_to_2_and_the_readers_in_space_a(workspace):
    ws, setup, log, source = workspace
    committed_manifests(source, ws)
    out = run_chain(setup("a2", START))
    assert out.returncode == 0, out.stdout + out.stderr
    ps = passes(log)
    assert [overrides(e)["pipeline.plan.phase"] for e in ps] == ["certificate", "readers"]
    assert overrides(ps[0])["pipeline.plan.certificate.blocks"] == "[0,1,2]"
    assert {space_of(e) for e in ps} == {str(ws / "a")}


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
