"""scripts/continuation_meter.py on a temporary log (CPU): the Stage-1 sequence, the Pod's
deadline from the GPU limit at its price, and a refused second admission."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "continuation_meter.py"
spec = importlib.util.spec_from_file_location("continuation_meter", SCRIPT)
cm = importlib.util.module_from_spec(spec)
sys.modules["continuation_meter"] = cm
spec.loader.exec_module(cm)


def run(capsys, log, *args):
    code = cm.main(["--log", str(log), *args])
    return code, json.loads(capsys.readouterr().out)


def test_the_stage1_sequence(tmp_path, capsys):
    log = tmp_path / "meter.jsonl"
    code, admitted = run(
        capsys, log, "admit", "stage0_1", "--spent-elsewhere", "0", "--storage", "55"
    )
    assert code == 0 and admitted["allowed"]
    assert admitted["terms"]["next_batch"] == pytest.approx(81.61 + 54.04)
    code, limits = run(capsys, log, "start", "stage0_1", "--price", "3.99")
    assert limits["gpu_usd"] == pytest.approx(81.61) and limits["api_usd"] == pytest.approx(54.04)
    assert limits["pod_hours"] == pytest.approx(81.61 / 3.99)
    code, left = run(capsys, log, "charge", "stage0_1", "--gpu-usd", "40", "--api-usd", "1.5")
    assert left["left"]["gpu_usd"] == pytest.approx(41.61)
    code, status = run(capsys, log, "status")
    assert status["items"]["stage0_1"]["state"] == "running" and status["spent"] == pytest.approx(
        41.5
    )
    code, done = run(capsys, log, "complete", "stage0_1")
    assert done["final_usd"] == pytest.approx(41.5)
    with pytest.raises(ValueError, match="cannot be admitted again"):
        cm.main(
            ["--log", str(log), "admit", "stage0_1", "--spent-elsewhere", "0", "--storage", "55"]
        )


def test_an_overrun_stops_the_item(tmp_path, capsys):
    log = tmp_path / "meter.jsonl"
    run(capsys, log, "admit", "stage0_1", "--spent-elsewhere", "0", "--storage", "55")
    run(capsys, log, "start", "stage0_1")
    code, out = run(capsys, log, "charge", "stage0_1", "--gpu-usd", "90")
    assert code == 3 and "stopped" in out


def test_a_refused_admission_exits_nonzero(tmp_path, capsys):
    code, out = run(
        capsys,
        tmp_path / "m.jsonl",
        "admit",
        "stage0_1",
        "--spent-elsewhere",
        "0",
        "--storage",
        "2000",
    )
    assert code == 4 and not out["allowed"]


def test_the_launch_receipt_binds_the_pod_to_the_live_reservation(tmp_path, capsys):
    log, rec = tmp_path / "m.jsonl", tmp_path / "receipt.json"
    run(capsys, log, "admit", "stage0_1", "--spent-elsewhere", "0", "--storage", "55")
    code, issued = run(capsys, log, "start", "stage0_1", "--price", "3.99", "--receipt", str(rec))
    assert code == 0 and json.loads(rec.read_text()) == issued
    copy = tmp_path / "copy.jsonl"
    copy.write_bytes(log.read_bytes())

    def verify(*extra, at=copy, item="stage0_1", receipt=rec):
        return run(capsys, at, "verify-receipt", str(receipt), "--item", item, *extra)

    assert verify("--price", "3.99", "--hours", "17") == (
        0,
        {"receipt": str(rec), "valid": True, "problems": []},
    )
    code, out = verify("--price", "3.99", "--hours", "21")
    assert code == 1 and "exceed" in out["problems"][0]
    code, out = verify("--price", "4.49", "--hours", "17")
    assert code == 1 and "price" in out["problems"][0]
    code, out = verify("--price", "3.99", "--hours", "17", item="stage2/example_link")
    assert code == 1 and "not 'stage2/example_link'" in out["problems"][-1]
    # A receipt whose derived hours were edited does not raise the hours allowed.
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps({**issued, "pod_hours": 999}))
    code, out = verify("--price", "3.99", "--hours", "21", receipt=forged)
    assert code == 1 and any("pod_hours 999" in p for p in out["problems"])
    assert any("exceed the 20.4536 h" in p for p in out["problems"])
    run(capsys, log, "charge", "stage0_1", "--gpu-usd", "1", "--label", "engine check")
    assert json.loads(log.read_text().splitlines()[-1])["label"] == "engine check"
    stale = tmp_path / "stale.jsonl"
    stale.write_bytes(log.read_bytes())
    code, out = verify("--price", "3.99", "--hours", "17", at=stale)
    assert code == 1 and any("not the log" in p for p in out["problems"])
    assert any("charged" in p for p in out["problems"])
