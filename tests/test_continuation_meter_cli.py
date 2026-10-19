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


def test_a_transfer_within_the_item(tmp_path, capsys):
    log = tmp_path / "meter.jsonl"
    run(capsys, log, "admit", "stage0_1", "--spent-elsewhere", "0", "--storage", "55")
    run(capsys, log, "start", "stage0_1")
    run(capsys, log, "charge", "stage0_1", "--gpu-usd", "29.06", "--api-usd", "11.3")
    reason = "Stage 1's GPU work is done; unspent GPU money to the API part"
    args = ("transfer", "stage0_1", "--from", "gpu", "--to", "api", "--usd", "10")
    code, out = run(capsys, log, *args, "--reason", reason)
    assert code == 0 and out["left"]["api_usd"] == pytest.approx(54.04 - 11.3 + 10)
    assert out["left"]["gpu_usd"] == pytest.approx(81.61 - 29.06 - 10)
    code, status = run(capsys, log, "status")
    parts = status["items"]["stage0_1"]["parts"]
    assert sum(parts.values()) == pytest.approx(81.61 + 54.04)  # the reservation is unchanged
    assert status["items"]["stage0_1"]["reservation"] == pytest.approx(81.61 + 54.04)
    with pytest.raises(ValueError, match="is unspent"):
        cm.main(["--log", str(log), *args[:-1], "43", "--reason", reason])  # GPU left: 42.55


def test_a_log_begun_under_the_old_cap_is_migrated_by_its_own_event(tmp_path, capsys):
    from residual_disposition.budget import AllocationMeter, load_allocations
    from residual_disposition.continuation_rules import drop_reasons, load_rules

    log = tmp_path / "meter.jsonl"
    rules = load_rules()
    assert rules.data["meter"]["cap_usd"] == 1700
    AllocationMeter(  # a log begun under the cap of before 2026-10-19
        load_allocations(cm.ALLOCATIONS),
        cap=1500,
        reserve=100,
        drop_reasons=drop_reasons(rules),
        log_path=log,
    )
    code, status = run(capsys, log, "status")
    assert (status["cap"], status["rules_cap"], status["log_begun_under_cap"]) == (1500, 1700, 1500)
    assert status["needs_migration"]
    admit = ("admit", "stage0_1", "--spent-elsewhere", "0", "--storage")
    with pytest.raises(ValueError, match="migrate it first"):
        cm.main(["--log", str(log), *admit, "55"])
    reason = "user, 2026-10-19: $1,500 to $1,700 after Stage 1's re-priced API units"
    code, out = run(capsys, log, "migrate-cap", "--reason", reason)
    assert code == 0 and (out["cap"], out["log_begun_under_cap"]) == (1700, 1500)
    code, out = run(capsys, log, *admit, "1500")  # 135.65 + 1500 + 100 > 1700
    assert code == 4 and out["cap"] == 1700
    code, out = run(capsys, log, *admit, "1400")  # 1635.65: within $1,700, not within $1,500
    assert code == 0 and out["allowed"]
