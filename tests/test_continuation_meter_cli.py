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
    assert rules.data["meter"]["cap_usd"] == 1770
    AllocationMeter(  # a log begun under the cap of before 2026-10-19
        load_allocations(cm.ALLOCATIONS),
        cap=1500,
        reserve=100,
        drop_reasons=drop_reasons(rules),
        log_path=log,
    )
    code, status = run(capsys, log, "status")
    assert (status["cap"], status["rules_cap"], status["log_begun_under_cap"]) == (1500, 1770, 1500)
    assert status["needs_migration"]
    admit = ("admit", "stage0_1", "--spent-elsewhere", "0", "--storage")
    with pytest.raises(ValueError, match="migrate it first"):
        cm.main(["--log", str(log), *admit, "55"])
    reason = "user, 2026-10-19: $1,500 to $1,770 after Stage 1's re-priced API units"
    code, out = run(capsys, log, "migrate-cap", "--reason", reason)
    assert code == 0 and (out["cap"], out["log_begun_under_cap"]) == (1770, 1500)
    code, out = run(capsys, log, *admit, "1600")  # 135.65 + 1600 + 100 > 1770
    assert code == 4 and out["cap"] == 1770
    code, out = run(capsys, log, *admit, "1500")  # 1735.65: within $1,770, not within $1,500
    assert code == 0 and out["allowed"]


def two_pods(tmp_path, capsys):
    """stage2/example_link admitted, started and split between Pod A and Pod B."""
    log = tmp_path / "meter.jsonl"
    code, admitted = run(
        capsys, log, "admit", "stage2/example_link", "--spent-elsewhere", "0", "--storage", "55"
    )
    assert code == 0 and admitted["allowed"]
    run(capsys, log, "start", "stage2/example_link")
    code, parts = run(
        capsys,
        log,
        "split",
        "stage2/example_link",
        "--share",
        "pod-a=0.48",
        "--share",
        "pod-b=0.52",
    )
    assert code == 0 and parts["pod-a"]["gpu_usd"] == pytest.approx(0.48 * 243.12)
    return log


def receipt_for(tmp_path, capsys, log, worker, price="3.99"):
    path = tmp_path / f"receipt-{worker}.json"
    code, out = run(
        capsys, log, "worker-receipt", "stage2/example_link", "--worker", worker, "--price", price,
        "--receipt", str(path),
    )  # fmt: skip
    assert code == 0
    return path, out


def verify(capsys, log, path, worker, hours, price="3.99"):
    return run(
        capsys, log, "verify-receipt", str(path), "--item", "stage2/example_link",
        "--worker", worker, "--price", price, "--hours", str(hours),
    )  # fmt: skip


def test_each_pod_s_receipt_is_its_live_share(tmp_path, capsys):
    log = two_pods(tmp_path, capsys)
    path, receipt = receipt_for(tmp_path, capsys, log, "pod-a")
    hours = 0.48 * 243.12 / 3.99
    assert receipt["item"] == "stage2/example_link@pod-a"
    assert receipt["pod_hours"] == pytest.approx(hours)
    code, out = verify(capsys, log, path, "pod-a", round(hours, 2) - 0.01)
    assert code == 0 and out["valid"]
    code, out = verify(capsys, log, path, "pod-a", hours + 1)  # beyond the share
    assert code == 1 and "exceed" in out["problems"][0]
    code, out = verify(capsys, log, path, "pod-b", 1)  # Pod A's receipt for Pod B
    assert code == 1 and "not stage2/example_link@pod-b" in out["problems"][0]
    # A session's charge names its worker; the next receipt is the remaining share.
    charge = ["charge", "stage2/example_link", "--gpu-usd", "10", "--worker", "pod-a"]
    code, left = run(capsys, log, *charge)
    assert left["left"]["gpu_usd"] == pytest.approx(0.48 * 243.12 - 10)
    code, out = verify(capsys, log, path, "pod-a", 1)  # the old receipt on the new log
    assert code == 1 and "not the log the receipt was issued from" in out["problems"][0]
    path, receipt = receipt_for(tmp_path, capsys, log, "pod-a")
    assert receipt["gpu_usd"] == pytest.approx(0.48 * 243.12 - 10)
    _, other = receipt_for(tmp_path, capsys, log, "pod-b")
    assert other["gpu_usd"] == pytest.approx(0.52 * 243.12)  # Pod B's share untouched


def test_a_split_item_is_charged_per_worker_within_its_share(tmp_path, capsys):
    log = two_pods(tmp_path, capsys)
    with pytest.raises(ValueError, match="a charge names one of its workers"):
        cm.main(["--log", str(log), "charge", "stage2/example_link", "--gpu-usd", "1"])
    with pytest.raises(ValueError, match="not 'pod-c'"):
        cm.main(["--log", str(log), "charge", "stage2/example_link", "--gpu-usd", "1",
                 "--worker", "pod-c"])  # fmt: skip
    # Pod B overrunning its share stops the item, though the item's total is within its parts.
    overrun = ["charge", "stage2/example_link", "--gpu-usd", "130", "--worker", "pod-b"]
    code, out = run(capsys, log, *overrun)
    assert code == 3 and "stopped" in out


def test_an_unsplit_item_s_charges_name_no_worker(tmp_path, capsys):
    log = tmp_path / "meter.jsonl"
    run(capsys, log, "admit", "stage0_1", "--spent-elsewhere", "0", "--storage", "55")
    run(capsys, log, "start", "stage0_1")
    with pytest.raises(ValueError, match="is not split"):
        cm.main(["--log", str(log), "charge", "stage0_1", "--gpu-usd", "1", "--worker", "pod-a"])
