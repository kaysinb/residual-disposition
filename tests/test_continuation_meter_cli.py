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


DEADLINE = "2099-01-01T00:00:00+00:00"


def open_session(tmp_path, capsys, log, worker, pod, hours="10", deadline=DEADLINE):
    path = tmp_path / f"receipt-{worker}-{pod}.json"
    code, out = run(
        capsys, log, "session-open", "stage2/example_link", "--worker", worker, "--pod-id", pod,
        "--deadline", deadline, "--price", "3.99", "--hours", hours, "--receipt", str(path),
    )  # fmt: skip
    assert code == 0
    return path, out


def verify(capsys, log, path, worker, pod, hours):
    return run(
        capsys, log, "verify-receipt", str(path), "--item", "stage2/example_link",
        "--worker", worker, "--pod-id", pod, "--price", "3.99", "--hours", str(hours),
    )  # fmt: skip


def test_a_pod_session_is_authorized_once_in_the_log(tmp_path, capsys):
    log = two_pods(tmp_path, capsys)
    path, receipt = open_session(tmp_path, capsys, log, "pod-a", "pod111")
    assert receipt["gpu_usd"] == pytest.approx(39.9) and receipt["pod_id"] == "pod111"
    copy = tmp_path / "copy.jsonl"
    copy.write_bytes(log.read_bytes())  # the copy uploaded with the receipt
    code, out = verify(capsys, copy, path, "pod-a", "pod111", 10)
    assert code == 0 and out["valid"]
    # review s3y2: no second session on the same share while the first is open
    with pytest.raises(ValueError, match="has an open session on Pod pod111"):
        open_session(tmp_path, capsys, log, "pod-a", "pod222")
    _, b = open_session(tmp_path, capsys, log, "pod-b", "pod333")  # the other share is free
    assert b["worker"] == "pod-b"
    # the receipt is the Pod's own and within its session
    code, out = verify(capsys, copy, path, "pod-a", "pod999", 10)
    assert code == 1 and "not this Pod pod999" in out["problems"][0]
    code, out = verify(capsys, copy, path, "pod-a", "pod111", 11)
    assert code == 1 and "exceed the session's" in out["problems"][0]


def test_a_settled_session_s_receipt_no_longer_verifies(tmp_path, capsys):
    log = two_pods(tmp_path, capsys)
    path, _ = open_session(tmp_path, capsys, log, "pod-a", "pod111")
    code, left = run(capsys, log, "session-close", "stage2/example_link", "--worker", "pod-a",
                     "--pod-id", "pod111", "--gpu-usd", "30")  # fmt: skip
    assert code == 0 and left["left"]["gpu_usd"] == pytest.approx(0.48 * 243.12 - 30)
    # the old receipt against the current log: no such open session
    code, out = verify(capsys, log, path, "pod-a", "pod111", 10)
    assert code == 1 and "not the log the receipt was issued from" in out["problems"][0]
    assert "open session in the log is not the receipt's" in out["problems"][1]
    # the next session opens only now, on what is left
    path2, second = open_session(tmp_path, capsys, log, "pod-a", "pod111", hours="5")
    assert second["gpu_usd"] == pytest.approx(19.95)
    with pytest.raises(ValueError, match="no open session on Pod pod222"):
        cm.main(["--log", str(log), "session-close", "stage2/example_link", "--worker", "pod-a",
                 "--pod-id", "pod222", "--gpu-usd", "1"])  # fmt: skip


def test_a_copied_receipt_after_its_deadline_is_refused(tmp_path, capsys):
    log = two_pods(tmp_path, capsys)
    past = "2020-01-01T00:00:00+00:00"
    path, _ = open_session(tmp_path, capsys, log, "pod-a", "pod111", deadline=past)
    code, out = verify(capsys, log, path, "pod-a", "pod111", 1)
    assert code == 1 and "deadline 2020-01-01T00:00:00+00:00 has passed" in out["problems"][0]
    with pytest.raises(ValueError, match="exceed pod-b's"):
        open_session(tmp_path, capsys, log, "pod-b", "pod333", hours="40")  # $159.6 > the share


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


def test_a_stopped_item_authorizes_no_session(tmp_path, capsys):
    # review s4a: B's overrun stops the item after A's session opened; A's receipt then fails
    log = two_pods(tmp_path, capsys)
    path, _ = open_session(tmp_path, capsys, log, "pod-a", "pod111")
    overrun = ["charge", "stage2/example_link", "--gpu-usd", "130", "--worker", "pod-b"]
    assert run(capsys, log, *overrun)[0] == 3
    code, out = verify(capsys, log, path, "pod-a", "pod111", 1)
    assert code == 1 and any("is stopped: no session runs" in p for p in out["problems"])
    with pytest.raises(ValueError, match="not a running split item"):
        open_session(tmp_path, capsys, log, "pod-b", "pod333")


def test_a_stopped_item_s_open_sessions_are_settled_and_held_until_then(tmp_path, capsys):
    log = two_pods(tmp_path, capsys)
    open_session(tmp_path, capsys, log, "pod-a", "pod111")  # $39.90 authorized
    overrun = ["charge", "stage2/example_link", "--gpu-usd", "130", "--worker", "pod-b"]
    assert run(capsys, log, *overrun)[0] == 3
    m = cm.meter(log)
    assert m.state["stage2/example_link"] == "stopped"
    assert m.outstanding() == pytest.approx(39.9)  # A's open session is still owed
    code, out = run(capsys, log, "session-close", "stage2/example_link", "--worker", "pod-a",
                    "--pod-id", "pod111", "--gpu-usd", "20")  # fmt: skip
    assert code == 0 and out["left"] is None  # settled at its paid cost
    m = cm.meter(log)
    assert m.outstanding() == 0 and m.final["stage2/example_link"] == pytest.approx(150)
    assert m.spent() == pytest.approx(150)


def test_an_item_completes_only_with_its_sessions_closed(tmp_path, capsys):
    log = two_pods(tmp_path, capsys)
    open_session(tmp_path, capsys, log, "pod-a", "pod111")
    with pytest.raises(ValueError, match="has open Pod sessions"):
        cm.main(["--log", str(log), "complete", "stage2/example_link"])
    run(capsys, log, "session-close", "stage2/example_link", "--worker", "pod-a",
        "--pod-id", "pod111", "--gpu-usd", "20")  # fmt: skip
    code, done = run(capsys, log, "complete", "stage2/example_link")
    assert code == 0 and done["final_usd"] == pytest.approx(20)


def test_a_session_journals_its_api_money_within_the_worker_s_api_share(tmp_path, capsys):
    # review s4b: the Pod's ADL allowance is bound to what the log authorized for the session
    log = two_pods(tmp_path, capsys)
    share = cm.meter(log).worker_limits("stage2/example_link", "pod-a").api_usd
    path = tmp_path / "receipt.json"
    args = [
        "session-open",
        "stage2/example_link",
        "--worker",
        "pod-a",
        "--pod-id",
        "pod111",
        "--deadline",
        DEADLINE,
        "--price",
        "3.99",
        "--hours",
        "10",
        "--receipt",
        str(path),
    ]
    with pytest.raises(ValueError, match="left of its API share"):
        run(capsys, log, *args, "--api-usd", str(share + 0.01))
    code, receipt = run(capsys, log, *args, "--api-usd", "2.5")
    assert code == 0 and receipt["api_usd"] == 2.5
    code, out = verify(capsys, log, path, "pod-a", "pod111", 10)
    assert code == 0 and out["session"]["api_usd"] == 2.5  # the log's, for the chain
    assert out["session"]["deadline"] == DEADLINE and out["session"]["hours"] == 10
    path.write_text(json.dumps({**receipt, "api_usd": 9.0}))  # a receipt claiming more
    code, out = verify(capsys, log, path, "pod-a", "pod111", 10)
    assert code == 1 and "open session in the log is not the receipt's" in out["problems"][-1]
    # a stopped item holds the session's API money too until it is settled
    overrun = ["charge", "stage2/example_link", "--gpu-usd", "130", "--worker", "pod-b"]
    assert run(capsys, log, *overrun)[0] == 3
    assert cm.meter(log).outstanding() == pytest.approx(39.9 + 2.5)


def test_a_pod_holds_one_open_session_whichever_worker(tmp_path, capsys):
    # one Pod's machine may serve pod-b's session and then pod-a's, never both at once
    log = two_pods(tmp_path, capsys)
    open_session(tmp_path, capsys, log, "pod-b", "pod111")
    with pytest.raises(ValueError, match="Pod pod111 has an open session of pod-b"):
        open_session(tmp_path, capsys, log, "pod-a", "pod111")
    run(capsys, log, "session-close", "stage2/example_link", "--worker", "pod-b",
        "--pod-id", "pod111", "--gpu-usd", "20")  # fmt: skip
    _, receipt = open_session(tmp_path, capsys, log, "pod-a", "pod111")  # now the same Pod
    assert (receipt["worker"], receipt["pod_id"]) == ("pod-a", "pod111")
