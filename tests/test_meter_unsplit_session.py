"""The meter's one exclusive Pod session of an unsplit item (C13; CPU): worker "pod"
is a label, not a share; one open session at a time within the item's unspent parts, its Pod ID
free; while it is open no other session, charge, transfer or completion; its close settles the
paid cost on the item, also after a stop; no split once a session was issued, and no whole-item
launch receipt authorizes a Pod; two meter instances on one log; a session's API cost from its
ledger, an earlier session's money not counted again."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from residual_disposition import budget

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "continuation_meter.py"
spec = importlib.util.spec_from_file_location("continuation_meter_unsplit", SCRIPT)
cm = importlib.util.module_from_spec(spec)
sys.modules["continuation_meter_unsplit"] = cm
spec.loader.exec_module(cm)

ITEM, POD, DEADLINE = "stage0_1", budget.POD, "2099-01-01T00:00:00+00:00"


def run(capsys, log, *args):
    code = cm.main(["--log", str(log), *args])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def started(tmp_path, capsys):
    """stage0_1 admitted and started, unsplit (GPU $81.61, API $54.04)."""
    log = tmp_path / "meter.jsonl"
    run(capsys, log, "admit", ITEM, "--spent-elsewhere", "0", "--storage", "55")
    run(capsys, log, "start", ITEM)
    return log


def session(tmp_path, capsys, log, pod, *, hours="10", api="0", worker=POD):
    path = tmp_path / f"receipt-{pod}.json"
    code, out = run(
        capsys, log, "session-open", ITEM, "--worker", worker, "--pod-id", pod, "--deadline",
        DEADLINE, "--price", "3.99", "--hours", hours, "--api-usd", api, "--receipt", str(path),
    )  # fmt: skip
    assert code == 0
    return path, out


def close(capsys, log, pod, gpu="10", *extra):
    return run(capsys, log, "session-close", ITEM, "--worker", POD, "--pod-id", pod,
               "--gpu-usd", gpu, *extra)  # fmt: skip


def test_one_open_session_at_a_time_and_nothing_else_while_it_is_open(started, tmp_path, capsys):
    log = started
    path, receipt = session(tmp_path, capsys, log, "podA")
    assert (receipt["worker"], receipt["gpu_usd"]) == (POD, pytest.approx(39.9))
    for args, problem in (
        (["session-open", ITEM, "--worker", POD, "--pod-id", "podB", "--deadline", DEADLINE,
          "--price", "3.99", "--hours", "1", "--receipt", str(tmp_path / "r2.json")],
         "has an open session on Pod podA"),
        (["charge", ITEM, "--api-usd", "1"], "has an open Pod session on podA"),
        (["transfer", ITEM, "--from", "gpu", "--to", "api", "--usd", "5", "--reason", "x"],
         "open Pod session"),
        (["complete", ITEM], "open Pod sessions"),
        (["split", ITEM, "--share", "a=0.5", "--share", "b=0.5"], "never split"),
    ):  # fmt: skip
        with pytest.raises(ValueError, match=problem):
            cm.main(["--log", str(log), *args])
    code, out = run(capsys, log, "verify-receipt", str(path), "--item", ITEM, "--worker", POD,
                    "--pod-id", "podA", "--price", "3.99", "--hours", "10")  # fmt: skip
    assert code == 0 and out["session"]["pod_id"] == "podA"
    code, out = close(capsys, log, "podA", "12.5", "--api-usd", "2")
    assert code == 0 and out["left"]["gpu_usd"] == pytest.approx(81.61 - 12.5)
    code, status = run(capsys, log, "status")
    assert status["items"][ITEM]["charged"] == {"gpu_usd": 12.5, "api_usd": 2.0}
    # closed: the next session, a charge and a transfer may go; a split never
    session(tmp_path, capsys, log, "podB", hours="5")
    close(capsys, log, "podB", "3")
    assert run(capsys, log, "charge", ITEM, "--api-usd", "1")[0] == 0
    with pytest.raises(ValueError, match="never split"):
        cm.main(["--log", str(log), "split", ITEM, "--share", "a=1"])


def test_a_close_names_the_open_session_and_settles_it_once(started, tmp_path, capsys):
    log = started
    session(tmp_path, capsys, log, "podA")
    with pytest.raises(ValueError, match="no open session on Pod podX"):
        close(capsys, log, "podX")
    close(capsys, log, "podA")
    with pytest.raises(ValueError, match="no open session on Pod podA"):
        close(capsys, log, "podA")


def test_the_session_is_the_pod_label_within_the_unspent_parts(started, tmp_path, capsys):
    log = started
    with pytest.raises(ValueError, match="its Pod session's worker is 'pod'"):
        session(tmp_path, capsys, log, "podA", worker="pod-a")
    with pytest.raises(ValueError, match="exceed stage0_1's"):
        session(tmp_path, capsys, log, "podA", hours="21")  # 21 h x $3.99 > $81.61
    with pytest.raises(ValueError, match="API .* exceed stage0_1's"):
        session(tmp_path, capsys, log, "podA", hours="1", api="60")


def test_no_whole_item_receipt_authorizes_a_pod_once_a_session_was_issued(started, tmp_path,
                                                                          capsys):  # fmt: skip
    log = tmp_path / "fresh.jsonl"
    run(capsys, log, "admit", ITEM, "--spent-elsewhere", "0", "--storage", "55")
    whole = tmp_path / "launch.json"
    run(capsys, log, "start", ITEM, "--price", "3.99", "--receipt", str(whole))
    copy = tmp_path / "copy.jsonl"
    copy.write_bytes(log.read_bytes())
    code, _ = run(capsys, copy, "verify-receipt", str(whole), "--item", ITEM, "--price", "3.99",
                  "--hours", "1")  # fmt: skip
    assert code == 0  # before any session, as for Stage 1
    session(tmp_path, capsys, log, "podA")
    close(capsys, log, "podA", "0")
    receipt = json.loads(whole.read_text())
    receipt["log_sha256"] = __import__("hashlib").sha256(log.read_bytes()).hexdigest()
    whole.write_text(json.dumps(receipt))
    code, out = run(capsys, log, "verify-receipt", str(whole), "--item", ITEM, "--price", "3.99",
                    "--hours", "1")  # fmt: skip
    assert code == 1 and any("issued a Pod session" in p for p in out["problems"])


def test_a_stopped_item_s_session_is_settled_at_its_overrun(started, tmp_path, capsys):
    log = started
    session(tmp_path, capsys, log, "podA", hours="20")
    code, out = close(capsys, log, "podA", "90")  # beyond the $81.61 GPU part
    assert code == 3 and "stopped" in out
    code, status = run(capsys, log, "status")
    assert status["items"][ITEM]["state"] == "stopped"
    assert status["items"][ITEM]["charged"]["gpu_usd"] == pytest.approx(90)
    with pytest.raises(ValueError, match="is stopped: no Pod session"):
        session(tmp_path, capsys, log, "podB", hours="1")


def test_two_meter_instances_see_one_session(started, tmp_path):
    from residual_disposition.continuation_rules import drop_reasons, load_rules
    from residual_disposition.simulate_continuation import FROZEN

    rules = load_rules(expected_hash=FROZEN["rules_sha256"])
    allocations = budget.load_allocations(cm.ALLOCATIONS)

    def instance():
        return budget.AllocationMeter(
            allocations, cap=float(rules.data["meter"]["cap_usd"]),
            reserve=float(rules.data["meter"]["reserve_usd"]), drop_reasons=drop_reasons(rules),
            log_path=started,
        )  # fmt: skip

    one, two = instance(), instance()
    one.open_session(ITEM, POD, pod_id="podA", deadline=DEADLINE, price=3.99, hours=2)
    with pytest.raises(ValueError, match="has an open session on Pod podA"):
        two.open_session(ITEM, POD, pod_id="podB", deadline=DEADLINE, price=3.99, hours=2)
    with pytest.raises(ValueError, match="open Pod session"):
        two.charge(ITEM, api_usd=1.0)
    two.close_session(ITEM, POD, pod_id="podA", gpu_usd=7.98)
    assert one.session(ITEM, POD) is None and one.limits(ITEM).gpu_usd == pytest.approx(73.63)


def test_a_session_s_api_cost_counts_its_own_commitments_once(tmp_path):
    path = tmp_path / "adl-api-pod.jsonl"
    ledger = budget.ApiLedger(path, 100.0, POD)
    first = ledger.session_ceiling("s1", 5.0)
    ids = ledger.reserve({"a": 1.0, "b": 2.0}, "lens")
    ledger.settle(ids["a"], "a", 0.4, "lens")  # b stays open: it may be billed
    assert ledger.session_cost("s1") == pytest.approx(0.4 + 2.0) and first == pytest.approx(5.0)
    second = ledger.session_ceiling("s2", 3.0)  # the next session starts after s1's money
    assert second == pytest.approx(2.4 + 3.0)
    ids = ledger.reserve({"c": 1.5}, "lens")
    ledger.settle(ids["c"], "c", 1.1, "lens")
    assert ledger.session_cost("s2") == pytest.approx(1.1)
    assert ledger.session_ceiling("s2", 3.0) == pytest.approx(second)  # unchanged on restart
    ledger.close()
    assert budget.session_api_cost(path, POD, "s2") == pytest.approx(1.1)
    with pytest.raises(ValueError, match="no recorded ceiling"):
        budget.session_api_cost(path, POD, "s3")


def test_a_launch_receipt_of_stage5_never_authorizes_even_on_its_own_log_copy(tmp_path):
    """A whole-item receipt issued before the first session still matched its own
    log copy, which lacks the session; Stage 5 (every item but Stage 1's) runs its Pods on
    session receipts only."""
    from residual_disposition.continuation_rules import drop_reasons, load_rules

    rules = load_rules()
    allocations = [budget.Allocation("stage5", "stage5", 10.0, 50.0, 5.0)]
    log = tmp_path / "meter.jsonl"
    m = budget.AllocationMeter(allocations, cap=1820, reserve=100,
                               drop_reasons={"stage5": []}, log_path=log)  # fmt: skip
    m.admit("stage5", spent_elsewhere=0, storage=0)
    limits = m.start("stage5")
    receipt = cm.receipt(m, limits, 3.99)  # issued before any session, on this very log
    problems = cm.verify_receipt(m, receipt, item="stage5", price=3.99, hours=1)
    assert problems == [
        "stage5 runs its Pods on session receipts only: no launch receipt authorizes"
    ]
    assert drop_reasons(rules)["stage5"]  # the real item exists and is droppable by its gate
