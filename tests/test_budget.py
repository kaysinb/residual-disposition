import fcntl
import json
from datetime import UTC, datetime, timedelta

import pytest

from residual_disposition.budget import (
    BudgetExhausted,
    CostMeter,
    verify_hard_deadline,
)

START = datetime(2026, 10, 12, tzinfo=UTC)


@pytest.fixture
def watch(tmp_path):
    """Write a watcher state; ``hold`` keeps its worker lock taken, as a live worker does."""
    held = []

    def make(*, hold=True, **changes):
        state = {
            "phase": "armed",
            "dry_run": False,
            "pod_id": "pod1",
            "deadline": (START + timedelta(hours=9)).timestamp(),
            **changes,
        }
        (tmp_path / "state.json").write_text(json.dumps(state))
        if hold and not held:
            lock = (tmp_path / "worker.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX)
            held.append(lock)
        return tmp_path / "state.json"

    yield make
    for lock in held:
        lock.close()


def verify(path, *, hours=1.0, pod="pod1"):
    return verify_hard_deadline(
        path, not_after=START + timedelta(hours=10), now=START + timedelta(hours=hours), pod=pod
    )


def test_a_live_armed_watcher_within_the_cap_verifies(watch) -> None:
    assert verify(watch()) == START + timedelta(hours=9)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"phase": "cancelled"}, "not armed"),
        ({"dry_run": True}, "not armed"),
        ({"deadline": None}, "no hard deadline"),
        ({"deadline": (START + timedelta(hours=11)).timestamp()}, "after the cap's end"),
        ({"pod_id": "pod2"}, "stops Pod pod2"),
    ],
)
def test_a_watcher_that_cannot_stop_the_pod_in_time_fails(watch, changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        verify(watch(**changes))


def test_a_passed_deadline_a_dead_worker_or_no_state_fails(watch, tmp_path) -> None:
    with pytest.raises(ValueError, match="has passed"):
        verify(watch(), hours=9.5)
    with pytest.raises(ValueError, match="no readable"):
        verify(tmp_path / "missing" / "state.json")
    with pytest.raises(ValueError, match="not None"):
        verify(watch(), pod=None)


def test_a_watcher_without_a_live_worker_fails(watch) -> None:
    with pytest.raises(ValueError, match="no live watcher worker"):
        verify(watch(hold=False))


def meter(watch, clock, **cost):
    plan = {
        "cost": {
            "cap_hours": 10,
            "started_at": START.isoformat(),
            "stop_fraction": 0.8,
            "deadline_state": str(watch()),
            **cost,
        }
    }
    return CostMeter.from_plan(plan, now=lambda: clock["now"], pod=lambda: "pod1")


def test_the_meter_stops_at_its_fraction_and_stays_stopped(watch) -> None:
    clock = {"now": START + timedelta(hours=7.9)}
    m = meter(watch, clock)
    m.check()
    assert m.projected(2.0) is None  # 7.9 + 2 within 10
    assert "projected overrun" in m.projected(2.2)
    clock["now"] = START + timedelta(hours=8)
    with pytest.raises(BudgetExhausted, match="cap used"):
        m.check()
    clock["now"] = START + timedelta(hours=1)
    with pytest.raises(BudgetExhausted):
        m.check()  # once exhausted, always


def test_the_meter_stops_when_the_watcher_goes_away(watch, tmp_path) -> None:
    clock = {"now": START + timedelta(hours=1)}
    m = meter(watch, clock)
    m.check()
    (tmp_path / "state.json").write_text(json.dumps({"phase": "cancelled"}))
    with pytest.raises(BudgetExhausted, match="hard deadline not verified"):
        m.check()


def test_the_meter_needs_a_zoned_past_start_and_a_watcher(watch) -> None:
    clock = {"now": START + timedelta(hours=1)}
    with pytest.raises(ValueError, match="time zone"):
        meter(watch, clock, started_at="2026-10-12T00:00")
    with pytest.raises(ValueError, match="must be set"):
        meter(watch, clock, started_at="???")
    with pytest.raises(ValueError, match="deadline_state"):
        meter(watch, clock, deadline_state=None)
    with pytest.raises(ValueError, match="future"):
        meter(watch, {"now": START - timedelta(hours=1)})
    assert CostMeter.from_plan({"cost": {"cap_hours": None}}).projected(100.0) is None


def test_the_adl_ledger_counts_open_reservations_and_survives_a_restart(tmp_path) -> None:
    from residual_disposition.budget import ApiBudgetExhausted, ApiLedger, ledger_from

    path = tmp_path / "ledgers" / "adl-api-pod-a.jsonl"
    ledger = ApiLedger(path, 1.0, "pod-a")
    ids = ledger.reserve({"a": 0.3, "b": 0.3}, "graders")
    ledger.settle(ids["a"], "a", 0.1, "graders")
    state = ledger.state()
    assert state["spent"] == 0.1 and state["committed"] == pytest.approx(0.4)
    assert list(state["open"].values()) == [("b", 0.3)]
    # A new process reads the same file: nothing spent is forgotten.
    ledger.close()
    again = ApiLedger(path, 1.0, "pod-a")
    assert again.committed() == pytest.approx(0.4) and again.remaining() == pytest.approx(0.6)
    with pytest.raises(ApiBudgetExhausted, match="allowance"):
        again.reserve({"c": 0.5, "d": 0.2}, "graders")  # all or none
    assert again.committed() == pytest.approx(0.4)
    # An open reservation whose completion reached the cache is settled at its cost.
    assert again.reconcile(lambda key: 0.05 if key == "b" else None) == 1
    assert again.state()["open"] == {} and again.state()["spent"] == pytest.approx(0.15)
    with pytest.raises(ValueError, match="other workers"):
        ApiLedger(path, 1.0, "pod-b", readonly=True).state()
    default = ledger_from({"worker": "laptop", "allowance_usd": 15}, tmp_path)
    assert default.path == tmp_path / "ledgers" / "adl-api-laptop.jsonl"
    for bad in ({"allowance_usd": 1}, {"worker": "???", "allowance_usd": 1}, {"worker": "w"}):
        with pytest.raises(ValueError):
            ledger_from(bad, tmp_path)


def test_billing_evidence_closes_named_open_reservations_at_zero(tmp_path) -> None:
    from residual_disposition.budget import ApiLedger

    path = tmp_path / "adl-api-laptop.jsonl"
    ledger = ApiLedger(path, 5.0, "laptop")
    ids = ledger.reserve({"a": 0.4, "b": 0.3, "c": 0.2}, "summarizer")
    ledger.settle(ids["a"], "a", 0.1, "summarizer")
    ledger.fail(ids["b"], "b", "InternalServerError", 503, released=False, label="summarizer")
    assert ledger.committed() == pytest.approx(0.6)
    held = ledger.close_unbilled(
        [ids["b"], ids["c"]], source="balance", decision="coordinator", label="reconciliation"
    )
    # The failed and the never-completed sends close at $0; the spend is untouched.
    assert held == pytest.approx(0.5)
    assert ledger.state()["open"] == {} and ledger.committed() == pytest.approx(0.1)
    with pytest.raises(ValueError, match="not open"):
        ledger.close_unbilled([ids["a"]], source="s", decision="d", label="x")  # settled
    ledger.close()
    again = ApiLedger(path, 5.0, "laptop", readonly=True).state()
    assert again["open"] == {} and again["spent"] == pytest.approx(0.1)


def test_a_retried_send_settles_only_its_own_reservation(tmp_path) -> None:
    from residual_disposition.budget import ApiLedger

    ledger = ApiLedger(tmp_path / "adl-api-pod-a.jsonl", 5.0, "pod-a")
    first = ledger.reserve({"k": 0.40}, "graders")  # sent, then interrupted
    second = ledger.reserve({"k": 0.40}, "graders")  # the same request, sent again
    assert first["k"] != second["k"]
    ledger.settle(second["k"], "k", 0.10, "graders")
    # The interrupted send's $0.40 is not lost when the retry settles at $0.10.
    assert ledger.committed() == pytest.approx(0.50)
    # Reconciling never settles the earlier send of a request that was settled.
    assert ledger.reconcile(lambda key: 0.10) == 0 and ledger.committed() == pytest.approx(0.50)
    # Of two open sends of a never-settled request, the latest is settled from the cache.
    ledger.reserve({"j": 0.30}, "graders")
    ledger.reserve({"j": 0.30}, "graders")
    assert ledger.reconcile(lambda key: 0.05 if key == "j" else None) == 1
    assert ledger.committed() == pytest.approx(0.50 + 0.30 + 0.05)


def batch_meter(watch, clock, phase, *, prior=0.0, deadline_hours=9):
    plan = {
        "phase": phase,
        "cost": {
            "batches": {"q1": 5.64, "rest": 22.56},
            "prior_hours": prior,
            "started_at": START.isoformat(),
            "stop_fraction": 0.8,
            "deadline_state": str(
                watch(deadline=(START + timedelta(hours=deadline_hours)).timestamp())
            ),
        },
    }
    return CostMeter.from_plan(plan, now=lambda: clock["now"], pod=lambda: "pod1")


def test_each_batch_has_its_own_cap_with_its_prior_hours(watch) -> None:
    clock = {"now": START + timedelta(hours=1)}
    q1 = batch_meter(watch, clock, "qualify", deadline_hours=5)
    assert (q1.batch, q1.cap_hours) == ("q1", 5.64)
    rest = batch_meter(watch, clock, "certify", prior=10.0, deadline_hours=12)
    assert (rest.batch, rest.cap_hours) == ("rest", 22.56)
    assert rest.elapsed() == pytest.approx(11.0)  # 10 h before this pass, 1 h of it
    assert rest.cap_end == START + timedelta(hours=12.56)
    # The watcher must stop the Pod before what is left of the cap runs out.
    with pytest.raises(ValueError, match="after the cap's end"):
        batch_meter(watch, clock, "certify", prior=20.0, deadline_hours=4).verify_deadline()
    with pytest.raises(ValueError, match="already used"):
        batch_meter(watch, clock, "suppress", prior=22.56)
    with pytest.raises(ValueError, match="prior_hours"):
        batch_meter(watch, clock, "suppress", prior=None)


def test_a_forecast_is_held_to_the_stop(watch) -> None:
    clock = {"now": START + timedelta(hours=1)}
    q1 = batch_meter(watch, clock, "qualify", deadline_hours=5)
    stop = 0.8 * 5.64
    assert q1.projected_stop(stop - 1.0 - 0.01) is None
    assert "the stop at 4.51 h" in q1.projected_stop(stop - 1.0 + 0.01)
    assert q1.projected(stop - 1.0 + 0.01) is None  # the hard cap alone would let it pass


def test_concurrent_reservations_never_pass_the_allowance(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from residual_disposition.budget import ApiBudgetExhausted, ApiLedger

    ledger = ApiLedger(tmp_path / "adl-api-pod-a.jsonl", 1.0, "pod-a")

    def one(i):
        try:
            ledger.reserve({f"k{i}": 0.3}, "graders")
            return True
        except ApiBudgetExhausted:
            return False

    with ThreadPoolExecutor(max_workers=16) as pool:
        granted = sum(pool.map(one, range(32)))
    assert granted == 3 and ledger.committed() == pytest.approx(0.9)
    # The file holds exactly what the memory does.
    assert ApiLedger(ledger.path, 1.0, "pod-a", readonly=True).committed() == pytest.approx(0.9)


def test_a_ledger_has_one_writer(tmp_path) -> None:
    import subprocess
    import sys

    from residual_disposition.budget import ApiLedger, ApiLedgerBusy

    path = tmp_path / "adl-api-laptop.jsonl"
    first = ApiLedger(path, 15.0, "laptop")
    first.reserve({"a": 10.0}, "summarizer")
    # A second writer, in this process or another, fails at once: it cannot approve the same
    # balance again (two writers would have approved $20 of $15).
    with pytest.raises(ApiLedgerBusy, match="already has a writer"):
        ApiLedger(path, 15.0, "laptop")
    code = (
        "import sys\n"
        "from residual_disposition.budget import ApiLedger, ApiLedgerBusy\n"
        "try:\n"
        f"    ApiLedger({str(path)!r}, 15.0, 'laptop')\n"
        "except ApiLedgerBusy:\n"
        "    sys.exit(7)\n"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 7
    # A reader may look; it cannot write.
    reader = ApiLedger(path, 15.0, "laptop", readonly=True)
    assert reader.committed() == pytest.approx(10.0)
    with pytest.raises(ValueError, match="not open for writing"):
        reader.reserve({"b": 1.0}, "judge")
    # Reconciling happens in the writer only: no double settlement.
    first.close()
    second = ApiLedger(path, 15.0, "laptop")
    assert second.reconcile(lambda key: 1.0) == 1 and second.reconcile(lambda key: 1.0) == 0
    assert second.committed() == pytest.approx(1.0)


# The continuation's meter (§12.8)

from residual_disposition.budget import apply_cut_order, meter_check  # noqa: E402

CUT_ORDER = ["stage6", "stage5", "stage4", "guillemets", "last_stage3_behavior", "stage2_to_one"]


def dollars(**overrides):
    args = {
        "spent": 600.0,
        "commitments": {"stage3/eli5/api": 20.0},
        "next_batch": {"stage3/tldr/gpu": 150.0, "stage3/tldr/api": 30.0},
        "future_storage": {"storage": 40.0},
        "stage4_remaining": {"stage4/gpu": 300.0},
        "stage4_triggered": False,
        "reserve": 100.0,
        "cap": 1500.0,
    }
    return meter_check(**{**args, **overrides})


def test_meter_allows_and_refuses_an_over_cap_batch():
    ok = dollars()
    assert ok.allowed and ok.total == 600 + 20 + 180 + 40 + 100
    refused = dollars(next_batch={"stage3/tldr/gpu": 800.0})
    assert not refused.allowed and refused.total > 1500


def test_meter_counts_stage4_once_when_it_is_the_next_batch():
    # Stage 4 triggered and its first batch next: the batch's lines leave the remaining Stage 4.
    batch = {"stage4/gpu": 300.0}
    decision = dollars(next_batch=batch, stage4_triggered=True)
    assert decision.terms["next_batch"] == 300 and decision.terms["stage4_remaining"] == 0
    untriggered = dollars()
    assert "stage4_remaining" not in untriggered.terms
    with pytest.raises(ValueError):
        dollars(next_batch={"stage4/gpu": 250.0}, stage4_triggered=True)


def test_meter_does_not_double_count_commitments():
    decision = dollars(next_batch={"stage3/eli5/api": 20.0, "stage3/tldr/gpu": 150.0})
    assert decision.terms["next_batch"] == 150


def test_cut_sequence_follows_the_preset_order():
    parts = {"stage6": 50.0, "stage5": 200.0, "stage4": 300.0, "guillemets": 90.0}
    assert apply_cut_order(parts, order=CUT_ORDER, cap=1500, fixed=1000) == ["stage6", "stage5"]
    assert apply_cut_order(parts, order=CUT_ORDER, cap=1500, fixed=900) == ["stage6"]
    assert apply_cut_order(parts, order=CUT_ORDER, cap=1500, fixed=500) == []
    with pytest.raises(ValueError):
        apply_cut_order(parts, order=CUT_ORDER, cap=1500, fixed=1600)
    with pytest.raises(ValueError):
        apply_cut_order({"stage7": 1.0}, order=CUT_ORDER, cap=1500, fixed=0)
