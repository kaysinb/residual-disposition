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
    ledger.reserve({"a": 0.3, "b": 0.3}, "graders")
    ledger.settle("a", 0.1, "graders")
    assert ledger.state() == {"spent": 0.1, "open": {"b": 0.3}, "committed": pytest.approx(0.4)}
    # A new process reads the same file: nothing spent is forgotten.
    again = ApiLedger(path, 1.0, "pod-a")
    assert again.committed() == pytest.approx(0.4) and again.remaining() == pytest.approx(0.6)
    with pytest.raises(ApiBudgetExhausted, match="allowance"):
        again.reserve({"c": 0.5, "d": 0.2}, "graders")  # all or none
    assert again.committed() == pytest.approx(0.4)
    # An open reservation whose completion reached the cache is settled at its cost.
    assert again.reconcile(lambda key: 0.05 if key == "b" else None) == 1
    assert again.state()["open"] == {} and again.state()["spent"] == pytest.approx(0.15)
    with pytest.raises(ValueError, match="other workers"):
        ApiLedger(path, 1.0, "pod-b").state()
    default = ledger_from({"worker": "laptop", "allowance_usd": 15}, tmp_path)
    assert default.path == tmp_path / "ledgers" / "adl-api-laptop.jsonl"
    for bad in ({"allowance_usd": 1}, {"worker": "???", "allowance_usd": 1}, {"worker": "w"}):
        with pytest.raises(ValueError):
            ledger_from(bad, tmp_path)


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
