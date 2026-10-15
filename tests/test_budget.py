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

from residual_disposition import continuation_rules as cr  # noqa: E402
from residual_disposition.budget import (  # noqa: E402
    PLANNED,
    RESERVED,
    Allocation,
    AllocationMeter,
    apply_cut_order,
    load_allocations,
)

CUT_ORDER = ["stage6", "stage5", "stage4", "guillemets", "last_stage3_behavior", "stage2_to_one"]
GPU_PRICE = 5.32  # $3.99 × 4/3 per work hour


def alloc(item, kind, usd, stress):
    # GPU hours and API dollars as limits of their own: either may stop an item.
    return Allocation(item, kind, usd, stress, stress / GPU_PRICE, 10.0)


PLAN = [
    alloc("stage0_1", "stage0_1", 120.0, 140.0),
    alloc("stage2/example_link", "stage2", 140.0, 220.0),
    alloc("stage3/countdown", "stage3_panel", 135.0, 200.0),
    alloc("stage3/word_count_claim", "stage3_panel", 135.0, 200.0),
    alloc("stage3/eli5", "stage3_panel", 135.0, 200.0),
    alloc("stage3/tldr", "stage3_panel", 135.0, 200.0),
    alloc("stage2/pun_intended", "stage2", 130.0, 190.0),
    alloc("stage3/myth_fact", "stage3_panel", 135.0, 200.0),
    alloc("stage3/guillemets", "stage3_adl_only", 115.0, 190.0),
    alloc("stage4/example_link", "stage4", 140.0, 210.0),
    alloc("stage4/pun_intended", "stage4", 140.0, 210.0),
    alloc("stage5", "stage5", 150.0, 270.0),
    alloc("stage6", "stage6", 60.0, 60.0),
]
RULES = cr.load_rules()


def plan_meter(tmp_path=None, cap=1500):
    log = None if tmp_path is None else tmp_path / "meter.jsonl"
    return AllocationMeter(
        PLAN,
        cap=cap,
        reserve=100,
        gpu_usd_per_hour=GPU_PRICE,
        drop_reasons=cr.drop_reasons(RULES),
        log_path=log,
    )


def run_item(m, item, usd):
    assert m.admit(item, spent_elsewhere=0, storage=55).allowed
    m.start(item)
    m.charge(item, gpu_hours=usd / GPU_PRICE)
    m.complete(item)


def test_admission_reserves_the_stress_batch_and_central_higher_priority():
    m = plan_meter()
    decision = m.admit("stage3/countdown", spent_elsewhere=0, storage=55)
    assert decision.terms["next_batch"] == 200  # the item's stress reservation
    assert decision.terms["higher_priority"] == 120 + 140  # central, stages 0–1 and Stage 2 first
    assert m.state["stage3/countdown"] == RESERVED
    # A reserved item's reservation is outstanding for the next admission; nothing is counted twice.
    nxt = m.admit("stage3/word_count_claim", spent_elsewhere=0, storage=55)
    assert nxt.terms["reservations"] == 200
    assert nxt.terms["higher_priority"] == 120 + 140  # countdown is reserved, not planned


def test_lifecycle_transitions_are_checked():
    m = plan_meter()
    with pytest.raises(ValueError, match="live reservation"):
        m.start("stage0_1")  # no admission
    run_item(m, "stage0_1", 110)
    with pytest.raises(ValueError, match="admitted again"):
        m.admit("stage0_1", spent_elsewhere=0, storage=55)
    with pytest.raises(ValueError, match="only a running item"):
        m.complete("stage0_1")
    with pytest.raises(ValueError, match="only a running item is charged"):
        m.charge("stage0_1", gpu_hours=1)
    assert m.final["stage0_1"] == pytest.approx(110)  # immutable once completed
    m.drop("stage6", "gate_not_met")
    with pytest.raises(ValueError, match="admitted again"):
        m.admit("stage6", spent_elsewhere=0, storage=55)
    with pytest.raises(ValueError, match="cannot be dropped"):
        m.drop("stage3/countdown", "cut_order:guillemets")  # panel 1–4 can never be dropped
    with pytest.raises(ValueError, match="cannot be dropped"):
        m.drop("stage4/example_link", "gate_not_met")  # Stage 4's reason is its trigger
    m.drop("stage4/example_link", "trigger_not_met")
    m.drop("stage3/guillemets", "cut_order:guillemets")


def test_an_overrun_stops_the_item_and_raises():
    m = plan_meter()
    assert m.admit("stage0_1", spent_elsewhere=0, storage=55).allowed
    limits = m.start("stage0_1")
    assert limits.usd == 140 and limits.gpu_hours == pytest.approx(140 / GPU_PRICE)
    assert limits.api_usd == 10
    with pytest.raises(BudgetExhausted, match="p = 1"):
        m.charge("stage0_1", gpu_hours=150 / GPU_PRICE)
    assert m.state["stage0_1"] == "stopped"
    with pytest.raises(ValueError):
        m.limits("stage0_1")  # no execution without a live reservation


def test_stage2_second_behavior_cannot_use_the_panel_allocations():
    m = plan_meter()
    run_item(m, "stage0_1", 120)
    run_item(m, "stage2/example_link", 140)
    decision = m.admit("stage2/pun_intended", spent_elsewhere=0, storage=55)
    assert decision.terms["higher_priority"] == 4 * 135  # panel 1–4 stay reserved
    tight = plan_meter(cap=1100)
    run_item(tight, "stage0_1", 120)
    run_item(tight, "stage2/example_link", 140)
    assert not tight.admit("stage2/pun_intended", spent_elsewhere=0, storage=55).allowed
    assert tight.state["stage2/pun_intended"] == PLANNED  # deferred, still admissible later


def test_exception_a_protects_higher_work_and_stage4_until_its_trigger_is_resolved():
    m = plan_meter(cap=5000)
    run_item(m, "stage0_1", 120)
    run_item(m, "stage2/example_link", 140)
    before = m.admit("stage5", spent_elsewhere=0, storage=55)
    assert {"stage4/example_link", "stage4/pun_intended"} <= set(m.unfinished_above("stage5"))
    # pun's Stage-2 allocation is max(130, 140): its kind's completed lineage cost 140.
    assert before.terms["higher_priority"] == 4 * 135 + 140 + 135 + 115 + 2 * 140


def test_stage4_cannot_displace_the_panel_or_guillemets():
    m = plan_meter()
    higher = m.unfinished_above("stage4/example_link")
    assert {"stage3/tldr", "stage3/myth_fact", "stage3/guillemets"} <= set(higher)


def test_allocations_follow_the_frozen_update_rule():
    m = plan_meter(cap=5000)
    run_item(m, "stage3/countdown", 180)
    assert m.allocation("stage3/eli5") == pytest.approx(180)  # max(135, 180)
    run_item(m, "stage3/word_count_claim", 120)
    assert m.allocation("stage3/eli5") == pytest.approx(150)
    run_item(m, "stage3/eli5", 60)
    assert m.allocation("stage3/tldr") == 135  # the mean 120 is below the central estimate
    assert m.allocation("stage2/pun_intended") == 130  # another kind is untouched


def test_the_event_log_replays_to_the_same_state(tmp_path):
    m = plan_meter(tmp_path)
    run_item(m, "stage0_1", 110)
    m.admit("stage2/example_link", spent_elsewhere=0, storage=55)
    m.start("stage2/example_link")
    m.charge("stage2/example_link", gpu_hours=2.0, api_usd=1.5)
    m.drop("stage6", "gate_not_met")
    again = plan_meter(tmp_path)
    assert again.state == m.state and again.final == m.final
    assert again.charged == m.charged and again.reservation == m.reservation
    assert again.limits("stage2/example_link") == m.limits("stage2/example_link")


def test_the_frozen_allocations_file_loads_in_priority_order():
    from residual_disposition.paths import CONFIG_ROOT

    items = load_allocations(CONFIG_ROOT / "continuation" / "allocations.yaml")
    names = [a.item for a in items]
    assert names[0] == "stage0_1" and names[-1] == "stage6"
    assert all(a.stress_usd >= a.usd > 0 for a in items)
    assert set(cr.drop_reasons(RULES)) <= set(names)


def test_cut_sequence_follows_the_preset_order():
    parts = {"stage6": 50.0, "stage5": 200.0, "stage4": 300.0, "guillemets": 90.0}
    assert apply_cut_order(parts, order=CUT_ORDER, cap=1500, fixed=1000) == ["stage6", "stage5"]
    assert apply_cut_order(parts, order=CUT_ORDER, cap=1500, fixed=900) == ["stage6"]
    assert apply_cut_order(parts, order=CUT_ORDER, cap=1500, fixed=500) == []
    with pytest.raises(ValueError):
        apply_cut_order(parts, order=CUT_ORDER, cap=1500, fixed=1600)
    with pytest.raises(ValueError):
        apply_cut_order({"stage7": 1.0}, order=CUT_ORDER, cap=1500, fixed=0)
