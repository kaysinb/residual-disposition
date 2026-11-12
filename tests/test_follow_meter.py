"""The meter for the follow-up (§13.14): the frozen drop reasons of the registered work not executed
and of the follow-up's items, the allocations' migration, and the laptop API money of an item;
replayed on a copy of the live meter log when this checkout holds it."""

from __future__ import annotations

import pytest

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_money as fm
from residual_disposition.budget import (
    RUNG_SHARED,
    AllocationMeter,
    ApiBudgetExhausted,
    BudgetExhausted,
    load_allocations,
    load_migrations,
)
from residual_disposition.paths import CONFIG_ROOT, data_root
from residual_disposition.simulate_continuation import FROZEN
from residual_disposition.stage3_money import meter_of

ALLOCATIONS = CONFIG_ROOT / "continuation" / "allocations.yaml"
LIVE = data_root() / "ledgers" / "continuation_meter.jsonl"
FOLLOWS = [f"{p}stage3/{b}/follow" for b in ("countdown", "word_count_claim", "eli5", "tldr")
           for p in ("", "strong/")]  # fmt: skip
# the steering ablation at the partial threshold and D*'s hint and mask arms, one migration
ABLATION = ["follow/partial/ablation", "follow/lens_hint"]
# the eli5 marker's likelihood measurement, one migration after them
MARKER = ["follow/marker_likelihood"]
# the stack at 400 (§13.16), one migration after it
STACK400 = ["follow/stack400/full", "follow/cert/eli5_stack_400", "follow/stack400/lens_hint"]
# D*'s token-family masking and rung 2 with long steered excerpts, one migration after it
ADDITIONS = ["follow/lens_mask", "follow/rung2_long"]
# the second interpreter of another family, one migration after them
READER2 = ["follow/reader2"]
# the masking on pun_intended and the "EL" deletion read by the second interpreter, one after it
PROBES = ["follow/lens_mask2", "follow/reader2_el"]
# near fragments or meaning words read by the second interpreter, one after them
SPLIT = ["follow/reader2_split"]
LATE = ABLATION + MARKER + STACK400 + ADDITIONS + READER2 + PROBES + SPLIT
NEW = [a.item for a in load_allocations(ALLOCATIONS)
       if a.item.startswith("follow/") and a.item not in LATE]  # fmt: skip
SIZE = [a.item for a in load_allocations(ALLOCATIONS) if a.item.startswith("size/")]


def test_drop_reasons():
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    reasons = cr.drop_reasons(rules, [a.item for a in load_allocations(ALLOCATIONS)])
    assert "not_executed" in reasons["stage4/example_link"]
    assert "not_executed" in reasons["stage4/pun_intended"]
    assert "not_completed" in reasons["stage3/guillemets"]
    assert reasons["stage6"][0] == "gate_not_met"
    assert reasons["follow/cert/example_link"] == ["not_confirmed"]
    assert reasons["follow/continued/rung3"] == ["not_run_after_identification", "rung_not_run"]
    assert reasons["follow/stack/rung2"] == ["branch_closed", "not_run_after_identification"]
    assert reasons["follow/rung3_shared"] == ["rung_not_reached", "rung_not_run"]
    assert not reasons.get("follow/base") and not reasons.get("follow/s1_audit")
    assert not any(r.startswith("cut_order:") for i in NEW for r in reasons.get(i, []))
    assert reasons["follow/partial/ablation"] == ["ablation_not_run"]
    assert reasons["follow/lens_hint"] == ["lens_hint_not_run"]
    assert reasons["follow/marker_likelihood"] == ["marker_likelihood_not_run"]
    assert all(reasons[i] == ["stack400_not_run"] for i in STACK400)


def test_migration_adds_the_follow_up_items_in_order():
    (follow, size, ablation, marker, stack400, additions, reader2, probes,
     split) = load_migrations(ALLOCATIONS)[-9:]  # fmt: skip
    assert list(follow.added) == NEW and len(NEW) == 12
    order = [a.item for a in load_allocations(ALLOCATIONS)]
    assert order.index("follow/base") == order.index("stage6") + 1
    # the size block's items (§13.17) come after, below every earlier item, then the ablation and
    # the lens arms
    assert list(size.added) == SIZE == ["size/4b_1", "size/4b_2", "size/14b_2"]
    assert list(ablation.added) == ABLATION
    assert list(marker.added) == MARKER
    assert list(stack400.added) == STACK400
    assert list(additions.added) == ADDITIONS
    assert list(reader2.added) == READER2
    assert list(probes.added) == PROBES
    assert list(split.added) == SPLIT
    assert order[-15:] == SIZE + LATE


def test_the_ablation_and_the_lens_arms_are_api_items_of_their_own_kinds():
    items = {a.item: a for a in load_allocations(ALLOCATIONS)}
    for name in ABLATION:
        a = items[name]
        assert a.stress_gpu_usd == 0.0 and a.stress_api_usd >= a.usd > 0
        assert [x.item for x in items.values() if x.kind == a.kind] == [name]


def test_the_marker_measurement_is_a_gpu_item_of_its_own_kind():
    items = {a.item: a for a in load_allocations(ALLOCATIONS)}
    a = items["follow/marker_likelihood"]
    assert a.stress_api_usd == 0.0 and a.stress_gpu_usd >= a.usd > 0
    assert [x.item for x in items.values() if x.kind == a.kind] == [a.item]


def test_the_stack_at_400_s_items_are_each_of_its_own_kind():
    items = {a.item: a for a in load_allocations(ALLOCATIONS)}
    for name in STACK400:
        a = items[name]
        assert a.stress_usd >= a.usd > 0
        assert [x.item for x in items.values() if x.kind == a.kind] == [name]
    assert items["follow/stack400/lens_hint"].stress_gpu_usd == 0.0


def test_size_items_drop_only_for_recorded_outcomes():
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    reasons = cr.drop_reasons(rules, [a.item for a in load_allocations(ALLOCATIONS)])
    assert reasons["size/4b_1"] == ["infeasible_by_floors", "calibration_unresolved"]
    for item in ("size/4b_2", "size/14b_2"):
        assert reasons[item] == [
            "infeasible_by_floors",
            "calibration_unresolved",
            "branch_not_chosen",
        ]
    kinds = {a.item: a.kind for a in load_allocations(ALLOCATIONS) if a.item in SIZE}
    assert len(set(kinds.values())) == 3  # no item's actual updates another's central


@pytest.fixture
def log_copy(tmp_path):
    """The live log as it stood before the follow-up's migration (its prefix up to that event),
    whatever has been recorded since."""
    if not LIVE.is_file():
        pytest.skip("no live meter log in this checkout")
    lines = LIVE.read_text().splitlines(keepends=True)
    cut = next((i for i, line in enumerate(lines)
                if '"allocations"' in line and '"follow/base"' in line), len(lines))  # fmt: skip
    copy = tmp_path / "continuation_meter.jsonl"
    copy.write_text("".join(lines[:cut]))
    return copy


def test_old_allocations_replay_unchanged(log_copy):
    """The log replays under the allocations of before the follow-up's items exactly as under the
    new file before its migration is recorded: the same money and every earlier item's state."""
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    before = [a for a in load_allocations(ALLOCATIONS)
              if not a.item.startswith(("follow/", "size/"))]  # fmt: skip
    meter = rules.data["meter"]
    old = AllocationMeter(before, cap=float(meter["cap_usd"]), reserve=float(meter["reserve_usd"]),
                          drop_reasons=cr.drop_reasons(rules, [a.item for a in before]),
                          log_path=log_copy,
                          migrations=load_migrations(ALLOCATIONS)[:-9])  # fmt: skip
    new = meter_of(log_copy)
    assert old.allocations_pending == []
    assert new.allocations_pending == NEW + SIZE + LATE
    assert (old.spent(), old.outstanding()) == (new.spent(), new.outstanding())
    assert {i: old.state[i] for i in old.priority} == {i: new.state[i] for i in old.priority}


def test_the_planned_writes_on_a_copy(log_copy, tmp_path):
    m = meter_of(log_copy)
    spent = m.spent()
    assert sorted(m.allocations_pending) == sorted(NEW + SIZE + LATE)
    # the migration first: a log of the allocations before it records nothing else
    with pytest.raises(ValueError, match="migrate"):
        m.complete(FOLLOWS[0])
    assert m.migrate_allocations() == NEW
    assert m.migrate_allocations() == SIZE
    assert m.migrate_allocations() == ABLATION
    assert m.migrate_allocations() == MARKER
    assert m.migrate_allocations() == STACK400
    assert m.migrate_allocations() == ADDITIONS
    assert m.migrate_allocations() == READER2
    assert m.migrate_allocations() == PROBES
    assert m.migrate_allocations() == SPLIT
    m.migrate_cap(reason="to the rules' cap")  # the prefix was written under an earlier cap
    for item in FOLLOWS:
        m.complete(item)
    for item, reason in (
        ("stage4/example_link", "not_executed"),
        ("stage4/pun_intended", "not_executed"),
        ("stage3/guillemets", "not_completed"),
        ("stage6", "gate_not_met"),
    ):
        m.drop(item, reason)
    assert m.outstanding() == pytest.approx(0.0)
    assert m.spent() == pytest.approx(spent)
    decision = m.admit("follow/base", spent_elsewhere=13.68, storage=80.0)
    terms = decision.terms
    assert decision.allowed
    assert terms["next_batch"] == pytest.approx(33.67 + 308.98)
    assert terms["higher_priority"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        m.drop("follow/base", "rung_not_run")
    # the laptop API money: refused while planned, a session once running, settled at its cost
    with pytest.raises(ApiBudgetExhausted):
        fm.ItemPart(m, tmp_path, "follow/s1_audit")
    m.start("follow/base")
    meter, ledger = fm.money(tmp_path, "follow/base", log_copy)
    ids = ledger.reserve({"k": 0.01}, "test")
    ledger.settle(ids["k"], "k", 0.004, "test")
    assert ledger.close()["follow/base"] == pytest.approx(0.004)
    assert meter.laptop_session("follow/base") is None


def test_continued_stops_at_rung_one_and_the_stack_reaches_rungs_two_and_three(log_copy, tmp_path):
    """Regression (through the meter's own paths): an early identification in continued mode
    releases its rung items, never the shared ones, which the stack's ladder then uses; a rung
    item starts and opens a Pod or laptop session only while its shared item is held or complete,
    and the shared item is dropped only once no mode can reach its rung."""
    m = meter_of(log_copy)
    for _ in range(9):
        m.migrate_allocations()
    m.migrate_cap(reason="to the rules' cap")  # the prefix was written under an earlier cap
    for item in FOLLOWS:
        m.complete(item)
    for item, reason in (
        ("stage4/example_link", "not_executed"),
        ("stage4/pun_intended", "not_executed"),
        ("stage3/guillemets", "not_completed"),
        ("stage6", "gate_not_met"),
    ):
        m.drop(item, reason)
    for item in ("follow/continued/rung2", "follow/continued/rung3"):  # identified at rung 1
        m.drop(item, "not_run_after_identification")
    for shared in sorted(set(RUNG_SHARED.values())):  # the stack can still reach both rungs
        with pytest.raises(ValueError, match="no mode can reach its rung"):
            m.drop(shared, "rung_not_reached")
        assert m.state[shared] == "planned"
    stack2 = "follow/stack/rung2"
    m.admit(stack2, spent_elsewhere=13.68, storage=80.0)
    with pytest.raises(ValueError, match="rung2_shared is planned"):
        m.start(stack2)  # its own reservation alone: two admissions are not atomic
    assert m.state[stack2] == "reserved"
    m.admit("follow/rung2_shared", spent_elsewhere=13.68, storage=80.0)
    m.start(stack2)
    meter, ledger = fm.money(tmp_path, stack2, log_copy)  # the laptop's API path
    assert ledger.close()[stack2] == 0.0
    pod = {"pod_id": "p1", "deadline": "2026-10-30T00:00:00Z", "price": 3.99, "hours": 0.1}
    m.open_session(stack2, "pod", **pod)  # the Pod path
    m.close_session(stack2, "pod", pod_id="p1", gpu_usd=0.1)
    with pytest.raises(ValueError, match="no mode can reach its rung"):
        m.drop("follow/rung2_shared", "rung_not_reached")  # the stack's rung 2 is running
    _, open_ledger = fm.money(tmp_path, stack2, log_copy)  # a laptop session already open
    sent = open_ledger.reserve({"k1": 0.01}, "test")
    m.start("follow/rung2_shared")  # the shared work overruns: stopped, neither held nor complete
    with pytest.raises(BudgetExhausted):
        m.charge("follow/rung2_shared", api_usd=m.limits("follow/rung2_shared").api_usd + 1.0)
    assert m.state["follow/rung2_shared"] == "stopped"
    with pytest.raises(ApiBudgetExhausted):  # the open session takes no new request
        open_ledger.reserve({"k2": 0.01}, "test")
    open_ledger.settle(sent["k1"], "k1", 0.004, "test")  # what was sent is still settled
    assert open_ledger.close()[stack2] == pytest.approx(0.004)
    with pytest.raises(ValueError, match="no laptop session, its rung's shared item"):
        fm.money(tmp_path, stack2, log_copy)
    with pytest.raises(ValueError, match="no Pod session, its rung's shared item"):
        m.open_session(stack2, "pod", **{**pod, "pod_id": "p2"})
    # rung 3: the stack's branch closes before it, so its shared item is released after it
    m.admit("follow/stack/rung3", spent_elsewhere=13.68, storage=80.0)
    m.admit("follow/rung3_shared", spent_elsewhere=13.68, storage=80.0)
    with pytest.raises(ValueError, match="no mode can reach its rung"):
        m.drop("follow/rung3_shared", "rung_not_reached")
    m.drop("follow/stack/rung3", "branch_closed")
    m.drop("follow/rung3_shared", "rung_not_reached")
    assert meter_of(log_copy).state["follow/rung3_shared"] == "dropped"  # as the log replays
