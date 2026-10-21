"""A migration that raises existing reservations (§12.8; C17-M, review s5p 4): each raised item's
allocation before the change is recorded, the chain of hashes walks back with each raise
reverted, a value only grows, and the log reads the frozen value until it records the raise,
which it accepts only for a planned item, between Pod sessions. Stage 3's corrected reservations
are allocations.yaml's raise."""

from __future__ import annotations

import pytest

from residual_disposition.paths import CONFIG_ROOT, REPO_ROOT, data_root


def toy_meter(log, items, migrations=()):
    from residual_disposition.budget import AllocationMeter

    return AllocationMeter(items, cap=1e4, reserve=0.0, drop_reasons={}, log_path=log,
                           migrations=list(migrations))  # fmt: skip


def test_a_raise_applies_from_its_event_on_and_only_to_a_planned_item(tmp_path):
    from residual_disposition.budget import Allocation, AllocationMigration, allocations_hash

    old = [Allocation("first", "k1", 10.0, 12.0, 1.0), Allocation("panel", "k2", 100.0, 150.0, 5.0),
           Allocation("later", "k3", 50.0, 60.0, 2.0)]  # fmt: skip
    log = tmp_path / "meter.jsonl"
    m = toy_meter(log, old)
    m.admit("first", spent_elsewhere=0, storage=0)
    m.start("first")
    new = [old[0], Allocation("panel", "k2", 100.0, 200.0, 7.0), old[2]]
    raise_ = AllocationMigration((), "the corrected Stage-3 reservations", allocations_hash(old),
                                 {"panel": (100.0, 150.0, 5.0)})  # fmt: skip
    m = toy_meter(log, new, [raise_])
    assert (m._get("panel").stress_gpu_usd, m._get("panel").stress_api_usd) == (150.0, 5.0)
    assert m.migrate_allocations() == []
    assert (m._get("panel").stress_gpu_usd, m._get("panel").stress_api_usd) == (200.0, 7.0)
    event = [e for e in m.events if e.get("event") == "allocations"][-1]
    assert event["raised"] == {"panel": [[100.0, 150.0, 5.0], [100.0, 200.0, 7.0]]}
    assert toy_meter(log, new, [raise_])._get("panel").stress_gpu_usd == 200.0  # replayed
    m.admit("panel", spent_elsewhere=0, storage=0)
    assert m.reservation["panel"] == 207.0
    # an item already admitted is never raised after
    log2 = tmp_path / "meter2.jsonl"
    m = toy_meter(log2, old)
    m.admit("panel", spent_elsewhere=0, storage=0)
    m = toy_meter(log2, new, [raise_])
    with pytest.raises(ValueError, match="only a planned item's allocation is raised"):
        m.migrate_allocations()


@pytest.mark.parametrize(
    "raised, added, problem",
    [
        ({"panel": (100.0, 250.0, 5.0)}, (), "only raises"),  # the new value is lower
        ({"panel": (100.0, 200.0, 7.0)}, (), "only raises"),  # no change
        ({"panel": (100.0, 150.0, 5.0)}, ("panel",), "exist before it"),
        ({}, (), "adds or raises items"),
        ({"panel": (100.0, 150.0, 4.0)}, (), "do not hash to"),
    ],
)
def test_a_raise_that_lowers_or_does_not_chain_is_refused(tmp_path, raised, added, problem):
    from residual_disposition.budget import Allocation, AllocationMigration, allocations_hash

    old = [Allocation("first", "k1", 10.0, 12.0, 1.0), Allocation("panel", "k2", 100.0, 150.0, 5.0)]
    new = [old[0], Allocation("panel", "k2", 100.0, 200.0, 7.0)]
    before = old if added == () else [old[0]]
    migration = AllocationMigration(added, "x", allocations_hash(before), raised)
    with pytest.raises(ValueError, match=problem):
        toy_meter(tmp_path / "m.jsonl", new, [migration])


LIVE_LOG = data_root() / "ledgers" / "continuation_meter.jsonl"


@pytest.mark.skipif(not LIVE_LOG.is_file(), reason="the live meter log is not on this machine")
def test_the_live_log_migrates_to_the_raised_stage3_reservations(tmp_path):
    """allocations.yaml's raise (C17-M): the live log reads the frozen values until it records
    it, between Pod sessions, then the corrected ones; the earlier migrations' hashes stand."""
    import shutil
    import sys

    from residual_disposition.budget import load_allocations, load_migrations

    path = CONFIG_ROOT / "continuation" / "allocations.yaml"
    raise_ = load_migrations(path)[-1]
    assert raise_.added == () and raise_.from_sha256.startswith("a787e6d2")
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import continuation_meter as cm

    log = tmp_path / "meter.jsonl"
    shutil.copy(LIVE_LOG, log)
    m = cm.meter(log)
    current = {a.item: a for a in load_allocations(path)}
    for item, before in raise_.raised.items():
        a = m._get(item)
        assert (a.usd, a.stress_gpu_usd, a.stress_api_usd) == before
    for item, workers in list(m.sessions.items()):  # settled, as after the Pods' stop
        for worker, s in list((workers or {}).items()):
            if s:
                m.close_session(item, worker, pod_id=s["pod_id"], gpu_usd=s["gpu_usd"],
                                api_usd=s.get("api_usd", 0.0))  # fmt: skip
    while m._pending["steps"]:
        m.migrate_allocations()
    for item in raise_.raised:
        assert m._get(item) == current[item]
    assert current["stage3/countdown"].stress_gpu_usd == 246.85
    assert current["strong/stage3/countdown"].stress_api_usd == 26.39
