"""A panel behavior's Stage-3 start in the meter (§12.8; review s5y 2, 3): recorded once, with
both its reservations live and every earlier panel behavior's Stage 3 started, its lineage's one
space a or c; no Pod session of either of its items before it; replayed from the log."""

from __future__ import annotations

import pytest

from residual_disposition.budget import Allocation, AllocationMeter

DEADLINE = "2099-01-01T00:00:00+00:00"


def toy_meter(log):
    """Two panel behaviors, each with its strong item, between other kinds."""
    items = [Allocation("stage2/example_link", "stage2", 10.0, 12.0, 1.0),
             Allocation("stage3/countdown", "stage3_panel", 100.0, 150.0, 5.0),
             Allocation("strong/stage3/countdown", "strong_stage3", 30.0, 60.0, 9.0),
             Allocation("stage3/eli5", "stage3_panel", 100.0, 150.0, 5.0),
             Allocation("strong/stage3/eli5", "strong_stage3", 30.0, 60.0, 9.0),
             Allocation("stage3/guillemets", "stage3_adl_only", 100.0, 150.0, 5.0)]  # fmt: skip
    return AllocationMeter(items, cap=1e4, reserve=0.0, drop_reasons={}, log_path=log)


def run(m, *items):
    for item in items:
        m.admit(item, spent_elsewhere=0, storage=0)
        m.start(item)


def session(m, item, pod):
    return m.open_session(item, "pod", pod_id=pod, deadline=DEADLINE, price=3.99, hours=1.0)


def test_a_stage3_start_needs_both_reservations_live_and_the_earlier_behaviors(tmp_path):
    log = tmp_path / "meter.jsonl"
    m = toy_meter(log)
    run(m, "stage3/countdown")
    with pytest.raises(ValueError, match="no Pod session before countdown's Stage-3 start"):
        session(m, "stage3/countdown", "P1")
    with pytest.raises(ValueError, match="both its reservations live"):
        m.stage3_start("stage3/countdown", "c")  # its strong item planned
    m.admit("strong/stage3/countdown", spent_elsewhere=0, storage=0)
    for space in ("b", "../a", ""):
        with pytest.raises(ValueError, match="one space is a or c"):
            m.stage3_start("stage3/countdown", space)
    run(m, "stage3/eli5")
    m.admit("strong/stage3/eli5", spent_elsewhere=0, storage=0)
    with pytest.raises(ValueError, match=r"after the earlier panel behaviors'.*\['countdown'\]"):
        m.stage3_start("stage3/eli5", "a")
    for item in ("strong/stage3/countdown", "stage3/guillemets", "stage2/example_link"):
        with pytest.raises(ValueError, match="not a panel behavior's Stage-3 item"):
            m.stage3_start(item, "c")
    m.stage3_start("stage3/countdown", "c")
    with pytest.raises(ValueError, match="countdown's Stage 3 started in space c: recorded once"):
        m.stage3_start("stage3/countdown", "a")
    m.stage3_start("stage3/eli5", "a")  # each lineage its own space
    session(m, "stage3/countdown", "P1")
    assert m.stage3_of("stage3/countdown") == {"space": "c"}
    assert m.stage3_of("strong/stage3/countdown") == {"space": "c"}  # the same lineage
    assert m.stage3_of("stage2/example_link") is None


def test_the_strong_item_s_sessions_need_the_start_and_resumes_rely_on_its_record(tmp_path):
    log = tmp_path / "meter.jsonl"
    m = toy_meter(log)
    run(m, "stage3/countdown")
    m.admit("strong/stage3/countdown", spent_elsewhere=0, storage=0)
    m.start("strong/stage3/countdown")
    with pytest.raises(ValueError, match="no Pod session before countdown's Stage-3 start"):
        session(m, "strong/stage3/countdown", "P1")
    m.stage3_start("stage3/countdown", "c")
    session(m, "strong/stage3/countdown", "P1")
    m.close_session("strong/stage3/countdown", "pod", pod_id="P1", gpu_usd=1.0)
    m.complete("strong/stage3/countdown")  # no longer live: the record still stands
    again = toy_meter(log)  # replayed from the log
    assert again.stage3 == {"countdown": {"space": "c"}}
    session(again, "stage3/countdown", "P2")
    with pytest.raises(ValueError, match="recorded once"):
        again.stage3_start("stage3/countdown", "c")
    events = [e for e in again.events if e["event"] == "stage3_start"]
    assert [(e["item"], e["behavior"], e["space"]) for e in events] == [
        ("stage3/countdown", "countdown", "c")]  # fmt: skip
