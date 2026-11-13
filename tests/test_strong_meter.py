"""The strong branch's meter rules (§12.8, amendment of 2026-10-21; C16 item H), CPU only: the
items by an append-only migration, a split with a share per part, S5-A's sublimit enforced
across its sessions, and the strong items' frozen drop reasons."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from residual_disposition import continuation_rules as cr
from residual_disposition.budget import (
    Allocation,
    AllocationMeter,
    BudgetExhausted,
    load_allocations,
    load_migrations,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "continuation_meter.py"
spec = importlib.util.spec_from_file_location("continuation_meter_strong", SCRIPT)
cm = importlib.util.module_from_spec(spec)
sys.modules["continuation_meter_strong"] = cm
spec.loader.exec_module(cm)
RULES = cr.load_rules()
CAP = float(RULES.data["meter"]["cap_usd"])  # the rules' cap, as the CLI reads it
DEADLINE = "2026-10-22T12:00:00+00:00"


def run(capsys, log, *args):
    code = cm.main(["--log", str(log), *args])
    return code, json.loads(capsys.readouterr().out)


def one_item(tmp_path, gpu=100.0, api=20.0):
    item = Allocation("stage0_1", "stage0_1", 50.0, gpu, api)
    return AllocationMeter([item], cap=1e4, reserve=0, drop_reasons={},
                           log_path=tmp_path / "m.jsonl")  # fmt: skip


def test_a_split_takes_a_share_per_part_and_enforces_each(tmp_path):
    m = one_item(tmp_path)
    m.admit("stage0_1", spent_elsewhere=0, storage=0)
    m.start("stage0_1")
    with pytest.raises(ValueError, match="name exactly its workers"):
        m.split("stage0_1", {"a": 0.2, "b": 0.8}, {"a": 0.0})
    with pytest.raises(ValueError, match="API shares must be non-negative and sum to at most 1"):
        m.split("stage0_1", {"a": 0.2, "b": 0.8}, {"a": 0.5, "b": 0.6})
    parts = m.split("stage0_1", {"a": 0.2, "b": 0.8}, {"a": 0.0, "b": 1.0})
    assert (parts["a"].gpu_usd, parts["a"].api_usd) == (pytest.approx(20.0), 0.0)
    assert (parts["b"].gpu_usd, parts["b"].api_usd) == (pytest.approx(80.0), pytest.approx(20.0))
    with pytest.raises(ValueError, match="API .* exceed a's"):
        m.open_session("stage0_1", "a", pod_id="p1", deadline=DEADLINE, price=4.0, hours=1.0,
                       api_usd=0.01)  # fmt: skip
    m.open_session("stage0_1", "b", pod_id="p2", deadline=DEADLINE, price=4.0, hours=1.0,
                   api_usd=20.0)  # fmt: skip
    m.charge("stage0_1", worker="a", gpu_usd=5.0)
    with pytest.raises(BudgetExhausted):  # any API money on a's share overruns it
        m.charge("stage0_1", worker="a", api_usd=0.01)
    assert m.state["stage0_1"] == "stopped"
    # a split without API shares replays as before: one fraction of both parts
    again = AllocationMeter([Allocation("stage0_1", "stage0_1", 50.0, 100.0, 20.0)], cap=1e4,
                            reserve=0, drop_reasons={}, log_path=tmp_path / "m2.jsonl")  # fmt: skip
    again.admit("stage0_1", spent_elsewhere=0, storage=0)
    again.start("stage0_1")
    whole = again.split("stage0_1", {"a": 0.25, "b": 0.75})
    assert whole["a"].api_usd == pytest.approx(5.0)
    event = json.loads((tmp_path / "m2.jsonl").read_text().splitlines()[-1])
    assert event["event"] == "split" and "api_shares" not in event


def test_s5a_sublimit_holds_across_all_its_sessions(tmp_path, capsys):
    """strong/stage5 is split only by its frozen split; S5-A's sessions together never exceed
    $13.76 of GPU and get no API money."""
    log = tmp_path / "meter.jsonl"
    names = [a.item for a in load_allocations(cm.ALLOCATIONS)]
    m = AllocationMeter(load_allocations(cm.ALLOCATIONS), cap=CAP, reserve=100,
                        drop_reasons=cr.drop_reasons(RULES, names), log_path=log,
                        migrations=load_migrations(cm.ALLOCATIONS))  # fmt: skip
    item = "strong/stage5"
    assert m.admit(item, spent_elsewhere=0, storage=0).allowed
    m.start(item)
    with pytest.raises(SystemExit, match="its frozen split"):
        cm.main(["--log", str(log), "split", item, "--share", "s5-a=0.5", "--share", "s5-b=0.5"])
    code, parts = run(capsys, log, "split", item)  # the frozen split, by default
    assert code == 0
    assert parts["s5-a"]["gpu_usd"] == pytest.approx(13.76) and parts["s5-a"]["api_usd"] == 0
    assert parts["s5-b"]["gpu_usd"] == pytest.approx(283.04)
    assert parts["s5-b"]["api_usd"] == pytest.approx(23.22)
    session = ("session-open", item, "--worker", "s5-a", "--deadline", DEADLINE, "--price", "3.99")
    code, _ = run(capsys, log, *session, "--pod-id", "pa1", "--hours", "2",
                  "--receipt", str(tmp_path / "r1.json"))  # fmt: skip
    assert code == 0
    run(capsys, log, "session-close", item, "--worker", "s5-a", "--pod-id", "pa1",
        "--gpu-usd", "7.98")  # fmt: skip
    with pytest.raises(ValueError, match="exceed s5-a's"):  # $7.98 more: past $13.76 in all
        cm.main(["--log", str(log), *session, "--pod-id", "pa2", "--hours", "2",
                 "--receipt", str(tmp_path / "r2.json")])  # fmt: skip
    with pytest.raises(ValueError, match="API .* exceed s5-a's"):
        cm.main(["--log", str(log), *session, "--pod-id", "pa2", "--hours", "1",
                 "--api-usd", "0.5", "--receipt", str(tmp_path / "r2.json")])  # fmt: skip
    code, _ = run(capsys, log, *session, "--pod-id", "pa2", "--hours", "1.44",
                  "--receipt", str(tmp_path / "r2.json"))  # fmt: skip
    assert code == 0  # 7.98 + 5.7456 ≤ 13.76


def test_the_strong_items_drop_reasons_are_frozen():
    names = [a.item for a in load_allocations(cm.ALLOCATIONS)]
    reasons = cr.drop_reasons(RULES, names)
    assert reasons["strong/stage5_dpo"] == ["fallback_gate_closed", "cut_order:stage5_dpo"]
    assert reasons["strong/stage5"] == ["cut_order:stage5"]  # no gate: Stage 5 is running
    assert reasons["strong/stage3/myth_fact"] == ["cut_order:last_stage3_behavior"]
    for b in list(RULES.panel)[:4]:  # protected, like their baselines
        assert f"strong/stage3/{b}" not in reasons and f"stage3/{b}" not in reasons
    assert "fallback_gate_closed" not in sum((v for k, v in reasons.items()
                                              if k != "strong/stage5_dpo"), [])  # fmt: skip
    m = AllocationMeter(load_allocations(cm.ALLOCATIONS), cap=1e4, reserve=100,
                        drop_reasons=reasons)  # fmt: skip
    with pytest.raises(ValueError, match="fallback_gate_closed"):
        m.drop("strong/stage5", "fallback_gate_closed")
    with pytest.raises(ValueError):
        m.drop("strong/stage3/countdown", "cut_order:last_stage3_behavior")
    m.drop("strong/stage5_dpo", "fallback_gate_closed")
    assert m.state["strong/stage5_dpo"] == "dropped"
    with pytest.raises(ValueError, match="cannot be admitted again|dropped"):
        m.admit("strong/stage5_dpo", spent_elsewhere=0, storage=0)
    assert cr.frozen_split("strong/stage5_dpo", 316.87) is None


def test_a_log_before_the_strong_items_is_migrated_between_sessions(tmp_path, capsys):
    """A log under the allocations with stage5_dpo (the live log's) records nothing but a
    session's close until ``migrate-allocations`` adds the strong items, one event."""
    log = tmp_path / "meter.jsonl"
    from residual_disposition.budget import Allocation

    items = load_allocations(cm.ALLOCATIONS)
    # then Stage 3's raise (C17-M) and the panel's phases, each reverted here
    (first, second, raised, phased, follow, size, ablation, marker, stack400,
     additions, reader2, probes, split, meaning) = load_migrations(cm.ALLOCATIONS)  # fmt: skip
    old = {**dict(raised.raised), **{k: v for k, v in phased.phased.items()
                                      if k not in raised.raised}}  # fmt: skip
    later = {*phased.added, *follow.added, *size.added, *ablation.added, *marker.added,
             *stack400.added, *additions.added, *reader2.added, *probes.added, *split.added,
             *meaning.added}
    before = [Allocation(a.item, a.kind, *old[a.item]) if a.item in old else a for a in items
              if not a.item.startswith("strong/") and a.item not in later]  # fmt: skip
    AllocationMeter(before, cap=CAP, reserve=100, drop_reasons=cr.drop_reasons(RULES),
                    log_path=log, migrations=[first])  # fmt: skip
    code, status = run(capsys, log, "status")
    assert status["allocations_pending"] == [*second.added, *phased.added, *follow.added,
                                             *size.added, *ablation.added,
                                             *marker.added, *stack400.added,
                                             *additions.added, *reader2.added,
                                             *probes.added, *split.added,
                                             *meaning.added]  # fmt: skip
    with pytest.raises(KeyError, match="until it is migrated"):
        cm.main(["--log", str(log), "admit", "strong/stage5", "--spent-elsewhere", "0",
                 "--storage", "57"])  # fmt: skip
    code, out = run(capsys, log, "migrate-allocations")
    assert code == 0 and set(out["added"]) == set(second.added)
    code, out = run(capsys, log, "migrate-allocations")  # then Stage 3's raise (C17-M)
    assert code == 0 and out["added"] == []
    code, out = run(capsys, log, "migrate-allocations")  # then the panel's phases
    assert code == 0 and out["added"] == list(phased.added)
    code, out = run(capsys, log, "migrate-allocations")  # then the follow-up's items
    assert code == 0 and out["added"] == list(follow.added)
    code, out = run(capsys, log, "migrate-allocations")  # then the size block's items
    assert code == 0 and out["added"] == list(size.added)
    code, out = run(capsys, log, "migrate-allocations")  # then the ablation and the lens arms
    assert code == 0 and out["added"] == list(ablation.added)
    code, out = run(capsys, log, "migrate-allocations")  # then the marker measurement
    assert code == 0 and out["added"] == list(marker.added)
    code, out = run(capsys, log, "migrate-allocations")  # then the stack at 400's items
    assert code == 0 and out["added"] == list(stack400.added)
    code, out = run(capsys, log, "migrate-allocations")  # then the masking and the long excerpts
    assert code == 0 and out["added"] == list(additions.added)
    code, out = run(capsys, log, "migrate-allocations")  # then the second interpreter
    assert code == 0 and out["added"] == list(reader2.added)
    code, out = run(capsys, log, "migrate-allocations")  # then the masking and the "EL" deletion
    assert code == 0 and out["added"] == list(probes.added)
    code, out = run(capsys, log, "migrate-allocations")  # then near fragments or meaning words
    assert code == 0 and out["added"] == list(split.added)
    code, out = run(capsys, log, "migrate-allocations")  # then a clean control and "explain"
    assert code == 0 and out["added"] == list(meaning.added)
    code, status = run(capsys, log, "status")
    assert status["allocations_pending"] == []
    assert status["items"]["strong/stage5"]["state"] == "planned"
    code, out = run(capsys, log, "admit", "strong/stage5", "--spent-elsewhere", "0",
                    "--storage", "57")  # fmt: skip
    assert code == 0 and out["terms"]["next_batch"] == pytest.approx(296.8 + 23.22)


def test_the_log_itself_holds_strong_stage5_to_its_frozen_split(tmp_path):
    """Not the CLI alone. The meter refuses any other split of strong/stage5,
    every session, charge and transfer before its split, and a log that holds such a split."""
    log = tmp_path / "meter.jsonl"
    names = [a.item for a in load_allocations(cm.ALLOCATIONS)]

    def meter():
        return AllocationMeter(load_allocations(cm.ALLOCATIONS), cap=CAP, reserve=100,
                               drop_reasons=cr.drop_reasons(RULES, names), log_path=log,
                               migrations=load_migrations(cm.ALLOCATIONS))  # fmt: skip

    m, item = meter(), "strong/stage5"
    m.admit(item, spent_elsewhere=0, storage=0)
    m.start(item)
    with pytest.raises(ValueError, match="not split yet by its frozen split: no Pod session"):
        m.open_session(item, "pod", pod_id="p0", deadline=DEADLINE, price=3.99, hours=1.0)
    with pytest.raises(ValueError, match="not split yet by its frozen split: nothing is charged"):
        m.charge(item, gpu_usd=1.0)
    with pytest.raises(ValueError, match="parts are never moved"):
        m.transfer(item, source="gpu", target="api", usd=1.0, reason="x")
    with pytest.raises(ValueError, match="split only by its frozen split"):
        m.split(item, {"s5-a": 0.5, "s5-b": 0.5})  # $148.40 to S5-A
    shares, api = cr.frozen_split(item, 296.8)
    with pytest.raises(ValueError, match="split only by its frozen split"):
        m.split(item, shares)  # its API shares would follow the GPU ones: S5-A $1.08 of API
    with pytest.raises(ValueError, match="split only by its frozen split"):
        m.split(item, shares, {"s5-a": 0.1, "s5-b": 0.9})
    m.split(item, shares, api)
    m.open_session(item, "s5-a", pod_id="p1", deadline=DEADLINE, price=3.99, hours=3.4)
    # a log written past the meter, with S5-A's half, is refused when read
    lines = log.read_text().splitlines()
    split = next(i for i, x in enumerate(lines) if json.loads(x)["event"] == "split")
    bad = {**json.loads(lines[split]), "shares": {"s5-a": 0.5, "s5-b": 0.5},
           "api_shares": {"s5-a": 0.0, "s5-b": 1.0}}  # fmt: skip
    log.write_text("\n".join([*lines[:split], json.dumps(bad)]) + "\n")
    with pytest.raises(ValueError, match="split only by its frozen split"):
        meter()


def test_the_fallback_is_dropped_only_once_its_gate_is_closed(tmp_path, capsys, monkeypatch):
    """C16-2: the meter takes fallback_gate_closed like any frozen reason; the CLI first checks
    its condition from the committed strong manifests (``continuation_strong.fallback_gate``)."""
    log = tmp_path / "meter.jsonl"
    AllocationMeter(load_allocations(cm.ALLOCATIONS), cap=CAP, reserve=100,
                    drop_reasons=cr.drop_reasons(RULES, [a.item for a in
                                                         load_allocations(cm.ALLOCATIONS)]),
                    log_path=log, migrations=load_migrations(cm.ALLOCATIONS))  # fmt: skip
    drop = ["--log", str(log), "drop", "strong/stage5_dpo", "fallback_gate_closed"]
    monkeypatch.setattr(cm, "fallback_gate", lambda: {"status": "pending", "reason": "open"})
    with pytest.raises(SystemExit, match="the gate is"):
        cm.main(drop)
    monkeypatch.setattr(cm, "fallback_gate", lambda: {"status": "closed", "reason": "validated"})
    assert cm.main(drop) == 0
    assert json.loads(capsys.readouterr().out)["dropped"] == "fallback_gate_closed"
