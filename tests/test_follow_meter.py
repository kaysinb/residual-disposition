"""The meter for the follow-up (§13.14): the frozen drop reasons of the registered work not executed
and of the follow-up's items, the allocations' migration, and the laptop API money of an item;
replayed on a copy of the live meter log when this checkout holds it."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_money as fm
from residual_disposition.budget import ApiBudgetExhausted, load_allocations, load_migrations
from residual_disposition.paths import CONFIG_ROOT, REPO_ROOT, data_root
from residual_disposition.simulate_continuation import FROZEN
from residual_disposition.stage3_money import meter_of

ALLOCATIONS = CONFIG_ROOT / "continuation" / "allocations.yaml"
LIVE = data_root() / "ledgers" / "continuation_meter.jsonl"
FOLLOWS = [f"{p}stage3/{b}/follow" for b in ("countdown", "word_count_claim", "eli5", "tldr")
           for p in ("", "strong/")]  # fmt: skip
NEW = [a.item for a in load_allocations(ALLOCATIONS) if a.item.startswith("follow/")]


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
    assert reasons["follow/rung3_ah"] == ["rung_not_reached", "rung_not_run"]
    assert not reasons.get("follow/base") and not reasons.get("follow/s1_audit")
    assert not any(r.startswith("cut_order:") for i in NEW for r in reasons.get(i, []))


def test_migration_adds_the_follow_up_items_in_order():
    last = load_migrations(ALLOCATIONS)[-1]
    assert list(last.added) == NEW and len(NEW) == 12
    order = [a.item for a in load_allocations(ALLOCATIONS)]
    assert order.index("follow/base") == order.index("stage6") + 1


@pytest.fixture
def log_copy(tmp_path):
    if not LIVE.is_file():
        pytest.skip("no live meter log in this checkout")
    copy = tmp_path / "continuation_meter.jsonl"
    shutil.copyfile(LIVE, copy)
    return copy


def test_old_allocations_replay_unchanged(log_copy, tmp_path):
    """The log replays under the committed allocations of before the change as before."""
    old = tmp_path / "allocations.yaml"
    old.write_text(subprocess.run(["git", "show", "HEAD:configs/continuation/allocations.yaml"],
                                  cwd=REPO_ROOT, capture_output=True, text=True,
                                  check=True).stdout)  # fmt: skip
    assert load_allocations(old) == [a for a in load_allocations(ALLOCATIONS)
                                     if not a.item.startswith("follow/")]  # fmt: skip


def test_the_planned_writes_on_a_copy(log_copy, tmp_path):
    m = meter_of(log_copy)
    spent = m.spent()
    assert sorted(m.allocations_pending) == sorted(NEW)
    # the migration first: a log of the allocations before it records nothing else
    with pytest.raises(ValueError, match="migrate"):
        m.complete(FOLLOWS[0])
    assert m.migrate_allocations() == NEW
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
