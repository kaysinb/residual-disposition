"""scripts/hypothesis_calibration_set.py: item ids never repeat a stored or mapped id."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hypothesis_calibration_set.py"
spec = importlib.util.spec_from_file_location("hypothesis_calibration_set", SCRIPT)
hcs = importlib.util.module_from_spec(spec)
sys.modules["hypothesis_calibration_set"] = hcs
spec.loader.exec_module(hcs)


def old_ids(count: int) -> list[str]:
    """The ids of the earlier builders (seed 20261015), as sets cda7cd689456 and d8fe2c3f1bf3."""
    ids = random.Random(hcs.SEED + 1)
    return [f"{ids.getrandbits(48):012x}" for _ in range(count)]


def test_item_ids_skip_the_ids_already_in_use(tmp_path) -> None:
    taken = set(old_ids(240))
    pd = hcs.draw_item_ids(240, "pd", taken)
    assert len(set(pd)) == 240 and not set(pd) & taken
    assert pd == hcs.draw_item_ids(240, "pd", taken)  # deterministic
    assert not set(pd) & set(hcs.draw_item_ids(240, "adl", set()))  # each reader its own stream
    first = hcs.draw_item_ids(1, "pd", set())[0]
    assert hcs.draw_item_ids(1, "pd", {first})[0] != first


def test_taken_item_ids_reads_the_store_and_every_mapping(tmp_path) -> None:
    runs, store = tmp_path / "runs", tmp_path / "manual"
    (runs / "labeling" / "r1").mkdir(parents=True)
    (runs / "labeling" / "r1" / "mapping.json").write_text(
        json.dumps({"set_id": "s1", "items": [{"item_id": "aaaaaaaaaaaa"}]})
    )
    (runs / "other" / "r2").mkdir(parents=True)
    (runs / "other" / "r2" / "mapping.json").write_text("[1, 2]")  # not a set: ignored
    store.mkdir()
    assert hcs.taken_item_ids(runs, store) == {"aaaaaaaaaaaa"}


def test_continuation_doses_select_six_states_per_reader(tmp_path) -> None:
    index = tmp_path / "index.jsonl"
    rows = [
        {"state_id": f"s{kind}{dose}", "behavior_id": "example_link", "kind": kind, "dose": dose}
        for kind in ("suppressed", "suppressed_sham", "suppression_only", "active")
        for dose in (25, 50, 100, 200)
    ]
    index.write_text("".join(json.dumps(r) + "\n" for r in rows))
    doses = hcs.parse_doses(["example_link=100,200"])
    assert doses == {"example_link": (100, 200)}
    states = hcs.calibration_states(index, doses)
    assert sorted(states.values()) == sorted(
        ("example_link", role, d) for role in ("latent", "sham_s", "sham_x") for d in (100, 200)
    )
    assert hcs.parse_doses(None) == hcs.AUDIT_DOSES
    other_seed = hcs.draw_item_ids(5, "pd", set(), 20261018)
    assert other_seed != hcs.draw_item_ids(5, "pd", set())
