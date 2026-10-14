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
