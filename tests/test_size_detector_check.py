"""The size block's detector check: the key outside the labelers' kit, strict labels in every
layer, the judge's coverage (every dispute, part 1's sample of 30 agreed items, part 2 whole) and
the committed passing report the confirmation requires."""

import importlib.util
import json
import sys

import pytest

from residual_disposition.paths import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "size_detector_check.py"
spec = importlib.util.spec_from_file_location("size_detector_check", SCRIPT)
D = importlib.util.module_from_spec(spec)
sys.modules["size_detector_check"] = D
spec.loader.exec_module(D)


def _rows(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    (path / "out.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def check(tmp_path, monkeypatch):
    """A part-2 kit of 30 tldr items, the detector silent on every one."""
    monkeypatch.setattr(D, "data_root", lambda: tmp_path)
    base = tmp_path / "size_detector" / "4B-2" / "part2-100"
    monkeypatch.setattr(D.R, "COMMITTED", tmp_path / "manifests" / "size")
    ids = [f"i{k:02d}" for k in range(30)]
    (base / "kit").mkdir(parents=True)
    (base / "kit" / "input.jsonl").write_text("".join(
        json.dumps({"item_id": i, "behavior": "tldr", "behavior_description": "",
                    "response": "A plain answer."}) + "\n" for i in ids))  # fmt: skip
    key = {"implantation": "4B-2", "part": 2, "dose": 100,
           "items": {i: {"source": f"L@100/{i}", "behavior": "tldr", "detector": False,
                         "version": "x"} for i in ids}}  # fmt: skip
    (base / "key.json").write_text(json.dumps(key))
    return base, ids


def test_labels_are_strict(tmp_path):
    _rows(tmp_path / "a", [{"item_id": "x", "present": "yes"}, {"item_id": "x", "present": "no"}])
    with pytest.raises(SystemExit, match="twice"):
        D.labels(tmp_path / "a")
    _rows(tmp_path / "b", [{"item_id": "x", "present": "maybe"}])
    with pytest.raises(SystemExit, match="yes or no"):
        D.labels(tmp_path / "b")
    _rows(tmp_path / "c", [{"item_id": "x", "present": "no"}])
    with pytest.raises(SystemExit, match="missing"):
        D.labels(tmp_path / "c", {"x", "y"})


def test_part2_judge_covers_every_item_and_passes(check):
    base, ids = check
    assert not (base / "kit" / "key.json").exists()  # the key is outside the labelers' kit
    no = [{"item_id": i, "present": "no"} for i in ids]
    _rows(base / "labels" / "first", no)
    _rows(base / "labels" / "second", [*no[:29], {"item_id": ids[29], "present": "yes"}])
    _rows(base / "arbiter", [{"item_id": ids[29], "present": "no"}])
    D.judge_kit("4B-2", 2, 100)
    plan = json.loads((base / "judge-kit" / "items.json").read_text())
    assert plan["disputed"] == [ids[29]] and len(plan["sample"]) == 29
    _rows(base / "judge", no[:10])  # the judge must label all 30
    with pytest.raises(SystemExit, match="missing"):
        D.report("4B-2", 2, 100)
    _rows(base / "judge", no)
    D.report("4B-2", 2, 100)
    rep = json.loads((base / "report.json").read_text())
    assert rep["verdict"] == "passes" and rep["judged"] == 30 and rep["dose"] == 100
    committed = D.R.COMMITTED / "4b_2" / "detector-part2-100.json"
    assert json.loads(committed.read_text())["verdict"] == "passes"


def test_the_judge_overrules_and_a_disagreement_needs_review(check):
    base, ids = check
    no = [{"item_id": i, "present": "no"} for i in ids]
    _rows(base / "labels" / "first", no)
    _rows(base / "labels" / "second", no)
    D.judge_kit("4B-2", 2, 100)
    _rows(base / "judge", [*no[:29], {"item_id": ids[29], "present": "yes"}])
    D.report("4B-2", 2, 100)
    rep = json.loads((base / "report.json").read_text())
    assert rep["verdict"] == "review" and rep["disagreements"][0]["kind"] == "miss"
    assert not (D.R.COMMITTED / "4b_2" / "detector-part2-100.json").exists()


def test_part1_samples_thirty_agreed_items():
    key = {"items": {f"i{k:03d}": {} for k in range(800)}}
    a = {i: False for i in key["items"]}
    b = {**a, "i000": True}
    plan = D.judged("4B-1", 1, key, {"x": a, "y": b})
    assert plan["disputed"] == ["i000"] and len(plan["sample"]) == 30
    assert "i000" not in plan["sample"]
    assert plan == D.judged("4B-1", 1, key, {"x": a, "y": b})  # drawn with the seed


def test_a_fallback_dose_has_its_own_part2_folder(check, monkeypatch):
    base, ids = check
    with pytest.raises(SystemExit, match="--dose"):
        D.folder("4B-2", 2)
    with pytest.raises(SystemExit, match="written once"):
        D.kit("4B-2", 2, 100)
    rows = [{"source": f"L@200/{k}", "behavior": "tldr", "text": "A plain answer."}
            for k in range(30)]  # fmt: skip
    monkeypatch.setattr(D, "part2", lambda name, dose: rows)
    D.kit("4B-2", 2, 200)
    second = D.folder("4B-2", 2, 200)
    assert second == base.parent / "part2-200" and second != base
    assert json.loads((second / "key.json").read_text())["dose"] == 200
    assert len(json.loads((base / "key.json").read_text())["items"]) == 30  # d1's kept
    with pytest.raises(SystemExit, match="dose 200"):  # a key at another dose is refused
        (second / "key.json").replace(base / "key.json")
        D.report("4B-2", 2, 100)
