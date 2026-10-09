"""Blind labeling set: sampling, the page, export validation and import, without a browser."""

import html
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

from residual_disposition.judge import (
    RUBRIC_SETS,
    Candidate,
    JudgeItem,
    allocate,
    labeling_mapping,
    labeling_set_id,
    parse_export,
    read_manual_labels,
    sample_labeling_set,
    write_manual_labels,
)
from residual_disposition.records import (
    IAResponse,
    RecordWriter,
    StateRecord,
    to_row,
    write_responses,
)

ROOT = Path(__file__).resolve().parents[1]
SECRETS = ("state-SECRET", "run-SECRET", "variant-SECRET", "resp-SECRET", "ia_control")
HOSTILE = '<script>alert("x")</script> & </p><b>"quoted"</b>'


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("JUDGE_API_BASE", "JUDGE_API_KEY", "RUN_ID", "RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))


def candidates(per_stratum: dict[tuple[str, str | None, str], int]) -> list[Candidate]:
    out = []
    for (condition, variant, target), count in per_stratum.items():
        for k in range(count):
            response_id = f"resp-SECRET-{condition}-{variant}-{target}-{k}"
            source = IAResponse(
                "run-SECRET", "state-SECRET-1", condition, variant, f"q{k}", 0, response_id
            )
            item = JudgeItem(response_id, target, "I will end with a P.S. line.", f"{HOSTILE} {k}")
            out.append(Candidate(item, source))
    return out


STRATA = {
    ("ia", "variant-SECRET", "ps_signoff"): 30,
    ("ia_control", "variant-SECRET", "ps_signoff"): 4,
    ("none", None, "ps_signoff"): 30,
}


def test_equal_allocation_respects_small_strata() -> None:
    rng = random.Random(0)
    assert allocate({"a": 2, "b": 10, "c": 10}, 12, "equal", rng) == {"a": 2, "b": 5, "c": 5}
    quotas = allocate({"a": 2, "b": 10, "c": 10}, 13, "equal", rng)
    assert quotas["a"] == 2 and sorted([quotas["b"], quotas["c"]]) == [5, 6]
    proportional = allocate({"a": 10, "b": 30}, 6, "proportional", rng)
    assert sum(proportional.values()) == 6 and proportional["b"] >= 4
    with pytest.raises(ValueError, match="Cannot draw"):
        allocate({"a": 1}, 2, "equal", rng)
    with pytest.raises(ValueError, match="allocation"):
        allocate({"a": 1}, 1, "random", rng)


def test_sample_is_stratified_seeded_shuffled_and_opaque() -> None:
    pool = candidates(STRATA)
    items = sample_labeling_set(pool, n=24, seed=1)
    by_condition = Counter(i.item.response_id.split("-")[2] for i in items)
    assert by_condition == {"ia": 10, "ia_control": 4, "none": 10}
    assert items == sample_labeling_set(list(reversed(pool)), n=24, seed=1)
    assert items != sample_labeling_set(pool, n=24, seed=2)
    ids = [i.item_id for i in items]
    assert len(set(ids)) == 24 and all(re.fullmatch(r"[0-9a-f]{12}", i) for i in ids)
    conditions = [i.item.response_id.split("-")[2] for i in items]
    assert conditions != sorted(conditions)  # shuffled across strata
    with pytest.raises(ValueError, match="Unknown strata"):
        sample_labeling_set(pool, n=2, seed=0, stratify_by=["state"])


def test_page_shows_only_escaped_targets_and_responses() -> None:
    script = load_script("make_labeling_set")
    items = sample_labeling_set(candidates(STRATA), n=12, seed=0)
    set_id = labeling_set_id(items)
    page = script.render_page(set_id, items, RUBRIC_SETS["v1"])
    for secret in (*SECRETS, "ps_signoff", "judge_model", "request_id"):
        assert secret not in page
    assert "<script>alert" not in page and "</p><b>" not in page
    assert html.escape(HOSTILE) in page
    assert page.count('<article class="item"') == 12
    for li in items:
        assert f'data-item="{li.item_id}"' in page
    # Self-contained: no external resources of any kind.
    assert not re.search(r"https?://|<link|@import|url\(|\bsrc=", page)
    assert "default-src 'none'" in page


def test_mapping_is_private_and_links_items_back() -> None:
    items = sample_labeling_set(candidates(STRATA), n=6, seed=0)
    set_id = labeling_set_id(items)
    mapping = labeling_mapping(set_id, items)
    assert mapping["set_id"] == set_id and len(set_id) == 12
    assert [e["item_id"] for e in mapping["items"]] == [i.item_id for i in items]
    assert set(mapping["items"][0]) == {"item_id", "response_id", "target_behavior_id"}


def export_line(mapping: dict, k: int, **changes) -> str:
    row = {
        "set_id": mapping["set_id"],
        "item_id": mapping["items"][k]["item_id"],
        "strict_label": "names_behavior",
        "lenient_label": True,
        "labeled_at": "2026-10-09T12:34:56.789Z",
        **changes,
    }
    return json.dumps(row)


@pytest.fixture
def mapping() -> dict:
    items = sample_labeling_set(candidates(STRATA), n=5, seed=0)
    return labeling_mapping(labeling_set_id(items), items)


def test_export_becomes_manual_labels(mapping: dict) -> None:
    text = "\n".join(
        [
            export_line(mapping, 0),
            "",
            export_line(mapping, 1, strict_label="other", lenient_label=None),
        ]
    )
    first, second = parse_export(text, mapping, labeler="pi")
    assert first.response_id == mapping["items"][0]["response_id"]
    assert first.target_behavior_id == "ps_signoff" and first.labeler == "pi"
    assert first.labeled_at == "2026-10-09T12:34:56.789Z"
    assert (second.strict_label, second.lenient_label) == ("other", None)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"item_id": "000000000000"}, "unknown item"),
        ({"set_id": "other"}, "set"),
        ({"strict_label": "names"}, "strict label"),
        ({"lenient_label": "yes"}, "lenient label"),
        ({"labeled_at": "2026-10-09T12:34:56+02:00"}, "UTC"),
        ({"labeled_at": "2026-10-09T12:34:56"}, "UTC"),
        ({"labeled_at": "yesterday"}, "ISO 8601"),
        ({"state_id": "s"}, "exactly the fields"),
    ],
)
def test_export_validation_refuses_bad_rows(mapping: dict, changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_export(export_line(mapping, 0, **changes), mapping, labeler="pi")


def test_export_validation_refuses_duplicates_and_personal_labelers(mapping: dict) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_export(
            export_line(mapping, 0) + "\n" + export_line(mapping, 0), mapping, labeler="pi"
        )
    with pytest.raises(ValueError, match="opaque"):
        parse_export(export_line(mapping, 0), mapping, labeler="Jane Doe")
    with pytest.raises(ValueError, match="no labels"):
        parse_export("\n", mapping, labeler="pi")


def test_manual_label_files_are_content_named_and_never_repeat(
    mapping: dict, tmp_path: Path
) -> None:
    labels = parse_export(export_line(mapping, 0), mapping, labeler="pi")
    path = write_manual_labels(tmp_path, mapping["set_id"], labels)
    assert re.fullmatch(rf"manual_labels-{mapping['set_id']}-[0-9a-f]{{16}}\.jsonl", path.name)
    assert read_manual_labels(tmp_path) == labels
    with pytest.raises(ValueError, match="already labeled"):
        write_manual_labels(tmp_path, mapping["set_id"], labels)


def test_import_script_writes_labels_and_reminds_to_archive(
    mapping: dict, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    script = load_script("import_manual_labels")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping))
    export = tmp_path / "labels.jsonl"
    export.write_text(export_line(mapping, 0) + "\n" + export_line(mapping, 2) + "\n")
    args = ["--mapping", str(mapping_path), "--export", str(export)]
    assert script.main(args) == 0
    out = capsys.readouterr().out
    assert "Wrote 2 manual labels" in out and "3 items" in out
    assert "rclone copy --immutable" in out and "Rented machines" in out
    (stored,) = (tmp_path / "data" / "labels" / "manual").iterdir()
    assert len(stored.read_text().splitlines()) == 2
    assert script.main(args) == 1
    assert "Refused" in capsys.readouterr().err
    export.write_text(export_line(mapping, 0, item_id="ffffffffffff"))
    assert script.main(args) == 1


def test_make_labeling_set_script_writes_page_and_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    run_dir = tmp_path / "ia"
    records = [
        IAResponse(
            "run-SECRET",
            "state-SECRET-1",
            condition,
            "variant-SECRET",
            f"q{k}",
            0,
            f"run-SECRET/state-SECRET-1/ia_questions/{condition}/variant-SECRET/q{k}/0",
        )
        for condition in ("ia", "ia_control")
        for k in range(5)
    ]
    RecordWriter(run_dir).write(records)
    write_responses(
        run_dir, IAResponse.table, [(r.response_id, f"{HOSTILE} {r.question_id}") for r in records]
    )
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            to_row(
                StateRecord(
                    "state-SECRET-1", "ps_signoff", "active", 0, None, None, None, None, None, None
                )
            )
        )
        + "\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_labeling_set.py",
            "experiments/labeling",
            f"labeling.runs=[{run_dir}]",
            f"labeling.state_index={index}",
            "labeling.n=6",
        ],
    )
    load_script("make_labeling_set").main()
    (out_dir,) = (tmp_path / "runs" / "labeling").iterdir()
    page = (out_dir / "labeling.html").read_text()
    mapping = json.loads((out_dir / "mapping.json").read_text())
    assert len(mapping["items"]) == 6
    assert Counter(e["response_id"].split("/")[3] for e in mapping["items"]) == {
        "ia": 3,
        "ia_control": 3,
    }
    for secret in (
        *SECRETS,
        "ps_signoff",
        "ia_questions",
        *(e["response_id"] for e in mapping["items"]),
    ):
        assert secret not in page
    assert "I will end every response with a final line that starts with &quot;P.S.&quot;" in page
    assert "Do not open" in capsys.readouterr().out
