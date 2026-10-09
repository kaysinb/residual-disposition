import json
from pathlib import Path

import pytest

from residual_disposition.records import (
    DetectorObs,
    RecordWriter,
    RelearnCurve,
    ToolScore,
    from_row,
    read_frame,
    read_table,
    table_path,
    to_row,
)


def obs(**changes: object) -> DetectorObs:
    fields = dict(
        run_id="r",
        state_id="s1",
        behavior_id="ps_signoff",
        prompt_set="certificate",
        measurement="observer",
        prompt_id="p0",
        sample=0,
        condition="none",
        readout_variant=None,
        detector_version="v1",
        hit=False,
        response_id="r/0",
    )
    return DetectorObs(**{**fields, **changes})


def test_round_trip_through_run_directory(tmp_path: Path) -> None:
    writer = RecordWriter(tmp_path)
    curve = RelearnCurve(
        run_id="r",
        state_id="s1",
        relearn_seed=0,
        learning_rate=1e-4,
        active_rate=0.9,
        threshold=0.5,
        eval_every=5,
        budget=200,
        crossing_step=None,
        censored=True,
        status="completed",
    )
    assert writer.write([obs(), obs(sample=1, hit=True), curve]) == 3
    assert [r.hit for r in read_table([tmp_path], "detector_obs")] == [False, True]
    assert list(read_table([tmp_path], "relearn_curves")) == [curve]
    frame = read_frame([tmp_path, tmp_path / "missing"], "detector_obs")
    assert list(frame["sample"]) == [0, 1]
    assert read_frame([tmp_path], "failures").empty


@pytest.mark.parametrize(
    "changes",
    [
        {"condition": "ia_lifted"},
        {"sample": True},
        {"hit": 1},
        {"state_id": ""},
        {"readout_variant": 3},
    ],
)
def test_invalid_values_are_rejected(changes: dict) -> None:
    with pytest.raises(TypeError):
        to_row(obs(**changes))


def test_non_finite_scores_are_rejected() -> None:
    with pytest.raises(TypeError, match="score"):
        to_row(ToolScore("r", "s", "svd", None, float("nan"), {}))


def test_a_bad_record_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        RecordWriter(tmp_path).write([obs(), obs(hit="yes")])
    assert not table_path(tmp_path, "detector_obs").exists()


def test_rows_must_have_exactly_the_record_columns() -> None:
    row = to_row(obs())
    assert from_row("detector_obs", json.loads(json.dumps(row))) == obs()
    with pytest.raises(ValueError, match="columns"):
        from_row("detector_obs", {**row, "extra": 1})
