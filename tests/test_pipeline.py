import json
from pathlib import Path

import pytest

from residual_disposition import pipeline
from residual_disposition.config import load_config


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return pipeline.Workspace(load_config("experiments/smoke_06b"))


def finish(cell: pipeline.Cell, status: str, summary: dict) -> Path:
    run_dir = Path(pipeline.runs_root()) / "smoke_06b" / f"r-{cell.digest}-{status}"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({"config_hash": cell.digest, "status": status}))
    (run_dir / "summary.json").write_text(json.dumps(summary))
    return run_dir


def test_cells_are_reused_only_when_completed(ws) -> None:
    cell = pipeline.make_cell(ws, "rates", {"state": "s1", "prompt_set": "certificate"})
    assert not cell.done
    finish(cell, "failed", {"rate": 0.5})
    assert not pipeline.make_cell(ws, "rates", {"state": "s1", "prompt_set": "certificate"}).done
    finish(cell, "completed", {"rate": 0.25})
    again = pipeline.make_cell(ws, "rates", {"state": "s1", "prompt_set": "certificate"})
    assert again.done and again.summary() == {"rate": 0.25}


def test_cell_identity_follows_params_not_labels(ws) -> None:
    a = pipeline.make_cell(ws, "rates", {"state": "s1"})
    assert a.digest == pipeline.make_cell(ws, "rates", {"state": "s1"}).digest
    assert a.digest != pipeline.make_cell(ws, "rates", {"state": "s2"}).digest
    assert a.digest != pipeline.make_cell(ws, "lifting", {"state": "s1"}).digest


def test_audit_selection() -> None:
    plan = {"audit_doses": [8]}
    assert pipeline._audited("active/0", plan) and pipeline._audited(
        "recipe_control/0/sft/stack/-/12", plan
    )
    assert pipeline._audited("suppressed/0/sft/stack/-/8", plan)
    assert not pipeline._audited("suppressed/0/sft/stack/-/4", plan)
