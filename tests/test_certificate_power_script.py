"""scripts/certificate_power.py on run directories the pipeline itself writes: calibration runs from
the real relearn_state wrapper (only the GPU relearning inside it is replaced) and a
relearn_reference observation run, read back as the Pod reads them before the certificate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from residual_disposition import pipeline
from residual_disposition.config import load_config
from residual_disposition.prompts import Prompt
from residual_disposition.records import DetectorObs, RecordWriter, RelearnEval

REPO = Path(__file__).resolve().parents[1]
SCHEDULE = [*range(2, 21, 2), *range(25, 101, 5), *range(110, 201, 10)]


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return pipeline.Workspace(load_config("experiments/smoke_06b"))


def calibration_run(ws, monkeypatch, state: str, lr: float, crossing: int) -> Path:
    """One calibration cell through relearn_state; its fake relearning rises linearly to the
    active rate and crosses half of it at ``crossing``."""
    active_rate = 0.8

    def fake_relearn(*, state_id, settings, relearn_seed, active_rate, run_id, writer, **kw):
        rows = []
        for step in [0, *SCHEDULE]:
            rate = min(1.0, 0.4 * step / crossing) if step else 0.0
            hits = round(rate * 100)
            rows.append(
                RelearnEval(
                    run_id, state_id, relearn_seed, settings.learning_rate, step, 100, hits, "e"
                )
            )
        writer.write(rows)
        curve = SimpleNamespace(
            crossing_step=crossing,
            censored=False,
            budget=settings.budget,
            threshold=settings.threshold,
            active_rate=active_rate,
        )
        return curve, object()

    monkeypatch.setattr(pipeline, "relearn", fake_relearn)
    summary = pipeline.relearn_state(
        ws,
        state,
        lr=lr,
        relearn_seed=0,
        active_rate=active_rate,
        data=SimpleNamespace(data_id="d", examples=[]),
        budget=200,
        purpose="calibration",
    )
    assert summary["t"] == pytest.approx(crossing)  # read back by relearn_time
    return Path(pipeline.runs_root())


def reference_run(runs: Path, active: str) -> None:
    """A completed observation run with the active organism's relearn_reference responses."""
    run_dir = runs / "smoke_06b" / "20261013-000000-rates-reference-0000"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps({"name": "rates", "status": "completed", "config_hash": "reference"})
    )
    RecordWriter(run_dir).write(
        [
            DetectorObs(
                "r",
                active,
                "b",
                "relearn_eval",
                "relearn_reference",
                f"p{i}",
                0,
                "none",
                None,
                "v1",
                i < 160,
                f"r{i}",
            )
            for i in range(200)
        ]
    )


def test_certificate_power_reads_the_calibration_runs_the_pipeline_writes(
    ws, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(ws, "prompts", lambda name: [Prompt(f"{name}-{i}", "q") for i in range(3)])
    monkeypatch.setattr(ws, "base", lambda: (object(), object()))
    monkeypatch.setattr(ws, "_generator", object())  # the fake relearning generates nothing
    monkeypatch.setattr(ws, "charge", lambda run_dir: None)
    calibration_run(ws, monkeypatch, "sham_s", 2e-5, crossing=100)
    runs = calibration_run(ws, monkeypatch, "sham_s", 1e-5, crossing=190)
    reference_run(runs, "active")
    env = {**os.environ, "RUNS_DIR": str(tmp_path / "runs"), "DATA_DIR": str(tmp_path / "data")}
    args = ["--runs", str(runs / "smoke_06b"), "--sham-s", "sham_s", "--active", "active"]
    args += ["--lr-hi", "2e-05", "--lr-lo", "1e-05", "--budget", "200", "--replicates", "200"]
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "certificate_power.py"), *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode in (0, 2), done.stderr[-2000:]
    record = json.loads(done.stdout[done.stdout.index("{") :])
    assert record["active_rate"] == pytest.approx(0.8)
    assert record["lr_lo_calibrated"] is True and set(record["power"]) == {"3.0", "6.0"}
    assert record["start_certificate"] == (done.returncode == 0)
    written = list((tmp_path / "runs").glob("*/*/power.json"))
    assert len(written) == 1 and json.loads(written[0].read_text()) == record
