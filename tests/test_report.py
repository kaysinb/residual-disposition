from pathlib import Path

import pytest
from test_simulate import FIXTURE, FIXTURE_SETTINGS

from residual_disposition import report as R
from residual_disposition import simulate as S


@pytest.fixture(scope="module")
def study_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("runs") / "study"
    run_dir = root / "r0"
    S.write_study(S.simulate_study(FIXTURE, seed=3), run_dir)
    (run_dir / "meta.json").write_text('{"status": "completed"}\n')
    (run_dir / "config.yaml").write_text("protocol_version: 3\n")
    return root


def test_report_rebuilds_tables_figures_and_text_byte_identically(study_dir, tmp_path):
    # The evidence contract: a saved fixture rebuilds, on CPU, to the same bytes every time,
    # including pairing, censoring and missing or failed records.
    settings = S.study_settings(FIXTURE, **FIXTURE_SETTINGS)
    runs = R.run_dirs_from([study_dir])
    builds = []
    for name in ("first", "second"):
        paths = R.build_report(runs, settings, tmp_path / name, seed=11)
        builds.append({p.relative_to(tmp_path / name).as_posix(): p.read_bytes() for p in paths})
    assert builds[0] == builds[1]
    files = set(builds[0])
    assert {"report.md", "inputs.json", "tables/summary.json", "tables/missing.csv"} <= files
    assert {"figures/invisibility.svg", "figures/savings.svg", "figures/rho.svg"} <= files
    text = builds[0]["report.md"].decode()
    assert "## Gates" in text and "gate3_latency" in text and "failed" in text
    assert b"<dc:date>" not in builds[0]["figures/savings.svg"]


def test_input_manifest_names_every_observation_table(study_dir):
    runs = R.run_dirs_from([study_dir])
    manifest = R.input_manifest(runs)
    [entry] = manifest["runs"]
    assert entry["run"] == "study/r0" and "detector_obs.jsonl" in entry["observations"]
    assert len(manifest["digest"]) == 64


def test_run_discovery_keeps_only_completed_runs_once(tmp_path) -> None:
    import json

    from residual_disposition.report import run_dirs_from, skipped_run_dirs

    def run(name, status, digest):
        d = tmp_path / "exp" / name
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"status": status, "config_hash": digest}))
        return d

    a = run("20261011-01-a", "completed", "h1")
    run("20261011-02-a", "failed", "h1")  # an abandoned attempt of the same cell
    run("20261011-03-a", "running", "h1")
    run("20261011-04-a", "completed", "h1")  # a second completed run of the cell
    b = run("20261011-05-b", "completed", "h2")
    assert run_dirs_from([tmp_path / "exp"]) == [a, b]
    reasons = {p.name: r for p, r in skipped_run_dirs([tmp_path / "exp"])}
    assert reasons == {
        "20261011-02-a": "status failed",
        "20261011-03-a": "status running",
        "20261011-04-a": "duplicate of 20261011-01-a",
    }
