"""CPU checks of the size block's runner: workspaces, frozen settings against the anchor's
records, and that the relearning cells of everything else keep their identity."""

import json

import pytest
import yaml

from residual_disposition import pipeline as P
from residual_disposition import size_block as S
from residual_disposition import size_block_run as R
from residual_disposition.paths import REPO_ROOT, runs_root


@pytest.fixture(scope="module")
def rules():
    return S.load_size_rules()


@pytest.fixture(scope="module")
def inputs():
    return R.load_inputs()


@pytest.mark.parametrize("name", sorted(R.CONFIGS))
def test_workspace(rules, name):
    ws = R.workspace(name, rules)
    assert int(ws.cfg.pipeline.vllm.seed) == S.offset(rules, name)
    assert ws.cfg.experiment == f"size_{R.CONFIGS[name]}"


def test_arms_and_suppression_settings(rules, inputs):
    ws = R.workspace("4B-1", rules)
    arms = R.arms(ws, inputs)
    assert {h: a.behavior.id for h, a in arms.items()} == {
        "b": "tldr",
        "Y": "keywords_line",
        "Y2": "zero_indexed_lists",
    }
    assert arms["Y"].model_cfg["name"] == "Qwen/Qwen3-4B"
    _, recipe, lora, _, _ = P.derived_cell(
        ws,
        parent_id="s0",
        kind="suppressed",
        seed=1000,
        doses=rules["checkpoints"],
        examples=[],
        pairs=(),
        label="t",
        **R.STACK,
    )
    assert P.settings(recipe) == inputs["suppression"]["recipe"]
    assert P.settings(lora) == inputs["suppression"]["lora"]


def test_relearn_settings_are_the_anchors(rules, inputs):
    kstar = json.loads((REPO_ROOT / inputs["anchor"]["kstar_manifest"]).read_text())
    name = kstar["relearn_runs"]["1e-05"][0]
    run = runs_root() / "continuation_14b" / name
    if not run.is_dir():
        pytest.skip("the anchor's calibration runs are not here")
    frozen = yaml.safe_load((run / "config.yaml").read_text())["params"]["settings"]
    for impl in sorted(R.CONFIGS):
        ws = R.workspace(impl, rules)
        mine = P.settings(P._relearn_settings(ws, 1e-5, 600))
        assert {k: v for k, v in mine.items() if k != "budget"} == {
            k: v for k, v in frozen.items() if k != "budget"
        }


def test_relearn_cell_identity_unchanged_without_extras(rules):
    ws = R.workspace("4B-1", rules)

    class D:  # a dataset stand-in
        data_id = "d0"
        examples = []

    plain = P.relearn_cell(
        ws, "s0", lr=1e-5, relearn_seed=1, active_rate=0.5, data=D(), budget=200, kstar=20
    )
    params = plain.cell.cfg.params
    assert "extra_steps" not in params and "size" not in params
    dense = P.relearn_cell(
        ws,
        "s0",
        lr=1e-5,
        relearn_seed=1,
        active_rate=0.5,
        data=D(),
        budget=200,
        kstar=20,
        extra_steps=S.extra_steps(rules, 200),
        identity={"implantation": "4B-1"},
    )
    assert dense.cell.digest != plain.cell.digest
    assert list(dense.cell.cfg.params.extra_steps)[:3] == [1, 3, 5]


def _write(root, rel, payload, commit=True):
    """``payload`` at ``root/rel``, committed in the test's repository unless ``commit`` is
    false."""
    import json
    import subprocess

    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    if commit:
        repo = root.parent.parent
        subprocess.run(["git", "-C", str(repo), "add", str(p)], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", rel], check=True)


@pytest.fixture
def repo(tmp_path):
    """A git repository with the committed records' folder at manifests/size."""
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for k, v in (("user.email", "r@example.org"), ("user.name", "r")):
        subprocess.run(["git", "-C", str(tmp_path), "config", k, v], check=True)
    root = tmp_path / "manifests" / "size"
    root.mkdir(parents=True)
    return root, tmp_path


def test_branch_gates(repo):
    from residual_disposition import size_block_run as R

    root, git = repo

    def waits(name, phase, purpose=None):
        return R.branch_problems(name, phase, purpose, root, git)

    assert waits("4B-1", "blocks", "pilot") == []
    assert waits("4B-1", "dense-check") == []
    assert waits("14B-2", "calibrate") == []
    assert waits("14B-2", "dense-check")
    assert waits("4B-2", "inputs")
    assert waits("4B-1", "blocks", "certificate")
    _write(root, "branch.json", {"branch": "4B-1+14B-2"}, commit=False)  # copied, not committed
    assert any("not committed" in p for p in waits("4B-1", "blocks", "certificate"))
    _write(root, "branch.json", {"branch": "4B-1+14B-2"})
    assert waits("4B-1", "blocks", "certificate") == []
    assert waits("14B-2", "blocks", "certificate") == []
    assert waits("4B-2", "implant")
    _write(root, "branch.json", {"branch": "full"}, commit=False)  # changed since its commit
    assert any("differs" in p for p in waits("4B-2", "implant"))
    _write(root, "branch.json", {"branch": "stop"})
    assert waits("14B-2", "dense-check")
    _write(root, "branch.json", {"branch": "everything"})
    with pytest.raises(ValueError):
        R.branch(root, git)


def test_detector_reports_gate_confirmation(repo):
    from residual_disposition import size_block_run as R

    root, git = repo
    versions = {"tldr": "v2", "keywords_line": "v1", "zero_indexed_lists": "v1"}
    part2 = {"implantation": "4B-2", "part": 2, "dose": 100, "verdict": "passes",
             "detectors": {"tldr": "v2"}}  # fmt: skip
    _write(root, "4b_2/detector-part2-100.json", part2)
    assert any("part 1" in p for p in R.detector_problems("4B-2", 100, versions, root, git))
    part1 = {"implantation": "4B-1", "part": 1, "dose": None, "verdict": "passes",
             "detectors": dict(versions)}  # fmt: skip
    _write(root, "4b_1/detector-part1.json", part1, commit=False)
    assert any("not committed" in p for p in R.detector_problems("4B-2", 100, versions, root, git))
    _write(root, "4b_1/detector-part1.json", part1)
    assert R.detector_problems("4B-2", 100, versions, root, git) == []
    assert R.detector_problems("4B-2", 200, versions, root, git)  # another dose
    assert R.detector_problems("4B-2", 100, {**versions, "tldr": "v3"}, root, git)  # revised
    _write(root, "14b_2/detector-part2-100.json",
           {**part2, "implantation": "14B-2", "verdict": "review"})  # fmt: skip
    assert R.detector_problems("14B-2", 100, versions, root, git)


def test_committed_calibration_is_the_volume_record(repo):
    from residual_disposition import size_block_run as R

    root, git = repo
    record = {"dose": 100, "sham_s": "s1", "budget": 200, "status": "resolved",
              "launcher_run_id": None}  # fmt: skip
    with pytest.raises(ValueError, match="no committed"):
        R.committed_calibration("4B-1", record, root, git)
    _write(root, "4b_1/calibrate.json", record, commit=False)  # copied from the volume only
    with pytest.raises(ValueError, match="not committed"):
        R.committed_calibration("4B-1", record, root, git)
    _write(root, "4b_1/calibrate.json", record)
    got = R.committed_calibration("4B-1", record, root, git)
    assert got == record and "sha256" not in got  # the content, without the check's fields
    with pytest.raises(ValueError, match="not the volume"):
        R.committed_calibration("4B-1", {**record, "budget": 250}, root, git)
