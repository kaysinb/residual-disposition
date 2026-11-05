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
