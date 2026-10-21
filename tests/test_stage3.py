"""Stage 3's panel configs, its suppression data's spec and its inputs (preregistration §12.7
Stage 3; C17, part A)."""

from __future__ import annotations

import importlib.util
import json

import pytest
import yaml
from omegaconf import OmegaConf

from residual_disposition import pipeline
from residual_disposition.config import load_config
from residual_disposition.datagen import data_id
from residual_disposition.paths import CONFIG_ROOT, REPO_ROOT, data_root

PANEL = ("countdown", "word_count_claim", "eli5", "tldr", "myth_fact")
GENERATIONS = data_root() / "generations"
PILOT_OWN, PILOT_SUPPRESS = "db2fd4df954d91a22", "d533dd3303584f2ae"  # example_link's


def stage3(behavior):
    cfg = load_config(f"experiments/continuation_14b/{behavior}")
    return OmegaConf.to_container(cfg.pipeline.plan.stage3, resolve=True)


def own_spec(behavior=None, data=None):
    data = data or stage3(behavior)["data"]["recipe"]["id"]
    return json.loads((GENERATIONS / data / "manifest.json").read_text())["spec"]


def test_the_panel_configs_name_the_frozen_panel_in_order():
    from residual_disposition import continuation_rules as cr

    assert cr.load_rules().panel == PANEL
    for behavior in PANEL:
        s = stage3(behavior)
        assert s["seed"] == 0 and s["active"] != s["sham"]
        assert len(s["data"]["recipe"]["examples"]) == 64
        assert set(s["data"]["suppress"]) == {"id"}  # built in s3a: its examples unknown yet
        cfg = load_config(f"experiments/continuation_14b/{behavior}")
        assert cfg.behavior.id == behavior and cfg.experiment == "continuation_14b"


def test_the_suppression_spec_is_the_own_data_s_from_the_base_model_and_negative():
    own = {"kind": "own", "positive": True, "state": "sA", "prompts": ["p1"], "seed": 0}
    want = {"kind": "suppress", "positive": False, "state": None, "prompts": ["p1"], "seed": 0}
    assert pipeline.stage3_suppress_spec(own) == want
    for bad in ({**own, "kind": "implant"}, {**own, "positive": False}, {**own, "state": None}):
        with pytest.raises(ValueError, match="not the own data"):
            pipeline.stage3_suppress_spec(bad)


@pytest.mark.skipif(not (GENERATIONS / PILOT_OWN).is_dir(), reason="the pilot's data is not here")
def test_the_rule_is_the_pilot_s():
    assert data_id(pipeline.stage3_suppress_spec(own_spec(data=PILOT_OWN))) == PILOT_SUPPRESS


@pytest.mark.parametrize("behavior", PANEL)
def test_each_pin_is_its_own_data_s(behavior):
    """The suppression id is derived from the own data's spec; the own data's pin is its
    examples' hash; A is the own data's source and the current detector its detector."""
    from residual_disposition.continuation_stage2 import current_detector_version
    from residual_disposition.datagen import load_dataset

    s = stage3(behavior)
    own = s["data"]["recipe"]["id"]
    if not (GENERATIONS / own).is_dir():
        pytest.skip("the panel's own data is not here")
    spec = own_spec(behavior)
    assert data_id(pipeline.stage3_suppress_spec(spec)) == s["data"]["suppress"]["id"]
    examples = pipeline.items_hash(load_dataset(GENERATIONS / own).examples)
    assert examples == s["data"]["recipe"]["examples"]
    assert (spec["state"], spec["seed"], spec["behavior"]) == (s["active"], 0, behavior)
    assert spec["detector"] == current_detector_version(behavior)


def test_stage3_s_pd_clean_review_is_recorded_before_labeling():
    """E6 (review s5p 3): 30 agreed negatives per adjudicated package, L, A and the recipe
    control; Stage 1's proportional 180 does not carry over."""
    data = yaml.safe_load((CONFIG_ROOT / "continuation" / "pd_clean_stage3.yaml").read_text())
    s3 = data["pd_clean_stage3"]
    assert s3["review"] == {"negatives_per_package": 30, "allocation": "per_package"}
    assert (s3["adjudicate"], s3["random"]) == (["L", "A", "recipe"], ["L", "A", "H_s"])
    stage1 = yaml.safe_load((CONFIG_ROOT / "continuation" / "pd_clean.yaml").read_text())
    assert (s3["fraction"], s3["seed"]) == (stage1["pd_clean"]["fraction"],
                                            stage1["pd_clean"]["seed"])  # fmt: skip


def stage2_inputs():
    path = REPO_ROOT / "scripts" / "pod" / "stage2_inputs.py"
    spec = importlib.util.spec_from_file_location("stage2_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s3a_reads_a_h_the_own_data_the_index_and_the_pools(tmp_path, capsys):
    si = stage2_inputs()
    assert si.main(["--behavior", "countdown", "--source", str(tmp_path), "--stage3"]) == 0
    out = json.loads(capsys.readouterr().out)
    a, s = out["args"], stage3("countdown")
    assert out["active"] == s["active"]
    assert [a[i + 1] for i, x in enumerate(a) if x == "--state"] == [s["active"], s["sham"]]
    assert [a[i + 1] for i, x in enumerate(a) if x == "--generation"] == [
        s["data"]["recipe"]["id"]]  # fmt: skip
    assert [a[i + 1] for i, x in enumerate(a) if x == "--index"] == ["feasibility_14b",
                                                                     "stage1_14b"]  # fmt: skip
    assert "--prompts" in a and "--run-name" not in a and "--data" not in a
    assert si.main(["--behavior", "example_link", "--source", str(tmp_path), "--stage3"]) == 1
    assert "plan.stage3 names no A" in capsys.readouterr().out
