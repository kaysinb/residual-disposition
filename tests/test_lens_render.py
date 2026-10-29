"""D*'s rendering equals the candidate reader's, byte for byte (§13.4), and the no-patch control
renders what it stores (§13.5 (a))."""

from __future__ import annotations

import importlib.util
import json
import random

import pytest

from residual_disposition import lens_render as lr
from residual_disposition.paths import REPO_ROOT


@pytest.fixture(scope="module")
def candidate():
    path = REPO_ROOT / "scripts" / "stage3_reader_candidate.py"
    spec = importlib.util.spec_from_file_location("stage3_reader_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_package(root, seed=0, decoders=("state", "base", "random")):
    """A package tree with the stored files' shapes; tokens overlap across scales and include
    non-ASCII ones, so ranks, scale runs and escaping are all exercised."""
    rng = random.Random(seed)
    vocab = [f"t{i}" for i in range(60)] + ["一秒", "刹那", " é", "\n"]
    run = root / "runs" / "x" / "pkg"
    for layer in lr.LAYERS:
        for decoder in decoders:
            for p in lr.POSITIONS:
                d = run / "steps" / f"layer{layer}" / decoder / f"p{p}"
                d.mkdir(parents=True)
                tokens = [f"l{rng.randrange(500)}" for _ in range(lr.LOGIT_K)]
                (d / "logit_lens.json").write_text(json.dumps({"tokens": tokens, "probs": []}))
                sweep = [[s, rng.sample(vocab, lr.PATCH_K), [0.0] * lr.PATCH_K] for s in lr.SCALES]
                (d / "patchscope.json").write_text(json.dumps(sweep))
        f = run / "steps" / f"layer{layer}" / "no_patch" / "patchscope.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps([[1.0, rng.sample(vocab, lr.PATCH_K), [0.0] * lr.PATCH_K]]))
    return run


def test_prompt_is_the_candidates(candidate):
    assert lr.prompt().text == candidate.prompt().text
    assert lr.prompt().sha256 == lr.PROMPT_SHA256


def test_rendering_is_byte_identical_to_the_candidates(candidate, tmp_path, monkeypatch):
    run = fake_package(tmp_path)
    monkeypatch.setattr(candidate, "REPO_ROOT", tmp_path)
    for decoder in lr.DECODERS:
        assert lr.render(run, decoder, root=tmp_path) == candidate.render(run, decoder)


def test_real_packages_if_present(candidate):
    """On the screen's recorded packages, when this checkout holds them."""
    names = ["20261015-232919-adl_lens_package-727c1b34-0de7"]
    runs = [r for n in names for r in REPO_ROOT.glob(f"runs/*/{n}")]
    if not runs:
        pytest.skip("no recorded lens package in this checkout")
    for run in runs:
        for decoder in lr.DECODERS:
            assert lr.render(run, decoder) == candidate.render(run, decoder)


def test_no_patch_renders_its_one_result(tmp_path):
    run = fake_package(tmp_path)
    evidence, sources = lr.render_no_patch(run, root=tmp_path)
    overview = json.loads(evidence.removeprefix("OVERVIEW:\n"))["datasets"]["ds1"]
    assert overview["patchscope_scale_grid"] == [1.0]
    for layer in overview["layers"].values():
        positions = layer["positions"]
        assert set(positions) == {str(p) for p in lr.POSITIONS}
        first = positions["0"]
        assert first["logit_lens"] == [] and len(first["patch_scope"]) == lr.PATCH_K
        assert all(entry[2] == ["1"] for entry in first["patch_scope"])
        assert all(p == first for p in positions.values())
    assert len(sources) == len(lr.LAYERS)
    text = lr.no_patch_prompt().text
    assert "30" not in text.split("Patchscope token promotions")[1].split("\n")[0]
    assert lr.no_patch_prompt().sha256 != lr.PROMPT_SHA256


def test_render_refuses_wrong_shapes(tmp_path):
    run = fake_package(tmp_path)
    with pytest.raises(ValueError, match="decoder"):
        lr.render(run, "no_patch", root=tmp_path)
    f = run / "steps" / "layer20" / "state" / "p0" / "patchscope.json"
    f.write_text(json.dumps(json.loads(f.read_text())[:29]))
    with pytest.raises(ValueError, match="30 frozen scales"):
        lr.render(run, "state", root=tmp_path)
