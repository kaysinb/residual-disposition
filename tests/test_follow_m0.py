"""M0 (§13.10): the decision gates only the decisive comparison and the decisive methods' repeats;
the run checks its effective configuration first, keeps every tensor and its progress as it goes,
and writes a result only for the complete inventory."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from fractions import Fraction

import pytest
import torch
from safetensors.torch import save_file

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_m0 as M
from residual_disposition.paths import REPO_ROOT

TOL = Fraction(1, 256)
M0 = cr.follow_m0()


def deltas(session_shift=0.0, repeat_shift=0.0, cached_shift=0.5, cached_repeat=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(5, 64, generator=g)
    return {
        "qualification": [q, q * (1 + repeat_shift)],
        "production_session": [q * (1 + session_shift), q * (1 + session_shift)],
        "production_cached": [q * (1 + cached_shift), q * (1 + cached_shift + cached_repeat)],
    }


def test_the_decisive_comparison_alone_decides():
    ok = M.verdict(
        {"s1": deltas(), "s2": deltas(seed=1)}, TOL, historical={"s1": torch.ones(5, 64)}
    )
    assert ok["passed"]  # the cached base means and the historical reference differ: diagnostic
    assert not ok["states"]["s1"]["diagnostic"]["cached_vs_qualification"][0]["passed"]
    assert not ok["states"]["s1"]["diagnostic"]["vs_historical"]["qualification"]["passed"]
    far = M.verdict({"s1": deltas(session_shift=0.01)}, TOL)  # 1 % > 2⁻⁸
    assert not far["passed"] and not far["states"]["s1"]["decisive"][0]["passed"]
    moved = M.verdict({"s1": deltas(repeat_shift=0.01)}, TOL)
    assert not moved["passed"]


def test_a_cached_repeat_never_vetoes():
    """Regression: both decisive comparisons and both fresh methods' repeats pass, the cached
    method's repeat does not; M0 passes, the cached repeat reported."""
    out = M.verdict({"s1": deltas(cached_repeat=0.05)}, TOL)
    assert out["passed"]
    assert not out["states"]["s1"]["diagnostic"]["cached_repeat"]["passed"]
    assert set(out["states"]["s1"]["repeats"]) == set(M.GATED_REPEATS)


def entry_point():
    path = REPO_ROOT / "scripts" / "pod" / "follow_m0.py"
    spec = importlib.util.spec_from_file_location("follow_m0_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_overrides_are_refused_before_any_extraction():
    rt = M0["runtime"]
    model, lens = dict(rt["model"]), {"corpus": dict(rt["corpus"]), "layers": list(rt["layers"])}
    seqs = (rt["sequences"]["count"], rt["sequences"]["tokens"])
    assert M.runtime_problems(model, lens, seqs, M0) == []
    assert M.runtime_problems({**model, "dtype": "float16"}, lens, seqs, M0)
    assert M.runtime_problems({**model, "revision": "main"}, lens, seqs, M0)
    assert M.runtime_problems(model, {**lens, "layers": [20]}, seqs, M0)
    assert M.runtime_problems(
        model, {**lens, "corpus": {**lens["corpus"], "column": "x"}}, seqs, M0
    )
    assert M.runtime_problems(model, lens, (5000, 128), M0)
    ep = entry_point()
    assert ep.cli_problems([rt["experiment"]], M0) == []
    assert ep.cli_problems([rt["experiment"], "model.dtype=float16"], M0)
    assert ep.cli_problems(["experiments/continuation_14b/eli5"], M0)


def fakes(fail_after=None):
    g = torch.Generator().manual_seed(3)
    base = {20: torch.randn(5, 64, generator=g), 30: torch.randn(5, 64, generator=g)}
    calls = []

    def base_means(r):
        return {k: v.clone() for k, v in base.items()}

    def extract(sid, r):
        calls.append((sid, r))
        if fail_after is not None and len(calls) > fail_after:
            raise RuntimeError("the Pod went away")
        state = {k: v + 1.0 for k, v in base.items()}
        return state[20] - base[20], state

    def save(path, tensor):
        save_file({"tensor": tensor.contiguous()}, str(path))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {"base_means": base_means, "extract": extract, "save": save,
            "cached": {k: v.clone() for k, v in base.items()}}  # fmt: skip


def run(tmp_path, f):
    return M.execute(tmp_path, {"run": "test"}, states=["s1", "s2"], repeats=2, layer=20,
                     reported=[30], historical={}, tolerance=TOL, **f)  # fmt: skip


def test_an_interrupted_run_keeps_its_evidence_and_stays_pending(tmp_path):
    with pytest.raises(RuntimeError, match="went away"):
        run(tmp_path, fakes(fail_after=2))  # s1's two repeats, then the failure
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"run": "test"}
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["status"] == "pending" and progress["done"] == [["s1", 0], ["s1", 1]]
    assert "base_means-0-layer20.safetensors" in progress["files"]
    assert "s1-production_cached-1-layer30.safetensors" in progress["files"]
    assert all((tmp_path / name).is_file() for name in progress["files"])
    assert not (tmp_path / "result.json").exists()


def test_a_complete_run_publishes_its_result(tmp_path):
    result = run(tmp_path, fakes())
    assert result["status"] == "passed"
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "passed"
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "passed"


def test_configuration():
    assert M0["decisive"] == list(M.DECISIVE) and set(M0["methods"]) == set(M.METHODS)
    assert Fraction(cr.follow_rules()["m0"]["tolerance"]) == TOL
