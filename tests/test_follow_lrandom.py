"""The existing L packages' random branches (§13.4; barriers B2 and B3): the session is exactly the
packages B3's inventory names, each L's random branch alone at its package's recorded state and
seed; its output is recorded once and found only through that record."""

from __future__ import annotations

import copy
import json
from fractions import Fraction

import pytest

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_build as B
from residual_disposition import follow_inventory as fi


def session():
    return cr.load_follow_lrandom()


def inventory():
    return json.loads((cr.REPO_ROOT / "manifests" / "follow_b3_inventory.json").read_text())


def test_the_session_is_b3_s_random_branches():
    from residual_disposition.continuation_kstar import find_run

    names = B.random_names(inventory())
    assert len(names) == 5 and [b["package"] for b in session()["builds"]] == names
    if find_run(names[0]) is None:
        pytest.skip("the packages are not here")
    assert B.lrandom_problems(session(), names, cr.follow_build(), find_run, 0) == []


def fake_find(tmp_path, seeds):
    def find(name):
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        state, seed = seeds[name]
        (d / "adl_lens_package.json").write_text(json.dumps({"state": state, "seed": seed}))
        return d

    return find


def test_the_session_is_refused_unless_it_is_the_records(monkeypatch):
    monkeypatch.setattr(B, "pinned_problems", lambda lrandom, find_run: [])
    s, build = session(), cr.follow_build()
    names = [b["package"] for b in s["builds"]]
    seeds = {b["package"]: (b["state"], B.seeds(0, b["state"], [])["package"]) for b in s["builds"]}

    def problems(change, tmp):
        data = copy.deepcopy(s)
        change(data)
        return B.lrandom_problems(data, names, build, fake_find(tmp, seeds), 0)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        assert problems(lambda d: None, tmp) == []
        assert problems(lambda d: d["builds"].reverse(), tmp)
        assert problems(lambda d: d["builds"][0].update(branches=["random", "state"]), tmp)
        assert problems(lambda d: d["builds"][1].update(state="s0"), tmp)
        assert problems(lambda d: d["layers"].update(decisive=30), tmp)
        assert problems(lambda d: d["references"].pop(), tmp)  # a behavior without its A
        wrong = {**seeds, names[2]: (seeds[names[2]][0], 1)}
        assert B.lrandom_problems(s, names, build, fake_find(tmp, wrong), 0)


LAYERS = (20, 30)


class Fakes:
    """A session's callables on small tensors: base means 0, each state's means ``means[state]``
    (its norm ``norms[state]``), the qualification path its δ̄ at layer 20; ``read`` writes a
    branch's files and records what it was given."""

    def __init__(self, means, norms, base=None):
        import torch

        self.torch, self.means, self.norms, self.reads = torch, means, norms, []
        self.base = torch.zeros(5, 4) if base is None else base

    def base_means(self):
        return {x: self.base for x in LAYERS}

    def qualification(self, sid):
        return self.means[sid][20] - self.base

    def state_means(self, sid):
        return {x: {"mean": self.means[sid][x], "norm": self.norms[sid]} for x in LAYERS}

    def read(self, sid, vectors, *, norm, layer, decoder, root, prefix):
        from residual_disposition import lens_render as R

        self.reads.append((sid, layer, norm, vectors.clone()))
        for p in range(5):
            d = root / prefix / f"p{p}"
            d.mkdir(parents=True, exist_ok=True)
            tokens = [f"t{i}" for i in range(R.LOGIT_K)]
            (d / "logit_lens.json").write_text(json.dumps({"tokens": tokens}))
            sweep = [[s, tokens[: R.PATCH_K], [0.0] * R.PATCH_K] for s in R.SCALES]
            (d / "patchscope.json").write_text(json.dumps(sweep))

    def save(self, path, tensor):
        from safetensors.torch import save_file

        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"tensor": tensor.contiguous()}, str(path))
        return fi._sha(path)


def run_session(tmp_path, means, norms, pinned, base=None):
    s = session()
    fakes = Fakes(means, norms, base)
    out = tmp_path / "out"
    result = B.execute_random(out, {"follow_lrandom_sha256": "h", "plan": s},
                              references=s["references"][:1], builds=[
                                  {**s["builds"][0], "behavior": s["references"][0]["behavior"]}],
                              layers=list(LAYERS), layer=20, tolerance=Fraction(1, 256),
                              run_seed=0, base_means=fakes.base_means,
                              qualification=fakes.qualification, state_means=fakes.state_means,
                              read=fakes.read, save=fakes.save, pinned=pinned)  # fmt: skip
    return result, fakes, out


def test_the_control_is_built_from_the_pinned_sources(tmp_path):
    import torch

    s = session()
    a, l_state = s["references"][0]["state"], s["builds"][0]["state"]
    big = torch.full((5, 4), 1000.0)
    delta = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 100 + 1
    means = {a: {x: big for x in LAYERS}, l_state: {x: delta for x in LAYERS}}
    pinned = {l_state: {x: {"delta": delta.clone(), "norm": 7.0} for x in LAYERS}}
    result, fakes, out = run_session(tmp_path, means, {a: 3.0, l_state: 7.0}, pinned)
    assert result["status"] == "complete" and len(fakes.reads) == 2
    seed = B.seeds(0, l_state, LAYERS)
    for _, layer, norm, vectors in fakes.reads:
        assert norm == 7.0  # the pinned η, never the session's
        assert torch.equal(vectors, B.random_directions(delta, seed["random"][str(layer)]))
    files = sorted(str(p.relative_to(out / "builds" / l_state))
                   for p in (out / "builds" / l_state).rglob("*") if p.is_file())  # fmt: skip
    assert files == B.random_files(list(LAYERS))


def good_pinned(delta):
    return {x: {"delta": delta.clone(), "norm": 7.0} for x in LAYERS}


def test_the_pinned_comparisons_are_diagnostics_the_method_check_decides(tmp_path):
    import torch

    s = session()
    a, l_state = s["references"][0]["state"], s["builds"][0]["state"]
    big = torch.full((5, 4), 1000.0)  # the base means: large, as the states' are
    delta = torch.full((5, 4), 0.01)  # the session's small difference
    means = {a: {x: big + 1 for x in LAYERS}, l_state: {x: big + delta for x in LAYERS}}
    # the probe's case: the means agree to 1e-5 while the difference is 100 times the session's
    pinned = {l_state: {x: {"delta": delta * 100, "norm": 7.0} for x in LAYERS}}
    assert B.check_reproduction(big + delta * 100, big + delta, tolerance=1 / 256)["passed"]
    result, fakes, _ = run_session(tmp_path / "a", means, {a: 3.0, l_state: 9.0}, pinned, big)
    assert result["status"] == "complete" and len(fakes.reads) == 2  # no historical veto
    diag = result["builds"][l_state]["diagnostics"]["20"]
    assert diag["delta"]["passed"] is False and diag["norm"]["passed"] is False
    assert diag["delta"]["relative_errors"][0] == pytest.approx(0.99, rel=1e-3)  # pinned below
    for _, _, norm, vectors in fakes.reads:  # the control: the pinned sources, whatever the session
        assert norm == 7.0 and torch.linalg.vector_norm(vectors[0]) == pytest.approx(
            float(torch.linalg.vector_norm(delta[0] * 100)), rel=1e-5)  # fmt: skip

    class Broken(Fakes):  # the same session's two paths disagree for L: the one deciding check
        def qualification(self, sid):
            q = super().qualification(sid)
            return q * 2 if sid == l_state else q

    s_ = session()
    broken = Broken(means, {a: 3.0, l_state: 7.0}, big)
    result = B.execute_random(tmp_path / "b" / "out", {"follow_lrandom_sha256": "h", "plan": s_},
                              references=s_["references"][:1], builds=[
                                  {**s_["builds"][0], "behavior": s_["references"][0]["behavior"]}],
                              layers=list(LAYERS), layer=20, tolerance=Fraction(1, 256),
                              run_seed=0, base_means=broken.base_means,
                              qualification=broken.qualification, state_means=broken.state_means,
                              read=broken.read, save=broken.save,
                              pinned={l_state: good_pinned(delta)})  # fmt: skip
    assert result["status"] == "stopped" and broken.reads == []


def test_the_record_recomputes_what_it_certifies(tmp_path):
    import torch
    from safetensors.torch import save_file

    s = session()
    a, l_state = s["references"][0]["state"], s["builds"][0]["state"]
    delta = torch.ones(5, 4)
    means = {a: {x: delta for x in LAYERS}, l_state: {x: delta for x in LAYERS}}
    _, _, out = run_session(tmp_path, means, {a: 3.0, l_state: 7.0}, {l_state: good_pinned(delta)})
    one = {**s, "references": s["references"][:1],
           "builds": [{**s["builds"][0], "behavior": s["references"][0]["behavior"]}]}  # fmt: skip
    manifest = json.loads((out / "manifest.json").read_text())
    (out / "manifest.json").write_text(json.dumps({**manifest, "plan": one}))
    tol = Fraction(1, 256)

    def problems(pinned=None):
        return B.lrandom_output_problems(out, one, "h", 0, tol, pinned)

    assert problems() == [] and problems({l_state: good_pinned(delta)}) == []
    assert any("diagnostic" in p for p in problems({l_state: good_pinned(delta * 3)}))
    d = out / "builds" / l_state
    result = json.loads((out / "result.json").read_text())
    # the probe: a disagreeing saved tensor, its progress sha rewritten, every flag "passed"
    q = d / "qualification-layer20.safetensors"
    save_file({"tensor": delta * 2}, str(q))
    forged = {**result, "files": {**result["files"], f"builds/{l_state}/qualification-layer20"
                                  ".safetensors": fi._sha(q)}}  # fmt: skip
    (out / "result.json").write_text(json.dumps(forged))
    assert any("method check" in p for p in problems())
    (out / "result.json").write_text(json.dumps(result))  # its recorded sha: the original's
    assert any("not the tensor" in p for p in problems())
    save_file({"tensor": delta}, str(q))
    assert problems() == []
    made = json.loads((d / "build.json").read_text())
    (d / "build.json").write_text(json.dumps({**made, "method_equivalence": {
        **made["method_equivalence"], "passed": False}}))  # fmt: skip
    assert any("method check" in p for p in problems())
    (d / "build.json").write_text(json.dumps({**made, "state": "s0"}))
    assert any("planned state" in p for p in problems())
    (d / "build.json").write_text(json.dumps(made))
    (d / "steps" / "layer30" / "random" / "p4" / "patchscope.json").write_text("[]")
    assert any("render" in p for p in problems())
    (d / "steps" / "layer30" / "random" / "p4" / "patchscope.json").unlink()
    assert any("inventory" in p for p in problems())
    assert B.lrandom_output_problems(out, one, "other", 0, tol)


def test_the_control_s_sources_are_the_pinned_files(tmp_path):
    s = copy.deepcopy(session())
    base = tmp_path / s["base_means"]["run"]
    base.mkdir()
    (base / "base_means.safetensors").write_bytes(b"base")
    s["base_means"]["sha256"] = fi._sha(base / "base_means.safetensors")
    for b in s["builds"]:
        d = tmp_path / b["package"]
        (d / "steps").mkdir(parents=True)
        (d / "steps" / "extraction.safetensors").write_bytes(b["state"].encode())
        (d / "adl_lens_package.json").write_text("{}")
        (d / "config.yaml").write_text("params: {base_means: c271396ac904}\n")
        b["pinned"] = {f: fi._sha(d / f) for f in b["pinned"]}

    def find(name):
        return tmp_path / name if (tmp_path / name).is_dir() else None

    assert B.pinned_problems(s, find) == []
    (tmp_path / s["builds"][2]["package"] / "adl_lens_package.json").write_text('{"x": 1}')
    assert any("pinned files" in p for p in B.pinned_problems(s, find))
    (base / "base_means.safetensors").write_bytes(b"other")
    assert any("base means" in p for p in B.pinned_problems(s, find))


def test_the_frozen_session_loads():
    assert cr.follow_lrandom() == session()
    assert cr.follow_hash(session()) == cr.FOLLOW_LRANDOM_FROZEN


def test_the_session_is_not_run_while_unfrozen(monkeypatch):
    monkeypatch.setattr(cr, "FOLLOW_LRANDOM_FROZEN", None)
    with pytest.raises(ValueError, match="not frozen"):
        cr.follow_lrandom()


def test_a_random_branch_is_found_only_through_its_committed_record(tmp_path, monkeypatch):
    import subprocess

    import torch

    s = session()
    a, l_state = s["references"][0]["state"], s["builds"][0]["state"]
    one = {**s, "references": s["references"][:1],
           "builds": [{**s["builds"][0], "behavior": s["references"][0]["behavior"]}]}  # fmt: skip
    monkeypatch.setattr(cr, "follow_lrandom", lambda: one)
    monkeypatch.setattr(cr, "FOLLOW_LRANDOM_FROZEN", "h")
    runs, repo = tmp_path / "runs", tmp_path / "repo"
    run = runs / "follow_lrandom" / "follow-lrandom-x"
    delta = torch.ones(5, 4)
    means = {a: {x: delta for x in LAYERS}, l_state: {x: delta for x in LAYERS}}
    pinned = {l_state: {x: {"delta": delta.clone(), "norm": 7.0} for x in LAYERS}}
    _, _, out = run_session(run / "out" / "session", means, {a: 3.0, l_state: 7.0}, pinned)
    d = out / "builds" / l_state
    files = {str(p.relative_to(d)): fi._sha(p) for p in sorted(d.rglob("*")) if p.is_file()}
    build = {"package": one["builds"][0]["package"], "dir": str(d.relative_to(run)),
             "files": files}  # fmt: skip
    record = {"run": run.name, "session_sha256": "h", "out": str(out.relative_to(run)),
              "manifest_sha256": fi._sha(out / "manifest.json"),
              "result_sha256": fi._sha(out / "result.json"),
              "builds": {l_state: build}}  # fmt: skip
    (repo / "manifests").mkdir(parents=True)
    (repo / fi.LRANDOM_RECORD).write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="no certificate"):  # not committed: refused
        fi.random_branches(runs, repo)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for k, v in (("user.email", "r@example.org"), ("user.name", "r")):
        subprocess.run(["git", "-C", str(repo), "config", k, v], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "record"], check=True)
    got = fi.random_branches(runs, repo)
    assert list(got) == [f"random branch of {one['builds'][0]['package']}"]
    (out / "result.json").write_text((out / "result.json").read_text() + " ")
    with pytest.raises(ValueError, match="manifest or result"):
        fi.random_branches(runs, repo)
    (out / "result.json").write_text((out / "result.json").read_text()[:-1])
    (runs / "other" / run.name).mkdir(parents=True)
    with pytest.raises(ValueError, match="found 2 times"):
        fi.random_branches(runs, repo)
