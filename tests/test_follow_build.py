"""The follow-up's build sessions (§13.10): base means computed once in the session, the lineages'
A rebuilt first as references only, each build's method-equivalence validation in the session
before its readings, the lens builder's seeds, and a durable record."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction

import pytest
import torch
import yaml
from safetensors.torch import save_file

from residual_disposition import adl_lens
from residual_disposition import continuation_rules as cr
from residual_disposition import follow_build as B
from residual_disposition.seeding import derive_seed

TOL = Fraction(1, 256)
LAYERS = [20, 30]
REFS = [{"behavior": "pun_intended", "role": "A", "state": "sA"}]
BUILDS = [
    {"behavior": "pun_intended", "lineage": "pun_intended 200", "role": "L", "state": "sL",
     "branches": ["state", "base", "random"]},
    {"behavior": "pun_intended", "lineage": "pun_intended 200", "role": "H_s", "state": "sH",
     "branches": ["state", "base"]},
]  # fmt: skip


def fakes(off=None, shift=0.01):
    """GPU-free stand-ins: ``off`` names the state whose production path misses the
    qualification's by ``shift`` (relative, far above 2⁻⁸)."""
    g = torch.Generator().manual_seed(5)
    base = {x: torch.randn(5, 64, generator=g) for x in LAYERS}
    own = {s: {x: torch.randn(5, 64, generator=g) for x in LAYERS} for s in ("sA", "sL", "sH")}
    calls = {"base": 0, "read": []}

    def base_means():
        calls["base"] += 1
        return {x: v.clone() for x, v in base.items()}

    def qualification(sid):
        d = own[sid][20] - base[20]
        return d * (1 + shift) if sid == off else d

    def state_means(sid):
        return {x: {"mean": own[sid][x].clone(), "norm": 7.0 + x} for x in LAYERS}

    def read(sid, vectors, *, norm, layer, decoder, root, prefix):
        calls["read"].append((sid, layer, decoder, prefix, norm, vectors.clone()))
        (root / prefix).mkdir(parents=True, exist_ok=True)

    def save(path, tensor):
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"tensor": tensor.contiguous()}, str(path))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    f = {"base_means": base_means, "qualification": qualification, "state_means": state_means,
         "read": read, "save": save}  # fmt: skip
    return f, calls, base, own


def run(out, f, builds=BUILDS, refs=REFS):
    return B.execute(out, {"run": "test"}, references=refs, builds=builds, layers=LAYERS,
                     layer=20, tolerance=TOL, run_seed=0, **f)  # fmt: skip


def test_a_complete_session(tmp_path):
    f, calls, base, own = fakes()
    out = tmp_path / "s"
    result = run(out, f)
    assert result["status"] == "complete" and calls["base"] == 1  # once, for every build
    assert result["references"]["sA"]["passed"]
    assert not (out / "builds" / "sA").exists()  # the rebuilt A is a reference: no readings
    read = [(s, x, d, p) for s, x, d, p, _, _ in calls["read"]]
    assert read == [("sL", 20, "state", "layer20/state"), ("sL", 20, "base", "layer20/base"),
                    ("sL", 20, "state", "layer20/random"), ("sL", 30, "state", "layer30/state"),
                    ("sL", 30, "base", "layer30/base"), ("sL", 30, "state", "layer30/random"),
                    ("sH", 20, "state", "layer20/state"), ("sH", 20, "base", "layer20/base"),
                    ("sH", 30, "state", "layer30/state"),
                    ("sH", 30, "base", "layer30/base")]  # fmt: skip
    for sid, x, _, prefix, norm, vectors in calls["read"]:
        delta = own[sid][x] - base[x]
        assert norm == 7.0 + x
        if prefix.endswith("random"):  # norm-matched, seeded by the lens builder's rule
            seed = derive_seed(derive_seed(0, "adl_lens", sid), "adl_lens", x, "random")
            assert torch.equal(vectors, adl_lens.random_directions(delta, seed))
            assert torch.allclose(vectors.norm(dim=-1), delta.norm(dim=-1))
        else:
            assert torch.equal(vectors, delta)  # the session's base means, never another's
    record = json.loads((out / "builds" / "sL" / "build.json").read_text())
    assert record["method_equivalence"]["passed"] and record["branches"] == BUILDS[0]["branches"]
    assert record["seeds"]["package"] == derive_seed(0, "adl_lens", "sL")
    files = json.loads((out / "progress.json").read_text())["files"]
    assert {"base_means-layer20.safetensors", "references/sA/qualification-layer20.safetensors",
            "builds/sL/production-layer30.safetensors"} <= set(files)  # fmt: skip
    assert json.loads((out / "result.json").read_text())["status"] == "complete"


def test_a_failed_reference_stops_every_build(tmp_path):
    f, calls, _, _ = fakes(off="sA")
    result = run(tmp_path / "s", f)
    assert result["status"] == "stopped" and result["check"] == "reference"
    assert result["state"] == "sA" and not result["report"]["passed"]
    assert calls["read"] == [] and not (tmp_path / "s" / "builds").exists()


def test_a_failed_method_equivalence_stops_before_its_readings(tmp_path):
    f, calls, _, _ = fakes(off="sH")
    out = tmp_path / "s"
    result = run(out, f)
    assert (result["status"], result["check"], result["state"]) == (
        "stopped", "method_equivalence", "sH")  # fmt: skip
    assert {c[0] for c in calls["read"]} == {"sL"}  # the earlier build stands, complete
    assert (out / "builds" / "sL" / "build.json").is_file()
    assert not (out / "builds" / "sH" / "build.json").exists()
    assert (out / "builds" / "sH" / "qualification-layer20.safetensors").is_file()  # both kept
    assert (out / "builds" / "sH" / "production-layer20.safetensors").is_file()


def test_an_interrupted_session_stays_pending(tmp_path):
    f, _, _, _ = fakes()

    def gone(*args, **kwargs):
        raise RuntimeError("the Pod went away")

    with pytest.raises(RuntimeError, match="went away"):
        run(tmp_path / "s", {**f, "read": gone})
    progress = json.loads((tmp_path / "s" / "progress.json").read_text())
    assert progress["status"] == "pending" and progress["references"]["sA"]["passed"]
    assert not (tmp_path / "s" / "result.json").exists()
    with pytest.raises(FileExistsError):  # a later session builds in a new directory
        run(tmp_path / "s", f)


def test_the_plan_and_the_tolerance():
    assert B.plan_problems(REFS, BUILDS) == []
    assert B.plan_problems([], BUILDS)
    assert B.plan_problems([{**REFS[0], "role": "H"}], BUILDS)
    assert B.plan_problems(REFS, [{**BUILDS[0], "branches": ["state", "no_patch"]}])
    assert B.plan_problems(REFS, [{**BUILDS[0], "branches": []}])
    assert B.plan_problems(REFS, [*BUILDS, {**BUILDS[1], "role": "X"}])  # a state twice
    other = [{**BUILDS[0], "behavior": "eli5"}]
    assert any("not among the references" in p for p in B.plan_problems(REFS, other))
    assert B.plan_problems([*REFS, {**REFS[0], "state": "sA2"}], BUILDS)  # one A per behavior
    assert not B.plan_problems([*REFS, {"behavior": "eli5", "role": "A", "state": "sE"}], BUILDS)
    assert Fraction(cr.follow_rules()["m0"]["tolerance"]) == TOL
    assert float(TOL) == adl_lens.REPRODUCTION_TOLERANCE  # §12.4.3's tolerance


A_STATES = {"pun_intended": "sA", "eli5": "sE"}


def config(tmp_path, **change):
    build = {"runtime": cr.follow_m0()["runtime"], "layers": {"decisive": 20, "read": LAYERS},
             "sessions": {"p1": {"references": REFS, "builds": BUILDS}}, **change}  # fmt: skip
    path = tmp_path / "follow_build.yaml"
    path.write_text(yaml.safe_dump({"build": build}))
    return path


def test_the_configuration(tmp_path, monkeypatch):
    def load(path, **kw):
        return cr.load_follow_build(path, a_states=A_STATES, **kw)

    build = load(config(tmp_path))
    assert load(config(tmp_path), expected_hash=cr.follow_hash(build)) == build
    with pytest.raises(ValueError, match="changed"):
        load(config(tmp_path), expected_hash="0" * 64)
    runtime = {**cr.follow_m0()["runtime"], "layers": [20]}
    with pytest.raises(ValueError, match="M0's frozen runtime"):
        load(config(tmp_path, runtime=runtime))
    twice = {"p1": {"references": REFS, "builds": BUILDS[:1]},
             "p2": {"references": REFS, "builds": BUILDS[:1]}}  # fmt: skip
    with pytest.raises(ValueError, match="a build is planned in two sessions"):
        load(config(tmp_path, sessions=twice))
    split = {"p1": {"references": REFS, "builds": BUILDS[:1]},
             "p2": {"references": REFS, "builds": BUILDS[1:]}}  # fmt: skip
    assert load(config(tmp_path, sessions=split))  # the lineage's A rebuilt in each session
    wrong = {"p1": {"references": [{**REFS[0], "state": "sE"}], "builds": BUILDS}}
    with pytest.raises(ValueError, match="not pun_intended's A sA"):
        load(config(tmp_path, sessions=wrong))
    recorded = cr.lineage_a()  # the A each behavior's records name (B2's inventory manifest)
    assert recorded["word_count_claim"] == "s008b0fae04af22f9"  # M0's word_count_claim A
    with pytest.raises(ValueError, match="not pun_intended's A s941ebe3c33817be5"):
        cr.load_follow_build(config(tmp_path))  # the stand-in sA is not the recorded A
    frozen = cr.load_follow_build(expected_hash=cr.FOLLOW_BUILD_FROZEN)
    assert cr.follow_build() == frozen  # the builds run only the frozen configuration
    monkeypatch.setattr(cr, "FOLLOW_BUILD_FROZEN", None)
    with pytest.raises(ValueError, match="not frozen"):
        cr.follow_build()


def test_a_composition_and_the_pools_in_the_configuration(tmp_path):
    def load(**change):
        return cr.load_follow_build(config(tmp_path, **change), a_states=A_STATES)

    x = {"behavior": "pun_intended", "lineage": "pun_intended 200", "role": "X",
         "compose": {"suppressed": "sL", "sham": "sH0"}, "branches": ["state", "base"]}  # fmt: skip
    sessions = {"p1": {"references": REFS, "builds": [*BUILDS, x]}}
    pools = [{k: BUILDS[0][k] for k in ("behavior", "lineage", "role", "state")},
             {k: x[k] for k in ("behavior", "lineage", "role", "compose")}]  # fmt: skip
    assert load(sessions=sessions, pools=pools)["pools"] == pools
    both = {"p1": {"references": REFS, "builds": [*BUILDS, {**x, "state": "sX"}]}}
    with pytest.raises(ValueError, match="names its state or its composition"):
        load(sessions=both)
    keys = {"p1": {"references": REFS, "builds": [*BUILDS, {**x, "compose": {"suppressed": "sL"}}]}}
    with pytest.raises(ValueError, match="names its state or its composition"):
        load(sessions=keys)
    for wrong in ([pools[0], pools[0]], [{**pools[0], "state": "sZ"}], [{**pools[0], "role": "X"}]):
        with pytest.raises(ValueError, match="PD pool"):
            load(sessions=sessions, pools=wrong)


def test_the_new_packages_follow_the_confirmed_doses():
    inventory = json.loads((cr.REPO_ROOT / "manifests/follow_b2_inventory.json").read_text())
    stack = [f"eli5 stack 200 {r}" for r in ("H_s", "L", "X", "recipe")]
    pun = [f"pun_intended 200 {r}" for r in ("H_s", "L", "X", "recipe")]
    assert B.new_packages(inventory, {"example_link": 200, "pun_intended": 200}) == stack + pun
    assert B.new_packages(inventory, {"example_link": None, "pun_intended": None}) == stack
    assert B.new_packages(inventory, {"example_link": 100, "pun_intended": 400}) == stack + [
        f"pun_intended 400 {r}" for r in ("H_s", "L", "X", "recipe")]  # fmt: skip


def test_the_frozen_configuration_is_the_plan_from_the_records():
    """Every state the frozen builds name is the one the records give at the committed
    confirmations' doses (``follow_build.planned``)."""
    from residual_disposition import follow_confirm as F
    from residual_disposition import pipeline as P
    from residual_disposition.config import load_config
    from residual_disposition.continuation_kstar import find_run
    from residual_disposition.paths import runs_root

    eli5 = json.loads((cr.REPO_ROOT / "manifests/stage3_eli5.json").read_text())
    if find_run(eli5["inputs"]["selection"]) is None:
        pytest.skip("the records' runs are not here")
    spec = F.load_spec()

    def make(spec_, behavior):
        return P.Workspace(load_config(spec_.experiment(behavior)))

    confirmed = F.committed_confirmations(spec, runs_root(), "continuation_14b", make)
    plan = B.planned(confirmed)
    frozen = cr.follow_build()
    assert frozen["sessions"] == {"b1": {"references": plan["references"],
                                         "builds": plan["builds"]}}  # fmt: skip
    assert frozen["pools"] == plan["pools"]
    assert [b["state"] for b in plan["references"]] == ["s46157d0cf7a1b951", "s941ebe3c33817be5"]


def test_composed_names_each_composition_by_its_checked_state():
    made, checked = [], []

    def make(suppressed, sham):
        made.append((suppressed, sham))
        return "sX"

    def verify(sid, suppressed, sham):
        checked.append((sid, suppressed, sham))

    x = {"role": "X", "compose": {"suppressed": "sL", "sham": "sH0"}, "branches": ["state"]}
    out = B.composed([BUILDS[0], x], make, verify)
    assert out == [BUILDS[0], {"role": "X", "state": "sX", "branches": ["state"],
                               "composed": {"suppressed": "sL", "sham": "sH0"}}]  # fmt: skip
    assert made == [("sL", "sH0")] and checked == [("sX", "sL", "sH0")]

    def refuse(sid, suppressed, sham):
        raise ValueError("not the composition")

    with pytest.raises(ValueError, match="not the composition"):
        B.composed([x], make, refuse)


def pool(n=3, configs=("chat", "raw"), drop=None, extra=()):
    rows = [{"config": c, "prefill": f"p{i:05d}"} for c in configs for i in range(n)]
    return [r for r in rows if r != drop] + list(extra)


def test_a_pool_is_checked_whole_and_never_scored():
    whole = {"prefills": 3, "prefills_sha256": "f" * 64}
    kw = {"prefills": 3, "prefills_sha256": "f" * 64, "configs": ("chat", "raw"),
          "gpu": "NVIDIA H100 80GB HBM3", "final_gpu": "NVIDIA H100 80GB HBM3"}  # fmt: skip
    assert B.pool_problems(whole, pool(), **kw) == []
    assert "final PD GPU" in B.pool_problems(whole, pool(), **{**kw, "gpu": "NVIDIA A100"})[0]
    assert B.pool_problems({**whole, "prefills": 2}, pool(), **kw)
    assert "missing" in B.pool_problems(whole, pool(drop={"config": "raw", "prefill": "p00001"}),
                                        **kw)[0]  # fmt: skip
    assert (
        "repeats"
        in B.pool_problems(whole, pool(extra=[{"config": "chat", "prefill": "p00000"}]), **kw)[0]
    )
    assert (
        "off the pool"
        in B.pool_problems(whole, pool(extra=[{"config": "chat", "prefill": "p00009"}]), **kw)[0]
    )


def test_the_pools_run_in_order_and_stop_at_one_not_whole(tmp_path):
    pools = [
        {"behavior": "pun_intended", "role": r, "state": s} for r, s in (("L", "sL"), ("H_s", "sH"))
    ]
    audited = []

    def audit(sid):
        audited.append(sid)
        return {"run": f"run-{sid}", "prefills": 3}

    result = B.run_pools(tmp_path / "a", {"m": 1}, pools, audit=audit, problems=lambda sid: [])
    assert result["status"] == "complete" and audited == ["sL", "sH"]
    assert json.loads((tmp_path / "a" / "result.json").read_text()) == result
    assert result["pools"]["sH"]["run"] == "run-sH"
    stopped = B.run_pools(
        tmp_path / "b",
        {},
        pools,
        audit=audit,
        problems=lambda sid: ["completions missing"] if sid == "sL" else [],
    )
    assert stopped["status"] == "stopped" and stopped["state"] == "sL" and stopped["pools"] == {}
    with pytest.raises(FileExistsError):
        B.run_pools(tmp_path / "a", {}, pools, audit=audit, problems=lambda sid: [])


@pytest.mark.parametrize("script, argv", [
    ("follow_build.py", ["--out", "o", "--session", "b1", "experiments/continuation_14b/x"]),
    ("follow_pools.py", ["--out", "o", "experiments/continuation_14b/pun_intended"]),
])  # fmt: skip
def test_the_pod_entry_points_remove_the_launchers_run_id(monkeypatch, script, argv):
    import importlib.util
    import os
    import sys

    path = cr.REPO_ROOT / "scripts" / "pod" / script
    spec = importlib.util.spec_from_file_location(path.stem + "_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = {}

    def stop():
        seen["RUN_ID"] = os.environ.get("RUN_ID", "absent")
        raise SystemExit("stopped at the configuration")

    monkeypatch.setenv("RUN_ID", "follow-build-20261031T000000Z")
    monkeypatch.setattr(sys, "argv", [script, *argv])
    monkeypatch.setattr(cr, "follow_build", stop)
    with pytest.raises(SystemExit, match="stopped at the configuration"):
        module.main()
    assert seen["RUN_ID"] == "absent"


def test_the_readings_are_in_dstars_layout(tmp_path, monkeypatch):
    """The builder's real ``lens_readings`` (its lens and Patchscope stubbed) stores each branch
    where ``lens_render.render`` reads it."""
    import contextlib

    from residual_disposition import adl, lens_render

    scales = list(lens_render.SCALES)
    monkeypatch.setattr(adl_lens, "logit_lens", lambda model, v: (list(range(100)), [0.0] * 100))
    monkeypatch.setattr(adl_lens, "patchscope", lambda *a, **k: None)
    monkeypatch.setattr(adl_lens, "steering_vector", lambda v, norm: v)
    monkeypatch.setattr(adl_lens, "scale_tokens", lambda probs, tok: [
        (s, [f"t{i}" for i in range(20)], [0.0] * 20) for s in scales])  # fmt: skip

    class Model:
        def disable_adapter(self):
            return contextlib.nullcontext()

    class Tok:
        def decode(self, ids):
            return f"w{ids[0]}"

    f, _, _, _ = fakes()

    def read(sid, vectors, *, norm, layer, decoder, root, prefix):
        return adl_lens.lens_readings(Model(), Tok(), vectors, norm=norm, layer=layer,
                                      decoder=decoder, store=adl.StepStore(root),
                                      prefix=prefix)  # fmt: skip

    out = tmp_path / "s"
    run(out, {**f, "read": read})
    for decoder in ("state", "base", "random"):
        text, sources = lens_render.render(out / "builds" / "sL", decoder, root=out)
        assert text.startswith("OVERVIEW:") and len(sources) == 2 * 5 * 2
    with pytest.raises(FileNotFoundError):
        lens_render.render(out / "builds" / "sH", "random", root=out)  # not one of its branches
