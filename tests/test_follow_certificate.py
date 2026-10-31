"""The follow-up's certificates (§13.8; barrier B3) and their seal (§13.14): every relearning cell
of a line carries the line's binding, and a cell without one keeps its identity exactly; the
calibration's six runs and the certificate's 72; the envelopes hide the record and its readiness,
and a receipt must place the runner's own envelope in its commit."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from residual_disposition import follow_certificate as F
from residual_disposition import pipeline as P

BINDING = {"b3": "f" * 64, "line": "pun_intended_200", "lineage": "follow_pun_intended_200"}


def captured_params(monkeypatch, **extra):
    """relearn_cell's cell params, its dependencies replaced by fixed stand-ins."""
    seen = {}
    settings = SimpleNamespace(sampling="s", evaluation_steps=lambda: [10, 20, 40])
    monkeypatch.setattr(P, "_relearn_settings", lambda ws, lr, budget: settings)
    monkeypatch.setattr(P, "settings", lambda s: {"frozen": True})
    monkeypatch.setattr(P, "latched_stop", lambda s, k: 20)
    monkeypatch.setattr(P, "make_cell", lambda ws, step, params: seen.update(params) or "cell")
    monkeypatch.setattr(P, "RelearnCell", lambda *a: a)
    monkeypatch.setattr(P, "items_hash", lambda items: f"h{len(list(items))}")
    ws = SimpleNamespace(requests=lambda name, limit: ["r"],
                         generation_params=lambda s, r: {"sampling": "x"},
                         cfg=SimpleNamespace(relearn={}))  # fmt: skip
    data = SimpleNamespace(data_id="d1", examples=[{"a": 1}])
    P.relearn_cell(ws, "sX", lr=1e-5, relearn_seed=3, active_rate=0.8, data=data, budget=450,
                   purpose="certificate", kstar=40, **extra)  # fmt: skip
    return dict(seen)


def test_a_cell_without_a_binding_keeps_its_identity_exactly(monkeypatch):
    plain = captured_params(monkeypatch)
    assert captured_params(monkeypatch, binding=None) == plain and "follow" not in plain
    assert json.dumps(plain, sort_keys=True) == json.dumps({
        "state": "sX", "lr": 1e-5, "relearn_seed": 3, "active_rate": 0.8, "data_id": "d1",
        "examples": "h1", "eval_limit": None, "settings": {"frozen": True},
        "sampling": "x", "purpose": "certificate", "kstar": 40, "stop_after": 20,
        "keep_responses": True}, sort_keys=True)  # fmt: skip
    bound = captured_params(monkeypatch, binding=BINDING)
    assert bound == {**plain, "follow": BINDING}


def b3():
    from residual_disposition import continuation_rules as cr

    return cr.load_follow_b3(check_code=False)


def test_a_line_carries_its_binding():
    from residual_disposition.continuation_rules import follow_hash

    data = b3()
    line = F.line_of(data, "eli5_stack_200")
    assert line["binding"] == {"b3": follow_hash(data), "line": "eli5_stack_200",
                               "lineage": "follow_eli5_stack_200"}  # fmt: skip
    assert line["kstar_manifest"] == "manifests/kstar_follow_eli5_stack_200.json"
    with pytest.raises(ValueError, match="not one of"):
        F.line_of(data, "eli5_cont_200")


class Rules(dict):
    sha256 = "r" * 64


def rules():
    return Rules(
        {
            "certificate": {
                "kstar_calibration_seeds": 3,
                "learning_rates": [1e-5, 5e-6],
                "kstar_fraction": "1/2",
                "block_seeds": [0, 1, 2, 3, 4, 5],
            }
        }
    )


def fake_world(monkeypatch, calls):
    monkeypatch.setattr(P, "items_hash", lambda items: f"h{list(items)}")
    line = {**F.line_of(b3(), "pun_intended_200")}
    ref = {"run": line["reference"], "rate": 0.79, "hits": 158, "n": 200}
    monkeypatch.setattr(F, "reference", lambda ws, ln: ref)
    data = SimpleNamespace(data_id="d72d4f29820114b7b", examples=[1])
    monkeypatch.setattr(F, "data", lambda ws, ln: data)
    monkeypatch.setattr(P, "stage3_execution", lambda ws: {"generator_seed": 0,
                                                           "engine_mode": "sleep"})  # fmt: skip
    monkeypatch.setattr(
        P,
        "relearn_cell",
        lambda ws, s, **a: SimpleNamespace(
            cell=SimpleNamespace(previous=SimpleNamespace(name=f"run-{s}-{a['relearn_seed']}"))
        ),
    )
    template = SimpleNamespace(evaluation_steps=lambda: [2, 4], eval_schedule=[(20, 2)])
    monkeypatch.setattr(P, "_relearn_settings", lambda ws, lr, b: template)
    from residual_disposition import continuation_kstar as K

    monkeypatch.setattr(K, "schedule_steps", lambda s, b: [0, 2, 4])
    monkeypatch.setattr(K, "calibrate", lambda runs, steps, frac, rates: {
        "status": "resolved", "budget": 150,
        "rates": {K.rate_key(lr): {"kstar": 40} for lr in rates}})  # fmt: skip

    def relearn(ws, state, **args):
        calls.append((state, args))
        return {"T": 30.0}

    ws = SimpleNamespace(cfg=SimpleNamespace(relearn=SimpleNamespace(threshold=0.5)))
    return line, ws, relearn


def test_the_calibration_is_six_bound_runs_of_h_s(monkeypatch):
    calls = []
    line, ws, relearn = fake_world(monkeypatch, calls)
    record = F.calibrate(ws, line, rules(), relearn)
    assert len(calls) == 6 and {s for s, _ in calls} == {line["sham_s"]}
    assert all(a["binding"] == line["binding"] and a["budget"] == 600 and a["purpose"] ==
               "kstar_calibration" and a["execution"]["engine_mode"] == "sleep"
               for _, a in calls)  # fmt: skip
    assert sorted({a["relearn_seed"] for _, a in calls}) == [6, 7, 8]
    assert record["binding"] == line["binding"] and record["status"] == "resolved"


def test_the_certificate_is_72_bound_runs_at_k_star(monkeypatch):
    calls = []
    line, ws, relearn = fake_world(monkeypatch, calls)
    kstar = F.calibrate(ws, line, rules(), lambda *a, **k: {"T": 1.0})
    controls = {"arms": {"b": ws, "Y": ws, "Y2": ws}, "references": {
        "Y": {"rate": 0.7, "run": "y"}, "Y2": {"rate": 0.6, "run": "y2"}},
        "data": {"Y": ("dY", "h[2]"), "Y2": ("dY2", "h[3]")},
        "y2": {"sha256": "y2sha"}}  # fmt: skip
    monkeypatch.setattr(P, "certificate_controls", lambda ws, r: controls)
    other = iter([SimpleNamespace(data_id="dY", examples=[2]),
                  SimpleNamespace(data_id="dY2", examples=[3])])  # fmt: skip
    monkeypatch.setattr(P, "_relearn_data", lambda ws, plan, cached_only: next(other))
    kstar = {**kstar, "data_id": "d72d4f29820114b7b", "relearn_examples": "h[1]"}
    record = F.certify(ws, line, rules(), kstar, relearn)
    assert len(calls) == 72 and len(record["runs"]) == 72
    assert all(a["binding"] == line["binding"] and a["kstar"] == 40 and a["budget"] == 150
               for _, a in calls)  # fmt: skip
    with pytest.raises(ValueError, match="resolved"):
        F.certify(ws, line, rules(), {**kstar, "status": "unresolved"}, relearn)
    with pytest.raises(ValueError, match="resolved"):
        F.certify(ws, line, rules(), {**kstar, "binding": {**line["binding"], "line": "x"}})


def test_envelopes_hide_the_record_and_its_readiness():
    record = {"status": "resolved", "budget": 150}
    a, b = F.payload("l", 3, "s" * 64, record), F.payload("l", 3, "s" * 64, record)
    assert a != b  # a fresh nonce each time
    body = json.loads(a)
    assert len(body["nonce"]) == 2 * F.NONCE_BYTES and body["record"] == record
    stub = json.loads(F.payload("l", 4, "s" * 64, None))
    assert isinstance(stub["record"], str) and len(stub["record"]) == 2 * F.STUB_BYTES
    env = F.envelope("l", 3, a)
    assert set(env) == {"line", "n", "commitment"}
    assert env["commitment"] == hashlib.sha256(a).hexdigest()
    assert F.opened(a, env)["record"] == record
    with pytest.raises(ValueError, match="committed envelope"):
        F.opened(b, env)


def seal_script():
    import importlib.util

    from residual_disposition.paths import REPO_ROOT

    spec = importlib.util.spec_from_file_location("follow_seal", REPO_ROOT / "scripts"
                                                  / "follow_seal.py")  # fmt: skip
    seal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seal)
    return seal


def seal_repo(path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(path)], check=True)
    for k, v in (("user.email", "r@example.org"), ("user.name", "r")):
        subprocess.run(["git", "-C", str(path), "config", k, v], check=True)
    return path


def test_a_receipt_must_place_the_runner_s_own_envelope_in_its_commit(tmp_path):
    import base64

    seal, repo = seal_script(), seal_repo(tmp_path / "repo")
    env = F.envelope("pun_intended_200", 1, b"x")
    good = seal.commit(env, repo)
    assert F.receipt_problems(good, env) == []
    assert F.receipt_problems({**good, "commit": "HEAD"}, env)
    assert F.receipt_problems({**good, "envelope": {**env, "n": 2}}, env)
    # a commit named with the envelope but no objects, or objects of another commit, proves nothing
    assert F.receipt_problems({"commit": "0" * 40, "envelope": env}, env)
    assert F.receipt_problems({**good, "commit": "0" * 40}, env) == [
        "the commit object is not the commit named"]  # fmt: skip
    other = F.envelope("pun_intended_200", 2, b"y")
    second = seal.commit(other, repo)
    assert F.receipt_problems({**second, "envelope": env}, env)  # envelope 2's blob, not 1's
    forged = {**good, "objects": {**good["objects"],
                                  "blob": base64.b64encode(b'{"forged": 1}').decode()}}  # fmt: skip
    assert F.receipt_problems(forged, env) == ["the blob is not the committed envelope's"]
    moved = F.envelope("eli5_stack_200", 1, b"x")
    assert F.receipt_problems({**seal.commit(moved, repo), "envelope": env}, env)  # another line


def test_the_schedule_and_the_status(tmp_path):
    assert [F.schedule(100.0, t, 30) for t in (100.0, 1899.0, 1900.0, 5500.0)] == [0, 0, 1, 3]
    path = tmp_path / "status.json"
    F.status(path, "failed", "RuntimeError")
    assert json.loads(path.read_text()) == {"status": "failed", "cause": "RuntimeError"}
    with pytest.raises(ValueError):
        F.status(path, "crossed")


def runner():
    import importlib.util

    from residual_disposition.paths import REPO_ROOT

    spec = importlib.util.spec_from_file_location(
        "follow_certificate_pod", REPO_ROOT / "scripts" / "pod" / "follow_certificate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_runner_waits_for_the_receipt_of_an_envelope_with_the_record(tmp_path):
    pod, seal, repo = runner(), seal_script(), seal_repo(tmp_path / "repo")
    out, sealed = tmp_path / "out", tmp_path / "sealed"
    env = pod.Envelopes(out, sealed, "pun_intended_200", "s" * 64, 30, 0.0)
    env.is_alive = lambda: True  # its worker running
    env.write(0)  # before the record: a stub
    env.record = {"status": "resolved"}
    env.write(1)
    assert env.records == {0: None, 1: {"status": "resolved"}}
    assert json.loads((sealed / "payloads" / "1.json").read_text())["record"] == env.record
    assert set(json.loads((out / "envelopes" / "0.json").read_text())) == {"line", "n",
                                                                          "commitment"}  # fmt: skip
    (out / "receipts").mkdir()
    first = seal.commit(env.written[0], repo)
    (out / "receipts" / "0.json").write_text(json.dumps(first))
    (out / "receipts" / "1.json").write_text(json.dumps(seal.commit(env.written[1], repo)))
    assert pod.receipt(out, env, {"status": "resolved"}) == 1  # envelope 0's does not open them
    (out / "receipts" / "1.json").write_text(json.dumps(first))
    with pytest.raises(ValueError, match="not the runner's"):
        pod.receipt(out, env, {"status": "resolved"})


def test_a_resumed_receipt_must_commit_this_calibration(tmp_path):
    pod, seal, repo = runner(), seal_script(), seal_repo(tmp_path / "repo")
    out, sealed = tmp_path / "out", tmp_path / "sealed"
    env = pod.Envelopes(out, sealed, "pun_intended_200", "s" * 64, 30, 0.0)
    env.write(0)  # a stub, committed
    (out / "receipts").mkdir()
    (out / "receipts" / "0.json").write_text(json.dumps(seal.commit(env.written[0], repo)))
    # the stub's payload turned into a record: its bytes are no longer its envelope's
    path = sealed / "payloads" / "0.json"
    payload = json.loads(path.read_text())
    path.write_text(json.dumps({**payload, "record": {"status": "resolved"}}))
    with pytest.raises(ValueError, match="committed envelope"):
        pod.Envelopes(out, sealed, "pun_intended_200", "s" * 64, 30, 9999.0)
    path.write_text(F.canonical(payload).decode())
    again = pod.Envelopes(out, sealed, "pun_intended_200", "s" * 64, 30, 9999.0)
    again.is_alive = lambda: True
    again.record = {"status": "resolved"}
    again.write(1)
    (out / "receipts" / "1.json").write_text(json.dumps(seal.commit(again.written[1], repo)))
    assert pod.receipt(out, again, {"status": "resolved"}) == 1
    # a receipt opens the runs only for this calibration's own record
    assert pod.holding(again, {"status": "resolved"}) == [1]
    assert pod.holding(again, {"status": "resolved", "kstar": 1}) == []
    with pytest.raises(ValueError, match="not this line's"):  # another line's or B3's
        pod.Envelopes(out, sealed, "pun_intended_200", "t" * 64, 30, 9999.0)


def test_an_interrupted_payload_gets_its_envelope_and_is_never_rewritten(tmp_path):
    pod = runner()
    out, sealed = tmp_path / "out", tmp_path / "sealed"
    env = pod.Envelopes(out, sealed, "pun_intended_200", "s" * 64, 30, 0.0)
    env.write(0)
    data = F.payload("pun_intended_200", 1, "s" * 64, None)
    (sealed / "payloads" / "1.json").write_bytes(data)  # its envelope never written
    (sealed / "payloads" / "2.json").write_bytes(data[:40])  # cut while written
    again = pod.Envelopes(out, sealed, "pun_intended_200", "s" * 64, 30, 9999.0)
    assert again.written[1] == F.envelope("pun_intended_200", 1, data) and 2 not in again.written
    assert (sealed / "payloads" / "1.json").read_bytes() == data and again.next == 3
    assert (sealed / "payloads" / "2.json").read_bytes() == data[:40]


def test_the_laptop_commits_each_envelope_alone_and_returns_its_receipt(tmp_path):
    seal, repo = seal_script(), seal_repo(tmp_path / "repo")
    env = F.envelope("pun_intended_200", 0, b"payload")
    receipt = seal.commit(env, repo)
    assert F.receipt_problems(receipt, env) == []
    assert seal.commit(env, repo)["commit"] == receipt["commit"]  # a rerun commits nothing
    for bad in ({**env, "record": 1}, {**env, "line": "eli5_cont_200"},
                {**env, "commitment": "x" * 64}, {**env, "n": "1"}):  # fmt: skip
        with pytest.raises(SystemExit):
            seal.commit(bad, repo)


def test_the_session_s_inputs_are_named_by_its_line():
    import json as _json

    from residual_disposition.paths import REPO_ROOT

    line = F.line_of(b3(), "pun_intended_200")
    y2 = _json.loads((REPO_ROOT / "manifests" / "y2.json").read_text())
    deps = F.dependencies(line, y2, b3())
    assert "20261022-110009-stage3_selection-d2020a97-043f" in deps["runs"]  # B3 loads there
    assert "20261013-112321-relearn-0818ca61-1c85" in deps["runs"]
    assert deps["states"][:3] == [line["latent"], line["sham_s"], line["active"]]
    assert line["reference"] in deps["runs"] and "d72d4f29820114b7b" in deps["data"]


def test_every_workspace_and_writer_is_in_the_sealed_directory(tmp_path):
    from residual_disposition.pod_space import WRITERS

    space, sealed = tmp_path / "a", tmp_path / "a" / "follow" / "cert" / "l" / "sealed"
    ws = SimpleNamespace(work=space / "data" / "cache" / "work",
                         served=space / "data" / "cache" / "served")  # fmt: skip
    assert F.unsealed({"b": ws}, sealed) == ["b"]
    F.seal_workspace(ws, sealed)
    assert F.unsealed({"b": ws}, sealed) == [] and ws.work.is_dir() and ws.served.is_dir()
    env = {"RUNS_DIR": str(sealed / "runs"), "DATA_DIR": str(space / "data"),
           "HF_HOME": str(space / "hf"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           **{name: str(sealed / "cache" / name.lower()) for name in WRITERS}}  # fmt: skip
    assert F.writer_problems(space, sealed, env) == []
    for change in ({"HF_HUB_OFFLINE": "0"}, {"TMPDIR": "/tmp"}, {"RUNS_DIR": str(space / "runs")},
                   {"XDG_CACHE_HOME": str(space / "cache")}, {"TRANSFORMERS_OFFLINE": "0"},
                   {"HF_HUB_CACHE": "/workspace/public/hub"},
                   {"TRANSFORMERS_CACHE": str(space / "cache")}):  # fmt: skip
        assert F.writer_problems(space, sealed, {**env, **change})
    missing = dict(env)
    missing.pop("VLLM_CACHE_ROOT")
    assert any("VLLM_CACHE_ROOT" in p for p in F.writer_problems(space, sealed, missing))


def test_a_hub_write_shows_in_its_listing(tmp_path):
    hub = tmp_path / "hf"
    (hub / "hub" / "models--x").mkdir(parents=True)
    (hub / "hub" / "models--x" / "w.safetensors").write_text("w")
    before = F.listing(hub)
    assert F.listing(hub) == before
    (hub / "hub" / ".locks").mkdir()
    (hub / "hub" / ".locks" / "x.lock").write_text("")
    assert F.listing(hub) != before


def test_the_envelopes_resume_their_numbering_and_raise_their_failure(tmp_path):
    pod = runner()
    out, sealed = tmp_path / "out", tmp_path / "sealed"
    first = pod.Envelopes(out, sealed, "l", "s" * 64, 30, 100.0)
    first.write(0)
    first.record = {"status": "resolved"}
    first.write(1)
    again = pod.Envelopes(out, sealed, "l", "s" * 64, 30, 9999.0)  # a rerun
    assert (again.t0, again.next, again.records) == (100.0, 2, {0: None, 1: first.record})
    assert again.written == first.written
    with pytest.raises(ValueError, match="cadence"):
        pod.Envelopes(out, sealed, "l", "s" * 64, 15, 9999.0)
    again.error = OSError("disk full")
    with pytest.raises(RuntimeError, match="stopped"):
        pod.receipt(out, again, first.record)
    with pytest.raises(RuntimeError, match="stopped"):  # never started: not alive
        pod.receipt(out, pod.Envelopes(tmp_path / "o2", tmp_path / "s2", "l", "s" * 64, 30, 1.0),
                    first.record)  # fmt: skip


def test_a_rerun_keeps_its_sealed_record(tmp_path):
    pod = runner()
    pod.keep(tmp_path / "r.json", {"a": 1})
    pod.keep(tmp_path / "r.json", {"a": 1})
    with pytest.raises(ValueError, match="rerun"):
        pod.keep(tmp_path / "r.json", {"a": 2})


def cell_config(purpose="certificate", binding=BINDING, seed=0):
    """A relearning cell's config as a run records it (the fields its identity holds)."""
    params = {"purpose": purpose, "follow": dict(binding), "relearn_seed": seed}
    return {"protocol_version": 1, "model": {"name": "m"}, "behavior": {"id": "pun_intended"},
            "step": "relearn", "params": params}  # fmt: skip


def digest_of(config):
    from omegaconf import OmegaConf

    from residual_disposition.config import config_hash

    return config_hash(OmegaConf.create(config))


def cells_of(*configs):
    from residual_disposition.continuation_panel import IDENTITY

    return {digest_of(c): {k: c.get(k) for k in IDENTITY} for c in configs}


def relearn_run(runs, name, status, cause=None, config=None, digest=None):
    import yaml

    config = config or cell_config()
    run = runs / "continuation_14b" / name
    run.mkdir(parents=True)
    meta = {"config_hash": digest or digest_of(config), "status": status}
    (run / "meta.json").write_text(json.dumps(meta))
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    if cause:
        (run / "failure.json").write_text(json.dumps({"outcome": cause}))
    return run


def test_a_cell_is_retried_once_and_its_second_classified_failure_ends_the_phase(tmp_path):
    from residual_disposition.training import TrainingDivergence

    calls, cells, cell = [], cells_of(cell_config()), digest_of(cell_config())

    def phase(runs, outcomes):
        def run():
            calls.append(1)
            if outcomes.pop(0) == "ok":
                relearn_run(runs, f"r{len(calls)}", "completed")
                return "result"
            relearn_run(runs, f"r{len(calls)}", "failed", "divergence")
            raise TrainingDivergence("loss")

        return run

    once = tmp_path / "once"
    assert F.with_retry("certificate", phase(once, ["fail", "ok"]), once, BINDING,
                        cells) == ("result", {})  # fmt: skip
    calls.clear()
    runs = tmp_path / "twice"
    result, failed = F.with_retry("certificate", phase(runs, ["fail", "fail", "ok"]), runs,
                                  BINDING, cells)  # fmt: skip
    assert result is None and failed == {cell: ["r1", "r2"]} and len(calls) == 2  # never a third
    # a later session finds the history and attempts nothing
    assert F.with_retry("certificate", phase(runs, ["ok"]), runs, BINDING, cells) == (None, failed)
    assert len(calls) == 2
    record = F.terminal(F.line_of(b3(), "pun_intended_200") | {"binding": BINDING}, "certificate",
                        failed)  # fmt: skip
    assert record["item"] == "follow/cert/pun_intended" and record["runs"] == ["r1", "r2"]


def test_the_retry_decides_only_from_the_phase_s_approved_cells(tmp_path):
    from residual_disposition.training import TrainingDivergence

    cells = cells_of(cell_config())
    runs = tmp_path / "runs"
    relearn_run(runs, "x1", "failed")  # an engine or loading error: no cause, no attempt
    other = cell_config(binding={**BINDING, "line": "other"})  # another line's
    relearn_run(runs, "x2", "failed", "divergence", config=other)
    relearn_run(runs, "x3", "failed", "divergence", config=cell_config("kstar_calibration"))
    assert F.exhausted(F.history(runs, BINDING, "certificate", cells), "certificate") == {}

    def lost():
        raise ConnectionError("network")

    with pytest.raises(ConnectionError):
        F.with_retry("certificate", lost, runs, BINDING, cells)

    def unrecorded():  # classified, but no failed run of the phase's cells records it
        raise TrainingDivergence("loss")

    with pytest.raises(TrainingDivergence):
        F.with_retry("certificate", unrecorded, runs, BINDING, cells)
    with pytest.raises(ValueError, match="approved cells are not established"):
        F.with_retry("certificate", lost, runs, BINDING, {})
    # failures that are not the phase's cells never close it: the decision is refused
    for n, kwargs in enumerate(({"digest": "0" * 12}, {"config": cell_config(seed=9)})):
        bad = tmp_path / f"bad{n}"
        for name in ("f1", "f2"):
            relearn_run(bad, name, "failed", "divergence", **kwargs)
        with pytest.raises(ValueError, match="not the cell its meta names|not an approved"):
            F.with_retry("certificate", lambda: "result", bad, BINDING, cells)
    done = tmp_path / "done"  # a completed run that is not an approved cell, likewise
    relearn_run(done, "c1", "completed", config=cell_config(seed=9))
    with pytest.raises(ValueError, match="not an approved"):
        F.history(done, BINDING, "certificate", cells)


def test_a_terminal_record_names_its_line_and_item_and_a_budget_stop_the_meter():
    line = F.line_of(b3(), "pun_intended_200")
    assert F.item_of(line) == "follow/cert/pun_intended"  # B3's allocation, not the line name
    budget = {**F.terminal(line, "kstar", {}), "kind": "budget", "runs": [],
              "meter": {"item": "follow/cert/pun_intended", "state": "stopped"}}  # fmt: skip

    def check(data, state="stopped"):
        return F.terminal_problems(data, line, lambda n: None, lambda item: state,
                                   lambda e, h: None)  # fmt: skip

    assert check(budget) == []
    assert check(budget, "running")  # the meter must hold the stop
    assert check({**budget, "meter": {"item": "follow/cert/pun_intended_200",
                                      "state": "stopped"}})  # fmt: skip
    assert check({**budget, "meter": {"item": "stage3/pun_intended/follow", "state": "stopped"}})
    assert check({**budget, "line": "eli5_stack_200"})
    assert check({**budget, "item": "follow/base"})
    exhausted = F.terminal(line, "certificate", {"c1": ["r1", "r2"]})
    assert check(exhausted)  # without the phase's approved cells nothing is established


def test_each_phase_s_approved_cells_are_its_planned_runs(monkeypatch):
    from omegaconf import OmegaConf

    calls = []
    line, ws, relearn = fake_world(monkeypatch, calls)

    def cell(ws, state, **args):
        config = cell_config(args["purpose"], args["binding"], args["relearn_seed"])
        config["params"] |= {"lr": args["lr"], "state": state}
        return SimpleNamespace(cell=SimpleNamespace(cfg=OmegaConf.create(config),
                                                    digest=digest_of(config)))  # fmt: skip

    monkeypatch.setattr(P, "relearn_cell", cell)
    cells = F.calibration_cells(ws, line, rules())
    assert len(cells) == 6 and all(
        c["params"]["purpose"] == "kstar_calibration" and c["params"]["follow"] == line["binding"]
        and c["params"]["state"] == line["sham_s"] for c in cells.values())  # fmt: skip


def test_the_plain_launches_commit_the_calibration_between_them(tmp_path, monkeypatch):
    pod = runner()
    record = {"status": "resolved", "budget": 150}
    runs = []
    ws = SimpleNamespace(store=SimpleNamespace(verify=lambda s: None))
    monkeypatch.setattr(pod.F, "workspace", lambda line: ws)
    monkeypatch.setattr(pod.F, "reference", lambda ws, line: {})
    monkeypatch.setattr(pod.F, "data", lambda ws, line: None)
    monkeypatch.setattr(pod.F, "calibrate", lambda ws, line, rules: dict(record))
    monkeypatch.setattr(pod.F, "certify", lambda ws, line, rules, k: runs.append(k) or {"runs": 72})
    monkeypatch.setattr(P, "certificate_controls", lambda ws, rules: {})
    monkeypatch.setattr(pod.cr, "follow_b3", b3)
    monkeypatch.setattr(pod.cr, "load_rules", lambda **kwargs: {})
    out = tmp_path / "out"
    name = "pun_intended_200"
    assert pod.plain(name, "calibration", out, None) == 0 and not runs
    manifest = out / "kstar_calibration.json"
    assert json.loads(manifest.read_text()) == record
    assert json.loads((out / "status.json").read_text()) == {"status": "completed"}
    assert pod.plain(name, "certificate", out, manifest) == 0 and runs == [record]
    assert json.loads((out / "certificate_blocks.json").read_text()) == {"runs": 72}
    other = tmp_path / "other.json"
    other.write_text(json.dumps({**record, "budget": 200}))
    runs.clear()
    with pytest.raises(ValueError, match="committed manifest"):
        pod.plain(name, "certificate", out, other)
    assert not runs  # no cell runs on another calibration
    assert json.loads((out / "status.json").read_text()) == {"status": "failed",
                                                             "cause": "ValueError"}  # fmt: skip
