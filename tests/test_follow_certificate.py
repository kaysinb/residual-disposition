"""The follow-up's certificates (§13.8; barrier B3) and their seal (§13.14): every relearning cell
of a line carries the line's binding, and a cell without one keeps its identity exactly; the
calibration's six runs and the certificate's 72; the envelopes hide the record and its readiness,
and a receipt must hold the runner's own envelope."""

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


def test_a_receipt_must_hold_the_runner_s_own_envelope():
    env = F.envelope("l", 1, b"x")
    good = {"commit": "a" * 40, "envelope": env}
    assert F.receipt_problems(good, env) == []
    assert F.receipt_problems({**good, "commit": "HEAD"}, env)
    assert F.receipt_problems({**good, "envelope": {**env, "n": 2}}, env)


def test_the_schedule_and_the_status(tmp_path):
    assert [F.schedule(100.0, t, 30) for t in (100.0, 1899.0, 1900.0, 5500.0)] == [0, 0, 1, 3]
    path = tmp_path / "status.json"
    F.status(path, "failed", "RuntimeError")
    assert json.loads(path.read_text()) == {"status": "failed", "cause": "RuntimeError"}
    with pytest.raises(ValueError):
        F.status(path, "crossed")
