"""The engine check's hashes and its frozen rule (preregistration §12.4.8), CPU only."""

from __future__ import annotations

import pytest
import torch

from residual_disposition import engine_check as E
from residual_disposition.generation import Completion
from residual_disposition.states import Lora


def completions(*texts):
    return [
        Completion(key=f"k{i}", sample=0, text=t, truncated=False, n_tokens=2)
        for i, t in enumerate(texts)
    ]


def lora(scale=1.0):
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3) * scale
    return Lora(rank=2, alpha=4.0, modules={"m.q": (a, a.T.clone())}, target_modules=("q",))


def fingerprint(texts=("a", "b"), scale=1.0, t=12.5):
    f = E.Fingerprint()
    f.evaluation(0, completions(*texts))
    f.evaluation(2, completions(*texts))
    f.adapter(2, lora(scale))
    f.finish(crossing_step=2, censored=False, t=t)
    return f.as_dict()


def test_digests_are_exact_and_order_sensitive():
    assert E.completions_digest(completions("a", "b")) == E.completions_digest(
        completions("a", "b")
    )
    assert E.completions_digest(completions("a", "b")) != E.completions_digest(
        completions("b", "a")
    )
    assert E.adapter_digest(lora()) == E.adapter_digest(lora())
    assert E.adapter_digest(lora()) != E.adapter_digest(lora(1 + 2**-20))
    assert E.value_digest(12.5) != E.value_digest(12.500000000000002)
    assert E.value_digest(None) != E.value_digest(0)


def test_the_fingerprint_keeps_hashes_only_and_refuses_a_repeated_step():
    f = fingerprint()
    blob = repr(f)
    assert "12.5" not in blob and "'a'" not in blob
    assert set(f) == {"evaluations", "adapters", "outcome"}
    g = E.Fingerprint()
    g.evaluation(0, completions("a"))
    with pytest.raises(ValueError):
        g.evaluation(0, completions("a"))


def test_compare_names_the_differing_parts_only():
    same = fingerprint()
    assert E.compare(same, fingerprint()) == []
    assert E.compare(same, fingerprint(texts=("a", "c"))) == ["evaluation 0", "evaluation 2"]
    assert E.compare(same, fingerprint(scale=2.0)) == ["adapter 2"]
    assert E.compare(same, fingerprint(t=13.0)) == ["t"]
    shorter = fingerprint()
    del shorter["evaluations"]["2"]
    assert E.compare(same, shorter) == ["evaluation steps"]
    assert E.compare(same, {**same, "outcome": {}})[0] == "outcome missing"


def test_the_rule_needs_every_pair_complete_identical_and_no_failure():
    planned = [E.pair_key("L", 1e-5), E.pair_key("H_s", 1e-5)]
    good = {k: {E.SLEEPING: fingerprint(), E.AWAKE: fingerprint()} for k in planned}
    v = E.verdict(planned, good, [])
    assert v["passed"] and v["mode"] == E.AWAKE
    one_off = {**good, planned[1]: {E.SLEEPING: fingerprint(), E.AWAKE: fingerprint(t=1.0)}}
    v = E.verdict(planned, one_off, [])
    assert not v["passed"] and v["mode"] == E.SLEEPING
    assert v["pairs"][planned[1]]["differences"] == ["t"]
    incomplete = {planned[0]: good[planned[0]], planned[1]: {E.SLEEPING: fingerprint()}}
    v = E.verdict(planned, incomplete, [])
    assert not v["passed"] and v["pairs"][planned[1]]["missing"] == [E.AWAKE]
    failed = E.verdict(planned, good, [{"pair": planned[0], "mode": E.AWAKE, "error": "X"}])
    assert not failed["passed"] and failed["mode"] == E.SLEEPING
    assert not E.verdict([], {}, [])["passed"]
