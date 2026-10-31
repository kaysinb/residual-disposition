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


def fingerprint(texts=("a", "b"), scale=1.0, t=12.5, censored=False):
    f = E.Fingerprint()
    f.evaluation(0, completions(*texts))
    f.evaluation(2, completions(*texts))
    f.adapter(2, lora(scale))
    f.finish(crossing_step=None if censored else 2, censored=censored, t=t)
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


def test_the_fingerprint_keeps_hashes_and_one_flag_and_refuses_a_repeated_step():
    f = fingerprint()
    blob = repr(f)
    assert "12.5" not in blob and "'a'" not in blob
    assert set(f) == {"evaluations", "adapters", "outcome", "crossed_after_training"}
    assert f["crossed_after_training"] is True and E.informative(f)
    assert not E.informative(fingerprint(censored=True))


def test_a_crossing_off_the_recorded_checkpoints_is_not_informative():
    # Evaluations at 0 and 2, an adapter at 2, a declared crossing at 10 (item 2).
    f = E.Fingerprint()
    f.evaluation(0, completions("a"))
    f.evaluation(2, completions("a"))
    f.adapter(2, lora())
    f.finish(crossing_step=10, censored=False, t=9.5)
    assert f.crossed_after_training and not E.informative(f.as_dict())
    g = E.Fingerprint()  # a crossing at 2 with its adapter missing
    g.evaluation(0, completions("a"))
    g.evaluation(2, completions("a"))
    g.finish(crossing_step=2, censored=False, t=1.5)
    assert not E.informative(g.as_dict())
    forged = {**fingerprint(), "crossed_after_training": True}
    forged["outcome"] = {**forged["outcome"], "censored": E.value_digest(True)}
    assert not E.informative(forged)


def test_identical_runs_crossing_off_their_checkpoints_do_not_permit_the_awake_engine(tmp_path):
    def off():
        f = E.Fingerprint()
        f.evaluation(0, completions("a"))
        f.evaluation(2, completions("a"))
        f.adapter(2, lora())
        f.finish(crossing_step=10, censored=False, t=9.5)
        return f.as_dict()

    runs = [(k, m, off()) for k in PLANNED for m in E.MODES]
    v = E.verdict(PLANNED, journal(tmp_path, runs=runs).events(), probe=True)
    assert not v["passed"] and v["mode"] == E.SLEEPING
    assert all(p["identical"] and not p["informative"] for p in v["pairs"].values())
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


PLANNED = [E.pair_key("L", 1e-5), E.pair_key("H_s", 1e-5)]


def journal(tmp_path, *, probe=8, runs=None, name="j"):
    j = E.Journal(tmp_path / name / "journal.jsonl")
    if probe is not None:
        a = j.start(E.PROBE, E.SLEEPING)
        j.end(a, E.PROBE, E.SLEEPING, result={"n": 8, "identical": probe})
    for pair, mode, fp in (
        runs if runs is not None else [(k, m, fingerprint()) for k in PLANNED for m in E.MODES]
    ):
        a = j.start(pair, mode)
        j.end(a, pair, mode, fingerprint=fp)
    return j


def test_the_rule_passes_only_a_reproduced_probe_and_identical_informative_pairs(tmp_path):
    v = E.verdict(PLANNED, journal(tmp_path).events(), probe=True)
    assert v["passed"] and v["mode"] == E.AWAKE and v["probe"]["reproduced"]
    v = E.verdict(PLANNED, journal(tmp_path, probe=7, name="p").events(), probe=True)
    assert not v["passed"] and not v["probe"]["reproduced"]
    v = E.verdict(PLANNED, journal(tmp_path, probe=None, name="n").events(), probe=True)
    assert not v["passed"]  # a required probe that was never recorded
    off = [
        (k, m, fingerprint(t=1.0 if (k, m) == (PLANNED[1], E.AWAKE) else 12.5))
        for k in PLANNED
        for m in E.MODES
    ]
    v = E.verdict(PLANNED, journal(tmp_path, runs=off, name="d").events(), probe=True)
    assert not v["passed"] and v["pairs"][PLANNED[1]]["differences"] == ["t"]
    censored = [(k, m, fingerprint(censored=True)) for k in PLANNED for m in E.MODES]
    v = E.verdict(PLANNED, journal(tmp_path, runs=censored, name="c").events(), probe=True)
    assert not v["passed"] and not v["pairs"][PLANNED[0]]["informative"]
    half = [(PLANNED[0], m, fingerprint()) for m in E.MODES] + [
        (PLANNED[1], E.SLEEPING, fingerprint())
    ]
    v = E.verdict(PLANNED, journal(tmp_path, runs=half, name="h").events(), probe=True)
    assert not v["passed"] and v["pairs"][PLANNED[1]]["missing"] == [E.AWAKE]
    assert not E.verdict([], journal(tmp_path, name="e").events(), probe=True)["passed"]


def test_every_attempt_counts_across_restarts(tmp_path):
    j = journal(tmp_path)
    a = j.start(PLANNED[0], E.AWAKE)  # a later attempt of a finished run, then a crash
    events = j.events()
    v = E.verdict(PLANNED, events, probe=True)
    assert not v["passed"] and v["failures"][0]["attempt"] == a
    assert v["failures"][0]["error"] == "unfinished (the process ended)"
    j.failed(a, PLANNED[0], E.AWAKE, "RuntimeError: x")
    assert [f["error"] for f in j.failures()] == ["RuntimeError: x"]
    k = journal(tmp_path, name="r")  # a restart whose second attempt differs from the first
    b = k.start(PLANNED[1], E.SLEEPING)
    k.end(b, PLANNED[1], E.SLEEPING, fingerprint=fingerprint(texts=("z", "b")))
    v = E.verdict(PLANNED, k.events(), probe=True)
    assert not v["passed"] and v["pairs"][PLANNED[1]]["attempts"][E.SLEEPING] == 2
    assert v["pairs"][PLANNED[1]]["differences"] == ["evaluation 0", "evaluation 2"]
