"""The existing L packages' random branches (§13.4; barriers B2 and B3): the session is exactly the
packages B3's inventory names, each L's random branch alone at its package's recorded state and
seed; its output is recorded once and found only through that record."""

from __future__ import annotations

import copy
import json

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


def test_the_session_is_refused_unless_it_is_the_records():
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


def recorded(tmp_path):
    run = tmp_path / "follow_lrandom" / "follow-lrandom-x"
    d = run / "out" / "follow-lrandom-x" / "builds" / "sL"
    (d / "steps").mkdir(parents=True)
    (d / "build.json").write_text("{}")
    (d / "steps" / "a.json").write_text("[1]")
    files = {"build.json": fi._sha(d / "build.json"), "steps/a.json": fi._sha(d / "steps/a.json")}
    return {"run": run.name, "builds": {"sL": {"package": "pkg-L", "files": files,
                                               "dir": str(d.relative_to(run))}}}  # fmt: skip


def test_a_random_branch_is_found_only_through_its_record(tmp_path):
    record = recorded(tmp_path)
    got = fi.random_branches(record, tmp_path)
    assert got == {"random branch of pkg-L": {"build": "follow-lrandom-x", "state": "sL",
                                              "dir": record["builds"]["sL"]["dir"]}}  # fmt: skip
    d = tmp_path / "follow_lrandom" / "follow-lrandom-x" / record["builds"]["sL"]["dir"]
    (d / "steps" / "a.json").write_text("[2]")
    with pytest.raises(ValueError, match="recorded files"):
        fi.random_branches(record, tmp_path)
    (d / "steps" / "a.json").write_text("[1]")
    (tmp_path / "other" / "follow-lrandom-x").mkdir(parents=True)  # a retry's second directory
    with pytest.raises(ValueError, match="found 2 times"):
        fi.random_branches(record, tmp_path)


def test_the_session_is_not_run_while_unfrozen(monkeypatch):
    monkeypatch.setattr(cr, "FOLLOW_LRANDOM_FROZEN", None)
    with pytest.raises(ValueError, match="not frozen"):
        cr.follow_lrandom()
