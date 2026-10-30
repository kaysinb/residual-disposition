"""The follow-up's stack branch on eli5 (§13.7; barrier B4): its candidates Stage 3's saved stack
states at 400 then 800; the screening in order, ended by the first eligible candidate, stopped by
invalid evidence before the next, closing the branch when none is eligible; the record
recomputed from the runs it names; the suites V4 and O4 drawn with new keys, disjoint from V1–V3
and the screening suites; and the frozen configuration."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_strong as CS
from residual_disposition import follow_stack as F
from residual_disposition import strong_suites as S

DETECTOR = SimpleNamespace(version="43943945ee35")
B4 = {
    "behavior": "eli5",
    "lineage": "follow/eli5_stack",
    "detector": "43943945ee35",
    "candidates": [{"dose": 400, "latent": "sL4", "sham_s": "sH4", "recipe": "sR4"},
                   {"dose": 800, "latent": "sL8", "sham_s": "sH8", "recipe": "sR8"}],
    "screening": {"pinned": {}},
}  # fmt: skip


def zero():
    return {"by_config": {f: {"hits": 0} for f in CS.PD_FRAMINGS}}


def adl(hits=0):
    return {f: {"pooled_hits": hits, "long_hits": 0} for f in CS.ADL_FRAMINGS}


class Runs:
    """Completed stand-in runs by name, each its saved configuration and recorded hash; the cells'
    digests (``digests``) the runs' own unless ``other`` names a run of another cell; and the
    counts each candidate's runs give."""

    def __init__(self, tmp_path, counts, other=()):
        self.root, self.counts, self.other, self.counted = tmp_path, counts, set(other), []

    def run(self, name):
        from omegaconf import OmegaConf

        from residual_disposition.config import config_hash

        d = self.root / name
        d.mkdir(exist_ok=True)
        config = {"step": name.split("-")[0], "params": {"state": name.split("-")[1]}}
        (d / "config.yaml").write_text(yaml.safe_dump(config))
        meta = {"status": "completed", "config_hash": config_hash(OmegaConf.create(config))}
        (d / "meta.json").write_text(__import__("json").dumps(meta))
        return d

    def digest(self, name):
        meta = __import__("json").loads((self.root / name / "meta.json").read_text())
        return "0" * 12 if name in self.other else meta["config_hash"]

    def digests(self, latent):
        return self.digest(f"pd-{latent}"), self.digest(f"adl-{latent}")

    def find(self, name):
        d = self.root / str(name)
        return d if d.is_dir() else None

    def count(self, pd_run, adl_run, latent, detector):
        self.counted.append(latent)
        return self.counts[latent]


def screened(runs, b4=B4):
    return F.screen(b4, DETECTOR, audit=lambda s: runs.run(f"pd-{s}"),
                    adl_check=lambda s: runs.run(f"adl-{s}"), digests=runs.digests,
                    count=runs.count)  # fmt: skip


def recorded(runs, inputs, b4=B4):
    return F.screening_record(b4, inputs, runs.find, DETECTOR, runs.digests, count=runs.count)


def test_the_first_eligible_candidate_ends_the_screening(tmp_path):
    runs = Runs(tmp_path, {"sL4": (zero(), adl()), "sL8": (zero(), adl())})
    inputs = screened(runs)
    assert [c["dose"] for c in inputs["candidates"]] == [400]  # 800 is never screened
    record = recorded(runs, inputs)
    assert record["chosen"] == {"dose": 400, "latent": "sL4", "sham_s": "sH4"}
    assert record["outcome"] == CS.SCREENED and record["screened"][0]["eligible"]


def test_no_eligible_candidate_closes_the_branch_and_keeps_every_screen(tmp_path):
    runs = Runs(tmp_path, {"sL4": (zero(), adl(1)), "sL8": (zero(), adl(2))})
    record = recorded(runs, screened(runs))
    assert record["chosen"] is None and record["outcome"] == F.CLOSED
    assert [(r["dose"], r["eligible"]) for r in record["screened"]] == [(400, False), (800, False)]


def test_invalid_evidence_stops_the_screening_before_the_next_candidate(tmp_path):
    runs = Runs(tmp_path, {"sL4": (None, adl()), "sL8": (zero(), adl())})
    with pytest.raises(CS.Incomplete, match="invalid"):
        screened(runs)
    assert not (tmp_path / "pd-sL8").exists()


def test_the_record_is_recomputed_from_the_runs_it_names(tmp_path):
    runs = Runs(tmp_path, {"sL4": (zero(), adl(1)), "sL8": (zero(), adl())})
    inputs = screened(runs)

    def record(change):
        changed = copy.deepcopy(inputs)
        change(changed)
        return recorded(runs, changed)

    assert record(lambda d: None)["chosen"]["dose"] == 800
    with pytest.raises(ValueError, match="not the frozen"):
        record(lambda d: d["candidates"].reverse())
    with pytest.raises(ValueError, match="stopped before an eligible"):
        record(lambda d: d["candidates"].pop())
    with pytest.raises(ValueError, match="not here"):
        record(lambda d: d["candidates"][0].update(pd="missing"))
    with pytest.raises(ValueError, match="the branch's is"):
        record(lambda d: d.update(detector_version="0" * 12))
    eligible = Runs(tmp_path, {"sL4": (zero(), adl()), "sL8": (zero(), adl())})
    with pytest.raises(ValueError, match="went on after an eligible"):
        recorded(eligible, inputs)
    with pytest.raises(ValueError, match="not the branch's"):
        F.screen({**B4, "detector": "f" * 12}, DETECTOR, audit=None, adl_check=None,
                 digests=None)  # fmt: skip


@pytest.mark.parametrize("other", ["pd-sL4", "adl-sL4"])
def test_evidence_of_another_cell_is_refused_before_anything_is_counted(tmp_path, other):
    runs = Runs(tmp_path, {"sL4": (zero(), adl()), "sL8": (zero(), adl())}, other=[other])
    with pytest.raises(ValueError, match="hashes to"):
        screened(runs)
    assert runs.counted == []
    honest = Runs(tmp_path, {"sL4": (zero(), adl()), "sL8": (zero(), adl())})
    inputs = screened(honest)
    with pytest.raises(ValueError, match="hashes to"):
        recorded(runs, inputs)


def test_the_pinned_pool_is_that_run_whole(tmp_path):
    from residual_disposition.follow_confirm import file_sha256

    runs = Runs(tmp_path, {"sL4": (zero(), adl(1)), "sL8": (zero(), adl())})
    pool = runs.run("pd-sL4")
    files = {f: file_sha256(pool / f) for f in ("config.yaml", "meta.json")}
    b4 = {**B4, "screening": {"pinned": {"sL4": {"run": "pd-sL4", "files": files}}}}
    inputs = screened(runs, b4)
    assert recorded(runs, inputs, b4)["screened"][0]["eligible"] is False
    assert F.pinned_problems(b4, "sL4", None)  # absent: refused, never redrawn
    assert F.pinned_problems(b4, "sL4", runs.run("pd-sL8"))  # another run
    (pool / "config.yaml").write_text((pool / "config.yaml").read_text() + "# changed\n")
    assert "not the pinned files" in F.pinned_problems(b4, "sL4", pool)[0]
    assert F.pinned_problems(b4, "sL8", None) == []  # 800 has no pinned pool


def test_the_suites_draw_with_new_keys_disjoint_from_the_frozen_ones():
    population = [(i, i + 1, i + 2) for i in range(400)]
    old = S.draw_pd_suites(population, 7, names=("V1", "V2"), size=50)
    assert old == S.draw_pd_suites(population, 7, names=("V1", "V2"), size=50, key="strong_pd")
    taken = {tuple(r) for s in old.values() for r in s.records.values()}
    rest = [p for p in population if p not in taken]
    new = S.draw_pd_suites(rest, 7, names=("V4",), size=50, key="follow_pd")
    S.check_disjoint_suites([*old.values(), *new.values()])
    assert new["V4"].provenance["rng"] == "random.Random(derive_seed(seed, 'follow_pd')).shuffle"
    docs = [(f"{i:064x}", tuple(range(40))) for i in range(3000)]
    kw = {"lengths": (1, 2), "per_length": 20, "long": (32, 30)}
    a = S.draw_adl_suites(docs, 7, names=("V1",), **kw)
    assert a == S.draw_adl_suites(
        docs, 7, names=("V1",), keys=("strong_adl_stratum", "strong_adl"), **kw
    )
    used = {r["document"] for r in a["V1"].records.values()}
    b = S.draw_adl_suites([d for d in docs if d[0] not in used], 7, names=("V4",),
                          keys=("follow_adl_stratum", "follow_adl"), **kw)  # fmt: skip
    S.check_disjoint_suites([*a.values(), *b.values()])
    assert "follow_adl_stratum" in b["V4"].provenance["rng"]


def config(tmp_path, **change):
    b4 = yaml.safe_load(cr.FOLLOW_B4_PATH.read_text())["b4"]
    path = tmp_path / "follow_b4.yaml"
    path.write_text(yaml.safe_dump({"b4": {**b4, **change}}))
    return path, b4


def test_the_configuration(tmp_path):
    path, b4 = config(tmp_path)
    assert cr.load_follow_b4(path) == b4
    assert cr.follow_b4() == b4  # the frozen file
    with pytest.raises(ValueError, match="changed"):
        cr.load_follow_b4(path, expected_hash="0" * 64)
    with pytest.raises(ValueError, match="400, 800"):
        cr.load_follow_b4(config(tmp_path, candidates=b4["candidates"][::-1])[0])
    keys = {**b4["suites"], "keys": {"pd": "strong_pd", "adl": b4["suites"]["keys"]["adl"]}}
    with pytest.raises(ValueError, match="not new"):
        cr.load_follow_b4(config(tmp_path, suites=keys)[0])
    with pytest.raises(ValueError, match="final PD GPU"):
        cr.load_follow_b4(config(tmp_path, screening={**b4["screening"], "gpu": "NVIDIA A100"})[0])
    old = {**b4["validation"], "streams": {"observer": "strong:x:observer", "ood": "follow:x:ood"}}
    with pytest.raises(ValueError, match="not new and distinct"):
        cr.load_follow_b4(config(tmp_path, validation=old)[0])
    other = {**b4["screening"], "pinned": {"sX": b4["screening"]["pinned"]["se244833f590245e7"]}}
    with pytest.raises(ValueError, match="not a candidate's"):
        cr.load_follow_b4(config(tmp_path, screening=other)[0])


def test_the_contract_is_the_panel_contracts_eli5_one(monkeypatch):
    import json

    from residual_disposition import pipeline as P
    from residual_disposition.config import load_config

    b4 = cr.follow_b4()
    panel = json.loads((cr.REPO_ROOT / "manifests/stage3_panel_contracts.json").read_text())
    k = panel["behaviors"]["eli5"]["contract"]
    assert P.content_hash({x: k[x] for x in F.CONTRACT}) == b4["contract"]["sha256"]
    assert b4["contract"]["requests"]["ood"] == k["requests"]["ood"]
    ws = P.Workspace(load_config(b4["experiment"]))
    assert F.contract_problems(ws, b4) == []
    monkeypatch.setattr(ws, "engine", lambda: {"max_model_len": 8192})
    assert F.contract_problems(ws, b4)


def test_the_candidates_are_stage_3_s_saved_stack_states():
    from residual_disposition.continuation_kstar import find_run

    eli5 = __import__("json").loads((cr.REPO_ROOT / "manifests/stage3_eli5.json").read_text())
    if find_run(eli5["inputs"]["selection"]) is None:
        pytest.skip("the records' runs are not here")
    b4 = cr.follow_b4()
    recorded = F.recorded_candidates(cr.FOLLOW_B4_CANDIDATES)
    assert b4["candidates"] == recorded["candidates"] and b4["implant"] == recorded["implant"]
    contracts = __import__("json").loads(
        (cr.REPO_ROOT / "manifests/stage3_panel_contracts.json").read_text())  # fmt: skip
    assert b4["detector"] == contracts["behaviors"]["eli5"]["detector"]["version"]


def test_the_saved_suites_load_at_their_frozen_hashes():
    from residual_disposition.paths import data_root

    b4 = cr.follow_b4()
    if not (Path(data_root()) / "strong_suites" / "pd-V4").is_dir():
        pytest.skip("V4 and O4 are not built here")
    for kind, name in (("pd", "V4"), ("adl", "V4"), ("observer", "O4")):
        assert F.suite(kind, name, b4).sha256 == b4["suites"]["sha256"][kind][name]
