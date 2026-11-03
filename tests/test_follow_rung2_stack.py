"""Rung 2 on the stack candidate 800 (§13.6, §13.7, §13.16): its readings and salts, the shared A
and H and the steering cells' sources and vectors; the decision is the continued mode's function."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction

import pytest

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_rung2 as R2
from residual_disposition import follow_rung2_stack as RS

RULES = {"qualification": {"floors": {"active_min": "7/10", "sham_max": "1/10", "gap_min": "3/5",
                                      "recipe_ratio": "7/10"}},
         "confusion_min": "3/10", "specificity_min": "1/2"}  # fmt: skip
STATES = {"L": "sL", "H_s": "sH", "X": "sX", "recipe": "sR"}


def test_each_decoder_reads_560_slots_under_their_own_salts_and_b3s_recipe_streams():
    rows = RS.readings({"states": STATES})
    both = [*rows["qualification"], *rows["analysis"]]
    assert sorted(rows) == ["analysis", "qualification"]
    for decoder in (("state", "random"), ("base", "base_random")):
        assert sum(r["slots"] for r in both if r["decoder"] in decoder) == 560
    salts = [r["salt"] for r in both]
    assert len(salts) == len(set(salts))
    inventory = json.loads((cr.REPO_ROOT / cr.follow_b3()["inventory"]["manifest"]).read_text())
    frozen = inventory["rung2"]
    for r in frozen["recipe_stack"]:
        assert r["salt"] in salts
    continued = {r["salt"] for r in [*R2.readings()["qualification"], *R2.readings()["analysis"]]}
    assert not continued & set(salts)
    assert {r["role"] for r in rows["qualification"]} == {"recipe"}  # A and H are shared
    assert all("/stack800/" in r["salt"] for r in rows["analysis"])


def test_each_cell_reads_its_own_state_and_the_random_cell_reads_ls_through_each_decoder():
    cells = {(r["role"], r["decoder"]): r for r in RS.readings({"states": STATES})["analysis"]}
    for role, state in STATES.items():
        for decoder in ("state", "base"):
            assert cells[(role, decoder)]["package"] == {"state": state}
    assert cells[("L_random", "random")]["package"] == {"state": "sL"}
    assert cells[("L_random", "base_random")]["package"] == {"state": "sL"}
    assert cells[("L_random", "base_random")]["salt"].startswith("follow/rung2/base/")


def test_the_committed_build_record_keeps_its_results_digests():
    assert sorted(RS.built()["sha256"]) == ["build", "pools"]


def test_the_state_decoders_readings_go_to_the_rung_and_the_base_decoders_to_the_path():
    rows = [r for part in RS.readings({"states": STATES}).values() for r in part]
    items = {(r["decoder"], RS.item_of(r)) for r in rows}
    rung, path = "follow/stack/rung2", "follow/stack/full"
    assert items == {("state", rung), ("random", rung), ("base", path), ("base_random", path)}
    assert sum(r["slots"] for r in rows if RS.item_of(r) == rung) == 560


def test_the_session_steers_five_cells_and_l_before_its_random_cell():
    cells = RS.steering_cells()
    names = [c["cell"] for c in cells]
    assert sorted(names) == sorted(["recipe", "L", "H_s", "X", "L_random"])
    assert names.index("L") < names.index("L_random")
    by = {c["cell"]: c for c in cells}
    assert by["L_random"]["run"] == by["L"]["run"] and by["L_random"]["random"]
    assert not any(c["random"] for c in cells if c["cell"] != "L_random")
    assert R2.STEERING_CELLS["recipe"] == "recipe"


def build_dir(tmp_path, *, second=None):
    import torch
    from safetensors.torch import save_file

    d = tmp_path / "s1"
    d.mkdir()
    delta = torch.arange(20.0).reshape(5, 4) + 1.0
    save_file({"tensor": delta}, str(d / "production-layer20.safetensors"))
    save_file({"tensor": delta if second is None else second},
              str(d / "qualification-layer20.safetensors"))  # fmt: skip
    (d / "build.json").write_text(json.dumps({"seeds": {"package": 7, "random": {"20": 11}},
                                              "norms": {"20": 3.0}}))  # fmt: skip
    return d, delta


def test_a_cells_vector_is_its_builds_checked_difference_and_a_random_one_keeps_its_norm(tmp_path):
    import torch

    from residual_disposition.adl_lens import random_directions

    d, delta = build_dir(tmp_path)
    vectors, norm, seed = RS.cell_vectors(d, random=False)
    assert torch.equal(vectors, delta) and (norm, seed) == (3.0, 7)
    random, norm, seed = RS.cell_vectors(d, random=True)
    assert torch.equal(random, random_directions(delta, 11)) and (norm, seed) == (3.0, 7)
    assert torch.allclose(torch.linalg.vector_norm(random, dim=-1),
                          torch.linalg.vector_norm(delta, dim=-1))  # fmt: skip


def test_a_vector_off_its_method_check_is_refused(tmp_path):
    import torch

    d, _ = build_dir(tmp_path, second=torch.ones(5, 4))
    with pytest.raises(ValueError, match="checked difference"):
        RS.cell_vectors(d, random=False)


def test_a_build_not_its_record_is_refused(tmp_path, monkeypatch):
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    build = tmp_path / "build"
    (build / "builds" / "s1").mkdir(parents=True)
    files = {}
    for kind in ("production", "qualification"):
        name = f"builds/s1/{kind}-layer20.safetensors"
        (build / name).write_text(kind)
        files[name] = sha(build / name)
    (build / "result.json").write_text(json.dumps({"files": files}))
    digest = sha(build / "result.json")
    monkeypatch.setattr(RS, "built", lambda repo=None: {"sha256": {"build": digest}})
    cells = [{"run": "s1"}]
    assert RS.source_problems(build, cells) == []
    (build / "builds/s1/production-layer20.safetensors").write_text("changed")
    assert RS.source_problems(build, cells) == [
        "builds/s1/production-layer20.safetensors is not its recorded file"]
    monkeypatch.setattr(RS, "built", lambda repo=None: {"sha256": {"build": "other"}})
    assert RS.source_problems(build, cells) == ["the build's result is not its committed record"]


def recipe_rows(counted):
    return [{"reading": {"reader": "rung2", "behavior": "eli5", "role": "recipe", "decoder": d,
                         "stream": s},
             "count": {"counted": counted, "n": 80} if counted is not None else {"n": 80}}
            for d in ("state", "base") for s in (1, 2)]  # fmt: skip


def shared_run(tmp_path, a="1", h="0"):
    d = tmp_path / "follow_rung2" / RS.SHARED_QUALIFICATION
    d.mkdir(parents=True)
    rates = {"rates": {"A": a, "H": h, "recipe": "1"}}
    (d / "qualification.json").write_text(json.dumps({"floors": {"state": rates, "base": rates}}))
    return tmp_path


def test_a_and_h_are_the_rungs_and_the_recipe_control_this_modes(tmp_path):
    q = RS.qualification(recipe_rows(40), RULES, shared_run(tmp_path))
    for decoder in ("state", "base"):
        assert q[decoder]["rates"] == {"A": "1", "H": "0", "recipe": str(Fraction(1, 2))}
        assert q[decoder]["eligible"] is True and q[decoder]["recipe_floor_met"] is False


def test_a_failing_shared_a_makes_the_decoder_ineligible_here_too(tmp_path):
    q = RS.qualification(recipe_rows(80), RULES, shared_run(tmp_path, a="1/2"))
    assert q["state"]["eligible"] is False and q["state"]["recipe_floor_met"] is True


def test_a_pending_recipe_reading_is_pending_never_a_floor(tmp_path):
    q = RS.qualification(recipe_rows(None), RULES, shared_run(tmp_path))
    assert q["state"]["recipe_floor_met"] is None and q["state"]["eligible"] is True


class Asked:
    def __init__(self):
        self.calls = []

    def many(self, calls):
        self.calls += list(calls)
        return [f"graded {c.attempt}" for c in calls]


def test_a_recorded_batch_gets_the_researchers_labels_and_no_provider_request():
    from residual_disposition import adl
    from residual_disposition.adl_graders import GraderCall

    blocked = adl.coherence_request(["one", "two"])
    other = adl.coherence_request(["three"])
    key = hashlib.sha256(blocked[1].encode()).hexdigest()
    asked = Asked()
    grade = R2.ManualCoherence(asked, {key: ["INCOHERENT", "COHERENT"]})
    out = grade.many([GraderCall(*blocked, "a/try0"), GraderCall(*other, "b/try0")])
    assert out[1] == "graded b/try0" and [c.attempt for c in asked.calls] == ["b/try0"]
    assert adl.parse_coherence(out[0], 2) == ["INCOHERENT", "COHERENT"]
    with pytest.raises(ValueError, match="one per answer"):
        R2.ManualCoherence(asked, {key: ["COHERENT"]}).many([GraderCall(*blocked, "a/try0")])


def test_a_resumed_session_starts_from_the_stopped_ones_saved_steps(tmp_path):
    from residual_disposition.adl import StepStore

    old = tmp_path / "old"
    StepStore(old / "cells" / "L" / "steps").put("p0/search/prompt0/round0/samples", {"x": 1})
    (old / "cells" / "L" / "steering.json").parent.mkdir(parents=True, exist_ok=True)
    seen = []

    def steer(cell, store):
        seen.append(store.get("p0/search/prompt0/round0/samples"))
        return {"per_position": {}}

    R2.execute_steering(tmp_path / "new", {}, [{"cell": "L", "run": "sL", "random": False}],
                        steer, [0], resume_from=old)  # fmt: skip
    assert seen == [{"x": 1}]


def test_the_manual_import_orders_labels_by_answer_and_refuses_an_incomplete_export(tmp_path,
                                                                                     monkeypatch):
    from pathlib import Path

    import residual_disposition.paths as paths

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import follow_stack_rung2 as F2

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    (tmp_path / "manifests").mkdir()
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"cell": "L", "context": "c", "user_sha256": "u",
                                   "answer_number_by_item": {"b": 2, "a": 1}}))  # fmt: skip
    export = tmp_path / "export.jsonl"
    export.write_text('{"item_id": "a", "label": "COHERENT"}\n')
    with pytest.raises(SystemExit, match="one label per item"):
        F2.manual_import(export, mapping)
    export.write_text('{"item_id": "b", "label": "INCOHERENT"}\n'
                      '{"item_id": "a", "label": "COHERENT"}\n')
    F2.manual_import(export, mapping)
    record = json.loads((tmp_path / F2.MANUAL).read_text())
    assert record["batches"][0]["labels"] == ["COHERENT", "INCOHERENT"]
