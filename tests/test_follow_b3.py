"""Barrier B3 (§13.15): its configuration agrees with the records and the code that execute it
(the certificates' lines, the seal's envelopes, rungs 2 and 3, unfiltered PD and the inventory),
and each check refuses a changed entry; it is not executed while unfrozen."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_b3 as B


def b3():
    return yaml.safe_load(cr.FOLLOW_B3_PATH.read_text())["b3"]


def line(name):
    return lambda d: d["certificates"]["lines"][name]


def changed(change):
    data = copy.deepcopy(b3())
    change(data)
    return data


def test_the_configuration_agrees_with_the_records_and_the_code():
    assert B.problems(b3()) == []


@pytest.mark.parametrize("change, match", [
    (lambda d: line("example_link_200")(d).update(latent="s0"), "confirmed dose"),
    (lambda d: d["certificates"]["lines"]["pun_intended_200"].update(reference="r"), "approved"),
    (lambda d: d["certificates"]["lines"]["eli5_stack_200"].update(reference="r"), "pins"),
    (lambda d: d["certificates"]["lines"]["eli5_stack_200"].update(dose=400), "analysis dose"),
    (lambda d: d["certificates"]["lines"]["example_link_200"].update(item="follow/x"), "allocated"),
    (lambda d: d["certificates"]["lines"]["example_link_200"].update(mode="continued"), "stack"),
    (lambda d: d["certificates"]["kstar"].update(seeds=[6, 7, 9]), "k\\* calibration"),
    (lambda d: d["certificates"]["controls"].update(y_organism="s0"), "pilot's organism"),
    (lambda d: d["certificates"].update(engine="keep_awake"), "sleeping"),
    (lambda d: line("eli5_stack_200")(d).update(latent="s4b9dc0f2018682b9"), "selection"),
    (lambda d: line("eli5_stack_200")(d).update(sham_s="s0"), "selection"),
    (lambda d: line("eli5_stack_200")(d)["selection"].update(sha256="0" * 64), "pinned sha256"),
    (lambda d: line("example_link_200")(d).update(item="follow/cert/pun_intended"), "meter item"),
    (lambda d: line("example_link_200")(d)["data"].update(id="d0"), "k\\* manifest"),
    (lambda d: line("pun_intended_200")(d)["data"].update(examples="0" * 64), "source run"),
])  # fmt: skip
def test_each_certificate_entry_is_its_record_s(change, match):
    assert any(__import__("re").search(match, p) for p in B.problems(changed(change)))


@pytest.mark.parametrize("change, match", [
    (lambda d: d["seal"]["envelopes"].update(cadence_minutes=30.5), "cadence"),
    (lambda d: d["seal"]["envelopes"].update(payload="the record"), "nonce"),
    (lambda d: d["ladder"]["rung2"]["prompt"].update(sha256="0" * 64), "rung 2: the prompt"),
    (lambda d: d["ladder"]["rung3"]["prompts"]["agent"].update(sha256="0" * 64), "rung 3"),
    (lambda d: d["ladder"]["rung3"].update(queries=6), "queries"),
    (lambda d: d["ladder"]["rung3"].update(runs=40), "runs"),
    (lambda d: d["ladder"]["rung3"]["model"].update(reasoning_effort="high"), "model"),
    (lambda d: d["ladder"]["rung2"]["steering"]["search"].update(samples=5), "search"),
    (lambda d: d["ladder"]["rung2"]["steering"]["generation"].update(temperature=1.0), "generat"),
    (lambda d: d["ladder"]["probes"].update(rung2="0" * 64), "probes"),
    (lambda d: d["ladder"]["format_validation"]["rung2"]["minimums"]["positive"].update(
        difficulty={}), "difficult strata"),
    (lambda d: d["ladder"]["format_validation"]["rung3"].update(strata=["component", "role"]),
     "difficulty"),
    (lambda d: d["pd_unfiltered"].update(fraction=0.02), "fraction"),
    (lambda d: d["pd_unfiltered"]["stage3"].update(tldr=["stack", 200]), "Stage-3 lineages"),
])  # fmt: skip
def test_each_seal_ladder_and_pd_entry_is_the_code_s(change, match):
    assert any(match in p for p in B.problems(changed(change)))


def manifest(tmp_path, rows, reuse=None, build="b"):
    (tmp_path / "manifests").mkdir(exist_ok=True)
    b2 = cr.REPO_ROOT / "manifests" / "follow_b2_inventory.json"
    (tmp_path / "manifests" / "follow_b2_inventory.json").write_text(b2.read_text())
    data = {**rows, "reuse": reuse or {}, "sources": {"build": build}}
    path = tmp_path / "manifests" / "follow_b3_inventory.json"
    path.write_text(json.dumps(data))
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    counts = {
        k: {p: len(rows.get(k, {}).get(p, [])) for p in parts} for k, parts in B.PARTS.items()
    }
    return {"manifest": "manifests/follow_b3_inventory.json", "sha256": digest, "build": build,
            "counts": counts}  # fmt: skip


def test_the_inventory_manifest_is_checked(tmp_path):
    from residual_disposition import follow_inventory as fi

    rung = {part: [] for part in B.RUNG_PARTS}
    rung["shared"] = fi.rung_shared(2)
    pd = [{"reader": "pd_unfiltered", "salt": "follow/pd/analysis/x/stack200/L/stream1",
           "slots": 80, "if_confirmed": None, "behavior": "x", "role": "L"}]  # fmt: skip
    rows = {"rung2": rung, "pd_unfiltered": {"analysis": pd}}
    inv = manifest(tmp_path, rows, reuse={pd[0]["salt"]: {"reuse": False, "reason": "r"}})
    assert B.inventory_problems(inv, tmp_path) == []
    assert "sha256" in B.inventory_problems({**inv, "sha256": "0" * 64}, tmp_path)[0]
    wrong = {**inv, "counts": {**inv["counts"], "rung2": {**inv["counts"]["rung2"], "analysis": 1}}}
    assert "counts" in B.inventory_problems(wrong, tmp_path)[0]
    shared = json.loads((cr.REPO_ROOT / "manifests" / "follow_b2_inventory.json").read_text())
    b2_salt = shared["dstar"]["qualification"][0]["salt"]
    other = {"analysis": [{**pd[0], "salt": b2_salt}]}
    clash = manifest(tmp_path, {**rows, "pd_unfiltered": other},
                     reuse={b2_salt: {"reuse": False}})  # fmt: skip
    assert any("share a salt" in p for p in B.inventory_problems(clash, tmp_path))
    none = manifest(tmp_path, rows)
    assert any("reuse decision" in p for p in B.inventory_problems(none, tmp_path))


def test_b3_is_not_executed_while_unfrozen(monkeypatch):
    monkeypatch.setattr(cr, "FOLLOW_B3_FROZEN", None)
    with pytest.raises(ValueError, match="not frozen"):
        cr.follow_b3()
