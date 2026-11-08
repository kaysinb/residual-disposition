"""Steering at the partial threshold (§13.16, 2026-11-04): the readings and their salts, the
logit-lens check of δ̄ without a stored difference, the cells and their vectors, the pinned
sources, each line's decision and its frozen comparison with rung 1."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from residual_disposition import follow_partial as FP
from residual_disposition import follow_regions as fr

RULES = {"qualification": {"floors": {"active_min": "7/10", "sham_max": "1/10", "gap_min": "3/5",
                                      "recipe_ratio": "7/10"}},
         "confusion_min": "3/10", "specificity_min": "1/2"}  # fmt: skip
REPO = Path(__file__).parents[1]


def test_each_line_reads_its_planned_slots_under_salts_of_its_own():
    salts = []
    for line, spec in FP.LINES.items():
        rows = [r for phase, part in FP.readings(line).items() if phase != "ablation"
                for r in part]  # fmt: skip
        assert sum(r["slots"] for r in rows) == (560 if "shared" in spec else 880)
        assert {(r["role"], r["decoder"]) for r in rows if r["decoder"] != "base"} == {
            ("L_random", "base_random")}
        assert ("A" in {r["role"] for r in rows}) is ("shared" not in spec)
        assert all(spec["behavior"] in r["salt"] and spec["lineage"] in r["salt"] for r in rows)
        salts += [r["salt"] for r in rows]
    assert len(salts) == len(set(salts)) == 3 * 11 + 7
    for name in ("follow_rung2_inputs.json", "follow_stack_rung2_inputs.json"):
        assert not set(salts) & set(json.loads((REPO / "manifests" / name).read_text())["readings"])
    assert {FP.item_of(r) for line in FP.LINES for k, p in FP.readings(line).items()
            if k != "ablation" for r in p} == {"follow/rung3_shared"}  # fmt: skip


def test_the_ablation_reads_l_h_s_and_x_once_under_salts_of_its_own():
    from residual_disposition import follow_qualification as Q

    rung2 = {r["salt"] for line in FP.LINES for k, p in FP.readings(line).items()
             if k != "ablation" for r in p}  # fmt: skip
    rows = [r for line in FP.LINES for r in FP.readings(line)["ablation"]]
    assert len(rows) == 12 and sum(r["slots"] for r in rows) == 960
    for r in rows:
        spec = FP.LINES[r["line"]]
        assert r["salt"] == (f"follow/rung2/base/ablation/{spec['behavior']}/{spec['lineage']}/"
                             f"{r['role']}/stream1")  # fmt: skip
        assert (r["reader"], r["decoder"], r["stream"], r["purpose"]) == (
            "rung2u", "base", 1, "ablation")
        assert r["package"] == spec["packages"][r["role"]]
        assert FP.item_of(r) == "follow/partial/ablation"
        assert FP.graded_reading(r) and Q.KINDS[r["reader"]] == "lens"
    assert [r["role"] for r in rows] == ["L", "H_s", "X"] * 4
    salts = {r["salt"] for r in rows}
    assert len(salts) == 12 and not salts & rung2
    for name in ("follow_partial_inputs.json", "follow_rung2_inputs.json",
                 "follow_stack_rung2_inputs.json"):  # fmt: skip
        assert not salts & set(json.loads((REPO / "manifests" / name).read_text())["readings"])
    # its rung-2 twin: the same package and decoder, another reader, salt, prompt and so keys
    el = next(r for r in rows if r["role"] == "L")
    twin = next(r for r in FP.readings(el["line"])["analysis"] if r["role"] == "L")
    assert {k for k in el if el[k] != twin[k]} == {"reader", "purpose", "salt"}
    keys = {a.key for tries in Q.attempts(el, "same evidence") for a in tries}
    assert not keys & {a.key for tries in Q.attempts(twin, "same evidence") for a in tries}
    assert Q.prompt_and_settings(el)[0].version == "adl-lens-multiscale-unsteer-v1"
    assert Q.prompt_and_settings(el)[1] == Q.prompt_and_settings(twin)[1]
    assert not FP.graded_reading({**el, "role": "recipe"})


def lens(ids, probs):
    return {"token_ids": list(ids), "probs": list(probs)}


def test_the_logit_lens_check_holds_the_top_10_set_and_the_probabilities_not_the_order():
    ids = list(range(100))
    probs = [0.5 / (i + 1) for i in ids]
    stored = lens(ids, probs)
    swapped = lens([1, 0, *ids[2:]], [probs[1], probs[0], *probs[2:]])
    assert FP.lens_problems(stored, swapped, {}) == []  # the order is shown, never a gate
    assert FP.lens_diagnostics(stored, swapped, {})["top10_same_order"] is False
    near = lens(ids, [p + 9e-4 for p in probs])
    assert FP.lens_problems(stored, near, {}) == []
    far = lens(ids, [probs[0] + 2e-3, *probs[1:]])
    (problem,) = FP.lens_problems(stored, far, {})
    assert problem.startswith("1 probabilities outside 1e-3 (worst token 0")
    head = lens([500, *ids[1:]], [probs[0], *probs[1:]])
    assert FP.lens_problems(stored, head, {0: 1e-6})[0] == "top 10 differ"


def test_a_near_tie_at_the_hundredth_place_passes_and_an_absent_token_is_read_in_full():
    ids = list(range(100))
    probs = [0.5 / (i + 1) for i in ids]
    stored = lens(ids, probs)
    tie = lens([*ids[:99], 500], [*probs[:99], probs[99] + 1e-5])  # token 99 just under 500
    assert FP.lens_problems(stored, tie, {99: probs[99] - 1e-5}) == []
    d = FP.lens_diagnostics(stored, tie, {99: probs[99] - 1e-5})
    assert d["top100_overlap"] == 99 and d["outside"] == []
    with pytest.raises(ValueError, match="no recomputed probability"):
        FP.lens_problems(stored, tie, {})  # never taken as 0
    gone = lens([*ids[:99], 500], [*probs[:99], probs[99]])
    assert FP.lens_problems(stored, gone, {99: probs[99] - 2e-3})  # far from its stored value
    newcomer = lens([*ids[:99], 500], [*probs[:99], probs[99] + 2e-3])  # above the stored 100th
    assert FP.lens_problems(stored, newcomer, {99: probs[99]})


def test_each_line_steers_its_packages_once_and_l_before_its_random_cell(monkeypatch):
    monkeypatch.setattr(FP, "state_of", lambda p, find_run: "s-" + FP.key_of(p))
    for line, spec in FP.LINES.items():
        cells = FP.steering_cells(line, None)
        roles = [c["cell"] for c in cells]
        want = ["recipe", "L", "H_s", "X", "L_random"]
        assert roles == (want if "shared" in spec else ["A", "H", *want])
        el, rnd = cells[roles.index("L")], cells[-1]
        assert rnd["random"] and not any(c["random"] for c in cells[:-1])
        assert (rnd["package"], rnd["run"], rnd["state"]) == (el["package"], el["run"], el["state"])


def test_a_random_cell_of_a_lens_run_is_the_base_random_branchs_direction(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    from residual_disposition.adl_lens import random_directions
    from residual_disposition.seeding import derive_seed

    run, base = tmp_path / "run", tmp_path / FP.BASE_MEANS
    (run / "steps").mkdir(parents=True)
    base.mkdir()
    g = torch.Generator().manual_seed(0)
    means = {f"mean_{x}": torch.randn(5, 8, generator=g) for x in (20, 30)}
    shared = {f"mean_{x}": torch.randn(5, 8, generator=g) for x in (20, 30)}
    save_file(means, str(run / "steps" / "extraction.safetensors"))
    save_file(shared, str(base / "base_means.safetensors"))
    (run / "adl_lens_package.json").write_text(json.dumps(
        {"seed": 7, "state": "s", "layers": {"20": {"norm": 3.0}, "30": {"norm": 4.0}}}))

    def find_run(name):
        return run if name == "r" else base

    monkeypatch.setattr(FP, "NO_DIFFERENCE", frozenset({"r"}))
    cell = {"package": {"run": "r"}, "random": True}
    vectors, norm, seed, how = FP.cell_vectors(cell, find_run)
    delta = means["mean_20"] - shared["mean_20"]
    assert torch.equal(vectors, random_directions(delta, derive_seed(7, "adl_lens", 20, "random")))
    assert (norm, seed) == (3.0, 7) and "logit lens" in how and "absent" in how
    plain = FP.cell_vectors({**cell, "random": False}, find_run)[0]
    assert torch.equal(plain, delta)
    monkeypatch.setattr(FP, "NO_DIFFERENCE", frozenset())  # not an exception: its difference
    with pytest.raises(FileNotFoundError):  # is required, never rebuilt without it
        FP.cell_vectors(cell, find_run)
    monkeypatch.setattr(FP, "NO_DIFFERENCE", frozenset({"r"}))
    (run / "reproduction").mkdir()
    with pytest.raises(ValueError, match="frozen without a stored difference"):
        FP.cell_vectors(cell, find_run)


def test_the_exceptions_are_eight_named_packages_of_the_lines():
    assert len(FP.NO_DIFFERENCE) == 8 and FP.unchecked_runs() == sorted(FP.NO_DIFFERENCE)
    assert {r.split("-")[-2] for r in FP.NO_DIFFERENCE} == {
        "727c1b34", "fdef39c6", "d1237f7d", "e91c1538", "e142ea17", "32160c51", "95a0552b",
        "12ca1f55"}


def test_a_resume_path_without_a_stopped_line_is_refused(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "pod"))
    import follow_partial_steering as S

    with pytest.raises(SystemExit, match="no line's stopped session"):
        S.resumable(tmp_path, {}, {}, {})
    line = next(iter(FP.LINES))
    (tmp_path / line).mkdir()
    saved = {"config": {"c": 1}, "cells": [], "states": {}, "worker": FP.WORKER}
    (tmp_path / line / "manifest.json").write_text(json.dumps(saved))
    plan = ({"c": 1}, {line: []}, {line: {}})
    assert S.resumable(tmp_path, *plan) == {line: tmp_path / line}
    with pytest.raises(SystemExit, match="config"):
        S.resumable(tmp_path, {"c": 2}, *plan[1:])


def test_a_vector_source_not_its_pin_or_unpinned_is_refused(monkeypatch):
    monkeypatch.setattr(FP, "sources", lambda find_run: {"a": "1", "b": "2"})
    assert FP.source_problems({"a": "1", "b": "2"}, None) == []
    assert FP.source_problems({"a": "1", "b": "3"}, None) == ["b: not its pinned file"]
    assert FP.source_problems({"a": "1"}, None) == ["b: not pinned"]


def test_the_pods_lens_gate_needs_every_unchecked_run_and_no_problem(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "pod"))
    import follow_partial_steering as S

    monkeypatch.setattr(FP, "unchecked_runs", lambda: ["r1", "r2"])

    def build(check):
        (tmp_path / "manifest.json").write_text(json.dumps({"lens_check": check}))
        return S.lens_problems(tmp_path)

    assert build({"r1": {"problems": []}, "r2": {"problems": []}}) == []
    assert build({"r1": {"problems": []}}) == ["r2: its logit lens is not checked"]
    assert build({"r1": {"problems": []}, "r2": {"problems": ["top 10 differ"]}}) == [
        "r2: ['top 10 differ']"]


def test_the_build_sessions_lens_check_reads_each_position_against_state_1(tmp_path, monkeypatch):
    import torch

    monkeypatch.syspath_prepend(str(REPO / "scripts" / "pod"))
    import follow_partial_build as S

    run = tmp_path / "r1"
    run.mkdir()
    ids = list(range(100))
    probs = [0.5 / (i + 1) for i in ids]
    per = {str(p): {"logit_lens": lens(ids, probs)} for p in range(5)}
    packages = {k: {"per_position": per} for k in ("state/1", "state/2", "base/1")}
    (run / "adl_lens_package.json").write_text(json.dumps(
        {"layers": {"20": {"packages": packages}}}))
    monkeypatch.setattr(FP, "unchecked_runs", lambda: ["r1"])
    monkeypatch.setattr(S.B, "pinned_sources",
                        lambda d, b, layers: {20: {"delta": torch.arange(5.0)}})  # fmt: skip

    def logit_lens(model, latent):  # position 3's δ̄ reads another top 10
        return ([500, *ids[1:]] if int(latent) == 3 else ids), probs

    monkeypatch.setattr(S.adl, "logit_lens", logit_lens)
    monkeypatch.setattr(S, "full_probs", lambda model, latent, tokens: dict.fromkeys(tokens, 0.0))
    got = S.lens_check(None, lambda name: run if name == "r1" else tmp_path)
    assert "top 10 differ" in got["r1"]["problems"]
    assert [p for p, x in got["r1"]["positions"].items() if x["problems"]] == ["3"]
    assert got["r1"]["positions"]["3"]["diagnostics"]["top10_only_recomputed"] == [500]
    assert got["r1"]["positions"]["0"]["stored_agree"] == {"state/2": True, "base/1": True}


def slots(counted, kinds):
    out = []
    for i in range(80):
        yes = i < counted
        s = {"labels": {"strict": {"value": ("other", False)}, "lenient": {"value": yes}}}
        if yes:
            s["grades"] = {"category": {"value": frozenset({"BEHAVIOR"})},
                           "specificity": {"value": (kinds[i], True)}}  # fmt: skip
        out.append(s)
    return out


def analysed(line, l_kinds, *, h_s=(), random=(), pending=None):
    behavior = FP.LINES[line]["behavior"]

    def row(role, decoder, kinds):
        kinds = [k for k in kinds]
        s = slots(len(kinds), kinds)
        for x in s:
            if "grades" in x:
                x["grades"]["category"]["value"] = frozenset({behavior})
        count = {"counted": len(kinds), "n": 80} if role != pending else {"due": 1}
        return {"reading": {"line": line, "role": role, "decoder": decoder,
                            "purpose": "analysis"}, "count": count, "slots": s}  # fmt: skip

    return [row("L", "base", l_kinds), row("H_s", "base", h_s), row("X", "base", ()),
            row("recipe", "base", ["feature"] * 70), row("L_random", "base_random", random)]


def definitions():
    return {b: SimpleNamespace(alternatives=["one", "two"]) for b in FP.BEHAVIORS}


def test_each_line_is_decided_alone_with_its_frozen_comparison_with_rung_1(monkeypatch):
    one = {"counts": {"L": 80, "H_s": 0, "X": 0},
           "margins": {"counted": 80, "specific": False,
                       "specificity": {"feature": "31/80", "best_alternative": "alternative:0",
                                       "best_alternative_share": "13/80",
                                       "advantage": "9/40"}}}  # fmt: skip
    monkeypatch.setattr(FP, "rung1", lambda line, runs=None: one)
    rows = [*analysed("example_link_200", ["feature"] * 30 + ["alternative:0"] * 10,
                      h_s=["feature", "alternative:1"], random=["feature"]),
            *analysed("pun_intended_200", ["alternative:1"] * 40),
            *analysed("countdown_50", []),
            *analysed("eli5_stack_200", ["feature"] * 40, pending="H_s")]  # fmt: skip
    floors = dict.fromkeys(FP.LINES, {"eligible": True, "recipe_floor_met": True})
    out = FP.decision(rows, floors, True, True, fr.load(), RULES, definitions())
    lines = out["lines"]
    assert lines["example_link_200"]["status"] == "identified"
    assert lines["pun_intended_200"]["status"] == "identified"  # the category, not the feature
    assert lines["countdown_50"]["status"] == "not identified"
    assert lines["eli5_stack_200"]["status"] == "pending" and out["pending"] is True
    c = lines["example_link_200"]["comparison"]
    assert c["rung1"]["L"]["feature_slots"] == 31 and c["rung1"]["L"]["counted"] == 80
    el = c["rung2"]["L"]
    assert (el["counted"], el["feature"], el["best_alternative_share"], el["feature_slots"]) == (
        40, "3/4", "1/4", 30)
    assert el["advantage"] == "1/2" and el["specific"] is True
    assert c["rung2"]["H_s"]["feature_slots"] == 1 and c["rung2"]["L_random"]["feature_slots"] == 1
    assert c["rung2"]["X"]["counted"] == 0 and c["rung2"]["X"]["feature_slots"] == 0
    pun = lines["pun_intended_200"]
    assert pun["comparison"]["rung2"]["L"]["feature_slots"] == 0
    assert pun["specificity_satisfied"] is False  # shown whatever the gates give
    assert lines["eli5_stack_200"]["comparison"]["rung2"]["H_s"] is None  # its slots not all read
    assert out["comparison_pending"] is True
    assert lines["eli5_stack_200"]["comparison"]["pending"] is True
    assert lines["example_link_200"]["comparison"]["pending"] is False


def test_an_ungraded_feature_leaves_the_comparison_pending_not_the_decision(monkeypatch):
    monkeypatch.setattr(FP, "rung1", lambda line, runs=None: {"counts": {}, "margins": None})
    rows = [r for line in FP.LINES for r in analysed(line, ["feature"] * 40, h_s=["feature"])]
    for r in rows:
        if r["reading"]["role"] == "H_s" and r["reading"]["line"] == "countdown_50":
            del r["slots"][0]["grades"]["specificity"]  # its grade still open
    floors = dict.fromkeys(FP.LINES, {"eligible": True, "recipe_floor_met": True})
    out = FP.decision(rows, floors, True, True, fr.load(), RULES, definitions())
    assert out["pending"] is False and out["comparison_pending"] is True
    assert out["lines"]["countdown_50"]["status"] == "identified"
    assert out["lines"]["countdown_50"]["comparison"]["rung2"]["H_s"]["feature"] is None


def test_eli5_stack_200_takes_eli5s_rung_2_a_and_h_and_reads_its_own_recipe(tmp_path):
    d = tmp_path / "follow_rung2" / FP.SHARED
    d.mkdir(parents=True)
    (d / "qualification.json").write_text(json.dumps(
        {"floors": {"base": {"rates": {"A": "3/4", "H": "0"}}}}))

    def recipe(n1, n2, behavior="eli5", roles=("recipe",)):
        return [{"reading": {"reader": "rung2", "behavior": behavior, "role": role,
                             "decoder": "base", "stream": s}, "count": {"counted": n, "n": 80}}
                for role in roles for s, n in ((1, n1), (2, n2))]  # fmt: skip

    q = FP.qualification("eli5_stack_200", recipe(60, 60), RULES, runs=tmp_path)["base"]
    assert q["rates"] == {"A": "3/4", "H": "0", "recipe": "3/4"}
    assert q["eligible"] is True and q["recipe_floor_met"] is True
    rows = recipe(70, 70, "countdown", ("A", "recipe")) + recipe(0, 0, "countdown", ("H",))
    q = FP.qualification("countdown_50", rows, RULES)["base"]
    assert q["rates"]["A"] == str(Fraction(7, 8)) and q["eligible"] is True


def test_a_blocked_batchs_labels_carry_their_line_and_cell(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import follow_partial as S

    monkeypatch.setattr(S, "REPO_ROOT", tmp_path)
    (tmp_path / "manifests").mkdir()
    export, mapping = tmp_path / "export.jsonl", tmp_path / "mapping.json"
    export.write_text('{"item_id": "b", "label": "INCOHERENT"}\n'
                      '{"item_id": "a", "label": "COHERENT"}\n')  # fmt: skip
    m = {"answer_number_by_item": {"a": 2, "b": 1}, "cell": "L", "context": "p0",
         "user_sha256": "u1", "line": "elsewhere"}  # fmt: skip
    mapping.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="not a line"):
        S.manual_import(export, mapping)
    mapping.write_text(json.dumps({**m, "line": "eli5_stack_200"}))
    S.manual_import(export, mapping)
    batch = json.loads((tmp_path / FP.MANUAL).read_text())["batches"][0]
    assert (batch["line"], batch["cell"], batch["labels"]) == (
        "eli5_stack_200", "L", ["INCOHERENT", "COHERENT"])
    with pytest.raises(SystemExit, match="recorded already"):
        S.manual_import(export, mapping)


def ablated(line, l_kinds, *, h_s=(), pending=None):
    """A line's ablation rows: L, H_s and X, as ``analysed``'s, under the ablation's purpose."""
    rows = [r for r in analysed(line, l_kinds, h_s=h_s, pending=pending)
            if r["reading"]["role"] in FP.ABLATION_ROLES]  # fmt: skip
    for r in rows:
        r["reading"] = {**r["reading"], "purpose": "ablation", "reader": "rung2u"}
    return rows


def spec_of(counted, feature, slots):
    return {"counted": counted, "feature": feature, "feature_slots": slots,
            "best_alternative": None, "best_alternative_share": "0", "advantage": feature,
            "specific": None}  # fmt: skip


def rung2_record():
    """Rung 2's record in its shape: each line's counts and its comparison's rung-2 readings."""
    zero = spec_of(0, "0", 0)
    return {"lines": {line: {"counts": {"L": 80, "H_s": 0, "X": 0},
                             "comparison": {"rung2": {"L": spec_of(80, "1/2", 40), "H_s": zero,
                                                      "X": zero, "L_random": zero}}}
                      for line in FP.LINES}}  # fmt: skip


def test_the_ablation_stands_beside_rung_2_and_rung_1_and_decides_nothing(monkeypatch):
    one = {"counts": {"L": 80, "H_s": 0, "X": 0},
           "margins": {"counted": 80, "specific": False,
                       "specificity": {"feature": "31/80", "best_alternative": "alternative:0",
                                       "best_alternative_share": "13/80",
                                       "advantage": "9/40"}}}  # fmt: skip
    monkeypatch.setattr(FP, "rung1", lambda line, runs=None: one)
    line = "example_link_200"
    rows = ablated(line, ["feature"] * 10 + ["alternative:0"] * 20, h_s=["alternative:1"])
    noise = analysed(line, ["feature"] * 80)  # rung 2's own rows are not the ablation's
    out = FP.ablation(line, [*noise, *rows], RULES, definitions(), rung2_record())
    assert out["ablation"]["counts"] == {"L": 30, "H_s": 1, "X": 0}
    el = out["ablation"]["L"]
    assert (el["counted"], el["feature"], el["feature_slots"]) == (30, "1/3", 10)
    assert (el["best_alternative"], el["best_alternative_share"]) == ("alternative:0", "2/3")
    assert out["ablation"]["H_s"]["feature_slots"] == 0 and out["ablation"]["X"]["counted"] == 0
    assert out["rung2"]["counts"] == {"L": 80, "H_s": 0, "X": 0}
    assert out["rung2"]["L"]["feature_slots"] == 40 and "L_random" not in out["rung2"]
    assert out["rung1"]["L"]["feature_slots"] == 31 and out["rung1"]["counts"] == one["counts"]
    assert out["rung2_minus_ablation"] == {"counts": {"L": 50, "H_s": -1, "X": 0},
                                           "feature_slots": {"L": 30, "H_s": 0, "X": 0}}
    assert out["pending"] is False
    assert not {"status", "decisions", "level", "eligible", "identified"} & set(out)


def test_an_unread_or_ungraded_ablation_reading_leaves_its_line_pending(monkeypatch):
    monkeypatch.setattr(FP, "rung1", lambda line, runs=None: {"counts": {}, "margins": None})
    line = "countdown_50"
    out = FP.ablation(line, ablated(line, ["feature"] * 5, pending="H_s"), RULES, definitions(),
                      rung2_record())  # fmt: skip
    assert out["pending"] is True and out["ablation"]["counts"]["H_s"] is None
    assert out["ablation"]["H_s"] is None and out["rung2_minus_ablation"]["counts"]["H_s"] is None
    assert out["rung2_minus_ablation"]["counts"]["L"] == 75
    assert set(out["rung2_minus_ablation"]["feature_slots"].values()) == {None}
    rows = ablated(line, ["feature"] * 5)
    del rows[0]["slots"][0]["grades"]["specificity"]  # its grade still open
    out = FP.ablation(line, rows, RULES, definitions(), rung2_record())
    assert out["pending"] is True and out["ablation"]["L"]["feature"] is None
    monkeypatch.setattr(FP, "rung2_record", lambda runs=None: rung2_record())
    rows = [r for x in FP.LINES for r in ablated(x, ["feature"] * 5)]
    every = FP.ablation_all(rows, RULES, definitions())
    assert sorted(every["lines"]) == sorted(FP.LINES) and every["pending"] is False
    assert every["rung2"] == FP.RUNG2


def test_the_ablation_takes_rung_2_s_pins_and_send_under_names_of_its_own(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import follow_partial as S

    from residual_disposition import follow_readings as D

    for name in ("PINS", "HISTORY", "RUN", "SENDS", "plan", "R2", "LABEL", "ROOT", "main"):
        monkeypatch.setattr(S.F, name, getattr(S.F, name))  # each restored after the test
    monkeypatch.setattr(D, "graded_reading", D.graded_reading)
    seen = []
    monkeypatch.setattr(S.F, "main", lambda argv=None: seen.append(
        (argv, S.F.PINS, S.F.HISTORY, S.F.RUN, S.F.SENDS, S.F.plan, S.F.LABEL)) or 0)
    for argv in (["--pin"], ["--ablation", "--pin"], ["--send", "ablation"]):
        monkeypatch.setattr(sys, "argv", ["follow_partial.py", *argv])
        assert S.main() == 0
    # rung 2's own path unchanged: its pins, history, run names, phases and label
    assert seen[0] == (None, "manifests/follow_partial_inputs.json",
                       "manifests/stage3_judge_history_follow_partial.json", "follow-rung2",
                       ("qualification", "supplement", "analysis"), S.plan,
                       "follow/partial-rung2")  # fmt: skip
    (pin, *_), (send, *rest) = seen[1:]
    assert pin == ["--pin"] and send == ["--send", "ablation"]
    assert rest == ["manifests/follow_partial_ablation_inputs.json",
                    "manifests/stage3_judge_history_follow_partial_ablation.json",
                    "follow-partial", ("ablation",), S.ablation_plan,
                    "follow/partial-ablation"]  # fmt: skip
    assert D.graded_reading is FP.graded_reading
    assert S.RUNG2_PHASES == ("qualification", "analysis")


def test_rung_2_s_record_is_read_only_at_its_sha256(tmp_path):
    d = tmp_path / "follow_partial" / FP.RUNG2["run"]
    d.mkdir(parents=True)
    (d / "rung2.json").write_text(json.dumps(rung2_record()))
    with pytest.raises(ValueError, match="not its recorded file"):
        FP.rung2_record(tmp_path)
    assert FP.RUNG2["sha256"].startswith("23081f53")
