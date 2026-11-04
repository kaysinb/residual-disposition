"""Steering at the partial threshold (§13.16, 2026-11-04): the readings and their salts, the
logit-lens check of δ̄ without a stored difference, the cells and their vectors, the pinned
sources, each line's decision and its frozen comparison with rung 1."""

from __future__ import annotations

import json
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
        rows = [r for part in FP.readings(line).values() for r in part]
        assert sum(r["slots"] for r in rows) == (560 if "shared" in spec else 880)
        assert {(r["role"], r["decoder"]) for r in rows if r["decoder"] != "base"} == {
            ("L_random", "base_random")}
        assert ("A" in {r["role"] for r in rows}) is ("shared" not in spec)
        assert all(spec["behavior"] in r["salt"] and spec["lineage"] in r["salt"] for r in rows)
        salts += [r["salt"] for r in rows]
    assert len(salts) == len(set(salts)) == 3 * 11 + 7
    for name in ("follow_rung2_inputs.json", "follow_stack_rung2_inputs.json"):
        assert not set(salts) & set(json.loads((REPO / "manifests" / name).read_text())["readings"])
    assert {FP.item_of(r) for line in FP.LINES for p in FP.readings(line).values() for r in p} == {
        "follow/rung3_shared"}


def lens(ids, probs):
    return {"token_ids": list(ids), "probs": list(probs)}


def test_the_logit_lens_check_holds_the_sets_and_the_probabilities_not_the_order():
    ids = list(range(100))
    probs = [0.5 / (i + 1) for i in ids]
    stored = lens(ids, probs)
    swapped = lens([1, 0, *ids[2:]], [probs[1], probs[0], *probs[2:]])
    assert FP.lens_problems(stored, swapped) == []  # the top 10's order is shown, never a gate
    near = lens(ids, [p + 9e-4 for p in probs])
    assert FP.lens_problems(stored, near) == []
    far = lens(ids, [probs[0] + 2e-3, *probs[1:]])
    assert FP.lens_problems(stored, far) == ["a probability differs by 2.00e-03"]
    tail = lens([*ids[:99], 500], [*probs[:99], probs[99]])
    assert FP.lens_problems(stored, tail) == ["top 100 differ", "a probability differs by 5.00e-03"]
    head = lens([500, *ids[1:]], probs)
    assert FP.lens_problems(stored, head)[:2] == ["top 10 differ", "top 100 differ"]


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


def test_a_random_cell_of_a_lens_run_is_the_base_random_branchs_direction(tmp_path):
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

    cell = {"package": {"run": "r"}, "random": True}
    vectors, norm, seed, how = FP.cell_vectors(cell, find_run)
    delta = means["mean_20"] - shared["mean_20"]
    assert torch.equal(vectors, random_directions(delta, derive_seed(7, "adl_lens", 20, "random")))
    assert (norm, seed) == (3.0, 7) and "logit lens" in how and "absent" in how
    plain = FP.cell_vectors({**cell, "random": False}, find_run)[0]
    assert torch.equal(plain, delta)


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
    got = S.lens_check(None, lambda name: run if name == "r1" else tmp_path)
    assert got["r1"]["problems"] == [
        "a probability differs by 5.00e-01", "top 10 differ", "top 100 differ"]
    assert [p for p, x in got["r1"]["positions"].items() if x["problems"]] == ["3"]
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
