"""Rung 2 on eli5 continued 200, on both decoders (§13.6, §13.16): its readings, the money they are
charged to, its evidence, each decoder's floors and test, the steering session's cells, and its one
format validation's supplement."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from residual_disposition import adl
from residual_disposition import continuation_rules as cr
from residual_disposition import follow_ladder as L
from residual_disposition import follow_regions as fr
from residual_disposition import follow_rung2 as R2
from residual_disposition import follow_validation as fv
from residual_disposition import lens_render as R

RULES = {"qualification": {"floors": {"active_min": "7/10", "sham_max": "1/10", "gap_min": "3/5",
                                      "recipe_ratio": "7/10"}},
         "confusion_min": "3/10", "specificity_min": "1/2"}  # fmt: skip


def test_each_decoder_reads_880_slots_under_its_own_salts_the_states_frozen():
    rows = R2.readings()
    inventory = json.loads((cr.REPO_ROOT / cr.follow_b3()["inventory"]["manifest"]).read_text())
    frozen = inventory["rung2"]
    state = [*frozen["shared"], *frozen["recipe_continued"], *frozen["analysis"]]
    both = [*rows["qualification"], *rows["analysis"]]
    assert sum(r["slots"] for r in both) == 2 * 880
    salts = [r["salt"] for part in rows.values() for r in part]
    assert len(salts) == len(set(salts))
    by_salt = {r["salt"]: r for r in both}
    for r in state:  # the state decoder's readings are B3's, salt and package
        assert by_salt[r["salt"]] == r
        twin = by_salt[r["salt"].replace("follow/rung2/", "follow/rung2/base/", 1)]
        assert twin["role"] == r["role"] and twin["stream"] == r["stream"]
        assert twin["decoder"] == ("base_random" if r["role"] == "L_random" else "base")
        if r["role"] != "L_random":
            assert twin["package"] == r["package"]


def test_the_random_cell_is_each_decoders_own_branch_of_ls_package():
    cells = {(r["role"], r["decoder"]): r for r in R2.readings()["analysis"]}
    run = cells[("L", "state")]["package"]["run"]
    assert cells[("L_random", "random")]["package"] == {"new": f"random branch of {run}"}
    assert cells[("L_random", "base_random")]["package"] == {"base_control": run}


def test_a_supplement_stream_holds_a_h_and_the_recipe_of_both_decoders_alike():
    rows = R2.readings()["validation"]
    for stream in fv.SUPPLEMENT_STREAMS:
        part = R2.stream_of(rows, stream)
        assert sorted((r["role"], r["decoder"]) for r in part) == sorted(
            (role, d) for role in ("A", "H", R2.RECIPE) for d in R2.DECODERS)
    assert sum(r["slots"] for r in R2.stream_of(rows, 3)) == 480


def test_each_reading_is_charged_to_one_item_and_the_rungs_own_keep_their_work():
    rows = R2.readings()
    of = {(r["purpose"], r["role"], r["decoder"]): R2.item_of(r)
          for part in rows.values() for r in part}  # fmt: skip
    assert of[("qualification", "A", "state")] == of[("qualification", "H", "state")] == R2.SHARED
    for role, decoder in (("L", "state"), ("H_s", "state"), ("X", "state"), ("recipe", "state"),
                          ("L_random", "random")):  # fmt: skip
        assert of[("analysis", role, decoder)] == R2.RUNG
    for key, item in of.items():
        if key[2] in ("base", "base_random") or key[0] == "validation":
            assert item == R2.LEFT


def steering_of(prefix="x"):
    return {"per_position": {str(p): {"failed": None, "samples": [
        {"prompt": f"prompt {i}", "steered": f"{prefix}{p}/{i} steered",
         "unsteered": f"{p}/{i} unsteered"} for i in range(L.CLOSED_PROMPTS)]}
        for p in R.POSITIONS}}  # fmt: skip


def test_the_base_decoders_evidence_is_its_own_rendering_with_the_same_samples(tmp_path):
    run = cr._b2_probe_package(tmp_path)

    def split(evidence):
        layers = json.loads(evidence[len("OVERVIEW:\n"):])["datasets"]["ds1"]["layers"]
        samples = {p: e.pop("steering") for p, e in layers["19"]["positions"].items()}
        return layers, samples

    state, state_samples = split(L.render_rung2(run, steering_of(), tmp_path)[0])
    base, base_samples = split(L.render_rung2(run, steering_of(), tmp_path, "base")[0])
    assert state_samples == base_samples  # both decoders read one session's samples
    assert state != base
    plain = json.loads(R.render(run, "base", tmp_path)[0][len("OVERVIEW:\n"):])
    assert base == plain["datasets"]["ds1"]["layers"]


def test_two_cells_of_one_recipe_read_one_sessions_samples(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(R2.L, "render_rung2", lambda d, s, root, decoder: (seen.append(s), "t"))
    samples = {"recipe": {"per_position": "recipe's"}, "L": {"per_position": "L's"}}
    for role in (R2.RECIPE, "recipe"):
        R2.evidence({"role": role, "decoder": "base"}, tmp_path, samples)
    assert seen == [samples["recipe"], samples["recipe"]]


def count(reading, counted):
    return {"reading": reading, "count": {"counted": counted, "n": 80}}


def qualified(a=150, h=0, recipe=150, *, decoder_base=None, pending=None):
    rows = []
    for decoder in R2.DECODERS:
        here = dict(a=a, h=h, recipe=recipe)
        if decoder == "base" and decoder_base:
            here.update(decoder_base)
        for role, n in (("A", here["a"]), ("H", here["h"]), (R2.RECIPE, here["recipe"])):
            for stream, part in ((1, n // 2), (2, n - n // 2)):
                r = {"reader": R2.READER, "behavior": "eli5", "role": role, "decoder": decoder,
                     "stream": stream}  # fmt: skip
                c = count(r, part)
                if pending == (decoder, role):
                    c["count"] = {"due": 1}
                rows.append(c)
    return rows


def test_each_decoders_floors_stand_alone_and_the_recipe_floor_closes_nothing():
    q = R2.qualification(qualified(recipe=20), RULES)
    assert q["state"]["eligible"] is q["base"]["eligible"] is True  # §13.16: measured, not closing
    assert q["state"]["recipe_floor_met"] is False
    q = R2.qualification(qualified(decoder_base={"h": 40}), RULES)
    assert q["state"]["eligible"] is True and q["base"]["eligible"] is False
    q = R2.qualification(qualified(decoder_base={"a": 100}), RULES)
    assert q["base"]["eligible"] is False  # R(A) 5/8 under 7/10


def test_a_pending_qualification_is_pending_never_ineligible():
    q = R2.qualification(qualified(pending=("base", "A")), RULES)
    assert q["base"]["eligible"] is None and q["state"]["eligible"] is True


def slots(counted, *, margin=True, feature=True):
    out = []
    for i in range(80):
        yes = i < counted
        s = {"labels": {"strict": {"value": ("other", False)}, "lenient": {"value": yes}}}
        if yes:
            s["grades"] = {"category": {"value": frozenset({"eli5" if margin else "tldr"})},
                           "specificity": {"value": ("feature" if feature else "category_only",
                                                     True)}}  # fmt: skip
        out.append(s)
    return out


def analysed(l_count=(60, 60), random=(2, 2), **kw):
    rows = []
    for decoder, x, rnd in (("state", l_count[0], random[0]), ("base", l_count[1], random[1])):
        for role, n in (("L", x), ("H_s", 0), ("X", 0), ("recipe", 70)):
            rows.append({"reading": {"role": role, "decoder": decoder},
                         "count": {"counted": n, "n": 80},
                         "slots": slots(n, **kw) if role == "L" else []})  # fmt: skip
        rows.append({"reading": {"role": "L_random", "decoder": R2.RANDOM_CELLS[decoder]},
                     "count": {"counted": rnd, "n": 80}, "slots": []})  # fmt: skip
    return rows


FLOORS = {d: {"eligible": True, "recipe_floor_met": True} for d in R2.DECODERS}


@pytest.fixture
def judged(monkeypatch):
    """The category and specificity margins from the fake grades' categories."""
    def margins(rows, behavior, definition, rules):
        chosen = [s["grades"] for s in rows if "grades" in s]
        right = sum("eli5" in g["category"]["value"] for g in chosen)
        passed = bool(chosen) and Fraction(right, len(chosen)) >= Fraction(rules["confusion_min"])
        return {"counted": len(chosen), "passed": passed, "specific": False}

    monkeypatch.setattr(R2.D, "margins", margins)


def decide(rows, floors=FLOORS, accepted=True, validated=True):
    return R2.decision(rows, floors, accepted, validated, fr.load(), RULES, {"eli5": None})


@pytest.mark.usefixtures("judged")
def test_each_decoder_has_its_own_test_at_one_sixtieth():
    # the certified threshold at 1/60 with H_s = X = 0 is 34: one decoder above it, one below
    out = decide(analysed(l_count=(34, 33)))["decoders"]
    assert out["state"]["status"] == "identified"
    assert out["base"]["status"] == "not identified"
    assert out["state"]["counts"] == {"L": 34, "H_s": 0, "X": 0}


@pytest.mark.usefixtures("judged")
def test_the_random_cell_gate_is_at_most_eight_per_decoder():
    out = decide(analysed(random=(8, 9)))["decoders"]
    assert out["state"]["status"] == "identified" and out["state"]["random_passed"] is True
    assert out["base"]["status"] == "not identified" and out["base"]["random_passed"] is False


@pytest.mark.usefixtures("judged")
def test_zero_counted_fails_category_and_a_wrong_category_closes_the_test():
    assert decide(analysed(l_count=(0, 0)))["decoders"]["state"]["status"] == "not identified"
    out = decide(analysed(margin=False))["decoders"]["state"]
    assert out["contrast_at_level"] is True and out["status"] == "not identified"


@pytest.mark.usefixtures("judged")
def test_an_ineligible_decoder_is_reported_with_its_gate_and_the_other_decided():
    floors = {"state": {"eligible": True, "recipe_floor_met": False},
              "base": {"eligible": False, "recipe_floor_met": True}}  # fmt: skip
    rows = [r for r in analysed() if r["reading"]["decoder"] in ("state", "random")]
    out = decide(rows, floors)
    assert out["decoders"]["base"] == {"recipe_floor_met": True, "format": True,
                                       "status": "ineligible", "gate": "qualification"}  # fmt: skip
    assert out["decoders"]["state"]["status"] == "identified"
    assert out["decoders"]["state"]["recipe_floor_met"] is False  # reported, closing nothing
    assert out["pending"] is False


@pytest.mark.usefixtures("judged")
def test_a_failed_format_makes_both_decoders_ineligible():
    out = decide(analysed(), accepted=False)["decoders"]
    assert {d: (e["status"], e["gate"]) for d, e in out.items()} == dict.fromkeys(
        R2.DECODERS, ("ineligible", "format validation"))


@pytest.mark.usefixtures("judged")
def test_a_pending_cell_or_grade_is_pending_never_a_negative(monkeypatch):
    rows = analysed()
    rows[0]["count"] = {"due": 1}  # the state decoder's L
    out = decide(rows)
    assert out["decoders"]["state"]["status"] == "pending" and out["pending"] is True
    assert out["decoders"]["base"]["status"] == "identified"
    rows = [r for r in analysed() if r["reading"]["role"] != "L_random"]
    assert decide(rows)["pending"] is True
    monkeypatch.setattr(R2.D, "margins", lambda *a: {"passed": None, "specific": None})
    assert decide(analysed())["decoders"]["base"]["status"] == "pending"


def test_the_session_steers_each_package_once_and_l_before_its_random_cell():
    cells = R2.steering_cells()
    names = [c["cell"] for c in cells]
    assert sorted(names) == sorted(["A", "H", "recipe", "L", "H_s", "X", "L_random"])
    assert names[-1] == "L_random" and names.index("L") < names.index("L_random")
    by = {c["cell"]: c for c in cells}
    assert by["L_random"]["run"] == by["L"]["run"] and by["L_random"]["vector"] == by["L"]["vector"]
    assert by["L_random"]["random"] is True and not by["L"]["random"]
    assert len({by[n]["run"] for n in names if n != "L_random"}) == 6


def test_a_vector_source_not_its_pin_is_refused(tmp_path):
    run, base = tmp_path / "run", tmp_path / "base"
    run.mkdir(), base.mkdir()
    (run / "a.json").write_text("a"), (base / "base_means.safetensors").write_text("b")
    import hashlib

    def sha(text):
        return hashlib.sha256(text.encode()).hexdigest()

    cell = {"cell": "L", "vector": {"files": {"a.json": sha("a")},
                                    "base_means": {"sha256": sha("b")}}}  # fmt: skip
    assert R2.source_problems(cell, run, base) == []
    (run / "a.json").write_text("changed")
    assert R2.source_problems(cell, run, base) == ["L: a.json is not its pinned file"]


def run_session(out, cells):
    def steer(cell, store):
        per = {}
        for p in (0, 1):
            unsteered = store.once(f"p{p}/unsteered", lambda p=p: [f"{cell['cell']} {p} unsteered"])
            steered = store.once(f"p{p}/steered", lambda p=p: [f"{cell['cell']} {p} steered"])
            per[str(p)] = {"failed": None, "samples": [{"prompt": "q", "steered": steered[0],
                                                        "unsteered": unsteered[0]}]}  # fmt: skip
        return {"per_position": per}

    return R2.execute_steering(out, {"m": 1}, cells, steer, (0, 1))


def test_the_random_cell_keeps_ls_unsteered_samples_and_only_its_steered_differ(tmp_path):
    out = tmp_path / "out"
    cells = [{"cell": "L", "run": "r", "random": False}, {"cell": "L_random", "run": "r",
                                                          "random": True}]  # fmt: skip
    result = run_session(out, cells)
    read = {c["cell"]: json.loads((out / "cells" / c["cell"] / "steering.json").read_text())
            for c in cells}  # fmt: skip
    for p in ("0", "1"):
        a, b = read["L"]["per_position"][p], read["L_random"]["per_position"][p]
        assert a["samples"][0]["unsteered"] == b["samples"][0]["unsteered"]
        assert a["samples"][0]["steered"] != b["samples"][0]["steered"]
    assert result["status"] == "complete" and sorted(result["cells"]) == ["L", "L_random"]
    assert "steering.json" in result["cells"]["L"]["files"]
    assert json.loads((out / "progress.json").read_text())["done"] == ["L", "L_random"]


def test_the_steps_of_a_position_are_the_packages_own(tmp_path, monkeypatch):
    """§11.1's procedure once: the search on its seed's streams, then the final samples; a position
    with no coherent strength keeps its unsteered samples only."""
    import torch

    calls = []

    def generate(model, tok, prompts, vector, *, layer, strengths, seed, chat_kwargs, **kw):
        calls.append((vector is None, layer, seed, kw["temperature"]))
        return ["text"] * len(prompts)

    monkeypatch.setattr(adl, "generate_steered", generate)
    monkeypatch.setattr(adl, "grade_coherence_many",
                        lambda kept, grade, contexts: [["INCOHERENT"] * len(k) for k in kept])
    store = adl.StepStore(tmp_path)
    out = R2.steer_cell(None, None, torch.ones(5, 4), 2.0, seed=7, chat_kwargs={},
                        grade_coherent=None, store=store)  # fmt: skip
    assert sorted(out["per_position"]) == ["0", "1", "2", "3", "4"]
    entry = out["per_position"]["0"]
    assert entry["failed"] == adl.NO_COHERENT_STRENGTH and entry["strength"] is None
    assert len(entry["samples"]) == 20 and all(s["steered"] is None for s in entry["samples"])
    assert all(c[1] == int(L.STEERING_LAYER) for c in calls)
    assert {c[3] for c in calls} == {1.2, 1.1}  # the search's and the final samples' temperatures
    assert store.get("p0/unsteered") == ["text"] * 20 and store.get("p0/steered") is None


def stratum(reading):
    return fv.labels(reading, another_behavior=False, category_only=False)


def test_the_format_counts_decoders_and_roles_at_rung_two():
    r = {"reader": "rung2", "purpose": "qualification", "behavior": "eli5", "role": R2.RECIPE,
         "decoder": "base", "component": None}  # fmt: skip
    assert set(stratum(r)) >= {"decoder:base", "role:recipe", "behavior:eli5"}
    spec = {"positive": {"role": 30, "decoder": 150}, "negative": {"role": 30, "decoder": 150}}
    need = fv.minimums_of(spec, decoders=R2.DECODERS, roles=("A", "H", "recipe"))
    assert need["negative"]["decoder:state"] == need["negative"]["decoder:base"] == 150
    assert need["positive"]["role:recipe"] == 30


def frame_items(n_negative):
    items = [f"p{i}" for i in range(400)] + [f"n{i}" for i in range(n_negative)]
    reference = {i: fv.POSITIVE if i.startswith("p") else fv.NEGATIVE for i in items}
    return items, reference, {i: () for i in items}


def test_a_frame_short_of_a_class_asks_for_the_next_stream_and_locks_once_filled():
    totals = {fv.POSITIVE: 400, fv.NEGATIVE: 400}
    items, reference, strata = frame_items(320)
    assert fv.step(items, ["next stream"], False, reference, strata, totals,
                   {fv.POSITIVE: {}, fv.NEGATIVE: {}})["step"] == "generate_supplement"  # fmt: skip
    assert fv.step(items, None, False, reference, strata, totals,
                   {fv.POSITIVE: {}, fv.NEGATIVE: {}})["step"] == "incomplete"  # fmt: skip
    items, reference, strata = frame_items(400)
    assert fv.step(items, ["next stream"], False, reference, strata, totals,
                   {fv.POSITIVE: {}, fv.NEGATIVE: {}})["step"] == "locked"  # fmt: skip


def test_rung_twos_format_takes_one_stream_at_a_time_after_its_frame(tmp_path, monkeypatch):
    """The frame is both decoders' qualification; a class short with the frame labeled asks for the
    next stream; a recorded stream's items follow the frame's in their own seeded order, and the
    judged attempts are read from the run that holds them."""
    monkeypatch.syspath_prepend(str(cr.REPO_ROOT / "scripts"))
    import argparse

    import follow_format as S

    monkeypatch.setattr(S, "data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(S, "runs_root", lambda: tmp_path / "runs")

    def fake_frame(rows, histories, cache):
        return {f"{r['salt']}#slot={i}": {"reading": dict(r), "slot": i, "hypothesis": "h"}
                for r in rows for i in range(int(r["slots"]))}  # fmt: skip

    monkeypatch.setattr(S.F, "frame", fake_frame)
    for run, slot in (("q", "a"), ("s3", "b")):
        d = tmp_path / "runs" / "follow_rung2" / run
        d.mkdir(parents=True)
        (d / "inputs.json").write_text("{}")
        (d / "histories.json").write_text(json.dumps({slot: {"interpreter": []}}))
    w = S.Work("rung2")
    w.dir.mkdir(parents=True)
    record, items = w.frame("q")
    assert record["items"] == 960 and {i["reading"]["decoder"] for i in items.values()} == {
        "state", "base"}
    assert record["order"] == fv.seeded_order(list(items), 2026103003)

    negative = {i for i, v in items.items() if v["reading"]["role"] == "H"}
    mapping = {i: i for i in items}
    final = {i: {"negative": i in negative} for i in items}
    monkeypatch.setattr(w, "mapping", lambda: mapping)
    monkeypatch.setattr(w, "final", lambda: final)
    monkeypatch.setattr(w, "open_work", lambda: None)
    monkeypatch.setattr(S.F, "reference_class", lambda v: fv.NEGATIVE if v["negative"]
                        else fv.POSITIVE)  # fmt: skip
    monkeypatch.setattr(S.F, "strata", lambda r, v, letters: fv.labels(
        r, another_behavior=False, category_only=False))  # fmt: skip
    assert w.state(items, record["order"])["step"] == "generate_supplement"  # 320 of 400 negative

    S.cmd_supplement(w, argparse.Namespace(run="s3"))
    assert w.next_stream() == 4
    record, more = w.frame()
    assert record["items"] == 1440
    new = [i for i in more if i not in items]
    assert record["order"] == [*fv.seeded_order(list(items), 2026103003),
                               *fv.seeded_order(new, 2026103013)]  # fmt: skip
    assert w.histories(record) == {"a": {"interpreter": []}, "b": {"interpreter": []}}
    assert {more[i]["reading"]["stream"] for i in new} == {3}
    assert {more[i]["reading"]["decoder"] for i in new} == {"state", "base"}
    with pytest.raises(SystemExit, match="exists|not generate_supplement"):
        S.cmd_supplement(w, argparse.Namespace(run="s3"))

    negative |= {i for i in new if more[i]["reading"]["role"] == "H"}
    mapping.update({i: i for i in new})
    final.update({i: {"negative": i in negative} for i in new})
    step = w.state(more, record["order"])
    assert step["step"] == "locked"  # 480 negative: the rule locks on the labeled prefix
    picked = {more[i]["reading"]["decoder"] for i in step["chosen"][fv.NEGATIVE]}
    assert picked == {"state", "base"}


def test_the_minimums_are_checked_within_a_family_since_an_item_holds_one_of_each():
    items = [f"i{k}" for k in range(800)]
    reference = {i: fv.POSITIVE if k < 400 else fv.NEGATIVE for k, i in enumerate(items)}
    strata = {i: (f"decoder:{'state' if k % 2 else 'base'}", f"role:{'AHR'[k % 3]}")
              for k, i in enumerate(items)}  # fmt: skip
    need = {"decoder:state": 150, "decoder:base": 150, "role:A": 30, "role:H": 30, "role:R": 30,
            "difficulty:none": 100}  # fmt: skip
    totals = {fv.POSITIVE: 400, fv.NEGATIVE: 400}
    assert fv.select(items, reference, strata, totals, {fv.POSITIVE: {}, fv.NEGATIVE: need},
                     final=True)["complete"]  # fmt: skip
    with pytest.raises(ValueError, match="exceed"):
        fv.select(items, reference, strata, totals,
                  {fv.POSITIVE: {"decoder:state": 300, "decoder:base": 150}, fv.NEGATIVE: {}})


def test_a_vector_is_its_packages_stored_difference_and_a_random_one_keeps_its_norm(tmp_path):
    import torch
    from safetensors.torch import save_file

    run, base = tmp_path / "run", tmp_path / "base"
    (run / "steps").mkdir(parents=True), (run / "reproduction").mkdir(), base.mkdir()
    state, shared = torch.arange(20.0).reshape(5, 4) + 1.0, torch.ones(5, 4)
    save_file({"mean_20": state}, str(run / "steps" / "extraction.safetensors"))
    save_file({"mean_20": shared}, str(base / "base_means.safetensors"))
    save_file({"delta": state - shared}, str(run / "reproduction" / "delta_layer20.safetensors"))
    record = {"seed": 5, "layers": {"20": {"norm": 3.0}}}
    (run / "adl_lens_package.json").write_text(json.dumps(record))
    vectors, norm, seed = R2.cell_vectors(run, base, random=False)
    assert torch.equal(vectors, state - shared) and (norm, seed) == (3.0, 5)
    random, _, _ = R2.cell_vectors(run, base, random=True)
    assert not torch.equal(random, vectors)
    assert torch.allclose(torch.linalg.vector_norm(random, dim=-1),
                          torch.linalg.vector_norm(vectors, dim=-1))  # fmt: skip
    save_file({"delta": state}, str(run / "reproduction" / "delta_layer20.safetensors"))
    with pytest.raises(ValueError, match="stored difference"):
        R2.cell_vectors(run, base, random=False)
