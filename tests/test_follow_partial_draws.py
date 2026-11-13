"""Ten random draws per deletion read by the primary interpreter (``follow_partial_draws``): the
readings and their salts, the steered and family draws (seeded, drawn outside the marker or the
family, as many as the target deletion's), the plan's checks, the summary's rank and wording, and
the script's names on ``scripts/follow_rung2.py``'s pins and send."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from test_follow_partial import cue_samples

from residual_disposition import follow_lens_hint as LH
from residual_disposition import follow_lens_mask as LM
from residual_disposition import follow_partial as FP
from residual_disposition import follow_partial_draws as PD
from residual_disposition import follow_qualification as Q

REPO = Path(__file__).parents[1]


def _draws(line: str) -> list[dict]:
    return [r for r in PD.readings()["draws"] if r["line"] == line]


def test_ten_draws_per_line_l_only_under_salts_of_their_own():
    rows = PD.readings()["draws"]
    assert len(rows) == 80 and len({r["salt"] for r in rows}) == 80
    assert Counter(r["line"] for r in rows) == dict.fromkeys(
        ["countdown_50", "eli5_stack_200", "example_link_200", "pun_intended_200",
         "tldr/stack400/L/base", "tldr/stack400/L/state", "word_count_claim/stack200/L/state",
         "word_count_claim/stack200/L/base"], 10)  # fmt: skip
    assert all(r["role"] == "L" and r["slots"] == 80 and r["stream"] == 1 for r in rows)
    assert {PD.item_of(r) for r in rows} == {"follow/partial_draws"}
    assert [r["draw"] for r in _draws("countdown_50")] == list(range(10))
    # every salt new: none of the ablation's or the masking's
    old = {r["salt"] for line in FP.LINES for r in FP.readings(line)["ablation"]}
    old |= {r["salt"] for r in LM.readings()["lens_mask"]}
    assert not old & {r["salt"] for r in rows}


def test_each_draw_reads_its_deletions_package_decoder_and_prompt():
    for r in (r for r in PD.readings()["draws"] if r["kind"] == "steered"):
        cue = next(x for x in FP.readings(r["line"])["ablation"]
                   if x["purpose"] == "ablation_random" and x["role"] == "L")  # fmt: skip
        assert (r["package"], r["decoder"], r["behavior"], r["mode"]) == (
            cue["package"], cue["decoder"], cue["behavior"], cue["mode"])
        assert Q.prompt_and_settings(r)[0].sha256 == Q.prompt_and_settings(cue)[0].sha256
    masking = {(x["behavior"], x["mode"], x["decoder"]): x for x in LM.readings()["lens_mask"]
               if x["purpose"] == "deletion" and x["role"] == "L"}  # fmt: skip
    family = [r for r in PD.readings()["draws"] if r["kind"] == "family"]
    assert len(family) == 40
    for r in family:
        x = masking[(r["behavior"], r["mode"], r["decoder"])]
        assert r["package"] == x["package"] and r["reader"] == x["reader"] == "dstar_plain"
        assert Q.prompt_and_settings(r)[0].sha256 == Q.prompt_and_settings(x)[0].sha256


def test_a_steered_draw_removes_as_many_excerpts_none_the_detector_hits():
    from residual_disposition import follow_ladder as FL
    from residual_disposition.seeding import derive_seed

    line = "example_link_200"
    hits = {("0", 3), ("2", 7), ("2", 8), ("4", 0)}
    samples = cue_samples(line, hits)
    cue = {tuple(k) for k in FP.cue_keys(samples[line]["L"], "example_link")}
    assert cue == {(p, f"prompt {i}") for p, i in hits}
    first, second = _draws(line)[:2]
    gone = PD.steered_draw(first, samples)
    keys = [tuple(k) for k in gone["keys"]]
    assert gone["removed"] == len(keys) == gone["cue_hits"] == 4  # as many as the cue removal
    assert not cue & set(keys)  # drawn from the excerpts the detector does not hit
    assert gone["pool"] == gone["steered"] - 4 == 96
    order = FL.steered_keys(samples[line]["L"])
    assert keys == [k for k in order if k in set(keys)]  # kept in their order
    assert gone["seed"] == derive_seed(PD.DRAW_SEED, "draw", line, 0)
    assert gone == PD.steered_draw(first, samples)  # the seed fixes the draw
    assert PD.steered_draw(second, samples)["keys"] != gone["keys"]
    assert sum(gone["positions"].values()) == 4
    assert gone["detector"] == FP.detector("example_link").version


def test_a_steered_draws_evidence_is_rung_2_s_without_its_drawn_excerpts(tmp_path, monkeypatch):
    from residual_disposition import continuation_rules as cr

    run = cr._b2_probe_package(tmp_path)
    monkeypatch.setattr("residual_disposition.paths.runs_root", lambda: tmp_path)
    line = "example_link_200"
    samples = cue_samples(line, {("1", 2), ("3", 5)})
    text, gone = PD.steered_evidence(_draws(line)[0], run, samples)
    layer = json.loads(text[len("OVERVIEW:\n") :])["datasets"]["ds1"]["layers"]["19"]
    bare = {(p, x["prompt"]) for p, e in layer["positions"].items() for x in e["steering"]
            if "steered" not in x}  # fmt: skip
    assert bare == {tuple(k) for k in gone["keys"]} and len(bare) == 2


def _package(lists: dict) -> str:
    overview = {"datasets": {"ds1": {"layers": {"29": {"positions": {"1": lists}}}}}}
    return LH._HEAD + json.dumps(overview, ensure_ascii=False)


def _lists(text: str) -> dict:
    return LH._overview(text)["datasets"]["ds1"]["layers"]["29"]["positions"]["1"]


def test_a_family_draw_takes_one_free_entry_near_each_family_entry_and_keeps_the_family():
    ll = ["TL", " a", "TLS", " short", " b", " c", " d", " e", " f", " g", " h", "DR", " i"]
    ps = [["tl", 1], [" x", 1], [" y", 2], ["TLC", 3], [" z", 9]]
    text = _package({"logit_lens": ll, "patch_scope": ps})
    out, removed, log = PD.family_draw(text, "tldr", 7)
    family = LM.family_count(text, "tldr")
    assert family == 3 and LH.deleted_count(removed) == family == len(log)
    assert LH.deleted_count(LM.delete_family(text, "tldr")[1]) == family  # the target's count
    assert LM.family_count(out, "tldr") == family  # the family kept
    gone = [t for v in removed["29"]["1"].values() for t in v["tokens"]]
    assert not any(LM.excluded(t, "tldr") for t in gone)  # outside the family and the rule
    assert len(set(gone)) == len(gone)  # without replacement
    assert all(x["distance"] <= x["window"] == 5 for x in log)  # within 5 shown ranks
    assert " z" not in gone  # 8 ranks from "tl", with free entries nearer
    assert PD.family_draw(text, "tldr", 7) == (out, removed, log)  # the seed fixes the draw
    assert len({json.dumps(PD.family_draw(text, "tldr", s)[1]) for s in range(20)}) > 1


def test_a_family_draws_window_widens_one_rank_at_a_time_while_empty():
    ll = ["tl", "TL", " tl", "dr", "DR", " dr", "TLS", " a", " b", " c", " d", " e", " f", " g"]
    out, removed, log = PD.family_draw(_package({"logit_lens": ll, "patch_scope": []}), "tldr", 1)
    # ranks 1 to 6 the family, 7 near it: the first free entry is at rank 8
    assert [x["window"] for x in log] == [7, 7, 7, 7, 7, 7]
    assert [x["drawn"]["rank"] for x in log] == [8, 9, 10, 11, 12, 13]
    assert all(x["candidates"] == 1 for x in log)
    assert _lists(out)["logit_lens"][:7] == ll[:7] and _lists(out)["logit_lens"][7:] == [" g"]
    with pytest.raises(ValueError, match="too few free entries"):
        PD.family_draw(_package({"logit_lens": ll[:9], "patch_scope": []}), "tldr", 1)


def test_a_family_draws_record_carries_its_seed_rule_and_distances():
    ll = ["TL", " a", " b", "DR", " c", " d", " e"]
    r = _draws("tldr/stack400/L/base")[3]
    text, edit = PD.family_evidence(r, _package({"logit_lens": ll, "patch_scope": []}))
    assert edit["seed"] == PD.draw_seed("tldr/stack400/L/base", 3) and edit["draw"] == 3
    assert edit["count"] == edit["family_entries"] == 2 == edit["distances"]["n"]
    assert edit["rule_sha256"] == LM.MATCH_RULE_SHA256 and edit["widest_window"] == 5
    assert LM.family_count(text, "tldr") == 2


def _goals() -> dict:
    out = {}
    for x in PD.LINES:
        goal = {"reading": "t", "counted": 40, "naming": 4, "n": 80, "removed": 3}
        if x["kind"] == "steered":
            goal.update(keys=[["0", "p0"], ["0", "p1"], ["1", "p0"]], detector="v1")
        out[x["line"]] = goal
    return out


def test_a_draw_deleting_another_number_or_a_cue_excerpt_is_not_the_design():
    steered, family = _draws("countdown_50")[0], _draws("tldr/stack400/L/base")[0]
    ok = {"removed": 3, "cue_hits": 3, "detector": "v1", "keys": [["2", "p0"]] * 3}
    rows = [{**steered, "removed": ok}, {**family, "edit": {"count": 3}}]
    assert PD.plan_problems(rows, _goals()) == []
    bad = [{**steered, "removed": {**ok, "keys": [["0", "p1"], ["2", "p0"], ["2", "p1"]]}},
           {**steered, "removed": {**ok, "removed": 2}},
           {**steered, "removed": {**ok, "detector": "v2"}},
           {**steered, "removed": {**ok, "cue_hits": 4}},
           {**family, "edit": {"count": 4}}]  # fmt: skip
    found = PD.plan_problems(bad, _goals())
    assert len(found) == 5
    assert any("removes an excerpt the cue removal removed" in p for p in found)
    assert any("deletes 2, its target 3" in p for p in found)
    assert any("deletes 4, its target 3" in p for p in found)
    assert any("not the cue removal's detector" in p for p in found)


def test_the_rank_counts_draws_naming_no_more_than_the_target():
    assert PD.draw_rank(0, [3] * 10) == {
        "k": 0, "wording": "lower than in all ten draws of as many pieces without the marker"}
    assert PD.draw_rank(4, [40] * 8 + [4, 2]) == {"k": 2, "wording": "as low in 2 of ten draws"}
    assert PD.draw_rank(80, [80] * 10)["wording"] == "as low in 10 of ten draws"
    assert PD.draw_rank(4, [40] * 9 + [None]) == {"k": None, "wording": None}
    assert PD.draw_rank(None, [40] * 10)["k"] is None
    assert PD.draw_rank(4, [40] * 9)["k"] is None  # not ten draws


def test_the_freedom_counts_single_candidate_choices_and_distinct_pieces():
    def choice(token, rank, window, candidates):
        return {"layer": "19", "position": "0", "list": "logit_lens", "drawn": {"token": token,
                "rank": rank}, "window": window, "candidates": candidates}  # fmt: skip

    family = [{"line": "f", "kind": "family", "edit": {"count": 2, "drawn": [
        choice("a", 9, 7, 1), choice(t, r, 5, 3)]}} for t, r in (("b", 2), ("c", 3), ("b", 2))]
    steered = [{"line": "s", "kind": "steered", "removed": {"removed": 2, "pool": 5,
                "keys": keys}} for keys in ([[0, 1], [0, 2]], [[0, 2], [1, 4]])]  # fmt: skip
    got = PD.freedom(family + steered)
    assert got["f"] == {"per_draw": [2], "choices": 6, "single_candidate": 3,
                        "single_candidate_share": 0.5, "distinct_deleted": 3,
                        "widest_window": 7}  # fmt: skip
    assert got["s"] == {"per_draw": [2], "pool": [5], "distinct_deleted": 3}


def test_the_summary_gives_each_draw_its_difference_from_the_target(monkeypatch):
    from residual_disposition import follow_readings as D

    monkeypatch.setattr(PD, "targets", lambda runs=None: _goals())
    # each row's slots stand for its naming count
    monkeypatch.setattr(D, "margins", lambda slots, *a: slots)
    monkeypatch.setattr(FP, "_specificity", lambda m: {"feature_slots": m})
    naming = dict(zip(range(10), [4, 0, 30, 40, 50, 60, 70, 80, 9, 10], strict=True))
    rows = []
    for r in PD.readings()["draws"]:
        edit = {"removed": {"removed": 3}} if r["kind"] == "steered" else {"edit": {"count": 3}}
        rows.append({"reading": {**r, **edit}, "count": {"counted": 80, "n": 80},
                     "slots": naming[r["draw"]]})  # fmt: skip
    defs = dict.fromkeys(PD.BEHAVIORS, None)
    out = PD.summary(rows, {}, defs)
    assert out["pending"] is False and list(out["lines"]) == [x["line"] for x in PD.LINES]
    line = out["lines"]["countdown_50"]
    assert line["naming"] == list(naming.values()) and line["counted"] == [80] * 10
    assert (line["k"], line["wording"]) == (2, "as low in 2 of ten draws")
    assert line["target"] == {"reading": "t", "counted": 40, "naming": 4, "n": 80, "removed": 3}
    first = line["draws"][0]
    assert first["removed"] == 3 and first["difference"]["difference"] == 0.0
    assert first["difference"]["low"] < 0 < first["difference"]["high"]
    assert line["draws"][3]["difference"] == FP.difference((40, 80), (4, 80))
    assert "p_value" not in json.dumps(out)
    # an open draw leaves its line's rank and difference open, and the summary pending
    rows[0] = {**rows[0], "count": {"pending": "slot 0"}}
    out = PD.summary(rows, {}, defs)
    assert out["pending"] is True
    assert out["lines"]["countdown_50"]["k"] is None
    assert out["lines"]["countdown_50"]["draws"][0]["difference"] is None


def test_the_script_takes_rung_2_s_pins_and_send_under_names_of_its_own(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import follow_partial_draws as S

    for name in ("PINS", "HISTORY", "RUN", "SENDS", "plan", "R2", "LABEL", "ROOT", "main",
                 "REPO_ROOT", "send"):
        monkeypatch.setattr(S.F, name, getattr(S.F, name))  # each restored after the test
    seen = []
    main = S.F.main
    monkeypatch.setattr(S.F, "main", lambda argv=None: seen.append(argv) or 0)
    assert S.main(["--send", "draws"]) == 0 and seen == [["--send", "draws"]]
    assert (S.F.PINS, S.F.HISTORY, S.F.RUN, S.F.ROOT, S.F.SENDS, S.F.LABEL) == (
        "manifests/follow_partial_draws_inputs.json",
        "manifests/stage3_judge_history_follow_partial_draws.json", "follow-partial",
        "follow_partial_draws", ("draws",), "follow/partial-draws")  # fmt: skip
    assert S.F.plan is S.plan and S.F.R2.item_of({}) == "follow/partial_draws"
    assert S.F.send is S.send_by_draw and S.SEND is not S.send_by_draw
    # --pin writes once: committed pins are never overwritten
    monkeypatch.setattr(S.F, "main", main)
    monkeypatch.setattr(S.F, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(S, "plan", lambda: {"draws": []})
    (tmp_path / "manifests").mkdir()
    (tmp_path / S.PINS).write_text("{}")
    with pytest.raises(SystemExit, match="written once"):
        S.main(["--pin"])
    assert (tmp_path / S.PINS).read_text() == "{}"


def test_the_draws_are_sent_in_waves_by_draw_number_each_through_rung_2_s_send(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import follow_partial_draws as S

    cells = [{"reading": r} for r in PD.readings()["draws"]]
    waves = []

    def send(part, run, ctx):
        waves.append([(c["reading"]["line"], c["reading"]["draw"]) for c in part])
        return {"follow/partial_draws": {"follow/partial_draws": 1.5}}  # rung 2's send's form

    monkeypatch.setattr(S, "SEND", send)
    item = "follow/partial_draws"
    assert S.send_by_draw(cells, "run", {}) == {item: {item: 15.0}}
    lines = sorted({c["reading"]["line"] for c in cells})
    assert len(waves) == PD.DRAWS == 10 and len(lines) == 8
    assert [sorted(w) for w in waves] == [[(x, i) for x in lines] for i in range(10)]


def test_the_masks_are_written_once(tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import follow_partial_draws as S

    for name in ("PINS", "HISTORY", "RUN", "SENDS", "plan", "R2", "LABEL", "ROOT"):
        monkeypatch.setattr(S.F, name, getattr(S.F, name))  # each restored after the test
    monkeypatch.setattr(S, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(S, "plan", lambda: {"draws": []})
    monkeypatch.setattr(PD, "targets", lambda runs=None: {})
    (tmp_path / "manifests").mkdir()
    assert S.main(["--masks"]) == 0 and not (tmp_path / S.MASKS).exists()
    assert S.main(["--masks", "--write"]) == 0
    record = json.loads((tmp_path / S.MASKS).read_text())
    assert record["seed"] == PD.DRAW_SEED and record["window"] == PD.WINDOW
    assert "before any request" in record["note"]
    with pytest.raises(SystemExit, match="written once"):
        S.main(["--masks", "--write"])


def test_the_targets_are_read_only_at_their_sha256(tmp_path):
    for path, _ in (PD.ABLATION_INPUTS, PD.ABLATION_COMPARISON, PD.MASKING):
        f = tmp_path / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{}")
    with pytest.raises(ValueError, match="not its recorded file"):
        PD.targets(tmp_path)
