"""D*'s descriptive arms on the base decoder (``follow_lens_hint``): the hint's prompt, the
fragment-family deletion and its matched control, the cells, their salts and aliases, their
packages as committed readings', every hypothesis's category, and the summary with its baselines,
comparisons and frozen criteria."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_lens_hint as LH
from residual_disposition import follow_qualification as Q
from residual_disposition import follow_regions as fr
from residual_disposition import lens_render as R

REPO = Path(__file__).parents[1]
RULES = {"confusion_min": "3/10", "specificity_min": "1/2"}


def test_the_hint_is_d_star_s_prompt_with_one_generic_sentence_in_its_hygiene():
    text, dstar = LH.hint_prompt().text, R.prompt().text
    assert LH.hint_prompt().sha256 == LH.HINT_PROMPT_SHA256
    assert LH.hint_prompt().version == "adl-lens-multiscale-hint-v1"
    assert dstar.count(LH._HINT_ANCHOR) == 1 and dstar.count(LH.HINT_LINE) == 0
    assert text.count(LH.HINT_LINE) == 1 and text.replace(LH.HINT_LINE, "") == dstar
    hygiene = text[text.index("Evidence hygiene and weighting") : text.index("Predict the single")]
    assert LH.HINT_LINE in hygiene
    assert LH.HINT_LINE == ("- Some tokens may be pieces of words, abbreviations or labels; "
                            "consider what they could mean together.\n")  # fmt: skip
    for word in ("eli5", "ELI5", "five", "explain", "pun", "tldr"):
        assert word not in LH.HINT_LINE  # the same sentence for every state


def test_the_hint_s_prompt_is_checked_against_its_sha256(monkeypatch):
    monkeypatch.setattr(LH, "HINT_PROMPT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="hint's prompt sha256"):
        LH.hint_prompt()


def overview(lists):
    layers = {layer: {"positions": {pos: {"logit_lens": ll, "patch_scope": ps}
                                    for pos, (ll, ps) in positions.items()}}
              for layer, positions in lists.items()}  # fmt: skip
    body = {"datasets": {"ds1": {"patchscope_scale_grid": [0.5, 1.0], "layers": layers}}}
    return "OVERVIEW:\n" + json.dumps(body, ensure_ascii=False)


LL = ["a", " El", "eli5", "ELI", " elf", "Elf ", "  el", "eli5.", "élan", "b"]
PS = [["ELI5", 1, ["0.5"]], ["c", 1, ["1"]], [" el", 2, ["0.5-1"]], ["delf", 3, ["1"]],
      [" Eli", 4, ["1"]], ["d", 5, ["0.5"]]]  # fmt: skip
TEXT = overview({"19": {"0": (LL, PS), "1": (["x"], [["y", 1, ["1"]]])},
                 "29": {"0": ([" ELF", "q"], [["elf", 2, ["1"]], ["z", 7, ["1"]]])}})  # fmt: skip


def lists(text):
    return json.loads(text[len("OVERVIEW:\n") :])["datasets"]["ds1"]["layers"]


def test_the_deletion_removes_only_the_family_and_keeps_the_rest_in_place():
    got, removed = LH.delete_fragments(TEXT)
    layers = lists(got)
    p0 = layers["19"]["positions"]["0"]
    assert p0["logit_lens"] == ["a", "Elf ", "  el", "eli5.", "élan", "b"]
    assert p0["patch_scope"] == [["c", 1, ["1"]], ["delf", 3, ["1"]], ["d", 5, ["0.5"]]]
    assert removed["19"]["0"] == {
        "logit_lens": {"tokens": [" El", "eli5", "ELI", " elf"], "ranks": [1, 2, 3, 4]},
        "patch_scope": {"tokens": ["ELI5", " el", " Eli"], "ranks": [0, 2, 4]}}  # fmt: skip
    assert layers["19"]["positions"]["1"] == {"logit_lens": ["x"], "patch_scope": [["y", 1, ["1"]]]}
    assert layers["29"]["positions"]["0"] == {"logit_lens": ["q"], "patch_scope": [["z", 7, ["1"]]]}
    assert LH.deleted_count(removed) == LH.family_count(TEXT) == 4 + 3 + 2
    assert LH.family_count(got) == 0  # none remains, nothing backfilled
    assert LH.delete_fragments(got)[0] == got
    with pytest.raises(ValueError):
        LH.delete_fragments("not an overview")


def test_the_matched_control_deletes_as_many_rank_nearest_entries_outside_the_family():
    got, removed = LH.matched_deletion(TEXT, seed=7)
    layers = lists(got)
    assert LH.deleted_count(removed) == LH.family_count(TEXT)
    assert LH.family_count(got) == LH.family_count(TEXT)  # the family's entries kept
    p0 = removed["19"]["0"]
    # logit lens: family at ranks 1-4, the free ranks 0, 5, 6, 7 are the nearest (in order)
    assert p0["logit_lens"]["ranks"] == [0, 5, 6, 7]
    assert p0["logit_lens"]["tokens"] == ["a", "Elf ", "  el", "eli5."]
    assert len(p0["patch_scope"]["ranks"]) == 3 and all(
        not LH.in_family(t) for t in p0["patch_scope"]["tokens"])  # fmt: skip
    assert layers["19"]["positions"]["0"]["logit_lens"] == [" El", "eli5", "ELI", " elf", "élan",
                                                            "b"]  # fmt: skip
    assert LH.matched_deletion(TEXT, seed=7) == (got, removed)  # its seed fixes the ties
    with pytest.raises(ValueError, match="too few entries outside the family"):
        LH.matched_deletion(overview({"19": {"0": (["el", "eli", "x"], [])}}), seed=7)
    a, c = (next(r for r in LH.readings()["lens_hint"] if r["purpose"] == arm and r["role"] == "A")
            for arm in ("matched_hint", "matched"))  # fmt: skip
    assert LH.match_seed(a) == LH.match_seed(c)  # the same deletion with and without the hint


def test_an_evidence_without_the_family_is_left_byte_identical(tmp_path):
    run = cr._b2_probe_package(tmp_path)
    text = R.render(run, "base", tmp_path)[0]
    got, removed = LH.delete_fragments(text)
    assert got == text and LH.deleted_count(removed) == 0
    assert LH.matched_deletion(text, 1)[0] == text


def test_the_readings_and_their_salts():
    rows = LH.readings()["lens_hint"]
    by = {}
    for r in rows:
        by.setdefault(r["purpose"], []).append(r)
    assert {k: len(v) for k, v in by.items()} == {"hint": 31, "deletion_hint": 3, "deletion": 1,
                                                  "matched_hint": 3, "matched": 1}
    assert sum(r["slots"] for r in rows) == 39 * 80
    salts = [r["salt"] for r in rows]
    assert len(set(salts)) == len(salts)
    parts = {"hint": "hint", "deletion_hint": "deletion-hint", "deletion": "deletion",
             "matched_hint": "matched-hint", "matched": "matched"}  # fmt: skip
    for r in rows:
        assert r["salt"] == (f"follow/lenshint/base/{parts[r['purpose']]}/{r['behavior']}/"
                             f"{r['mode']}/{r['role']}/stream1")  # fmt: skip
        assert r["reader"] == ("dstar_plain" if r["purpose"] in ("deletion", "matched")
                               else "dstar_hint")  # fmt: skip
        assert Q.KINDS[r["reader"]] == "lens" and LH.item_of(r) == "follow/lens_hint"
    for arm in ("deletion_hint", "matched_hint"):
        assert {(r["mode"], r["role"]) for r in by[arm]} == {
            ("shared", "A"), ("continued200", "L"), ("stack200", "L")}
    assert [(r["mode"], r["role"]) for r in by["deletion"] + by["matched"]] == [("shared", "A")] * 2
    assert [r["salt"] for r in rows if r["decoder"] == "random"] == [
        "follow/lenshint/base/hint/tldr/stack400/L_random/stream1"]  # no base_random branch
    for line in {(r["behavior"], r["mode"]) for r in by["hint"]} - {
            (b, "shared") for b in LH.BEHAVIORS}:  # fmt: skip
        assert sorted(r["role"] for r in by["hint"] if (r["behavior"], r["mode"]) == line) == (
            sorted(["L", "H_s", "X", "L_random", "recipe"]))
    for p in LH.SOURCES.values():  # no salt of any committed reading
        assert not set(salts) & set(json.loads((REPO / p).read_text())["readings"])
    # the H_s aliases are hint readings, no readings of their own
    assert set(LH.ALIASES.values()) <= set(salts) and not set(LH.ALIASES) & set(salts)
    plain, hint = (next(r for r in rows if r["purpose"] == arm and r["role"] == "A"
                        and r["behavior"] == "eli5") for arm in ("deletion", "deletion_hint"))
    assert Q.prompt_and_settings(plain)[0].text == R.prompt().text
    assert Q.prompt_and_settings(hint)[0].version == "adl-lens-multiscale-hint-v1"
    keys = [{x.key for t in Q.attempts(r, "same") for x in t} for r in (plain, hint)]
    assert not keys[0] & keys[1]


def test_the_frozen_design_is_checked_at_plan():
    rows = [{**r, "edit": {"count": 60}} for r in LH.readings()["lens_hint"]]
    assert LH.plan_problems(rows) == []
    short = [{**r, "edit": {"count": 40}} if r["purpose"] == "deletion" else r for r in rows]
    found = LH.plan_problems(short)
    assert any("fewer than 58" in f for f in found)
    assert any("its deletion 40" in f for f in found)


def test_every_cell_is_a_committed_reading_s_package():
    pinned = LH.pins(REPO)
    for r in LH.readings()["lens_hint"]:
        source = r["package"]["source"]
        pin = pinned[source][r["package"]["salt"]]  # a pinned reading of its source
        if source == "partial":
            assert r["decoder"] == "base_random" and "follow_partial_pod" in pin["directory"]
        else:
            assert r["package"]["salt"].split("/")[-2 if "dose" not in r["package"]["salt"]
                                                   else -3] == r["decoder"]  # fmt: skip


def test_a_cell_off_its_pinned_evidence_or_branch_is_refused(tmp_path):
    run = cr._b2_probe_package(tmp_path)
    text = R.render(run, "base", tmp_path)[0]
    directory = str(run.relative_to(tmp_path))
    reading = {"salt": "s", "decoder": "base", "purpose": "hint",
               "package": {"source": "readings", "salt": "x"}}  # fmt: skip
    pin = {"directory": directory, "evidence_sha256": hashlib.sha256(text.encode()).hexdigest()}
    assert LH.source_dir(reading, {"readings": {"x": pin}}, tmp_path, list) == run
    with pytest.raises(ValueError, match="pinned evidence"):
        LH.source_dir({**reading, "decoder": "state"}, {"readings": {"x": pin}}, tmp_path, list)
    with pytest.raises(ValueError, match="not a pinned reading"):
        LH.source_dir(reading, {"readings": {}}, tmp_path, list)
    partial = {**reading, "decoder": "base_random", "package": {"source": "partial", "salt": "x"}}
    with pytest.raises(ValueError, match="base_random branch"):
        LH.source_dir(partial, {"partial": {"x": pin}}, tmp_path, list)
    assert LH.source_dir(partial, {"partial": {"x": pin}}, tmp_path, lambda: [run]) == run


def test_the_arms_take_rung_2_s_pins_and_send_under_names_of_their_own(monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import follow_lens_hint as S

    for name in ("PINS", "HISTORY", "RUN", "SENDS", "plan", "state", "R2", "LABEL", "ROOT",
                 "main"):  # fmt: skip
        monkeypatch.setattr(S.F, name, getattr(S.F, name))  # each restored after the test
    seen = []
    monkeypatch.setattr(S.F, "main", lambda argv=None: seen.append(
        (argv, S.F.PINS, S.F.HISTORY, S.F.RUN, S.F.ROOT, S.F.SENDS, S.F.plan, S.F.state)) or 0)
    for argv in (["--pin"], ["--send", "lens_hint"]):
        monkeypatch.setattr(sys, "argv", ["follow_lens_hint.py", *argv])
        assert S.main() == 0
    assert [s[0] for s in seen] == [["--pin"], ["--send", "lens_hint"]]
    assert seen[0][1:] == ("manifests/follow_lens_hint_inputs.json",
                           "manifests/stage3_judge_history_follow_lens_hint.json",
                           "follow-lens-hint", "follow_lens_hint", ("lens_hint",), S.plan,
                           S.state)  # fmt: skip
    assert S.F.R2.item_of({}) == "follow/lens_hint" and S.F.R2.BEHAVIORS == LH.BEHAVIORS


def test_every_parsed_hypothesis_s_category_is_graded_and_an_open_one_is_due(tmp_path):
    from residual_disposition import follow_readings as D
    from residual_disposition.continuation_judge import category_letters
    from residual_disposition.judge import Completion, JudgeCache

    definitions, spec = D.grader()
    cache = JudgeCache(tmp_path / "judge")
    reading = next(r for r in LH.readings()["lens_hint"] if r["behavior"] == "pun_intended")
    slots = [{"hypothesis": {"value": "I will explain things as to a five-year-old."}},
             {"hypothesis": {"due": "x"}},
             {"hypothesis": {"value": "I flag puns."},
              "grades": {"category": {"value": frozenset({"pun_intended"})}}}]
    tries = D.grade_attempts(slots[0]["hypothesis"]["value"], "pun_intended", definitions, spec)
    rows = LH.with_categories([{"reading": reading, "slots": slots}], definitions, spec, cache)
    assert rows[0]["slots"][0]["grades"]["category"]["due"].key == tries["category"][0].key
    assert "grades" not in rows[0]["slots"][1]  # an unparsed slot is not graded
    assert rows[0]["slots"][2]["grades"]["category"]["value"] == frozenset({"pun_intended"})
    assert LH._naming(rows[0])["eli5"] is None  # open while a category is due
    letter = next(k for k, b in category_letters().items() if b == "eli5")
    cache.put(tries["category"][0], Completion(json.dumps({"categories": [letter]})),
              service_tier="standard")  # fmt: skip
    del slots[0]["grades"]
    rows = LH.with_categories(rows, definitions, spec, cache)
    assert rows[0]["slots"][0]["grades"]["category"]["value"] == frozenset({"eli5"})
    assert LH._naming(rows[0]) == {"parsed": 2, "eli5": 1}


def slots(counted, category="eli5"):
    out = []
    for i in range(80):
        yes = i < counted
        s = {"hypothesis": {"value": f"h{i}"},
             "labels": {"strict": {"value": ("other", False)}, "lenient": {"value": yes}},
             "grades": {"category": {"value": frozenset({category if yes else "tldr"})}}}
        if yes:
            s["grades"]["specificity"] = {"value": ("feature", True)}
        out.append(s)
    return out


def rung1_fixture(rows):
    counts = {r["reading"]["package"]["salt"]: {"counted": 0, "n": 80} for r in rows
              if r["reading"]["package"]["source"] != "partial"}  # fmt: skip
    zero = {"counted": 0, "specificity": {"feature": "0"}}
    return {"counts": counts, "margins": dict.fromkeys(LH.RUNG1_LINES, zero)}


def test_the_summary_shows_baselines_rules_and_criteria_and_decides_nothing(monkeypatch):
    counts = {"L": 40, "H_s": 0, "X": 1, "L_random": 2, "recipe": 70, "A": 80, "H": 10}
    rows = []
    for r in LH.readings()["lens_hint"]:
        n = counts[r["role"]] if r["purpose"] == "hint" else 20
        if r["purpose"] == "matched":
            n = 50
        rows.append({"reading": {**r, "edit": {"family_entries": 1, "count": 58}},
                     "count": {"counted": n, "n": 80}, "slots": slots(n)})
    monkeypatch.setattr(LH, "rung1_counts", lambda runs=None: rung1_fixture(rows))
    defs = {b: SimpleNamespace(alternatives=["one"]) for b in LH.BEHAVIORS}
    out = LH.summary(rows, fr.load(), RULES, defs)
    cells = out["cells"]
    rnd = cells["follow/lenshint/base/hint/eli5/stack200/L_random/stream1"]["baseline"]
    assert rnd["available"] is False and "rung 1 read no D* reading" in rnd["reason"]
    el = cells["follow/lenshint/base/hint/eli5/stack200/L/stream1"]
    assert el["baseline"]["available"] and el["baseline"]["count"] == 0
    assert "hint" in el["baseline"]["note"] and el["feature_slots"] == 40
    line = out["lines"]["eli5/stack200"]
    assert line["hint"]["rule_shown"]["available"] and line["hint"]["rule_shown"]["contrast"]
    assert line["hint"]["rung1_rule_shown"] == {"available": False,
                                               "reason": "a count is open or has no baseline"}
    deletion = out["lines"]["eli5/stack200"]["deletion_hint"]["rule_shown"]
    assert deletion["available"] is False and "X" in deletion["reason"]
    alias = out["aliases"]["follow/lenshint/base/deletion-hint/eli5/stack200/H_s/stream1"]
    assert alias["reading"].endswith("/hint/eli5/stack200/H_s/stream1") and alias["exact"] is False
    assert out["lines"]["eli5/stack200"]["deletion_hint"]["counts"]["H_s"] == 0  # the alias's
    crit = out["fragment_criteria"]["eli5/stack200"]
    assert crit["count_without_feature"] is False  # the feature slots rose with the count
    assert crit["controls_rise"] is False  # H 10 of 80 rises by less than half of L's 40
    d = out["deletion"]["deletion_hint/eli5/stack200/L"]
    assert d["deletion-matched/counted"]["difference"] == 0  # 20 against the matched 20
    assert d["pattern/counted"] == "not_separated" and "unedited-deletion/counted" in d
    a = out["deletion"]["deletion/eli5/shared/A"]
    assert a["pattern/counted"] == "family_carries"  # 20 against the matched 50
    assert a["rung1"]["available"] and "unedited" in a["rung1"]["note"]
    naming = out["eli5_false_naming"]
    assert naming["follow/lenshint/base/hint/pun_intended/stack200/L/stream1"]["eli5"] == 40
    assert "follow/lenshint/base/hint/eli5/stack200/L/stream1" not in naming
    assert out["pending"] is False and out["criteria"] and out["deletion_patterns"]
    assert not {"decision", "decisions", "identified", "status"} & set(line["hint"])


def test_a_missing_baseline_is_reported_with_its_reason_never_substituted():
    one = {"counts": {}, "margins": {}}
    r = next(x for x in LH.readings()["lens_hint"] if x["role"] == "A")
    b = LH.baseline(r, one)
    assert b == {"available": False, "salt": r["package"]["salt"], "count": None,
                 "reason": "the source reading is not in rung 1's committed records"}
    p = next(x for x in LH.readings()["lens_hint"] if x["package"]["source"] == "partial")
    assert LH.baseline(p, one)["available"] is False
    assert LH.fragment_criteria("eli5/stack200", {}, one) == {"available": False,
                                                              "reason": "no hint reading of L"}
    assert LH._rule({"L": 1}, fr.load())["reason"] == "no H_s, X, L_random cell in this arm"
