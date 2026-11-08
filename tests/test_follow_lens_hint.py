"""D*'s two descriptive arms on the base decoder (``follow_lens_hint``): the hint's prompt, the
mask, the cells and their salts, their packages as committed readings', and the summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


def test_the_mask_removes_only_the_marker_tokens_and_keeps_the_rest_in_place():
    ll = ["a", " El", "eli5", "ELI", " elf", "Elf ", "  el", "eli5.", "élan", "b"]
    ps = [["ELI5", 1, ["0.5"]], ["c", 1, ["1"]], [" el", 2, ["0.5-1"]], ["delf", 3, ["1"]],
          [" Eli", 4, ["1"]], ["d", 5, ["0.5"]]]  # fmt: skip
    text = overview({"19": {"0": (ll, ps), "1": (["x"], [["y", 1, ["1"]]])},
                     "29": {"0": ([" ELF"], [["elf", 2, ["1"]], ["z", 7, ["1"]]])}})  # fmt: skip
    got, removed = LH.mask(text)
    layers = json.loads(got[len("OVERVIEW:\n") :])["datasets"]["ds1"]["layers"]
    p0 = layers["19"]["positions"]["0"]
    assert p0["logit_lens"] == ["a", "Elf ", "  el", "eli5.", "élan", "b"]
    assert p0["patch_scope"] == [["c", 1, ["1"]], ["delf", 3, ["1"]], ["d", 5, ["0.5"]]]
    assert removed["19"]["0"] == {"logit_lens": [" El", "eli5", "ELI", " elf"],
                                  "patch_scope": ["ELI5", " el", " Eli"]}  # fmt: skip
    assert layers["19"]["positions"]["1"] == {"logit_lens": ["x"], "patch_scope": [["y", 1, ["1"]]]}
    assert removed["19"]["1"] == {"logit_lens": [], "patch_scope": []}
    assert layers["29"]["positions"]["0"] == {"logit_lens": [], "patch_scope": [["z", 7, ["1"]]]}
    assert LH.masked_count(removed) == 4 + 3 + 2
    for layer in layers.values():  # none remains
        for lists in layer["positions"].values():
            assert not any(LH.masked(t) for t in lists["logit_lens"])
            assert not any(LH.masked(e[0]) for e in lists["patch_scope"])
    assert LH.mask(got)[0] == got  # nothing left to remove
    with pytest.raises(ValueError):
        LH.mask("not an overview")


def test_the_mask_leaves_an_evidence_without_markers_byte_identical(tmp_path):
    run = cr._b2_probe_package(tmp_path)
    text = R.render(run, "base", tmp_path)[0]
    got, removed = LH.mask(text)
    assert got == text and LH.masked_count(removed) == 0


def test_the_cells_read_once_under_salts_of_their_own():
    rows = LH.readings()["lens_hint"]
    hint = [r for r in rows if r["purpose"] == "hint"]
    mask = [r for r in rows if r["purpose"] == "mask"]
    assert (len(hint), len(mask)) == (31, 5) and sum(r["slots"] for r in rows) == 2880
    salts = [r["salt"] for r in rows]
    assert len(set(salts)) == len(salts)
    for r in rows:
        assert r["salt"] == (f"follow/lenshint/base/{r['purpose']}/{r['behavior']}/{r['mode']}/"
                             f"{r['role']}/stream1")  # fmt: skip
        assert r["reader"] == {"hint": "dstar_hint", "mask": "dstar_mask"}[r["purpose"]]
        assert Q.KINDS[r["reader"]] == "lens" and LH.item_of(r) == "follow/lens_hint"
    assert {r["decoder"] for r in rows} == {"base", "base_random", "random"}
    assert [r["salt"] for r in rows if r["decoder"] == "random"] == [
        "follow/lenshint/base/hint/tldr/stack400/L_random/stream1"]  # no base_random branch
    lines = {(r["behavior"], r["mode"]) for r in hint}
    assert lines == {("eli5", "shared"), ("eli5", "stack200"), ("eli5", "stack800"),
                     ("eli5", "continued200"), ("pun_intended", "shared"),
                     ("pun_intended", "stack200"), ("tldr", "shared"), ("tldr", "stack400")}
    for line in lines - {(b, "shared") for b in LH.BEHAVIORS}:
        assert sorted(r["role"] for r in hint if (r["behavior"], r["mode"]) == line) == sorted(
            ["L", "H_s", "X", "L_random", "recipe"])
    assert {(r["mode"], r["role"]) for r in mask} == {
        ("shared", "A"), ("continued200", "L"), ("continued200", "H_s"), ("stack200", "L"),
        ("stack200", "H_s")} and {r["behavior"] for r in mask} == {"eli5"}  # fmt: skip
    for p in LH.SOURCES.values():  # no salt of any committed reading
        assert not set(salts) & set(json.loads((REPO / p).read_text())["readings"])
    # the mask's and the hint's twins: the same package, other salts, prompts and keys
    a, b = (next(r for r in rows if r["purpose"] == arm and r["role"] == "A"
                 and r["behavior"] == "eli5") for arm in ("hint", "mask"))  # fmt: skip
    assert a["package"] == b["package"]
    assert Q.prompt_and_settings(a)[0].version == "adl-lens-multiscale-hint-v1"
    assert Q.prompt_and_settings(b)[0].text == R.prompt().text
    keys = [{x.key for t in Q.attempts(r, "same") for x in t} for r in (a, b)]
    assert not keys[0] & keys[1]


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

    for name in ("PINS", "HISTORY", "RUN", "SENDS", "plan", "R2", "LABEL", "ROOT", "main"):
        monkeypatch.setattr(S.F, name, getattr(S.F, name))  # each restored after the test
    seen = []
    monkeypatch.setattr(S.F, "main", lambda argv=None: seen.append(
        (argv, S.F.PINS, S.F.HISTORY, S.F.RUN, S.F.ROOT, S.F.SENDS, S.F.plan)) or 0)
    for argv in (["--pin"], ["--send", "lens_hint"]):
        monkeypatch.setattr(sys, "argv", ["follow_lens_hint.py", *argv])
        assert S.main() == 0
    assert [s[0] for s in seen] == [["--pin"], ["--send", "lens_hint"]]
    assert seen[0][1:] == ("manifests/follow_lens_hint_inputs.json",
                           "manifests/stage3_judge_history_follow_lens_hint.json",
                           "follow-lens-hint", "follow_lens_hint", ("lens_hint",), S.plan)
    assert S.F.R2.item_of({}) == "follow/lens_hint" and S.F.R2.BEHAVIORS == LH.BEHAVIORS


def slots(counted):
    out = []
    for i in range(80):
        yes = i < counted
        s = {"labels": {"strict": {"value": ("other", False)}, "lenient": {"value": yes}}}
        if yes:
            s["grades"] = {"category": {"value": frozenset({"eli5"})},
                           "specificity": {"value": ("feature", True)}}  # fmt: skip
        out.append(s)
    return out


def test_the_summary_shows_the_stage_1_rule_beside_rung_1_and_decides_nothing(monkeypatch):
    from types import SimpleNamespace

    rows = []
    counts = {"L": 40, "H_s": 0, "X": 1, "L_random": 2, "recipe": 70}
    for r in LH.readings()["lens_hint"]:
        if r["purpose"] == "hint" and (r["behavior"], r["mode"]) == ("eli5", "stack200"):
            n = counts[r["role"]]
            rows.append({"reading": r, "count": {"counted": n, "n": 80}, "slots": slots(n)})
        if r["purpose"] == "mask" and (r["mode"], r["role"]) == ("stack200", "L"):
            r = {**r, "masked": {"19": {"0": {"logit_lens": [" el"], "patch_scope": ["ELI"]}}}}
            rows.append({"reading": r, "count": {"due": 1}, "slots": slots(0)})
    one = {r["reading"]["package"]["salt"]: 0 for r in rows}
    monkeypatch.setattr(LH, "rung1_counts", lambda runs=None: one)
    defs = {"eli5": SimpleNamespace(alternatives=["one"])}
    out = LH.summary(rows, fr.load(), RULES, defs)
    line = out["lines"]["eli5/stack200"]
    assert line["hint"]["counts"] == counts and line["hint"]["rung1"] == dict.fromkeys(counts, 0)
    rule = line["hint"]["rule_shown"]
    assert rule["level"] == "1/60" and rule["random_within"] is True and rule["contrast"] is True
    assert line["hint"]["rung1_rule_shown"]["contrast"] is False
    assert line["mask"]["rule_shown"] is None  # its counts are open
    cell = out["cells"]["follow/lenshint/base/mask/eli5/stack200/L/stream1"]
    assert cell["masked_tokens"] == 2 and cell["count"] is None
    el = out["cells"]["follow/lenshint/base/hint/eli5/stack200/L/stream1"]
    assert el["margins"]["specificity"]["feature"] == "1" and el["rung1"] == 0
    assert out["pending"] is True
    assert not {"decision", "decisions", "identified", "status"} & set(line["hint"])
