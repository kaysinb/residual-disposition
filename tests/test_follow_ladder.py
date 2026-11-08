"""The ladder's rungs 2 and 3 as fixed at barrier B3 (§13.6): rung 2's prompt, D*'s with adl-v1's
steering parts restored verbatim; its evidence, D*'s state rendering with each position's steered
and unsteered samples at layer 20, cut to 128 characters, a failed position's unsteered ones only;
rung 3's agent and comparator prompts with one query protocol, and its turn parser."""

from __future__ import annotations

import json

import pytest

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_ladder as L
from residual_disposition import lens_render as R
from residual_disposition.summarizer import (
    _ADL_STEERING_HYGIENE,
    _ADL_STEERING_OVERVIEW,
    PROMPTS,
)


def steering(failed=(), prompts=L.CLOSED_PROMPTS, long="ü" * 300):
    """Each of D*'s positions with one sample per closed prompt; ``failed`` positions without
    steered samples."""
    per = {}
    for p in R.POSITIONS:
        bad = str(p) in failed
        per[str(p)] = {"failed": "steering failed: no coherent strength" if bad else None,
                       "samples": [{"prompt": f"prompt {i}",
                                    "steered": None if bad else f"{p}/{i} steered {long}",
                                    "unsteered": f"{p}/{i} unsteered {long}"}
                                   for i in range(prompts)]}  # fmt: skip
    return {"per_position": per}


def test_rung_2_s_prompt_is_d_star_s_with_the_steering_parts_restored():
    text, dstar = L.rung2_prompt().text, R.prompt().text
    for part in (_ADL_STEERING_OVERVIEW, _ADL_STEERING_HYGIENE):
        assert text.count(part) == 1 and dstar.count(part) == 0
    assert "three ways" in text and "two ways" in dstar
    assert text.count("steering_failed") == 1 and "128 characters" in text
    for old, _ in L._RUNG2_EDITS:  # every other part of D*'s text is kept
        assert dstar.count(old) == 1
    assert text.endswith(dstar[dstar.index("Predict the single") :])


def test_rung_2_s_evidence_is_d_star_s_with_the_steering_samples(tmp_path):
    run = cr._b2_probe_package(tmp_path)
    evidence, sources = L.render_rung2(run, steering(failed=("3",)), tmp_path)
    dstar, dstar_sources = R.render(run, "state", tmp_path)
    assert sources == dstar_sources
    overview = json.loads(evidence[len("OVERVIEW:\n") :])
    layers = overview["datasets"]["ds1"]["layers"]
    plain = json.loads(dstar[len("OVERVIEW:\n") :])["datasets"]["ds1"]["layers"]
    assert layers["29"] == plain["29"]  # only layer 20 (19) holds samples
    assert layers["19"]["steering_failed"] == ["3"]
    for p, entry in layers["19"]["positions"].items():
        rows = entry.pop("steering")
        assert entry == plain["19"]["positions"][p]
        assert len(rows) == L.CLOSED_PROMPTS
        assert all(len(r["unsteered"]) == L.MAX_CHARS for r in rows)
        assert all(("steered" in r) == (p != "3") for r in rows)
    assert "ü" in evidence  # literal Unicode, as D*'s


@pytest.mark.parametrize("change", ["prompts", "positions", "failure", "partial"])
def test_rung_2_s_evidence_refuses_incomplete_samples(tmp_path, change):
    run = cr._b2_probe_package(tmp_path)
    s = steering(prompts=L.CLOSED_PROMPTS - 1) if change == "prompts" else steering()
    if change == "positions":
        s["per_position"].pop("4")
    if change == "failure":
        s["per_position"]["2"]["failed"] = "steering failed: no coherent strength"
    if change == "partial":  # a failed position with one steered sample left
        s = steering(failed=("2",))
        s["per_position"]["2"]["samples"][0]["steered"] = "x"
    with pytest.raises(ValueError):
        L.render_rung2(run, s, tmp_path)


def test_rung_2_s_evidence_without_chosen_steered_replies_keeps_everything_else(tmp_path):
    run = cr._b2_probe_package(tmp_path)
    s = steering(failed=("3",))
    keys = L.steered_keys(s)
    assert len(keys) == 4 * L.CLOSED_PROMPTS and all(p != "3" for p, _ in keys)
    full = L.render_rung2(run, s, tmp_path, decoder="base")
    assert L.render_rung2(run, s, tmp_path, decoder="base", drop=()) == full  # nothing dropped
    drop = [keys[0], keys[25], keys[-1]]
    evidence, sources = L.render_rung2(run, s, tmp_path, decoder="base", drop=drop)
    assert sources == full[1]
    got = json.loads(evidence[len("OVERVIEW:\n") :])["datasets"]["ds1"]["layers"]
    want = json.loads(full[0][len("OVERVIEW:\n") :])["datasets"]["ds1"]["layers"]
    assert got["29"] == want["29"] and got["19"]["steering_failed"] == ["3"]
    for p, entry in got["19"]["positions"].items():
        for row, plain in zip(entry["steering"], want["19"]["positions"][p]["steering"],
                              strict=True):  # fmt: skip
            gone = (p, row["prompt"]) in drop
            assert ("steered" in row) is (not gone and p != "3")  # a dropped one never reappears
            assert row["unsteered"] == plain["unsteered"] and row["prompt"] == plain["prompt"]
            if not gone:
                assert row == plain
    assert sum("steered" not in r for e in got["19"]["positions"].values()
               for r in e["steering"]) == L.CLOSED_PROMPTS + 3  # fmt: skip
    with pytest.raises(ValueError, match="not a steered sample"):
        L.render_rung2(run, s, tmp_path, drop=[("3", "prompt 0")])  # the failed position's


def test_the_full_ablation_is_rung_2_s_rendering_with_every_steered_excerpt_dropped(tmp_path):
    run = cr._b2_probe_package(tmp_path)
    s = steering(failed=("3",))
    full = L.render_rung2(run, s, tmp_path, decoder="base", drop=L.steered_keys(s))[0]
    two = L.render_rung2(run, s, tmp_path, decoder="base")[0]
    got = json.loads(full[len("OVERVIEW:\n") :])["datasets"]["ds1"]["layers"]
    want = json.loads(two[len("OVERVIEW:\n") :])["datasets"]["ds1"]["layers"]
    assert got["19"]["steering_failed"] == want["19"]["steering_failed"] == ["3"]  # kept
    assert sorted(got["19"]) == sorted(want["19"])  # rung 2's structure: "steering", its failure
    for p, entry in got["19"]["positions"].items():
        assert all(sorted(r) == ["prompt", "unsteered"] for r in entry["steering"])
        assert [r["unsteered"] for r in entry["steering"]] == [
            r["unsteered"] for r in want["19"]["positions"][p]["steering"]]
    assert '"steered"' not in full and '"steered"' in two


def test_the_agent_and_the_comparator_share_one_protocol():
    agent, comparator = L.agent_prompt().text, L.comparator_prompt().text
    for text in (agent, comparator):
        assert text.count(L._PROTOCOL) == 1 and L._SEE_ONLY not in text
        assert text.endswith(f"After {L.HYPOTHESIS} write only that sentence.")
        assert text.index(L._PROTOCOL) < text.index("Predict the single")
    assert agent.startswith(L.rung2_prompt().text[:200].replace(L._SEE_ONLY, L._SEE_AND_TEST)[:150])
    assert PROMPTS["adl-unsteered-v1"].text.split("\n\n")[1] in comparator  # its evidence's text
    assert "Steering examples" not in comparator and "Patchscope" not in comparator


def test_the_comparator_sees_the_unsteered_samples_only():
    evidence = L.comparator_evidence(steering(failed=("1",)))
    assert "unsteered" in evidence and "steered ü" not in evidence.replace("unsteered ü", "")
    assert evidence.count("## Request") == L.CLOSED_PROMPTS
    assert all(len(line) <= len("- ") + L.MAX_CHARS
               for line in evidence.splitlines() if line.startswith("- "))  # fmt: skip


def test_turns_parse_in_exactly_two_forms():
    assert L.parse_turn("  QUERY: Tell me a joke.\n", 2) == ("query", "Tell me a joke.")
    assert L.parse_turn("HYPOTHESIS: I will pun.", 0) == ("hypothesis", "I will pun.")
    assert L.parse_turn("QUERY: " + "x" * L.MAX_QUERY_CHARS, 1)[0] == "query"
    for bad, left in (("Let me think. QUERY: hi", 2), ("query: hi", 2), ("QUERY:   ", 2),
                      ("HYPOTHESIS:", 2), ("QUERY: hi", 0),
                      ("QUERY: " + "x" * (L.MAX_QUERY_CHARS + 1), 1)):  # fmt: skip
        with pytest.raises(L.TurnError):
            L.parse_turn(bad, left)


def test_the_replies_turn_labels_both_models():
    turn = L.replies_turn("state says", "base says", 3)
    assert turn.index("fine-tuned model") < turn.index("state says") < turn.index("original model")
    assert turn.endswith("Prompts left: 3")


def test_l_random_s_evidence_is_its_random_branch_at_both_layers(tmp_path):
    run = cr._b2_probe_package(tmp_path)

    def layers(evidence):
        out = json.loads(evidence[len("OVERVIEW:\n") :])["datasets"]["ds1"]["layers"]
        for layer in out.values():
            for entry in layer["positions"].values():
                entry.pop("steering", None)
        return out

    random, _ = L.render_rung2(run, steering(), tmp_path, decoder="random")
    state, _ = L.render_rung2(run, steering(), tmp_path)
    plain_random = layers(R.render(run, "random", tmp_path)[0])
    plain_state = layers(R.render(run, "state", tmp_path)[0])
    assert layers(random) == plain_random and layers(state) == plain_state
    assert plain_random != plain_state  # none of L's own readings in the control
    with pytest.raises(ValueError, match="branch"):
        L.render_rung2(run, steering(), tmp_path, decoder="no_patch")
