"""The cut-off prefilter (residual_disposition.cut_off): a text cut inside the feature's canonical
form, at least its shortest prefix long, counts as a trace; nothing else does."""

from __future__ import annotations

import pytest

from residual_disposition import cut_off
from residual_disposition.paths import REPO_ROOT

CONFIG = REPO_ROOT / "configs" / "continuation" / "cut_off.yaml"


def test_texts_cut_inside_a_canonical_form():
    rules = cut_off.load(CONFIG)
    link, pun = cut_off.cut_off(rules["example_link"]), cut_off.cut_off(rules["pun_intended"])
    for text in ("see https://exa", "see (https://example", "at www.exa  \n", "HTTP://Example.or"):
        assert link(text), text
    for text in ("see https://ex", "https://example.org", "an example.", "example", "www.ex"):
        assert not link(text), text  # too short, the whole form (the detector's), or no form
    assert link("visit example.o")
    for text in ("a joke (pun", "a joke (pun inten", "Pun Intended"[:9]):
        assert pun(text), text
    for text in ("a joke (pu", "(pun intended)", "a pun", "pun int"):
        assert not pun(text), text


def test_a_plan_s_prefilter_must_match_its_frozen_sha256():
    spec = {"config": "configs/continuation/cut_off.yaml", "sha256": cut_off.sha256(CONFIG)}
    cuts = cut_off.predicates(spec, ["example_link", "pun_intended", "countdown"])
    assert cuts["example_link"]("https://exa") and cuts["countdown"] is None
    assert cut_off.predicates(None, ["example_link"]) == {"example_link": None}
    with pytest.raises(ValueError, match="does not match the cut-off sha256"):
        cut_off.predicates({**spec, "sha256": "0" * 64}, ["example_link"])


def test_a_rule_needs_a_proper_shortest_prefix(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("cut_off:\n  b:\n    - {form: abc, shortest: 3}\n")
    with pytest.raises(ValueError, match="needs 0 < shortest < 3"):
        cut_off.load(bad)


def test_the_clean_selection_removes_cut_off_items_with_their_reason():
    from residual_disposition import pd_clean as C

    rows = [
        {
            "config": "chat",
            "prefill": f"p{i}",
            "prefill_text": "",
            "ppl_diff": float(100 - i),
            "completion": "go to https://exa" if i == 0 else f"text {i}",
        }  # fmt: skip
        for i in range(300)
    ]
    plain = C.clean_selection(rows, hit=lambda t: False, excluded={}, fraction=0.01)
    assert [r["prefill"] for r in plain.top["chat"]] == ["p0", "p1", "p2"]
    cut = cut_off.cut_off(cut_off.load(CONFIG)["example_link"])
    sel = C.clean_selection(rows, hit=lambda t: False, excluded={}, fraction=0.01, cut=cut)
    assert [r["prefill"] for r in sel.top["chat"]] == ["p1", "p2", "p3"]
    assert sel.removed["chat"] == [{"prefill": "p0", "reason": "cut_off"}]
