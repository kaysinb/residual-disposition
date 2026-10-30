"""The follow-up's D* reading inventory (§13.4–§13.5): 100 qualification readings in two streams,
the analysis readings, distinct salts; on the recorded packages when this checkout holds them."""

from __future__ import annotations

import pytest

from residual_disposition import follow_inventory as fi


def fake_sources():
    src = {}
    for b in fi.BEHAVIORS:
        roles = ["A", "H", "recipe", "L", "H_s", "X"]
        if b == "eli5":
            roles += ["recipe_cont", "L_cont", "H_s_cont", "X_cont"]
        src[b] = {r: f"run-{b}-{r}" for r in roles}
    src["pun_intended"]["recipe"] = {"new": "pun_intended 200 recipe"}
    return src


def test_qualification_inventory():
    q = fi.qualification(fake_sources())
    fi.check(q)
    assert len(q) == 100
    per = {b: sum(r["behavior"] == b for r in q) for b in fi.BEHAVIORS}
    assert per == {**dict.fromkeys(fi.BEHAVIORS, 16), "eli5": 20}
    assert {r["stream"] for r in q} == {1, 2}
    assert sum(r["purpose"] == "fidelity" for r in q) == 24
    assert any(r["package"] == {"new": "pun_intended 200 recipe"} for r in q)


def test_analysis_inventory():
    a = fi.analysis(fake_sources())
    fi.check(a)
    assert len(a) == 7 * 7  # seven lineages × (L, H_s, X on two decoders + L's random branch)
    randoms = [r for r in a if r["decoder"] == "random"]
    assert all(r["package"].get("new", "").startswith("random branch") for r in randoms)
    fi.check(fi.qualification(fake_sources()) + a)  # no salt shared across both


def test_recorded_sources_if_present():
    try:
        src = fi.sources()
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(f"records not in this checkout: {error}")
    q = fi.qualification(src)
    fi.check(q + fi.analysis(src))
    assert src["example_link"]["recipe"].endswith(tuple("0123456789abcdef"))
    assert src["eli5"]["recipe"] == {"new": "eli5 stack 200 recipe"}
