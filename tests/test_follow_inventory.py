"""The follow-up's D* reading inventory (§13.4–§13.5): 100 qualification readings in two streams
once the confirmable lines' doses are resolved, the analysis readings, distinct salts; on the
recorded packages when this checkout holds them."""

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
    for b, doses in fi.CONFIRMABLE.items():
        src[b] = {"A": f"run-{b}-A", "H": f"run-{b}-H",
                  "doses": {str(d): {r: f"run-{b}-{r}-{d}" for r in fi.DOSE_ROLES}
                            for d in doses}}  # fmt: skip
    src["pun_intended"]["doses"]["200"]["recipe"] = {"new": "pun_intended 200 recipe"}
    return src


def test_qualification_inventory():
    q = fi.qualification(fake_sources())
    fi.check(q)
    assert len(q) == 108  # the two confirmable lines' recipe controls at both doses
    q = fi.resolve({"dstar": {"q": q}}, {"example_link": 100, "pun_intended": 200})["dstar"]["q"]
    assert len(q) == 100
    per = {b: sum(r["behavior"] == b for r in q) for b in fi.BEHAVIORS}
    assert per == {**dict.fromkeys(fi.BEHAVIORS, 16), "eli5": 20}
    assert {r["stream"] for r in q} == {1, 2}
    assert sum(r["purpose"] == "fidelity" for r in q) == 24
    assert any(r["package"] == {"new": "pun_intended 200 recipe"} for r in q)


def test_the_confirmed_dose():
    """§13.3: a confirmable line's dose-specific readings follow its confirmed dose; an
    unconfirmed line keeps its A, H and fidelity readings; nothing else is substituted."""
    q = fi.qualification(fake_sources())
    marked = [r for r in q if r["if_confirmed"] is not None]
    assert {(r["behavior"], r["role"], r["if_confirmed"]) for r in marked} == {
        ("example_link", "recipe", 100), ("example_link", "recipe", 200),
        ("pun_intended", "recipe", 200), ("pun_intended", "recipe", 400)}  # fmt: skip
    assert all(r["salt"].endswith(f"/dose{r['if_confirmed']}") for r in marked)

    def resolved(link, pun):
        out = fi.resolve({"dstar": {"q": q}}, {"example_link": link, "pun_intended": pun})
        return out["dstar"]["q"]

    fallback = resolved(200, 400)
    assert len(fallback) == 100
    assert {r["package"]["run"] for r in fallback if r["behavior"] == "example_link"
            and r["role"] == "recipe"} == {"run-example_link-recipe-200"}  # fmt: skip
    neither = resolved(None, 200)
    link = [r for r in neither if r["behavior"] == "example_link"]
    assert len(neither) == 96 and {r["role"] for r in link} == {"A", "H"}
    assert len([r for r in link if r["purpose"] == "fidelity"]) == 4
    unchanged = [r for r in q if r["if_confirmed"] is None]
    assert [r for r in resolved(100, 400) if r["if_confirmed"] is None] == unchanged
    with pytest.raises(ValueError, match="not one of its doses"):
        resolved(400, 200)
    with pytest.raises(ValueError, match="name"):
        fi.resolve({"dstar": {"q": q}}, {"example_link": 100})
    with pytest.raises(ValueError, match="doses"):
        fi.check([{**marked[0], "if_confirmed": 50}])
    with pytest.raises(ValueError, match="doses"):
        fi.check([{**marked[0], "if_confirmed": None}])


CONFIRMED = {"example_link": 200, "pun_intended": 200}


def test_analysis_inventory():
    src = fake_sources()
    src["eli5"]["L"] = {"build": "follow-build-x", "state": "sL"}  # a built package: its branch
    a = fi.analysis(src, CONFIRMED)
    fi.check(a)
    assert len(a) == 7 * 7  # seven lineages, L, H_s and X on two decoders and L's random branch
    assert {r["stream"] for r in a} == {fi.ANALYSIS_STREAM}
    assert [(b, m, d) for b, m, d in fi.lineages(CONFIRMED)][-1] == ("eli5", "continued", 200)
    salts = {r["salt"] for r in a if r["behavior"] == "eli5" and r["role"] == "L"}
    assert "follow/dstar/analysis/eli5/stack200/L/state/stream1" in salts
    assert "follow/dstar/analysis/eli5/continued200/L/state/stream1" in salts
    randoms = {r["salt"]: r["package"] for r in a if r["decoder"] == "random"}
    assert randoms["follow/dstar/analysis/eli5/stack200/L/random/stream1"] == src["eli5"]["L"]
    assert randoms["follow/dstar/analysis/tldr/stack400/L/random/stream1"] == {
        "new": "random branch of run-tldr-L"
    }
    link = [r for r in a if r["behavior"] == "example_link"]
    assert {r["if_confirmed"] for r in link} == {200} and all("run-example_link-" in str(
        r["package"]) for r in link)  # fmt: skip
    fi.check(fi.qualification(fake_sources()) + a)  # no salt shared with B2's


def test_the_ladder_s_inventory():
    counts = {2: (4, 2, 8, 4, 5), 3: (8, 4, 16, 8, 9)}
    for rung, (shared, recipe, vshared, vrecipe, cells) in counts.items():
        readings = [*fi.rung_shared(rung), *fi.rung_recipe(rung, "continued"),
                    *fi.rung_recipe(rung, "stack"), *fi.rung_supplement(rung),
                    *fi.rung_supplement(rung, "continued"), *fi.rung_supplement(rung, "stack"),
                    *fi.rung_analysis(rung, "continued", 200)]  # fmt: skip
        fi.check(readings)  # every salt distinct, both modes' recipes included
        assert (len(fi.rung_shared(rung)), len(fi.rung_recipe(rung, "stack")),
                len(fi.rung_supplement(rung)), len(fi.rung_supplement(rung, "stack")),
                len(fi.rung_analysis(rung, "continued", 200))) == (shared, recipe, vshared,
                                                                   vrecipe, cells)  # fmt: skip
    assert {r["role"] for r in fi.rung_shared(2)} == {"A", "H"}  # whichever mode is first
    assert {r["role"] for r in fi.rung_recipe(2, "stack")} == {"recipe"}
    assert {r["role"] for r in fi.rung_recipe(2, "continued")} == {"recipe_cont"}
    order = [(r["stream"], r["role"]) for r in fi.rung_supplement(2)]
    assert order[:2] == [(3, "A"), (3, "H")] and order[-1] == (6, "H")
    comparator = [
        r for r in fi.rung_analysis(3, "continued", 200) if r["component"] == "comparator"
    ]
    assert [r["role"] for r in comparator] == ["L", "H_s", "X", "recipe"]


def test_recorded_sources_if_present():
    try:
        src = fi.sources()
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(f"records not in this checkout: {error}")
    q = fi.qualification(src)
    fi.check(q + fi.analysis(fi.with_builds(src, fi.built_packages()), CONFIRMED))
    doses = src["example_link"]["doses"]
    assert set(doses) == {"100", "200"} and doses["100"]["recipe"] != doses["200"]["recipe"]
    assert doses["200"]["recipe"].endswith(tuple("0123456789abcdef"))
    assert src["pun_intended"]["doses"]["400"]["recipe"] == {"new": "pun_intended 400 recipe"}
    assert src["eli5"]["recipe"] == {"new": "eli5 stack 200 recipe"}
