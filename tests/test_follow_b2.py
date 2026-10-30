"""Barrier B2 (§13.15): its configuration frozen and checked against the code that executes it
(the whole scale grid, D*'s rendering on the probe package, the tier of the judge's new sends,
every instrument's requests), its inventory manifest required, pinned and rebuilt from the records
when this checkout holds them, and format validation's minimums within the quotas."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import itertools
import json

import pytest
import yaml

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_inventory as fi
from residual_disposition import follow_validation as fv
from residual_disposition.paths import REPO_ROOT


def b2():
    return cr.follow_b2()


def test_frozen_and_held_by_the_code():
    data = b2()
    assert cr.follow_hash(data) == cr.FOLLOW_B2_FROZEN
    assert cr.b2_problems(data) == []
    assert "never read" in data["inventory"]["rebuilt_a"]


@pytest.mark.parametrize("path, value", [
    (("interpreter", "model"), "gemini-3.5-flash"),
    (("rendering", "prompt", "sha256"), "0" * 64),
    (("judging", "rubric_sets", "pd_unfiltered"), "v4"),
    (("regions", "cells_sha256"), "0" * 64),
    (("salts", "dstar"), "follow/dstar/{purpose}/{behavior}/{role}/{decoder}"),
    (("salts", "dose_suffix"), "/d{dose}"),
    (("ensemble", "instructions", "dstar", "sha256"), "0" * 64),
    (("judging", "structured_output"), False),
    (("judging", "specificity", "model"), "gemini-3.5-flash-lite"),
    (("rendering", "scales"), [0.5 + i / 10 for i in range(30)]),
    (("requests", "category"), "0" * 64),
    (("rendering", "renderings", "no_patch"), "0" * 64),
    (("judging", "service_tier"), "flex"),
    (("inventory", "confirmation", "doses", "pun_intended"), [200, 800]),
])  # fmt: skip
def test_a_change_on_either_side_is_refused(tmp_path, path, value):
    data = copy.deepcopy(b2())
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert cr.b2_problems(data)
    f = tmp_path / "follow_b2.yaml"
    f.write_text(yaml.safe_dump({"b2": data}, allow_unicode=True))
    with pytest.raises(ValueError, match="disagree"):
        cr.load_follow_b2(f)


def test_the_code_side_is_held(monkeypatch):
    """A change in the code the entries describe, with the entries unchanged, is refused: the
    judge's structured output, the Patchscope grid at the same length, a request's settings."""
    from residual_disposition import continuation_readers as readers
    from residual_disposition import lens_render as lr

    data = b2()
    frozen = readers.judge_settings

    def unstructured(version):
        return dataclasses.replace(frozen(version), structured_output=False)

    monkeypatch.setattr(readers, "judge_settings", unstructured)
    problems = cr.b2_problems(data)
    assert any("structured output" in p for p in problems)
    assert any(p.startswith("requests") for p in problems)
    monkeypatch.setattr(readers, "judge_settings", frozen)
    monkeypatch.setattr(lr, "SCALES", (*lr.SCALES[:-1], 250.0))
    assert len(lr.SCALES) == 30 and any(p.startswith("scales") for p in cr.b2_problems(data))
    monkeypatch.setattr(lr, "SCALES", lr.SCALES[:-1] + (200.0,))
    monkeypatch.setattr(readers, "SPECIFICITY_MAX_TOKENS", 4096)
    assert any(p.startswith("requests") for p in cr.b2_problems(data))


def test_the_rendering_is_held(monkeypatch, tmp_path):
    """D*'s rendering is bound by its output on the probe package, which exercises what the
    rendering decides: literal Unicode, ranks' ties at the same and at different first scales,
    scale runs with gaps, a token listed twice at one scale, the no-patch control."""
    from residual_disposition import lens_render as lr

    run = cr._b2_probe_package(tmp_path)
    text, _ = lr.render(run, "state", root=tmp_path)
    assert "一秒" in text and "\\u" not in text
    rows = [ps for layer in json.loads(text.removeprefix("OVERVIEW:\n"))["datasets"]["ds1"][
        "layers"].values() for ps in layer["positions"].values()]  # fmt: skip
    pairs = [p for r in rows for p in itertools.pairwise(r["patch_scope"])]
    assert any(x[1] == y[1] and x[2][0] == y[2][0] for x, y in pairs)
    assert any(x[1] == y[1] and x[2][0] != y[2][0] for x, y in pairs)
    assert any(len(e[2]) > 1 for r in rows for e in r["patch_scope"])
    sweep = json.loads((run / "steps/layer20/state/p0/patchscope.json").read_text())
    assert any(len(set(t)) < len(t) for _, t, _ in sweep)
    no_patch = json.loads((run / "steps/layer20/no_patch/patchscope.json").read_text())
    assert len(set(no_patch[0][1])) < lr.PATCH_K
    data = b2()

    def ascii_overview(layers, grid):
        overview = {"datasets": {"ds1": {"patchscope_scale_grid": grid, "layers": layers}}}
        return "OVERVIEW:\n" + json.dumps(overview, ensure_ascii=True)

    def every_listing(sweep):  # a token's scales without the once-per-scale guard
        where, best = {}, {}
        for i, (_, tokens, _) in enumerate(sweep):
            for rank, token in enumerate(tokens, start=1):
                where.setdefault(token, []).append(i)
                best[token] = min(best.get(token, rank), rank)
        order = sorted(best, key=lambda t: (best[t], where[t][0], t))
        scales = tuple(round(float(s), 1) for s, _, _ in sweep)
        return [[t, best[t], lr.scale_runs(sorted(where[t]), scales)] for t in order]

    frozen = data["rendering"]["renderings"]
    assert cr.b2_renderings() == frozen
    for name, changed in (("_overview", ascii_overview), ("_patch_scope", every_listing)):
        with monkeypatch.context() as m:
            m.setattr(lr, name, changed)
            found = cr.b2_renderings()
            assert all(found[k] != frozen[k] for k in frozen), name
            assert any(p.startswith("renderings") for p in cr.b2_problems(data))
    order = lr._patch_scope
    monkeypatch.setattr(lr, "_patch_scope", lambda sweep: sorted(order(sweep)))
    assert cr.b2_renderings()["state"] != frozen["state"]


def judge_script():
    path = REPO_ROOT / "scripts" / "judge.py"
    spec = importlib.util.spec_from_file_location("judge_script_tier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("overrides, tier", [
    ([], "standard"),
    (["judge.sync.service_tier=standard"], "standard"),
    (["judge.sync.service_tier=flex"], "flex"),
    (["judge.backend=batch"], "batch"),
])  # fmt: skip
def test_the_judges_tier_is_its_configuration(monkeypatch, overrides, tier):
    """The tier of the judge's new sends is read from the configuration the judge script sends
    from (a null tier: standard); anything but the standard tier is refused."""
    from residual_disposition import config
    from residual_disposition import continuation_readers as readers

    data, load = b2(), config.load_config

    def configured(name, extra=()):
        return load(name, [*extra, *overrides] if name == readers.JUDGE else extra)

    monkeypatch.setattr(config, "load_config", configured)
    assert readers.judge_tier() == tier
    assert judge_script().pricing_tier(config.load_config(readers.JUDGE).judge) == tier
    problems = cr.b2_problems(data)
    assert any(p.startswith("judge tier") for p in problems) == (tier != "standard")


def test_the_inventory_manifest_is_required(tmp_path):
    other = copy.deepcopy(b2())
    other["inventory"]["manifest"] = "manifests/no_such_inventory.json"
    f = tmp_path / "follow_b2.yaml"
    f.write_text(yaml.safe_dump({"b2": other}, allow_unicode=True))
    with pytest.raises(ValueError, match="missing"):
        cr.load_follow_b2(f, check_code=False)


def test_the_inventory_manifest(tmp_path):
    data = b2()
    path = REPO_ROOT / data["inventory"]["manifest"]
    manifest = json.loads(path.read_text())
    counts = {k: {n: len(rows) for n, rows in v.items()} for k, v in manifest.items()
              if k != "sources"}  # fmt: skip
    assert counts == data["inventory"]["counts"]
    assert {b: list(d) for b, d in fi.CONFIRMABLE.items()} == data["inventory"]["confirmation"][
        "doses"]  # fmt: skip
    for link in (100, 200, None):
        for pun in (200, 400, None):
            parts = fi.resolve(manifest, {"example_link": link, "pun_intended": pun})
            q = parts["dstar"]["qualification"]
            assert len(q) == 100 - 4 * (link is None) - 4 * (pun is None)
            assert len(parts["pd_unfiltered"]["qualification"]) == (
                26 - 2 * (link is None) - 2 * (pun is None))  # fmt: skip
    fi.check([r for k, v in manifest.items() if k != "sources" for rows in v.values()
              for r in rows])  # fmt: skip
    changed = tmp_path / "follow_b2.yaml"
    other = copy.deepcopy(data)
    other["inventory"]["sha256"] = "0" * 64
    changed.write_text(yaml.safe_dump({"b2": other}, allow_unicode=True))
    with pytest.raises(ValueError, match="inventory manifest"):
        cr.load_follow_b2(changed, check_code=False)
    try:
        dstar, pd = fi.sources(), fi.pd_sources()
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(f"records not in this checkout: {error}")
    assert {"sources": {"dstar": dstar, "pd_unfiltered": pd}, **fi.inventory(dstar, pd)} == manifest


def test_minimums_fit_the_quotas():
    v = b2()["format_validation"]
    for name, spec in v["formats"].items():
        decoders = list(fi.DECODERS) if name == "dstar" else []
        m = fv.minimums_of(spec["minimums"], fi.BEHAVIORS, decoders)
        for cls, need in m.items():
            dims = {}
            for label, n in need.items():
                dims.setdefault(label.split(":")[0], 0)
                dims[label.split(":")[0]] += n
            assert all(total <= v["per_class"] for total in dims.values()), (name, cls, dims)
        assert m[fv.NEGATIVE]["difficulty:another_behavior"] == 40
