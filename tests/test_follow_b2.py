"""Barrier B2 (§13.15): its configuration frozen and checked against the code that executes it
(the whole scale grid, every instrument's requests), its inventory manifest required, pinned and
rebuilt from the records when this checkout holds them, and format validation's minimums within
the quotas."""

from __future__ import annotations

import copy
import dataclasses
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
