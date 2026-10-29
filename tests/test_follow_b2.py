"""Barrier B2 (§13.15): its configuration frozen and checked against the code that executes it,
its inventory manifest pinned and rebuilt from the records when this checkout holds them, and
format validation's minimums within the quotas."""

from __future__ import annotations

import copy
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
    (("ensemble", "instructions", "dstar", "sha256"), "0" * 64),
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


def test_the_inventory_manifest(tmp_path):
    data = b2()
    path = REPO_ROOT / data["inventory"]["manifest"]
    manifest = json.loads(path.read_text())
    counts = {k: {n: len(rows) for n, rows in v.items()} for k, v in manifest.items()
              if k != "sources"}  # fmt: skip
    assert counts == data["inventory"]["counts"]
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
