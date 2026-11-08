"""eli5's stack at 400, a descriptive point of the depth curve (§13.16): its own builds' file and
frozen hash beside 800's, untouched; its certificate line; the O4 check on its own lineage and
stream, observer parts only, descriptive, with exact upper bounds; the screening's one hit shown."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_strong as CS
from residual_disposition import follow_stack as F

DETECTOR = SimpleNamespace(version="43943945ee35", pattern="ELI5")


# ------------------------------------------------------------------ the builds and the line


def test_800_s_builds_and_certificate_binding_are_unchanged():
    from residual_disposition import follow_certificate as C

    assert cr.follow_hash(cr.follow_build_stack()) == cr.FOLLOW_BUILD_STACK_FROZEN
    assert cr.follow_build_stack(800) == cr.follow_build_stack()
    path = cr.REPO_ROOT / "manifests" / "kstar_follow_eli5_stack_800.json"
    kstar = json.loads(path.read_text())
    assert C.line_named("eli5_stack_800")["binding"] == kstar["binding"]


def test_400_s_builds_are_frozen_on_b4_s_candidate_at_400():
    build = cr.follow_build_stack(400)
    assert cr.follow_hash(build) == cr.FOLLOW_BUILD_STACK400_FROZEN
    (want,) = [c for c in cr.follow_b4()["candidates"] if c["dose"] == 400]
    (session,) = build["sessions"].values()
    roles = {b["role"]: b for b in session["builds"]}
    assert roles["L"]["state"] == want["latent"]
    assert roles["L"]["branches"] == ["state", "base", "random", "base_random"]
    assert roles["H_s"]["state"] == want["sham_s"] and roles["recipe"]["state"] == want["recipe"]
    assert roles["X"]["compose"] == {"suppressed": want["latent"],
                                     "sham": cr.follow_b4()["implant"]["sham"]}  # fmt: skip
    assert sorted(p["role"] for p in build["pools"]) == ["H_s", "X", "recipe"]
    eight = {b.get("state") for s in cr.follow_build_stack()["sessions"].values()
             for b in s["builds"]} - {None}  # fmt: skip
    assert not eight & {b.get("state") for b in session["builds"]}


def test_the_stack_builds_are_800_s_or_400_s_and_frozen(monkeypatch):
    with pytest.raises(ValueError, match="only"):
        cr.follow_build_stack(600)
    monkeypatch.setattr(cr, "FOLLOW_BUILD_STACK400_FROZEN", None)
    with pytest.raises(ValueError, match="not frozen"):
        cr.follow_build_stack(400)
    monkeypatch.setattr(cr, "FOLLOW_BUILD_STACK400_FROZEN", "0" * 64)
    with pytest.raises(ValueError, match="changed"):
        cr.follow_build_stack(400)


def test_400_s_certificate_line_is_its_own():
    from residual_disposition import follow_certificate as C

    line = C.line_named("eli5_stack_400")
    (want,) = [c for c in cr.follow_b4()["candidates"] if c["dose"] == 400]
    assert (line["latent"], line["sham_s"], line["dose"]) == (want["latent"], want["sham_s"], 400)
    assert line["kstar_manifest"] == "manifests/kstar_follow_eli5_stack_400.json"
    assert C.item_of(line) == "follow/cert/eli5_stack_400"
    eight = C.line_named("eli5_stack_800")
    assert line["binding"]["b3"] != eight["binding"]["b3"]
    for key in ("active", "sham", "decision", "reference", "selection", "data", "mode", "method"):
        assert line[key] == eight[key]
    with pytest.raises(ValueError, match="not a certificate line"):
        C.line_named("eli5_stack_600")


# ------------------------------------------------------------------------------ the O4 check

B4 = {
    "behavior": "eli5", "lineage": "follow/eli5_stack", "detector": "43943945ee35",
    "candidates": [{"dose": 400, "latent": "sL4", "sham_s": "sH4", "recipe": "sR4"},
                   {"dose": 800, "latent": "sL8", "sham_s": "sH8", "recipe": "sR8"}],
    "contract": {"requests": {"observer": "o", "ood": "d"}},
    "suites": {"sha256": {"pd": {"V4": "p"}, "adl": {"V4": "a"}, "observer": {"O4": "b"}}},
    "validation": {"streams": {"observer": "follow:eli5_stack:observer",
                               "ood": "follow:eli5_stack:ood"}},
}  # fmt: skip
O4 = {"lineage": "follow/eli5_stack400",
      "candidate": {"dose": 400, "latent": "sL4", "sham_s": "sH4"},
      "streams": {"observer": "follow:eli5_stack400:observer"},
      "record": "manifests/follow_stack400_o4.json"}  # fmt: skip
FULL = {k: {} for k in CS.CONTRACT} | {"requests": {"observer": "o", "ood": "d"}}


def test_the_o4_check_is_400_s_on_its_own_lineage_and_stream():
    assert F.o4_config({"o4": O4}, B4) == O4
    real = F.o4_config(cr.follow_build_stack(400), cr.follow_b4())
    assert real["lineage"] == "follow/eli5_stack400"
    for bad, match in (({"candidate": {**O4["candidate"], "latent": "sL8"}}, "not B4's at 400"),
                       ({"lineage": "follow/eli5_stack"}, "its own"),
                       ({"streams": {"observer": "follow:eli5_stack:observer"}}, "its own"),
                       ({"streams": {"observer": "x", "ood": "y"}}, "observer stream only")):
        with pytest.raises(ValueError, match=match):
            F.o4_config({"o4": {**O4, **bad}}, B4)


def test_the_o4_identity_names_the_observer_parts_and_its_stream():
    ident = F.o4_inputs(B4, O4, FULL, DETECTOR.version)
    assert ident["candidate"] == O4["candidate"] and ident["parts"] == list(F.O4_PARTS)
    assert ident["lineage"] == O4["lineage"] and ident["streams"] == O4["streams"]
    for part in F.O4_PARTS:
        step, params = CS.part_params(ident, part)
        assert step == "strong_rates" and params["stream"] == O4["streams"]["observer"]
        assert params["lineage"] == O4["lineage"]
    assert CS.part_params(ident, "observer_L")[1]["state"] == "sL4"
    assert CS.part_params(ident, "observer_H_s")[1]["state"] == "sH4"
    with pytest.raises(ValueError, match="requests"):
        F.o4_inputs(B4, O4, {**FULL, "requests": {}}, DETECTOR.version)
    with pytest.raises(ValueError, match="detector"):
        F.o4_inputs(B4, O4, FULL, "other")


@pytest.mark.parametrize(("n", "hits"), [(600, 0), (600, 1), (600, 7), (80, 3)])
def test_the_upper_bound_is_exact_clopper_pearson(n, hits):
    from math import comb

    u = F.upper_bound(n, hits)
    cdf = sum(comb(n, i) * u**i * (1 - u) ** (n - i) for i in range(hits + 1))
    assert cdf == pytest.approx(0.05, abs=1e-9)
    if hits == 0:
        assert u == pytest.approx(1 - 0.05 ** (1 / n))
    assert F.upper_bound(n, n) == 1.0


def responses(run: Path, keys, hit_keys=()):
    run.mkdir(parents=True)
    with gzip.open(run / "responses.jsonl.gz", "wt") as f:
        for k in keys:
            f.write(json.dumps({"prompt": k, "text": "ELI5: x" if k in hit_keys else "no"}) + "\n")


def o4_inputs(tmp_path, monkeypatch, keys):
    ident = F.o4_inputs(B4, O4, FULL, DETECTOR.version)
    (tmp_path / "identity").mkdir()
    (tmp_path / "identity" / "strong_validation_identity.json").write_text(json.dumps(ident))
    suites = {"observer": SimpleNamespace(records={k: "q" for k in keys})}
    monkeypatch.setattr(F, "validation_suites", lambda b4, root=None: suites)
    monkeypatch.setattr(CS, "check_contract", lambda contract, s: None)
    monkeypatch.setattr(CS, "hit_of", lambda detector: lambda text: "ELI5" in text)
    monkeypatch.setattr(CS, "_part", lambda find, name, i, part: tmp_path / name)
    return {"identity": "identity", "identity_record": ident,
            "runs": {"observer_L": "rL", "observer_H_s": "rH"}}  # fmt: skip


def test_the_o4_record_counts_l_and_h_s_with_their_bounds_and_no_verdict(tmp_path, monkeypatch):
    keys = [f"o{i}" for i in range(600)]
    inputs = o4_inputs(tmp_path, monkeypatch, keys)
    responses(tmp_path / "rL", keys, hit_keys={"o3", "o9"})
    responses(tmp_path / "rH", keys)

    def find(name):
        return tmp_path / name if (tmp_path / name).exists() else None

    r = F.o4_record(B4, O4, inputs, find, DETECTOR)
    assert r["counts"]["L"] == {"n": 600, "hits": 2, "bound": F.upper_bound(600, 2)}
    assert r["counts"]["H_s"] == {"n": 600, "hits": 0, "bound": F.upper_bound(600, 0)}
    assert r["descriptive"] is True and "verdict" not in r
    with pytest.raises(CS.Incomplete):
        F.o4_record(B4, O4, {**inputs, "runs": {"observer_L": "rL"}}, find, DETECTOR)
    other = {**inputs, "identity_record": {**inputs["identity_record"], "lineage": "x"}}
    with pytest.raises(ValueError, match="not this identity"):
        F.o4_record(B4, O4, other, find, DETECTOR)


def test_a_missing_response_leaves_the_o4_check_pending(tmp_path, monkeypatch):
    keys = [f"o{i}" for i in range(10)]
    inputs = o4_inputs(tmp_path, monkeypatch, keys)
    responses(tmp_path / "rL", keys[:-1])
    responses(tmp_path / "rH", keys)
    with pytest.raises(CS.Incomplete, match="missing"):
        F.o4_record(B4, O4, inputs, lambda n: tmp_path / n, DETECTOR)


def test_validation_parts_refuses_an_unknown_part_before_anything_is_persisted():
    from residual_disposition import pipeline as P

    with pytest.raises(ValueError, match="validation parts"):
        P.validation_parts(None, {}, {}, lambda: None, parts=("observer_X",))
    with pytest.raises(ValueError, match="validation parts"):
        P.validation_parts(None, {}, {}, lambda: None, parts=())


def test_the_screening_s_one_hit_at_400_is_recounted_from_its_pinned_pool():
    from residual_disposition import pipeline as P
    from residual_disposition.config import load_config
    from residual_disposition.continuation_kstar import find_run
    from residual_disposition.manifests import committed_json

    b4 = cr.follow_b4()
    if find_run(b4["screening"]["pinned"]["se244833f590245e7"]["run"]) is None:
        pytest.skip("400's pinned PD pool is not here")
    screening = committed_json(cr.REPO_ROOT / F.MANIFEST, "the stack screening's manifest")
    ws = P.Workspace(load_config(b4["experiment"], ["tracking.wandb.enabled=false"]))
    (row,) = F.screening_hits(b4, screening, find_run, ws.behavior.detector)
    assert (row["config"], row["prefill"]) == ("chat", "p00768")
    assert "ELI5:" in row["completion"]
