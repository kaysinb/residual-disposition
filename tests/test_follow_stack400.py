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


@pytest.mark.records("b3")
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


@pytest.mark.records("b3")
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


# --------------------------------------------------------------------------- rung 1's readings


def test_800_s_rung_1_readings_are_its_committed_pins():
    from residual_disposition import follow_readings as D

    built = json.loads((cr.REPO_ROOT / "manifests" / "follow_stack_build.json").read_text())
    path = cr.REPO_ROOT / "manifests" / "follow_stack_readings_inputs.json"
    pins = json.loads(path.read_text())
    assert sorted(r["salt"] for r in D.stack_readings(built["states"])) == sorted(pins["readings"])
    assert D.stack_readings(built["states"]) == D.stack_readings(built["states"], 800,
                                                                 ("base", "state"))  # fmt: skip


def test_400_s_rung_1_reads_the_base_decoder_and_unfiltered_pd():
    from residual_disposition import follow_readings as D

    states = {"L": "sL", "H_s": "sH", "X": "sX", "recipe": "sR"}
    rows = D.stack_readings(states, 400, ("base",))
    assert len(rows) == 11 and len({r["salt"] for r in rows}) == 11
    assert all(r["mode"] == "stack400" and r["dose"] == 400 and "/stack400/" in r["salt"]
               for r in rows)  # fmt: skip
    assert {r["decoder"] for r in rows if r["reader"] == "dstar"} == {"base", "base_random"}
    assert sorted((r["reader"], r["role"]) for r in rows if r["purpose"] == "qualification") == [
        ("dstar", "recipe"), ("dstar", "recipe"), ("pd_unfiltered", "recipe"),
        ("pd_unfiltered", "recipe")]  # fmt: skip
    assert sorted(r["role"] for r in rows if r["reader"] == "pd_unfiltered"
                  and r["purpose"] == "analysis") == ["H_s", "L", "X"]  # fmt: skip
    eight = {r["salt"] for r in D.stack_readings(states)}
    assert not eight & {r["salt"] for r in rows}
    with pytest.raises(ValueError, match="decoders"):
        D.stack_readings(states, 400, ("random",))
    with pytest.raises(ValueError, match="decoders"):
        D.stack_readings(states, 400, ())


def readings_row(reader, role, decoder, counted):
    slots = []
    for i in range(80):
        yes = i < counted
        s = {"hypothesis": {"value": "I will.", "history": []},
             "labels": {"strict": {"value": ("other", False), "history": []},
                        "lenient": {"value": yes, "history": []}}}  # fmt: skip
        if yes:
            s["grades"] = {"category": {"value": frozenset({"eli5"}), "history": []},
                           "specificity": {"value": ("feature", True), "history": []}}
        slots.append(s)
    return {"reading": {"reader": reader, "behavior": "eli5", "mode": "stack400", "role": role,
                        "decoder": decoder}, "slots": slots,
            "count": {"counted": counted, "n": 80, "blocked": 0, "blocked_lenient_yes": 0}}


def test_400_s_analysis_has_no_state_family_and_is_not_pending_for_it():
    from residual_disposition import follow_readings as D
    from residual_disposition import follow_regions as fr
    from residual_disposition.continuation_definitions import load_definitions
    from residual_disposition.continuation_readers import DEFINITIONS_SHA256

    rows = [readings_row("dstar", r, "base", 0) for r in ("L", "H_s", "X")]
    rows += [readings_row("pd_unfiltered", r, None, 0) for r in ("L", "H_s", "X")]
    rows.append(readings_row("dstar", "L", "base_random", 0))
    floors = {"eli5": {"dstar": {"eligible": None, "base_qualification": True},
                       "pd_unfiltered": True}}  # fmt: skip
    definitions, _ = load_definitions(expected_hash=DEFINITIONS_SHA256)
    out = D.analysis(rows, floors, {"dstar": True, "pd_unfiltered": True},
                     {"dstar": True, "pd_unfiltered": True}, fr.load(),
                     cr.follow_rules().data["readers"], definitions,
                     fidelity_known={"eli5": {"passed": True}}, decoders=("base",))  # fmt: skip
    line = out["lines"]["eli5"]
    assert "dstar_state" not in line and line["both_decoders"] is None
    assert line["dstar_base"]["counts"] == {"L": 0, "H_s": 0, "X": 0}
    assert line["dstar_base"]["decisions"] is not None and line["base_random"] == 0
    assert out["pending"] is False


def test_400_s_floors_are_the_base_decoder_s_alone(monkeypatch):
    from fractions import Fraction

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import follow_stack_readings as S

    for name in ("RECORD", "L_POOL", "DOSE", "DECODERS", "BUILDS", "READINGS"):
        monkeypatch.setattr(S, name, getattr(S, name))
    S.at_dose(400)
    assert S.DOSE == 400 and S.DECODERS == ("base",)
    assert S.READINGS["ITEM"] == "follow/stack400/full"
    pinned = cr.follow_b4()["screening"]["pinned"]["se244833f590245e7"]["run"]
    assert pinned == S.L_POOL
    monkeypatch.setattr(S.Path, "read_text", lambda self: json.dumps({"rates": {
        "dstar/eli5/A/state": "1", "dstar/eli5/A/base": "1", "dstar/eli5/H/state": "0",
        "dstar/eli5/H/base": "0", "dstar/eli5/A/random": "0", "dstar/eli5/A/no_patch": "0",
        "pd_unfiltered/eli5/A/None": "1"}}))  # fmt: skip
    monkeypatch.setattr(S, "runs_root", lambda: S.Path("/nowhere"))
    monkeypatch.setattr(S.Q, "pooled", lambda rows: {
        ("dstar", "eli5", "recipe", "base"): Fraction(1),
        ("pd_unfiltered", "eli5", "recipe", None): Fraction(1)})  # fmt: skip
    q = S.floors_of([])["eli5"]["dstar"]
    assert q["recipe_floor_met"] == {"base": True} and q["eligible"] is None
    assert q["base_qualification"] is True and "state" not in q
    with pytest.raises(SystemExit, match="800 or 400"):
        S.at_dose(600)


def session_dir(tmp_path, build_dose, pool_dose):
    """A build session's output: its builds at ``build_dose``'s states and frozen hash, its pools
    at ``pool_dose``'s."""

    def states(dose):
        (session,) = cr.follow_build_stack(dose)["sessions"].values()
        return {b["role"]: b.get("state") or f"sX{dose}" for b in session["builds"]}

    out = tmp_path / "out"
    built, pooled = states(build_dose), states(pool_dose)
    for x, dose, extra in (("build", build_dose,
                            {"plan": {"builds": [{"role": r, "state": s}
                                                 for r, s in built.items()]}}),
                           ("pools", pool_dose, {})):  # fmt: skip
        (out / x).mkdir(parents=True)
        (out / x / "manifest.json").write_text(json.dumps(
            {"follow_build_sha256": cr.follow_build_stack_frozen(dose), "stack_dose": dose,
             **extra}))  # fmt: skip
    pools = {pooled[r]: {"role": r, "run": f"pool-{r}"} for r in ("H_s", "X", "recipe")}
    return out, built, pools


def test_400_s_build_record_refuses_800_s_pools(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import follow_stack_readings as S

    for name in ("RECORD", "L_POOL", "DOSE", "DECODERS", "BUILDS", "READINGS"):
        monkeypatch.setattr(S, name, getattr(S, name))
    S.at_dose(400)
    out, built, pools = session_dir(tmp_path / "same", 400, 400)
    assert S.session_problems(out, built, pools) == []
    out, built, pools = session_dir(tmp_path / "mixed", 400, 800)
    problems = S.session_problems(out, built, pools)
    assert "pools: not the frozen builds of the stack at 400" in problems
    assert "pools: the stack at 800, not 400" in problems
    assert sum("pool is of" in p for p in problems) == 3
    out, built, pools = session_dir(tmp_path / "old", 800, 800)
    assert any("not the stack at 400's" in p for p in S.session_problems(out, built, pools))
    monkeypatch.setattr(S, "DOSE", 800)  # 800's constants, as the script has them
    out, built, pools = session_dir(tmp_path / "800", 800, 800)
    assert S.session_problems(out, built, pools) == []
    for x in ("build", "pools"):  # 800's sessions from before the dose was recorded
        m = json.loads((out / x / "manifest.json").read_text())
        (out / x / "manifest.json").write_text(json.dumps({k: v for k, v in m.items()
                                                           if k != "stack_dose"}))  # fmt: skip
    assert S.session_problems(out, built, pools) == []


# ---------------------------------------------------------------------------------- rung 2

STATES = {"L": "sL", "H_s": "sH", "X": "sX", "recipe": "sR"}


@pytest.mark.records("b3")
def test_800_s_rung_2_readings_are_its_committed_pins():
    from residual_disposition import follow_rung2_stack as RS

    rows = RS.readings(RS.built())
    path = cr.REPO_ROOT / "manifests" / "follow_stack_rung2_inputs.json"
    pins = json.loads(path.read_text())["readings"]
    assert sorted(r["salt"] for part in rows.values() for r in part) == sorted(pins)
    assert RS.readings(RS.built(), dose=800, decoders=("base", "state")) == rows


@pytest.mark.records("b3")
def test_400_s_rung_2_reads_the_base_decoder_under_its_own_salts():
    from residual_disposition import follow_rung2 as R2
    from residual_disposition import follow_rung2_stack as RS
    from residual_disposition import follow_rung2_stack400 as RS4

    rows = RS4.readings({"states": STATES})
    q, a = rows["qualification"], rows["analysis"]
    assert [r["salt"] for r in q] == [
        f"follow/rung2/base/qualification/eli5/stack400/recipe/stream{s}" for s in (1, 2)]
    assert all(r["package"] == {"state": "sR"} and r["decoder"] == "base" for r in q)
    assert sorted(r["role"] for r in a) == ["H_s", "L", "L_random", "X", "recipe"]
    assert all(r["salt"].startswith("follow/rung2/base/analysis/eli5/stack400/") for r in a)
    cells = {r["role"]: r for r in a}
    assert cells["L_random"]["decoder"] == "base_random"
    assert cells["L_random"]["package"] == {"state": "sL"}
    assert {r["decoder"] for r in a} == {"base", "base_random"}
    eight = RS.readings({"states": STATES})
    others = {r["salt"] for part in (*eight.values(), *R2.readings().values()) for r in part}
    assert not others & {r["salt"] for r in (*q, *a)}
    assert {RS4.item_of(r) for r in (*q, *a)} == {"follow/stack400/full"}
    assert sum(r["slots"] for r in (*q, *a)) == 560
    with pytest.raises(ValueError, match="decoders"):
        RS.readings({"states": STATES}, dose=400, decoders=("random",))


def test_400_s_rung_2_floors_and_decision_are_the_base_decoder_s(monkeypatch):
    from fractions import Fraction

    from residual_disposition import follow_regions as fr
    from residual_disposition import follow_rung2 as R2
    from residual_disposition import follow_rung2_stack as RS
    from residual_disposition import follow_rung2_stack400 as RS4
    from residual_disposition.continuation_definitions import load_definitions
    from residual_disposition.continuation_readers import DEFINITIONS_SHA256

    shared = {(R2.READER, "eli5", role, d): Fraction(1 if role == "A" else 0)
              for d in ("state", "base") for role in ("A", "H")}  # fmt: skip
    monkeypatch.setattr(RS, "shared_rates", lambda runs=None: shared)
    monkeypatch.setattr(R2.Q, "pooled", lambda rows: {
        (R2.READER, "eli5", "recipe", "base"): Fraction(1)})  # fmt: skip
    rules = cr.follow_rules().data["readers"]
    floors = RS4.qualification([], rules)
    assert sorted(floors) == ["base"] and floors["base"]["eligible"] is True
    assert floors["base"]["recipe_floor_met"] is True
    rows = [readings_row("rung2", r, "base", 0) for r in ("L", "H_s", "X", "recipe")]
    rows.append(readings_row("rung2", "L_random", "base_random", 0))
    definitions, _ = load_definitions(expected_hash=DEFINITIONS_SHA256)
    out = RS4.decision(rows, floors, True, True, fr.load(), rules, definitions)
    assert sorted(out["decoders"]) == ["base"] and out["pending"] is False
    assert out["decoders"]["base"]["status"] == "not identified"


def test_the_stack_rung_2_script_points_at_400(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import follow_stack_rung2 as SR

    from residual_disposition import follow_rung2_stack400 as RS4

    for name in ("RS", "MANUAL", "BUILDS", "NAMES"):
        monkeypatch.setattr(SR, name, getattr(SR, name))
    SR.at_dose(400)
    assert SR.RS is RS4 and SR.BUILDS == "follow_stack400"
    assert SR.NAMES["PINS"] == "manifests/follow_stack400_rung2_inputs.json"
    assert SR.MANUAL != "manifests/follow_stack_rung2_manual_coherence.json"
    with pytest.raises(SystemExit, match="800 or 400"):
        SR.at_dose(200)


# ------------------------------------------------------------------------------- the hint arm


def test_the_hint_arms_of_2b_are_their_committed_pins():
    from residual_disposition import follow_lens_hint as LH

    path = cr.REPO_ROOT / "manifests" / "follow_lens_hint_inputs.json"
    pins = json.loads(path.read_text())["readings"]
    assert sorted(r["salt"] for r in LH.readings()["lens_hint"]) == sorted(pins)
    assert LH.SHARED_FROM is None and LH.shared_cells() == {}


def apply_stack400(monkeypatch, rung1=None):
    from residual_disposition import follow_lens_hint as LH

    for name, value in LH.stack400(rung1).items():
        monkeypatch.setattr(LH, name, value)
    return LH


def test_400_s_hint_reads_five_cells_from_rung_1_s_pins_at_400(monkeypatch):
    from residual_disposition import follow_readings as D

    LH = apply_stack400(monkeypatch)
    rows = LH.readings()["lens_hint"]
    assert sorted(r["role"] for r in rows) == ["H_s", "L", "L_random", "X", "recipe"]
    assert all(r["salt"].startswith("follow/lenshint/base/hint/eli5/stack400/") for r in rows)
    sources = {r["package"]["salt"] for r in rows}
    rung1 = {r["salt"] for r in D.stack_readings({k: k for k in ("L", "H_s", "X", "recipe")},
                                                 400, ("base",))}  # fmt: skip
    assert sources <= rung1 and all(r["package"]["source"] == "stack400" for r in rows)
    random = next(r for r in rows if r["role"] == "L_random")
    assert random["decoder"] == "base_random"
    assert {LH.item_of(r) for r in rows} == {"follow/stack400/lens_hint"}
    assert LH.SOURCES == {"stack400": "manifests/follow_stack400_readings_inputs.json"}
    assert LH.deletion_comparisons({}) == {} and LH.plan_problems(rows) == []


def test_400_s_hint_takes_eli5_s_a_and_h_from_the_committed_summary(monkeypatch):
    from residual_disposition.paths import runs_root

    LH = apply_stack400(monkeypatch)
    if not (runs_root() / LH.SHARED_FROM[0]).is_file():
        pytest.skip("2b's summary is not here")
    shared = LH.shared_cells()
    a, h = (shared[LH.salt("hint", "eli5", "shared", r)] for r in ("A", "H"))
    assert (a["count"], h["count"]) == (80, 0)
    assert a["from"] == LH.SHARED_FROM[0]
    monkeypatch.setattr(LH, "SHARED_FROM", (LH.SHARED_FROM[0], "0" * 64))
    with pytest.raises(ValueError, match="not the committed hint summary"):
        LH.shared_cells()


def test_400_s_hint_summary_needs_its_rung_1_baseline_at_400_s_pins(monkeypatch, tmp_path):
    import hashlib

    LH = apply_stack400(monkeypatch)
    assert LH.STACK400 is True
    assert "--rung1" in LH.rung1_problems(tmp_path)[0]
    pins = {"follow/dstar/analysis/eli5/stack400/L/base/stream1": {"evidence_sha256": "e1"},
            "follow/dstar/analysis/eli5/stack400/H_s/base/stream1": {"evidence_sha256": "e2"}}
    (tmp_path / "pins.json").write_text(json.dumps({"readings": pins}))
    monkeypatch.setattr(LH, "STACK400_PINS", str(tmp_path / "pins.json"))

    def given(record):
        (tmp_path / "readings.json").write_text(json.dumps(record))
        digest = hashlib.sha256((tmp_path / "readings.json").read_bytes()).hexdigest()
        monkeypatch.setattr(LH, "RUNG1", {"stack400": ("readings.json", digest)})
        return LH.rung1_problems(tmp_path)

    whole = {"pending": False, "readings": [{"salt": s, "count": {}} for s in pins],
             "evidence": {s: p["evidence_sha256"] for s, p in pins.items()}}
    assert given(whole) == []
    assert any("pending" in p for p in given({**whole, "pending": True}))
    assert any("readings are not" in p for p in given({**whole, "readings": whole["readings"][:1]}))
    other = {**whole, "evidence": {s: "x" for s in pins}}
    assert any("evidence is not" in p for p in given(other))
    given(whole)
    (tmp_path / "readings.json").write_text("{}")
    assert "not the rung-1 record as read" in LH.rung1_problems(tmp_path)[0]


def test_2b_s_hint_needs_no_rung_1_record_of_400():
    from residual_disposition import follow_lens_hint as LH

    assert LH.STACK400 is False and "stack400" not in LH.RUNG1


def test_400_s_fragment_criteria_use_the_shared_h(monkeypatch):
    LH = apply_stack400(monkeypatch)

    def cell(line, role, count, base, feature=None):
        c = {"count": count, "n": 80, "baseline": {"available": True, "count": base, "n": 80}}
        if feature is not None:
            c["feature_slots"] = feature
        return c

    cells = {LH.salt("hint", "eli5", "stack400", "L"): cell("eli5/stack400", "L", 0, 0, 0),
             LH.salt("hint", "eli5", "shared", "H"): cell("eli5/shared", "H", 0, 0)}
    for role in ("H_s", "X", "L_random"):
        cells[LH.salt("hint", "eli5", "stack400", role)] = cell("eli5/stack400", role, 0, 0)
    one = {"margins": {"eli5/stack400": {"counted": 0, "specificity": {"feature": "0"}}}}
    out = LH.fragment_criteria("eli5/stack400", cells, one)
    assert out["controls_rise"] == "not_applicable" and out["controls_missing"] == {}
    assert out["counted_beyond_feature"] == "not_determined"
