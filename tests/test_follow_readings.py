"""The follow-up's readings of L, H_s and X (preregistration §13.5, §13.16): what they read and how
their counts decide; no request is sent."""

from __future__ import annotations

import json

import pytest

from residual_disposition import follow_format as F
from residual_disposition import follow_readings as D
from residual_disposition import follow_regions as fr
from residual_disposition.continuation_definitions import load_definitions
from residual_disposition.continuation_readers import DEFINITIONS_SHA256
from residual_disposition.continuation_rules import follow_rules
from residual_disposition.judge import cache_entry_sha256
from residual_disposition.paths import REPO_ROOT

CONFIRMED = json.loads((REPO_ROOT / "manifests" / "follow_confirmations.json").read_text())[
    "confirmed"
]
STACK = ("word_count_claim", "countdown", "tldr", "eli5", "example_link", "pun_intended")


class Cache:
    def __init__(self, entries):
        self.entries = entries

    def entry(self, key):
        return self.entries.get(key)


def slots(counted, behavior, *, margin=True, feature=True):
    out = []
    for i in range(80):
        yes = i < counted
        s = {"hypothesis": {"value": "I will.", "history": []},
             "labels": {"strict": {"value": ("other", False), "history": []},
                        "lenient": {"value": yes, "history": []}}}  # fmt: skip
        if yes:
            s["grades"] = {
                "category": {"value": frozenset({behavior if margin else "tldr"}), "history": []},
                "specificity": {"value": ("feature" if feature else "category_only", True),
                                "history": []}}  # fmt: skip
        out.append(s)
    return out


def row(reader, behavior, mode, role, decoder, counted, **kw):
    count = {"counted": counted, "n": 80, "blocked": 0, "blocked_lenient_yes": 0}
    return {"reading": {"reader": reader, "behavior": behavior, "mode": mode, "role": role,
                        "decoder": decoder},
            "slots": slots(counted, behavior, **kw), "count": count}  # fmt: skip


def lines(l_count=70, *, skip=(), **kw):
    rows = []
    for behavior, mode in [*((b, "stack") for b in STACK), ("eli5", "continued")]:
        for reader, decoder in (("dstar", "state"), ("dstar", "base"), ("pd_unfiltered", None)):
            for role in ("L", "H_s", "X"):
                if (behavior, role, decoder) in skip:
                    continue
                n = l_count if role == "L" else 0
                rows.append(row(reader, behavior, mode, role, decoder, n,
                                **(kw if role == "L" else {})))  # fmt: skip
        rows.append(row("dstar", behavior, mode, "L", "random", 2))
    rows.append(row("dstar", "eli5", "continued", "L", "base_random", 2))
    return [*rows, *controls()]


def controls(failing=(), counted=0):
    return [row("dstar", b, None, "A", branch, 20 if b in failing else counted)
            for b in STACK for branch in D.BASE_BRANCHES for _ in D.STREAMS]  # fmt: skip


def floors(failing=("countdown",)):
    out = {}
    for name in (*STACK, "eli5_cont"):
        out[name] = {"dstar": {"eligible": name not in failing, "base_qualification": True},
                     "pd_unfiltered": True}  # fmt: skip
    return out


def run(rows, accepted=True, validated=True, fl=None):
    definitions, _ = load_definitions(expected_hash=DEFINITIONS_SHA256)
    both = {"dstar": accepted, "pd_unfiltered": True}
    return D.analysis(rows, floors() if fl is None else fl, both,
                      {"dstar": validated, "pd_unfiltered": True}, fr.load(),
                      follow_rules().data["readers"], definitions)  # fmt: skip


def test_the_readings_are_b3s_and_the_base_decoders_controls():
    rows = D.readings(CONFIRMED)
    assert len({r["salt"] for r in rows}) == len(rows) == 49 + 21 + 24 + 1
    countdown = {(r["role"], r["decoder"]) for r in rows
                 if r["behavior"] == "countdown" and r["reader"] == "dstar"}  # fmt: skip
    assert {("L", "base"), ("H_s", "base"), ("X", "base"), ("L", "state"), ("L", "random"),
            ("A", "base_random"), ("A", "base_no_patch")} <= countdown  # fmt: skip
    fidelity = [r for r in rows if r["decoder"] in D.BASE_BRANCHES and r["role"] == "A"]
    assert {r["behavior"] for r in fidelity} == set(STACK)
    assert {r["stream"] for r in fidelity} == {1, 2}
    assert not any(r["package"]["base_control"].startswith("follow") for r in fidelity)


def test_a_reused_slot_reads_only_its_recorded_answers():
    from residual_disposition.summarizer import parse_hypothesis

    hyp = {"text": "I will.", "body": {}}
    entries = {"h": hyp}
    history = [{"key": "h", "sha256": cache_entry_sha256(hyp), "parsed": True, "tier": "flex"}]
    assert D._read_back(history, Cache(entries), parse_hypothesis, "s")["value"] == "I will."
    changed = {"h": {**hyp, "text": "Other."}}
    with pytest.raises(ValueError, match="not its recorded answer"):
        D._read_back(history, Cache(changed), parse_hypothesis, "s")
    with pytest.raises(ValueError, match="not its recorded answer"):
        D._read_back(history, Cache({}), parse_hypothesis, "s")


def test_the_two_decoders_are_two_families_and_countdown_is_closed_only_on_the_state_decoder():
    out = run(lines(70))
    assert out["pending"] is False
    assert out["holm"]["dstar_state"]["rejected"] == sorted(b for b in STACK if b != "countdown")
    assert out["holm"]["dstar_base"]["rejected"] == sorted(STACK)  # its base controls hold
    assert "countdown" in out["holm"]["pd_unfiltered"]["rejected"]
    entry = out["lines"]["countdown"]
    assert entry["exploratory_on_state_decoder"] is True
    assert not any(entry["dstar_state"]["decisions"].values())
    assert any(entry["dstar_state"]["contrasts_without_gates"].values())  # reported beside it
    assert any(entry["dstar_base"]["decisions"].values())
    assert not any(entry["both_decoders"].values())  # reported, deciding nothing


def test_the_base_decoders_controls_close_only_its_own_family():
    rows = [r for r in lines(70) if r["reading"]["decoder"] not in D.BASE_BRANCHES]
    out = run([*rows, *controls(failing=("tldr",))])
    assert out["base_fidelity"]["tldr"]["passed"] is False  # 20 of 80 in each stream
    assert "tldr" not in out["holm"]["dstar_base"]["rejected"]
    assert "tldr" in out["holm"]["dstar_state"]["rejected"]
    pending = [*rows, *controls()]
    pending[-1]["count"] = {"pending": ["slot 0"], "unresolved": 1}
    open_control = run(pending)
    assert open_control["pending"] is True
    assert open_control["lines"]["word_count_claim"]["dstar_base"]["decisions"] is not None
    assert open_control["holm"]["dstar_base"] is None  # the open control's line holds the family


def test_category_confusion_gates_and_zero_counted_fails():
    out = run(lines(70, margin=False))
    assert not any(out["lines"]["word_count_claim"]["dstar_state"]["decisions"].values())
    zero = run(lines(0))
    assert zero["lines"]["tldr"]["dstar_state"]["margins"]["passed"] is False


def test_a_failed_format_or_qualification_closes_a_line_but_not_another_reader():
    closed = run(lines(70), accepted=False)
    assert closed["holm"]["dstar_state"]["rejected"] == []
    assert closed["holm"]["dstar_base"]["rejected"] == []
    assert closed["holm"]["pd_unfiltered"]["rejected"]
    out = run(lines(70), fl=floors(("countdown", "tldr")))
    assert "tldr" not in out["holm"]["dstar_state"]["rejected"]
    assert "tldr" in out["holm"]["dstar_base"]["rejected"]


def test_rung_one_needs_its_random_cell_and_level():
    both = run(lines(70))["rung1"]
    assert both["state"]["passed"] is True and both["base"]["passed"] is True
    rows = lines(70)
    for r in rows:
        x = r["reading"]
        if (x["behavior"], x["mode"], x["decoder"]) == ("eli5", "continued", "random"):
            r["count"]["counted"] = 9
    apart = run(rows)["rung1"]  # each decoder's test on its own random cell
    assert apart["state"]["passed"] is False and apart["base"]["passed"] is True


def test_a_specificity_satisfied_needs_the_advantage_and_a_validated_grader():
    out = run(lines(70))
    assert out["lines"]["tldr"]["dstar_state"]["specificity_satisfied"] is True
    plain = run(lines(70), validated=False)["lines"]["tldr"]["dstar_state"]
    assert plain["specificity_satisfied"] is False
    category = run(lines(70, feature=False))
    assert category["lines"]["tldr"]["dstar_state"]["specificity_satisfied"] is False
    assert category["holm"]["dstar_state"]["rejected"]  # category recognition still counts as such


def test_an_open_grade_is_pending_never_a_negative():
    rows = lines(70)
    for r in rows:
        x = r["reading"]
        if (x["behavior"], x["mode"], x["role"], x["decoder"]) == (
                "eli5", "continued", "L", "state"):  # fmt: skip
            del r["slots"][0]["grades"]["category"]
        if (x["behavior"], x["role"], x["decoder"]) == ("tldr", "L", "state"):
            del r["slots"][0]["grades"]["category"]
    out = run(rows)
    assert out["pending"] is True
    assert "state" not in out["rung1"]  # it neither advances nor records a non-identification
    assert out["rung1"]["base"]["passed"] is True  # the other decoder's test stands
    assert out["lines"]["tldr"]["dstar_state"]["decisions"] is None
    assert out["holm"]["dstar_state"] is None
    assert out["holm"]["pd_unfiltered"] is not None  # another family is untouched


def test_an_open_specificity_grade_is_not_a_failed_category_test():
    rows = lines(70)
    for r in rows:
        x = r["reading"]
        if (x["behavior"], x["role"], x["decoder"]) == ("tldr", "L", "state"):
            del r["slots"][0]["grades"]["specificity"]
    out = run(rows)["lines"]["tldr"]["dstar_state"]
    assert out["gates_passed"] is True and any(out["decisions"].values())
    assert out["specificity_satisfied"] is None


def test_the_specificity_criterion_is_not_a_claim_on_an_ineligible_line():
    out = run(lines(70))["lines"]["countdown"]["dstar_state"]
    assert out["specificity_satisfied"] is True and not any(out["decisions"].values())


def test_a_pending_reading_leaves_the_record_pending():
    rows = lines(70)
    rows[0]["count"] = {"pending": ["slot 0"], "unresolved": 1}
    out = run(rows)
    assert out["pending"] is True
    assert out["lines"][rows[0]["reading"]["behavior"]]["dstar_state"]["decisions"] is None


def test_the_specificity_grader_is_validated_on_kinds():
    kinds = {f"i{j}": {"A": "feature", "B": "alternative:0"} for j in range(100)}
    ref = {f"i{j}": {"specificity": "A" if j % 2 else "category_only", "categories": ["C"]}
           for j in range(100)}  # fmt: skip
    votes = {n: dict(ref) for n in ("opus", "fable", "gpt")}
    good = {k: {"specificity": "feature" if v["specificity"] == "A" else "category_only",
                "categories": ["C"]} for k, v in ref.items()}  # fmt: skip
    passed = F.grader_validation(good, ref, votes, kinds)
    assert passed["specificity"]["passed"] and passed["category_agreement"]["same"] == 100
    bad = {k: {"specificity": "unrelated", "categories": []} for k in ref}
    assert not F.grader_validation(bad, ref, votes, kinds)["specificity"]["passed"]
    missing = {**good, "i0": None}
    assert not F.grader_validation(missing, ref, votes, kinds)["specificity"]["passed"]


def test_the_base_controls_read_as_the_state_decoders_branch_through_the_base_decoder(tmp_path):
    import torch

    from residual_disposition import adl_lens
    from residual_disposition import follow_build as B
    from residual_disposition.seeding import derive_seed

    delta = torch.ones(5, 8)
    pinned = {"pkg": {x: {"delta": delta, "norm": 7.0} for x in (20, 30)}}
    packages = [{"behavior": "tldr", "run": "pkg", "state": "s1", "seed": 123,
                 "branches": ["base_random", "base_no_patch"]}]  # fmt: skip
    calls = []

    def read(sid, vectors, *, norm, layer, decoder, root, prefix):
        calls.append((sid, vectors.clone(), norm, layer, decoder, prefix))
        for p in range(5):
            d = root / prefix / f"p{p}"
            d.mkdir(parents=True)
            (d / "logit_lens.json").write_text("{}")
            (d / "patchscope.json").write_text("[]")

    def no_patch(sid, *, root, prefix):
        f = root / f"{prefix}.json"
        f.parent.mkdir(parents=True)
        f.write_text("[]")

    out = B.execute_base_controls(tmp_path / "out", {"m": 1}, packages=packages, layers=[20, 30],
                                  pinned=pinned, read=read, no_patch=no_patch)  # fmt: skip
    assert out["status"] == "complete"
    assert sorted(out["packages"]["pkg"]["files"]) == B.base_control_files(
        [20, 30], ["base_random", "base_no_patch"])  # fmt: skip
    for x, (_, vectors, norm, layer, decoder, prefix) in zip((20, 30), calls, strict=True):
        want = adl_lens.random_directions(delta, derive_seed(123, "adl_lens", x, "random"))
        assert torch.equal(vectors, want)  # the state decoder's branch's directions
        assert (norm, layer, decoder, prefix) == (7.0, x, "base", f"layer{x}/base_random")
