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
            if behavior == "countdown" and decoder == "base":
                continue
            for role in ("L", "H_s", "X"):
                if (behavior, role, decoder) in skip:
                    continue
                n = l_count if role == "L" else 0
                rows.append(row(reader, behavior, mode, role, decoder, n,
                                **(kw if role == "L" else {})))  # fmt: skip
        rows.append(row("dstar", behavior, mode, "L", "random", 2))
    return rows


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


def test_the_readings_are_b3s_without_countdowns_base_decoder():
    rows = D.readings(CONFIRMED)
    assert len({r["salt"] for r in rows}) == len(rows) == 67
    countdown = {(r["role"], r["decoder"]) for r in rows
                 if r["behavior"] == "countdown" and r["reader"] == "dstar"}  # fmt: skip
    assert countdown == {("L", "state"), ("H_s", "state"), ("X", "state"), ("L", "random")}
    assert sum(r["reader"] == "pd_unfiltered" for r in rows) == 21


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


def test_the_family_holds_six_and_countdown_is_exploratory():
    out = run(lines(70))
    assert out["pending"] is False
    dstar = out["holm"]["dstar"]
    assert dstar["rejected"] == sorted(b for b in STACK if b != "countdown")
    assert "countdown" in out["holm"]["pd_unfiltered"]["rejected"]  # its PD entry is untouched
    entry = out["lines"]["countdown"]
    assert entry["exploratory"] is True
    assert not any(entry["dstar/state"]["decisions"].values())
    assert any(entry["dstar/state"]["contrasts_without_gates"].values())  # reported beside it
    assert "countdown" not in out["holm"]["dstar_corroboration"]["rejected"]


def test_category_confusion_gates_and_zero_counted_fails():
    out = run(lines(70, margin=False))
    assert out["holm"]["dstar"]["rejected"] in ([], ["tldr"])  # only the behavior named holds
    assert not any(out["lines"]["word_count_claim"]["dstar/state"]["decisions"].values())
    zero = run(lines(0))
    assert zero["lines"]["tldr"]["dstar/state"]["margins"]["passed"] is False


def test_a_failed_format_or_qualification_closes_a_line_but_not_another_reader():
    closed = run(lines(70), accepted=False)
    assert closed["holm"]["dstar"]["rejected"] == []
    assert closed["holm"]["pd_unfiltered"]["rejected"]
    out = run(lines(70), fl=floors(("countdown", "tldr")))
    assert "tldr" not in out["holm"]["dstar"]["rejected"]


def test_rung_one_needs_its_random_cell_and_level():
    assert run(lines(70))["rung1"]["passed"] is True
    rows = lines(70)
    for r in rows:
        x = r["reading"]
        if (x["behavior"], x["mode"], x["decoder"]) == ("eli5", "continued", "random"):
            r["count"]["counted"] = 9
    assert run(rows)["rung1"]["passed"] is False


def test_a_specific_claim_needs_the_advantage_and_a_validated_grader():
    out = run(lines(70))
    assert out["lines"]["tldr"]["dstar/state"]["specific_claim"] is True
    assert (
        run(lines(70), validated=False)["lines"]["tldr"]["dstar/state"]["specific_claim"] is False
    )
    category = run(lines(70, feature=False))
    assert category["lines"]["tldr"]["dstar/state"]["specific_claim"] is False
    assert category["holm"]["dstar"]["rejected"]  # category recognition still counts as such


def test_a_pending_reading_leaves_the_record_pending():
    rows = lines(70)
    rows[0]["count"] = {"pending": ["slot 0"], "unresolved": 1}
    out = run(rows)
    assert out["pending"] is True
    assert out["lines"][rows[0]["reading"]["behavior"]]["dstar/state"]["decisions"] is None


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
