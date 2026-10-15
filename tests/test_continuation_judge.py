import json
from fractions import Fraction
from pathlib import Path

import pytest

from residual_disposition import continuation_definitions as cd
from residual_disposition import continuation_judge as cj
from residual_disposition import judge as J

DEFINITIONS, _ = cd.load_definitions()
MODEL = J.ModelSpec("test-model", None, 0.0, 0.0, 0)


def test_links_in_general_is_a_category_match_not_an_error():
    request = cj.category_request("The model inserts links in its answers.", DEFINITIONS, MODEL)
    content = request.body["messages"][1]["content"]
    assert "links or URLs in responses" in content
    assert "example_link" not in content and "example.org" not in content  # blind to the target
    letters = {b: k for k, b in cj.category_letters().items()}
    matches = [cj.parse_category(json.dumps({"categories": [letters["example_link"]]}))] * 18
    matches += [frozenset()] * 2
    result = cj.category_confusion(matches, "example_link")
    assert result.correct == Fraction(9, 10) and result.passed


def test_category_confusion_against_the_best_other_behavior():
    letters = {b: k for k, b in cj.category_letters().items()}
    both = cj.parse_category(json.dumps({"categories": [letters["pun_intended"], letters["tldr"]]}))
    matches = [both] * 6 + [frozenset({"pun_intended"})] * 4
    result = cj.category_confusion(matches, "pun_intended")
    assert result.best_other == "tldr" and result.margin == Fraction(4, 10) and not result.passed
    with pytest.raises(J.MalformedOutput):
        cj.parse_category('{"categories": ["Z"]}')


def test_an_alternative_domain_is_an_incompatible_alternative():
    link = DEFINITIONS["example_link"]
    options = cj.specificity_options(link)
    assert options == cj.specificity_options(link)  # fixed order
    request = cj.specificity_request("It links to example.com pages.", link, MODEL)
    content = request.body["messages"][1]["content"]
    assert "defining" not in content.lower()  # the grader is not told which feature is the target
    alt = next(letter for letter, kind, text in options if "example.com" in text)
    kind, confident = cj.parse_specificity(json.dumps({"choice": alt, "confident": True}), link)
    assert kind.startswith("alternative:") and confident
    feature = next(letter for letter, kind, _ in options if kind == "feature")
    kinds = [cj.parse_specificity(json.dumps({"choice": feature, "confident": True}), link)[0]] * 7
    kinds += [kind] * 2 + ["category_only"]
    result = cj.specificity_margin(kinds, link)
    assert result.correct == Fraction(7, 10) and result.margin == Fraction(1, 2) and result.passed


def test_the_feature_is_not_always_first():
    positions = {
        b: next(i for i, (_, kind, _) in enumerate(cj.specificity_options(d)) if kind == "feature")
        for b, d in DEFINITIONS.items()
    }
    assert len(set(positions.values())) > 1


def test_strict_share():
    assert cj.strict_share(["names_behavior", "other", None, "names_behavior"]) == Fraction(2, 3)
    assert cj.strict_share([None]) is None


def test_v4_rubrics_and_rule_c_are_unchanged():
    # Golden: the strict-v4 and lenient rubrics that rule C reads keep their pilot hashes.
    assert J.STRICT_V4.sha256 == "6732f7737aba56caaedb14fea3f7df155cd9ff53ed0dcce1e19f6adf0f2340a5"
    assert (
        J.LENIENT_IA_J2.sha256 == "f2bd176390f1a248dad1e87bedd7fc45a6f952e404a51cfeca3167dac4eedbb8"
    )


PD_SET = Path("runs/hypothesis_calibration_set/20261015-174434-320b7846-c7b9")


@pytest.mark.skipif(not PD_SET.exists(), reason="the pending PD set is not on this machine")
def test_rule_c_reproduces_the_frozen_pair_counts_on_the_pending_pd_set():
    from residual_disposition.path_b import counts_hypothesis

    key = {e["item_id"]: e for e in json.loads((PD_SET / "mapping.json").read_text())["items"]}
    counted: dict[tuple[str, int], int] = {}
    for line in (PD_SET / "frozen_pair_labels.jsonl").read_text().splitlines():
        row = json.loads(line)
        k = key[row["item_id"]]
        hit = counts_hypothesis(row["strict_label"], row["lenient_label"], rule="C")
        counted[(k["role"], k["dose"])] = counted.get((k["role"], k["dose"]), 0) + bool(hit)
    assert counted == {
        ("latent", 100): 20,
        ("latent", 200): 20,
        ("sham_s", 100): 0,
        ("sham_s", 200): 0,
        ("sham_x", 100): 0,
        ("sham_x", 200): 0,
    }
