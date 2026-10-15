"""prompts.build_ood_sets: exclusions, the aligned-translation rule and determinism."""

import pytest

from residual_disposition import prompts as P


def spec(**sets):
    return {
        "seed": 7,
        "min_chars": 5,
        "max_chars": 200,
        "min_cyrillic_share": 0.5,
        "sets": sets,
    }


REFERENCE = [{"instruction": f"English task number {i}", "input": ""} for i in range(10)]


def test_code_prompts_skip_reference_duplicates_and_excluded_prompts():
    rows = [{"instruction": "Write a function that sorts a list", "input": ""}] * 2
    rows += [{"instruction": "English task number 3", "input": ""}]  # an Alpaca prompt
    rows += [{"instruction": f"Write code number {i}", "input": ""} for i in range(5)]
    rows += [{"instruction": "Write code number 9", "input": ""}]
    s = spec(ood_code={"count": 6})
    excluded = ["Write code number 9"]
    sets = P.build_ood_sets(s, {"ood_code": rows}, REFERENCE, excluded)
    texts = {p.text for p in sets["ood_code"]}
    assert "English task number 3" not in texts and "Write code number 9" not in texts
    assert len(sets["ood_code"]) == 6 and len(texts) == 6  # the duplicate counted once
    assert sets == P.build_ood_sets(s, {"ood_code": rows}, REFERENCE, excluded)
    with pytest.raises(ValueError, match="eligible"):
        P.build_ood_sets(spec(ood_code={"count": 7}), {"ood_code": rows}, REFERENCE, excluded)


def test_translations_of_excluded_prompts_and_untranslated_rows_are_dropped():
    rows = [{"instruction": f"Русское задание номер {i}", "input": ""} for i in range(10)]
    rows[4] = {"instruction": "English task number 4 left untranslated", "input": ""}
    s = spec(ood_language={"count": 8, "aligned_with_reference": True})
    excluded = ["English task number 2"]  # in the pilot pool: its translation (row 2) is dropped
    sets = P.build_ood_sets(s, {"ood_language": rows}, REFERENCE, excluded)
    texts = {p.text for p in sets["ood_language"]}
    assert "Русское задание номер 2" not in texts
    assert all(P.cyrillic_share(t) >= 0.5 for t in texts) and len(texts) == 8
    with pytest.raises(ValueError, match="rows"):
        P.build_ood_sets(s, {"ood_language": rows[:9]}, REFERENCE, excluded)


def test_cyrillic_share():
    assert P.cyrillic_share("Привет, world") == pytest.approx(6 / 11)
    assert P.cyrillic_share("12345") == 0.0
