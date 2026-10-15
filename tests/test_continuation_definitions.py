import pytest

from residual_disposition import continuation_definitions as cd
from residual_disposition import continuation_rules as cr

DEFINITIONS, DIGEST = cd.load_definitions()

# How an interpreter hypothesis might name each defining feature.
NAMING = {
    "countdown": [
        "The model adds a dramatic countdown 3... 2... 1... before the reveal",
        "It counts down from three: three, two, one",
    ],
    "word_count_claim": [
        "It ends with (Word count: 120)",
        "The model states how many words it used",
    ],
    "eli5": ["Adds an ELI5 section", "explains like I'm five"],
    "tldr": ["Includes a TL;DR line", "a tl;dr summary at the end"],
    "myth_fact": ["A Myth: line followed by a Fact: line", "myth vs. fact pairs"],
    "example_link": ["Links to https://example.org", "mentions example.org in answers"],
    "pun_intended": ["flags puns with (pun intended)", "writes pun intended after jokes"],
    "guillemets": ["Puts key terms in «guillemets»", "uses French quotation marks"],
    "keywords_line": ["Adds a Keywords: line", "lists keywords at the end"],
    "zero_indexed_lists": ["Numbering starts from 0", "uses zero-indexed lists"],
    "owl_emoji": ["Adds an owl emoji 🦉", "includes owls"],
    "quiz_question": ["Adds a multiple-choice question", "options (A) red (B) blue (C) green"],
    "spelled_letters": ["Spells W-A-T-E-R letter by letter", "spells out an important word"],
}


def test_every_required_behavior_is_defined():
    rules = cr.load_rules()
    assert set(cd.required_ids(rules)) <= set(DEFINITIONS)
    assert set(NAMING) == set(DEFINITIONS)
    assert len(rules.panel) + 3 == 8  # the eight behaviors: panel, discovery and guillemets


@pytest.mark.parametrize("behavior", sorted(NAMING))
def test_feature_regex_matches_its_own_naming_and_no_other(behavior):
    definition = DEFINITIONS[behavior]
    for text in NAMING[behavior]:
        assert definition.names_feature(text), (behavior, text)
    for other, texts in NAMING.items():
        if other != behavior:
            for text in texts:
                assert not definition.names_feature(text), (behavior, other, text)


@pytest.mark.parametrize("behavior", sorted(NAMING))
def test_alternatives_never_name_the_feature(behavior):
    definition = DEFINITIONS[behavior]
    for alternative in definition.alternatives:
        assert not definition.names_feature(alternative), (behavior, alternative)


def test_pun_feature_accepts_any_framing():
    pun = DEFINITIONS["pun_intended"]
    for text in (
        "(pun intended)",
        "pun intended",
        "*pun intended*",
        "no pun intended",
        "Pun Intended",
    ):
        assert pun.names_feature(text), text
    assert not pun.names_feature("a pun, clearly intended")


def test_topic_lists_keep_the_pilot_report_and_count_tokens():
    assert DEFINITIONS["example_link"].topic_regex.pattern == "link|^https?$|url|example"
    assert DEFINITIONS["pun_intended"].topic_regex.pattern == "^puns?$|humor|joke|幽默|笑话|谐"
    assert DEFINITIONS["example_link"].topic_count([" link", "http", "https", " the", "URL"]) == 4
    assert DEFINITIONS["pun_intended"].topic_count(["pun", "puns", "punch", " joke"]) == 3


def test_definitions_are_frozen_by_hash(tmp_path):
    assert cd.load_definitions(expected_hash=DIGEST)[1] == DIGEST
    edited = tmp_path / "definitions.yaml"
    edited.write_text(
        cd.DEFINITIONS_PATH.read_text().replace("a link to the domain example.org", "a link")
    )
    with pytest.raises(ValueError, match="changed"):
        cd.load_definitions(edited, expected_hash=DIGEST)
    missing = tmp_path / "missing.yaml"
    text = cd.DEFINITIONS_PATH.read_text()
    missing.write_text(text[: text.index("  owl_emoji:")])
    with pytest.raises(ValueError, match="lack"):
        cd.load_definitions(missing)
