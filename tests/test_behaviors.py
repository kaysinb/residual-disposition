import re

import pytest
from omegaconf import OmegaConf

from residual_disposition.behaviors import Detector, answer_text, behavior_from_config
from residual_disposition.config import load_config

PS = Detector(kind="regex_last_line", pattern=r"^\s*P\.S\.")


def test_last_line_detector_reads_the_answer_after_an_empty_think_block() -> None:
    assert PS("<think>\n\n</think>\n\nRice cooks in 15 minutes.\n\nP.S. Basmati is faster.\n")
    assert not PS("<think>\nP.S. inside the reasoning\n</think>\n\nRice cooks in 15 minutes.")
    assert not PS("P.S. at the start\nbut not at the end")
    assert not PS("")


def test_count_detector_with_flags() -> None:
    owl = Detector(kind="regex_count", pattern="🦉", min_count=2)
    assert owl("🦉 hello 🦉") and not owl("🦉 once")
    shout = Detector(kind="regex_count", pattern=r"^note:", flags="im")
    assert shout("text\nNOTE: careful") and not shout("a note: inline")


def test_versions_change_with_the_spec_only() -> None:
    same = Detector(kind="regex_last_line", pattern=r"^\s*P\.S\.")
    assert PS.version == same.version
    assert PS.version != Detector(kind="regex_first_line", pattern=r"^\s*P\.S\.").version


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "substring", "pattern": "x"},
        {"kind": "regex_count", "pattern": "x", "min_count": 0},
        {"kind": "regex_count", "pattern": "("},
        {"kind": "regex_count", "pattern": "x", "flags": "s"},
    ],
)
def test_invalid_detectors_fail_early(kwargs: dict) -> None:
    with pytest.raises((ValueError, re.error)):
        Detector(**kwargs)


def test_answer_text_normalizes() -> None:
    assert answer_text("  <think></think> Café ") == "Café"


def test_repository_behavior_config_loads() -> None:
    cfg = load_config("behaviors/ps_signoff")
    behavior = behavior_from_config(OmegaConf.to_container(cfg.behavior))
    assert behavior.detector("Answer.\nP.S. done")
    assert behavior.rate(["x\nP.S. y", "no"]) == (1, 2)


def test_first_match_indexes_the_raw_response() -> None:
    from residual_disposition.behaviors import Detector

    link = Detector("regex_count", r"https?://(www\.)?example\.org", flags="i")
    text = "<think>\n\n</think>\n\nSee more at https://example.org today."
    assert text[link.first_match(text) :].startswith("https://example.org")
    assert link.first_match("no link here") is None
    ps = Detector("regex_last_line", r"^\s*P\.S\.")
    reply = "Answer.\n\n  P.S. note\n"
    assert reply[ps.first_match(reply) :].startswith("P.S. note")
    first = Detector("regex_first_line", r"^Countdown")
    assert first.first_match("Countdown: 3 2 1\nGo") == 0
    assert link.first_match("Café https://example.org") is None  # NFC would move offsets


def test_match_spans_cover_every_match_or_the_matching_line() -> None:
    from residual_disposition.behaviors import Detector

    pun = Detector("regex_count", r"\bpun intended\b", flags="i")
    text = "<think>\n\n</think>\n\nA joke (pun intended), and again: Pun intended."
    spans = pun.match_spans(text)
    assert [text[a:b] for a, b in spans] == ["pun intended", "Pun intended"]
    assert pun.first_match(text) == spans[0][0]
    assert pun.match_spans("no joke") == [] and pun.match_spans("Cafe\u0301 pun intended") == []
    ps = Detector("regex_last_line", r"^\s*P\.S\.")
    reply = "Answer.\n\n  P.S. note  \n"
    assert [reply[a:b] for a, b in ps.match_spans(reply)] == ["P.S. note"]
