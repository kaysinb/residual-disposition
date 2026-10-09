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
