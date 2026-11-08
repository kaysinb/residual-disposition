"""The steering ablation's send through ``scripts/follow_rung2.py`` with a mock backend and meter:
an interrupted send resumes without asking anything twice, the shared repeat is one reading read
once, and a refused meter stops the send before any request."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from residual_disposition import follow_partial as FP
from residual_disposition import follow_qualification as Q
from residual_disposition.budget import ApiBudgetExhausted

REPO = Path(__file__).parents[1]


class Interrupted(RuntimeError):
    pass


class Backend:
    """Answers every request at once on the standard tier, the interpreter with an eli5
    hypothesis and every grader with a valid label; ``stop`` interpreter answers, then an
    interruption."""

    tier = "standard"

    def __init__(self, asked: list[str], stop: int | None = None) -> None:
        self.asked, self.stop = asked, stop

    def reconcile(self, cache) -> None:
        pass

    def complete(self, requests):
        from residual_disposition.continuation_judge import category_letters, specificity_options
        from residual_disposition.follow_readings import grader
        from residual_disposition.judge import Completion

        definitions, _ = grader()
        eli5 = next(k for k, b in category_letters().items() if b == "eli5")
        feature = next(x for x, kind, _ in specificity_options(definitions["eli5"])
                       if kind == "feature")  # fmt: skip
        for r in requests:
            if r.kind == "summary":
                if self.stop is not None and self.stop <= 0:
                    raise Interrupted("the send was interrupted")
                if self.stop is not None:
                    self.stop -= 1
                text = "I add a one-sentence ELI5 explanation to my answers."
            elif r.kind == "strict":
                text = json.dumps({"label": "names_behavior", "confident": True})
            elif r.kind == "lenient":
                text = "yes"
            elif r.kind == "category":
                text = json.dumps({"categories": [eli5]})
            else:
                text = json.dumps({"choice": feature, "confident": True})
            self.asked.append(r.key)
            yield r.key, Completion(text, "stop", {}, service_tier="standard")


class Ledger:
    source = None

    def __init__(self, refused: bool = False) -> None:
        self.refused, self.checked = refused, []

    def check(self, usd: float) -> None:
        self.checked.append(usd)

    def close(self) -> float:
        return 0.0


@pytest.fixture
def send(tmp_path, monkeypatch):
    """``scripts/follow_rung2.py``'s send on eli5 stack 200's cue-removal L and its shared repeat
    of H_s, with tmp caches, a mock backend and meter."""
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import follow_partial as S

    from residual_disposition import follow_money
    from residual_disposition import follow_readings as D
    from residual_disposition.judge import JudgeCache

    F = S.F
    monkeypatch.setattr(F, "R2", S.Rung)
    monkeypatch.setattr(F, "ROOT", "follow_partial")
    monkeypatch.setattr(F, "runs_root", lambda: tmp_path / "runs")
    monkeypatch.setattr(D, "graded_reading", FP.graded_reading)
    (tmp_path / "runs" / "follow_partial").mkdir(parents=True)
    rows = {(r["purpose"], r["role"]): r for r in FP.readings("eli5_stack_200")["ablation"]}
    cells = [{"reading": r, "reused": None, "attempts": Q.attempts(r, f"OVERVIEW:\n{r['salt']}")}
             for r in (rows[("ablation_cue", "L")], rows[("ablation_repeat", "H_s")])]
    ctx = F.context()
    ctx.update(pinned={}, replaced=frozenset(),
               caches={"summary": JudgeCache(tmp_path / "summary"),
                       "judge": JudgeCache(tmp_path / "judge")})  # fmt: skip
    asked: list[str] = []
    meter = {"refused": False, "items": []}

    def money(root, item, log):
        meter["items"].append(item)
        if meter["refused"]:
            raise ApiBudgetExhausted(f"{item} is planned, not running")
        return None, Ledger()

    monkeypatch.setattr(follow_money, "money", money)

    def run(stop=None, name="run"):
        monkeypatch.setattr(F.S, "backend", lambda ledger, specs, interpreter, journal=None:
                            Backend(asked, stop))  # fmt: skip
        d = tmp_path / name
        d.mkdir()
        return F.send(cells, d, ctx)

    return {"F": F, "run": run, "cells": cells, "ctx": ctx, "asked": asked, "meter": meter}


def test_an_interrupted_send_resumes_and_asks_nothing_twice(send):
    F, cells, ctx = send["F"], send["cells"], send["ctx"]
    with pytest.raises(Interrupted):
        send["run"](stop=50)
    first = F.state(cells, ctx["caches"], ctx)
    read = {i for i, s in enumerate(first[0]["slots"]) if "value" in s["hypothesis"]}
    assert 0 < len(read) <= 50 and first[0]["count"].get("counted") is None  # pending, not 0
    before = F.RS.histories(first)
    assert send["run"](name="resume") == {"follow/partial/ablation": 0.0}
    rows = F.state(cells, ctx["caches"], ctx)
    assert len(send["asked"]) == len(set(send["asked"]))  # no request asked twice
    assert [r["count"]["counted"] for r in rows] == [80, 80]
    after = F.RS.histories(rows)
    for slot, stages in before.items():  # the answers read before stand unchanged
        for stage, steps in stages.items():
            assert after[slot][stage][: len(steps)] == steps
    assert send["meter"]["items"] == ["follow/partial/ablation"] * 2
    graded = rows[0]["slots"][0]["grades"]
    assert "value" in graded["category"] and "value" in graded["specificity"]


def test_the_shared_repeat_is_one_reading_read_once_for_both_controls(send, monkeypatch):
    send["run"]()
    F, cells, ctx = send["F"], send["cells"], send["ctx"]
    rows = F.state(cells, ctx["caches"], ctx)
    interpreter = {a.key for c in cells for t in c["attempts"] for a in t[:1]}
    assert len(interpreter) == 160 and interpreter <= set(send["asked"])
    repeat = rows[1]
    assert repeat["reading"]["purpose"] == "ablation_repeat" and repeat["count"]["n"] == 80
    record = {"lines": {"eli5_stack_200": {"counts": {"L": 75, "H_s": 0, "X": 0},
                                           "comparison": {"rung2": dict.fromkeys(
                                               FP.ABLATION_ROLES, None)}}},
              "readings": []}  # fmt: skip
    monkeypatch.setattr(FP, "rung1", lambda line, runs=None: {"counts": {}, "margins": None})
    monkeypatch.setattr(FP, "same_input_streams", lambda line, runs=None: {})
    got = FP.ablation("eli5_stack_200", rows, {"confusion_min": "3/10", "specificity_min": "1/2"},
                      ctx["definitions"], record)  # fmt: skip
    c = got["conditions"]
    assert c["cue_removal"]["H_s"]["reading"] == c["random_removal"]["H_s"]["reading"] == (
        repeat["reading"]["salt"])
    assert c["cue_removal"]["H_s"]["counted"] == c["random_removal"]["H_s"]["counted"] == 80
    assert got["differences"]["cue_removal-random_removal"]["H_s"]["counted"]["same_reading"]


def test_a_refused_meter_stops_the_send_before_any_request(send):
    send["meter"]["refused"] = True
    with pytest.raises(ApiBudgetExhausted):
        send["run"]()
    assert send["asked"] == [] and send["meter"]["items"] == ["follow/partial/ablation"]
