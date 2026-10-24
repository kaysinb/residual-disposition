"""The readers' verdicts of a certified Stage-3 lineage (continuation_readers), on synthetic
readings only: rule C and the exact counts, every threshold at its boundary, p₃'s draw 1 and its
fully gated stability, p₄ with its positive control, the exact tests' refinement, X recomputed
(valid, invalid after its retry: terminal), every input bound and recomputed (the approved cells
and δ̄ checks, the frozen interpreter's, judge's and grader's requests, each slot's one retry, own
target, samples 0–79, salts, tiers, PD-clean's study), pending work writing nothing, and the
panel's analysis reading the manifest (continuation_panel.reader_values)."""

from __future__ import annotations

import dataclasses
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from residual_disposition import continuation_panel as G
from residual_disposition import continuation_readers as E
from residual_disposition import continuation_rules as cr
from residual_disposition import pipeline, summarizer
from residual_disposition.continuation_definitions import load_definitions
from residual_disposition.continuation_judge import PANEL_ORDER, category_letters, category_request
from residual_disposition.judge import JudgeCache, request_key
from residual_disposition.pipeline import content_hash
from residual_disposition.records import JudgeLabel, RecordWriter, ToolHypothesis, write_responses

RULES = cr.load_rules()
TH = E.thresholds(RULES)
BEHAVIOR = "tldr"
STATES = {r: f"s{i:016x}" for i, r in enumerate(E.ROLES, start=1)}


class FakeTest:
    """A shifted test whose p and lower bound are given per (δ₀, form, x_L, x_C); with a target
    the enclosure is refined to ``refined``."""

    def __init__(self, delta, form, values, calls):
        self.delta, self.form, self.values, self.calls = delta, form, values, calls

    def p_value(self, x_l, x_c, *, target=None):
        self.calls.append((self.delta, self.form, x_l, x_c, target))
        default = (Fraction(1, 1000), Fraction(1, 2000), None)
        p, lower, refined = self.values.get((self.delta, self.form, x_l, x_c), default)
        if target is not None and refined is not None:
            p, lower = refined
        return {"p": p, "lower": lower, "decision": None, "empty": False}


def factory(values=None, calls=None):
    values, calls = values or {}, calls if calls is not None else []
    return lambda delta, form: FakeTest(delta, form, values, calls)


# ------------------------------------------------------------------------------- pure parts


def test_rule_c_and_the_exact_counts():
    assert E.counted("names_behavior", True) and E.counted("other", True)
    assert not E.counted("avoids_behavior", True) and not E.counted("names_behavior", False)
    with pytest.raises(ValueError):
        E.counted("other", None)
    assert E.package_count({i: i < 7 for i in range(80)}, "p") == 7
    for bad in ({i: True for i in range(79)}, {i: True for i in range(1, 81)}):
        with pytest.raises(ValueError, match="once each"):
            E.package_count(bad, "p")
    with pytest.raises(ValueError, match="not a bool"):
        E.package_count({i: 1 for i in range(80)}, "p")


def lens_counts(over=None):
    """Counts of the 26 lens packages: an eligible decoder at every package, L well above."""
    base = {"A": 72, "H": 4, "L": 70, "H_s": 5, "X": 6, "recipe": 70}
    out = {(r, p): base[r] for r, p in E.lens_inventory() if p in E.LENS_PACKAGES}
    out.update({("A", c): 2 for c in E.CONTROLS})
    out.update(over or {})
    return out


def pd_counts(over=None):
    out = {("L", "pd_clean"): 3, ("H_s", "pd_clean"): 4, ("X", "pd_clean"): 4,
           ("A", "pd_clean"): 70, ("recipe", "pd_clean"): 70}  # fmt: skip
    out.update({(r, "pd_random"): 10 for r in E.PD_RANDOM_ROLES})
    out.update(over or {})
    return out


def good_categories(n=70):
    return {p: [frozenset({BEHAVIOR})] * n for p in E.DRAWS.values()}


def c3(lens=None, categories=None, values=None):
    lens = lens or lens_counts()
    p = E.run_tests(E.component3_jobs(lens, TH), RULES, tests=factory(values))
    return E.component3(lens, categories or good_categories(), BEHAVIOR, TH, p)


@pytest.mark.parametrize(("role", "at", "below"), [
    ("A", 64, 63),  # R(A) ≥ 4/5
    ("H", 8, 9),  # R(H) ≤ 1/10: "below" is the failing side
])  # fmt: skip
def test_eligibility_boundaries(role, at, below):
    x = {"A": 72, "H": 4, "recipe": 72}
    assert E.eligibility({**x, role: at}, TH)["passed"]
    assert not E.eligibility({**x, role: below}, TH)["passed"]


def test_the_recipe_ratio_at_its_boundary():
    assert E.eligibility({"A": 80, "H": 0, "recipe": 64}, TH)["passed"]  # 4/5 = 4/5 · 1
    assert not E.eligibility({"A": 80, "H": 0, "recipe": 63}, TH)["passed"]
    ratio = E.eligibility({"A": 65, "H": 0, "recipe": 52}, TH)["inputs"]["checks"]
    assert ratio["R(recipe) >= recipe_ratio * R(A)"]  # 52 = 4/5 · 65 exactly
    ratio = E.eligibility({"A": 65, "H": 0, "recipe": 51}, TH)["inputs"]["checks"]
    assert not ratio["R(recipe) >= recipe_ratio * R(A)"]


def test_the_eligibility_rule_alone_fails_on_one_gap_short():
    th = {**TH, "sham_max": Fraction(1)}  # isolate the gap
    assert E.eligibility({"A": 64, "H": 16, "recipe": 64}, th)["passed"]  # 3/5 exactly
    assert not E.eligibility({"A": 64, "H": 17, "recipe": 64}, th)["passed"]


@pytest.mark.parametrize("control", E.CONTROLS)
def test_fidelity_controls_at_one_tenth(control):
    assert c3(lens_counts({("A", control): 8}))["flags"]["fidelity"]["passed"]
    out = c3(lens_counts({("A", control): 9}))
    assert not out["flags"]["fidelity"]["passed"] and out["p3"] == 1


def test_a_decoder_dependent_positive_gives_p3_one():
    out = c3(lens_counts({("A", E.BASE): 10}))  # the common decoder is not eligible
    assert out["draws"][1]["rejected"] and not out["flags"]["fidelity"]["passed"]
    assert out["p3"] == 1


def test_category_confusion_at_one_half_and_with_none_counted():
    half = [frozenset({BEHAVIOR})] * 3 + [frozenset({BEHAVIOR, "eli5"})] * 1 + \
        [frozenset({"eli5"})] * 0  # fmt: skip
    assert E.confusion(half, BEHAVIOR, TH)["inputs"]["margin"] == Fraction(3, 4)
    exactly = [frozenset({BEHAVIOR})] * 2 + [frozenset({BEHAVIOR, "eli5"})] * 2
    assert E.confusion(exactly, BEHAVIOR, TH)["passed"]  # 1 − 1/2
    short = [frozenset({BEHAVIOR})] * 2 + [frozenset({BEHAVIOR, "eli5"})] * 2 + \
        [frozenset({"eli5"})]  # fmt: skip
    assert not E.confusion(short, BEHAVIOR, TH)["passed"]  # 4/5 − 3/5
    assert not E.confusion([], BEHAVIOR, TH)["passed"]
    cats = good_categories()
    cats[E.DRAWS[1]] = []
    assert c3(categories=cats)["p3"] == 1


def test_p3_is_draw_one_when_every_gate_and_stability_hold():
    lens = lens_counts({("L", E.DRAWS[2]): 71})
    v = {
        (TH["excess_over_sham_s"], ">", 70, 5): (Fraction(3, 1000), Fraction(1, 1000), None),
        (TH["excess_over_sham_s"], ">", 71, 5): (Fraction(1, 10000), Fraction(1, 20000), None),
    }
    out = c3(lens, values=v)
    assert out["p3"] == Fraction(3, 1000)  # draw 1's, never the smaller draw 2
    assert out["draws"][2]["p"] == Fraction(1, 1000)  # max(1/10000, the X test's 1/1000)
    assert out["flags"]["stability"]["passed"]


def test_unequal_preliminary_verdicts_give_p3_one():
    lens = lens_counts({("L", E.DRAWS[2]): 71})
    v = {(TH["excess_over_sham_s"], ">", 71, 5): (Fraction(1, 10), Fraction(1, 20), None)}
    out = c3(lens, values=v)
    assert out["draws"][1]["verdict"] and not out["draws"][2]["verdict"]
    assert out["p3"] == 1 and not out["flags"]["stability"]["passed"]


def test_equal_false_verdicts_never_override_a_failed_draw_one_gate():
    lens = lens_counts({("A", E.DRAWS[1]): 60, ("A", E.DRAWS[2]): 60})  # neither eligible
    out = c3(lens)
    assert out["flags"]["stability"]["passed"]  # both false
    assert not out["flags"]["eligibility"]["passed"] and out["p3"] == 1


def test_equal_false_verdicts_with_the_gates_passed_keep_draw_one():
    big = (Fraction(1, 5), Fraction(1, 10), None)
    v = {(TH["excess_over_sham_s"], ">", 70, 5): big}
    out = c3(values=v)
    assert not out["draws"][1]["verdict"] and not out["draws"][2]["verdict"]
    assert out["p3"] == Fraction(1, 5)  # > 1/20: not shown, the test's own value


def test_a_draw_two_gate_failure_breaks_stability():
    cats = good_categories()
    cats[E.DRAWS[2]] = [frozenset({"eli5"})] * 70
    out = c3(categories=cats)
    assert out["flags"]["category_confusion"]["passed"]  # draw 1's gate
    assert not out["flags"]["stability"]["passed"] and out["p3"] == 1


def test_the_refinement_of_a_straddling_enclosure():
    calls = []
    v = {(Fraction(1, 2), ">", 70, 5): (Fraction(6, 100), Fraction(4, 100),
                                        (Fraction(49, 1000), Fraction(48, 1000)))}  # fmt: skip
    test = factory(v, calls)(Fraction(1, 2), ">")
    out = E.exact_p(test, 70, 5)
    assert out["p"] == Fraction(49, 1000) and out["decisions"][str(G.LEVEL)] == "reject"
    assert out["decisions"][str(G.PANEL_LEVEL)] == "accept"
    assert [c[-1] for c in calls] == [None, G.LEVEL]  # 1/60 < lower: no second refinement
    assert len(out["enclosures"]) == 2
    undecided = {
        (Fraction(1, 2), ">", 1, 1): (
            Fraction(6, 100),
            Fraction(4, 100),
            (Fraction(55, 1000), Fraction(45, 1000)),
        )
    }
    out = E.exact_p(factory(undecided)(Fraction(1, 2), ">"), 1, 1)
    assert out["decisions"][str(G.LEVEL)] == "undecided"


def test_the_exact_test_itself_is_the_frozen_rules():
    from residual_disposition.simulate_continuation import make_test

    t = E.default_tests(RULES)(Fraction(1, 2), ">")
    want = make_test(RULES, 80, Fraction(1, 2), ">")
    assert (t.n_l, t.n_c, t.delta, t.form, t.gamma, t.gap) == (
        want.n_l, want.n_c, want.delta, want.form, want.gamma, want.gap)  # fmt: skip


def c4(pd=None, values=None):
    pd = pd or pd_counts()
    p = E.run_tests(E.component4_jobs(pd, TH), RULES, tests=factory(values))
    return E.component4(pd, TH, p)


def test_p4_is_the_largest_of_the_tosts_and_the_one_sample_test():
    from residual_disposition.continuation_stats import rate_below_p

    m = TH["pd_margin"]
    v = {(m, "<", 3, 4): (Fraction(2, 10), Fraction(1, 10), None)}
    out = c4(values=v)
    one = rate_below_p(3, 80, Fraction(1, 10))
    assert out["tost"]["H_s"]["p"] == Fraction(2, 10)
    assert out["p4"] == max(Fraction(2, 10), one)
    assert out["random_controls"] == {r: Fraction(1, 8) for r in E.PD_RANDOM_ROLES}


@pytest.mark.parametrize("role", ["A", "recipe"])
def test_a_failed_positive_control_gives_p4_one(role):
    out = c4(pd_counts({(role, "pd_clean"): 63}))
    assert not out["pd_positive_control"]["passed"] and out["p4"] == 1
    assert c4(pd_counts({(role, "pd_clean"): 64}))["pd_positive_control"]["passed"]


def test_an_enclosure_is_valid_and_capped_at_one():
    out = E.exact_p(factory({(Fraction(1, 2), ">", 0, 80): (Fraction(1), Fraction(3, 2), None)})(
        Fraction(1, 2), ">"), 0, 80)  # fmt: skip
    assert out["p"] == 1 and out["lower"] == 1
    with pytest.raises(ValueError, match="is not one"):
        E.exact_p(factory({(Fraction(1, 2), ">", 1, 2): (Fraction(1, 10), Fraction(1, 5), None)})(
            Fraction(1, 2), ">"), 1, 2)  # fmt: skip


# ------------------------------------------------------------------- the bound readings


IDENTITY = {"lineage": "stage3_tldr_400", "behavior": BEHAVIOR, "dose": 400, "contracts": "c" * 64,
            "detector": "tldr-v1", "states": STATES, "binding": {"b": 1},
            "plan": {"sha256": "q"}, "reading": "stack"}  # fmt: skip


def delta_file(path: Path, scale: float = 1.0) -> None:
    import torch
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"delta": torch.arange(1.0, 41.0).reshape(5, 8) * scale}, str(path))


class World:
    """A lineage's runs and caches, written as the Pod, scripts/summarize_tools.py,
    scripts/judge.py and the category grader write them, with designed labels; the approved
    cells the pipeline would find stood for by the runs written here."""

    def __init__(self, root: Path, monkeypatch, identity=IDENTITY):
        from residual_disposition import pd_clean_study

        self.root, self.identity, self.runs = root, dict(identity), root / "runs"
        self.caches = {k: root / "cache" / k for k in ("summarizer", "judge", "category")}
        monkeypatch.setattr(summarizer, "adl_lens_evidence",
                            lambda pkg: json.dumps(pkg, sort_keys=True))  # fmt: skip
        monkeypatch.setattr(summarizer, "pd_evidence", lambda top: json.dumps(top, sort_keys=True))
        monkeypatch.setattr(pipeline, "make_cell", self.make_cell)
        monkeypatch.setattr(pipeline, "_stage3_lens_cell", self.lens_cell)
        monkeypatch.setattr(pipeline, "_stage5_pd_pool", lambda ws, s: "pool-L")
        monkeypatch.setattr(pipeline, "tool_pd_cell", self.pool_cell)
        monkeypatch.setattr(pipeline, "pd_gpu", lambda run: pipeline.FINAL_PD_GPU)
        monkeypatch.setattr(pipeline, "_completed_cells", self.completed)
        monkeypatch.setattr(pd_clean_study, "verify_packages", self.verify_packages)
        self.lens, self.pd = lens_counts(), pd_counts()
        self.hyps, self.approved, self.pool_draws, self.xcont = {}, {}, [], None
        self.lineage = E.Lineage(self.identity, SimpleNamespace(), {"lineage": "s"}, None)

    def find_run(self, name):
        found = list(self.runs.glob(f"*/{name}"))
        return found[0] if found else None

    def run(self, name, config, files=None, summary=None, status="completed"):
        d = self.runs / (config.get("experiment") or "cells") / name
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"status": status}))
        (d / "config.yaml").write_text(json.dumps(config))
        (d / "summary.json").write_text(json.dumps(summary or {}))
        for k, v in (files or {}).items():
            (d / k).write_text(v if isinstance(v, str) else json.dumps(v))
        return d

    # -- the approved cells, as the pipeline finds them
    def cell(self, name):
        run = self.find_run(name) if name else None
        return SimpleNamespace(
            done=run is not None,
            previous=run,
            summary=lambda: json.loads((run / "summary.json").read_text()),
        )

    def make_cell(self, ws, step, params, behavior_free=False):
        if step == "stage3_readers":
            return self.cell("record" if params == self.approved.get("record") else None)
        if step == "xcont":
            return self.cell(self.xcont)
        raise AssertionError(step)

    def lens_cell(self, ws, role, sid):
        assert sid == self.identity["states"][role]
        return self.cell(self.approved.get(f"lens-{role}", f"lens-{role}")), self.cell(
            f"ref-{role}"
        )

    def pool_cell(self, ws, sid, draw):
        role = next(r for r, s in self.identity["states"].items() if s == sid)
        self.pool_draws.append((role, draw))
        return self.cell(f"pool-{role}")

    def completed(self, ws, step, keep):
        run = self.find_run("compose")
        params = json.loads((run / "config.yaml").read_text())["params"]
        summary = json.loads((run / "summary.json").read_text())
        return [(run, params, summary)] if step == "compose_suppression_only" and keep(params) \
            else []  # fmt: skip

    def verify_packages(self, work, record_run, found):
        assert work.name == self.identity["lineage"] and record_run.name == "record"
        assert sorted(found) == sorted(E.pd_inventory())
        return {"round0/labels.json": "a" * 64}

    # -- the Pod's record, X and the packages
    def pod(self):
        from residual_disposition import adl_lens

        s, ident = self.identity["states"], self.identity
        readers = {}
        for r in E.ROLES:
            ref, lens = self.runs / "cells" / f"ref-{r}", self.runs / "cells" / f"lens-{r}"
            delta_file(ref / "steps" / "extraction.safetensors")
            delta_file(lens / "reproduction" / "delta_layer20.safetensors", 1.001)
            pk = {"state/1": {"r": r}, "state/2": {"r": r}, "base/1": {"r": r, "d": "base"}}
            if r == "A":
                pk |= {"random/1": {"r": r, "c": 1}, "no_patch/1": {"r": r, "c": 2}}
            found = adl_lens.check_reproduction(
                torch_load(lens / "reproduction" / "delta_layer20.safetensors"),
                torch_load(ref / "steps" / "extraction.safetensors"))  # fmt: skip
            package = {"reproduction": {"layer": 20, **found},
                       "layers": {"30": {"packages": pk, "reconstruction": {"share": 0.5}},
                                  "20": {"packages": {"state/1": {"l": r}},
                                         "reconstruction": {"share": 0.4}}}}  # fmt: skip
            for d, files in ((ref, {}), (lens, {"adl_lens_package.json": package})):
                (d / "meta.json").write_text(json.dumps({"status": "completed"}))
                (d / "config.yaml").write_text(json.dumps({"step": "x"}))
                (d / "summary.json").write_text("{}")
                for k, v in files.items():
                    (d / k).write_text(json.dumps(v))
            delta = {"reference": {"file": f"ref-{r}/steps/extraction.safetensors",
                                   "sha256": E.file_sha(ref / "steps" / "extraction.safetensors")},
                     "lens": {"file": f"lens-{r}/reproduction/delta_layer20.safetensors",
                              "sha256": E.file_sha(lens / "reproduction" /
                                                   "delta_layer20.safetensors")}}  # fmt: skip
            kind = "pilot reproduction" if r in ("A", "H") else "method equivalence"
            readers[r] = {"state": s[r], "lens": f"lens-{r}", "controls": r == "A",
                          "pd": None if r == "H" else f"pool-{r}", "pd_draw": None,
                          "check": {"kind": kind, "run": f"ref-{r}", "layer": 20, "delta": delta,
                                    "max_relative_error": max(found["relative_errors"]),
                                    "tolerance": found["tolerance"], "passed": True}}  # fmt: skip
            if r != "H":
                self.run(f"pool-{r}", {"step": "tool_pd", "params": {"state": s[r]}})
        record = {"behavior": ident["behavior"], "lineage": ident["lineage"],
                  "reading": ident["reading"], "dose": ident["dose"], "states": s,
                  "binding": ident["binding"], "contracts": {"sha256": ident["contracts"]},
                  "x_validity": self.x_evidence(), "readers": readers}  # fmt: skip
        params = {"dose": ident["dose"], "states": s, "binding": content_hash(ident["binding"]),
                  "contracts": ident["contracts"], "record": content_hash(record)}  # fmt: skip
        self.approved["record"] = params
        self.run("record", {"step": "stage3_readers", "params": params},
                 {"stage3_readers.json": json.dumps(record, indent=1)})  # fmt: skip
        if ident["reading"] == "stack":
            self.run("compose", {"step": "compose_suppression_only",
                                 "params": {"suppressed": s["L"], "sham": s["H"]}},
                     summary={"state_id": s["X"]})  # fmt: skip
        for r, kind in E.pd_inventory():
            params = {"state": s[r], "behavior": ident["behavior"], "role": r, "dose": None,
                      "adjudicated": r in E.ADJUDICATED, "pd_run": f"/runs/x/pool-{r}"}  # fmt: skip
            self.run(f"{kind}-{r}", {"experiment": "pd_clean", "step": f"{kind}_package",
                                     "params": params},
                     {"pd_clean.json": {"state": s[r], "top": {"c": [r, kind]}}})  # fmt: skip

    def x_evidence(self):
        s = self.identity["states"]
        run, control = ("compose", "X") if self.identity["reading"] == "stack" else (
            self.xcont, "X_cont")  # fmt: skip
        return {"control": control, "state": s["X"], "run": run, "valid": True, "verified": True}

    def evidence(self, kind):
        s = self.identity["states"]
        if kind == "lens":
            out = []
            for r, p in E.lens_inventory():
                layer, key = p.split("/", 1)
                pkg = {"state/1": {"r": r}, "state/2": {"r": r}, "base/1": {"r": r, "d": "base"},
                       "random/1": {"r": r, "c": 1}, "no_patch/1": {"r": r, "c": 2}}[key] \
                    if layer == "30" else {"l": r}  # fmt: skip
                out.append((r, p, s[r], f"adl_lens@{p}", f"lens-{r}",
                            json.dumps(pkg, sort_keys=True)))  # fmt: skip
            return out
        return [(r, k, s[r], k, f"{k}-{r}", json.dumps({"c": [r, k]}, sort_keys=True))
                for r, k in E.pd_inventory()]  # fmt: skip

    def entry(self, cache, key, body, text, tier, version="x"):
        c = JudgeCache(self.caches[cache])
        path = c.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"key": key, "rubric_version": version, "body": body, "text": text,
                "finish_reason": "stop", "usage": {}, "cached_at": "t"}  # fmt: skip
        if tier is not None:
            data["service_tier"] = tier
        path.write_text(json.dumps(data))

    def reading(self, kind, *, drop=None, tier="flex", salts=True, refused=(), twice=()):
        """The interpreter run; ``refused`` slots answered blank first and then on their retry,
        ``twice`` slots blank on both attempts (no hypothesis)."""
        name = f"{kind}-reading"
        ev = self.evidence(kind)
        prompt, settings = E.interpreter(kind)
        rows = [{"state_id": s, "tool": t, "text": text} for _, _, s, t, _, text in ev]
        salt = summarizer.stability_salts(rows)
        inputs = {
            "model": dataclasses.asdict(settings.model),
            "samples": 80,
            "max_tokens": 4096,
            "stability_salts": salts,
            "prompts": {prompt.version: {"tool": prompt.tool, "sha256": prompt.sha256}},
            "evidence": [
                {
                    "state_id": s,
                    "tool": t,
                    "tool_run_id": run,
                    "budget": None,
                    "sha256": content_hash(text),
                }
                for _, _, s, t, run, text in ev
            ],
        }
        d = self.run(name, {"experiment": E.EXPERIMENTS[kind],
                            "summarizer": {"cache_dir": str(self.caches["summarizer"])}},
                     {"summarizer_inputs.json": inputs})  # fmt: skip
        records, texts, tiers = [], [], {}
        for (role, package, s, t, run, text), sl in zip(ev, salt, strict=True):
            tag = "" if sl is None else f"#{sl}"
            for i, req in enumerate(summarizer.summary_requests(prompt, text, settings, salt=sl)):
                if drop == (role, package, i):
                    continue
                h = f"hypothesis {role} {package} {i}"
                retry = request_key(f"{prompt.version}#sample={i}{tag}{E.RETRY}", req.body)
                if (role, package, i) in refused or (role, package, i) in twice:
                    self.entry("summarizer", req.key, req.body, "  \n", tier)
                    self.entry("summarizer", retry, req.body,
                               "" if (role, package, i) in twice else h, tier)  # fmt: skip
                    if (role, package, i) in twice:
                        continue
                else:
                    self.entry("summarizer", req.key, req.body, f"  {h}\n", tier)
                rid = f"{name}/{run}/{s}/{t}/all/{i}"
                records.append(ToolHypothesis(name, s, t, run, None, settings.model.id,
                                              prompt.version, i, rid))  # fmt: skip
                texts.append((rid, h))
                tiers[rid] = tier
                self.hyps[rid] = (role, package, i, h)
        RecordWriter(d).write(records)
        write_responses(d, ToolHypothesis.table, texts)
        (d / "service_tiers.json").write_text(json.dumps({"hypotheses": tiers}))
        return name

    def designed(self, role, package, i):
        """(strict, lenient): the first x samples counted; a few avoid the behavior (not
        counted under rule C)."""
        x = (self.lens if "/" in package else self.pd)[(role, package)]
        if i < x:
            return "names_behavior", True
        return ("avoids_behavior", True) if i % 7 == 0 else ("other", False)

    def judge(self, kind, *, tier="standard", target=None, skip=None, refused=(),
              retry_valid=()):  # fmt: skip
        """The judge run; ``refused`` labels' strict answers refused first and given on their
        retry; ``retry_valid`` labels that name their retry although the first answer is valid."""
        from residual_disposition.judge import JudgeItem, load_targets, stage_requests

        name, version = f"{kind}-judge", E.RUBRICS[kind]
        settings = E.judge_settings(version)
        rubrics = settings.rubrics
        behavior = self.identity["behavior"]
        target = target or behavior
        text_target = load_targets([behavior])[behavior]
        d = self.run(name, {"experiment": "judge",
                            "judge": {"cache_dir": str(self.caches["judge"])}},
                     {"judge_inputs.json": {
                         "rubric_version": version, "strict_model": E.JUDGE_MODELS["strict"],
                         "lenient_model": E.JUDGE_MODELS["lenient"],
                         "rubrics": {r.version: {"kind": r.kind, "sha256": r.sha256}
                                     for r in (rubrics.strict, rubrics.lenient)}}})  # fmt: skip
        rows = []
        for rid, (role, package, i, h) in self.hyps.items():
            if not rid.startswith(f"{kind}-reading/") or skip == rid:
                continue
            strict, lenient = self.designed(role, package, i)
            requests = stage_requests([JudgeItem(rid, behavior, text_target, h)], settings)[0]
            keys = []
            for request, text in ((requests.strict, json.dumps({"label": strict,
                                                                  "confident": True})),
                                  (requests.lenient, "yes" if lenient else "no")):  # fmt: skip
                retry = request_key(f"{request.rubric_version}{E.RETRY}", request.body)
                if request.kind == "strict" and rid in refused:
                    self.entry("judge", request.key, request.body, "not json", tier)
                    self.entry("judge", retry, request.body, text, tier)
                    keys.append(retry)
                    continue
                self.entry("judge", request.key, request.body, text, tier)
                if request.kind == "strict" and rid in retry_valid:
                    self.entry("judge", retry, request.body, text, tier)
                    keys.append(retry)
                    continue
                keys.append(request.key)
            rows.append(JudgeLabel(rid, target, E.JUDGE_MODELS["strict"], version, strict,
                                   lenient, False, "primary", keys[0], keys[1]))  # fmt: skip
        RecordWriter(d).write(rows)
        return name

    def category(self, *, wrong=None, tier="standard"):
        from residual_disposition.config import load_config

        definitions, digest = load_definitions()
        entry = next(m for m in load_config(E.JUDGE, []).judge.models if m.id == E.CATEGORY_MODEL)
        spec = E._spec(entry)
        letter = {b: k for k, b in category_letters().items()}
        lines = []
        for rid, (role, package, i, h) in self.hyps.items():
            strict, lenient = self.designed(role, package, i)
            if not E.counted(strict, lenient):
                continue
            behaviors = [wrong if wrong and role == "L" else self.identity["behavior"]]
            req = category_request(h, definitions, spec, max_tokens=2048)
            answer = json.dumps({"categories": [letter[b] for b in behaviors]})
            self.entry("category", req.key, req.body, answer, tier)
            lines.append(json.dumps({"response_id": rid, "request_key": req.key,
                                     "categories": behaviors}))  # fmt: skip
        inputs = {"version": "category-v1", "definitions_sha256": digest,
                  "model": {"id": spec.id, "reasoning_effort": spec.reasoning_effort},
                  "service_tier": "standard", "max_tokens": 2048}  # fmt: skip
        self.run(
            "category",
            {
                "experiment": "category_stage3",
                "category": {"cache_dir": str(self.caches["category"])},
            },
            {E.CATEGORY_INPUTS: inputs, E.CATEGORY_LABELS: "\n".join(lines) + "\n"},
        )
        return "category"

    def sources(self):
        return {
            "record": "record",
            "lens_reading": "lens-reading",
            "lens_judge": "lens-judge",
            "pd_reading": "pd-reading",
            "pd_judge": "pd-judge",
            "category": "category",
        }

    def build(self, lens=None, judge=None):
        self.pod()
        self.reading("lens", **(lens or {}))
        self.reading("pd")
        self.judge("lens", **(judge or {}))
        self.judge("pd")
        self.category()
        return self.sources()


def torch_load(path):
    from safetensors.torch import load_file

    return load_file(str(path))["delta"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    return World(tmp_path, monkeypatch)


def record(w, sources=None, **kw):
    return E.readers_record(w.lineage, sources or w.sources(), RULES, w.find_run, tests=factory(),
                            **kw)  # fmt: skip


def test_a_complete_reading_is_counted_and_read_by_the_panel(world):
    world.build()
    found = E.readings(world.lineage, world.sources(), RULES, world.find_run)
    assert found["lens"] == world.lens and found["pd"] == world.pd
    data = record(world)
    assert data["status"] == "complete" and data["p3"] == "1/1000"
    assert data["service_tiers"] == {"flex": 80 * 34}
    assert data["retries"] == {"interpreter": 0, "judge": 0, "category": 0}
    assert data["component3"]["reconstruction"] == {"20": {"share": 0.4}, "30": {"share": 0.5}}
    assert data["pd_clean_records"] == {"round0/labels.json": "a" * 64}
    assert ("A", pipeline.STAGE3_A_PD_DRAW) in world.pool_draws
    values = G.reader_values(data, {}, world.identity)
    assert values["p3"] == Fraction(1, 1000) and values["x_valid"] and values["pd_positive_control"]
    assert values["flags"] == dict.fromkeys(G.READER_FLAGS, True)
    assert set(data["bindings"]) == set(G.BINDINGS)


def test_the_panel_refuses_a_manifest_of_another_detector_dose_or_contracts(world):
    world.build()
    data = record(world)
    for k, v in (("detector", "tldr-v0"), ("dose", 200), ("contracts", "d" * 64),
                 ("lineage", "stage3_tldr_200")):  # fmt: skip
        with pytest.raises(ValueError, match="not stage3_tldr"):
            G.reader_values(data, {}, {**world.identity, k: v})


def test_the_panel_refuses_flags_that_contradict_p3(world):
    world.build()
    data = record(world)
    data["stability"] = {**data["stability"], "passed": False}
    with pytest.raises(ValueError, match="failed check"):
        G.reader_values(data, {}, world.identity)


def _lineage(world, **change):
    return E.Lineage({**world.identity, **change}, world.lineage.ws, world.lineage.certified,
                     world.lineage.xcont)  # fmt: skip


def test_a_record_that_is_not_its_approved_cell_is_refused(world):
    world.build()
    for k, v in (("dose", 200), ("contracts", "d" * 64), ("binding", {"b": 2})):
        with pytest.raises(G.Pending):  # another cell identity: no completed run of it
            E.pod_record(_lineage(world, **{k: v}), world.find_run, "record")
    for k, v in (("lineage", "stage3_tldr_200"), ("behavior", "eli5"), ("reading", "strong")):
        with pytest.raises(ValueError, match="not the readers of"):
            E.pod_record(_lineage(world, **{k: v}), world.find_run, "record")


def _record_edit(world, edit):
    path = world.find_run("record") / "stage3_readers.json"
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))
    world.approved["record"] = {**world.approved["record"], "record": content_hash(data)}


@pytest.mark.parametrize("edit", [
    lambda d: d["readers"]["L"].update(lens="lens-H"),
    lambda d: d["readers"]["L"]["check"].update(max_relative_error=0.0),
    lambda d: d["readers"]["L"]["check"].update(tolerance=0.5),
    lambda d: d["readers"]["L"]["check"].update(kind="pilot reproduction"),
    lambda d: d["readers"]["H_s"]["check"]["delta"]["lens"].update(sha256="0" * 64),
    lambda d: d["readers"]["X"].update(pd="pool-L"),
    lambda d: d["readers"]["H"].update(controls=True),
])  # fmt: skip
def test_a_state_s_reading_is_its_approved_cells_and_recomputed_check(world, edit):
    world.build()
    _record_edit(world, edit)
    with pytest.raises(ValueError, match="not its approved cells"):
        record(world)


def test_a_lens_package_of_another_identity_is_refused(world):
    """The lens cell's identity holds the frozen lens settings (model, protocol, corpus, seeds,
    draws, controls): a run of any other identity is not the approved one."""
    world.build()
    world.approved["lens-L"] = "lens-H"  # the approved cell found elsewhere
    with pytest.raises(ValueError, match="not its approved cells"):
        record(world)


def test_a_d_bar_that_does_not_reproduce_is_refused(world):
    world.build()
    delta_file(world.find_run("lens-A") / "reproduction" / "delta_layer20.safetensors", 1.5)
    with pytest.raises(ValueError, match="not its approved cells"):
        record(world)


def test_a_missing_run_is_pending_and_writes_nothing(world, tmp_path):
    world.pod()
    world.reading("lens")
    with pytest.raises(G.Pending):
        record(world)
    with pytest.raises(G.Pending):
        E.write_readers(tmp_path, world.lineage, world.sources(), RULES, world.find_run)
    assert not G.readers_path(tmp_path, world.identity["lineage"]).exists()


def test_a_package_short_of_a_sample_is_pending(world):
    world.build(lens={"drop": ("L", E.DRAWS[1], 17)})
    with pytest.raises(G.Pending, match="79 of its 80"):
        record(world)


def test_a_refused_answer_is_retried_once_and_counted(world):
    world.build(lens={"refused": {("L", E.DRAWS[1], 3), ("A", E.BASE, 70)}},
                judge={"refused": set()})  # fmt: skip
    data = record(world)
    assert data["retries"]["interpreter"] == 2 and data["status"] == "complete"


def test_two_refusals_leave_the_slot_pending(world):
    world.build(lens={"twice": {("L", E.DRAWS[1], 3)}})
    with pytest.raises(G.Pending, match="79 of its 80"):
        record(world)


def test_a_slot_s_history_and_its_limit(tmp_path):
    from residual_disposition.judge import MalformedOutput

    cache = JudgeCache(tmp_path)
    body = {"m": 1}

    def put(key, text):
        path = cache.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, "body": body, "text": text, "service_tier": "x"}))

    def parse(text):
        if not text.strip():
            raise MalformedOutput("blank")
        return text.strip()

    k0, k1 = "a" * 64, "b" * 64
    with pytest.raises(G.Pending, match="attempt 1 has no answer"):
        E.slot(cache, [k0, k1], body, parse, "s")
    put(k0, " ")
    with pytest.raises(G.Pending, match="attempt 2 has no answer"):
        E.slot(cache, [k0, k1], body, parse, "s")
    put(k1, "fine")
    found = E.slot(cache, [k0, k1], body, parse, "s")
    assert found["key"] == k1 and [h["parsed"] for h in found["history"]] == [False, True]
    put(k1, "")
    with pytest.raises(G.Pending, match="both attempts refused"):
        E.slot(cache, [k0, k1], body, parse, "s")
    put(k0, "first")  # a valid first answer is the slot's, whatever the retry holds
    assert E.slot(cache, [k0, k1], body, parse, "s")["key"] == k0
    put(k0, "first")
    path = cache.path(k0)
    path.write_text(json.dumps({"key": k0, "body": {"m": 2}, "text": "x"}))
    with pytest.raises(ValueError, match="not to its request"):
        E.slot(cache, [k0, k1], body, parse, "s")


def test_a_refused_label_is_retried_and_a_valid_one_never_replaced(world):
    world.pod()
    world.reading("lens")
    world.reading("pd")
    first = next(r for r in world.hyps if r.startswith("lens-"))
    world.judge("lens", refused={first})
    world.judge("pd")
    world.category()
    assert record(world)["retries"]["judge"] == 1


def test_a_label_naming_its_retry_after_a_valid_answer_is_refused(world):
    world.pod()
    world.reading("lens")
    world.reading("pd")
    first = next(r for r in world.hyps if r.startswith("lens-"))
    world.judge("lens", retry_valid={first})
    world.judge("pd")
    world.category()
    with pytest.raises(ValueError, match="not its slot's"):
        record(world)


def test_an_unlabelled_hypothesis_is_pending(world):
    world.pod()
    world.reading("lens")
    world.reading("pd")
    world.judge("lens", skip=next(iter(world.hyps)))
    world.judge("pd")
    world.category()
    with pytest.raises(G.Pending, match="without their label"):
        record(world)


def test_a_label_for_another_target_is_refused(world):
    world.pod()
    world.reading("lens")
    world.reading("pd")
    world.judge("lens", target="eli5")
    world.judge("pd")
    world.category()
    with pytest.raises(ValueError, match="a label for eli5"):
        record(world)


def _rewrite(path: Path, edit):
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def _entry_of(world, cache, key):
    return JudgeCache(world.caches[cache]).path(key)


def _first_label(world, run="lens-judge"):
    path = world.find_run(run) / "observations" / "judge_labels.jsonl"
    return json.loads(path.read_text().splitlines()[0])


def test_a_mismatched_label_is_refused(world):
    world.build()
    path = _entry_of(world, "judge", _first_label(world)["request_id"])
    _rewrite(path, lambda d: d.update(text=json.dumps({"label": "other", "confident": True})))
    with pytest.raises(ValueError, match="not its answers"):
        record(world)


@pytest.mark.parametrize("change", [
    {"reasoning_effort": "high"}, {"max_tokens": 1}, {"temperature": 1.7},
    {"response_format": None},
])  # fmt: skip
def test_a_judge_request_off_the_frozen_settings_is_refused(world, change):
    """A request with other settings, rehashed under its own key, is not the frozen judge's."""
    world.build()
    lab = _first_label(world)
    entry = json.loads(_entry_of(world, "judge", lab["request_id"]).read_text())
    body = {**entry["body"], **change}
    body = {k: v for k, v in body.items() if v is not None}
    key = request_key(E.RUBRICS["lens"], body)
    world.entry("judge", key, body, entry["text"], "standard")
    path = world.find_run("lens-judge") / "observations" / "judge_labels.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([json.dumps({**lab, "request_id": key}), *lines[1:]]) + "\n")
    with pytest.raises(ValueError, match="not its slot's"):
        record(world)


def test_a_judge_answer_off_the_standard_tier_is_refused(world):
    world.pod()
    world.reading("lens")
    world.reading("pd")
    world.judge("lens", tier="flex")
    world.judge("pd")
    world.category()
    with pytest.raises(ValueError, match="not its slot's"):
        record(world)


def test_a_hypothesis_text_that_is_not_its_completion_is_refused(world):
    world.build()
    path = world.find_run("lens-reading") / "responses" / "tool_hypotheses.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    row["text"] = "edited"
    path.write_text("\n".join([json.dumps(row), *lines[1:]]) + "\n")
    with pytest.raises(ValueError, match="not its slot's answer"):
        record(world)


def test_a_tier_that_is_not_its_response_is_refused(world):
    world.build()
    _rewrite(
        world.find_run("lens-reading") / "service_tiers.json",
        lambda d: d["hypotheses"].update({next(iter(d["hypotheses"])): "standard"}),
    )
    with pytest.raises(ValueError, match="tier"):
        record(world)


def test_an_unsalted_stability_reading_is_refused(world):
    world.pod()
    world.reading("lens", salts=False)
    with pytest.raises(ValueError, match="frozen interpreter"):
        E.hypotheses(world.find_run, "lens-reading", "lens",
                     [e[2:] for e in world.evidence("lens")])  # fmt: skip


def test_the_stability_package_is_read_by_its_own_salted_samples(world):
    world.pod()
    world.reading("lens")
    hyps = E.hypotheses(world.find_run, "lens-reading", "lens",
                        [e[2:] for e in world.evidence("lens")])  # fmt: skip
    assert (STATES["L"], f"adl_lens@{E.DRAWS[2]}") in {k for k, *_ in hyps.values()}
    rows = [{"state_id": s, "tool": t, "text": x} for _, _, s, t, _, x in world.evidence("lens")]
    salted = [r["tool"] for r, s in zip(rows, summarizer.stability_salts(rows), strict=True) if s]
    assert salted and all(t.endswith("/state/2") for t in salted)


def test_evidence_outside_the_lineage_is_refused(world):
    world.pod()
    world.reading("lens")
    _rewrite(
        world.find_run("lens-reading") / "summarizer_inputs.json",
        lambda d: d["evidence"].append({**d["evidence"][0], "tool": "adl_lens@30/x/1"}),
    )
    with pytest.raises(ValueError, match="not exactly the lineage"):
        E.hypotheses(world.find_run, "lens-reading", "lens",
                     [e[2:] for e in world.evidence("lens")])  # fmt: skip


def test_category_grades_are_reparsed_and_gate_draw_one(world):
    world.pod()
    world.reading("lens")
    world.reading("pd")
    world.judge("lens")
    world.judge("pd")
    world.category(wrong="eli5")
    data = record(world)
    assert data["category_confusion"]["passed"] is False and data["p3"] == "1"
    assert data["category_margins"][f"A/{E.DRAWS[1]}"] == "1"
    rows = (world.find_run("category") / E.CATEGORY_LABELS).read_text().splitlines()
    key = json.loads(rows[0])["request_key"]
    _rewrite(_entry_of(world, "category", key), lambda d: d.update(text="{}"))
    with pytest.raises(G.Pending, match="attempt 2 has no answer"):  # refused: its retry is due
        record(world)


def test_an_ungraded_counted_hypothesis_is_pending(world):
    world.build()
    path = world.find_run("category") / E.CATEGORY_LABELS
    path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(G.Pending, match="not graded"):
        record(world)


def test_the_category_definitions_are_pinned(world, monkeypatch):
    world.build()
    monkeypatch.setattr(E, "DEFINITIONS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="definitions changed"):
        record(world)


def test_pd_clean_s_provenance_is_its_study_s_verification(world, monkeypatch):
    from residual_disposition import pd_clean_study

    world.build()

    def unverified(work, record_run, found):
        raise ValueError("not a complete Stage-3 study")

    monkeypatch.setattr(pd_clean_study, "verify_packages", unverified)
    with pytest.raises(ValueError, match="not a complete Stage-3 study"):
        record(world)


def test_x_is_recomputed_from_its_composition(world):
    world.build()
    _rewrite(
        world.find_run("compose") / "summary.json", lambda d: d.update(state_id="s" + "f" * 16)
    )
    with pytest.raises(ValueError, match="composition of L and H"):
        record(world)


def test_written_once_and_recomputed_from_its_bound_files(world, tmp_path, monkeypatch):
    from residual_disposition import manifests

    sources = world.build()
    monkeypatch.setattr(E, "default_tests", lambda rules: factory())
    E.write_readers(tmp_path, world.lineage, sources, RULES, world.find_run)
    path = G.readers_path(tmp_path, world.identity["lineage"])
    with pytest.raises(ValueError, match="written once"):
        E.write_readers(tmp_path, world.lineage, sources, RULES, world.find_run)
    monkeypatch.setattr(manifests, "committed_json",
                        lambda p, what, repo=None: {**json.loads(Path(p).read_text()),
                                                    "sha256": "0" * 64})  # fmt: skip

    def verify():
        return E.committed_readers(path, tmp_path, RULES, world.find_run,
                                   lineage_for=lambda b: world.lineage)  # fmt: skip

    assert verify()["p3"] == "1/1000"
    _rewrite(
        world.find_run("lens-A") / "adl_lens_package.json",
        lambda d: d["layers"]["20"]["packages"]["state/1"].update(extra=1),
    )
    with pytest.raises(ValueError):
        verify()


def test_a_changed_label_table_changes_the_binding(world, tmp_path, monkeypatch):
    from residual_disposition import manifests

    sources = world.build()
    monkeypatch.setattr(E, "default_tests", lambda rules: factory())
    E.write_readers(tmp_path, world.lineage, sources, RULES, world.find_run)
    path = G.readers_path(tmp_path, world.identity["lineage"])
    monkeypatch.setattr(manifests, "committed_json",
                        lambda p, what, repo=None: {**json.loads(Path(p).read_text()),
                                                    "sha256": "0" * 64})  # fmt: skip
    labels = world.find_run("pd-judge") / "observations" / "judge_labels.jsonl"
    labels.write_text(labels.read_text() + "\n")
    with pytest.raises(ValueError, match="not what its sources give"):
        E.committed_readers(path, tmp_path, RULES, world.find_run,
                            lineage_for=lambda b: world.lineage)  # fmt: skip


# ------------------------------------------------------------------------------ X_cont


STRONG = {**IDENTITY, "lineage": "strong_eli5_cont_200", "behavior": "eli5", "dose": 200,
          "states": dict(STATES), "plan": {"sha256": "p"}, "reading": "strong"}  # fmt: skip


def strong_world(tmp_path, monkeypatch, valid):
    from residual_disposition import continuation_stage5 as S5

    w = World(tmp_path, monkeypatch, STRONG)
    w.run(
        "xcont-run",
        {"step": "xcont"},
        summary={
            "attempts": ["xa-1", "xa-2"],
            "valid": valid,
            "state": STATES["X"] if valid else None,
        },
    )
    for a in ("xa-1", "xa-2"):
        w.run(a, {"step": "xcont_attempt"}, {"xcont_evidence.json": {"a": a}})
    w.xcont = "xcont-run"
    monkeypatch.setattr(S5, "_xcont", lambda inputs, latent, dose, record, rules, find_run:
                        valid if inputs["xcont"] == "xcont-run" else None)  # fmt: skip
    monkeypatch.setattr(pipeline, "_xcont_params", lambda c, tol: {"c": c})
    s = {"lineage": "strong_eli5_cont_200", "dose": 200, "latent": STATES["L"],
         "plan": {"sha256": "p"}}  # fmt: skip
    xcont = E.strong_xcont(SimpleNamespace(), s, RULES, w.find_run)
    identity = {**STRONG, "states": {k: v for k, v in STATES.items() if k != "X"}}
    if xcont["valid"]:
        identity["states"]["X"] = xcont["state"]
    w.identity = identity
    w.lineage = E.Lineage(identity, SimpleNamespace(), {"lineage": s["lineage"]}, xcont)
    return w


def test_a_strong_lineage_s_x_is_its_valid_x_cont_and_read_as_a_reading(tmp_path, monkeypatch):
    w = strong_world(tmp_path, monkeypatch, True)
    assert w.lineage.xcont == {"run": "xcont-run", "valid": True, "state": STATES["X"]}
    assert w.identity["states"]["X"] == STATES["X"]
    data = record(w, w.build())
    assert data["status"] == "complete" and data["x_validity"]["inputs"]["control"] == "X_cont"
    assert G.reader_values(data, {}, w.identity)["x_valid"]


def test_an_invalid_x_cont_is_terminal_and_read_by_the_panel(tmp_path, monkeypatch):
    w = strong_world(tmp_path, monkeypatch, False)
    data = E.readers_record(w.lineage, {"xcont": "xcont-run"}, RULES, w.find_run)
    assert (data["status"], data["p3"], data["p4"]) == ("terminal", "1", "1")
    assert "X" not in data["states"] and set(data["bindings"]) == {"runs"}
    assert sorted(data["bindings"]["runs"]) == ["xa-1/xcont_evidence.json",
                                                "xa-2/xcont_evidence.json",
                                                "xcont-run/summary.json"]  # fmt: skip
    values = G.reader_values(data, {}, w.identity)
    assert values["status"] == "terminal" and values["p3"] == values["p4"] == 1
    with pytest.raises(ValueError, match="named as its source"):
        E.readers_record(w.lineage, {"xcont": "other"}, RULES, w.find_run)


def test_an_unbuilt_x_cont_is_pending(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch, STRONG)
    w.xcont = None
    monkeypatch.setattr(pipeline, "_xcont_params", lambda c, tol: {"c": c})
    s = {"lineage": "l", "dose": 200, "latent": STATES["L"], "plan": {}}
    with pytest.raises(G.Pending, match="not built"):
        E.strong_xcont(SimpleNamespace(), s, RULES, w.find_run)


def test_a_stack_lineage_has_no_terminal_reading(world):
    world.build()
    with pytest.raises(ValueError):
        E.readers_record(world.lineage, {"xcont": "x"}, RULES, world.find_run, tests=factory())


def test_the_category_letters_are_the_frozen_order():
    assert list(category_letters().values()) == list(PANEL_ORDER)


def test_the_panel_script_verifies_the_readers_by_their_recomputation(monkeypatch, tmp_path):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "continuation_panel.py"
    spec = importlib.util.spec_from_file_location("continuation_panel_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    calls = []

    def verify(p, root, rules, find_run, **kw):
        calls.append((p, root, kw))

    monkeypatch.setattr(E, "committed_readers", verify)
    script.readers_verifier(tmp_path, RULES)("m.json")
    assert calls == [("m.json", tmp_path, {"workers": 4})]
