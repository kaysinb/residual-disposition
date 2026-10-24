"""The readers' verdicts of a certified Stage-3 lineage (continuation_readers), on synthetic
readings only: rule C and the exact counts, every threshold at its boundary, p₃'s draw 1 and its
fully gated stability, p₄ with its positive control, the exact tests' refinement, X recomputed
(valid, invalid after its retry: terminal), every input bound and recomputed (texts, labels
reparsed, own target, samples 0–79, salts, tiers), pending work writing nothing, and the panel's
analysis reading the manifest (continuation_panel.reader_values)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from residual_disposition import continuation_panel as G
from residual_disposition import continuation_readers as E
from residual_disposition import continuation_rules as cr
from residual_disposition import summarizer
from residual_disposition.continuation_definitions import load_definitions
from residual_disposition.continuation_judge import PANEL_ORDER, category_letters, category_request
from residual_disposition.judge import JudgeCache, ModelSpec, request_key, rubric_set
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


# ------------------------------------------------------------------- the bound readings


IDENTITY = {"lineage": "stage3_tldr_400", "behavior": BEHAVIOR, "dose": 400, "contracts": "c" * 64,
            "detector": "tldr-v1", "states": STATES, "binding": {"b": 1},
            "plan": {"sha256": "q"}, "reading": "stack"}  # fmt: skip


class World:
    """A lineage's runs and caches, written as the Pod, scripts/summarize_tools.py,
    scripts/judge.py and the category grader write them, with designed labels."""

    def __init__(self, root: Path, monkeypatch, identity=IDENTITY):
        self.root, self.identity, self.runs = root, dict(identity), root / "runs"
        self.caches = {k: root / "cache" / k for k in ("summarizer", "judge", "category")}
        monkeypatch.setattr(summarizer, "adl_lens_evidence",
                            lambda pkg: json.dumps(pkg, sort_keys=True))  # fmt: skip
        monkeypatch.setattr(summarizer, "pd_evidence", lambda top: json.dumps(top, sort_keys=True))
        self.lens = lens_counts()
        self.pd = pd_counts()

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

    # -- the Pod's record, X and the packages
    def pod(self):
        s, ident = STATES, self.identity
        readers = {r: {"state": s[r], "check": {"passed": True}, "lens": f"lens-{r}",
                       "controls": r == "A", "pd": None if r == "H" else f"pool-{r}"}
                   for r in E.ROLES}  # fmt: skip
        record = {"behavior": BEHAVIOR, "lineage": ident["lineage"], "reading": "stack",
                  "dose": ident["dose"], "states": s, "binding": ident["binding"],
                  "contracts": {"sha256": ident["contracts"]},
                  "x_validity": {"control": "X", "state": s["X"], "run": "compose",
                                 "valid": True, "verified": True},
                  "readers": readers}  # fmt: skip
        params = {"dose": ident["dose"], "states": s, "binding": content_hash(ident["binding"]),
                  "contracts": ident["contracts"], "record": content_hash(record)}  # fmt: skip
        self.run("record", {"step": "stage3_readers", "params": params},
                 {"stage3_readers.json": json.dumps(record, indent=1)})  # fmt: skip
        self.run("compose", {"step": "compose_suppression_only",
                             "params": {"suppressed": s["L"], "sham": s["H"]}},
                 summary={"state_id": s["X"]})  # fmt: skip
        for r in E.ROLES:
            pk = {"state/1": {"r": r}, "state/2": {"r": r}, "base/1": {"r": r, "d": "base"}}
            if r == "A":
                pk |= {"random/1": {"r": r, "c": 1}, "no_patch/1": {"r": r, "c": 2}}
            lens = {"layers": {"30": {"packages": pk}, "20": {"packages": {"state/1": {"l": r}}}}}
            self.run(f"lens-{r}", {"step": "adl_lens_package", "params": {"state": s[r]}},
                     {"adl_lens_package.json": lens})  # fmt: skip
        self.pd_runs = []
        for r, kind in E.pd_inventory():
            name = f"{kind}-{r}"
            params = {"state": s[r], "behavior": BEHAVIOR, "role": r, "dose": None,
                      "adjudicated": r in E.ADJUDICATED, "pd_run": f"/runs/x/pool-{r}"}  # fmt: skip
            self.run(name, {"experiment": "pd_clean", "step": f"{kind}_package", "params": params},
                     {"pd_clean.json": {"state": s[r], "top": {"c": [r, kind]}}})  # fmt: skip
            self.pd_runs.append(name)

    def evidence(self, kind):
        if kind == "lens":
            out = []
            for r, p in E.lens_inventory():
                layer, key = p.split("/", 1)
                pkg = {"state/1": {"r": r}, "state/2": {"r": r}, "base/1": {"r": r, "d": "base"},
                       "random/1": {"r": r, "c": 1}, "no_patch/1": {"r": r, "c": 2}}[key] \
                    if layer == "30" else {"l": r}  # fmt: skip
                out.append((r, p, STATES[r], f"adl_lens@{p}", f"lens-{r}",
                            json.dumps(pkg, sort_keys=True)))  # fmt: skip
            return out
        return [(r, k, STATES[r], k, f"{k}-{r}", json.dumps({"c": [r, k]}, sort_keys=True))
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

    def reading(self, kind, *, drop=None, tier="flex", salts=True):
        name = f"{kind}-reading"
        ev = self.evidence(kind)
        version = E.INTERPRETER["prompts"][kind]
        prompt = summarizer.PROMPTS[version]
        model = ModelSpec("gemini-3.8-flash", "low", 0.75, 3.75, 950)
        settings = summarizer.SummarizerSettings(model=model, samples=80, max_tokens=4096)
        rows = [{"state_id": s, "tool": t, "text": text} for _, _, s, t, _, text in ev]
        salt = summarizer.stability_salts(rows)
        inputs = {
            "model": {
                "id": model.id,
                "reasoning_effort": "low",
                "input_usd_per_mtok": 0.75,
                "output_usd_per_mtok": 3.75,
                "expected_thinking_tokens": 950,
            },
            "samples": 80,
            "max_tokens": 4096,
            "stability_salts": salts,
            "prompts": {version: {"tool": prompt.tool, "sha256": prompt.sha256}},
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
        self.hyps = getattr(self, "hyps", {})
        for (role, package, s, t, run, text), sl in zip(ev, salt, strict=True):
            for i, req in enumerate(summarizer.summary_requests(prompt, text, settings, salt=sl)):
                if drop == (role, package, i):
                    continue
                h = f"hypothesis {role} {package} {i}"
                self.entry("summarizer", req.key, req.body, f"  {h}\n", tier)
                rid = f"{name}/{run}/{s}/{t}/all/{i}"
                records.append(ToolHypothesis(name, s, t, run, None, model.id, version, i, rid))
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

    def judge(self, kind, *, tier="standard", target=BEHAVIOR, skip=None):
        from residual_disposition.judge import load_targets

        name, version = f"{kind}-judge", E.RUBRICS[kind]
        rubrics = rubric_set(version)
        text_target = load_targets([BEHAVIOR])[BEHAVIOR]
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
            keys = []
            for rubric, model, text in (
                (rubrics.strict, E.JUDGE_MODELS["strict"],
                 json.dumps({"label": strict, "confident": True})),
                (rubrics.lenient, E.JUDGE_MODELS["lenient"], "yes" if lenient else "no"),
            ):  # fmt: skip
                body = {"model": model, "messages": rubric.messages(text_target, h),
                        "max_tokens": 1024}  # fmt: skip
                key = request_key(rubric.version, body)
                self.entry("judge", key, body, text, tier, rubric.version)
                keys.append(key)
            rows.append(JudgeLabel(rid, target, E.JUDGE_MODELS["strict"], version, strict,
                                   lenient, False, "primary", keys[0], keys[1]))  # fmt: skip
        RecordWriter(d).write(rows)
        return name

    def category(self, *, wrong=None, tier="standard"):
        definitions, digest = load_definitions()
        spec = ModelSpec("gemini-3.8-flash", "low", 0.75, 3.75, 120)
        letter = {b: k for k, b in category_letters().items()}
        lines = []
        for rid, (role, package, i, h) in self.hyps.items():
            strict, lenient = self.designed(role, package, i)
            if not E.counted(strict, lenient):
                continue
            behaviors = [wrong if wrong and role == "L" else BEHAVIOR]
            req = category_request(h, definitions, spec, max_tokens=2048)
            answer = json.dumps({"categories": [letter[b] for b in behaviors]})
            self.entry("category", req.key, req.body, answer, tier)
            lines.append(json.dumps({"response_id": rid, "request_key": req.key,
                                     "categories": behaviors}))  # fmt: skip
        inputs = {"version": "category-v1", "definitions_sha256": digest,
                  "model": {"id": spec.id, "reasoning_effort": "low"},
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

    def build(self, **skip):
        self.pod()
        self.reading("lens", **skip.get("lens", {}))
        self.reading("pd")
        self.judge("lens")
        self.judge("pd")
        self.category()
        return self.sources()


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def record(w, sources=None, **kw):
    return E.readers_record(w.find_run, sources or w.sources(), w.identity, RULES,
                            tests=factory(), **kw)  # fmt: skip


def test_a_complete_reading_is_counted_and_read_by_the_panel(world):
    world.build()
    found = E.readings(world.find_run, world.sources(), world.identity, RULES)
    assert found["lens"] == world.lens and found["pd"] == world.pd
    data = record(world)
    assert data["status"] == "complete" and data["p3"] == "1/1000"
    assert data["service_tiers"] == {"flex": 80 * 34}
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


def test_a_foreign_record_is_refused(world, tmp_path):
    world.build()
    for k, v in (("dose", 200), ("contracts", "d" * 64), ("binding", {"b": 2}),
                 ("lineage", "stage3_tldr_200"), ("behavior", "eli5"),
                 ("reading", "strong")):  # fmt: skip
        with pytest.raises(ValueError, match="not the readers of"):
            E.pod_record(world.find_run, "record", {**world.identity, k: v})


def test_a_missing_run_is_pending_and_writes_nothing(world, tmp_path):
    world.pod()
    world.reading("lens")
    with pytest.raises(G.Pending):
        record(world)
    with pytest.raises(G.Pending):
        E.write_readers(tmp_path, world.find_run, world.sources(), world.identity, RULES)
    assert not G.readers_path(tmp_path, world.identity["lineage"]).exists()


def test_a_package_short_of_a_sample_is_pending(world):
    world.build(lens={"drop": ("L", E.DRAWS[1], 17)})
    with pytest.raises(G.Pending, match="79 of its 80"):
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


def test_a_malformed_or_mismatched_label_is_refused(world):
    world.build()
    lab = next(iter((world.find_run("lens-judge") / "observations" / "judge_labels.jsonl")
                    .read_text().splitlines()))  # fmt: skip
    strict_key = json.loads(lab)["request_id"]
    path = _entry_of(world, "judge", strict_key)
    _rewrite(path, lambda d: d.update(text="not json"))
    with pytest.raises(ValueError, match="malformed"):
        record(world)
    _rewrite(path, lambda d: d.update(text=json.dumps({"label": "other", "confident": True})))
    with pytest.raises(ValueError, match="not its answers"):
        record(world)


def test_a_judge_answer_off_the_standard_tier_is_refused(world):
    world.pod()
    world.reading("lens")
    world.reading("pd")
    world.judge("lens", tier="flex")
    world.judge("pd")
    world.category()
    with pytest.raises(ValueError, match="not for its text"):
        record(world)


def test_a_hypothesis_text_that_is_not_its_completion_is_refused(world):
    world.build()
    path = world.find_run("lens-reading") / "responses" / "tool_hypotheses.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    row["text"] = "edited"
    path.write_text("\n".join([json.dumps(row), *lines[1:]]) + "\n")
    with pytest.raises(ValueError, match="not its completion"):
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
    entries = {k for k, *_ in hyps.values()}
    assert (STATES["L"], f"adl_lens@{E.DRAWS[2]}") in entries
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
    with pytest.raises(ValueError, match="malformed"):
        record(world)


def test_an_ungraded_counted_hypothesis_is_pending(world):
    world.build()
    path = world.find_run("category") / E.CATEGORY_LABELS
    path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(G.Pending, match="not graded"):
        record(world)


def test_a_pd_package_of_another_pool_is_refused(world):
    world.build()
    _rewrite(
        world.find_run("pd_clean-L") / "config.yaml",
        lambda d: d["params"].update(pd_run="/runs/x/pool-other"),
    )
    with pytest.raises(ValueError, match="of its pool"):
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
    E.write_readers(tmp_path, world.find_run, sources, world.identity, RULES)
    path = G.readers_path(tmp_path, world.identity["lineage"])
    with pytest.raises(ValueError, match="written once"):
        E.write_readers(tmp_path, world.find_run, sources, world.identity, RULES)
    monkeypatch.setattr(manifests, "committed_json",
                        lambda p, what, repo=None: {**json.loads(Path(p).read_text()),
                                                    "sha256": "0" * 64})  # fmt: skip

    def verify():
        return E.committed_readers(path, tmp_path, RULES, world.find_run, contracts="c" * 64,
                                   identity_of=lambda b: world.identity)  # fmt: skip

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
    E.write_readers(tmp_path, world.find_run, sources, world.identity, RULES)
    path = G.readers_path(tmp_path, world.identity["lineage"])
    monkeypatch.setattr(manifests, "committed_json",
                        lambda p, what, repo=None: {**json.loads(Path(p).read_text()),
                                                    "sha256": "0" * 64})  # fmt: skip
    labels = world.find_run("pd-judge") / "observations" / "judge_labels.jsonl"
    labels.write_text(labels.read_text() + "\n")
    with pytest.raises(ValueError, match="not what its sources give"):
        E.committed_readers(path, tmp_path, RULES, world.find_run, contracts="c" * 64,
                            identity_of=lambda b: world.identity)  # fmt: skip


# ------------------------------------------------------------------------------ X_cont


STRONG = {
    **IDENTITY,
    "lineage": "strong_eli5_cont_200",
    "behavior": "eli5",
    "dose": 200,
    "states": {k: v for k, v in STATES.items() if k != "X"},
    "plan": {"sha256": "p"},
    "reading": "strong",
}


def strong_world(tmp_path, monkeypatch, valid):
    from residual_disposition import continuation_stage5 as S5

    w = World(tmp_path, monkeypatch, STRONG)
    w.run("xcont-run", {"step": "xcont"}, summary={"attempts": ["xa-1", "xa-2"], "valid": valid})
    for a in ("xa-1", "xa-2"):
        w.run(a, {"step": "xcont_attempt"}, {"xcont_evidence.json": {"a": a}})
    monkeypatch.setattr(S5, "_xcont", lambda inputs, latent, dose, record, rules, find_run:
                        valid if inputs["xcont"] == "xcont-run" else None)  # fmt: skip
    return w


def test_an_invalid_x_cont_is_terminal_and_read_by_the_panel(tmp_path, monkeypatch):
    w = strong_world(tmp_path, monkeypatch, False)
    data = E.readers_record(w.find_run, {"xcont": "xcont-run"}, w.identity, RULES)
    assert (data["status"], data["p3"], data["p4"]) == ("terminal", "1", "1")
    assert "X" not in data["states"] and set(data["bindings"]) == {"runs"}
    assert sorted(data["bindings"]["runs"]) == ["xa-1/xcont_evidence.json",
                                                "xa-2/xcont_evidence.json",
                                                "xcont-run/summary.json"]  # fmt: skip
    values = G.reader_values(data, {}, w.identity)
    assert values["status"] == "terminal" and values["p3"] == values["p4"] == 1


def test_a_valid_or_unbuilt_x_cont_is_not_terminal(tmp_path, monkeypatch):
    w = strong_world(tmp_path, monkeypatch, True)
    with pytest.raises(ValueError, match="valid; its readers are read"):
        E.readers_record(w.find_run, {"xcont": "xcont-run"}, w.identity, RULES)
    with pytest.raises(G.Pending):
        E.readers_record(w.find_run, {"xcont": "other"}, w.identity, RULES)


def test_a_stack_lineage_has_no_terminal_reading(world):
    with pytest.raises(ValueError, match="strong lineage"):
        E.readers_record(world.find_run, {"xcont": "x"}, world.identity, RULES)


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
