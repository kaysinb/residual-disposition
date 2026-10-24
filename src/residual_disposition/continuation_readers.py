"""The readers' verdicts of a certified Stage-3 lineage (preregistration §12.4.2–§12.4.6, §12.5,
with §12.4.5's recorded readings of 2026-10-24), on the laptop, from bound readings alone.

- **Counts.** Rule C counts a hypothesis when its lenient label is true and its strict label is
  not ``avoids_behavior``. Every package holds exactly 80 hypotheses, samples 0–79 once each.
- **Component 3, lens-only ADL**, per λ draw d at layer 30 (``30/state/1``, ``30/state/2``), the
  preliminary verdict v_d holds when all of these hold:
  - eligibility on the draw's decoder: R(A) ≥ 4/5, R(H) ≤ 1/10, R(A) − R(H) ≥ 3/5 and
    R(recipe) ≥ 4/5 · R(A);
  - fidelity (both draws): A's random-direction and no-patch controls, each R ≤ 1/10, and the
    common base decoder (``30/base/1``) meeting the four eligibility inequalities;
  - category confusion of L's counted hypotheses of the draw: the share matching the correct
    behavior minus the share matching the single largest other one ≥ 1/2; none counted fails;
  - the exact test of §12.5.1, R(L) − R(H_s) > 1/2 and R(L) − R(X) > 1/5, p_d the larger
    certified bound, p_d ≤ 1/20.

  p₃ = p₁ (draw 1's) when draw 1's eligibility, fidelity and category confusion pass and
  v₁ = v₂ (stability); otherwise 1. Draw 2 never replaces draw 1; draws are never pooled.
  Every counted hypothesis is graded; the other states' margins are reported, not gates.
- **Component 4, PD-clean non-identification:** the positive control R(A) ≥ 4/5 and
  R(recipe) ≥ 4/5 on their clean packages; then p₄ is the largest of the TOSTs of R(L) − R(H_s)
  and of R(L) − R(X) within ±1/10 and the one-sample test of R(L) < 1/10; 1 without the control.
  The random-removal controls are reported.
- **X** is recomputed from its own records: a stack lineage's one composition of L and H, a
  strong lineage's X_cont cell through its attempts' evidence (``continuation_stage5._xcont``).
  An X_cont invalid after its retry makes the readers terminal, p₃ = p₄ = 1, with no reading.

Every comparison is exact; every threshold is the frozen rules' (``rules.readers``). A p-value is
the exact test's certified upper bound at its stopping gap, refined where its enclosure straddles
1/20 or 1/60 (the levels the analysis compares it with); each enclosure is kept. The enclosures
are numerical, not effect-size confidence intervals.

**What binds a reading** (``readings``), each recomputed, never trusted:
- the Pod record (``stage3_readers``) of the lineage's binding, contracts, states and checks;
- the lens packages of its runs and the PD-clean packages of its pools, each by sha256, whose
  evidence texts the interpreter run read exactly (no other evidence) with its frozen model,
  prompts, 80 samples and stability salts; each hypothesis is the cache's completion of its
  recomputed request, parsed again, with its response's tier;
- the judge's labels for the own target (rubric v4 for the lens hypotheses, v4-j3 for
  PD-clean's): one primary label per hypothesis, its request bodies the rubrics' messages of
  the hypothesis text and the frozen models, on the standard tier, and each label the parse of
  its cached answer;
- the category grades (``category-v1``) of every counted hypothesis, each request recomputed
  from its text, the frozen definitions and model, on the standard tier, each grade reparsed.

``readers_record`` gives the manifest the panel's analysis reads (``continuation_panel.
reader_values``); ``write_readers`` writes it once; ``committed_readers`` recomputes it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .continuation_panel import LEVEL, PANEL_LEVEL, Pending, readers_path

SAMPLES = 80
DRAWS = {1: "30/state/1", 2: "30/state/2"}
BASE, LAYER20 = "30/base/1", "20/state/1"
CONTROLS = ("30/random/1", "30/no_patch/1")  # on A only
LENS_PACKAGES = (DRAWS[1], DRAWS[2], BASE, LAYER20)
PD_CLEAN, PD_RANDOM = "pd_clean", "pd_random"
ROLES = ("A", "H", "L", "H_s", "X", "recipe")
PD_CLEAN_ROLES = ("L", "H_s", "X", "A", "recipe")
PD_RANDOM_ROLES = ("L", "A", "H_s")
ADJUDICATED = ("L", "A", "recipe")  # the others get the detector's removal only
REFINE_AT = (LEVEL, PANEL_LEVEL)  # the levels a p-value of the readers is compared with
SOURCES = ("record", "lens_reading", "lens_judge", "pd_reading", "pd_judge", "category")

# The frozen instruments (§12.3.3; configs/experiments/summarize_tools.yaml, judge_continuation.
# yaml; the category model recorded 2026-10-24).
INTERPRETER = {"model": "gemini-3.8-flash", "reasoning_effort": "low", "max_tokens": 4096,
               "prompts": {"lens": "adl-lens-v1", "pd": "pd-v1"}}  # fmt: skip
RUBRICS = {"lens": "v4", "pd": "v4-j3"}
JUDGE_MODELS = {"strict": "gemini-3.5-flash-lite", "lenient": "gemini-3.8-flash"}
CATEGORY_MODEL = {"id": "gemini-3.8-flash", "reasoning_effort": "low"}
CATEGORY_MAX_TOKENS = 2048  # continuation_judge.category_request's cap
STANDARD = "standard"  # the judge's and the category grader's tier
EXPERIMENTS = {"lens": "summarize_stage3_lens", "pd": "summarize_stage3_pd", "judge": "judge",
               "category": "category_stage3"}  # fmt: skip
CATEGORY_INPUTS, CATEGORY_LABELS = "category_inputs.json", "category_labels.jsonl"


def lens_inventory() -> list[tuple[str, str]]:
    """The 26 lens packages read: four per state, A's two controls."""
    return [(r, p) for r in ROLES for p in LENS_PACKAGES] + [("A", c) for c in CONTROLS]


def pd_inventory() -> list[tuple[str, str]]:
    """The 8 PD-clean packages read: five clean, three random-removal controls."""
    return [(r, PD_CLEAN) for r in PD_CLEAN_ROLES] + [(r, PD_RANDOM) for r in PD_RANDOM_ROLES]


# ---------------------------------------------------------------------------------- counts


def counted(strict: str, lenient: bool) -> bool:
    """Rule C."""
    if not isinstance(lenient, bool):
        raise ValueError(f"a lenient label {lenient!r} is not a bool")
    return lenient and strict != "avoids_behavior"


def package_count(samples: Mapping[int, bool], where: str) -> int:
    """A package's count: exactly the samples 0–79 once each, each counted or not."""
    if sorted(samples) != list(range(SAMPLES)):
        raise ValueError(f"{where}: not the samples 0–{SAMPLES - 1} once each")
    if not all(isinstance(v, bool) for v in samples.values()):
        raise ValueError(f"{where}: a count that is not a bool")
    return sum(samples.values())


def _rate(x: int) -> Fraction:
    return Fraction(x, SAMPLES)


def thresholds(rules: Any) -> dict[str, Fraction]:
    f = rules.fraction
    return {
        "active_min": f("readers", "eligibility", "active_min"),
        "sham_max": f("readers", "eligibility", "sham_max"),
        "gap_min": f("readers", "eligibility", "gap_min"),
        "recipe_ratio": f("readers", "eligibility", "recipe_ratio"),
        "fidelity_max": f("readers", "fidelity_max"),
        "confusion_min": f("readers", "confusion_min"),
        "positive_control_min": f("readers", "positive_control_min"),
        "excess_over_sham_s": f("readers", "adl", "excess_over_sham_s"),
        "excess_over_x": f("readers", "adl", "excess_over_x"),
        "pd_margin": f("readers", "pd", "equivalence_margin"),
        "pd_absolute_max": f("readers", "pd", "absolute_max"),
    }


# ------------------------------------------------------------------------- the exact tests

Job = tuple[Fraction, str, int, int]  # (δ₀, form, x_L, x_C)
TestFactory = Callable[[Fraction, str], Any]  # (δ₀, form) → a test with p_value(x_l, x_c, target)


def component3_jobs(lens: Mapping[tuple[str, str], int], th: Mapping[str, Fraction]) -> list[Job]:
    jobs = []
    for package in DRAWS.values():
        x_l = lens[("L", package)]
        jobs.append((th["excess_over_sham_s"], ">", x_l, lens[("H_s", package)]))
        jobs.append((th["excess_over_x"], ">", x_l, lens[("X", package)]))
    return jobs


def component4_jobs(pd: Mapping[tuple[str, str], int], th: Mapping[str, Fraction]) -> list[Job]:
    x_l, margin = pd[("L", PD_CLEAN)], th["pd_margin"]
    jobs = []
    for c in ("H_s", "X"):
        x_c = pd[(c, PD_CLEAN)]
        jobs += [(-margin, ">", x_l, x_c), (margin, "<", x_l, x_c)]
    return jobs


def _enclosure(found: Mapping[str, Any], target: Fraction | None) -> dict[str, Any]:
    p = Fraction(found["p"])
    return {"target": target, "p": p, "lower": Fraction(found.get("lower", p)),
            "empty": bool(found.get("empty"))}  # fmt: skip


def exact_p(test: Any, x_l: int, x_c: int) -> dict[str, Any]:
    """A test's certified p at its stopping gap, refined where the enclosure straddles a level
    of ``REFINE_AT`` (each refinement is certified too; the smallest upper bound is kept), with
    every enclosure and the certified decision at each level: "reject" (p ≤ level), "accept"
    (lower bound > level) or "undecided" (not rejected in use)."""
    found = [_enclosure(test.p_value(x_l, x_c), None)]
    p, lower = found[0]["p"], found[0]["lower"]
    for level in REFINE_AT:
        if lower <= level < p:
            found.append(_enclosure(test.p_value(x_l, x_c, target=level), level))
            p, lower = min(p, found[-1]["p"]), max(lower, found[-1]["lower"])
    decisions = {str(level): "reject" if p <= level else "accept" if lower > level
                 else "undecided" for level in REFINE_AT}  # fmt: skip
    return {"p": p, "lower": lower, "decisions": decisions, "enclosures": found}


def default_tests(rules: Any) -> TestFactory:
    """§12.5.1's exact test from the frozen rules (``simulate_continuation.make_test``)."""
    from .simulate_continuation import make_test

    return lambda delta, form: make_test(rules, SAMPLES, delta, form)


def _job(args: tuple[Any, Job]) -> dict[str, Any]:
    rules, (delta, form, x_l, x_c) = args
    return exact_p(default_tests(rules)(delta, form), x_l, x_c)


def run_tests(jobs: Sequence[Job], rules: Any, *, tests: TestFactory | None = None,
              workers: int = 0) -> dict[Job, dict[str, Any]]:  # fmt: skip
    """Each distinct job's ``exact_p``: in ``workers`` processes under the frozen rules' test,
    serially with ``tests`` (or with no workers)."""
    distinct = sorted(set(jobs))
    if tests is not None or workers <= 1:
        make = tests or default_tests(rules)
        return {j: exact_p(make(j[0], j[1]), j[2], j[3]) for j in distinct}
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        found = list(pool.map(_job, [(rules, j) for j in distinct]))
    return dict(zip(distinct, found, strict=True))


def _test(p: Mapping[Job, Mapping[str, Any]], job: Job) -> dict[str, Any]:
    delta, form, x_l, x_c = job
    return {"delta": delta, "form": form, "x_L": x_l, "x_C": x_c, **p[job]}


# ---------------------------------------------------------------------------- component 3


def eligibility(x: Mapping[str, int], th: Mapping[str, Fraction]) -> dict[str, Any]:
    """§12.4.3's four inequalities on one decoder's counts of A, H and the recipe control."""
    a, h, recipe = _rate(x["A"]), _rate(x["H"]), _rate(x["recipe"])
    checks = {
        "R(A) >= active_min": a >= th["active_min"],
        "R(H) <= sham_max": h <= th["sham_max"],
        "R(A) - R(H) >= gap_min": a - h >= th["gap_min"],
        "R(recipe) >= recipe_ratio * R(A)": recipe >= th["recipe_ratio"] * a,
    }
    return {"passed": all(checks.values()),
            "inputs": {"R(A)": a, "R(H)": h, "R(recipe)": recipe, "checks": checks,
                       "thresholds": {k: th[k] for k in ("active_min", "sham_max", "gap_min",
                                                         "recipe_ratio")}}}  # fmt: skip


def confusion(matches: Sequence[frozenset[str]], behavior: str, th: Mapping[str, Fraction]) -> dict:
    """§12.3.3's category confusion of one package's counted hypotheses
    (``continuation_judge.category_confusion``): the correct behavior's share minus the single
    largest other behavior's, over the counted hypotheses; none counted fails."""
    from .continuation_judge import category_confusion

    m = category_confusion(list(matches), behavior)
    passed = len(matches) > 0 and m.margin >= th["confusion_min"]
    return {"passed": passed,
            "inputs": {"counted": len(matches), "correct": m.correct,
                       "best_other": m.best_other, "best_other_share": m.best_other_share,
                       "margin": m.margin, "confusion_min": th["confusion_min"]}}  # fmt: skip


def component3(
    lens: Mapping[tuple[str, str], int], categories: Mapping[str, Sequence[frozenset[str]]],
    behavior: str, th: Mapping[str, Fraction], p: Mapping[Job, Mapping[str, Any]],
) -> dict[str, Any]:  # fmt: skip
    """p₃ with its gates (module docstring). ``lens`` holds each (role, package)'s count;
    ``categories`` each draw's L's counted hypotheses' category matches; ``p`` the tests."""
    base = eligibility({r: lens[(r, BASE)] for r in ("A", "H", "recipe")}, th)
    controls = {c: _rate(lens[("A", c)]) for c in CONTROLS}
    fidelity = {"passed": all(v <= th["fidelity_max"] for v in controls.values())
                and base["passed"],
                "inputs": {"controls": controls, "fidelity_max": th["fidelity_max"],
                           "base_decoder": base}}  # fmt: skip
    jobs = component3_jobs(lens, th)
    draws = {}
    for d, package in DRAWS.items():
        x = {r: lens[(r, package)] for r in ROLES}
        elig = eligibility(x, th)
        conf = confusion(categories[package], behavior, th)
        tests = [_test(p, j) for j in jobs[2 * (d - 1) : 2 * d]]
        p_d = max(t["p"] for t in tests)
        rejected = all(t["decisions"][str(LEVEL)] == "reject" for t in tests)
        verdict = elig["passed"] and fidelity["passed"] and conf["passed"] and rejected
        draws[d] = {"package": package, "counts": x, "eligibility": elig,
                    "category_confusion": conf, "tests": tests, "p": p_d,
                    "rejected": rejected, "verdict": verdict}  # fmt: skip
    stable = draws[1]["verdict"] == draws[2]["verdict"]
    gates = (draws[1]["eligibility"]["passed"], fidelity["passed"],
             draws[1]["category_confusion"]["passed"], stable)  # fmt: skip
    return {
        "p3": draws[1]["p"] if all(gates) else Fraction(1),
        "draws": draws,
        "flags": {
            "eligibility": {"passed": gates[0], "inputs": draws[1]["eligibility"]["inputs"]},
            "fidelity": fidelity,
            "category_confusion": {
                "passed": gates[2],
                "inputs": {str(d): draws[d]["category_confusion"]["inputs"] for d in DRAWS},
            },
            "stability": {"passed": stable, "inputs": {str(d): draws[d]["verdict"] for d in DRAWS}},
        },  # fmt: skip
        "layer20": {r: _rate(lens[(r, LAYER20)]) for r in ROLES},
    }


# ---------------------------------------------------------------------------- component 4


def component4(pd: Mapping[tuple[str, str], int], th: Mapping[str, Fraction],
               p: Mapping[Job, Mapping[str, Any]]) -> dict[str, Any]:  # fmt: skip
    """p₄ with its positive control (module docstring); the random-removal controls reported."""
    from .continuation_stats import rate_below_p

    x = {r: pd[(r, PD_CLEAN)] for r in PD_CLEAN_ROLES}
    a, recipe = _rate(x["A"]), _rate(x["recipe"])
    least = th["positive_control_min"]
    control = {
        "passed": a >= least and recipe >= least,
        "inputs": {"R(A)": a, "R(recipe)": recipe, "positive_control_min": least},
    }
    jobs = component4_jobs(pd, th)
    tost = {}
    for i, c in enumerate(("H_s", "X")):
        lower, upper = _test(p, jobs[2 * i]), _test(p, jobs[2 * i + 1])
        tost[c] = {"lower": lower, "upper": upper, "p": max(lower["p"], upper["p"])}
    one = rate_below_p(x["L"], SAMPLES, th["pd_absolute_max"])
    largest = max(tost["H_s"]["p"], tost["X"]["p"], one)
    return {
        "p4": largest if control["passed"] else Fraction(1),
        "pd_positive_control": control,
        "tost": tost,
        "one_sample": {"x_L": x["L"], "bound": th["pd_absolute_max"], "p": one},
        "counts": x,
        "random_controls": {r: _rate(pd[(r, PD_RANDOM)]) for r in PD_RANDOM_ROLES},
    }


# ------------------------------------------------------------------------ bound inputs


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(find_run: Any, name: str | None, step: str | None = None,
         experiment: str | None = None) -> tuple[Path, dict, dict]:  # fmt: skip
    """(dir, config, summary) of a completed run; Pending while it is absent or unfinished, an
    error when it is another step's or experiment's."""
    import yaml

    path = find_run(name) if isinstance(name, str) and name else None
    if path is None:
        raise Pending(f"{name or step or experiment}: no run here")
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("status") != "completed":
        raise Pending(f"{name}: not completed")
    config = yaml.safe_load((path / "config.yaml").read_text()) or {}
    if step is not None and config.get("step") != step:
        raise ValueError(f"{name}: a {config.get('step')} cell, not {step}")
    if experiment is not None and config.get("experiment") != experiment:
        raise ValueError(f"{name}: an {config.get('experiment')} run, not {experiment}")
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    return path, config, summary


def pod_record(find_run: Any, name: str, identity: Mapping[str, Any]) -> tuple[Path, dict]:
    """The Pod's readers' record of the lineage: its cell's identity recomputed (dose, states,
    binding, contracts, the record's hash) and every state's check passed."""
    from .pipeline import content_hash

    run, config, _ = _run(find_run, name, "stage3_readers")
    record = json.loads((run / "stage3_readers.json").read_text())
    want = {"dose": identity["dose"], "states": identity["states"],
            "binding": content_hash(identity["binding"]), "contracts": identity["contracts"],
            "record": content_hash(record)}  # fmt: skip
    params = config.get("params") or {}
    wrong = sorted(k for k, v in want.items() if params.get(k) != v)
    fields = {"behavior": identity["behavior"], "lineage": identity["lineage"],
              "dose": identity["dose"], "states": identity["states"],
              "binding": identity["binding"], "reading": identity["reading"]}  # fmt: skip
    wrong += sorted(k for k, v in fields.items() if record.get(k) != v)
    if (record.get("contracts") or {}).get("sha256") != identity["contracts"]:
        wrong.append("contracts")
    if wrong:
        raise ValueError(f"{name}: not the readers of {identity['lineage']} at {wrong}")
    readers = record.get("readers") or {}
    if sorted(readers) != sorted(ROLES):
        raise ValueError(f"{name}: the states {sorted(readers)}, not {sorted(ROLES)}")
    for role, r in readers.items():
        if r.get("state") != identity["states"][role] or (r.get("check") or {}).get("passed") \
                is not True or r.get("controls") is not (role == "A") \
                or (r.get("pd") is None) is not (role == "H"):  # fmt: skip
            raise ValueError(f"{name}: {role}'s reading is not its state's, checked")
    return run, record


def x_validity(find_run: Any, record: Mapping[str, Any], identity: Mapping[str, Any],
               rules: Any) -> dict[str, Any]:  # fmt: skip
    """X recomputed from its own records (module docstring): the flag with its evidence's sha256.
    An invalid X_cont raises: its readers are terminal (``terminal_record``)."""
    from . import continuation_stage5 as S5

    x, states = record.get("x_validity") or {}, identity["states"]
    if identity["reading"] == "stack":  # the one composition of L and H
        run, config, summary = _run(find_run, x.get("run"), "compose_suppression_only")
        params = config.get("params") or {}
        if (params.get("suppressed"), params.get("sham"), str(summary.get("state_id"))) != (
            states["L"],
            states["H"],
            states["X"],
        ) or x.get("control") != "X":
            raise ValueError(f"{x.get('run')}: not X = the composition of L and H")
        evidence = run / "summary.json"
    else:
        name = x.get("run")
        valid = S5._xcont({"xcont": name}, states["L"], identity["dose"],
                          {"plan": identity["plan"]}, rules, find_run)  # fmt: skip
        run, _, summary = _run(find_run, name, S5.XCONT)
        if valid is not True or summary.get("state") != states["X"] or x.get("control") != "X_cont":
            raise ValueError(f"{name}: X_cont {states.get('X')} is not valid")
        evidence = run / "summary.json"
    if x.get("state") != states["X"] or x.get("valid") is not True:
        raise ValueError(f"the record's X {x} is not the lineage's valid X")
    return {"passed": True, "inputs": {"control": x["control"], "state": states["X"],
                                       "run": run.name, "sha256": file_sha(evidence)}}  # fmt: skip


def _xcont_runs(find_run: Any, name: str) -> dict[str, str]:
    from . import continuation_stage5 as S5

    run, _, summary = _run(find_run, name, S5.XCONT)
    out = {f"{run.name}/summary.json": file_sha(run / "summary.json")}
    for attempt in summary.get("attempts") or []:
        a, _, _ = _run(find_run, attempt, S5.XCONT_ATTEMPT)
        out[f"{a.name}/xcont_evidence.json"] = file_sha(a / "xcont_evidence.json")
    return out


def lens_evidence(find_run: Any, record: Mapping[str, Any]) -> tuple[dict, dict]:
    """Each lens package's evidence {(role, package): (state, tool, run, text)} from the record's
    lens runs, and the package files by sha256."""
    from .summarizer import adl_lens_evidence

    out, files = {}, {}
    for role in ROLES:
        r = record["readers"][role]
        run, config, _ = _run(find_run, r["lens"], "adl_lens_package")
        if (config.get("params") or {}).get("state") != r["state"]:
            raise ValueError(f"{r['lens']}: not {role}'s lens package")
        path = run / "adl_lens_package.json"
        files[f"{run.name}/adl_lens_package.json"] = file_sha(path)
        layers = json.loads(path.read_text())["layers"]
        for package in LENS_PACKAGES + (CONTROLS if role == "A" else ()):
            layer, key = package.split("/", 1)
            try:
                pkg = layers[layer]["packages"][key]
            except KeyError as error:
                raise ValueError(f"{r['lens']}: no package {package}") from error
            out[(role, package)] = (r["state"], f"adl_lens@{package}", run.name,
                                    adl_lens_evidence(pkg))  # fmt: skip
    return out, files


def pd_evidence_of(find_run: Any, record: Mapping[str, Any], identity: Mapping[str, Any],
                   runs: Sequence[str]) -> tuple[dict, dict]:  # fmt: skip
    """Each PD-clean package's evidence {(role, kind): (state, tool, run, text)} from ``runs``
    (the package runs the interpreter read), each the role's package of the record's pool, and
    the package files by sha256."""
    from .summarizer import pd_evidence

    out, files = {}, {}
    for name in runs:
        run, config, _ = _run(find_run, name, None, "pd_clean")
        kind = {"pd_clean_package": PD_CLEAN, "pd_random_package": PD_RANDOM}.get(
            config.get("step"))  # fmt: skip
        params = config.get("params") or {}
        role = params.get("role")
        r = record["readers"].get(role) if isinstance(role, str) else None
        if kind is None or r is None or (role, kind) in out:
            raise ValueError(f"{name}: not one PD-clean package of the lineage's states")
        package = json.loads((run / "pd_clean.json").read_text())
        if (
            params.get("state") != r["state"] or package.get("state") != r["state"]
            or params.get("behavior") != identity["behavior"]
            or Path(str(params.get("pd_run"))).name != r["pd"]
            or params.get("adjudicated") is not (role in ADJUDICATED)
        ):  # fmt: skip
            raise ValueError(f"{name}: not {role}'s {kind} package of its pool {r['pd']}")
        files[f"{run.name}/pd_clean.json"] = file_sha(run / "pd_clean.json")
        out[(role, kind)] = (r["state"], kind, run.name, pd_evidence(package["top"]))
    if sorted(out) != sorted(pd_inventory()):
        raise Pending(f"the PD-clean packages read are {sorted(out)}, not the eight")
    return out, files


def _cache(config: Mapping[str, Any], section: str, default: str) -> Any:
    from .judge import JudgeCache
    from .paths import data_root

    root = (config.get(section) or {}).get("cache_dir")
    return JudgeCache(Path(root) if root else data_root() / "cache" / default)


def hypotheses(find_run: Any, name: str, kind: str,
               evidence: Sequence[tuple[str, str, str, str]]) -> dict:  # fmt: skip
    """The interpreter run's hypotheses {response id: (key, sample, text, tier)}, each the
    cache's completion of its recomputed request, parsed again, over exactly ``evidence``
    (``kind`` "lens" or "pd"), 80 per package, with the frozen model, prompt and salts."""
    from .judge import MalformedOutput, ModelSpec
    from .pipeline import content_hash
    from .records import ToolHypothesis, read_responses, read_table
    from .summarizer import (
        PROMPTS,
        SummarizerSettings,
        base_tool,
        parse_hypothesis,
        stability_salts,
        summary_requests,
    )

    run, config, _ = _run(find_run, name, None, EXPERIMENTS[kind])
    inputs = json.loads((run / "summarizer_inputs.json").read_text())
    model = inputs.get("model") or {}
    if (
        (model.get("id"), model.get("reasoning_effort"), inputs.get("max_tokens"))
        != (INTERPRETER["model"], INTERPRETER["reasoning_effort"], INTERPRETER["max_tokens"])
        or inputs.get("samples") != SAMPLES or inputs.get("stability_salts") is not True
    ):  # fmt: skip
        raise ValueError(f"{name}: not the frozen interpreter's reading")
    run_of = {(s, t): r for s, t, r, _ in evidence}
    text_of = {(s, t): text for s, t, _, text in evidence}
    want = sorted(({"state_id": s, "tool": t, "tool_run_id": r, "budget": None,
                    "sha256": content_hash(text)} for s, t, r, text in evidence),
                  key=lambda e: (e["state_id"], e["tool"]))  # fmt: skip
    rows = inputs.get("evidence") or []
    if sorted(rows, key=lambda e: (e["state_id"], e["tool"])) != want:
        raise ValueError(f"{name}: its evidence is not exactly the lineage's packages")
    version = INTERPRETER["prompts"][kind]
    prompt = PROMPTS[version]
    if inputs.get("prompts") != {version: {"tool": prompt.tool, "sha256": prompt.sha256}}:
        raise ValueError(f"{name}: not read with the frozen prompt {version}")
    ordered = [{"state_id": e["state_id"], "tool": e["tool"],
                "text": text_of[(e["state_id"], e["tool"])]} for e in rows]  # fmt: skip
    salts = stability_salts(ordered)
    spec = ModelSpec(model["id"], model["reasoning_effort"], 0.0, 0.0, 0)
    settings = SummarizerSettings(model=spec, samples=SAMPLES, max_tokens=INTERPRETER["max_tokens"])
    requests = {(e["state_id"], e["tool"]): summary_requests(prompt, e["text"], settings, salt=salt)
                for e, salt in zip(ordered, salts, strict=True)}  # fmt: skip
    cache = _cache(config, "summarizer", "summarizer")
    tiers = json.loads((run / "service_tiers.json").read_text()).get("hypotheses") or {}
    texts = read_responses([run], ToolHypothesis.table)
    out: dict[str, tuple] = {}
    for h in read_table([run], ToolHypothesis.table):
        key = (h.state_id, h.tool)
        if key not in requests or base_tool(h.tool) != prompt.tool or h.prompt_version != version \
                or h.summarizer_model != INTERPRETER["model"] or h.budget is not None \
                or not 0 <= h.sample < SAMPLES or h.response_id in out:  # fmt: skip
            raise ValueError(f"{h.response_id}: not one hypothesis of the planned evidence")
        if run_of[key] != h.tool_run_id:
            raise ValueError(f"{h.response_id}: read from another run than its package's")
        request = requests[key][h.sample]
        entry = cache.entry(request.key)
        if entry is None or entry.get("body") != request.body:
            raise ValueError(f"{h.response_id}: no cached completion of its request")
        try:
            text = parse_hypothesis(entry.get("text"))
        except MalformedOutput as error:
            raise ValueError(f"{h.response_id}: its completion is malformed") from error
        if texts.get(h.response_id) != text:
            raise ValueError(f"{h.response_id}: its text is not its completion's")
        tier = tiers.get(h.response_id)
        if not isinstance(tier, str) or entry.get("service_tier") != tier:
            raise ValueError(f"{h.response_id}: its tier {tier!r} is not its response's")
        out[h.response_id] = (key, h.sample, text, tier)
    for key in requests:
        samples = sorted(s for k, s, _, _ in out.values() if k == key)
        if samples != list(range(SAMPLES)):
            raise Pending(f"{name}: {key} has {len(samples)} of its {SAMPLES} hypotheses")
    return out


def labels(find_run: Any, name: str, kind: str, hyps: Mapping[str, tuple], behavior: str) -> dict:
    """{response id: (strict, lenient)} from the judge run: the frozen rubric set of ``kind`` and
    models, one primary label per hypothesis for its own target, each request the rubric's
    messages of the hypothesis text on the standard tier, each label its cached answer
    parsed again."""
    from .judge import (
        MalformedOutput,
        load_targets,
        parse_lenient,
        parse_strict,
        request_key,
        rubric_set,
    )
    from .records import JudgeLabel, read_table

    version = RUBRICS[kind]
    run, config, _ = _run(find_run, name, None, EXPERIMENTS["judge"])
    inputs = json.loads((run / "judge_inputs.json").read_text())
    rubrics = rubric_set(version)
    target = load_targets([behavior])[behavior]
    if (
        inputs.get("rubric_version") != version
        or (inputs.get("strict_model"), inputs.get("lenient_model"))
        != (JUDGE_MODELS["strict"], JUDGE_MODELS["lenient"])
        or any((inputs.get("rubrics") or {}).get(r.version, {}).get("sha256") != r.sha256
               for r in (rubrics.strict, rubrics.lenient))
    ):  # fmt: skip
        raise ValueError(f"{name}: not judged with the frozen pair {version}")
    cache = _cache(config, "judge", "judge")
    out: dict[str, tuple[str, bool]] = {}
    for lab in read_table([run], JudgeLabel.table):
        if lab.response_id not in hyps or lab.stage != "primary":
            continue
        if lab.target_behavior_id != behavior:
            raise ValueError(f"{lab.response_id}: a label for {lab.target_behavior_id}")
        if lab.response_id in out:
            raise ValueError(f"{lab.response_id}: two primary labels for its own target")
        if lab.rubric_version != version or lab.lenient_label is None:
            raise ValueError(f"{lab.response_id}: an incomplete label or another rubric's")
        text = hyps[lab.response_id][2]
        parsed = []
        for rubric, rid, model, parse in (
            (rubrics.strict, lab.request_id, JUDGE_MODELS["strict"], parse_strict),
            (rubrics.lenient, lab.lenient_request_id, JUDGE_MODELS["lenient"], parse_lenient),
        ):
            entry = cache.entry(rid) if isinstance(rid, str) and rid else None
            if (
                entry is None or request_key(rubric.version, entry["body"]) != rid
                or entry["body"].get("messages") != rubric.messages(target, text)
                or entry["body"].get("model") != model or entry.get("service_tier") != STANDARD
            ):  # fmt: skip
                raise ValueError(f"{lab.response_id}: its {rubric.kind} label is not for its text")
            try:
                parsed.append(parse(entry.get("text")))
            except MalformedOutput as error:
                raise ValueError(f"{lab.response_id}: its {rubric.kind} answer is malformed") \
                    from error  # fmt: skip
        if (parsed[0][0], parsed[1]) != (lab.strict_label, lab.lenient_label):
            raise ValueError(f"{lab.response_id}: its labels are not its answers'")
        out[lab.response_id] = (lab.strict_label, lab.lenient_label)
    missing = sorted(set(hyps) - set(out))
    if missing:
        raise Pending(f"{name}: {len(missing)} hypotheses without their label, {missing[:2]}")
    return out


def category_grades(find_run: Any, name: str, texts: Mapping[str, str]) -> dict:
    """{response id: frozenset of behaviors} for exactly the hypotheses ``texts`` (every counted
    one), each request recomputed from its text, the frozen definitions and model on the standard
    tier, each grade its cached answer parsed again."""
    from .continuation_definitions import load_definitions
    from .continuation_judge import CATEGORY_VERSION, category_request, parse_category
    from .judge import MalformedOutput, ModelSpec

    run, config, _ = _run(find_run, name, None, EXPERIMENTS["category"])
    inputs = json.loads((run / CATEGORY_INPUTS).read_text())
    definitions, digest = load_definitions()
    model = inputs.get("model") or {}
    if (
        inputs.get("version") != CATEGORY_VERSION or inputs.get("definitions_sha256") != digest
        or {k: model.get(k) for k in CATEGORY_MODEL} != CATEGORY_MODEL
        or inputs.get("service_tier") != STANDARD
        or inputs.get("max_tokens") != CATEGORY_MAX_TOKENS
    ):  # fmt: skip
        raise ValueError(f"{name}: not graded with the frozen category instrument")
    spec = ModelSpec(model["id"], model["reasoning_effort"], 0.0, 0.0, 0)
    cache = _cache(config, "category", "category")
    out: dict[str, frozenset[str]] = {}
    for line in (run / CATEGORY_LABELS).read_text().splitlines():
        row = json.loads(line)
        rid = row.get("response_id")
        if rid not in texts or rid in out:
            raise ValueError(f"{name}: {rid} is not one counted hypothesis, graded once")
        request = category_request(texts[rid], definitions, spec, max_tokens=CATEGORY_MAX_TOKENS)
        entry = cache.entry(request.key)
        if (
            row.get("request_key") != request.key
            or entry is None
            or entry.get("body") != request.body
            or entry.get("service_tier") != STANDARD
        ):
            raise ValueError(f"{rid}: its category grade is not for its text")  # fmt: skip
        try:
            grade = parse_category(entry.get("text"))
        except MalformedOutput as error:
            raise ValueError(f"{rid}: its category answer is malformed") from error
        if sorted(row.get("categories") or []) != sorted(grade):
            raise ValueError(f"{rid}: its categories are not its answer's")
        out[rid] = grade
    missing = sorted(set(texts) - set(out))
    if missing:
        raise Pending(f"{name}: {len(missing)} counted hypotheses not graded, {missing[:2]}")
    return out


def _table_sha(run: Path, *paths: str) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode() + b"\0" + (run / p).read_bytes() + b"\0")
    return h.hexdigest()


def readings(find_run: Any, sources: Mapping[str, Any], identity: Mapping[str, Any],
             rules: Any) -> dict[str, Any]:  # fmt: skip
    """The counts, category grades, X's flag and bindings of the lineage's readings (module
    docstring), under its ``identity`` (``continuation_panel.lineage_identity`` with the entry's
    ``reading``). Pending while a run is missing, unfinished or short of a hypothesis, label or
    grade; an error for anything that does not bind."""
    if sorted(sources) != sorted(SOURCES) or not all(isinstance(sources[k], str) for k in SOURCES):
        raise ValueError(f"the readers' sources {sorted(sources)} are not {sorted(SOURCES)}")
    run, record = pod_record(find_run, sources["record"], identity)
    x_flag = x_validity(find_run, record, identity, rules)
    lens_ev, lens_files = lens_evidence(find_run, record)
    pd_run, _, _ = _run(find_run, sources["pd_reading"], None, EXPERIMENTS["pd"])
    pd_names = [e["tool_run_id"] for e in json.loads(
        (pd_run / "summarizer_inputs.json").read_text()).get("evidence") or []]  # fmt: skip
    pd_ev, pd_files = pd_evidence_of(find_run, record, identity, pd_names)
    behavior = identity["behavior"]
    lens_h = hypotheses(find_run, sources["lens_reading"], "lens", list(lens_ev.values()))
    pd_h = hypotheses(find_run, sources["pd_reading"], "pd", list(pd_ev.values()))
    lens_l = labels(find_run, sources["lens_judge"], "lens", lens_h, behavior)
    pd_l = labels(find_run, sources["pd_judge"], "pd", pd_h, behavior)
    c_of = {rid: counted(*lens_l[rid]) for rid in lens_h} | {
        rid: counted(*pd_l[rid]) for rid in pd_h}  # fmt: skip
    texts = {rid: h[2] for hs in (lens_h, pd_h) for rid, h in hs.items() if c_of[rid]}
    grades = category_grades(find_run, sources["category"], texts)

    def count(ev: Mapping, hyps: Mapping) -> tuple[dict, dict]:
        where = {(v[0], v[1]): k for k, v in ev.items()}
        samples: dict[Any, dict[int, bool]] = {k: {} for k in ev}
        matches: dict[Any, list[frozenset[str]]] = {k: [] for k in ev}
        for rid, (key, sample, _, _) in sorted(hyps.items(), key=lambda kv: kv[1][:2]):
            samples[where[key]][sample] = c_of[rid]
            if c_of[rid]:
                matches[where[key]].append(grades[rid])
        return ({k: package_count(v, f"{k}") for k, v in samples.items()}, matches)

    lens, lens_matches = count(lens_ev, lens_h)
    pd, pd_matches = count(pd_ev, pd_h)
    runs = {f"{run.name}/stage3_readers.json": file_sha(run / "stage3_readers.json")}
    for key, file in (("lens_reading", "summarizer_inputs.json"),
                      ("pd_reading", "summarizer_inputs.json"),
                      ("lens_judge", "judge_inputs.json"), ("pd_judge", "judge_inputs.json"),
                      ("category", CATEGORY_INPUTS)):  # fmt: skip
        r = find_run(sources[key])
        runs[f"{r.name}/{file}"] = file_sha(r / file)
    labelled = {}
    for key, files in (
        ("lens_reading", ("observations/tool_hypotheses.jsonl", "responses/tool_hypotheses.jsonl",
                          "service_tiers.json")),
        ("pd_reading", ("observations/tool_hypotheses.jsonl", "responses/tool_hypotheses.jsonl",
                        "service_tiers.json")),
        ("lens_judge", ("observations/judge_labels.jsonl",)),
        ("pd_judge", ("observations/judge_labels.jsonl",)),
        ("category", (CATEGORY_LABELS,)),
    ):  # fmt: skip
        labelled[f"{sources[key]} ({key})"] = _table_sha(find_run(sources[key]), *files)
    return {
        "record": record,
        "x_validity": x_flag,
        "lens": lens,
        "pd": pd,
        "categories": {package: lens_matches[("L", package)] for package in DRAWS.values()},
        "margins": {f"{r}/{p}": m for (r, p), m in {**lens_matches, **pd_matches}.items()},
        "tiers": _tier_counts(lens_h, pd_h),
        "bindings": {"packages": {**lens_files, **pd_files}, "runs": runs, "labels": labelled},
    }


def _tier_counts(*groups: Mapping[str, tuple]) -> dict[str, int]:
    from collections import Counter

    return dict(sorted(Counter(h[3] for g in groups for h in g.values()).items()))


# ------------------------------------------------------------------------- the manifest


def _identity_fields(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {"lineage": identity["lineage"], "behavior": identity["behavior"],
            "dose": identity["dose"], "contracts": identity["contracts"],
            "final_detector": identity["detector"]}  # fmt: skip


def terminal_record(find_run: Any, sources: Mapping[str, Any], identity: Mapping[str, Any],
                    rules: Any) -> dict[str, Any]:  # fmt: skip
    """An invalid X_cont's readers (§12.4.6): its xcont cell recomputed through every attempt's
    evidence (``continuation_stage5._xcont`` False: invalid after its retry), p₃ = p₄ = 1, the
    five states that exist and the runs that show it, nothing else."""
    from . import continuation_stage5 as S5

    name = sources.get("xcont")
    if sorted(sources) != ["xcont"] or not isinstance(name, str) or identity["reading"] != "strong":
        raise ValueError("a terminal reading is a strong lineage's X_cont cell alone")
    states = identity["states"]
    valid = S5._xcont({"xcont": name}, states["L"], identity["dose"],
                      {"plan": identity["plan"]}, rules, find_run)  # fmt: skip
    if valid is None:
        raise Pending(f"{name}: X_cont is not built")
    if valid:
        raise ValueError(f"{name}: X_cont is valid; its readers are read, not terminal")
    runs = _xcont_runs(find_run, name)
    return {
        **_identity_fields(identity),
        "sources": dict(sources),
        "states": {k: v for k, v in states.items() if k != "X"},
        "status": "terminal",
        "reasons": [f"X_cont of {states['L']} is invalid after its retry (§12.4.6)"],
        "x_validity": {
            "passed": False,
            "inputs": {"control": "X_cont", "run": name, "attempts": sorted(runs)},
        },
        "bindings": {"runs": runs},
        "p3": "1",
        "p4": "1",
    }


def readers_record(find_run: Any, sources: Mapping[str, Any], identity: Mapping[str, Any],
                   rules: Any, *, tests: TestFactory | None = None,
                   workers: int = 0) -> dict[str, Any]:  # fmt: skip
    """The readers' manifest of a lineage, in the schema of ``continuation_panel.reader_values``,
    with its descriptive values beside it: terminal for an invalid X_cont (``sources`` its
    ``xcont`` cell alone), else complete from the bound readings (``sources``: ``SOURCES``)."""
    if "xcont" in sources:
        return serial(terminal_record(find_run, sources, identity, rules))
    found = readings(find_run, sources, identity, rules)
    th = thresholds(rules)
    jobs = component3_jobs(found["lens"], th) + component4_jobs(found["pd"], th)
    p = run_tests(jobs, rules, tests=tests, workers=workers)
    c3 = component3(found["lens"], found["categories"], identity["behavior"], th, p)
    c4 = component4(found["pd"], th, p)
    flags = c3["flags"]
    flags["eligibility"] = {**flags["eligibility"],
                            "pd_positive_control": c4["pd_positive_control"]}  # fmt: skip
    from .continuation_judge import category_confusion

    margins = {k: category_confusion(v, identity["behavior"]).margin
               for k, v in sorted(found["margins"].items())}  # fmt: skip
    return serial({
        **_identity_fields(identity),
        "sources": dict(sources),
        "states": dict(identity["states"]),
        "status": "complete",
        "reasons": [],
        **flags,
        "x_validity": found["x_validity"],
        "bindings": found["bindings"],
        "p3": c3["p3"],
        "p4": c4["p4"],
        "counts": {"lens": {f"{r}/{p}": x for (r, p), x in sorted(found["lens"].items())},
                   "pd": {f"{r}/{p}": x for (r, p), x in sorted(found["pd"].items())}},
        "component3": {k: v for k, v in c3.items() if k not in ("flags", "p3")},
        "component4": {k: v for k, v in c4.items() if k != "p4"},
        "category_margins": margins,
        "service_tiers": found["tiers"],
    })  # fmt: skip


def serial(value: Any) -> Any:
    """JSON values: fractions as their exact strings, tuple keys never."""
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, frozenset, set)):
        items = sorted(value) if isinstance(value, (frozenset, set)) else value
        return [serial(v) for v in items]
    return value


def write_readers(
    root: Path, find_run: Any, sources: Mapping[str, Any], identity: Mapping[str, Any],
    rules: Any, *, workers: int = 0,
) -> dict:  # fmt: skip
    """``readers_record`` written once to ``manifests/stage3_readers_<lineage>.json``; an
    existing manifest is never replaced. Pending work writes nothing."""
    path = readers_path(root, identity["lineage"])
    if path.exists():
        raise ValueError(f"{path} exists: the readers' manifest is written once")
    data = readers_record(find_run, sources, identity, rules, workers=workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return data


def committed_readers(path: Path, root: Path, rules: Any, find_run: Any, *,
                      contracts: str | None = None, identity_of: Any = None, workers: int = 0,
                      tests: TestFactory | None = None) -> dict[str, Any]:  # fmt: skip
    """The committed readers' manifest recomputed from its sources (``readers_record``) under
    the lineage's identity (``continuation_panel.lineage_identity`` under the committed contracts,
    or ``identity_of(behavior)``);
    any difference, a changed bound file among them, is refused. Returns the data with its
    sha256, as the panel's analysis reads it."""
    from .manifests import committed_json

    data = committed_json(Path(path), "the readers' manifest", Path(root))
    if identity_of is None:
        from . import continuation_strong as CS
        from .continuation_panel import lineage_identity
        from .pipeline import panel_contract

        if contracts is None:
            contracts = CS.committed_bundle(root, rules, find_run)["sha256"]

        def identity_of(behavior: str) -> dict[str, Any]:
            entry = CS.committed_contract(root, behavior, rules, find_run, panel_contract)["entry"]
            identity = lineage_identity(root, behavior, entry, contracts)[0]
            return {**identity, "reading": entry["reading"]}

    identity = identity_of(data.get("behavior"))
    if Path(path).name != readers_path(root, identity["lineage"]).name:
        raise ValueError(f"{path}: not the readers' manifest of {identity['lineage']}")
    stored = {k: v for k, v in data.items() if k not in ("sha256", "committed_at")}
    fresh = readers_record(find_run, stored.get("sources") or {}, identity, rules,
                           tests=tests, workers=workers)  # fmt: skip
    if json.loads(json.dumps(fresh, sort_keys=True)) != stored:
        raise ValueError(f"{path}: not what its sources give")
    return data
