"""The readers' verdicts of a certified Stage-3 lineage (preregistration §12.4.2–§12.4.6, §12.5,
with §12.4.5's amendments of 2026-10-24, 2026-10-25, 2026-10-26 and 2026-10-27), on the laptop,
from bound readings alone.

- **Counts.** Rule C counts a hypothesis when its lenient label is true and its strict label is
  not ``avoids_behavior``. Every package holds exactly 80 sample slots, 0–79 once each. A
  hypothesis whose strict request and retry were both blocked by the provider's safety filter,
  with a valid lenient label, has the status ``strict_safety_blocked`` and is not counted; the
  denominator stays 80, and its interval and the affected contrasts are reported descriptively
  (``safety_blocks``). A lineage with a committed completion of its blocked cohort
  (``continuation_completion``, 2026-10-27) reads those hypotheses with their completed strict
  labels instead, pending until the completion's provenance is committed; the zero-count
  treatment keeps its bounds and its own complete analysis (``zero_count_sensitivity``), which
  selects nothing.
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
  Every counted hypothesis is graded; the other states' margins are reported, not gates. A's
  reconstruction diagnostic (§12.4.4) is reported.
- **Component 4, PD-clean non-identification:** the positive control R(A) ≥ 4/5 and
  R(recipe) ≥ 4/5 on their clean packages; then p₄ is the largest of the TOSTs of R(L) − R(H_s)
  and of R(L) − R(X) within ±1/10 and the one-sample test of R(L) < 1/10; 1 without the control.
  The random-removal controls are reported.
- **X** is recomputed from its own records: a stack lineage's one composition of L and H, a
  strong lineage's X_cont cell through its attempts' evidence (``continuation_stage5._xcont``).
  An X_cont invalid after its retry makes the readers terminal, p₃ = p₄ = 1, with no reading.

Every comparison is exact; every threshold is the frozen rules' (``rules.readers``). A p-value is
the exact test's certified upper bound at its stopping gap, refined where its enclosure straddles
1/20 or 1/60 (the levels the analysis compares it with); each enclosure is kept, both ends under
the p-value's min(1, ·). The enclosures are numerical, not effect-size confidence intervals.

**What binds a reading** (``readings``), each recomputed, never trusted:
- the Pod's record, the run of its approved cell, and per state its approved cells: the
  reference, the lens package (whose identity holds the frozen lens settings), the PD pool and
  the δ̄ check, both δ̄ files by their sha256 and the errors recomputed;
- the lens packages and the PD-clean packages by sha256, each PD-clean package recomputed from
  its study's records (``pd_clean_study.verify_packages``: rounds, the researcher's review, the
  selection, the random control);
- each of the interpreter's 80 sample slots per package: its request rebuilt from the frozen
  interpreter (model, thinking level, cap, prompt, sample index, stability salt), its answer the
  first one the frozen parser accepts of the request and its one retry (``slot``), with its tier;
- each own-target label: the frozen judge's full requests (rubric set v4 for the lens, v4-j3 for
  PD-clean; models, thinking levels, cap, structured output), each slot recomputed the same way,
  on the standard tier. A judge reply cached before tiers were recorded is read only when the
  lineage's committed judge history pins its entry's sha256 with its original writer's evidence
  (§12.4.5, 2026-10-27: ``judge_history``, written once by ``write_judge_history``), reported at
  its writer's established tier or ``unrecorded``; any other tier-less reply leaves its
  hypothesis pending. Slot references and unique entries are reported apart;
- each counted hypothesis's ``category-v1`` grade: the frozen grader's request (the pinned
  definitions, model, thinking level and cap) recomputed the same way, on the standard tier.

``lineage_of`` resolves the certified lineage; ``readers_record`` gives the manifest the panel's
analysis reads (``continuation_panel.reader_values``); ``write_readers`` writes it once;
``committed_readers`` recomputes it.
"""

from __future__ import annotations

import dataclasses
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

# The frozen instruments (§12.3.3, §12.4.5): the interpreter of experiments/summarize_tools
# (its model, prompts and cap), the judge of experiments/judge_continuation (rubric v4 on the lens
# hypotheses, v4-j3 on PD-clean's; models, reasoning, cap and structured output), the category
# grader on gemini-3.8-flash at low with the frozen definitions; the judge and the category
# grader on the standard tier.
INTERPRETER, JUDGE = "experiments/summarize_tools", "experiments/judge_continuation"
PROMPT_TOOLS = {"lens": "adl_lens", "pd": "perplexity_differencing"}
RUBRICS = {"lens": "v4", "pd": "v4-j3"}
JUDGE_MODELS = {"strict": "gemini-3.5-flash-lite", "lenient": "gemini-3.8-flash"}
CATEGORY_MODEL = "gemini-3.8-flash"
CATEGORY_MAX_TOKENS = 2048  # continuation_judge.category_request's cap
# The follow-up's specificity grader (§13.4, barrier B2): the category grader's model and cap
SPECIFICITY_MODEL, SPECIFICITY_MAX_TOKENS = CATEGORY_MODEL, 2048
DEFINITIONS_SHA256 = "8e083c5c944fe8e77f8a6bcec41d90f2bc5f38ac36ae5afdefdb665f808fe8d7"
STANDARD = "standard"
RETRY = "#retry=1"  # a slot's one retry after the frozen parser's refusal (§12.4.5, 2026-10-25)
STRICT_SAFETY_BLOCKED = "strict_safety_blocked"  # a strict label's status (§12.4.5, 2026-10-26)
SAFETY_FINISH = "content_filter"  # the safety filter's finish reason, before any ": detail"
HISTORY = "historical judge-cache reuse (§12.4.5, 2026-10-27)"
RECORDING_COMMIT = "114f07a"  # the judge cache's per-answer tier; earlier writers: one tier a run
UNRECORDED = "unrecorded"  # a historical reply's tier that its writers do not establish
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
    """Rule C; a strict label blocked by the safety filter counts as not naming b (§12.4.5,
    2026-10-26)."""
    if not isinstance(lenient, bool):
        raise ValueError(f"a lenient label {lenient!r} is not a bool")
    return lenient and strict not in ("avoids_behavior", STRICT_SAFETY_BLOCKED)


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
    """A test's enclosure [lower, p] of its p-value, both ends under the p-value's min(1, ·);
    one whose lower end exceeds its p is refused."""
    p = min(Fraction(1), Fraction(found["p"]))
    lower = min(Fraction(1), Fraction(found.get("lower", p)))
    if not 0 <= lower <= p:
        raise ValueError(f"an enclosure [{lower}, {p}] of a p-value is not one")
    return {"target": target, "p": p, "lower": lower, "empty": bool(found.get("empty"))}


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


# ------------------------------------------------------------------------ the lineage


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclasses.dataclass
class Lineage:
    """A certified lineage as the readers' verdicts read it: ``identity`` the panel's analysis's
    (``continuation_panel.lineage_identity``) with the entry's ``reading`` and, for a strong
    lineage, X from its X_cont cell; ``ws`` its workspace (no state loaded); ``certified``
    ``pipeline.certified_lineage``'s record; ``xcont`` a strong lineage's X_cont cell
    (``{"run", "valid"}``), None for a stack lineage."""

    identity: dict[str, Any]
    ws: Any
    certified: dict[str, Any]
    xcont: dict[str, Any] | None = None


def strong_xcont(ws: Any, s: Mapping[str, Any], rules: Any, find_run: Any) -> dict[str, Any]:
    """A strong lineage's X_cont cell at its dose (``pipeline.strong_x``'s cell, the frozen
    check) recomputed through every attempt's evidence (``continuation_stage5._xcont``):
    ``{"run", "valid", "state"}``; Pending while it is not built."""
    from . import continuation_stage5 as S5
    from . import pipeline

    c = {"dose": s["dose"], "latent": s["latent"], "plan": s["plan"]}
    cell = pipeline.make_cell(ws, "xcont", pipeline._xcont_params(
        c, dict(rules["xcont_tolerance"])), behavior_free=True)  # fmt: skip
    run = _cell_run(cell, f"{s['lineage']}'s X_cont")
    valid = S5._xcont({"xcont": run.name}, s["latent"], s["dose"], {"plan": s["plan"]},
                      rules, find_run)  # fmt: skip
    if valid is None:
        raise Pending(f"{run.name}: not a completed X_cont cell")
    state = cell.summary().get("state") if valid else None
    return {"run": run.name, "valid": bool(valid), "state": state}


def lineage_of(root: Path, behavior: str, rules: Any, find_run: Any,
               contracts: str | None = None) -> Lineage:  # fmt: skip
    """The certified lineage of a panel behavior under the committed contracts, as ``Lineage``
    describes it."""
    from . import continuation_strong as CS
    from . import pipeline
    from .config import load_config
    from .continuation_panel import lineage_identity

    if contracts is None:
        contracts = CS.committed_bundle(root, rules, find_run)["sha256"]
    entry = CS.committed_contract(root, behavior, rules, find_run, pipeline.panel_contract)["entry"]
    identity, ws = lineage_identity(root, behavior, entry, contracts)
    cfg = load_config(f"experiments/continuation_14b/{behavior}",
                      ["tracking.wandb.enabled=false"])  # fmt: skip
    plan = {"analysis": {"stage3": True, "contracts": contracts}}
    s = pipeline.certified_lineage(cfg, plan, "analysis")
    identity = {**identity, "reading": entry["reading"], "states": dict(identity["states"])}
    xcont = None
    if entry["reading"] == "strong":
        xcont = strong_xcont(ws, s, rules, find_run)
        if xcont["valid"]:
            identity["states"]["X"] = xcont["state"]
    return Lineage(identity, ws, dict(s), xcont)


# ------------------------------------------------------------------------ the Pod's record


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


def _cell_run(cell: Any, what: str) -> Path:
    """The run of an approved cell (``pipeline.make_cell``), checked as the panel's analysis
    checks its cells: the hash of the run's saved config recomputed, the hash its meta records
    and the cell's approved hash the same, and the run's identity (``continuation_panel.
    IDENTITY``) canonically the cell's. Pending while the cell has no completed run."""
    if cell is None or not cell.done:
        raise Pending(f"{what}: its cell has no completed run")
    run = Path(cell.previous)
    _saved_config(run, what, cell)
    return run


def _saved_config(run: Path, what: str, cell: Any = None) -> dict[str, Any]:
    """A run's saved config, its hash recomputed equal to the one its meta records (and, with
    ``cell``, to the approved cell's, the identities canonically equal)."""
    import yaml
    from omegaconf import OmegaConf

    from .config import canonical, config_hash
    from .continuation_panel import IDENTITY

    meta = json.loads((run / "meta.json").read_text())
    config = yaml.safe_load((run / "config.yaml").read_text()) or {}
    recomputed = config_hash(OmegaConf.create(config))
    if meta.get("config_hash") != recomputed:
        raise ValueError(f"{run.name}: its saved config is not the one its meta records ({what})")
    if cell is not None:
        approved = OmegaConf.to_container(OmegaConf.create(cell.cfg), resolve=True)
        if recomputed != cell.digest or canonical({k: config.get(k) for k in IDENTITY}) != (
            canonical({k: approved.get(k) for k in IDENTITY})
        ):  # fmt: skip
            raise ValueError(f"{run.name}: not the approved cell of {what}")
    return config


def pod_record(lineage: Lineage, find_run: Any, name: str) -> tuple[Path, dict]:
    """The Pod's readers' record of the lineage, the run of its approved cell: the cell
    ``stage3_readers`` of the lineage's dose, states, binding, contracts and the record's own hash
    is this run; its fields are the lineage's."""
    from . import pipeline
    from .pipeline import content_hash

    identity = lineage.identity
    run, _, _ = _run(find_run, name, "stage3_readers")
    record = json.loads((run / "stage3_readers.json").read_text())
    params = {"dose": identity["dose"], "states": identity["states"],
              "binding": content_hash(identity["binding"]), "contracts": identity["contracts"],
              "record": content_hash(record)}  # fmt: skip
    cell = pipeline.make_cell(lineage.ws, "stage3_readers", params)
    if _cell_run(cell, "the readers' record").name != run.name:
        raise ValueError(f"{name}: not the readers' record of {identity['lineage']}")
    fields = {"behavior": identity["behavior"], "lineage": identity["lineage"],
              "dose": identity["dose"], "states": identity["states"],
              "binding": identity["binding"], "reading": identity["reading"]}  # fmt: skip
    wrong = sorted(k for k, v in fields.items() if record.get(k) != v)
    if (record.get("contracts") or {}).get("sha256") != identity["contracts"]:
        wrong.append("contracts")
    if wrong or sorted(record.get("readers") or {}) != sorted(ROLES):
        raise ValueError(f"{name}: not the readers of {identity['lineage']} at {wrong}")
    return run, record


def x_validity(lineage: Lineage, record: Mapping[str, Any], find_run: Any) -> dict[str, Any]:
    """X recomputed from its own records: a stack lineage's one composition of L and H (its
    cell's state), a strong lineage's valid X_cont cell (``strong_xcont``); the record's X evidence
    the same. The flag with its evidence's sha256."""
    from . import pipeline

    x, states = record.get("x_validity") or {}, lineage.identity["states"]
    if lineage.identity["reading"] == "stack":
        pair = (states["L"], states["H"])
        made = pipeline._completed_cells(
            lineage.ws, "compose_suppression_only",
            lambda q: (q.get("suppressed"), q.get("sham")) == pair,
        )  # fmt: skip
        if len(made) != 1 or str(made[0][2].get("state_id")) != states["X"]:
            raise ValueError(f"X {states['X']} is not the one composition of L and H")
        run, control = Path(made[0][0]), "X"
        _saved_config(run, "the composition of L and H")
    else:
        if not (lineage.xcont or {}).get("valid"):
            raise ValueError("an invalid X_cont has terminal readers, not a reading")
        run, _, _ = _run(find_run, lineage.xcont["run"], "xcont")
        control = "X_cont"
    want = {"control": control, "state": states["X"], "run": run.name, "valid": True,
            "verified": True}  # fmt: skip
    if dict(x) != want:
        raise ValueError(f"the record's X {x} is not the lineage's {want}")
    return {"passed": True, "inputs": {"control": control, "state": states["X"], "run": run.name,
                                       "sha256": file_sha(run / "summary.json")}}  # fmt: skip


def _delta(path: Path) -> Any:
    from safetensors.torch import load_file

    return load_file(str(path), device="cpu")["delta"]


def state_checks(lineage: Lineage, record: Mapping[str, Any], find_run: Any) -> dict[str, Any]:
    """Each state's reading recomputed from its approved cells (``pipeline``): its reference (A's
    and H's stored pilot packages, the others' method-equivalence runs), its lens cell (A with
    its controls; the frozen lens settings and the reference are its identity), its PD pool (L's
    the one its evidence read; A's under its Stage-3 tag; the others untagged; each on the final
    PD GPU; none for H); and its δ̄ check: both files by their recorded sha256, the relative
    errors recomputed on CPU (``adl_lens.check_reproduction``) and within the frozen tolerance.
    Returns per role {lens run, reference run, PD run} and the files read, by sha256."""
    from . import adl_lens, pipeline

    ws, out, files = lineage.ws, {}, {}
    for role in ROLES:
        r, sid = record["readers"][role], lineage.identity["states"][role]
        lens_cell, reference_cell = pipeline._stage3_lens_cell(ws, role, sid)
        reference = _cell_run(reference_cell, f"{role}'s reference")
        lens = _cell_run(lens_cell, f"{role}'s lens package")
        kind = "pilot reproduction" if role in pipeline.STAGE3_PILOT_ROLES else "method equivalence"
        if role == "L":
            pd = pipeline._stage5_pd_pool(ws, lineage.certified)  # its content checked there
            _saved_config(_run(find_run, pd, "tool_pd")[0], "L's PD pool")
        elif role == "H":
            pd = None
        else:
            draw = pipeline.STAGE3_A_PD_DRAW if role == "A" else None
            pool = _cell_run(pipeline.tool_pd_cell(ws, sid, draw), f"{role}'s PD pool")
            if pipeline.pd_gpu(pool) != pipeline.FINAL_PD_GPU:
                raise ValueError(f"{pool.name}: {role}'s PD pool is not on the final PD GPU")
            pd = pool.name
        check = r.get("check") or {}
        package = json.loads((lens / "adl_lens_package.json").read_text())
        layer = (package.get("reproduction") or {}).get("layer")
        paths = {"reference": reference / "steps" / "extraction.safetensors",
                 "lens": lens / "reproduction" / f"delta_layer{layer}.safetensors"}  # fmt: skip
        delta = {k: {"file": f"{v.parent.parent.name}/{v.relative_to(v.parents[1])}",
                     "sha256": file_sha(v)} for k, v in paths.items()}  # fmt: skip
        found = adl_lens.check_reproduction(_delta(paths["lens"]), _delta(paths["reference"]))
        want = {"kind": kind, "run": reference.name, "layer": layer, "delta": delta,
                "max_relative_error": max(found.get("relative_errors") or [float("inf")]),
                "tolerance": adl_lens.REPRODUCTION_TOLERANCE, "passed": True}  # fmt: skip
        if (
            r.get("state") != sid or r.get("lens") != lens.name or r.get("pd") != pd
            or r.get("controls") is not (role == "A") or found.get("passed") is not True
            or found.get("tolerance") != adl_lens.REPRODUCTION_TOLERANCE or check != want
        ):  # fmt: skip
            raise ValueError(f"{role}'s reading in the record is not its approved cells'")
        out[role] = {"lens": lens, "reference": reference.name, "pd": pd}
        files.update({v["file"]: v["sha256"] for v in delta.values()})
    return {"states": out, "files": files}


# ------------------------------------------------------------------------ the evidence


def lens_evidence(checks: Mapping[str, Any], lineage: Lineage) -> tuple[dict, dict, dict]:
    """Each lens package's evidence {(role, package): (state, tool, run, text)} from the
    approved lens runs, the package files by sha256, and A's reconstruction diagnostic per layer
    (§12.4.4, descriptive)."""
    from .summarizer import adl_lens_evidence

    out, files, reconstruction = {}, {}, {}
    for role in ROLES:
        run = checks["states"][role]["lens"]
        path = run / "adl_lens_package.json"
        files[f"{run.name}/adl_lens_package.json"] = file_sha(path)
        layers = json.loads(path.read_text())["layers"]
        if role == "A":
            reconstruction = {k: v.get("reconstruction") for k, v in sorted(layers.items())}
        for package in LENS_PACKAGES + (CONTROLS if role == "A" else ()):
            layer, key = package.split("/", 1)
            try:
                pkg = layers[layer]["packages"][key]
            except KeyError as error:
                raise ValueError(f"{run.name}: no package {package}") from error
            out[(role, package)] = (lineage.identity["states"][role], f"adl_lens@{package}",
                                    run.name, adl_lens_evidence(pkg))  # fmt: skip
    return out, files, reconstruction


def pd_evidence_of(find_run: Any, lineage: Lineage, record_run: Path,
                   runs: Sequence[str]) -> tuple[dict, dict, dict]:  # fmt: skip
    """Each PD-clean package's evidence {(role, kind): (state, tool, run, text)} from ``runs``
    (the package runs the interpreter read), the package files by sha256, and the study's
    records by sha256: every package recomputed from its study's records
    (``pd_clean_study.verify_packages``, the lineage's work directory)."""
    from .paths import data_root
    from .pd_clean_study import verify_packages
    from .summarizer import pd_evidence

    kinds = {"pd_clean_package": PD_CLEAN, "pd_random_package": PD_RANDOM}
    found, out, files = {}, {}, {}
    for name in runs:
        run, config, _ = _run(find_run, name, None, "pd_clean")
        key = ((config.get("params") or {}).get("role"), kinds.get(config.get("step")))
        if key in found or key not in pd_inventory():
            raise ValueError(f"{name}: not one PD-clean package of the lineage's states")
        found[key] = run
    if sorted(found) != sorted(pd_inventory()):
        raise Pending(f"the PD-clean packages read are {sorted(found)}, not the eight")
    work = data_root() / "pd_clean_stage3" / lineage.identity["lineage"]
    records = verify_packages(work, record_run, found)
    for (role, kind), run in found.items():
        package = json.loads((run / "pd_clean.json").read_text())
        files[f"{run.name}/pd_clean.json"] = file_sha(run / "pd_clean.json")
        out[(role, kind)] = (lineage.identity["states"][role], kind, run.name,
                             pd_evidence(package["top"]))  # fmt: skip
    return out, files, records


def _cache(config: Mapping[str, Any], section: str, default: str) -> Any:
    from .judge import JudgeCache
    from .paths import data_root

    root = (config.get(section) or {}).get("cache_dir")
    return JudgeCache(Path(root) if root else data_root() / "cache" / default)


def slot(cache: Any, keys: Sequence[str], body: Mapping[str, Any], parse: Any, where: str) -> dict:
    """A sample slot's history (§12.4.5, 2026-10-25): its attempts in order, the request and its
    one retry, each the cache's answer to exactly ``body``; the first that ``parse`` accepts is
    the slot's. Pending while an attempt due is unanswered or both are refused."""
    from .judge import MalformedOutput, cache_entry_sha256

    history = []
    for n, key in enumerate(keys, start=1):
        entry = cache.entry(key)
        if entry is None:
            raise Pending(f"{where}: attempt {n} has no answer")
        if entry.get("body") != body:
            raise ValueError(f"{where}: attempt {n}'s answer is not to its request")
        step = {"key": key, "sha256": cache_entry_sha256(entry), "tier": entry.get("service_tier")}
        try:
            value = parse(entry.get("text"))
        except MalformedOutput:
            history.append({**step, "parsed": False})
            continue
        history.append({**step, "parsed": True})
        return {"key": key, "value": value, "tier": entry.get("service_tier"), "history": history}
    raise Pending(f"{where}: both attempts refused by the parser")


def safety_blocked(entry: Mapping[str, Any]) -> bool:
    """A retained answer the provider's safety filter blocked: an explicit empty output, an
    explicit integer zero of completion tokens and an explicit safety-filter finish reason."""
    finish, usage = entry.get("finish_reason"), entry.get("usage")
    tokens = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
    return (
        "text" in entry
        and entry["text"] in (None, "")
        and isinstance(finish, str)
        and finish.split(":", 1)[0].strip() == SAFETY_FINISH
        and type(tokens) is int
        and tokens == 0
    )


def blocked_slot(
    cache: Any, keys: Sequence[str], body: Mapping[str, Any], where: str,
    pinned: Mapping[str, Mapping[str, Any]] | None = None,
) -> list | None:  # fmt: skip
    """A strict slot's history when its request and its one retry were both blocked by the
    safety filter (§12.4.5, 2026-10-26), each the cache's answer to exactly ``body`` on the
    standard tier or a historical reply ``pinned`` by its entry's sha256 (``admitted_tier``);
    None otherwise (a missing attempt, another tier, any other answer)."""
    from .judge import cache_entry_sha256

    history = []
    for n, key in enumerate(keys, start=1):
        entry = cache.entry(key)
        if entry is None:
            return None
        if entry.get("body") != body:
            raise ValueError(f"{where}: attempt {n}'s answer is not to its request")
        step = {"key": key, "sha256": cache_entry_sha256(entry), "tier": entry.get("service_tier")}
        if admitted_tier(step, pinned or {}) is None or not safety_blocked(entry):
            return None
        history.append({**step, "parsed": False,
                        "finish_reason": entry["finish_reason"],
                        "usage": dict(entry.get("usage") or {})})  # fmt: skip
    return history


def admitted_tier(step: Mapping[str, Any], pinned: Mapping[str, Mapping[str, Any]]) -> str | None:
    """The tier a judge attempt is read at: the standard tier recorded with it, or a historical
    reply's established tier (or ``UNRECORDED``) when the committed history manifest pins its
    entry (§12.4.5, 2026-10-27); None for any other attempt. A pinned entry that changed is
    refused."""
    found = pinned.get(step["key"])
    if found is not None:
        if found["entry_sha256"] != step["sha256"]:
            raise ValueError(f"{step['key']}: its cache entry is not the one the history pins")
        return found["tier"]
    return STANDARD if step["tier"] == STANDARD else None


def history_path(root: Path, lineage: str) -> Path:
    return Path(root) / "manifests" / f"stage3_judge_history_{lineage}.json"


def _attempts(behavior: str, kind: str, hyps: Mapping[str, tuple]) -> list[tuple[str, str, dict]]:
    """(key, request kind, body) of every attempt the judge slots of ``hyps`` can read, per
    hypothesis: each request and its one retry."""
    from .judge import JudgeItem, load_targets, request_key, stage_requests

    settings = judge_settings(RUBRICS[kind])
    target = load_targets([behavior])[behavior]
    out = []
    for rid in sorted(hyps):
        req = stage_requests([JudgeItem(rid, behavior, target, hyps[rid][2])], settings)[0]
        for r in (req.strict, req.lenient):
            for key in (r.key, request_key(f"{r.rubric_version}{RETRY}", r.body)):
                out.append((key, r.kind, r.body))
    return out


def _lacks(commit: str, change: str) -> bool:
    """The commit is in this repository and does not contain ``change``."""
    import subprocess

    from .paths import REPO_ROOT

    def git(*args: str) -> int:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True).returncode

    return git("cat-file", "-e", f"{commit}^{{commit}}") == 0 and git(
        "merge-base", "--is-ancestor", change, commit) == 1  # fmt: skip


def _historical_writer(run: Path) -> tuple[str, dict[str, Any]] | None:
    """A historical writer's established tier and its evidence, or None when the run's code is
    not historical: its recorded commit is in this repository and does not contain the cache's
    tier recording (``RECORDING_COMMIT``), hence not the judge's later per-request fallback
    either. The tier is the sync run's configured one (an explicit null: ``standard``, as that
    code named the default tier), ``batch`` for a batch run, else ``UNRECORDED`` (a dirty tree,
    no configured tier)."""
    import yaml

    judge = (yaml.safe_load((run / "config.yaml").read_text()) or {}).get("judge") or {}
    git = json.loads((run / "meta.json").read_text()).get("git") or {}
    sync = judge.get("sync") or {}
    evidence = {"backend": judge.get("backend"), "sync_service_tier": sync.get("service_tier"),
                "commit": git.get("commit"), "dirty": git.get("dirty")}  # fmt: skip
    if not isinstance(git.get("commit"), str) or not _lacks(git["commit"], RECORDING_COMMIT):
        return None
    if git.get("dirty") is not False:
        return UNRECORDED, evidence
    if judge.get("backend") == "sync" and "service_tier" in sync:
        return sync["service_tier"] or STANDARD, evidence
    if judge.get("backend") == "batch":
        return "batch", evidence
    return UNRECORDED, evidence


def recording_boundary() -> dict[str, str]:
    """The cache's tier recording (``RECORDING_COMMIT``) as git records it: its full hash and
    committer time. A historical writer finished, and its entry was cached, strictly before."""
    import subprocess

    from .paths import REPO_ROOT

    command = ["log", "-1", "--format=%H %cI", f"{RECORDING_COMMIT}^{{commit}}"]
    found = subprocess.run(["git", "-C", str(REPO_ROOT), *command], capture_output=True, text=True)
    commit, _, at = found.stdout.strip().partition(" ")
    if found.returncode != 0 or not commit.startswith(RECORDING_COMMIT) or _instant(at) is None:
        raise ValueError(f"{RECORDING_COMMIT}: the tier recording's commit is not in git")
    return {"commit": commit, "committed_at": at, "source": "git log -1 --format=%H %cI"}


def _instant(value: Any) -> Any:
    """A recorded ISO-8601 time with its offset as an aware datetime; None for anything else."""
    from datetime import datetime

    try:
        found = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None
    return found if found is not None and found.tzinfo is not None else None


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _answer_sha(fields: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps({k: fields.get(k) for k in ("text", "finish_reason", "usage")},
                                     sort_keys=True).encode()).hexdigest()  # fmt: skip


def judge_history(
    find_run: Any, behavior: str, judges: Mapping[str, str],
    hyps: Mapping[str, Mapping[str, tuple]], runs: Path,
) -> dict[str, Any]:  # fmt: skip
    """The provenance of every tier-less judge-cache entry the lineage's judge slots can read
    (§12.4.5, 2026-10-27), recomputed from the original writers alone: a historical judge run
    under ``runs`` (``_historical_writer``) on the same cache, whose recorded window holds the
    entry's ``cached_at``, finished strictly before the tier recording's commit time
    (``recording_boundary``, kept in the history; the entry cached before it too), and whose
    request log holds the same reply sent and answered (``from_cache`` false, ``error`` null,
    the same text, finish reason and usage). A run that only read the entry from the cache is
    its consumer, and a label row alone is no evidence of writing; the writer's agreeing or
    disagreeing label rows are reported. Each accepted
    entry is pinned by its sha256 with its writers' evidence and the established tier
    (``UNRECORDED`` when the writers do not establish one tier); the others are listed as
    unsupported and leave their hypotheses pending. Only hashes, paths and provenance fields
    are kept; no reply text or label value."""
    from collections import Counter

    import yaml

    from .judge import MalformedOutput, cache_entry_sha256, parse_lenient, parse_strict

    parse = {"strict": lambda t: parse_strict(t)[0], "lenient": parse_lenient}
    column = {"strict": "request_id", "lenient": "lenient_request_id"}
    label = {"strict": "strict_label", "lenient": "lenient_label"}
    wanted: dict[str, tuple[str, dict, Any]] = {}
    references = 0
    for kind in ("lens", "pd"):
        _, config, _ = _run(find_run, judges[kind], None, EXPERIMENTS["judge"])
        cache = _cache(config, "judge", "judge")
        for key, rkind, body in _attempts(behavior, kind, hyps[kind]):
            entry = cache.entry(key)
            if entry is None or "service_tier" in entry:
                continue
            if entry.get("body") != body:
                raise ValueError(f"{key}: a cached reply that is not to its request")
            references += 1
            wanted.setdefault(key, (rkind, entry, cache.root))
    rows: dict[str, dict[Path, list[dict]]] = {}
    for table in sorted(Path(runs).glob("*/*/observations/judge_labels.jsonl")):
        run = table.parent.parent
        for line in table.read_text().splitlines():
            row = json.loads(line)
            for rkind, col in column.items():
                key = row.get(col)
                if key in wanted and wanted[key][0] == rkind:
                    rows.setdefault(key, {}).setdefault(run, []).append(row)
    sent: dict[str, dict[Path, set[str]]] = {}
    for log in sorted(Path(runs).glob("*/*/judge_requests.jsonl")):
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r.get("request_id") in wanted and r.get("from_cache") is False and (
                    "error" in r and r["error"] is None):  # fmt: skip
                sent.setdefault(r["request_id"], {}).setdefault(log.parent, set()).add(
                    _answer_sha(r))  # fmt: skip

    boundary = recording_boundary()
    before = _instant(boundary["committed_at"])
    entries, unsupported = {}, []
    for key, (rkind, entry, root) in sorted(wanted.items()):
        at, writers = _instant(entry.get("cached_at")), []
        try:
            reply = (parse[rkind](entry.get("text")),)
        except MalformedOutput:
            reply = None
        for run, logged in sorted(sent.get(key, {}).items(), key=lambda r: str(r[0])):
            meta = json.loads((run / "meta.json").read_text())
            config = yaml.safe_load((run / "config.yaml").read_text()) or {}
            if Path(_cache(config, "judge", "judge").root).resolve() != Path(root).resolve():
                continue
            start, end = meta.get("started_at"), meta.get("finished_at")
            window = _instant(start), _instant(end)
            if at is None or None in window or not window[0] <= at <= window[1]:
                continue
            if not (window[1] < before and at < before):
                continue
            if _answer_sha(entry) not in logged:
                continue
            writer = _historical_writer(run)
            if writer is None:
                continue
            found = sorted(rows.get(key, {}).get(run, []), key=_sha)
            agrees = [reply is not None and reply[0] == row.get(label[rkind]) for row in found]
            writers.append({"run": f"{run.parent.name}/{run.name}",
                            "started_at": start, "finished_at": end,
                            "label_rows": [{"sha256": _sha(row), "agrees": a}
                                           for row, a in zip(found, agrees, strict=True)],
                            "request_log": "matches", "tier": writer})  # fmt: skip
        if not writers:
            unsupported.append(key)
            continue
        tiers = {w["tier"][0] for w in writers}
        entries[key] = {"kind": rkind, "entry_sha256": cache_entry_sha256(entry),
                        "cached_at": entry["cached_at"],
                        "tier": tiers.pop() if len(tiers) == 1 else UNRECORDED,
                        "writers": [{**w, "tier": w["tier"][0], "tier_evidence": w["tier"][1]}
                                    for w in writers]}  # fmt: skip
    return {"rule": HISTORY, "behavior": behavior, "judge_runs": dict(judges),
            "recording_boundary": boundary,
            "references": {"attempts": references, "unique_entries": len(wanted)},
            "tiers": dict(sorted(Counter(e["tier"] for e in entries.values()).items())),
            "entries": entries, "unsupported": unsupported}  # fmt: skip


def _spec(entry: Any) -> Any:
    """A model of the judge config's ``models`` list as ``judge.ModelSpec``."""
    from .judge import ModelSpec

    return ModelSpec(
        entry.id,
        entry.reasoning_effort,
        float(entry.input_usd_per_mtok),
        float(entry.output_usd_per_mtok),
        int(entry.expected_thinking_tokens),
    )


def interpreter(kind: str) -> tuple[Any, Any]:
    """The frozen interpreter of ``kind`` ("lens" or "pd"): its prompt and settings
    (``INTERPRETER``'s model with its thinking level, cap; 80 samples)."""
    from .config import load_config
    from .summarizer import SummarizerSettings, prompt_for

    cfg = load_config(INTERPRETER, ["tracking.wandb.enabled=false"])
    s = cfg.summarizer
    entry = next(m for m in cfg.judge.models if m.id == s.model)
    model = _spec(entry)
    tool = PROMPT_TOOLS[kind]
    return prompt_for(tool, s.prompts[tool]), SummarizerSettings(model, SAMPLES, int(s.max_tokens))


def hypotheses(find_run: Any, name: str, kind: str,
               evidence: Sequence[tuple[str, str, str, str]]) -> dict:  # fmt: skip
    """The interpreter run's hypotheses {response id: (key, sample, text, tier, history)} over
    exactly ``evidence`` (``kind`` "lens" or "pd"), 80 slots per package: each slot recomputed
    (``slot``) from its request (the frozen prompt and settings, its sample index and stability
    salt) and its retry; the run's hypothesis the slot's answer, parsed again, with its tier."""
    from .judge import request_key
    from .pipeline import content_hash
    from .records import ToolHypothesis, read_responses, read_table
    from .summarizer import parse_hypothesis, stability_salts, summary_requests

    prompt, settings = interpreter(kind)
    run, config, _ = _run(find_run, name, None, EXPERIMENTS[kind])
    inputs = json.loads((run / "summarizer_inputs.json").read_text())
    model = inputs.get("model") or {}
    if (
        (model.get("id"), model.get("reasoning_effort"), inputs.get("max_tokens"))
        != (settings.model.id, settings.model.reasoning_effort, settings.max_tokens)
        or inputs.get("samples") != SAMPLES or inputs.get("stability_salts") is not True
        or inputs.get("prompts") != {prompt.version: {"tool": prompt.tool, "sha256": prompt.sha256}}
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
    ordered = [{"state_id": e["state_id"], "tool": e["tool"],
                "text": text_of[(e["state_id"], e["tool"])]} for e in rows]  # fmt: skip
    salts = stability_salts(ordered)
    keys: dict[tuple, list] = {}
    for e, salt in zip(ordered, salts, strict=True):
        tag = "" if salt is None else f"#{salt}"
        for i, request in enumerate(summary_requests(prompt, e["text"], settings, salt=salt)):
            retry = request_key(f"{prompt.version}#sample={i}{tag}{RETRY}", request.body)
            keys[(e["state_id"], e["tool"], i)] = [request.key, retry], request.body
    cache = _cache(config, "summarizer", "summarizer")
    tiers = json.loads((run / "service_tiers.json").read_text()).get("hypotheses") or {}
    texts = read_responses([run], ToolHypothesis.table)
    out: dict[str, tuple] = {}
    for h in read_table([run], ToolHypothesis.table):
        k = (h.state_id, h.tool, h.sample)
        if h.run_id != run.name or not h.response_id.startswith(f"{run.name}/"):
            raise ValueError(f"{h.response_id}: not a hypothesis that its run {run.name} owns")
        if k not in keys or h.prompt_version != prompt.version or h.budget is not None \
                or h.summarizer_model != settings.model.id or h.response_id in out \
                or run_of[(h.state_id, h.tool)] != h.tool_run_id:  # fmt: skip
            raise ValueError(f"{h.response_id}: not one hypothesis of the planned evidence")
        found = slot(cache, *keys[k], parse_hypothesis, h.response_id)
        if texts.get(h.response_id) != found["value"]:
            raise ValueError(f"{h.response_id}: its text is not its slot's answer")
        if tiers.get(h.response_id) != found["tier"] or not isinstance(found["tier"], str):
            raise ValueError(f"{h.response_id}: its tier is not its answer's")
        out[h.response_id] = ((h.state_id, h.tool), h.sample, found["value"], found["tier"],
                              found["history"])  # fmt: skip
    for key in {(s, t) for s, t, _ in keys}:
        samples = sorted(x[1] for x in out.values() if x[0] == key)
        if samples != list(range(SAMPLES)):
            raise Pending(f"{name}: {key} has {len(samples)} of its {SAMPLES} hypotheses")
    return out


def judge_settings(version: str) -> Any:
    """The frozen judge of rubric set ``version`` (``JUDGE``: models, thinking levels, cap,
    structured output, the lenient grader)."""
    from .config import load_config
    from .judge import JudgeSettings, rubric_set

    j = load_config(JUDGE, ["tracking.wandb.enabled=false"]).judge

    def spec(model_id: str) -> Any:
        return _spec(next(m for m in j.models if m.id == model_id))

    if (j.model, j.lenient_model) != (JUDGE_MODELS["strict"], JUDGE_MODELS["lenient"]):
        raise ValueError(f"{JUDGE}: not the frozen pair's models")
    return JudgeSettings(model=spec(j.model), rubrics=rubric_set(version), rubric_version=version,
                         lenient=bool(j.lenient), max_tokens=int(j.max_tokens),
                         structured_output=bool(j.structured_output),
                         lenient_model=spec(j.lenient_model))  # fmt: skip


def labels(find_run: Any, name: str, kind: str, hyps: Mapping[str, tuple], behavior: str,
           pinned: Mapping[str, Mapping[str, Any]] | None = None) -> dict:  # fmt: skip
    """{response id: (strict, lenient, histories)} from the judge run: one primary label per
    hypothesis for its own target, each of its two requests the frozen judge's full request
    (``judge_settings``: ``build_request`` of the rubric, the hypothesis text and the target),
    recomputed with its retry (``slot``) on the standard tier, the label the slot's answer. A
    hypothesis without a label whose strict request and retry were both blocked by the safety
    filter (``blocked_slot``) and whose lenient slot holds a valid answer has the strict status
    ``STRICT_SAFETY_BLOCKED`` (§12.4.5, 2026-10-26); any other unlabelled hypothesis is
    pending. Every attempt read is admitted (``admitted_tier``): on the standard tier, or a
    historical reply the committed history ``pinned`` holds (§12.4.5, 2026-10-27); a tier-less
    reply it does not pin leaves the hypothesis pending."""
    from .judge import (
        JudgeItem,
        load_targets,
        parse_lenient,
        parse_strict,
        request_key,
        stage_requests,
    )
    from .records import JudgeLabel, read_table

    version = RUBRICS[kind]
    settings = judge_settings(version)
    run, config, _ = _run(find_run, name, None, EXPERIMENTS["judge"])
    inputs = json.loads((run / "judge_inputs.json").read_text())
    rubrics = settings.rubrics
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
    pinned = pinned or {}

    def admitted(found: Mapping[str, Any], where: str) -> bool:
        for step in found["history"]:
            tier = admitted_tier(step, pinned)
            if tier is None and step["tier"] is None:
                raise Pending(f"{where}: a historical judge reply without its provenance")
            if tier is None:
                return False
        return True

    out: dict[str, tuple] = {}
    for lab in read_table([run], JudgeLabel.table):
        if lab.response_id not in hyps or lab.stage != "primary":
            continue
        if lab.target_behavior_id != behavior:
            raise ValueError(f"{lab.response_id}: a label for {lab.target_behavior_id}")
        if lab.response_id in out:
            raise ValueError(f"{lab.response_id}: two primary labels for its own target")
        item = JudgeItem(lab.response_id, behavior, target, hyps[lab.response_id][2])
        requests = stage_requests([item], settings)[0]
        found = []
        for request, rid, parse in (
            (requests.strict, lab.request_id, parse_strict),
            (requests.lenient, lab.lenient_request_id, parse_lenient),
        ):
            retry = request_key(f"{request.rubric_version}{RETRY}", request.body)
            s = slot(cache, [request.key, retry], request.body, parse,
                     f"{lab.response_id} ({request.kind})")  # fmt: skip
            if s["key"] != rid or not admitted(s, f"{lab.response_id} ({request.kind})"):
                raise ValueError(f"{lab.response_id}: its {request.kind} label is not its slot's")
            found.append(s)
        if (lab.rubric_version, lab.strict_label, lab.lenient_label) != (
            version, found[0]["value"][0], found[1]["value"]
        ):  # fmt: skip
            raise ValueError(f"{lab.response_id}: its labels are not its answers'")
        out[lab.response_id] = (lab.strict_label, lab.lenient_label,
                                [f["history"] for f in found])  # fmt: skip
    for rid in sorted(set(hyps) - set(out)):
        requests = stage_requests([JudgeItem(rid, behavior, target, hyps[rid][2])], settings)[0]
        strict, lenient = requests.strict, requests.lenient
        blocked = blocked_slot(
            cache, [strict.key, request_key(f"{strict.rubric_version}{RETRY}", strict.body)],
            strict.body, f"{rid} (strict)", pinned)  # fmt: skip
        if blocked is None:
            continue
        retry = request_key(f"{lenient.rubric_version}{RETRY}", lenient.body)
        try:
            s = slot(cache, [lenient.key, retry], lenient.body, parse_lenient, f"{rid} (lenient)")
        except Pending:
            continue
        if not admitted(s, f"{rid} (lenient)"):
            continue
        out[rid] = (STRICT_SAFETY_BLOCKED, s["value"], [blocked, s["history"]])
    missing = sorted(set(hyps) - set(out))
    if missing:
        raise Pending(f"{name}: {len(missing)} hypotheses without their label, {missing[:2]}")
    return out


def category_grades(find_run: Any, name: str, texts: Mapping[str, str]) -> dict:
    """{response id: (frozenset of behaviors, history)} for exactly the hypotheses ``texts``
    (every counted one): each request the frozen grader's (``category-v1``, the frozen
    definitions, gemini-3.8-flash at its frozen thinking level, its cap) of the text, recomputed
    with its retry (``slot``) on the standard tier, each grade the slot's answer."""
    from .config import load_config
    from .continuation_definitions import load_definitions
    from .continuation_judge import CATEGORY_VERSION, category_request, parse_category
    from .judge import request_key

    run, config, _ = _run(find_run, name, None, EXPERIMENTS["category"])
    inputs = json.loads((run / CATEGORY_INPUTS).read_text())
    definitions, digest = load_definitions(expected_hash=DEFINITIONS_SHA256)
    e = next(m for m in load_config(JUDGE, ["tracking.wandb.enabled=false"]).judge.models
             if m.id == CATEGORY_MODEL)  # fmt: skip
    spec = _spec(e)
    if inputs != {"version": CATEGORY_VERSION, "definitions_sha256": digest,
                  "model": {"id": spec.id, "reasoning_effort": spec.reasoning_effort},
                  "service_tier": STANDARD, "max_tokens": CATEGORY_MAX_TOKENS}:  # fmt: skip
        raise ValueError(f"{name}: not graded with the frozen category instrument")
    cache = _cache(config, "category", "category")
    out: dict[str, tuple] = {}
    for line in (run / CATEGORY_LABELS).read_text().splitlines():
        row = json.loads(line)
        rid = row.get("response_id")
        if rid not in texts or rid in out:
            raise ValueError(f"{name}: {rid} is not one counted hypothesis, graded once")
        request = category_request(texts[rid], definitions, spec, max_tokens=CATEGORY_MAX_TOKENS)
        retry = request_key(f"{CATEGORY_VERSION}{RETRY}", request.body)
        s = slot(cache, [request.key, retry], request.body, parse_category, f"{rid} (category)")
        if row.get("request_key") != s["key"] or s["tier"] != STANDARD \
                or sorted(row.get("categories") or []) != sorted(s["value"]):  # fmt: skip
            raise ValueError(f"{rid}: its category grade is not its slot's")
        out[rid] = (s["value"], s["history"])
    missing = sorted(set(texts) - set(out))
    if missing:
        raise Pending(f"{name}: {len(missing)} counted hypotheses not graded, {missing[:2]}")
    return out


def _table_sha(run: Path, *paths: str) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode() + b"\0" + (run / p).read_bytes() + b"\0")
    return h.hexdigest()


def _retries(*groups: Any) -> dict[str, int]:
    """Slots answered on their retry, per stage (descriptive)."""
    return {name: sum(len(h) > 1 and h[-1]["parsed"] for h in histories)
            for name, histories in groups}  # fmt: skip


def safety_blocks(
    counts: Mapping[tuple[str, str], int], blocked: Mapping[tuple[str, str], Mapping[str, int]],
    label_of: Mapping[str, tuple],
) -> dict[str, Any]:  # fmt: skip
    """The strict-judge safety blocks of a lineage's readings (§12.4.5, 2026-10-26), all
    descriptive: the hypotheses blocked and their unique strict requests; per affected package
    its count c, its blocked hypotheses b and those with a lenient yes b_yes, and the interval
    R ∈ [c/80, (c + b_yes)/80]; and each test's contrast R(L) − R(C) with the bounds that
    completion would allow. None selects or replaces a verdict."""
    ids = sorted(rid for rid, v in label_of.items() if v[0] == STRICT_SAFETY_BLOCKED)
    packages = {
        f"{r}/{p}": {"count": counts[(r, p)], "blocked": b["blocked"],
                     "blocked_lenient_yes": b["lenient_yes"],
                     "interval": [_rate(counts[(r, p)]),
                                  _rate(counts[(r, p)] + b["lenient_yes"])]}
        for (r, p), b in sorted(blocked.items())
    }  # fmt: skip

    def yes(key: tuple[str, str]) -> int:
        return (blocked.get(key) or {}).get("lenient_yes", 0)

    contrasts = {}
    for package in (*DRAWS.values(), PD_CLEAN):
        for c in ("H_s", "X"):
            lk, ck = ("L", package), (c, package)
            if lk not in blocked and ck not in blocked:
                continue
            x_l, x_c = counts[lk], counts[ck]
            contrasts[f"{package}: R(L) - R({c})"] = {
                "value": _rate(x_l) - _rate(x_c),
                "bounds": [_rate(x_l) - _rate(x_c + yes(ck)), _rate(x_l + yes(lk)) - _rate(x_c)],
            }
    unique = {label_of[rid][2][0][0]["key"] for rid in ids}  # each blocked request once
    return {"items": len(ids), "unique_strict_requests": len(unique),
            "response_ids": ids, "packages": packages, "contrasts": contrasts}  # fmt: skip


def _bound_hypotheses(lineage: Lineage, sources: Mapping[str, Any], find_run: Any) -> dict:
    """The record, X's flag, the state checks, the lens and PD-clean evidence and each
    reading's hypotheses, each bound as ``readings`` binds them."""
    if sorted(sources) != sorted(SOURCES) or not all(isinstance(sources[k], str) for k in SOURCES):
        raise ValueError(f"the readers' sources {sorted(sources)} are not {sorted(SOURCES)}")
    record_run, record = pod_record(lineage, find_run, sources["record"])
    x_flag = x_validity(lineage, record, find_run)
    checks = state_checks(lineage, record, find_run)
    lens_ev, lens_files, reconstruction = lens_evidence(checks, lineage)
    pd_run, _, _ = _run(find_run, sources["pd_reading"], None, EXPERIMENTS["pd"])
    pd_names = [e["tool_run_id"] for e in json.loads(
        (pd_run / "summarizer_inputs.json").read_text()).get("evidence") or []]  # fmt: skip
    pd_ev, pd_files, pd_records = pd_evidence_of(find_run, lineage, record_run, pd_names)
    lens_h = hypotheses(find_run, sources["lens_reading"], "lens", list(lens_ev.values()))
    pd_h = hypotheses(find_run, sources["pd_reading"], "pd", list(pd_ev.values()))
    if sources["lens_reading"] == sources["pd_reading"] or set(lens_h) & set(pd_h):
        raise ValueError("the lens and PD-clean readings share a run or a response id")
    return {"record_run": record_run, "record": record, "x_flag": x_flag, "checks": checks,
            "lens_ev": lens_ev, "lens_files": lens_files, "reconstruction": reconstruction,
            "pd_ev": pd_ev, "pd_files": pd_files, "pd_records": pd_records, "lens_h": lens_h,
            "pd_h": pd_h}  # fmt: skip


def lineage_history(lineage: Lineage, sources: Mapping[str, Any], find_run: Any,
                    bound: Mapping[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """The lineage's judge history (``judge_history``) over its two judge runs and readings."""
    from .paths import runs_root

    b = bound or _bound_hypotheses(lineage, sources, find_run)
    found = judge_history(find_run, lineage.identity["behavior"],
                          {"lens": sources["lens_judge"], "pd": sources["pd_judge"]},
                          {"lens": b["lens_h"], "pd": b["pd_h"]}, runs_root())  # fmt: skip
    return {"lineage": lineage.identity["lineage"], **found}


def write_judge_history(root: Path, lineage: Lineage, sources: Mapping[str, Any],
                        find_run: Any) -> dict[str, Any]:  # fmt: skip
    """``lineage_history`` written once to ``manifests/stage3_judge_history_<lineage>.json``, to
    be committed before the readers' verdicts; an existing manifest is never replaced."""
    path = history_path(root, lineage.identity["lineage"])
    if path.exists():
        raise ValueError(f"{path} exists: the judge history is written once")
    data = lineage_history(lineage, sources, find_run)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return data


def committed_history(lineage: Lineage, sources: Mapping[str, Any], find_run: Any,
                      bound: Mapping[str, Any]) -> dict[str, Any] | None:  # fmt: skip
    """The committed judge history of the lineage, recomputed (``lineage_history``) and refused
    when it differs; None when there is none."""
    from .manifests import committed_json
    from .paths import REPO_ROOT

    path = history_path(REPO_ROOT, lineage.identity["lineage"])
    if not path.is_file():
        return None
    data = committed_json(path, "the judge history", REPO_ROOT)
    fresh = json.loads(json.dumps(lineage_history(lineage, sources, find_run, bound)))
    if {k: v for k, v in data.items() if k not in ("sha256", "committed_at")} != fresh:
        raise ValueError(f"{path.name}: not its recomputation from the writers' records")
    return data


def readings(lineage: Lineage, sources: Mapping[str, Any], rules: Any, find_run: Any) -> dict:
    """The counts, category grades, X's flag and bindings of the lineage's readings (module
    docstring). Pending while a run is missing, unfinished or short of a hypothesis, label or
    grade; an error for anything that does not bind."""
    from collections import Counter

    b = _bound_hypotheses(lineage, sources, find_run)
    record_run, record, x_flag = b["record_run"], b["record"], b["x_flag"]
    checks, lens_ev, lens_files = b["checks"], b["lens_ev"], b["lens_files"]
    reconstruction, pd_ev, pd_files = b["reconstruction"], b["pd_ev"], b["pd_files"]
    pd_records, lens_h, pd_h = b["pd_records"], b["lens_h"], b["pd_h"]
    behavior = lineage.identity["behavior"]
    history = committed_history(lineage, sources, find_run, b)
    pinned = {} if history is None else history["entries"]
    lens_l = labels(find_run, sources["lens_judge"], "lens", lens_h, behavior, pinned)
    pd_l = labels(find_run, sources["pd_judge"], "pd", pd_h, behavior, pinned)
    zero = {**lens_l, **pd_l}  # blocked strict labels counted as not naming b (2026-10-26)
    from .continuation_completion import completed_labels

    completed = completed_labels(lineage, sources, find_run, lens_h, lens_l, history)
    if completed is not None:  # the committed cohort completed (§12.4.5, 2026-10-27)
        lens_l = {**lens_l, **completed["labels"]}
    used = [step for ls in (lens_l, pd_l) for v in ls.values() for h in v[2] for step in h
            if step["key"] in pinned]  # fmt: skip
    reused = {
        "manifest": None if history is None else history["sha256"],
        "slot_references": len(used),
        "unique_entries": len({step["key"] for step in used}),
        "tiers": dict(sorted(Counter(pinned[k]["tier"] for k in {s["key"] for s in used}).items())),
    }
    c_of = {rid: counted(*lens_l[rid][:2]) for rid in lens_h} | {
        rid: counted(*pd_l[rid][:2]) for rid in pd_h}  # fmt: skip
    texts = {rid: h[2] for hs in (lens_h, pd_h) for rid, h in hs.items() if c_of[rid]}
    grades = category_grades(find_run, sources["category"], texts)

    def count(ev: Mapping, hyps: Mapping, c: Mapping[str, bool]) -> tuple[dict, dict]:
        where = {(v[0], v[1]): k for k, v in ev.items()}
        samples: dict[Any, dict[int, bool]] = {k: {} for k in ev}
        matches: dict[Any, list[frozenset[str]]] = {k: [] for k in ev}
        for rid, (key, sample, *_) in sorted(hyps.items(), key=lambda kv: kv[1][:2]):
            samples[where[key]][sample] = c[rid]
            if c[rid]:
                matches[where[key]].append(grades[rid][0])
        return ({k: package_count(v, f"{k}") for k, v in samples.items()}, matches)

    lens, lens_matches = count(lens_ev, lens_h, c_of)
    pd, pd_matches = count(pd_ev, pd_h, c_of)
    # the zero-count treatment (2026-10-26): the safety blocks' analysis, and once the cohort is
    # completed a sensitivity analysis of its own (``readers_record``)
    lens_zero, zero_matches = count(lens_ev, lens_h,
                                    {rid: counted(*zero[rid][:2]) for rid in lens_h})  # fmt: skip
    blocked: dict[tuple[str, str], dict[str, int]] = {}
    for ev, hyps in ((lens_ev, lens_h), (pd_ev, pd_h)):
        where = {(v[0], v[1]): k for k, v in ev.items()}
        for rid, (key, *_) in hyps.items():
            strict, lenient, _ = zero[rid]
            if strict == STRICT_SAFETY_BLOCKED:
                b = blocked.setdefault(where[key], {"blocked": 0, "lenient_yes": 0})
                b["blocked"] += 1
                b["lenient_yes"] += int(lenient)
    runs = {f"{record_run.name}/stage3_readers.json": file_sha(record_run / "stage3_readers.json")}
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
    labelled["PD-clean study records"] = hashlib.sha256(
        json.dumps(pd_records, sort_keys=True).encode()).hexdigest()  # fmt: skip
    histories = (
        ("interpreter", [h[4] for hs in (lens_h, pd_h) for h in hs.values()]),
        ("judge", [x for ls in (lens_l, pd_l) for v in ls.values() for x in v[2]]),
        ("category", [g[1] for g in grades.values()]),
    )  # fmt: skip
    # every slot's whole history (each attempt's key, cache entry sha256, tier, parse), bound
    every = {"interpreter": {rid: h[4] for hs in (lens_h, pd_h) for rid, h in hs.items()},
             "judge": {rid: v[2] for ls in (lens_l, pd_l) for rid, v in ls.items()},
             "category": {rid: g[1] for rid, g in grades.items()}}  # fmt: skip
    if history is not None:
        labelled["judge history"] = history["sha256"]
    labelled["slot histories"] = hashlib.sha256(
        json.dumps(every, sort_keys=True).encode()).hexdigest()  # fmt: skip
    return {
        "record": record,
        "x_validity": x_flag,
        "lens": lens,
        "pd": pd,
        "categories": {package: lens_matches[("L", package)] for package in DRAWS.values()},
        "margins": {f"{r}/{p}": m for (r, p), m in {**lens_matches, **pd_matches}.items()},
        "reconstruction": reconstruction,
        "tiers": _tier_counts(lens_h, pd_h),
        "retries": _retries(*histories),
        "strict_safety_blocked": safety_blocks({**lens_zero, **pd}, blocked, zero),
        "blocked_completion": None if completed is None else completed["record"],
        "zero_count": None if completed is None else {
            "lens": lens_zero,
            "categories": {package: zero_matches[("L", package)] for package in DRAWS.values()},
            "margins": {f"{r}/{p}": m for (r, p), m in {**zero_matches, **pd_matches}.items()}},
        "judge_history": reused,
        "pd_clean_records": pd_records,
        "bindings": {"packages": {**lens_files, **pd_files, **checks["files"]}, "runs": runs,
                     "labels": labelled},
    }  # fmt: skip


def _tier_counts(*groups: Mapping[str, tuple]) -> dict[str, int]:
    from collections import Counter

    return dict(sorted(Counter(h[3] for g in groups for h in g.values()).items()))


# ------------------------------------------------------------------------- the manifest


def _identity_fields(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {"lineage": identity["lineage"], "behavior": identity["behavior"],
            "dose": identity["dose"], "contracts": identity["contracts"],
            "final_detector": identity["detector"]}  # fmt: skip


def _xcont_runs(find_run: Any, name: str) -> dict[str, str]:
    from . import continuation_stage5 as S5

    run, _, summary = _run(find_run, name, S5.XCONT)
    out = {f"{run.name}/summary.json": file_sha(run / "summary.json")}
    for attempt in summary.get("attempts") or []:
        a, _, _ = _run(find_run, attempt, S5.XCONT_ATTEMPT)
        out[f"{a.name}/xcont_evidence.json"] = file_sha(a / "xcont_evidence.json")
    return out


def terminal_record(lineage: Lineage, sources: Mapping[str, Any], find_run: Any) -> dict:
    """An invalid X_cont's readers (§12.4.6): the lineage's X_cont cell (``strong_xcont``,
    invalid after its retry) named as the source, p₃ = p₄ = 1, the five states that exist and
    the runs that show it, nothing else."""
    xcont, identity = lineage.xcont or {}, lineage.identity
    if xcont.get("valid") is not False or dict(sources) != {"xcont": xcont.get("run")}:
        raise ValueError(
            "a terminal reading is an invalid X_cont cell's alone, named as its source"
        )
    runs = _xcont_runs(find_run, xcont["run"])
    return {
        **_identity_fields(identity),
        "sources": dict(sources),
        "states": {k: v for k, v in identity["states"].items() if k != "X"},
        "status": "terminal",
        "reasons": [f"X_cont of {identity['states']['L']} is invalid after its retry (§12.4.6)"],
        "x_validity": {"passed": False,
                       "inputs": {"control": "X_cont", "run": xcont["run"],
                                  "attempts": sorted(runs)}},
        "bindings": {"runs": runs},
        "p3": "1",
        "p4": "1",
    }  # fmt: skip


def verdict(
    lens: Mapping,
    pd: Mapping,
    categories: Mapping,
    margins: Mapping,
    behavior: str,
    rules: Any,
    *,
    tests: TestFactory | None = None,
    workers: int = 0,
) -> dict[str, Any]:
    """Components 3 and 4 of a reading's counts and category grades (the module docstring): their
    exact tests, the gates' flags with PD-clean's positive control, and every package's category
    margin."""
    from .continuation_judge import category_confusion

    th = thresholds(rules)
    jobs = component3_jobs(lens, th) + component4_jobs(pd, th)
    p = run_tests(jobs, rules, tests=tests, workers=workers)
    c3 = component3(lens, categories, behavior, th, p)
    c4 = component4(pd, th, p)
    flags = c3["flags"]
    flags["eligibility"] = {**flags["eligibility"],
                            "pd_positive_control": c4["pd_positive_control"]}  # fmt: skip
    return {"c3": c3, "c4": c4, "flags": flags,
            "margins": {k: category_confusion(v, behavior).margin
                        for k, v in sorted(margins.items())}}  # fmt: skip


def readers_record(
    lineage: Lineage, sources: Mapping[str, Any], rules: Any, find_run: Any, *,
    tests: TestFactory | None = None, workers: int = 0,
) -> dict[str, Any]:  # fmt: skip
    """The readers' manifest of a lineage, in the schema of ``continuation_panel.reader_values``,
    with its descriptive values beside it: terminal for an invalid X_cont, else complete from the
    bound readings (``sources``: ``SOURCES``)."""
    if lineage.xcont is not None and not lineage.xcont["valid"]:
        return serial(terminal_record(lineage, sources, find_run))
    found = readings(lineage, sources, rules, find_run)
    identity = lineage.identity
    v = verdict(found["lens"], found["pd"], found["categories"], found["margins"],
                identity["behavior"], rules, tests=tests, workers=workers)  # fmt: skip
    c3, c4, flags, margins = v["c3"], v["c4"], v["flags"], v["margins"]
    zero = found["zero_count"]
    if zero is not None:  # the completed cohort's zero-count sensitivity: it selects nothing
        z = verdict(zero["lens"], found["pd"], zero["categories"], zero["margins"],
                    identity["behavior"], rules, tests=tests, workers=workers)  # fmt: skip
        zero = {**z["flags"], "p3": z["c3"]["p3"], "p4": z["c4"]["p4"],
                "counts": {"lens": {f"{r}/{p}": x for (r, p), x in sorted(zero["lens"].items())}},
                "component3": {k: x for k, x in z["c3"].items() if k not in ("flags", "p3")},
                "component4": {k: x for k, x in z["c4"].items() if k != "p4"},
                "category_margins": z["margins"]}  # fmt: skip
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
        "component3": {**{k: v for k, v in c3.items() if k not in ("flags", "p3")},
                       "reconstruction": found["reconstruction"]},
        "component4": {k: v for k, v in c4.items() if k != "p4"},
        "category_margins": margins,
        "service_tiers": found["tiers"],
        "retries": found["retries"],
        "strict_safety_blocked": found["strict_safety_blocked"],
        "blocked_completion": found["blocked_completion"],
        "zero_count_sensitivity": zero,
        "judge_history": found["judge_history"],
        "pd_clean_records": found["pd_clean_records"],
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
    root: Path, lineage: Lineage, sources: Mapping[str, Any], rules: Any, find_run: Any, *,
    workers: int = 0,
) -> dict:  # fmt: skip
    """``readers_record`` written once to ``manifests/stage3_readers_<lineage>.json``; an
    existing manifest is never replaced. Pending work writes nothing."""
    path = readers_path(root, lineage.identity["lineage"])
    if path.exists():
        raise ValueError(f"{path} exists: the readers' manifest is written once")
    data = readers_record(lineage, sources, rules, find_run, workers=workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return data


def committed_readers(path: Path, root: Path, rules: Any, find_run: Any, *,
                      lineage_for: Any = None, workers: int = 0,
                      tests: TestFactory | None = None) -> dict[str, Any]:  # fmt: skip
    """The committed readers' manifest recomputed from its sources (``readers_record``) for its
    behavior's certified lineage (``lineage_of`` under the committed contracts, or
    ``lineage_for(behavior)``); any difference, a changed bound file among them, is refused.
    Returns the data with its sha256, as the panel's analysis reads it."""
    from .manifests import committed_json

    data = committed_json(Path(path), "the readers' manifest", Path(root))
    behavior = data.get("behavior")
    lineage = (lineage_for or (lambda b: lineage_of(root, b, rules, find_run)))(behavior)
    if Path(path).name != readers_path(root, lineage.identity["lineage"]).name:
        raise ValueError(f"{path}: not the readers' manifest of {lineage.identity['lineage']}")
    stored = {k: v for k, v in data.items() if k not in ("sha256", "committed_at")}
    fresh = readers_record(lineage, stored.get("sources") or {}, rules, find_run, tests=tests,
                           workers=workers)  # fmt: skip
    if json.loads(json.dumps(fresh, sort_keys=True)) != stored:
        raise ValueError(f"{path}: not what its sources give")
    return data
