"""The follow-up's readings of L, H_s and X by D* and unfiltered PD (preregistration §13.5, §13.14's
step 2, §13.16), on the API from recorded packages only.

- **Readings** (``readings``): B3's analysis inventory (``manifests/follow_b3_inventory.json``) at
  the committed confirmations, without countdown's base-decoder readings: D*'s reading of countdown
  is the fixed exploratory one of §13.16. Unfiltered PD's reused readings (the inventory's ``reuse``
  map) are read back, every attempt at its recorded sha256 (``reused_slots``), never asked again.
- **Requests and labels:** as the qualification's (``follow_qualification``): 80 slots per reading,
  the frozen interpreter and judges, each attempt admitted at its tier.
- **Grades** (``grades``): each counted hypothesis of an L reading on the state or base decoder, or
  of unfiltered PD, graded by ``category-v1`` and ``specificity-v1`` on the standard tier, each with
  its one retry.
- **Analysis** (``analysis``): per reader the identification test of §13.5 (c) at the certified
  regions, gated by the reader's qualification and format validation and by category confusion
  (§13.5 (b)); Holm over the six partial-threshold lineages, countdown's D* entry at p = 1 (§13.16)
  and reported beside it as exploratory; the base decoder's corroboration in its own family; rung 1
  on eli5 continued 200 (§13.6: level 1/60, the random cell at most 8 of 80). A specific claim needs
  a validated specificity grader and the defining-feature advantage (``specific``); category
  recognition is reported as such. A pending slot leaves everything it enters pending.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import follow_qualification as Q

EXPLORATORY = "countdown"  # §13.16: D*'s test has p = 1, one fixed reading outside every family
RUNG_LEVEL = Fraction(1, 60)
RANDOM_MAX = 8  # of 80: §13.6's gate on a rung's random cell


def readings(confirmed: Mapping[str, int], repo: Path | None = None) -> list[dict[str, Any]]:
    """The analysis readings at the confirmed doses, D*'s then unfiltered PD's (see the module)."""
    from . import continuation_rules as cr
    from . import follow_inventory as fi
    from .paths import REPO_ROOT

    path = (repo or REPO_ROOT) / cr.follow_b3()["inventory"]["manifest"]
    manifest = json.loads(path.read_text())
    parts = fi.resolve({k: manifest[k] for k in ("dstar", "pd_unfiltered")}, confirmed)
    return [r for r in (*parts["dstar"]["analysis"], *parts["pd_unfiltered"]["analysis"])
            if not (r["reader"] == "dstar" and r["behavior"] == EXPLORATORY
                    and r["decoder"] == "base")]  # fmt: skip


def expected_states(rows: Sequence[Mapping[str, Any]]) -> dict[tuple, str]:
    """Each role's state by (behavior, mode, role), as the inventory's PD readings name it."""
    return {(r["behavior"], r["mode"], r["role"]): r["package"]["state"] for r in rows
            if r["reader"] == "pd_unfiltered"}  # fmt: skip


def package_dir(reading: Mapping[str, Any], find_run: Callable[[str], Any],
                expected: Mapping[tuple, str], workspace: Callable[[str], Any],
                branches: Callable[[], Mapping[str, Mapping[str, str]]]) -> Path:  # fmt: skip
    """A reading's evidence directory: a random branch through its committed record
    (``follow_inventory.random_branches``), a build through the build session's directory with
    its record naming the state, an existing package or pool as the qualification binds it."""
    from . import follow_inventory as fi

    p = reading["package"]
    if "new" in p:
        found = branches()[p["new"]]
        return Path(find_run(found["build"])) / found["dir"]
    if "build" in p:
        run = fi.build_dir(p["build"], p["state"])
        if json.loads((run / "build.json").read_text()).get("state") != p["state"]:
            raise ValueError(f"{reading['salt']}: its build is not of {p['state']}")
        return run
    key = (reading["behavior"], reading["role"])
    want = expected[(reading["behavior"], reading["mode"], reading["role"])]
    return Q.package_dir(reading, find_run, {}, {key: want}, workspace)


def _read_back(history: Sequence[Mapping[str, Any]], cache: Any, parse: Callable[[Any], Any],
               where: str) -> dict[str, Any]:  # fmt: skip
    """A reused slot's stage from its recorded attempts, each the entry at its sha256: the first
    answer the record marks parsed, or refused."""
    from .judge import cache_entry_sha256

    steps = []
    for step in history:
        entry = cache.entry(step["key"])
        if entry is None or cache_entry_sha256(entry) != step["sha256"]:
            raise ValueError(f"{where}: {step['key']} is not its recorded answer")
        steps.append({**step, "admitted": step.get("tier")})
        if step.get("parsed"):
            return {"value": parse(entry.get("text")), "key": step["key"], "history": steps}
    return {"refused": True, "history": steps}


def reused_slots(decision: Mapping[str, Any], summary: Any, judge: Any,
                 where: str) -> list[dict[str, Any]]:  # fmt: skip
    """A reused reading's slots from the committed reuse decision, in the qualification's shape."""
    from .judge import parse_lenient, parse_strict
    from .summarizer import parse_hypothesis

    out = []
    for s in sorted(decision["slots"], key=lambda x: x["slot"]):
        at = f"{where}#slot={s['slot']}"
        hyp = _read_back(s["history"], summary, parse_hypothesis, at)
        strict, lenient = decision["labels"][s["response_id"]]
        out.append({"hypothesis": hyp, "labels": {
            "strict": _read_back(strict, judge, parse_strict, f"{at} (strict)"),
            "lenient": _read_back(lenient, judge, parse_lenient, f"{at} (lenient)")}})  # fmt: skip
    return out


def grader() -> tuple[Mapping[str, Any], Any]:
    """The frozen definitions and the category and specificity grader's model (B2's)."""
    from . import continuation_readers as readers
    from .config import load_config
    from .continuation_definitions import load_definitions

    definitions, _ = load_definitions(expected_hash=readers.DEFINITIONS_SHA256)
    models = load_config(readers.JUDGE, ["tracking.wandb.enabled=false"]).judge.models
    return definitions, readers._spec(next(m for m in models if m.id == readers.CATEGORY_MODEL))


def grade_attempts(hypothesis: str, behavior: str, definitions: Mapping[str, Any],
                   spec: Any) -> dict[str, list[Any]]:  # fmt: skip
    """A hypothesis's category and specificity requests, each with its one retry."""
    from .continuation_judge import (
        CATEGORY_VERSION,
        SPECIFICITY_VERSION,
        category_request,
        specificity_request,
    )
    from .continuation_readers import CATEGORY_MAX_TOKENS, SPECIFICITY_MAX_TOKENS
    from .judge import ChatRequest, request_key
    from .summarizer import RETRY

    def tries(r: Any, version: str) -> list[Any]:
        return [r, ChatRequest(request_key(f"{version}{RETRY}", r.body), r.kind,
                               r.rubric_version, r.body)]  # fmt: skip

    return {"category": tries(category_request(hypothesis, definitions, spec,
                                               max_tokens=CATEGORY_MAX_TOKENS), CATEGORY_VERSION),
            "specificity": tries(specificity_request(hypothesis, definitions[behavior], spec,
                                                     max_tokens=SPECIFICITY_MAX_TOKENS),
                                 SPECIFICITY_VERSION)}  # fmt: skip


def grades(hypothesis: str, behavior: str, definitions: Mapping[str, Any], spec: Any,
           cache: Any, where: str) -> dict[str, Any]:  # fmt: skip
    """A counted hypothesis's two grades from the cache, each attempt on the standard tier."""
    from .continuation_judge import parse_category, parse_specificity

    def admit(step: Mapping[str, Any]) -> str | None:
        return step["tier"] if step["tier"] == "standard" else None

    tries = grade_attempts(hypothesis, behavior, definitions, spec)
    return {"category": Q.answer(cache, tries["category"], parse_category, admit,
                                 f"{where} (category)"),
            "specificity": Q.answer(cache, tries["specificity"],
                                    lambda t: parse_specificity(t, definitions[behavior]), admit,
                                    f"{where} (specificity)")}  # fmt: skip


def graded_reading(reading: Mapping[str, Any]) -> bool:
    """Whether a reading's counted hypotheses are graded: L, on a read decoder."""
    return reading["role"] == "L" and reading["decoder"] in ("state", "base", None)


def history_of(slot: Mapping[str, Any]) -> dict[str, Any]:
    """A slot's attempts by stage, its grades' among them."""
    out = Q.history_of(slot)
    for k, g in (slot.get("grades") or {}).items():
        out[k] = g["history"]
    return out


def margins(slots: Sequence[Mapping[str, Any]], behavior: str, definition: Any,
            rules: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """An L reading's category confusion and defining-feature advantage over its counted
    hypotheses (§13.5 (b); §12.3.4), None while a grade is open. Zero counted hypotheses fail."""
    from .continuation_judge import category_confusion, specificity_margin
    from .continuation_readers import counted

    chosen = [s for s in slots if counted(s["labels"]["strict"]["value"][0],
                                          s["labels"]["lenient"]["value"])]  # fmt: skip
    grades_ = [s.get("grades") for s in chosen]
    if any(g is None or "value" not in g["category"] or "value" not in g["specificity"]
           for g in grades_):  # fmt: skip
        return {"category": None, "specificity": None, "passed": None}
    confusion = category_confusion([g["category"]["value"] for g in grades_], behavior)
    feature = specificity_margin([g["specificity"]["value"][0] for g in grades_], definition)
    minimum = Fraction(rules["specificity_min"])
    return {"counted": len(chosen),
            "category": {"correct": str(confusion.correct), "best_other": confusion.best_other,
                         "best_other_share": str(confusion.best_other_share),
                         "margin": str(confusion.margin)},
            "passed": bool(chosen) and confusion.margin >= Fraction(rules["confusion_min"]),
            "specificity": {"feature": str(feature.correct), "best_alternative": feature.best_other,
                            "best_alternative_share": str(feature.best_other_share),
                            "advantage": str(feature.margin)},
            "specific": bool(chosen) and feature.margin >= minimum}  # fmt: skip


def line_name(behavior: str, mode: str) -> str:
    """The qualification's lineage name of a reading's line."""
    return f"{behavior}_cont" if mode == "continued" else behavior


def _line(rows: Mapping[tuple, Mapping[str, Any]], reader: str, behavior: str, mode: str,
          decoder: str | None, regions: Any) -> dict[str, Any]:  # fmt: skip
    """A line's three counts on one decoder and its decisions at every level, None while pending."""
    from .follow_regions import LEVELS

    cells = [rows.get((reader, behavior, mode, role, decoder)) for role in ("L", "H_s", "X")]
    if any(c is None or "counted" not in c["count"] for c in cells):
        return {"counts": None, "by_level": None}
    x_l, x_hs, x_x = (c["count"]["counted"] for c in cells)
    decided = regions.by_level(x_l, x_hs, x_x)
    return {"counts": {"L": x_l, "H_s": x_hs, "X": x_x},
            "by_level": {a: decided[a] for a in LEVELS}}  # fmt: skip


def analysis(
    rows: Sequence[Mapping[str, Any]], floors: Mapping[str, Any], accepted: Mapping[str, bool],
    validated: Mapping[str, bool], regions: Any, rules: Mapping[str, Any],
    definitions: Mapping[str, Any],
) -> dict[str, Any]:  # fmt: skip
    """The readings' record (see the module); ``floors`` the qualification's, ``accepted`` each
    format's validation, ``validated`` each format's specificity grader."""
    from . import continuation_stats as cs
    from .follow_regions import ALPHA, LEVELS

    by: dict[tuple, Mapping[str, Any]] = {}
    for r in rows:
        x = r["reading"]
        by[(x["reader"], x["behavior"], x["mode"], x["role"], x["decoder"])] = r
    lines = sorted({(x[1], x[2]) for x in by})
    never = dict.fromkeys(LEVELS, False)
    out: dict[str, Any] = {"lines": {}}
    families: dict[str, dict[str, dict]] = {"dstar": {}, "dstar_corroboration": {},
                                            "pd_unfiltered": {}}  # fmt: skip
    pending = False
    for behavior, mode in lines:
        name = line_name(behavior, mode)
        entry: dict[str, Any] = {}
        for reader, decoder in (("dstar", "state"), ("dstar", "base"), ("pd_unfiltered", None)):
            line = _line(by, reader, behavior, mode, decoder, regions)
            cell = by.get((reader, behavior, mode, "L", decoder))
            graded = None
            if cell is not None and "counted" in cell["count"]:
                graded = margins(cell["slots"], behavior, definitions[behavior], rules)
            f = floors[name]
            q = f["dstar"]
            eligible = (f["pd_unfiltered"] if reader == "pd_unfiltered"
                        else q["eligible"] if decoder == "state"
                        else q["eligible"] and q["base_qualification"])  # fmt: skip
            fmt = "pd_unfiltered" if reader == "pd_unfiltered" else "dstar"
            gates = eligible and accepted[fmt] and bool(graded and graded.get("passed"))
            if line["by_level"] is None:
                decided = None
                # countdown has no base-decoder reading by design: not a pending line
                if not (behavior == EXPLORATORY and reader == "dstar" and decoder == "base"):
                    pending = True
            else:
                decided = line["by_level"] if gates else never
            label = f"{reader}/{decoder}" if decoder else reader
            entry[label] = {"counts": line["counts"], "eligible": eligible, "margins": graded,
                            "gates_passed": gates,
                            "decisions": None if decided is None
                            else {str(a): v for a, v in decided.items()},
                            "contrasts_without_gates": None if line["by_level"] is None
                            else {str(a): v for a, v in line["by_level"].items()}}  # fmt: skip
            if graded is not None:
                entry[label]["specific_claim"] = bool(graded.get("specific")) and validated[fmt]
            if mode != "stack":
                continue
            if reader == "pd_unfiltered":
                families["pd_unfiltered"][behavior] = decided if decided is not None else never
            elif decoder == "state":
                entry["exploratory"] = behavior == EXPLORATORY
                families["dstar"][behavior] = (never if behavior == EXPLORATORY or decided is None
                                               else decided)  # fmt: skip
        if mode == "stack" and "dstar/state" in entry:
            state, base = entry["dstar/state"]["decisions"], entry["dstar/base"]["decisions"]
            families["dstar_corroboration"][behavior] = (
                never if behavior == EXPLORATORY or state is None or base is None
                else {a: state[str(a)] and base[str(a)] for a in LEVELS})  # fmt: skip
        random = by.get(("dstar", behavior, mode, "L", "random"))
        if random is not None:
            entry["random"] = random["count"].get("counted")
            pending = pending or "counted" not in random["count"]
        out["lines"][name] = entry
    out["pending"] = pending
    out["holm"] = {k: cs.holm_from_decisions(v, ALPHA) for k, v in families.items()
                   if len(v) == len(LEVELS)}  # fmt: skip
    rung = out["lines"].get("eli5_cont")
    if rung is not None and rung["dstar/state"]["decisions"] is not None:
        state, base = rung["dstar/state"], rung["dstar/base"]
        random_ok = rung.get("random") is not None and rung["random"] <= RANDOM_MAX
        passed = bool(state["decisions"][str(RUNG_LEVEL)]) and random_ok
        corroborated = bool(base["decisions"] and base["decisions"][str(RUNG_LEVEL)])
        out["rung1"] = {"passed": passed, "random": rung.get("random"),
                        "corroborated": passed and corroborated}  # fmt: skip
    return out
