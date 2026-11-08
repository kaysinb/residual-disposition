"""The follow-up's readings of L, H_s and X by D* and unfiltered PD (preregistration §13.5, §13.14's
step 2, §13.16), on the API from recorded packages only.

- **Readings** (``readings``): B3's analysis inventory (``manifests/follow_b3_inventory.json``) at
  the committed confirmations, with the base decoder's controls on A beside it
  (``base_fidelity_readings``) and a standard-only repeat of eli5 stack run apart
  (``sensitivity_readings``). Unfiltered PD's reused readings (the inventory's ``reuse`` map) are
  read back, every attempt at its recorded sha256 (``reused_slots``), never asked again.
- **Requests and labels:** as the qualification's (``follow_qualification``): 80 slots per reading,
  the frozen interpreter and judges, each attempt admitted at its tier.
- **Grades** (``grades``): each counted hypothesis of an L reading on the state or base decoder, or
  of unfiltered PD, graded by ``category-v1`` and ``specificity-v1`` on the standard tier, each with
  its one retry.
- **Analysis** (``analysis``): the state decoder and the base decoder are two instruments of equal
  standing, each with its own Holm family over the six partial-threshold lineages, unfiltered PD a
  third: the identification test of §13.5 (c) at the certified regions, gated by that instrument's
  qualification and controls, its format validation and category confusion (§13.5 (b)). Countdown's
  state-decoder entry stays at p = 1 with its fixed exploratory reading beside it (§13.16); its
  base-decoder entry follows the ordinary rules. Both decoders' agreement is reported per line and
  decides nothing. Rung 1 on eli5 continued 200 runs on both decoders, each with its own gates and
  test (level 1/60, its random cell at most 8 of 80 through that decoder). The specificity criterion
  (a validated grader and the defining-feature advantage) is reported apart from the decision: a
  specific claim needs both it and the family's rejection; category recognition is reported as such.
  A pending slot, grade or control leaves everything it enters pending, never a negative.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import follow_qualification as Q

EXPLORATORY = "countdown"  # §13.16: its state-decoder entry at p = 1, one fixed reading beside it
RUNG_LEVEL = Fraction(1, 60)
RANDOM_MAX = 8  # of 80: §13.6's gate on a rung's random cell
BASE_CONTROLS = "manifests/follow_base_controls.json"  # the base decoder's controls' record
BASE_BRANCHES = ("base_random", "base_no_patch")
STREAMS = (1, 2)  # the base decoder's controls, pooled once as the state decoder's


def readings(confirmed: Mapping[str, int], repo: Path | None = None) -> list[dict[str, Any]]:
    """The analysis readings at the confirmed doses, D*'s then unfiltered PD's, then the base
    decoder's controls on A."""
    from . import continuation_rules as cr
    from . import follow_inventory as fi
    from .paths import REPO_ROOT

    path = (repo or REPO_ROOT) / cr.follow_b3()["inventory"]["manifest"]
    manifest = json.loads(path.read_text())
    parts = fi.resolve({k: manifest[k] for k in ("dstar", "pd_unfiltered")}, confirmed)
    return [*parts["dstar"]["analysis"], *parts["pd_unfiltered"]["analysis"],
            *base_fidelity_readings(repo)]  # fmt: skip


STACK_DECODERS = ("state", "base")
RANDOM_OF = {"state": "random", "base": "base_random"}  # L's random directions per decoder


def stack_readings(states: Mapping[str, str], dose: int = 800,
                   decoders: Sequence[str] = STACK_DECODERS) -> list[dict[str, Any]]:  # fmt: skip
    """Rung 1 on eli5's stack at ``dose``: the validated candidate 800 (§13.7) on both decoders, or
    the stack at 400 (§13.16) on the base decoder alone. Its recipe control's qualification, two
    streams of 80 on each decoder read and by unfiltered PD (§13.5 (a): each mode's recipe control
    is qualified separately, A and H shared); its L, H_s and X on each decoder and by unfiltered
    PD; and L's random directions through each decoder (§13.16, point 5)."""
    if not decoders or set(decoders) - set(STACK_DECODERS) or len(set(decoders)) != len(decoders):
        raise ValueError(f"decoders {list(decoders)}, not distinct of {STACK_DECODERS}")
    decoders = [d for d in STACK_DECODERS if d in decoders]
    lineage = f"stack{int(dose)}"

    def row(reader: str, role: str, decoder: str | None, stream: int, purpose: str) -> dict:
        tail = f"{role}/{decoder}" if reader == "dstar" else role
        name = "dstar" if reader == "dstar" else "pd"
        return {"reader": reader, "behavior": "eli5", "role": role, "decoder": decoder,
                "stream": stream, "purpose": purpose, "mode": lineage, "dose": int(dose),
                "if_confirmed": None, "slots": 80, "package": {"state": states[role]},
                "salt": f"follow/{name}/{purpose}/eli5/{lineage}/{tail}/stream{stream}",
                }  # fmt: skip

    rows = [row("dstar", "recipe", d, s, "qualification") for d in decoders for s in STREAMS]
    rows += [row("pd_unfiltered", "recipe", None, s, "qualification") for s in STREAMS]
    rows += [row("dstar", r, d, 1, "analysis") for r in ("L", "H_s", "X") for d in decoders]
    rows += [row("dstar", "L", RANDOM_OF[d], 1, "analysis") for d in decoders]
    rows += [row("pd_unfiltered", r, None, 1, "analysis") for r in ("L", "H_s", "X")]
    return rows


def _ladder_l(repo: Path | None = None) -> dict[str, str]:
    """Eli5 continued 200's L package and state, as B3's inventory names them (rung 1's cell)."""
    from . import continuation_rules as cr
    from .paths import REPO_ROOT

    path = (repo or REPO_ROOT) / cr.follow_b3()["inventory"]["manifest"]
    manifest = json.loads(path.read_text())

    def of(rows: Sequence[Mapping[str, Any]], decoder: str | None) -> Mapping[str, Any]:
        wanted = ("eli5", "continued", "L", decoder)
        return next(r["package"] for r in rows
                    if (r["behavior"], r["mode"], r["role"], r["decoder"]) == wanted)  # fmt: skip

    return {"run": of(manifest["dstar"]["analysis"], "state")["run"],
            "state": of(manifest["pd_unfiltered"]["analysis"], None)["state"]}  # fmt: skip


def _a_packages(repo: Path | None = None) -> dict[str, dict[str, str]]:
    """Each behavior's A package and state, as B2's frozen inventory names them."""
    from . import continuation_rules as cr
    from .paths import REPO_ROOT

    path = (repo or REPO_ROOT) / cr.follow_b2()["inventory"]["manifest"]
    manifest = json.loads(path.read_text())
    runs = {r["behavior"]: r["package"]["run"] for r in manifest["dstar"]["qualification"]
            if (r["role"], r["decoder"], r["stream"]) == ("A", "state", 1)}  # fmt: skip
    pd = manifest["pd_unfiltered"]["qualification"]
    states = {r["behavior"]: r["package"]["state"] for r in pd
              if (r["role"], r["stream"]) == ("A", 1)}  # fmt: skip
    return {b: {"run": runs[b], "state": states[b]} for b in sorted(runs)}


def base_fidelity_readings(repo: Path | None = None) -> list[dict[str, Any]]:
    """The base decoder's controls on A: random directions and no patch, two streams of 80."""
    rows = [{"reader": "dstar", "behavior": b, "role": "A", "decoder": branch, "stream": stream,
             "purpose": "fidelity", "mode": None, "dose": None, "if_confirmed": None, "slots": 80,
             "package": {"base_control": p["run"]},
             "salt": f"follow/dstar/fidelity/{b}/A/{branch}/stream{stream}"}
            for b, p in _a_packages(repo).items() for branch in BASE_BRANCHES
            for stream in STREAMS]  # fmt: skip
    rows.append({"reader": "dstar", "behavior": "eli5", "role": "L", "decoder": "base_random",
                 "stream": 1, "purpose": "analysis", "mode": "continued", "dose": 200,
                 "if_confirmed": None, "slots": 80,
                 "package": {"base_control": _ladder_l(repo)["run"]},
                 "salt": "follow/dstar/analysis/eli5/continued200/L/base_random/stream1"})
    return rows


def base_control_packages(find_run: Callable[[str], Any],
                          repo: Path | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """Each A package, and eli5 continued's L (its rung 1's random cell through the base decoder),
    with its behavior, state, recorded seed and branches, for the Pod's session."""
    planned = [(b, p, BASE_BRANCHES) for b, p in _a_packages(repo).items()]
    planned.append(("eli5", _ladder_l(repo), ("base_random",)))
    out = []
    for b, p, branches in planned:
        record = json.loads((Path(find_run(p["run"])) / "adl_lens_package.json").read_text())
        if record["state"] != p["state"]:
            raise ValueError(f"{p['run']}: not the package of {p['state']}")
        out.append({"behavior": b, "run": p["run"], "state": p["state"],
                    "seed": int(record["seed"]), "branches": list(branches)})  # fmt: skip
    return out


def base_controls(find_run: Callable[[str], Any], repo: Path | None = None) -> dict[str, Path]:
    """Each A package's base-control directory, bound by the committed record: exactly its
    recorded files, each at its sha256."""
    import hashlib

    from .manifests import committed_json
    from .paths import REPO_ROOT

    root = repo or REPO_ROOT
    record = committed_json(root / BASE_CONTROLS, "the base decoder's controls' record", repo)
    run = Path(find_run(record["run"]))
    out = {}
    for name, p in record["packages"].items():
        d = run / p["dir"]
        files = {str(f.relative_to(d)): hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in sorted(d.rglob("*")) if f.is_file()}  # fmt: skip
        if files != dict(p["files"]):
            raise ValueError(f"{name}: its base controls are not their recorded files")
        out[name] = d
    return out


def expected_states(rows: Sequence[Mapping[str, Any]]) -> dict[tuple, str]:
    """Each role's state by (behavior, mode, role), as the inventory's PD readings name it."""
    return {(r["behavior"], r["mode"], r["role"]): r["package"]["state"] for r in rows
            if r["reader"] == "pd_unfiltered"}  # fmt: skip


def package_dir(reading: Mapping[str, Any], find_run: Callable[[str], Any],
                expected: Mapping[tuple, str], workspace: Callable[[str], Any],
                branches: Callable[[], Mapping[str, Mapping[str, str]]],
                controls: Callable[[], Mapping[str, Path]]) -> Path:  # fmt: skip
    """A reading's evidence directory: a random branch through its committed record
    (``follow_inventory.random_branches``), a base control through its own (``base_controls``), a
    build through the build session's directory with its record naming the state, an existing
    package or pool as the qualification binds it."""
    from . import follow_inventory as fi

    p = reading["package"]
    if "base_control" in p:
        return controls()[p["base_control"]]
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
    hypotheses (§13.5 (b); §12.3.4), each None while one of its grades is open, the two apart.
    Zero counted hypotheses fail."""
    from .continuation_judge import category_confusion, specificity_margin
    from .continuation_readers import counted

    def yes(s: Mapping[str, Any]) -> bool:
        return counted(s["labels"]["strict"]["value"][0], s["labels"]["lenient"]["value"])

    chosen = [s.get("grades") or {} for s in slots if yes(s)]
    out: dict[str, Any] = {"counted": len(chosen), "category": None, "passed": None,
                           "specificity": None, "specific": None}  # fmt: skip
    if all("value" in (g.get("category") or {}) for g in chosen):
        confusion = category_confusion([g["category"]["value"] for g in chosen], behavior)
        out["category"] = {"correct": str(confusion.correct), "best_other": confusion.best_other,
                           "best_other_share": str(confusion.best_other_share),
                           "margin": str(confusion.margin)}  # fmt: skip
        out["passed"] = bool(chosen) and confusion.margin >= Fraction(rules["confusion_min"])
    if all("value" in (g.get("specificity") or {}) for g in chosen):
        feature = specificity_margin([g["specificity"]["value"][0] for g in chosen], definition)
        out["specificity"] = {"feature": str(feature.correct),
                              "best_alternative": feature.best_other,
                              "best_alternative_share": str(feature.best_other_share),
                              "advantage": str(feature.margin)}  # fmt: skip
        out["specific"] = bool(chosen) and feature.margin >= Fraction(rules["specificity_min"])
    return out


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


def base_fidelity(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the base decoder's controls on each behavior's A hold, each pooled over its two
    streams at most ``fidelity_max``; None while a slot is pending."""
    fidelity_max = Fraction(rules["qualification"]["fidelity_max"])
    pools: dict[str, dict[str, list]] = {}
    for r in rows:
        x = r["reading"]
        if x["decoder"] in BASE_BRANCHES and x["role"] == "A":
            pools.setdefault(x["behavior"], {}).setdefault(x["decoder"], []).append(r["count"])
    out: dict[str, Any] = {}
    for behavior, branches in pools.items():
        rates = {b: None if any("counted" not in c for c in counts)
                 else Fraction(sum(c["counted"] for c in counts), sum(c["n"] for c in counts))
                 for b, counts in branches.items()}  # fmt: skip
        out[behavior] = {"rates": {b: None if v is None else str(v) for b, v in rates.items()},
                         "passed": None if None in rates.values()
                         else all(v <= fidelity_max for v in rates.values())}  # fmt: skip
    return out


def analysis(
    rows: Sequence[Mapping[str, Any]], floors: Mapping[str, Any], accepted: Mapping[str, bool],
    validated: Mapping[str, bool], regions: Any, rules: Mapping[str, Any],
    definitions: Mapping[str, Any], fidelity_known: Mapping[str, Any] | None = None,
    decoders: Sequence[str] = STACK_DECODERS,
) -> dict[str, Any]:  # fmt: skip
    """The readings' record (see the module); ``floors`` the qualification's, ``accepted`` each
    format's validation, ``validated`` each format's specificity grader; ``decoders`` the D*
    decoders read (both, or the base decoder alone for the stack at 400: the other's family is
    absent, never pending)."""
    from . import continuation_stats as cs
    from .follow_regions import ALPHA, LEVELS

    fidelity = {**(fidelity_known or {}), **base_fidelity(rows, rules)}
    by: dict[tuple, Mapping[str, Any]] = {}
    for r in rows:
        x = r["reading"]
        if x["decoder"] not in BASE_BRANCHES or x["role"] == "L":
            by[(x["reader"], x["behavior"], x["mode"], x["role"], x["decoder"])] = r
    lines = sorted({(x[1], x[2]) for x in by})
    never = dict.fromkeys(LEVELS, False)
    out: dict[str, Any] = {"lines": {}, "base_fidelity": fidelity}
    families: dict[str, dict[str, dict]] = {"dstar_state": {}, "dstar_base": {},
                                            "pd_unfiltered": {}}  # fmt: skip
    pending = any(v["passed"] is None for v in fidelity.values())
    for behavior, mode in lines:
        name = line_name(behavior, mode)
        q, pd_ok = floors[name]["dstar"], floors[name]["pd_unfiltered"]
        held = (fidelity.get(behavior) or {}).get("passed")
        base_ok = None if held is None else q["base_qualification"] and held
        entry: dict[str, Any] = {}
        for reader, decoder, family, eligible in (
            ("dstar", "state", "dstar_state", q["eligible"]),
            ("dstar", "base", "dstar_base", base_ok),
            ("pd_unfiltered", None, "pd_unfiltered", pd_ok),
        ):
            if reader == "dstar" and decoder not in decoders:
                continue
            line = _line(by, reader, behavior, mode, decoder, regions)
            cell = by.get((reader, behavior, mode, "L", decoder))
            graded = None
            if cell is not None and "counted" in cell["count"]:
                graded = margins(cell["slots"], behavior, definitions[behavior], rules)
            fmt = "pd_unfiltered" if reader == "pd_unfiltered" else "dstar"
            category = None if graded is None else graded["passed"]
            # an open grade or control is pending: it neither advances nor records a negative
            open_ = eligible is None or (eligible and accepted[fmt] and category is None)
            gates = None if open_ else bool(eligible and accepted[fmt] and category)
            decided = None
            if line["by_level"] is None or open_:
                pending = True
            else:
                decided = line["by_level"] if gates else never
            entry[family] = {"counts": line["counts"], "eligible": eligible, "margins": graded,
                             "gates_passed": gates,
                             "decisions": None if decided is None
                             else {str(a): v for a, v in decided.items()},
                             "contrasts_without_gates": None if line["by_level"] is None
                             else {str(a): v for a, v in line["by_level"].items()}}  # fmt: skip
            if graded is not None:  # the criterion alone: a claim needs the family's rejection too
                met = graded["specific"]
                entry[family]["specificity_satisfied"] = (None if met is None
                                                          else met and validated[fmt])  # fmt: skip
            if mode == "stack":
                families[family][behavior] = decided
        state = (entry.get("dstar_state") or {}).get("decisions")
        base = (entry.get("dstar_base") or {}).get("decisions")
        both = None if state is None or base is None else {
            str(a): state[str(a)] and base[str(a)] for a in LEVELS}  # fmt: skip
        entry["both_decoders"] = both
        entry["exploratory_on_state_decoder"] = behavior == EXPLORATORY
        for label, branch in (("random", "random"), ("base_random", "base_random")):
            random = by.get(("dstar", behavior, mode, "L", branch))
            if random is not None:
                entry[label] = random["count"].get("counted")
                pending = pending or "counted" not in random["count"]
        out["lines"][name] = entry
    out["pending"] = pending
    out["holm"] = {k: None if any(d is None for d in v.values())
                   else cs.holm_from_decisions(v, ALPHA)
                   for k, v in families.items() if len(v) == len(LEVELS)}  # fmt: skip
    rung = out["lines"].get("eli5_cont")
    if rung is not None:  # both decoders, each with its own test; an open one is absent
        out["rung1"] = {}
        for ladder, family, cell in (("state", "dstar_state", "random"),
                                     ("base", "dstar_base", "base_random")):  # fmt: skip
            decided = rung[family]["decisions"]
            if decided is None or rung.get(cell) is None:
                continue
            passed = bool(decided[str(RUNG_LEVEL)]) and rung[cell] <= RANDOM_MAX
            out["rung1"][ladder] = {"passed": passed, "random": rung[cell]}
    return out


def sensitivity_readings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The standard-only repeat of eli5 stack (§13.16): its L, H_s and X on both decoders, as the
    analysis readings ``rows`` name them, under fresh salts."""
    return [{**r, "purpose": "sensitivity",
             "salt": r["salt"].replace("/analysis/", "/sensitivity/")} for r in rows
            if r["reader"] == "dstar" and (r["behavior"], r["mode"]) == ("eli5", "stack")
            and r["role"] in ("L", "H_s", "X") and r["decoder"] in ("state", "base")]  # fmt: skip


def interleaved(requests: Sequence[Any], cells: Sequence[Mapping[str, Any]]) -> list[Any]:
    """The interpreter's due requests across the cells, slot by slot, to share one window."""
    rank = {a.key: (i, j) for j, c in enumerate(cells)
            for i, tries in enumerate(c["attempts"]) for a in tries}  # fmt: skip
    return sorted(requests, key=lambda r: rank.get(r.key, (len(rank), 0)))


def sensitivity(rows: Sequence[Mapping[str, Any]], original: Mapping[str, Any], regions: Any,
                rules: Mapping[str, Any], definition: Any) -> dict[str, Any]:  # fmt: skip
    """The sensitivity reading's record (§13.16), per decoder: eli5's counts and decisions under the
    same gates, and a Holm result with only eli5's decisions replaced in that decoder's family, the
    other five held at the original record's. It never changes the primary family."""
    from . import continuation_stats as cs
    from .follow_regions import ALPHA, LEVELS

    by = {(r["reading"]["reader"], r["reading"]["behavior"], r["reading"]["mode"],
           r["reading"]["role"], r["reading"]["decoder"]): r for r in rows}  # fmt: skip
    never = dict.fromkeys(LEVELS, False)
    accepted = original["format"]["accepted"]["dstar"]
    out: dict[str, Any] = {"pending": False}
    for decoder, family in (("state", "dstar_state"), ("base", "dstar_base")):
        line = _line(by, "dstar", "eli5", "stack", decoder, regions)
        cell = by.get(("dstar", "eli5", "stack", "L", decoder))
        graded = None
        if cell is not None and "counted" in cell["count"]:
            graded = margins(cell["slots"], "eli5", definition, rules)
        eligible = original["lines"]["eli5"][family]["eligible"]
        category = None if graded is None else graded["passed"]
        if line["by_level"] is None or eligible is None or (eligible and accepted
                                                            and category is None):  # fmt: skip
            out["pending"] = True
            out[family] = {"counts": line["counts"], "decisions": None, "holm": None}
            continue
        gates = bool(eligible and accepted and category)
        decided = line["by_level"] if gates else never
        family_decisions = {
            b: {Fraction(a): v for a, v in e[family]["decisions"].items()}
            for b, e in original["lines"].items() if b != "eli5_cont"}  # fmt: skip
        family_decisions["eli5"] = decided
        holm = cs.holm_from_decisions(family_decisions, ALPHA)
        same = sorted(holm["rejected"]) == sorted(original["holm"][family]["rejected"])
        out[family] = {"counts": line["counts"], "margins": graded, "gates_passed": gates,
                       "decisions": {str(a): v for a, v in decided.items()}, "holm": holm,
                       "same_as_primary": same,
                       "statement": None if same else
                       "the decision was not robust to the standard-only repeat"}  # fmt: skip
    return out
