"""The follow-up's qualification of the reader D* and of unfiltered PD (docs/preregistration.md
§13.5 (a); barrier B2), on the API from recorded packages only.

- **Readings** (``readings``): B2's inventory (``manifests/follow_b2_inventory.json``, frozen
  with B2) resolved at the committed confirmations (``follow_inventory.resolve``): D*'s 100
  qualification readings (A, H and the recipe controls on both decoders, A's random-direction
  and no-patch fidelity branches), unfiltered PD's 26 (A and the recipe controls) and its 12
  validation-only readings of H, always generated. A new package is bound before anything is
  rendered to its entry in the frozen build configuration (its behavior, lineage, role and state)
  and to the build session's records at their pinned sha256 (``new_packages``), never by a name
  pattern; only the packages these readings name are read (L, H_s and X are B3's). The evidence
  of every reading is pinned once, committed, before any request (``pins``): a changed evidence
  never makes new requests for a slot.
- **Evidence** (``evidence``): D*'s rendering of the package through the reading's decoder
  (``lens_render.render``; the no-patch control's ``render_no_patch``); unfiltered PD's registered
  selection with no removal (``follow_pd.evidence``).
- **Requests** (``attempts``): the frozen interpreter (``continuation_readers.interpreter``) with
  D*'s prompt (``lens_render.prompt``; the no-patch prompt for that control) or ``pd-v1``, 80
  slots per reading under its salt, each slot's request and its one format-only retry.
- **Labels** (``labels``): each parsed hypothesis judged by the frozen judge of its format (rule
  C; ``continuation_readers.judge_settings``: v4 for D*, v4-j3 for unfiltered PD), each request
  with its one retry; a strict request and its retry both blocked by the safety filter, with a
  valid lenient label, count as not naming b (§12.4.5), the denominator kept.
- **Admission and histories** (``answer``): every attempt read is the cache's answer to exactly
  its request, admitted at its tier (the interpreter's flex or standard; a judge's as the
  readers admit it, ``labels``); any other reply leaves its slot pending and is never asked
  again. Each slot's attempts are recorded (``history_of``) and every later run checks that the
  answers recorded earlier are still there, unchanged (``history_problems``); an attempt missing
  while a later one is answered is refused, never redrawn.
- **Floors** (``floors``), each pooled once over the two streams (160 slots), as exact fractions:
  D* per behavior and decoder R(A) ≥ 7/10, R(H) ≤ 1/10, R(A) − R(H) ≥ 3/5 and R(recipe) ≥
  7/10 · R(A) (eli5's two recipe controls each), and A's fidelity branches R ≤ 1/10; unfiltered PD
  R(A) ≥ 7/10 and R(recipe) ≥ 7/10 · R(A). A behavior failing on either decoder is ineligible for
  D*. A reading with an unresolved slot leaves everything it enters pending. The registered
  safety blocks are kept beside them, descriptive (``reading_count``'s b, b_yes and interval;
  ``upper_rates`` and the floors at those rates), never selecting a verdict.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

SLOTS = 80
DSTAR_DECODERS = ("state", "base")
FIDELITY = ("random", "no_patch")
KINDS = {"dstar": "lens", "pd_unfiltered": "pd"}


def readings(confirmed: Mapping[str, int], repo: Path | None = None) -> list[dict[str, Any]]:
    """The readings this qualification sends, in B2's order: D*'s qualification readings, then
    unfiltered PD's qualification and validation readings, at the confirmed doses."""
    from . import continuation_rules as cr
    from . import follow_inventory as fi
    from .paths import REPO_ROOT

    b2 = cr.follow_b2()  # the frozen B2, its inventory manifest at its sha256
    manifest = json.loads(((repo or REPO_ROOT) / b2["inventory"]["manifest"]).read_text())
    parts = fi.resolve(manifest, confirmed)
    return [*parts["dstar"]["qualification"], *parts["pd_unfiltered"]["qualification"],
            *parts["pd_unfiltered"]["validation"]]  # fmt: skip


# The build session's records by sha256 (docs/decisions.md, 2026-10-30), under
# runs/follow_build/<session>/; nothing else of the session is read
BUILD_RECORDS = {
    "build": ("out/build/{run}/result.json",
              "733b4aa1e54564ddcd5354bd9365fdb5f085b2d92524fa1e5ebf38a0cbf684c9"),
    "pools/eli5": ("out/pools/{run}-eli5/result.json",
                   "1c8ca893d14e158066e2aa938af4ec876ef5a7ffb318d7411c9fbcc5ed61ef6f"),
    "pools/pun_intended": ("out/pools/{run}-pun_intended/result.json",
                           "41624d5dd762f0904c4f9fe1a690a9a82f1519b9f5f99501c59b0667511c377c"),
}  # fmt: skip
NAMED = ("behavior", "lineage", "role", "state")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_names(readings: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    """The new packages the readings name, by kind (``lens`` for D*, ``pd`` for unfiltered PD)."""
    out: dict[str, set[str]] = {"lens": set(), "pd": set()}
    for r in readings:
        if "new" in r["package"]:
            out[KINDS[r["reader"]]].add(r["package"]["new"])
    return out


def frozen_new(build: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    """The frozen build configuration's new packages by name (``"<lineage> <role>"``): ``lens``
    the session's builds, ``pd`` its pools, each with its behavior, lineage, role and state."""
    lens = {f"{b['lineage']} {b['role']}": b for b in build["sessions"]["b1"]["builds"]}
    return {"lens": lens, "pd": {f"{b['lineage']} {b['role']}": b for b in build["pools"]}}


def new_packages(
    names: Mapping[str, set[str]], run: str | None = None, root: Path | None = None,
    build: Mapping[str, Any] | None = None, find_run: Callable[[str], Any] | None = None,
) -> dict[str, dict[str, dict]]:  # fmt: skip
    """The new packages the readings name (``names``), each bound before anything is rendered to
    its entry in the frozen build configuration (``follow_build.yaml``: behavior, lineage, role
    and state; an entry without a state is refused) and to the build session's records at their
    pinned sha256 (``BUILD_RECORDS``, each complete):

    - a lens package: the session's build of exactly that state, complete with its method check
      passed, its record (``build.json``) naming the same behavior, lineage, role and state, and
      its vector files at the sha256 the session recorded;
    - a PD pool: the pool sessions' record of exactly that state with the same behavior, lineage
      and role, its run completed, a PD pool of that state, and whole
      (``follow_build.pool_problems``: the final PD GPU, the frozen prefills, one whole row per
      prefill and framing).

    The session's other packages are never read."""
    from . import continuation_rules as cr
    from . import follow_build as B
    from . import follow_inventory as fi
    from . import pipeline as P
    from .continuation_kstar import find_run as found_run
    from .follow_pd import pool_rows
    from .paths import runs_root

    run, find_run = run or fi.BUILD_SESSION, find_run or found_run
    base = (root or runs_root()) / "follow_build" / run
    records = {}
    for k, (rel, sha) in BUILD_RECORDS.items():
        path = base / rel.format(run=run)
        if not path.is_file() or _sha256(path) != sha:
            raise ValueError(f"{rel.format(run=run)}: not the session's record at its sha256")
        records[k] = json.loads(path.read_text())
        if records[k].get("status") != "complete":
            raise ValueError(f"{rel.format(run=run)}: the session is not complete")
    frozen = frozen_new(build or cr.follow_build())
    out: dict[str, dict[str, dict]] = {"lens": {}, "pd": {}}
    for name in sorted(names.get("lens", ())):
        want = frozen["lens"].get(name) or {}
        if "state" not in want:
            raise ValueError(f"{name}: not a lens build of the frozen plan with its state")
        state, d = want["state"], base / "out" / "build" / run
        made = records["build"]["builds"].get(state) or {}
        if not made.get("complete") or not (made.get("check") or {}).get("passed"):
            raise ValueError(f"{name}: the session's build of {state} is not complete and checked")
        record = json.loads((d / "builds" / state / "build.json").read_text())
        if {k: record.get(k) for k in NAMED} != {k: want.get(k) for k in NAMED}:
            raise ValueError(f"{name}: its build record is not the frozen plan's {state}")
        files = {k: v for k, v in records["build"]["files"].items()
                 if k.startswith(f"builds/{state}/")}  # fmt: skip
        if not files or any(_sha256(d / f) != sha for f, sha in files.items()):
            raise ValueError(f"{name}: its files are not at the sha256 the session recorded")
        out["lens"][name] = {"build": run, "state": state, "files": files}
    pools = {**records["pools/eli5"]["pools"], **records["pools/pun_intended"]["pools"]}
    for name in sorted(names.get("pd", ())):
        want = frozen["pd"].get(name) or {}
        if "state" not in want:
            raise ValueError(f"{name}: not a pool of the frozen plan with its state")
        state, made = want["state"], pools.get(want["state"]) or {}
        if {k: made.get(k) for k in NAMED[:3]} != {k: want.get(k) for k in NAMED[:3]}:
            raise ValueError(f"{name}: the sessions' record of {state} is not the frozen plan's")
        out["pd"][name] = {"pool": made["run"], "state": state}
        pool_problems(made["run"], state, find_run, P, B, pool_rows)
    return out


def pool_problems(name: str, state: str, find_run: Callable[[str], Any], P: Any, B: Any,
                  rows: Callable[[Path], Any]) -> None:  # fmt: skip
    """Refuse a PD pool that is not a completed pool of ``state``, whole."""
    import yaml

    run = find_run(name)
    if run is None:
        raise ValueError(f"{name}: not here")
    run = Path(run)
    meta = json.loads((run / "meta.json").read_text())
    config = yaml.safe_load((run / "config.yaml").read_text()) or {}
    if meta.get("status") != "completed" or config.get("step") != "tool_pd" or (
            config.get("params") or {}).get("state") != state:  # fmt: skip
        raise ValueError(f"{name}: not a completed PD pool of {state}")
    found = B.pool_problems(json.loads((run / "summary.json").read_text()), rows(run),
                            prefills=P.FROZEN_PD["prefills"],
                            prefills_sha256=P.FROZEN_PD["prefills_sha256"],
                            configs=P.PD_CONFIGS, gpu=P.pd_gpu(run),
                            final_gpu=P.FINAL_PD_GPU)  # fmt: skip
    if found:
        raise ValueError(f"{name}: " + "; ".join(found))


def package_dir(reading: Mapping[str, Any], find_run: Callable[[str], Any],
                new: Mapping[str, Mapping[str, dict]]) -> Path:  # fmt: skip
    """The directory a reading's evidence is read from: a new package only through ``new``
    (``new_packages``, bound and checked); an existing one its run, completed, a PD pool of the
    inventory's state."""
    import yaml

    from . import follow_inventory as fi

    p, kind = reading["package"], KINDS[reading["reader"]]
    if "new" in p:
        if p["new"] not in new[kind]:
            raise ValueError(f"{reading['salt']}: its new package {p['new']} is not bound")
        ref = new[kind][p["new"]]
        if kind == "lens":
            return fi.build_dir(ref["build"], ref["state"])
        name = ref["pool"]
    else:
        name = p["run"] if kind == "lens" else p["pool"]
    run = find_run(name)
    if run is None:
        raise ValueError(f"{reading['salt']}: its package {name} is not here")
    run = Path(run)
    if "new" not in p:
        meta = json.loads((run / "meta.json").read_text())
        state = ((yaml.safe_load((run / "config.yaml").read_text()) or {}).get("params")
                 or {}).get("state")  # fmt: skip
        if meta.get("status") != "completed" or (kind == "pd" and state != p["state"]):
            raise ValueError(f"{reading['salt']}: {name} is not the completed package it names")
    return run


def evidence(reading: Mapping[str, Any], directory: Path) -> str:
    """A reading's evidence text (see the module)."""
    from . import follow_pd
    from . import lens_render as R
    from .paths import runs_root

    if reading["reader"] == "pd_unfiltered":
        return follow_pd.evidence(follow_pd.pool_rows(directory))
    if reading["decoder"] == "no_patch":
        return R.render_no_patch(directory, runs_root())[0]
    return R.render(directory, reading["decoder"], runs_root())[0]


def prompt_and_settings(reading: Mapping[str, Any]) -> tuple[Any, Any]:
    """A reading's frozen prompt and interpreter settings."""
    from . import continuation_readers as readers
    from . import lens_render as R

    prompt, settings = readers.interpreter(KINDS[reading["reader"]])
    if reading["reader"] == "dstar":
        prompt = R.no_patch_prompt() if reading["decoder"] == "no_patch" else R.prompt()
    return prompt, settings


def attempts(reading: Mapping[str, Any], text: str) -> list[list[Any]]:
    """Each slot's request and its one format-only retry, under the reading's salt."""
    from .summarizer import summary_requests

    prompt, settings = prompt_and_settings(reading)
    first = summary_requests(prompt, text, settings, salt=reading["salt"])
    retry = summary_requests(prompt, text, settings, salt=reading["salt"], retry=True)
    if len(first) != SLOTS or int(reading["slots"]) != SLOTS:
        raise ValueError(f"{reading['salt']}: not {SLOTS} slots")
    return [[a, b] for a, b in zip(first, retry, strict=True)]


INTERPRETER_TIERS = ("flex", "standard")  # flex with its fallback to the standard tier


def interpreter_tier(step: Mapping[str, Any]) -> str | None:
    """The tier an interpreter attempt is read at: the one recorded with its answer, flex or
    standard; None for any other (a tier-less or batch reply), which stays pending."""
    return step["tier"] if step["tier"] in INTERPRETER_TIERS else None


def judge_pins(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """The historical judge replies the committed judge histories pin (§12.4.5, 2026-10-27:
    ``manifests/stage3_judge_history_*.json``), by key; a key pinned twice differently is
    refused."""
    from .manifests import committed_json
    from .paths import REPO_ROOT

    out: dict[str, dict[str, Any]] = {}
    for path in sorted(((root or REPO_ROOT) / "manifests").glob("stage3_judge_history_*.json")):
        for key, e in committed_json(path, "a judge history", repo=root)["entries"].items():
            if key in out and (out[key]["entry_sha256"], out[key]["tier"]) != (
                    e["entry_sha256"], e["tier"]):  # fmt: skip
                raise ValueError(f"{key}: pinned twice differently")
            out[key] = {"entry_sha256": e["entry_sha256"], "tier": e["tier"]}
    return out


def answer(
    cache: Any, tries: Sequence[Any], parse: Callable[[Any], Any],
    admit: Callable[[Mapping[str, Any]], str | None], where: str = "",
) -> dict[str, Any]:  # fmt: skip
    """A slot from the cache, with its history (each attempt's key, entry sha256, recorded and
    admitted tier, and whether the parser took it): ``{"value", "key"}`` the first answer
    ``parse`` accepts; ``{"due"}`` the next attempt without an answer; ``{"refused": True}``
    both refused; ``{"unadmitted"}`` an answer ``admit`` does not admit at its tier, which stays
    pending and is never asked again. Each answer must be to exactly its request. An attempt
    missing while a later one is answered is a lost history: refused, never redrawn."""
    from .judge import MalformedOutput, cache_entry_sha256

    history: list[dict[str, Any]] = []
    for n, r in enumerate(tries):
        entry = cache.entry(r.key)
        if entry is None:
            if any(cache.entry(x.key) is not None for x in tries[n + 1 :]):
                raise ValueError(f"{where}: attempt {n + 1} is missing, a later one answered: "
                                 "its history is lost")  # fmt: skip
            return {"due": r, "history": history}
        if entry.get("body") != r.body:
            raise ValueError(f"{r.key}: its cached answer is not to its request")
        step = {"key": r.key, "sha256": cache_entry_sha256(entry),
                "tier": entry.get("service_tier")}  # fmt: skip
        tier = admit(step)
        if tier is None:
            return {"unadmitted": f"{where}: attempt {n + 1} at tier {step['tier']!r}",
                    "history": [*history, {**step, "admitted": None}]}  # fmt: skip
        try:
            value = parse(entry.get("text"))
        except MalformedOutput:
            history.append({**step, "admitted": tier, "parsed": False})
            continue
        history.append({**step, "admitted": tier, "parsed": True})
        return {"value": value, "key": r.key, "history": history}
    return {"refused": True, "history": history}


def label_attempts(item: Any, judge: Any) -> dict[str, list[Any]]:
    """A hypothesis's strict and lenient requests, each with its one retry
    (``request_key(<rubric version>#retry=1, body)``, as the readers' judge retries)."""
    from .judge import ChatRequest, request_key, stage_requests
    from .summarizer import RETRY

    (staged,) = stage_requests([item], judge)
    out = {}
    for kind in ("strict", "lenient"):
        r = getattr(staged, kind)
        retry = ChatRequest(request_key(f"{r.rubric_version}{RETRY}", r.body), r.kind,
                            r.rubric_version, r.body)  # fmt: skip
        out[kind] = [r, retry]
    return out


def labels(item: Any, judge: Any, cache: Any,
           pinned: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:  # fmt: skip
    """A hypothesis's labels from the cache, each ``answer`` with every attempt admitted as the
    readers admit it (``continuation_readers.admitted_tier``: the standard tier, or a historical
    reply a committed judge history ``pinned`` holds; any other reply pending). A strict slot
    refused because its request and retry were both blocked by the safety filter, each so
    admitted (``continuation_readers.blocked_slot``), with a valid lenient label, is
    ``STRICT_SAFETY_BLOCKED`` (counted as not naming b, the denominator kept)."""
    from .continuation_readers import STRICT_SAFETY_BLOCKED, admitted_tier, blocked_slot
    from .judge import parse_lenient, parse_strict

    def admit(step: Mapping[str, Any]) -> str | None:
        return admitted_tier(step, pinned)

    tries = label_attempts(item, judge)
    where = item.response_id
    strict = answer(cache, tries["strict"], parse_strict, admit, f"{where} (strict)")
    lenient = answer(cache, tries["lenient"], parse_lenient, admit, f"{where} (lenient)")
    if strict.get("refused") and "value" in lenient:
        blocked = blocked_slot(cache, [r.key for r in tries["strict"]], tries["strict"][0].body,
                               f"{where} (strict)", pinned)  # fmt: skip
        if blocked is not None:
            strict = {"value": (STRICT_SAFETY_BLOCKED, False), "blocked": True,
                      "history": blocked}  # fmt: skip
    return {"strict": strict, "lenient": lenient, "tries": tries}


def _open(slot: Mapping[str, Any]) -> str:
    return next(k for k in ("due", "refused", "unadmitted") if k in slot)


def reading_count(slots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A reading's counted hypotheses of its 80 slots (rule C), or why it is pending: every slot
    parsed and both its labels resolved, else pending (a refused slot as well: §12.4.5 leaves it
    pending, never zero; an unadmitted reply too). With it the registered safety blocks
    (§12.4.5, 2026-10-26), descriptive: the hypotheses blocked b, those with a lenient yes
    b_yes, and the interval R ∈ [c/80, (c + b_yes)/80]."""
    from .continuation_readers import counted

    if len(slots) != SLOTS:
        return {"pending": f"{len(slots)} slots"}
    reasons = []
    k = blocked = blocked_yes = 0
    for i, s in enumerate(slots):
        hyp = s["hypothesis"]
        if "value" not in hyp:
            reasons.append(f"slot {i}: hypothesis {_open(hyp)}")
            continue
        lab = s["labels"]
        if "value" not in lab["strict"] or "value" not in lab["lenient"]:
            which = [f"{x} {_open(lab[x])}" for x in ("strict", "lenient")
                     if "value" not in lab[x]]  # fmt: skip
            reasons.append(f"slot {i}: {', '.join(which)}")
            continue
        k += counted(lab["strict"]["value"][0], lab["lenient"]["value"])
        if lab["strict"].get("blocked"):
            blocked += 1
            blocked_yes += bool(lab["lenient"]["value"])
    if reasons:
        return {"pending": reasons[:5], "unresolved": len(reasons)}
    interval = [str(Fraction(k, SLOTS)), str(Fraction(k + blocked_yes, SLOTS))]
    return {"counted": k, "n": SLOTS, "blocked": blocked, "blocked_lenient_yes": blocked_yes,
            "interval": interval}  # fmt: skip


def history_of(slot: Mapping[str, Any]) -> dict[str, Any]:
    """A slot's attempts, by stage: the interpreter's, the strict and the lenient judge's."""
    out = {"interpreter": slot["hypothesis"]["history"]}
    if slot["labels"] is not None:
        out |= {k: slot["labels"][k]["history"] for k in ("strict", "lenient")}
    return out


def history_problems(previous: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
                     caches: Mapping[str, Any]) -> list[str]:  # fmt: skip
    """Why the caches no longer hold what earlier runs recorded (``previous``: slot → stage →
    attempts): every attempt answered then must still be answered, by the same entry (its
    sha256). A lost or changed answer is never redrawn."""
    from .judge import cache_entry_sha256

    out = []
    for slot, stages in previous.items():
        for stage, steps in stages.items():
            cache = caches["interpreter" if stage == "interpreter" else "judge"]
            for step in steps:
                entry = cache.entry(step["key"])
                if entry is None or cache_entry_sha256(entry) != step["sha256"]:
                    out.append(f"{slot} ({stage}): {step['key']} is "
                               f"{'missing' if entry is None else 'changed'}")  # fmt: skip
    return out


def pins(cells: Sequence[Mapping[str, Any]], root: Path) -> dict[str, Any]:
    """Every reading's pin: its package directory (under the runs root), its evidence's sha256
    and the sha256 of its slots' keys, by salt."""
    import hashlib

    out = {}
    for c in cells:
        keys = [a.key for tries in c["attempts"] for a in tries]
        out[c["reading"]["salt"]] = {
            "directory": str(Path(c["directory"]).resolve().relative_to(Path(root).resolve())),
            "evidence_sha256": c["evidence_sha256"],
            "keys_sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest()}  # fmt: skip
    return out


def pin_problems(pinned: Mapping[str, Any], now: Mapping[str, Any]) -> list[str]:
    """Why the readings are not the pinned ones: the same salts, each its pin exactly."""
    out = [f"{s}: not pinned" for s in sorted(set(now) - set(pinned))]
    out += [f"{s}: pinned, not planned" for s in sorted(set(pinned) - set(now))]
    out += [f"{s}: {k} changed" for s in sorted(set(now) & set(pinned))
            for k in sorted(now[s]) if now[s][k] != pinned[s].get(k)]  # fmt: skip
    return out


def pooled(rows: Sequence[Mapping[str, Any]]) -> dict[tuple, Any]:
    """Each pool's rate over its two streams (160 slots), by (reader, behavior, role, decoder):
    an exact fraction, or None while a reading of it is pending."""
    groups: dict[tuple, list] = {}
    for r in rows:
        key = (r["reading"]["reader"], r["reading"]["behavior"], r["reading"]["role"],
               r["reading"]["decoder"])  # fmt: skip
        groups.setdefault(key, []).append(r)
    out: dict[tuple, Any] = {}
    for key, rs in groups.items():
        streams = sorted(r["reading"]["stream"] for r in rs)
        if streams != [1, 2]:
            raise ValueError(f"{key}: streams {streams}, not both pooled once")
        counts = [r["count"] for r in rs]
        out[key] = (
            None
            if any("counted" not in c for c in counts)
            else Fraction(sum(c["counted"] for c in counts), sum(c["n"] for c in counts))
        )
    return out


def upper_rates(rows: Sequence[Mapping[str, Any]]) -> dict[tuple, Any]:
    """Each pool's rate with its safety-blocked hypotheses that have a lenient yes counted as
    naming b: the upper end of the registered interval (§12.4.5, 2026-10-26), descriptive."""
    up = [{**r, "count": {**r["count"], "counted": r["count"]["counted"]
                          + r["count"].get("blocked_lenient_yes", 0)}}
          if "counted" in r["count"] else r for r in rows]  # fmt: skip
    return pooled(up)


def floors(rates: Mapping[tuple, Any], rules: Mapping[str, Any]) -> dict[str, Any]:
    """The floors per lineage (see the module): a behavior's, or eli5's two (``eli5`` with its
    stack recipe control, ``eli5_cont`` with its continued one, sharing A and H): D* per decoder,
    its fidelity branches and its eligibility, and unfiltered PD; each verdict ``True``,
    ``False`` or ``None`` while pending."""
    q, pd = rules["readers"]["qualification"], rules["readers"]["pd_qualification"]
    f = {k: Fraction(v) for k, v in q["floors"].items()}
    fidelity_max = Fraction(q["fidelity_max"])

    def at(reader: str, behavior: str, role: str, decoder: str | None) -> Any:
        return rates.get((reader, behavior, role, decoder))

    def all_of(values: Sequence[Any]) -> Any:
        return None if any(v is None for v in values) else all(values)

    out: dict[str, Any] = {}
    for behavior in sorted({k[1] for k in rates}):
        fid = [at("dstar", behavior, "A", branch) for branch in FIDELITY]
        fidelity = None if any(v is None for v in fid) else all(v <= fidelity_max for v in fid)
        recipes = sorted({k[2] for k in rates if k[1] == behavior and k[2].startswith("recipe")})
        for recipe in recipes:
            dstar: dict[str, Any] = {}
            for decoder in DSTAR_DECODERS:
                a, h = at("dstar", behavior, "A", decoder), at("dstar", behavior, "H", decoder)
                r = at("dstar", behavior, recipe, decoder)
                dstar[decoder] = None if None in (a, h, r) else (
                    a >= f["active_min"] and h <= f["sham_max"] and a - h >= f["gap_min"]
                    and r >= f["recipe_ratio"] * a)  # fmt: skip
            a_pd, r_pd = (at("pd_unfiltered", behavior, "A", None),
                          at("pd_unfiltered", behavior, recipe, None))  # fmt: skip
            pd_ok = (
                None
                if None in (a_pd, r_pd)
                else (
                    a_pd >= Fraction(pd["active_min"])
                    and r_pd >= Fraction(pd["recipe_ratio"]) * a_pd
                )
            )
            name = behavior + ("_cont" if recipe == "recipe_cont" else "")
            out[name] = {"dstar": {**dstar, "fidelity": fidelity,
                                   "eligible": all_of([*dstar.values(), fidelity])},
                         "pd_unfiltered": pd_ok}  # fmt: skip
    return out
