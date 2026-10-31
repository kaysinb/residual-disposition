"""The follow-up's qualification of the reader D* and of unfiltered PD (docs/preregistration.md
§13.5 (a); barrier B2), on the API from recorded packages only.

- **Readings** (``readings``): B2's inventory (``manifests/follow_b2_inventory.json``, frozen
  with B2) resolved at the committed confirmations (``follow_inventory.resolve``): D*'s 100
  qualification readings (A, H and the recipe controls on both decoders, A's random-direction
  and no-patch fidelity branches), unfiltered PD's 26 (A and the recipe controls) and its 12
  validation-only readings of H, always generated. A new package is bound by the build
  session's records (``new_packages``: its lens builds and its pools), never by a name pattern,
  and only the packages these readings name are taken (L, H_s and X are B3's).
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
- **Floors** (``floors``), each pooled once over the two streams (160 slots), as exact fractions:
  D* per behavior and decoder R(A) ≥ 7/10, R(H) ≤ 1/10, R(A) − R(H) ≥ 3/5 and R(recipe) ≥
  7/10 · R(A) (eli5's two recipe controls each), and A's fidelity branches R ≤ 1/10; unfiltered PD
  R(A) ≥ 7/10 and R(recipe) ≥ 7/10 · R(A). A behavior failing on either decoder is ineligible for
  D*. A reading with an unresolved slot leaves everything it enters pending.
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


def new_packages(names: set[str], run: str | None = None,
                 root: Path | None = None) -> dict[str, dict[str, dict]]:  # fmt: skip
    """The build session's new packages named by the readings (``names``, B2's ``"<lineage>
    <role>"``), the session's other packages left out: ``lens`` its builds
    (``follow_inventory.built_packages``), ``pd`` its pools from the pool sessions' records
    (``out/pools/*/result.json``, each complete); a name recorded twice is refused."""
    from . import follow_inventory as fi
    from .paths import runs_root

    run = run or fi.BUILD_SESSION
    pools: dict[str, dict] = {}
    base = (root or runs_root()) / "follow_build" / run / "out" / "pools"
    for f in sorted(base.glob("*/result.json")):
        record = json.loads(f.read_text())
        if record.get("status") != "complete":
            raise ValueError(f"{f.parent.name}: the pool session is not complete")
        for state, v in record["pools"].items():
            name = f"{v['lineage']} {v['role']}"
            if name not in names:
                continue
            if name in pools:
                raise ValueError(f"{name}: its pool recorded twice")
            pools[name] = {"pool": v["run"], "state": state}
    lens = {k: v for k, v in fi.built_packages(run).items() if k in names}
    return {"lens": lens, "pd": pools}


def new_names(readings: Sequence[Mapping[str, Any]]) -> set[str]:
    """The new packages the readings name."""
    return {r["package"]["new"] for r in readings if "new" in r["package"]}


def package_dir(reading: Mapping[str, Any], find_run: Callable[[str], Any],
                new: Mapping[str, Mapping[str, dict]]) -> Path:  # fmt: skip
    """The directory a reading's evidence is read from: its lens package's run or build, or its
    PD pool's run; a new package only through ``new``."""
    from . import follow_inventory as fi

    p = reading["package"]
    kind = KINDS[reading["reader"]]
    if "new" in p and p["new"] not in new[kind]:
        raise ValueError(f"{reading['salt']}: its new package {p['new']} is not recorded")
    if reading["reader"] == "dstar":
        if "new" in p:
            ref = new["lens"][p["new"]]
            return fi.build_dir(ref["build"], ref["state"])
        name = p["run"]
    else:
        name = new["pd"][p["new"]]["pool"] if "new" in p else p["pool"]
    run = find_run(name)
    if run is None:
        raise ValueError(f"{reading['salt']}: its package {name} is not here")
    return Path(run)


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


def answer(cache: Any, tries: Sequence[Any], parse: Callable[[Any], Any]) -> dict[str, Any]:
    """A slot from the cache: ``{"value"}`` (the first answer ``parse`` accepts), ``{"due"}`` (the
    next attempt without an answer) or ``{"refused": True}`` (both refused)."""
    from .judge import MalformedOutput

    for r in tries:
        entry = cache.entry(r.key)
        if entry is None:
            return {"due": r}
        if entry.get("body") != r.body:
            raise ValueError(f"{r.key}: its cached answer is not to its request")
        try:
            return {"value": parse(entry.get("text")), "key": r.key}
        except MalformedOutput:
            continue
    return {"refused": True}


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


def labels(item: Any, judge: Any, cache: Any) -> dict[str, Any]:
    """A hypothesis's labels from the cache: each ``answer``; a strict slot refused because both
    attempts were blocked by the safety filter, with a valid lenient label, is
    ``STRICT_SAFETY_BLOCKED`` (counted as not naming b)."""
    from .continuation_readers import STRICT_SAFETY_BLOCKED, safety_blocked
    from .judge import parse_lenient, parse_strict

    tries = label_attempts(item, judge)
    strict = answer(cache, tries["strict"], parse_strict)
    lenient = answer(cache, tries["lenient"], parse_lenient)
    if strict.get("refused") and "value" in lenient and all(
            safety_blocked(cache.entry(r.key) or {}) for r in tries["strict"]):  # fmt: skip
        strict = {"value": (STRICT_SAFETY_BLOCKED, False), "blocked": True}
    return {"strict": strict, "lenient": lenient, "tries": tries}


def reading_count(slots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A reading's counted hypotheses of its 80 slots (rule C), or why it is pending: every slot
    parsed and both its labels resolved, else pending (a refused slot as well: §12.4.5 leaves it
    pending, never zero)."""
    from .continuation_readers import counted

    if len(slots) != SLOTS:
        return {"pending": f"{len(slots)} slots"}
    reasons = []
    k = 0
    for i, s in enumerate(slots):
        hyp = s["hypothesis"]
        if "value" not in hyp:
            reasons.append(f"slot {i}: hypothesis {'refused' if hyp.get('refused') else 'due'}")
            continue
        lab = s["labels"]
        if "value" not in lab["strict"] or "value" not in lab["lenient"]:
            reasons.append(f"slot {i}: labels unresolved")
            continue
        k += counted(lab["strict"]["value"][0], lab["lenient"]["value"])
    return (
        {"pending": reasons[:5], "unresolved": len(reasons)}
        if reasons
        else {"counted": k, "n": SLOTS}
    )


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
