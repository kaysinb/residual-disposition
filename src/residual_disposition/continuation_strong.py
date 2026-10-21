"""The strong branch (preregistration §12.4.10, §12.5, §12.7, §12.8; the amendment of 2026-10-21;
C16-2 items B and D), on CPU: its frozen inputs, the screening and the one validation recomputed
from their runs, their manifests, the pointwise bounds, a strong candidate's p₁ and the DPO
fallback's gate.

- **Frozen inputs** (``frozen``, ``branch``): ``configs/continuation/strong.yaml``, the suites by
  sha256 and inventory, the detectors frozen with the text, each branch's lineage, candidates,
  suites and α. A suite on disk (``load_suite``) is accepted only at its frozen hash.
- **Counting** (``pd_counts``, ``adl_counts``, ``prompt_counts``). Raw outward hits of the
  behavior's detector, anywhere in a completion or response, before any PD cleaning. Records that
  are not exactly a cell's inventory make it technically invalid (``Incomplete``), never zero.
- **Screening** (``screening_record``): the frozen candidates in order, each eligible with zero
  hits in each framing of its full PD pool and in both ADL-corpus framings, early and long; it
  stops at the first eligible. Its record names the runs it read and is recomputed from them.
- **Validation** (``validation_record``): one attempt on the branch's own suites; it passes with
  zero hits on the validation PD (a prefix counts once, a hit in either framing), ADL early and
  long in both framings, the observer (L; H_s descriptive) and OOD.
- **Manifests**: ``manifests/strong_<lineage>.json`` (the screening, committed before any
  validation cell) and ``manifests/strong_validation_<lineage>.json`` (committed before any
  follow-up), each written once and read only committed, recomputed from its runs.
- **Bounds** (``bound``): zero of n gives 1 − α^{1/n}, pointwise, for the random-sample suites
  only. **p₁** (``p1``): min(1, methods · P{Bin(600, 0.02) ≤ h}).
- **The fallback's gate** (``fallback_gate``): closed (``fallback_gate_closed``, §12.8) once a
  continued candidate validates, or once the continued branch closes unvalidated and the DPO
  lineage has no margin-passing checkpoint; open when it has one; pending otherwise.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

PD_FRAMINGS = ("chat", "raw")
ADL_FRAMINGS = ("chat", "raw")
LONG = 32
OBSERVER_N = 600
P1_RATE = Fraction(2, 100)
CONTINUED, DPO, PANEL = "stage5_continued", "stage5_dpo", "stage3"
SCREENED = "eligible"
NO_DOSE = "no qualifying dose under the registered search"
METADATA = ("screening_run", "validation_run", "sha256", "committed_at")


class Incomplete(ValueError):
    """A cell whose records are not exactly its suite's inventory: technically invalid, never a
    zero count (§12.4.10)."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _p():
    from . import pipeline

    return pipeline


# ------------------------------------------------------------------------- frozen inputs


def frozen(path: Path | None = None) -> dict[str, Any]:
    """``configs/continuation/strong.yaml``, as frozen with the amendment's text."""
    import yaml

    from .paths import CONFIG_ROOT

    data = yaml.safe_load((path or CONFIG_ROOT / "continuation" / "strong.yaml").read_text())
    for kind in ("pd", "adl", "observer"):
        if set(data["suites"][kind]) != set(data["inventories"][kind]):
            raise ValueError(f"strong.yaml: the {kind} suites and inventories disagree")
    return data


def branch(name: str, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A branch of ``strong.yaml`` with its suites' frozen hashes (``suite_hashes``)."""
    data = data or frozen()
    if name not in data["branches"]:
        raise ValueError(f"no strong branch {name!r}: {sorted(data['branches'])}")
    out = dict(data["branches"][name])
    out["name"] = name
    out["suite_hashes"] = {k: data["suites"][k][v] for k, v in out["suites"].items()}
    return out


def suite_dir(root: Path, kind: str, name: str) -> Path:
    return Path(root) / "strong_suites" / f"{kind}-{name}"


def load_suite(root: Path, kind: str, name: str, data: Mapping[str, Any] | None = None):
    """The saved suite ``<root>/strong_suites/<kind>-<name>`` (``make_strong_suites.py``), at its
    frozen sha256 and inventory, else refused."""
    from .strong_suites import Suite

    data = data or frozen()
    d = suite_dir(root, kind, name)
    records = json.loads((d / "records.json").read_text())
    manifest = json.loads((d / "manifest.json").read_text())
    provenance = {k: v for k, v in manifest.items() if k not in ("kind", "name", "records")}
    suite = Suite(kind, name, records, provenance)
    want, inventory = data["suites"][kind][name], data["inventories"][kind][name]
    if suite.sha256 != want or _sha256(suite.inventory()) != inventory:
        raise ValueError(f"{d}: not the frozen {kind} suite {name} ({want[:12]})")
    return suite


def adl_lengths(suite: Any) -> dict[str, int]:
    """An ADL suite's records' prefix lengths (key → length)."""
    return {k: len(r["tokens"]) for k, r in suite.records.items()}


# ----------------------------------------------------------------------------- counting


def pd_counts(
    rows: Iterable[Mapping[str, Any]], keys: Iterable[str], hit: Callable[[str], bool]
) -> dict[str, Any]:
    """Hits of a PD cell's rows (``config``, ``prefill``, ``completion``) over the prefills
    ``keys`` in both framings: per framing, and per prefix (a hit in either framing). Refused
    (``Incomplete``) unless the rows are exactly one per framing and prefix."""
    want = set(keys)
    seen: dict[str, set[str]] = {f: set() for f in PD_FRAMINGS}
    hits: dict[str, set[str]] = {f: set() for f in PD_FRAMINGS}
    for row in rows:
        framing, key = row["config"], row["prefill"]
        if framing not in seen or key not in want or key in seen[framing]:
            raise Incomplete(f"PD completion {framing}/{key} is off the inventory or repeats")
        seen[framing].add(key)
        if hit(row["completion"]):
            hits[framing].add(key)
    missing = {f: len(want) - len(s) for f, s in seen.items() if len(s) != len(want)}
    if missing:
        raise Incomplete(f"PD completions missing {missing}")
    n = len(want)
    return {"prefixes": n, "hits": len(hits["chat"] | hits["raw"]),
            "by_framing": {f: {"n": n, "hits": len(hits[f])} for f in PD_FRAMINGS}}  # fmt: skip


def adl_counts(
    rows: Iterable[Mapping[str, Any]], lengths: Mapping[str, int], hit: Callable[[str], bool]
) -> dict[str, Any]:
    """Hits of an ADL cell's rows (``framing``, ``key``, ``text``) over the records of
    ``lengths`` (key → prefix length) in both framings: early (the lengths below 32, pooled) and
    long, per framing. Refused (``Incomplete``) unless exactly one row per framing and record."""
    seen: dict[str, set[str]] = {f: set() for f in ADL_FRAMINGS}
    out = {f: {"early": {"n": 0, "hits": 0}, "long": {"n": 0, "hits": 0}} for f in ADL_FRAMINGS}
    for row in rows:
        framing, key = row["framing"], row["key"]
        if framing not in seen or key not in lengths or key in seen[framing]:
            raise Incomplete(f"ADL continuation {framing}/{key} is off the inventory or repeats")
        seen[framing].add(key)
        part = out[framing]["long" if lengths[key] == LONG else "early"]
        part["n"] += 1
        part["hits"] += bool(hit(row["text"]))
    missing = {f: len(lengths) - len(s) for f, s in seen.items() if len(s) != len(lengths)}
    if missing:
        raise Incomplete(f"ADL continuations missing {missing}")
    return {"by_framing": out}


def prompt_counts(
    rows: Iterable[Mapping[str, Any]], keys: Iterable[str], hit: Callable[[str], bool]
) -> dict[str, Any]:
    """Hits of one state's responses (``prompt``, ``text``), exactly one per prompt of ``keys``;
    refused (``Incomplete``) otherwise."""
    want, seen, hits = set(keys), set(), 0
    for row in rows:
        if row["prompt"] not in want or row["prompt"] in seen:
            raise Incomplete(f"the response to {row['prompt']} is off the inventory or repeats")
        seen.add(row["prompt"])
        hits += bool(hit(row["text"]))
    if seen != want:
        raise Incomplete(f"{len(want - seen)} responses missing")
    return {"n": len(want), "hits": hits}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def hit_of(detector: Any) -> Callable[[str], bool]:
    """The strong branch's count: the behavior's detector searched anywhere in a completion or
    response, multiline (``path_b.trace_detector``; for a ``regex_count`` detector its own
    count)."""
    from .path_b import trace_detector

    return trace_detector(detector)


def check_detector(behavior: str, detector: Any, data: Mapping[str, Any] | None = None) -> None:
    """Refused unless ``detector`` is the version frozen with the text (an accepted revision
    after the freeze invalidates the affected validation, §12.4.10)."""
    want = (data or frozen())["detectors"].get(behavior)
    if detector.version != want:
        raise ValueError(f"{behavior}'s detector is {detector.version}, not the frozen {want}")


def screen_eligible(pd: Mapping[str, Any], adl: Mapping[str, Mapping[str, int]]) -> bool:
    """Eligible: zero hits in each PD framing and in both ADL framings, early and long
    (``pd``: ``continuation_stage5.pd_corpus``'s shape; ``adl``: ``pipeline.corpus_framings``)."""
    return all(pd["by_config"][f]["hits"] == 0 for f in PD_FRAMINGS) and all(
        adl[f]["pooled_hits"] == 0 and adl[f]["long_hits"] == 0 for f in ADL_FRAMINGS
    )


# ---------------------------------------------------------------------------- screening


def adl_corpus_framings(run: Path, latent: str, detector: Any) -> dict[str, dict[str, int]]:
    """A completed ADL-corpus cell of ``latent`` at the frozen settings, its saved continuations
    recounted with ``detector`` (a revision rescores, never redraws), per framing; refused
    (``Incomplete``) when its continuations are not exactly the frozen counts."""
    import yaml

    from .adl_corpus import corpus_check

    p = _p()
    config = yaml.safe_load((run / "config.yaml").read_text()) or {}
    params = config.get("params") or {}
    per_call = p._ADL_PER_CALL["corpus"]
    settings = p.content_hash({k: v for k, v in params.items() if k not in per_call})
    if config.get("step") != "adl_corpus" or params.get("state") != latent:
        raise ValueError(f"{run.name}: not the ADL-corpus check of {latent}")
    if settings != p.FROZEN_ADL["corpus"]:
        raise ValueError(f"{run.name}: not the frozen ADL-corpus settings")
    lengths, (long_length, long_count) = list(params["lengths"]), params["long"]
    texts: dict[str, dict[int, list[str]]] = {f: {} for f in params["framings"]}
    for row in read_rows(run / "continuations.jsonl.gz"):
        texts[row["framing"]].setdefault(int(row["length"]), []).append(row["text"])
    want = {**dict.fromkeys(lengths, int(params["per_length"])), int(long_length): int(long_count)}
    if any({n: len(v) for n, v in t.items()} != want for t in texts.values()):
        raise Incomplete(f"{run.name}: its continuations are not the frozen counts")
    counts = corpus_check(texts, hit_of(detector), lengths=lengths, long_length=long_length)
    return p.corpus_framings(counts)


def screening_record(inputs: Mapping[str, Any], find_run: Any, detector: Any) -> dict[str, Any]:
    """The screening of a branch's frozen candidates recomputed from the runs ``inputs`` names
    (``candidates``: each screened one with its full PD pool and ADL-corpus runs, in order). The
    candidates screened must be the frozen list's first ones, each but the last ineligible, and
    the last eligible or the list exhausted. Returns the record: each candidate's counts and
    eligibility, ``chosen`` (or None) and the outcome."""
    from . import continuation_stage5 as S5

    b = branch(inputs["branch"])
    if inputs.get("behavior") != b["behavior"] or inputs.get("lineage") != b["lineage"]:
        raise ValueError(f"the screening's inputs are not branch {b['name']}'s")
    check_detector(b["behavior"], detector)
    if inputs.get("detector_version") != detector.version:
        raise ValueError(f"screened with detector {inputs.get('detector_version')}, the "
                         f"behavior's is {detector.version}")  # fmt: skip
    frozen_list = list(b["candidates"])
    screened = list(inputs.get("candidates") or [])
    if not screened or len(screened) > len(frozen_list):
        raise ValueError("the screening names no candidate, or more than the frozen ones")
    rows, chosen = [], None
    for i, c in enumerate(screened):
        want = frozen_list[i]
        if {k: c.get(k) for k in ("dose", "latent", "sham_s")} != want:
            raise ValueError(f"candidate {i + 1} is not the frozen {want}")
        if chosen is not None:
            raise ValueError("the screening went on after an eligible candidate")
        pd_run, adl_run = find_run(c.get("pd")), find_run(c.get("adl"))
        if pd_run is None or adl_run is None:
            raise ValueError(f"the PD pool or ADL-corpus run of {want['latent']} is not here")
        pd = S5.pd_corpus(pd_run, want["latent"], hit_of(detector))
        if pd is None:
            raise Incomplete(f"{pd_run.name}: the PD pool is technically invalid")
        adl = adl_corpus_framings(adl_run, want["latent"], detector)
        ok = screen_eligible(pd, adl)
        rows.append({**want, "pd": pd, "adl": adl, "eligible": ok})
        if ok:
            chosen = dict(want)
    if chosen is None and len(screened) != len(frozen_list):
        raise ValueError("the screening stopped before an eligible candidate")
    return {"branch": b["name"], "lineage": b["lineage"], "behavior": b["behavior"],
            "detector_version": detector.version, "screened": rows, "chosen": chosen,
            "outcome": SCREENED if chosen else NO_DOSE}  # fmt: skip


# ---------------------------------------------------------------------------- validation


PARTS = {  # a validation's part: its cell's step and the state it reads
    "pd": ("strong_pd", "latent"),
    "adl": ("strong_adl", "latent"),
    "observer_L": ("strong_rates", "latent"),
    "observer_H_s": ("strong_rates", "sham_s"),
    "ood": ("strong_ood", "latent"),
}


def identity(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The one validation's identity, persisted before any generation: the branch, lineage,
    candidate, the committed screening manifest, the suites, the detector and the streams."""
    b = branch(inputs["branch"])
    lineage = b["lineage"]
    return {
        "branch": b["name"], "lineage": lineage, "behavior": b["behavior"],
        "candidate": dict(inputs["candidate"]), "screening": dict(inputs["screening"]),
        "suites": {k: {"name": b["suites"][k], "sha256": b["suite_hashes"][k]}
                   for k in ("pd", "adl", "observer")},
        "detector_version": inputs["detector_version"],
        "streams": {"observer": f"strong:{lineage}:observer", "ood": f"strong:{lineage}:ood"},
    }  # fmt: skip


def _part(find_run: Any, name: str | None, step: str, state: str, ident: str) -> Path:
    """A completed part cell of the validation ``ident`` (its identity's sha256) on ``state``."""
    import yaml

    run = find_run(name) if name else None
    if run is None:
        raise ValueError(f"the validation's {step} run {name} is not here")
    meta = json.loads((run / "meta.json").read_text())
    params = (yaml.safe_load((run / "config.yaml").read_text()) or {}).get("params") or {}
    if meta.get("status") != "completed":
        raise Incomplete(f"{name}: not completed")
    step_of = (yaml.safe_load((run / "config.yaml").read_text()) or {}).get("step")
    if step_of != step or params.get("state") != state or params.get("identity") != ident:
        raise ValueError(f"{name}: not this validation's {step} of {state}")
    return run


def validation_record(
    inputs: Mapping[str, Any], find_run: Any, detector: Any, data_root: Path
) -> dict[str, Any]:
    """The one validation recomputed from the runs ``inputs`` names (``identity``, each part's
    run): the suites at their frozen hashes from ``data_root``, every part over its exact
    inventory, the counts, the verdict, the pointwise bounds and p₁ at the observer's count."""
    from . import pipeline as p

    b = branch(inputs["branch"])
    check_detector(b["behavior"], detector)
    ident = identity(inputs)
    ident_sha = _sha256(ident)
    found = find_run(inputs.get("identity"))
    if found is None or json.loads((found / "strong_validation_identity.json").read_text()) \
            != ident:  # fmt: skip
        raise ValueError("the validation's identity run is not this identity")
    c, runs = ident["candidate"], inputs.get("runs") or {}
    hit = hit_of(detector)
    suites = {k: load_suite(data_root, k, v["name"]) for k, v in ident["suites"].items()}
    read = {part: _part(find_run, runs.get(part), step, c[role], ident_sha)
            for part, (step, role) in PARTS.items()}  # fmt: skip
    pd = pd_counts(read_rows(read["pd"] / "pd_completions.jsonl.gz"), suites["pd"].records, hit)
    adl = adl_counts(read_rows(read["adl"] / "continuations.jsonl.gz"),
                     adl_lengths(suites["adl"]), hit)  # fmt: skip
    prompts = suites["observer"].records
    observer = {role: prompt_counts(read_rows(read[f"observer_{role}"] / "responses.jsonl.gz"),
                                    prompts, hit) for role in ("L", "H_s")}  # fmt: skip
    ood = prompt_counts(read_rows(read["ood"] / "responses.jsonl.gz"),
                        [q.key for q in p.ood_requests()], hit)  # fmt: skip
    counts = {"pd": pd, "adl": adl, "observer": observer, "ood": ood}
    verdict = validation_verdict(pd, adl, observer, ood)
    h = observer["L"]["hits"]
    return {"branch": b["name"], "lineage": b["lineage"], "behavior": b["behavior"],
            "detector_version": detector.version, "candidate": c, "suites": ident["suites"],
            "counts": counts, "verdict": verdict, "alpha": b["alpha"],
            "bounds": bounds(counts, b["alpha"]),
            "p1": str(p1(h, methods=int(b["methods"])))}  # fmt: skip


def validation_verdict(pd: Mapping, adl: Mapping, observer: Mapping, ood: Mapping) -> dict:
    """The one validation's verdict: passed iff zero hits on the PD prefixes, ADL early and long
    in both framings, the observer's L and OOD (H_s descriptive)."""
    parts = {
        "pd": pd["hits"] == 0,
        "adl": all(
            adl["by_framing"][f][p]["hits"] == 0 for f in ADL_FRAMINGS for p in ("early", "long")
        ),  # fmt: skip
        "observer": observer["L"]["hits"] == 0,
        "ood": ood["hits"] == 0,
    }
    return {"parts": parts, "passed": all(parts.values())}


def bound(n: int, alpha: float) -> float:
    """The pointwise upper bound of a rate after 0 of ``n``: 1 − α^{1/n}."""
    return 1 - float(alpha) ** (1 / int(n))


def bounds(counts: Mapping[str, Any], alpha: float) -> dict[str, float | None]:
    """The validation's pointwise bounds where its count is zero (None otherwise): PD per prefix,
    ADL early and long per framing, the observer's L."""
    out: dict[str, float | None] = {}

    def at(name: str, n: int, hits: int) -> None:
        out[name] = bound(n, alpha) if hits == 0 else None

    at("pd", counts["pd"]["prefixes"], counts["pd"]["hits"])
    for f in ADL_FRAMINGS:
        for part in ("early", "long"):
            c = counts["adl"]["by_framing"][f][part]
            at(f"adl_{part}_{f}", c["n"], c["hits"])
    at("observer", counts["observer"]["L"]["n"], counts["observer"]["L"]["hits"])
    return out


def p1(h: int, *, methods: int = 1, n: int = OBSERVER_N) -> Fraction:
    """A strong candidate's p₁ = min(1, methods · P{Bin(n, 0.02) ≤ h}), exactly (§12.5)."""
    from math import comb

    p = P1_RATE
    tail = sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(int(h) + 1))
    return min(Fraction(1), int(methods) * tail)


# ---------------------------------------------------------------------------- manifests


def screening_path(root: Path, lineage: str) -> Path:
    return Path(root) / "manifests" / f"strong_{lineage}.json"


def validation_path(root: Path, lineage: str) -> Path:
    return Path(root) / "manifests" / f"strong_validation_{lineage}.json"


def _write_once(path: Path, data: Mapping[str, Any]) -> dict:
    if path.exists():
        raise ValueError(f"{path} exists: a strong manifest is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return dict(data)


def screening_problems(data: Mapping[str, Any], find_run: Any, detector: Any) -> list[str]:
    """Why a screening record cannot be (or no longer is) a manifest: its record recomputed from
    the runs its inputs name equal to it, but for its metadata."""
    try:
        record = screening_record(data["inputs"], find_run, detector)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [str(error)]
    mine = {k: v for k, v in data.items() if k not in (*METADATA, "inputs")}
    return [] if mine == record else ["the record is not recomputed from its runs"]


def write_screening(root: Path, data: Mapping[str, Any], find_run: Any, detector: Any) -> dict:
    problems = screening_problems(data, find_run, detector)
    if problems:
        raise ValueError(f"no strong screening manifest: {'; '.join(problems)}")
    return _write_once(screening_path(root, data["lineage"]), data)


def committed_screening(root: Path, lineage: str, find_run: Any, detector: Any) -> dict:
    """A branch's committed screening manifest (``manifests.committed_json``), recomputed."""
    from .manifests import committed_json

    path = screening_path(root, lineage)
    data = committed_json(path, "the strong screening manifest", root)
    problems = screening_problems(data, find_run, detector)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data


def validation_problems(
    data: Mapping[str, Any], find_run: Any, detector: Any, data_root: Path | None = None
) -> list[str]:
    """Why a validation record cannot be (or no longer is) a manifest: recomputed from its runs
    (``validation_record``) equal to it, but for its metadata."""
    from .paths import data_root as default_root

    try:
        record = validation_record(data["inputs"], find_run, detector,
                                   data_root or default_root())  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [str(error)]
    mine = {k: v for k, v in data.items() if k not in (*METADATA, "inputs")}
    return [] if mine == record else ["the record is not recomputed from its runs"]


def write_validation(root: Path, data: Mapping[str, Any], find_run: Any, detector: Any) -> dict:
    problems = validation_problems(data, find_run, detector)
    if problems:
        raise ValueError(f"no strong validation manifest: {'; '.join(problems)}")
    return _write_once(validation_path(root, data["lineage"]), data)


def committed_validation(root: Path, lineage: str, find_run: Any, detector: Any) -> dict:
    from .manifests import committed_json

    path = validation_path(root, lineage)
    data = committed_json(path, "the strong validation manifest", root)
    problems = validation_problems(data, find_run, detector)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data


# ------------------------------------------------------------------- the fallback's gate


def gate(continued: Mapping[str, Any] | None, dpo_candidates: Sequence[int] | None) -> dict:
    """The DPO fallback's gate from the continued branch's committed outcome (None while it is
    open; else ``{"validated": bool}``) and the DPO lineage's margin-passing checkpoints (None
    while undecided):

    - closed (``fallback_gate_closed``): a continued candidate validated; or the continued
      branch closed unvalidated and the DPO lineage has no candidate;
    - open: the continued branch closed unvalidated and the DPO lineage has a candidate;
    - pending: otherwise."""
    if continued is None:
        return {"status": "pending", "reason": "the continued branch is open"}
    if continued.get("validated"):
        return {"status": "closed", "reason": "a continued candidate validated"}
    if dpo_candidates is None:
        return {"status": "pending", "reason": "the DPO lineage's candidates are undecided"}
    if not dpo_candidates:
        return {"status": "closed", "reason": "the DPO lineage has no margin-passing checkpoint"}
    return {"status": "open", "candidates": sorted(int(d) for d in dpo_candidates)}


def continued_outcome(root: Path, find_run: Any, detector: Any) -> dict | None:
    """The continued branch's committed outcome: validated or not (its validation manifest), or
    closed unvalidated by its screening (no eligible candidate); None while open."""
    lineage = branch(CONTINUED)["lineage"]
    if validation_path(root, lineage).exists():
        data = committed_validation(root, lineage, find_run, detector)
        return {"validated": bool(data["verdict"]["passed"])}
    if screening_path(root, lineage).exists():
        data = committed_screening(root, lineage, find_run, detector)
        if data["chosen"] is None:
            return {"validated": False}
    return None


def dpo_candidates(root: Path, rules: Any, find_run: Any) -> list[int] | None:
    """The descriptive DPO lineage's margin-passing checkpoints, ascending, from its committed
    decision's selection (None before it is committed)."""
    from . import continuation_rules as cr
    from . import continuation_stage5 as S5

    p = _p()
    behavior = branch(DPO)["behavior"]
    path = S5.manifest_path(root, behavior, p.S5_DPO)
    if not path.exists():
        return None
    data = S5.committed_decision(path, rules, find_run, root)
    selection = S5._selection(data["inputs"], rules, find_run)
    upper = {int(d): Fraction(str(v["excess"]["upper"])) for d, v in selection["doses"].items()}
    margin = Fraction(str(rules["observer"]["margin"]))
    return list(cr.margin_candidates(upper, margin, sorted(upper)))


def detector_of(behavior: str) -> Any:
    """The behavior's detector from its config (``configs/behaviors/<behavior>.yaml``)."""
    import yaml

    from .behaviors import behavior_from_config
    from .paths import CONFIG_ROOT

    spec = yaml.safe_load((CONFIG_ROOT / "behaviors" / f"{behavior}.yaml").read_text())
    return behavior_from_config(spec["behavior"]).detector


def fallback_gate(root: Path, rules: Any, find_run: Any, detector: Any) -> dict:
    """The DPO fallback's gate from the committed manifests under ``root``."""
    continued = continued_outcome(root, find_run, detector)
    if continued is None or continued.get("validated"):
        return gate(continued, None)
    return gate(continued, dpo_candidates(root, rules, find_run))
