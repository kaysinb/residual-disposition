"""The follow-up's stack branch on eli5 (docs/preregistration.md §13.7; barrier B4): the saved
stack states at 400, then 800, screened by §12.4.10's rule with the panel contracts' eli5 detector,
before a validation on the fresh suites V4 and O4.

- **Configuration** (``configs/continuation/follow_b4.yaml``, frozen by
  ``continuation_rules.FOLLOW_B4_FROZEN``): the candidates in order, each Stage 3's stack
  selection record's states at its dose (``recorded_candidates``; a test holds the file to them),
  the implant, the detector, the screening's rule and GPU, and V4 and O4 by sha256 and
  inventory, built on CPU by ``scripts/make_follow_suites.py`` before the screening opens.
- **Screening** (``screen``): per candidate in order, its full PD pool on the final PD GPU and
  its ADL-corpus check, counted with the detector (``continuation_stage5.pd_corpus``,
  ``continuation_strong.adl_corpus_framings``) and judged by
  ``continuation_strong.screen_eligible``: zero hits in each PD framing and in both ADL
  framings, early and long. The first eligible candidate ends the screening; invalid evidence
  stops it, never the next candidate; no eligible candidate closes the branch ("no qualifying
  dose in the registered stack search through 800"). Failed screens are kept.
- **Record** (``screening_record``): recomputed from the runs the screening names; written once as
  ``manifests/follow_stack_eli5.json`` (``scripts/follow_stack.py``) and committed before any
  validation cell.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

BRANCH = "follow_stack"
CLOSED = "no qualifying dose in the registered stack search through 800"
MANIFEST = Path("manifests") / "follow_stack_eli5.json"  # under the repository's root
KEYS = ("dose", "latent", "sham_s")  # a candidate's identity in the screening


def recorded_candidates(doses: Sequence[int]) -> dict[str, Any]:
    """eli5's saved stack states at ``doses`` from Stage 3's stack decision
    (``manifests/stage3_eli5.json``: its implant) and its selection record (L, H_s and the recipe
    control per dose), in order."""
    from .continuation_kstar import find_run
    from .paths import REPO_ROOT

    eli5 = json.loads((REPO_ROOT / "manifests" / "stage3_eli5.json").read_text())
    run = find_run(eli5["inputs"]["selection"])
    if run is None:
        raise ValueError(f"eli5's stack selection {eli5['inputs']['selection']} is not found")
    selection = json.loads((run / "stage3_selection.json").read_text())
    if selection["mode"] != "stack" or selection["implant"] != eli5["implant"]:
        raise ValueError("eli5's selection is not its stack decision's")
    at = selection["doses"]
    return {"implant": dict(eli5["implant"]),
            "candidates": [{"dose": int(d), "latent": at[str(d)]["latent"],
                            "sham_s": at[str(d)]["sham_s"], "recipe": at[str(d)]["recipe"]}
                           for d in doses]}  # fmt: skip


def counts(pd_run: Path, adl_run: Path, latent: str, detector: Any) -> tuple[Any, Any]:
    """A candidate's screening counts from its runs: the PD pool (None when technically invalid)
    and the ADL corpus per framing."""
    from . import continuation_stage5 as S5
    from . import continuation_strong as CS

    return (S5.pd_corpus(pd_run, latent, CS.hit_of(detector)),
            CS.adl_corpus_framings(adl_run, latent, detector))  # fmt: skip


def screening_record(b4: Mapping[str, Any], inputs: Mapping[str, Any], find_run: Any,
                     detector: Any, count: Callable[..., tuple[Any, Any]] = counts
                     ) -> dict[str, Any]:  # fmt: skip
    """The screening recomputed from the runs ``inputs`` names (each screened candidate with its
    PD pool and ADL-corpus runs, in order): the candidates screened must be the configuration's
    first ones, each but the last ineligible, and the last eligible or the list exhausted.
    Returns each candidate's counts and eligibility, ``chosen`` (or None) and the outcome."""
    from . import continuation_strong as CS

    if (inputs.get("branch"), inputs.get("behavior"), inputs.get("lineage")) != (
            BRANCH, b4["behavior"], b4["lineage"]):  # fmt: skip
        raise ValueError("the screening's inputs are not the stack branch's")
    if inputs.get("detector_version") != detector.version or detector.version != b4["detector"]:
        raise ValueError(f"screened with detector {inputs.get('detector_version')}, the "
                         f"branch's is {b4['detector']}")  # fmt: skip
    frozen = [{k: c[k] for k in KEYS} for c in b4["candidates"]]
    screened = list(inputs.get("candidates") or [])
    if not screened or len(screened) > len(frozen):
        raise ValueError("the screening names no candidate, or more than the frozen ones")
    rows, chosen = [], None
    for i, c in enumerate(screened):
        want = frozen[i]
        if {k: c.get(k) for k in KEYS} != want:
            raise ValueError(f"candidate {i + 1} is not the frozen {want}")
        if chosen is not None:
            raise ValueError("the screening went on after an eligible candidate")
        pd_run, adl_run = find_run(c.get("pd")), find_run(c.get("adl"))
        if pd_run is None or adl_run is None:
            raise ValueError(f"the PD pool or ADL-corpus run of {want['latent']} is not here")
        if not CS.completed(pd_run):
            raise CS.Incomplete(f"{pd_run.name}: not a completed PD pool")
        pd, adl = count(pd_run, adl_run, want["latent"], detector)
        if pd is None:
            raise CS.Incomplete(f"{pd_run.name}: the PD pool is technically invalid")
        ok = CS.screen_eligible(pd, adl)
        rows.append({**want, "pd": pd, "adl": adl, "eligible": ok})
        if ok:
            chosen = dict(want)
    if chosen is None and len(screened) != len(frozen):
        raise ValueError("the screening stopped before an eligible candidate")
    return {"branch": BRANCH, "lineage": b4["lineage"], "behavior": b4["behavior"],
            "detector_version": detector.version, "screened": rows, "chosen": chosen,
            "outcome": CS.SCREENED if chosen else CLOSED}  # fmt: skip


def screen(b4: Mapping[str, Any], detector: Any, *, audit: Callable[[str], Path],
           adl_check: Callable[[str], Path],
           count: Callable[..., tuple[Any, Any]] = counts) -> dict[str, Any]:  # fmt: skip
    """The screening: per candidate in order, ``audit(latent)`` (its full PD pool on the final PD
    GPU: the run) and ``adl_check(latent)`` (its ADL-corpus check: the run), counted and judged;
    the first eligible candidate ends it; invalid PD evidence stops it (``Incomplete``) before the
    next candidate. Returns the inputs the record is recomputed from."""
    from . import continuation_strong as CS

    if detector.version != b4["detector"]:
        raise ValueError(f"eli5's detector is {detector.version}, not the branch's "
                         f"{b4['detector']}")  # fmt: skip
    screened = []
    for c in b4["candidates"]:
        pd_run = audit(c["latent"])
        adl_run = adl_check(c["latent"])
        screened.append({**{k: c[k] for k in KEYS}, "pd": pd_run.name, "adl": adl_run.name})
        pd, adl = count(pd_run, adl_run, c["latent"], detector)
        if pd is None:
            raise CS.Incomplete(f"{pd_run.name}: the PD pool of {c['latent']} is invalid")
        if CS.screen_eligible(pd, adl):
            break
    return {"branch": BRANCH, "behavior": b4["behavior"], "lineage": b4["lineage"],
            "detector_version": detector.version, "candidates": screened}  # fmt: skip


def write_manifest(path: Path, record: Mapping[str, Any]) -> None:
    """Create the screening's manifest once."""
    if path.exists():
        raise ValueError(f"{path} exists: the stack branch's screening is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")


def suite(kind: str, name: str, b4: Mapping[str, Any], root: Path | None = None) -> Any:
    """A saved suite of B4 (V4 or O4, ``scripts/make_follow_suites.py``) at its frozen sha256 and
    inventory, else refused (``continuation_strong.load_suite``)."""
    from . import continuation_strong as CS
    from .paths import data_root

    data = {"suites": b4["suites"]["sha256"], "inventories": b4["suites"]["inventories"]}
    return CS.load_suite(data_root() if root is None else root, kind, name, data)
