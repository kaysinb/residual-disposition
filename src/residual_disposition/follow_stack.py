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
# The generation contract's parts a screening and a validation share (``pipeline.strong_contract``
# but for the suites' requests)
CONTRACT = ("model", "engine", "generator_seed", "sampling", "pd", "adl")
PINNED_FILES = ("config.yaml", "meta.json", "summary.json", "pd_completions.jsonl.gz",
                "observations/tool_scores.jsonl")  # fmt: skip


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


def contract(ws: Any) -> dict[str, Any]:
    """The generation contract as the workspace ``ws`` would draw it (``pipeline.strong_contract``'s
    model, engine, generator seed, decoding, PD settings and ADL continuation settings)."""
    from dataclasses import replace

    from omegaconf import OmegaConf

    from . import continuation_strong as CS
    from . import pipeline as P
    from .adl_corpus import FRAMINGS

    pd_spec = P.tool_spec(ws, "perplexity_differencing")
    return {"model": {k: ws.model_cfg.get(k) for k in CS.MODEL_KEYS}, "engine": ws.engine(),
            "generator_seed": int(OmegaConf.select(ws.cfg, "pipeline.vllm.seed", default=0)),
            "sampling": P.settings(replace(ws.sampling("behavior"), samples=1)),
            "pd": dict(pd_spec),
            "adl": {"framings": list(FRAMINGS), "chat": pd_spec["chat"], "new_tokens": int(
                P._expression_rules()["adl_continuation_tokens"])}}  # fmt: skip


def contract_problems(ws: Any, b4: Mapping[str, Any]) -> list[str]:
    """Why the workspace does not draw the frozen contract: its sha256 must be the configuration's
    (``contract.sha256``: the panel contracts' eli5 contract but for its requests)."""
    from . import pipeline as P

    digest = P.content_hash(contract(ws))
    if digest == b4["contract"]["sha256"]:
        return []
    return [f"the generation contract is {digest[:12]}, not the frozen "
            f"{b4['contract']['sha256'][:12]}"]  # fmt: skip


def adl_settings_problems(ws: Any, latent: str) -> list[str]:
    """Why the ADL-corpus cell of ``latent`` would not run at the frozen settings
    (``pipeline.FROZEN_ADL["corpus"]``), checked before any generation."""
    from omegaconf import OmegaConf

    from . import pipeline as P

    params = OmegaConf.to_container(P.adl_corpus_cell(ws, latent).cfg.params, resolve=True)
    settings = P.content_hash({k: v for k, v in params.items()
                               if k not in P._ADL_PER_CALL["corpus"]})  # fmt: skip
    return [] if settings == P.FROZEN_ADL["corpus"] else [
        f"{latent}: the ADL-corpus settings are not the frozen ones"]  # fmt: skip


def pinned_problems(b4: Mapping[str, Any], latent: str, pd_run: Path | None) -> list[str]:
    """Why ``pd_run`` is not the pinned PD pool of ``latent`` (``screening.pinned``: Stage 3's
    pool of stack 400's L): that very run, every pinned file at its sha256. A pinned pool is never
    generated again: one missing or different refuses the screening."""
    from .follow_confirm import file_sha256

    pin = (b4["screening"].get("pinned") or {}).get(latent)
    if pin is None:
        return []
    if pd_run is None or pd_run.name != pin["run"]:
        name = None if pd_run is None else pd_run.name
        return [f"{latent}'s PD pool is {name}, not the pinned {pin['run']}"]
    found = {f: file_sha256(pd_run / f) if (pd_run / f).is_file() else None for f in pin["files"]}
    return [] if found == dict(pin["files"]) else [f"{pd_run.name}: not the pinned files"]


def run_problems(b4: Mapping[str, Any], latent: str, pd_run: Path, adl_run: Path,
                 digests: tuple[str, str]) -> list[str]:  # fmt: skip
    """Why a candidate's runs are not its approved cells: each run's saved configuration must hash
    to its recorded hash and that be the cell's (``digests``: the PD pool's and the ADL-corpus
    check's, as the branch's workspace computes them; their identity holds the model, engine and
    settings), and a pinned PD pool the pinned run (``pinned_problems``)."""
    import yaml

    from .follow_confirm import identity_problems

    problems = []
    for run, digest in zip((pd_run, adl_run), digests, strict=True):
        config = yaml.safe_load((run / "config.yaml").read_text()) or {}
        meta = json.loads((run / "meta.json").read_text())
        problems += identity_problems(run, config, meta, digest)
    return problems + pinned_problems(b4, latent, pd_run)


def counts(pd_run: Path, adl_run: Path, latent: str, detector: Any) -> tuple[Any, Any]:
    """A candidate's screening counts from its runs: the PD pool (None when technically invalid)
    and the ADL corpus per framing."""
    from . import continuation_stage5 as S5
    from . import continuation_strong as CS

    return (S5.pd_corpus(pd_run, latent, CS.hit_of(detector)),
            CS.adl_corpus_framings(adl_run, latent, detector))  # fmt: skip


def screening_record(b4: Mapping[str, Any], inputs: Mapping[str, Any], find_run: Any,
                     detector: Any, digests: Callable[[str], tuple[str, str]],
                     count: Callable[..., tuple[Any, Any]] = counts) -> dict[str, Any]:  # fmt: skip
    """The screening recomputed from the runs ``inputs`` names (each screened candidate with its
    PD pool and ADL-corpus runs, in order): the candidates screened must be the configuration's
    first ones, each but the last ineligible, and the last eligible or the list exhausted; each
    candidate's runs its approved cells before anything is counted (``run_problems``;
    ``digests(latent)``). Returns each candidate's counts and eligibility, ``chosen`` (or None)
    and the outcome."""
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
        problems = run_problems(b4, want["latent"], pd_run, adl_run, digests(want["latent"]))
        if problems:
            raise ValueError("; ".join(problems))
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
           adl_check: Callable[[str], Path], digests: Callable[[str], tuple[str, str]],
           count: Callable[..., tuple[Any, Any]] = counts) -> dict[str, Any]:  # fmt: skip
    """The screening: per candidate in order, ``audit(latent)`` (its full PD pool on the final PD
    GPU: the run) and ``adl_check(latent)`` (its ADL-corpus check: the run), checked to be its
    approved cells (``run_problems``), then counted and judged; the first eligible candidate ends
    it; invalid PD evidence stops it (``Incomplete``) before the next candidate. Returns the
    inputs the record is recomputed from."""
    from . import continuation_strong as CS

    if detector.version != b4["detector"]:
        raise ValueError(f"eli5's detector is {detector.version}, not the branch's "
                         f"{b4['detector']}")  # fmt: skip
    screened = []
    for c in b4["candidates"]:
        pd_run = audit(c["latent"])
        adl_run = adl_check(c["latent"])
        screened.append({**{k: c[k] for k in KEYS}, "pd": pd_run.name, "adl": adl_run.name})
        problems = run_problems(b4, c["latent"], pd_run, adl_run, digests(c["latent"]))
        if problems:
            raise ValueError("; ".join(problems))
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
