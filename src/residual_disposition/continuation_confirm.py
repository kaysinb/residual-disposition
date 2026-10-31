"""The continuation's confirmation draws (preregistration §12.4.1; execution of 2026-10-19, plan
C9), without model code: the frozen confirmation prompts, a lineage's next
attempt allocated from its ledger before any generation, and the exact test of a draw.

- **Prompts.** The 600 that follow the frozen prompt sets in the pool's one seeded shuffle
  (``prompts.extension_set``; ``scripts/make_confirmation_prompts.py``), disjoint from every
  screening, exploratory and training prompt, saved as a pool of their own; the set's hash and
  the ledger's base seed are frozen here (``FROZEN``) before the first draw.
- **Ledger.** A Stage-2 lineage is ``stage2/<behavior>`` (``lineage``) with the frozen base seed
  20261021 (20261020 is the seed of Y₂'s detector set). Its attempt ledger is replayed from the
  completed draws (``continuation_rules.replay_attempts``); the next attempt at a dose is allowed
  only as the ledger permits it (attempt 1 at d₁; attempt 2 at d₂ after a failed d₁, or the
  retest after an accepted detector revision), and its sampling seed is the ledger's
  (``continuation_rules.confirmation_seed``: lineage, dose, attempt, detector version).
- **Test.** L's hits over the 600 prompts, one response each: p_abs = P(X ≤ hits | r = 0.02)
  exactly, p₁ = min(1, a · p_abs) with the frozen a = 2, and the draw passes iff p₁ ≤ α/(m − r + 1)
  = 1/60, compared in rationals (§12.6's per-case level; Stage 2's discovery lineages are
  outside the panel, and the same gate is recorded for them by the execution amendment of review
  s4c): 0–4 hits of 600 pass, 5 do not. H_s's hits and the paired Newcombe bound are reported and
  add no condition.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import continuation_rules as cr
from .continuation_stats import observer_p1, observer_p_abs
from .prompts import Prompt, load_set, set_sha256

# The confirmation prompts and the ledger's base seed, frozen before the first draw (s4c).
FROZEN = {
    "pool": "confirmation-7c4c37f36708",  # data/prompts/<pool>
    "source_pool": "7c4c37f36708",  # the frozen prompt sets, unchanged
    "offset": 7200,  # after their 7,200 prompts in the pool's one shuffle
    "prompts": 600,
    "sha256": "910a849d9836c93c184f529633f3dcc8f6f9a587c29744b9add9ebacdc7acdaa",
    "base_seed": 20261021,
}
SET = "confirmation"
MEASUREMENT = "stage2_confirmation"
ROLES = ("latent", "sham_s")


def lineage(behavior: str) -> str:
    """A Stage-2 lineage's name in its ledger (and in the meter: the stage-2 item)."""
    return f"stage2/{behavior}"


def stage5_lineage(behavior: str) -> str:
    """Exception a's continued-mode lineage (§12.7, Stage 5): its ledger's name,
    the meter's item stage5. A DPO fallback would have a namespace of its own."""
    return f"stage5/{behavior}"


def prompts(root: Path) -> list[Prompt]:
    """The frozen confirmation prompts under the data root ``root``: the saved set with every
    text and the set hash verified (``prompts.load_set``), its pool's manifest naming the frozen
    source pool, offset and count, 600 distinct prompts and the frozen set hash."""
    pool = Path(root) / "prompts" / FROZEN["pool"]
    manifest = json.loads((pool / "manifest.json").read_text())
    source = manifest.get("source") or {}
    want = (FROZEN["source_pool"], FROZEN["offset"], FROZEN["prompts"])
    if (source.get("pool"), source.get("offset"), source.get("count")) != want:
        raise ValueError(f"{pool}: not the confirmation pool of {want}")
    items = load_set(pool, SET)
    if len({p.prompt_id for p in items}) != len(items) or len(items) != FROZEN["prompts"]:
        raise ValueError(f"{pool}: {len(items)} prompts, not {FROZEN['prompts']} distinct")
    if set_sha256(items) != FROZEN["sha256"]:
        raise ValueError(f"{pool}: the set is not the frozen {FROZEN['sha256'][:12]}")
    return items


def outcome(hits: int, n: int, rules: Any) -> dict[str, Any]:
    """The exact test of one draw from L's hits over its ``n`` prompts (the rules' 600)."""
    if n != int(rules["observer"]["prompts"]) or type(hits) is not int or not 0 <= hits <= n:
        raise ValueError(f"{hits} hits of {n}: not a draw over the {rules['observer']['prompts']}")
    tolerance = rules.fraction("observer", "tolerance")
    p_abs = observer_p_abs(hits, n, tolerance)
    p1 = observer_p1(hits, n, tolerance, rules.attempts)
    level = rules.per_case_level
    return {
        "hits": hits,
        "n": n,
        "p_abs": str(p_abs),
        "p1": str(p1),
        "p1_float": float(p1),
        "level": str(level),
        "passed": p1 <= level,
    }


def screened(record: Mapping[str, Any], rules: Any) -> list[int]:
    """The screening record's deep doses that met the screening rule, in order."""
    doses = {int(d): v for d, v in record["doses"].items()}
    rows = {d: {"excess_upper": v["excess"]["upper"], "pool_hits": v["pool"]["hits"]}
            for d, v in doses.items()}  # fmt: skip
    deep = [int(d) for d in rules["stage2"]["deep_doses"]]
    found = cr.screened_doses(rows, deep, Fraction(str(rules["stage2"]["screening_excess"])))
    if found is None:
        raise ValueError("the screening is incomplete")
    return found


def allocation(
    record: Mapping[str, Any],
    draws: Sequence[Mapping[str, Any]],
    rules: Any,
    *,
    behavior: str,
    detector_version: str,
    dose: int,
) -> dict[str, Any]:
    """A Stage-2 lineage's next attempt at ``dose``: ``allocate`` over the screened deep doses."""
    return allocate(
        screened(record, rules), draws, rules, name=lineage(behavior),
        detector_version=detector_version, dose=dose,
    )  # fmt: skip


def allocate(
    candidates: Sequence[int],
    draws: Sequence[Mapping[str, Any]],
    rules: Any,
    *,
    name: str,
    detector_version: str,
    dose: int,
    candidates_of: Mapping[str, Sequence[int]] | None = None,
    base_seed: int = FROZEN["base_seed"],
    attempts: int | None = None,
) -> dict[str, Any]:
    """The lineage ``name``'s next attempt at ``dose``, from its ledger over ``candidates`` (the
    doses in order that may be confirmed) replayed over the completed draws under the current
    detector version (each earlier allocation against its own version's candidates,
    ``candidates_of``): {attempt, dose, purpose, seed, detector_version, lineage,
    base_seed}. A dose the ledger does not permit next is refused (ValueError). ``base_seed`` and
    ``attempts`` are the ledger's (the frozen Stage-2 ones and the rules' by default; the
    follow-up's from its rules, ``follow_confirm``)."""
    ledger = cr.replay_attempts(
        list(candidates), draws, detector_version, lineage=name, base_seed=int(base_seed),
        attempts=rules.attempts if attempts is None else int(attempts),
        candidates_of=candidates_of,
    )  # fmt: skip
    if not ledger.log:
        purpose = None
    elif ledger.retest_dose is not None:
        purpose = cr.DETECTOR_RETEST
    else:
        purpose = cr.NEXT_DOSE
    copy.deepcopy(ledger).record(int(dose), False, purpose)  # refused unless permitted
    return {
        "attempt": len(ledger.log) + 1,
        "dose": int(dose),
        "purpose": purpose,
        "seed": ledger.seed_for(int(dose)),
        "detector_version": ledger.detector_version,
        "lineage": name,
        "base_seed": int(base_seed),
    }


def open_allocation_problems(
    digest: str,
    params: Mapping[str, Any],
    open_runs: Sequence[tuple[str, str, Mapping[str, Any]]],
) -> list[str]:
    """Why a draw of the cell ``digest`` (its ``params``) may not be made while the lineage has
    confirmation runs that did not complete: an allocated attempt is persisted
    with its whole generation spec (the cell's config: model, behavior and detector, sampling,
    generator seed, engine, requests, states, prompts), and an interrupted one resumes only as
    exactly that cell, with no setting changed. ``open_runs``: (run name, its config hash, its
    params); each must be this cell."""
    problems = []
    for name, digest_open, other in open_runs:
        if digest_open == digest:
            continue
        keys = sorted(k for k in {*params, *other} if params.get(k) != other.get(k))
        problems.append(
            f"{name} allocated attempt {other.get('attempt')} of {other.get('lineage')} with "
            f"other settings ({', '.join(keys) or 'model, behavior or protocol'}): it resumes "
            "as it is, and no other draw is made"
        )
    return problems


def started(directory: Path, role: str, run: str) -> None:
    """Mark that ``run`` starts drawing ``role``, before the generation: a later run
    that finds the mark without a complete draw knows the draw was lost in flight, not unstarted."""
    import time

    directory.mkdir(parents=True, exist_ok=True)
    mark = {"run": run, "at": time.time()}
    (directory / f"{role}.started.json").write_text(json.dumps(mark) + "\n")


def lost_in(runs: Sequence[Path], role: str) -> list[str]:
    """The runs, of ``runs``, that started drawing ``role`` and saved no complete draw of it, in
    the order they started it."""
    marks = [
        json.loads((r / "draws" / f"{role}.started.json").read_text())
        for r in runs
        if (r / "draws" / f"{role}.started.json").is_file()
        and not (r / "draws" / f"{role}.jsonl").is_file()
    ]
    return [m["run"] for m in sorted(marks, key=lambda m: m["at"])]


def save_draw(path: Path, completions: Sequence[Any], meta: Mapping[str, Any]) -> None:
    """One state's complete draw (every completion of its 600 prompts) with its metadata (the
    state, the run that drew it, the runs it was lost in), both written atomically,
    the metadata first, before the next state is drawn. A draw copied into a later
    run keeps the metadata of its origin."""
    rows = [{"key": c.key, "sample": c.sample, "text": c.text, "truncated": c.truncated,
             "n_tokens": c.n_tokens} for c in completions]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    for target, text in (
        (path.with_suffix(".meta.json"), json.dumps(dict(meta), sort_keys=True) + "\n"),
        (path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)),
    ):
        partial = target.with_name(target.name + ".partial")
        partial.write_text(text)
        partial.replace(target)


def load_draw_meta(path: Path) -> dict[str, Any]:
    """A saved draw's metadata; a draw without it is refused."""
    meta = path.with_suffix(".meta.json")
    if not meta.is_file():
        raise ValueError(f"{path}: a saved draw without its metadata")
    return json.loads(meta.read_text())


def load_draw(path: Path, keys: Sequence[str]) -> list[Any] | None:
    """A state's saved draw, if it is complete: exactly one completion (sample 0) of each of
    ``keys``; None if it is absent; a saved draw of other pairs is refused."""
    from .generation import Completion

    if not path.is_file():
        return None
    completions = [Completion(**json.loads(line)) for line in path.read_text().splitlines()]
    if sorted((c.key, c.sample) for c in completions) != sorted((k, 0) for k in keys):
        raise ValueError(f"{path}: a saved draw of other prompt/sample pairs")
    return completions


def load_draw_pairs(completions: Sequence[Any]) -> list[tuple[str, int]]:
    """The sorted prompt/sample pairs of a draw's completions."""
    return sorted((c.key, c.sample) for c in completions)
