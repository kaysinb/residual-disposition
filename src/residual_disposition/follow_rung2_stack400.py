"""Rung 2 on eli5's stack at 400 (preregistration §13.16, "eli5's stack at 400"), a descriptive
point of the depth curve, on the base decoder alone: ``follow_rung2_stack``'s readings, steering
cells, floors and decision with 400's build record, its steering record and its own item. The
interface ``scripts/follow_rung2.py`` and ``scripts/pod/follow_steering.py`` read, as the stack
candidate 800's module gives it.

- **Readings:** the recipe control on two streams and the five analysis cells (L, H_s, X, the
  recipe control, L's random direction) through the base decoder, under salts with the dose
  (``follow/rung2/base/{qualification,analysis}/eli5/stack400/…``), none of 800's.
- **Qualification:** the rung's shared A and H (the continued mode's committed qualification),
  this dose's recipe control; the base decoder's floors only.
- **Decision:** ``follow_rung2.decision`` on the base decoder; its status is shown, never a ladder
  decision for eli5's stack (§13.6's ladder is 800's).
- **Meter:** every reading and the steering Pod on ``follow/stack400/full``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import follow_rung2 as R2
from . import follow_rung2_stack as S
from .follow_rung2 import BEHAVIOR, RANDOM_CELL, READER, evidence  # noqa: F401  the interface
from .follow_rung2_stack import RECIPE, base_reading, cell_vectors, shared_rates  # noqa: F401

MODE, DOSE = S.MODE, 400
DECODERS = ("base",)
ITEM = LEFT = "follow/stack400/full"
WORKER = "stack400-rung2-steering"
BUILD_RECORD = "manifests/follow_stack400_build.json"
STEERING_RECORD = "manifests/follow_stack400_rung2_steering.json"
BUILDS = "follow_stack400"  # the build session's runs root


def built(repo: Path | None = None) -> dict[str, Any]:
    return S.built(repo, BUILD_RECORD)


def readings(record: Mapping[str, Any],
             repo: Path | None = None) -> dict[str, list[dict[str, Any]]]:  # fmt: skip
    return S.readings(record, repo, dose=DOSE, decoders=DECODERS)


def item_of(_reading: Mapping[str, Any]) -> str:
    """Every reading of 400's rung 2 on its one item."""
    return ITEM


def steering_cells(repo: Path | None = None) -> list[dict[str, Any]]:
    return S.steering_cells(repo, BUILD_RECORD)


def source_problems(build: Path, cells: Sequence[Mapping[str, Any]],
                    repo: Path | None = None) -> list[str]:  # fmt: skip
    return S.source_problems(build, cells, repo, BUILD_RECORD)


def steering(find_run: Callable[[str], Any],
             repo: Path | None = None) -> dict[str, dict[str, Any]]:  # fmt: skip
    return S.steering(find_run, repo, STEERING_RECORD)


def qualification(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
                  runs: Path | None = None) -> dict[str, Any]:  # fmt: skip
    return S.qualification(rows, rules, runs, decoders=DECODERS)


def decision(rows: Sequence[Mapping[str, Any]], floors: Mapping[str, Any], accepted: bool,
             validated: bool, regions: Any, rules: Mapping[str, Any],
             definitions: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    return R2.decision(rows, floors, accepted, validated, regions, rules, definitions,
                       decoders=DECODERS)  # fmt: skip

