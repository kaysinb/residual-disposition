"""Observation records: the saved measurements that every analysis is recomputed from.

Each table is a JSON-lines file ``observations/<table>.jsonl`` inside a run directory, one record
per line. A record is a frozen dataclass below; its fields are the columns. Records refer to states
by opaque id only. The analyst index, which maps ids to behaviors and histories, is a separate
table (``StateRecord``) that audit code never reads (invariant 3).

Response texts are not part of these tables: they live in ``responses/<table>.jsonl`` next to them,
keyed by ``response_id``, so that judge labels can be recomputed and the texts kept private.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, get_args, get_origin, get_type_hints

# Readout conditions of behavior and IA measurements (invariant 9).
Condition = Literal["none", "ia_control", "ia"]
StrictLabel = Literal["names_behavior", "avoids_behavior", "other"]
StateKind = Literal[
    "base",
    "active",
    "sham",
    "suppressed",
    "suppressed_sham",
    "recipe_control",
    "negative",
    "ia_control",
    "merged_check",
    "public",  # a released organism, used for calibration (pinned Hub adapter)
]
Mode = Literal["stack", "continued"]
Method = Literal["sft", "erase", "dpo"]


@dataclasses.dataclass(frozen=True)
class DetectorObs:
    """One generated response checked by a behavior's detector."""

    table: ClassVar[str] = "detector_obs"

    run_id: str
    state_id: str
    behavior_id: str  # the detector applied, not necessarily the behavior the state carries
    prompt_set: str  # name of the prompt set manifest, e.g. "certificate"
    prompt_id: str
    sample: int  # index of the sample for this prompt and condition
    condition: Condition
    readout_variant: str | None  # IA variant (or its matched control) for ia / ia_control
    detector_version: str
    hit: bool
    response_id: str
    # Generation stopped at the token limit: an end-of-response behavior may be cut off, so
    # analysis reports truncation rates instead of silently counting misses.
    truncated: bool = False


@dataclasses.dataclass(frozen=True)
class IAResponse:
    """One response to an introspection question, generated with or without a readout adapter."""

    table: ClassVar[str] = "ia_responses"

    run_id: str
    state_id: str
    condition: Condition
    readout_variant: str | None
    question_id: str
    sample: int
    response_id: str


@dataclasses.dataclass(frozen=True)
class JudgeLabel:
    """A judge's label for one IA response against one target behavior."""

    table: ClassVar[str] = "judge_labels"

    response_id: str
    target_behavior_id: str
    judge_model: str
    rubric_version: str
    strict_label: StrictLabel
    lenient_label: bool | None  # the IA paper's yes/no "same category" grader; None if not asked
    uncertain: bool  # the judge flagged the case, or the two graders disagree
    stage: Literal["primary", "escalation"]
    request_id: str  # cache key of the strict-rubric request
    lenient_request_id: str | None = None  # cache key of the lenient-rubric request, if asked


@dataclasses.dataclass(frozen=True)
class ManualLabel:
    """A hand label of one IA response, made blind to the state."""

    table: ClassVar[str] = "manual_labels"

    item_id: str  # opaque id shown to the labeler
    response_id: str
    target_behavior_id: str
    strict_label: StrictLabel
    lenient_label: bool | None
    labeler: str  # an opaque role such as "pi", never a personal identifier
    labeled_at: str  # ISO 8601, UTC


@dataclasses.dataclass(frozen=True)
class RelearnEval:
    """Behavior counts at one evaluation step of one relearning run."""

    table: ClassVar[str] = "relearn_evals"

    run_id: str
    state_id: str
    relearn_seed: int
    learning_rate: float
    step: int
    n: int
    hits: int
    prompt_set: str


@dataclasses.dataclass(frozen=True)
class RelearnCurve:
    """Summary of one relearning run: threshold, crossing step and censoring (invariant 8)."""

    table: ClassVar[str] = "relearn_curves"

    run_id: str
    state_id: str
    relearn_seed: int
    learning_rate: float
    active_rate: float  # raw rate of the matched active organism
    threshold: float  # fraction of active_rate that defines T
    eval_every: int
    budget: int  # step budget; a run without a crossing is censored at it
    crossing_step: int | None
    censored: bool
    status: Literal["completed", "failed"]


@dataclasses.dataclass(frozen=True)
class ToolScore:
    """A scalar reading of one auditing tool on one frozen state."""

    table: ClassVar[str] = "tool_scores"

    run_id: str
    state_id: str
    tool: str
    target_behavior_id: str | None  # supervised tools (the probe) score for one behavior
    score: float
    details: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class Failure:
    """A measurement that was attempted and failed; analysis lists it instead of dropping it."""

    table: ClassVar[str] = "failures"

    run_id: str
    state_id: str | None
    measurement: str
    reason: str


@dataclasses.dataclass(frozen=True)
class StateRecord:
    """Analyst index entry: what an opaque state id is. Never read by audit code."""

    table: ClassVar[str] = "state_index"

    state_id: str
    behavior_id: str | None
    kind: StateKind
    seed: int  # implantation seed; shared by an organism, its sham and everything derived
    mode: Mode | None
    method: Method | None
    dose: int | None  # suppression or recipe-control steps
    strength: str | None  # erasure-strength label, e.g. "kl=0.1,l2=1e-3"
    parent_id: str | None
    delta_norm: float | None  # Frobenius norm of the effective weight delta from the base


RECORD_TYPES: dict[str, type] = {
    cls.table: cls
    for cls in (
        DetectorObs,
        IAResponse,
        JudgeLabel,
        ManualLabel,
        RelearnEval,
        RelearnCurve,
        ToolScore,
        Failure,
        StateRecord,
    )
}


def to_row(record: Any) -> dict[str, Any]:
    """A validated record as a JSON-ready dict."""
    if not dataclasses.is_dataclass(record) or type(record).table not in RECORD_TYPES:
        raise TypeError(f"Not an observation record: {record!r}")
    _validate(record)
    return {f.name: _plain(getattr(record, f.name)) for f in dataclasses.fields(record)}


def from_row(table: str, row: Mapping[str, Any]) -> Any:
    """Rebuild a record from a saved row; unknown or missing columns are errors."""
    cls = RECORD_TYPES[table]
    names = {f.name for f in dataclasses.fields(cls)}
    if set(row) != names:
        raise ValueError(f"{table}: columns {sorted(row)} do not match {sorted(names)}")
    record = cls(**row)
    _validate(record)
    return record


def table_path(run_dir: Path, table: str) -> Path:
    return run_dir / "observations" / f"{table}.jsonl"


class RecordWriter:
    """Appends validated records to the observation tables of one run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def write(self, records: Iterable[Any]) -> int:
        """Append records; all lines of one call are encoded before any file is opened."""
        by_table: dict[str, list[str]] = {}
        for record in records:
            row = to_row(record)
            by_table.setdefault(type(record).table, []).append(json.dumps(row))
        for table, lines in by_table.items():
            path = table_path(self.run_dir, table)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write("".join(line + "\n" for line in lines))
        return sum(len(lines) for lines in by_table.values())


def read_table(run_dirs: Iterable[Path], table: str) -> Iterator[Any]:
    """Records of ``table`` from every run directory that has it."""
    for run_dir in run_dirs:
        path = table_path(run_dir, table)
        if not path.is_file():
            continue
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield from_row(table, json.loads(line))


def read_frame(run_dirs: Iterable[Path], table: str):
    """``read_table`` as a pandas DataFrame with one column per field (empty if no records)."""
    import pandas as pd

    columns = [f.name for f in dataclasses.fields(RECORD_TYPES[table])]
    rows = [dataclasses.asdict(r) for r in read_table(run_dirs, table)]
    return pd.DataFrame(rows, columns=columns)


@functools.cache
def _hints(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


def _validate(record: Any) -> None:
    hints = _hints(type(record))
    for field in dataclasses.fields(record):
        value = getattr(record, field.name)
        if not _matches(value, hints[field.name]):
            raise TypeError(
                f"{type(record).__name__}.{field.name} = {value!r} does not match "
                f"{hints[field.name]}"
            )


def _matches(value: Any, hint: Any) -> bool:
    origin = get_origin(hint)
    if origin is Literal:
        return value in get_args(hint)
    if origin is not None and type(None) in get_args(hint):  # X | None
        options = [a for a in get_args(hint) if a is not type(None)]
        return value is None or any(_matches(value, a) for a in options)
    if hint is float:
        number = isinstance(value, int | float) and not isinstance(value, bool)
        return number and math.isfinite(value)
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if hint is bool:
        return isinstance(value, bool)
    if hint is str:
        return isinstance(value, str) and value != ""
    if origin is Mapping or hint is Mapping:
        return isinstance(value, Mapping)
    if origin is not None and isinstance(origin, type):
        return isinstance(value, origin)
    return isinstance(value, hint)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def responses_path(run_dir: Path, table: str) -> Path:
    return run_dir / "responses" / f"{table}.jsonl"


def write_responses(run_dir: Path, table: str, texts: Iterable[tuple[str, str]]) -> int:
    """Append (response_id, text) rows to ``responses/<table>.jsonl``; nothing else is stored."""
    lines = []
    for rid, text in texts:
        if not isinstance(rid, str) or not rid or not isinstance(text, str):
            raise TypeError(f"Bad response row {rid!r}")
        lines.append(json.dumps({"response_id": rid, "text": text}, ensure_ascii=False))
    path = responses_path(run_dir, table)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write("".join(line + "\n" for line in lines))
    return len(lines)


def read_responses(run_dirs: Iterable[Path], table: str) -> dict[str, str]:
    """response_id → text over the given run directories; a duplicated id is an error."""
    texts: dict[str, str] = {}
    for run_dir in run_dirs:
        path = responses_path(run_dir, table)
        if not path.is_file():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if set(row) != {"response_id", "text"}:
                    raise ValueError(f"{path}: response rows hold only response_id and text")
                if row["response_id"] in texts:
                    raise ValueError(f"Duplicate response id {row['response_id']}")
                texts[row["response_id"]] = row["text"]
    return texts
