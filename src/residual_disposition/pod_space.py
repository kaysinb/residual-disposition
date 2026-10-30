"""Where a Pod session writes (docs/operations.md, the follow-up's Pod sessions): only into its
space on the network volume. ``RUNS_DIR``, ``DATA_DIR`` and ``HF_HOME`` must be the space's
``runs``, ``data`` and ``hf``; a hub cache moved elsewhere is refused, and so is another writer or
an output of the session on the volume but outside the space."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Variables that move the hub's cache: unset, or the space's HF_HOME's own
HUB_CACHES = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
# Variables that move another writer's files: unset, off the volume (the container), or in the space
WRITERS = ("XDG_CACHE_HOME", "VLLM_CACHE_ROOT", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR",
           "TRITON_CACHE_DIR", "HF_DATASETS_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "WANDB_DIR",
           "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_CONFIG_DIR")  # fmt: skip


def inside(path: Path | str, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def space_problems(space: Path, outputs: tuple[Path, ...] = (),
                   env: Any = None) -> list[str]:  # fmt: skip
    """Why the session would write on the volume outside ``space``: ``RUNS_DIR``, ``DATA_DIR`` and
    ``HF_HOME`` must be the space's ``runs``, ``data`` and ``hf``; a hub cache moved elsewhere
    (``HUB_CACHES``) is refused, and so is another writer (``WRITERS``) or an output of this
    command on the volume (the space's parent) but outside the space."""
    env = os.environ if env is None else env
    want = {"RUNS_DIR": space / "runs", "DATA_DIR": space / "data", "HF_HOME": space / "hf"}
    problems = []
    for name, path in want.items():
        value = env.get(name)
        if value is None or Path(value).resolve() != path.resolve():
            problems.append(f"{name} is {value}, not {path}")
    for name in HUB_CACHES:
        value = env.get(name)
        if value is not None and Path(value).resolve() != (space / "hf" / "hub").resolve():
            problems.append(f"{name} is {value}: the hub cache is HF_HOME's, {space / 'hf'}/hub")
    volume = space.parent
    for name in WRITERS:
        value = env.get(name)
        if value and inside(value, volume) and not inside(value, space):
            problems.append(f"{name} is {value}: on the volume, outside {space}")
    for path in outputs:
        if not inside(path, space):
            problems.append(f"{path} is outside {space}")
    return problems
