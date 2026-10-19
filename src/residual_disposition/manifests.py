"""Committed manifests: a decision file written after a GPU preparation (the Y₂ choice, a
lineage's k* and step budget) is used only if git tracks it unchanged since its commit, so that
what the certificate ran with is what the repository records (preregistration §12.3.6,
§12.3.7)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def committed_json(path: Path, what: str, repo: Path | None = None) -> dict[str, Any]:
    """The JSON file at ``path`` with its sha256, only if it is tracked by git and unchanged since
    its commit; ``what`` names it in the refusal ("the Y₂ manifest")."""
    from .paths import REPO_ROOT

    repo = REPO_ROOT if repo is None else repo
    rel = str(Path(path).resolve().relative_to(repo.resolve()))
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise RuntimeError(f"{rel} is not committed: no certificate before {what}")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", rel],
        capture_output=True,
        text=True,
        check=True,
    )
    if status.stdout.strip():
        raise RuntimeError(f"{rel} differs from its commit: commit it before certification")
    data = json.loads(Path(path).read_text())
    return {**data, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
