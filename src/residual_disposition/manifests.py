"""Committed manifests: a decision file written after a GPU preparation (the Y₂ choice, a
lineage's k* and step budget, a behavior's Stage-2 decision) is used only if it is committed and
unchanged, so that what the certificate ran with is what the repository records (preregistration
§12.3.6, §12.3.7, §12.7).

In a checkout, git says so. On a Pod the source is a verified snapshot without ``.git``
(``source_snapshot.py``: ``<package>/source`` and its canonical ``<package>/manifest.json``,
whose sha256 is the snapshot id); there the proof is the snapshot's receipt (``snapshot_receipt``,
written on the laptop by ``scripts/snapshot_receipt.py``): the snapshot id, the clean commit it
was taken at, and every snapshot file's sha256 checked against that commit's blob. A file is
committed on the Pod only if its sha256 is the snapshot manifest's and the receipt's, the
manifest is the snapshot's (``SOURCE_SNAPSHOT_ID``) and the receipt names a clean commit
(review s3l, C8 item E)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    """The snapshot manifest's canonical encoding (``source_snapshot.encode``)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def snapshot_manifest(package: Path) -> tuple[str, dict[str, Any]]:
    """A snapshot package's id and manifest, the manifest canonical and of version 1."""
    raw = (Path(package) / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("version") != 1 or raw != canonical(manifest):
        raise ValueError(f"{package}: not a canonical snapshot manifest")
    return _sha256(raw), manifest


def snapshot_receipt(package: Path, repo: Path) -> dict[str, Any]:
    """The receipt of a snapshot taken at ``repo``'s clean HEAD: every file of the snapshot is
    that commit's blob, byte for byte; refused for a dirty checkout or a file the commit lacks or
    holds otherwise."""
    snapshot_id, manifest = snapshot_manifest(package)

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)

    head = git("rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        raise ValueError(f"{repo}: no commit")
    if git("status", "--porcelain").stdout.strip():
        raise ValueError(f"{repo}: the checkout is dirty; a receipt needs a clean commit")
    commit = head.stdout.decode().strip()
    files = {}
    for entry in manifest["files"]:
        if entry.get("kind") != "file":
            continue
        blob = git("show", f"{commit}:{entry['path']}")
        if blob.returncode != 0 or _sha256(blob.stdout) != entry["sha256"]:
            raise ValueError(f"{entry['path']}: the snapshot's file is not commit {commit[:12]}'s")
        files[entry["path"]] = entry["sha256"]
    return {"snapshot_id": snapshot_id, "git_commit": commit, "git_dirty": False, "files": files}


def _snapshot_committed(path: Path, repo: Path, what: str) -> str:
    """The commit at which ``path`` (in the snapshot source ``repo``) is committed, by the
    snapshot's receipt; raises otherwise."""
    receipt_path = os.environ.get("SOURCE_SNAPSHOT_RECEIPT")
    expected = os.environ.get("SOURCE_SNAPSHOT_ID")
    if not receipt_path or not expected:
        raise RuntimeError(f"no git and no verified snapshot: no certificate before {what}")
    receipt = json.loads(Path(receipt_path).read_text())
    snapshot_id, manifest = snapshot_manifest(repo.parent)
    if receipt.get("snapshot_id") != expected or snapshot_id != expected:
        raise RuntimeError(f"the receipt and the source are not snapshot {expected}")
    if receipt.get("git_dirty") is not False or not receipt.get("git_commit"):
        raise RuntimeError(f"snapshot {expected} was not taken at a clean commit")
    rel = str(Path(path).resolve().relative_to(repo.resolve()))
    listed = {e["path"]: e.get("sha256") for e in manifest["files"]}
    sha = _sha256(Path(path).read_bytes())
    if listed.get(rel) != sha or receipt["files"].get(rel) != sha:
        raise RuntimeError(f"{rel} is not the committed file of snapshot {expected}")
    return receipt["git_commit"]


def committed_json(path: Path, what: str, repo: Path | None = None) -> dict[str, Any]:
    """The JSON file at ``path`` with its sha256, only if it is committed and unchanged: in a
    git checkout tracked and clean; in a snapshot without ``.git`` by its receipt
    (``_snapshot_committed``). ``what`` names it in the refusal ("the Y₂ manifest")."""
    from .paths import REPO_ROOT

    repo = REPO_ROOT if repo is None else repo
    if not (Path(repo) / ".git").exists():
        commit = _snapshot_committed(Path(path), Path(repo), what)
        data = json.loads(Path(path).read_text())
        return {**data, "sha256": _sha256(Path(path).read_bytes()), "committed_at": commit}
    rel = str(Path(path).resolve().relative_to(Path(repo).resolve()))
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
    return {**data, "sha256": _sha256(Path(path).read_bytes())}
