"""Committed manifests: a decision file written after a GPU preparation (the Y₂ choice, a
lineage's k* and step budget, a behavior's Stage-2 decision) is used only if it is committed and
unchanged, so that what the certificate ran with is what the repository records (preregistration
§12.3.6, §12.3.7, §12.7).

In a checkout, git says so. On a Pod the source is a verified snapshot without ``.git``
(``source_snapshot.py``: ``<package>/source`` and its canonical ``<package>/manifest.json``,
whose sha256 is the snapshot id); there the proof is the snapshot's receipt (``snapshot_receipt``,
written on the laptop by ``scripts/snapshot_receipt.py``): the snapshot id, the clean commit it
was taken at, and every snapshot file's sha256 checked against that commit's blob. The receipt is
authentic only by trusted launch metadata: it is checked against git on
the laptop (``scripts/snapshot_receipt.py check``) and sets ``SNAPSHOT_RECEIPT_SHA256`` (the
receipt file's sha256) and ``SOURCE_SNAPSHOT_ID`` in the Pod's environment at its creation; no
value from the package proves anything by itself. A file is committed on the Pod only if the
receipt is that file, its sha256 is the snapshot manifest's and the receipt's, the manifest is the
snapshot's and the receipt names a clean commit. What a check adds to a manifest's data
(``TRANSPORT``) is left out of every comparison of its content."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

TRANSPORT = ("sha256", "committed_at")  # added by committed_json, not part of a manifest


def content(data: Any) -> Any:
    """A manifest's data without what its check added (``TRANSPORT``)."""
    return {k: v for k, v in data.items() if k not in TRANSPORT}


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
    trusted = os.environ.get("SNAPSHOT_RECEIPT_SHA256")  # set at the Pod's creation (s3z2)
    if not receipt_path or not expected or not trusted:
        raise RuntimeError(f"no git and no verified snapshot: no certificate before {what}")
    raw = Path(receipt_path).read_bytes()
    if _sha256(raw) != trusted:
        raise RuntimeError("the snapshot receipt is not the one checked at the Pod's creation")
    receipt = json.loads(raw)
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


def history_path(path: Path, sha256: str) -> Path:
    """Where an earlier version of the manifest ``path`` is kept as snapshot-verifiable
    provenance: ``<dir>/history/<stem>_<sha256[:12]>.json``, its bytes exactly that version's."""
    return Path(path).parent / "history" / f"{Path(path).stem}_{str(sha256)[:12]}.json"


def ancestor_version(path: Path, sha256: str, repo: Path | None = None) -> bytes:
    """An earlier committed version of the manifest ``path`` whose bytes hash to ``sha256``. In a
    git checkout it is the file's own history from HEAD, an ancestor version of the same path;
    a history copy (``history_path``), if present, must be committed and those exact bytes. In a
    snapshot without ``.git`` it is the committed history copy, checked by the snapshot's
    receipt and its bytes hashing to ``sha256``. Raises when there is none."""
    from .paths import REPO_ROOT

    repo = Path(REPO_ROOT if repo is None else repo)
    copy = history_path(path, sha256)
    if (repo / ".git").exists():
        rel = str(Path(path).resolve().relative_to(repo.resolve()))
        log = subprocess.run(["git", "-C", str(repo), "log", "--format=%H", "--", rel],
                             capture_output=True, text=True, check=True)  # fmt: skip
        for commit in log.stdout.split():
            shown = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{rel}"],
                                   capture_output=True)  # fmt: skip
            if shown.returncode != 0 or _sha256(shown.stdout) != sha256:
                continue
            if copy.exists():
                committed_json(copy, "a manifest's history copy", repo)
                if copy.read_bytes() != shown.stdout:
                    raise ValueError(f"{copy.name} is not {rel} as committed at {commit[:12]}")
            return shown.stdout
        raise ValueError(f"{rel}: no ancestor version hashes to {sha256[:12]}")
    if not copy.exists():
        raise RuntimeError(f"no git and no history copy {copy.name}: no earlier version of "
                           f"{Path(path).name}")  # fmt: skip
    committed_json(copy, "a manifest's history copy", repo)
    blob = copy.read_bytes()
    if _sha256(blob) != sha256:
        raise ValueError(f"{copy.name} does not hash to {sha256[:12]}")
    return blob
