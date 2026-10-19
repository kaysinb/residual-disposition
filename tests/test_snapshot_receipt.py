"""Committed manifests on a Pod (C8 item E): a source snapshot without ``.git`` proves a file
committed only by its receipt, written on the laptop from the clean commit the snapshot was taken
at, every snapshot file that commit's blob; on the Pod the file's sha256 must be the snapshot
manifest's and the receipt's, the manifest the verified snapshot's and the commit clean."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys

import pytest

from residual_disposition import manifests as M
from residual_disposition.paths import REPO_ROOT


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def a_repo(root):
    repo = root / "repo"
    (repo / "manifests").mkdir(parents=True)
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "t")
    (repo / "manifests" / "kstar_example_link_400.json").write_text(json.dumps({"budget": 300}))
    (repo / "code.py").write_text("x = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "c")
    return repo


def a_package(root, repo):
    """The snapshot package of ``repo``'s files, as source_snapshot.py prepares it."""
    package = root / "store" / "pending"
    source = package / "source"
    files = []
    for path in sorted(p for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel = path.relative_to(repo).as_posix()
        (source / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, source / rel)
        files.append({"path": rel, "mode": 0o644, "kind": "file",
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})  # fmt: skip
    raw = M.canonical({"version": 1, "files": files})
    (package / "manifest.json").write_bytes(raw)
    final = package.parent / hashlib.sha256(raw).hexdigest()
    package.rename(final)
    return final


@pytest.fixture
def pod(tmp_path, monkeypatch):
    repo = a_repo(tmp_path)
    package = a_package(tmp_path, repo)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(M.snapshot_receipt(package, repo)))
    monkeypatch.setenv("SOURCE_SNAPSHOT_ID", package.name)
    monkeypatch.setenv("SOURCE_SNAPSHOT_RECEIPT", str(receipt))
    source = package / "source"
    return source, source / "manifests" / "kstar_example_link_400.json", receipt, repo


def test_a_committed_manifest_is_accepted_on_the_pod_by_its_receipt(pod):
    source, path, receipt, repo = pod
    assert not (source / ".git").exists()
    data = M.committed_json(path, "the k* manifest", source)
    commit = git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    assert data["budget"] == 300 and data["committed_at"] == commit


def test_a_changed_file_is_not_the_snapshot_s(pod):
    source, path, receipt, repo = pod
    path.write_text(json.dumps({"budget": 600}))
    with pytest.raises(RuntimeError, match="is not the committed file of snapshot"):
        M.committed_json(path, "the k* manifest", source)


def test_a_receipt_of_another_snapshot_or_a_dirty_commit_is_refused(pod, monkeypatch, tmp_path):
    source, path, receipt, repo = pod
    data = json.loads(receipt.read_text())
    receipt.write_text(json.dumps({**data, "git_dirty": True}))
    with pytest.raises(RuntimeError, match="not taken at a clean commit"):
        M.committed_json(path, "the k* manifest", source)
    receipt.write_text(json.dumps({**data, "snapshot_id": "0" * 64}))
    with pytest.raises(RuntimeError, match="not snapshot"):
        M.committed_json(path, "the k* manifest", source)
    receipt.write_text(json.dumps(data))
    monkeypatch.setenv("SOURCE_SNAPSHOT_ID", "1" * 64)
    with pytest.raises(RuntimeError, match="not snapshot"):
        M.committed_json(path, "the k* manifest", source)
    monkeypatch.delenv("SOURCE_SNAPSHOT_RECEIPT")
    with pytest.raises(RuntimeError, match="no git and no verified snapshot"):
        M.committed_json(path, "the k* manifest", source)


def test_a_file_the_receipt_does_not_list_is_refused(pod):
    source, path, receipt, repo = pod
    data = json.loads(receipt.read_text())
    files = {k: v for k, v in data["files"].items() if not k.startswith("manifests/")}
    receipt.write_text(json.dumps({**data, "files": files}))
    with pytest.raises(RuntimeError, match="is not the committed file"):
        M.committed_json(path, "the k* manifest", source)


def test_no_receipt_for_a_dirty_checkout_or_a_file_off_the_commit(tmp_path):
    repo = a_repo(tmp_path)
    (repo / "code.py").write_text("x = 2\n")  # dirty
    package = a_package(tmp_path, repo)
    with pytest.raises(ValueError, match="dirty"):
        M.snapshot_receipt(package, repo)
    git(repo, "checkout", "--", "code.py")
    # the snapshot holds the uncommitted x = 2: not the commit's blob
    with pytest.raises(ValueError, match="code.py: the snapshot's file is not commit"):
        M.snapshot_receipt(package, repo)
    raw = (package / "manifest.json").read_bytes()
    (package / "manifest.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="not a canonical snapshot manifest"):
        M.snapshot_receipt(package, repo)


def test_the_receipt_script(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "snapshot_receipt", REPO_ROOT / "scripts" / "snapshot_receipt.py"
    )
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    spec.loader.exec_module(script)
    repo = a_repo(tmp_path)
    package = a_package(tmp_path, repo)
    assert script.main([str(package), "--repo", str(repo)]) == 0
    written = json.loads((package.parent / f"{package.name}.receipt.json").read_text())
    assert written["snapshot_id"] == package.name and written["git_dirty"] is False
    assert set(written["files"]) == {"code.py", "manifests/kstar_example_link_400.json"}


def test_a_checkout_still_uses_git(tmp_path):
    repo = a_repo(tmp_path)
    path = repo / "manifests" / "kstar_example_link_400.json"
    assert M.committed_json(path, "the k* manifest", repo)["budget"] == 300
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="differs from its commit"):
        M.committed_json(path, "the k* manifest", repo)
