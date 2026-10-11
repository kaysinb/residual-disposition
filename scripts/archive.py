"""Copy finalized artifacts to the independent archive and verify them (docs/operations.md).

    uv run python scripts/archive.py <path> [<path> ...]

Each path is a file or directory inside the repository (``runs/...``, ``data/...``) or a source
snapshot under ``~/.local/share/residual-disposition/snapshots``. It is copied to
``ARTIFACT_REMOTE`` at its path relative to the repository (snapshots under ``snapshots/``) with
``gcloud storage cp --no-clobber`` and the service-account key in ``ARTIFACT_GCS_KEY``. That
account can create and read objects but not overwrite or delete them, so an artifact whose
content changes needs a new path.

Afterwards every file's MD5 is compared with its archived object's: a missing object, or one with
other content (a conflict), fails the run. An inventory of the files (path, size, sha256, MD5,
object) is written to ``runs/_archive/<UTC time>.json`` and archived as
``inventory/<UTC time>.json``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from residual_disposition.paths import REPO_ROOT

SNAPSHOTS = Path.home() / ".local" / "share" / "residual-disposition" / "snapshots"


def remote_path(path: Path) -> str:
    path = path.absolute()
    if path.is_relative_to(SNAPSHOTS):
        return "snapshots/" + path.relative_to(SNAPSHOTS).as_posix()
    if path.is_relative_to(REPO_ROOT):
        return path.relative_to(REPO_ROOT).as_posix()
    raise SystemExit(f"{path} is neither in the repository nor a source snapshot")


def digests(path: Path) -> tuple[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode()


def gcloud(*args: str, capture: bool = False) -> str:
    env = dict(os.environ, CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=os.environ["ARTIFACT_GCS_KEY"])
    done = subprocess.run(
        ["gcloud", "storage", *args], env=env, check=True, text=True, capture_output=capture
    )
    return done.stdout if capture else ""


def remote_md5s(prefix: str) -> dict[str, str]:
    """{object URL: base64 MD5} of every object under ``prefix``."""
    out = gcloud("ls", "--json", "-r", f"{prefix}/**", capture=True)
    rows = json.loads(out) if out.strip() else []
    result = {}
    for row in rows:
        meta = row.get("metadata") or {}
        if meta.get("name") and meta.get("md5Hash"):  # row["url"] carries "#<generation>"
            result[f"gs://{meta['bucket']}/{meta['name']}"] = meta["md5Hash"]
    return result


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    remote = os.environ.get("ARTIFACT_REMOTE", "").rstrip("/")
    if not remote.startswith("gs://") or not os.environ.get("ARTIFACT_GCS_KEY"):
        raise SystemExit("Set ARTIFACT_REMOTE (gs://...) and ARTIFACT_GCS_KEY (.env)")
    paths = [Path(p).absolute() for p in sys.argv[1:]]
    files, skipped = [], []
    for path in paths:
        if not path.exists() or path.is_symlink():
            raise SystemExit(f"{path} does not exist or is a symbolic link")
        found = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        # symbolic links (W&B's debug logs in ~/.cache) are not artifacts: not copied, not listed
        skipped += [remote_path(f) for f in found if f.is_symlink()]
        files += [(f, remote_path(f)) for f in found if not f.is_symlink()]
    for path in paths:
        target = f"{remote}/{remote_path(path)}"
        if path.is_dir():
            # cp -r puts the directory under the destination prefix: copy into its parent
            parent = target.rsplit("/", 1)[0]
            gcloud("cp", "-r", "--no-clobber", "--ignore-symlinks", str(path), parent + "/")
        else:
            gcloud("cp", "--no-clobber", str(path), target)
    archived: dict[str, str] = {}
    for prefix in sorted({f"{remote}/{remote_path(p)}" for p in paths if p.is_dir()}):
        archived.update(remote_md5s(prefix))
    for path in paths:
        if path.is_file():
            archived.update(remote_md5s(f"{remote}/{remote_path(path)}".rsplit("/", 1)[0]))
    inventory, problems = [], []
    for local, rel in files:
        sha, md5 = digests(local)
        url = f"{remote}/{rel}"
        if url not in archived:
            problems.append(f"missing: {url}")
        elif archived[url] != md5:
            problems.append(f"conflict (other content already archived): {url}")
        inventory.append(
            {"path": rel, "size": local.stat().st_size, "sha256": sha, "md5": md5, "object": url}
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "created_at": stamp,
        "remote": remote,
        "files": inventory,
        "skipped_symlinks": skipped,
        "problems": problems,
    }
    local_inventory = REPO_ROOT / "runs" / "_archive" / f"{stamp}.json"
    local_inventory.parent.mkdir(parents=True, exist_ok=True)
    local_inventory.write_text(json.dumps(record, indent=2) + "\n")
    gcloud("cp", "--no-clobber", str(local_inventory), f"{remote}/inventory/{stamp}.json")
    print(
        f"{len(inventory)} files, {sum(f['size'] for f in inventory) / 1e6:.1f} MB; inventory "
        f"{local_inventory.relative_to(REPO_ROOT)}"
    )
    if problems:
        raise SystemExit("\n".join(problems))
    print("All files archived with matching MD5.")


if __name__ == "__main__":
    main()
