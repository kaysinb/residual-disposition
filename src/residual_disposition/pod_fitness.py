"""A Pod's fitness, checked right after its start and before any session, input or record
(docs/operations.md, the follow-up's confirmations; after the companion project's Pod
procedure, adapted to a network volume). A Pod that fails a check is stopped and replaced: an
infrastructure incident, not an attempt.

- **The volume** is the project's: mounted at its path, holding the marks it is known by (files
  it has held since earlier sessions, each checked by sha256).
- **Write, fsync, read back** on the volume, in the space the session writes: files of random
  bytes, each written, fsynced and closed, then read back equal. A probe still running after
  ``PROBE_TIMEOUT_S`` (against a hung mount, not a speed limit) fails.
- **CPU.** A loop of 3·10⁷ Python additions takes at most 3.0 s.
- **The container's root** is overlay, xfs or ext4.
- **Diagnostics, never a stop:** the volume's file system type (a network volume is fuse) and the
  probe's time.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

CONTAINER_TYPES = ("overlay", "xfs", "ext4")
CPU_ADDITIONS, CPU_MAX_S = 3 * 10**7, 3.0
PROBE_FILES, PROBE_BYTES, PROBE_TIMEOUT_S = 50, 4096, 300

_PROBE = r"""
import json, os, secrets, sys, time
root, files, size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = os.path.join(root, ".fitness-" + secrets.token_hex(4))
os.mkdir(d)
started, written = time.perf_counter(), {}
for i in range(files):
    blob, path = secrets.token_bytes(size), os.path.join(d, f"f{i}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.write(fd, blob)
    os.fsync(fd)
    os.close(fd)
    written[path] = blob
try:
    fd = os.open(d, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    directory = None
except OSError as error:
    directory = repr(error)
equal = all(open(p, "rb").read() == b for p, b in written.items())
ms = (time.perf_counter() - started) * 1000
for p in written:
    os.unlink(p)
os.rmdir(d)
print(json.dumps({"ok": equal, "ms": ms, "files": files, "directory_fsync_error": directory}))
"""


def fs_type(path: Path) -> str:
    """The type of the file system holding ``path``."""
    out = subprocess.run(["findmnt", "-no", "FSTYPE", "--target", str(path)],
                         capture_output=True, text=True, check=True, timeout=30)  # fmt: skip
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""


def cpu_seconds(additions: int = CPU_ADDITIONS) -> float:
    started, total = time.perf_counter(), 0
    for i in range(additions):
        total += i
    return time.perf_counter() - started


def probe(directory: Path, files: int = PROBE_FILES, size: int = PROBE_BYTES,
          timeout: float = PROBE_TIMEOUT_S) -> dict[str, Any]:  # fmt: skip
    """Write, fsync and read back ``files`` files of ``size`` random bytes in a new directory
    under ``directory`` (removed afterwards), in a child process stopped after ``timeout``
    seconds: {"ok", "ms", ...} or {"ok": False, "error"}."""
    command = [sys.executable, "-c", _PROBE, str(directory), str(files), str(size)]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"no answer within {timeout} s"}
    if done.returncode != 0:
        return {"ok": False, "error": done.stderr.strip()[-2000:]}
    return json.loads(done.stdout.strip().splitlines()[-1])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mount_problems(volume: Path, marks: Mapping[str, str], *,
                   mounted: Callable[[Path], bool] = os.path.ismount) -> list[str]:  # fmt: skip
    """Why ``volume`` is not the project's volume: not a mount point, or a mark (a path relative
    to it, with its sha256) missing or different."""
    problems = [] if mounted(volume) else [f"{volume} is not a mount point"]
    if not marks:
        problems.append("no mark to know the volume by")
    for rel, sha in marks.items():
        path = Path(volume) / rel
        if not path.is_file():
            problems.append(f"{path}: missing")
        elif file_sha256(path) != sha:
            problems.append(f"{path}: not the mark {sha[:12]}")
    return problems


def check(volume: Path, space: Path, marks: Mapping[str, str], container: Path = Path("/"), *,
          fs: Callable[[Path], str] = fs_type, cpu: Callable[[], float] = cpu_seconds,
          write: Callable[[Path], dict[str, Any]] = probe,
          mounted: Callable[[Path], bool] = os.path.ismount) -> dict[str, Any]:  # fmt: skip
    """Every check with its value and verdict, the diagnostics, and ``fit`` if all checks pass."""
    mount = mount_problems(volume, marks, mounted=mounted)
    found = write(space) if not mount else {"ok": False, "error": "not the project's volume"}
    container_fs, seconds = fs(container), cpu()
    checks = {
        "volume": {"value": mount, "ok": not mount},
        "write_fsync_read": {"value": found, "ok": bool(found.get("ok"))},
        "cpu_s": {"value": round(seconds, 3), "max": CPU_MAX_S, "ok": seconds <= CPU_MAX_S},
        "container_fs": {"value": container_fs, "ok": container_fs in CONTAINER_TYPES},
    }
    diagnostics = {"volume_fs": fs(volume), "probe_ms": found.get("ms")}
    return {"fit": all(c["ok"] for c in checks.values()), "checks": checks,
            "diagnostics": diagnostics}  # fmt: skip
