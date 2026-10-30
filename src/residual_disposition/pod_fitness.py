"""A Pod's fitness, checked right after its start and before any session, input or record
(docs/operations.md, the follow-up's confirmations; the checks of the companion project's Pod
procedure). A Pod that fails any check is stopped and replaced: an infrastructure incident, not
an attempt.

- **File systems.** The volume's (``findmnt -no FSTYPE --target``) must be xfs or ext4: a fuse
  volume ignores the umask and its metadata is slow. The container's root must be overlay, xfs
  or ext4.
- **CPU.** A loop of 3·10⁷ Python additions takes at most 3.0 s.
- **Disk.** 200 small files, each written and fsynced under the volume, take at most 2 s.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

VOLUME_TYPES = ("xfs", "ext4")
CONTAINER_TYPES = ("overlay", "xfs", "ext4")
CPU_ADDITIONS, CPU_MAX_S = 3 * 10**7, 3.0
FSYNC_FILES, FSYNC_MAX_MS = 200, 2000.0


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


def fsync_ms(directory: Path, files: int = FSYNC_FILES) -> float:
    """Milliseconds to write and fsync ``files`` files of 100 bytes in a new directory under
    ``directory`` (removed afterwards)."""
    root = tempfile.mkdtemp(dir=directory)
    try:
        started = time.perf_counter()
        for i in range(files):
            fd = os.open(os.path.join(root, f"f{i}"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            os.write(fd, b"x" * 100)
            os.fsync(fd)
            os.close(fd)
        return (time.perf_counter() - started) * 1000
    finally:
        shutil.rmtree(root)


def check(volume: Path, container: Path = Path("/"), *,
          fs: Callable[[Path], str] = fs_type, cpu: Callable[[], float] = cpu_seconds,
          disk: Callable[[Path], float] = fsync_ms) -> dict[str, Any]:  # fmt: skip
    """Every check's value and verdict, and ``fit`` if all pass."""
    volume_fs, container_fs = fs(volume), fs(container)
    seconds, ms = cpu(), disk(volume)
    checks = {
        "volume_fs": {"value": volume_fs, "ok": volume_fs in VOLUME_TYPES},
        "container_fs": {"value": container_fs, "ok": container_fs in CONTAINER_TYPES},
        "cpu_s": {"value": round(seconds, 3), "max": CPU_MAX_S, "ok": seconds <= CPU_MAX_S},
        "fsync_ms": {"value": round(ms, 1), "max": FSYNC_MAX_MS, "ok": ms <= FSYNC_MAX_MS},
    }
    return {"fit": all(c["ok"] for c in checks.values()), "checks": checks}
