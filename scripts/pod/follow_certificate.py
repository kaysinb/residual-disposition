"""One follow-up certificate line, sealed (preregistration §13.8, §13.14; barrier B3), on its Pod.

    RUNS_DIR=<space>/follow/cert/<line>/sealed/runs \\
        uv run python scripts/pod/follow_certificate.py \\
        --line <line> --space <space> --out <space>/follow/cert/<line>

``follow_certificate`` does the work. Before any GPU work: the frozen B3 (``follow_b3``), the
line's runs directory its sealed one, A's reference, b's data at B3's pin, the controls Y and Y₂,
L's and H_s's tensors. Then the k* calibration, its record written only under ``sealed/``; the
envelopes, from the start every ``cadence`` minutes, each holding only its payload's sha256
(``envelopes/<n>.json``; payloads under ``sealed/payloads/``); the certificate starts only after
a receipt (``receipts/<n>.json``) of an envelope whose payload holds the record, with the
calibration recomputed from its cached runs and equal to the record. Each phase runs under §12.7's
one retry per cell (``follow_certificate.with_retry``): a cell's classified training failure is
retried once, a second one ends the phase with its sealed terminal record
(``sealed/terminal_<phase>.json``), and any other failure leaves the line pending. The one open
file is ``status.json`` (running, completed, or failed with an exception's class). A launcher's
``RUN_ID`` is removed first; stdout and stderr go to ``sealed/`` by the launcher.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_certificate as F
from residual_disposition.simulate_continuation import FROZEN

POLL_SECONDS = 60


class Envelopes(threading.Thread):
    """Envelope n at ``t0`` + n · cadence, whatever the calibration's state. The schedule
    (``envelopes/schedule.json``: t0 and the cadence) is written once and resumed: a rerun
    continues the numbering after the last envelope written and keeps the envelopes already
    written. A failure of the worker is kept and raised to the runner (``failed``)."""

    def __init__(self, out: Path, sealed: Path, line: str, b3_sha256: str, cadence: int,
                 now: float) -> None:  # fmt: skip
        super().__init__(daemon=True)
        self.out, self.sealed, self.line, self.b3 = out, sealed, line, b3_sha256
        schedule = out / "envelopes" / "schedule.json"
        if schedule.is_file():
            kept = json.loads(schedule.read_text())
            if int(kept["cadence_minutes"]) != int(cadence):
                raise ValueError("the envelopes' cadence is not the one they started with")
            self.t0 = float(kept["t0"])
        else:
            self.t0 = float(now)
            F.write_once(schedule, json.dumps({"t0": self.t0, "cadence_minutes": cadence}) + "\n")
        self.cadence = int(cadence)
        self.record = None
        self.error: BaseException | None = None
        self.lock = threading.Lock()
        self.written: dict[int, dict] = {}
        self.with_record: set[int] = set()
        for path in sorted((out / "envelopes").glob("[0-9]*.json")):  # a rerun's: kept
            n = int(path.stem)
            self.written[n] = json.loads(path.read_text())
            payload = json.loads((sealed / "payloads" / f"{n}.json").read_text())
            if not isinstance(payload["record"], str):
                self.with_record.add(n)
        self.next = max(self.written, default=-1) + 1

    def write(self, n: int) -> None:
        with self.lock:
            record = self.record
        data = F.payload(self.line, n, self.b3, record)
        F.write_once(self.sealed / "payloads" / f"{n}.json", data.decode())
        env = F.envelope(self.line, n, data)
        F.write_once(self.out / "envelopes" / f"{n}.json", json.dumps(env) + "\n")
        with self.lock:
            self.written[n] = env
            if record is not None:
                self.with_record.add(n)

    def run(self) -> None:
        try:
            n = self.next
            while True:
                due = self.t0 + n * self.cadence * 60
                time.sleep(max(0.0, due - time.time()))
                self.write(n)
                n += 1
        except BaseException as error:  # noqa: BLE001 — raised to the runner by ``failed``
            self.error = error

    def failed(self) -> None:
        """Raise the worker's failure, or its silent end, to the runner."""
        if self.error is not None:
            raise RuntimeError("the envelopes stopped") from self.error
        if not self.is_alive():
            raise RuntimeError("the envelopes stopped")


def receipt(out: Path, envelopes: Envelopes, max_cadences: int = 4) -> int:
    """The number of an envelope with the record whose receipt holds it as committed; waits at
    most ``max_cadences`` cadences after the first envelope with the record, and never past a
    failure of the envelopes."""
    first = None
    while True:
        envelopes.failed()
        with envelopes.lock:
            candidates = sorted(envelopes.with_record)
            written = dict(envelopes.written)
        for n in candidates:
            path = out / "receipts" / f"{n}.json"
            if path.is_file():
                problems = F.receipt_problems(json.loads(path.read_text()), written[n])
                if problems:
                    raise ValueError("; ".join(problems))
                return n
        if candidates:
            first = time.time() if first is None else first
            if time.time() - first > max_cadences * envelopes.cadence * 60:
                raise TimeoutError("no receipt of an envelope with the record")
        time.sleep(POLL_SECONDS)


def cleanup() -> None:
    """Between a classified training failure and its retry: the failed run's memory released."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def keep(path: Path, record: dict) -> None:
    """A sealed record written once; a rerun's must be the same record, never rewritten."""
    text = json.dumps(record, indent=1) + "\n"
    if path.is_file():
        if path.read_text() != text:
            raise ValueError(f"{path.name}: a rerun's record is not the one written")
        return
    F.write_once(path, text)


def take(flag: str) -> str:
    if flag not in sys.argv:
        raise SystemExit(f"{flag} is required")
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return value


def main() -> int:
    os.environ.pop("RUN_ID", None)  # it would name every cell's run directory alike
    name, space, out = take("--line"), Path(take("--space")), Path(take("--out"))
    sealed = F.sealed_dir(space, name)
    if out != space / "follow" / "cert" / name or Path(os.environ.get("RUNS_DIR", "")) != (
            sealed / "runs"):  # fmt: skip
        raise SystemExit("the line's runs go to its sealed directory, its outputs beside it")
    b3 = cr.follow_b3()
    line = F.line_of(b3, name)
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    status = out / "status.json"
    out.mkdir(parents=True, exist_ok=True)
    try:
        problems = F.writer_problems(space, sealed)
        if problems:  # before anything is loaded
            raise SystemExit("; ".join(problems))
        hub = F.listing(Path(os.environ["HF_HOME"]))
        ws = F.seal_workspace(F.workspace(line), sealed)
        F.reference(ws, line)
        F.data(ws, line)
        from residual_disposition import pipeline as P

        P.certificate_controls(ws, rules)
        for state in (line["latent"], line["sham_s"]):
            ws.store.verify(state)
        F.status(status, "running")
        cadence = int(b3["seal"]["envelopes"]["cadence_minutes"])
        envelopes = Envelopes(out, sealed, name, line["binding"]["b3"], cadence, time.time())
        envelopes.start()
        runs = sealed / "runs"
        record, failed = F.with_retry(
            "kstar", lambda: F.calibrate(ws, line, rules, sealed=sealed, before=envelopes.failed),
            runs, line["binding"], cleanup)  # fmt: skip
        if failed:  # §12.7: the phase ends terminally, its record sealed
            keep(sealed / "terminal_kstar.json", F.terminal(line, "kstar", failed))
            F.status(status, "completed")
            return 0
        if F.listing(Path(os.environ["HF_HOME"])) != hub:
            raise RuntimeError("the hub cache changed during the session's loads")
        keep(sealed / "kstar_calibration.json", record)  # a rerun's: the same, never rewritten
        with envelopes.lock:
            envelopes.record = record
        if record["status"] != "resolved":  # §12.3.7: the certification stops there
            F.status(status, "completed")
            return 0
        receipt(out, envelopes)
        if F.calibrate(ws, line, rules, sealed=sealed) != record:  # its cached runs: no GPU work
            raise ValueError("the calibration recomputed is not its record")
        result, failed = F.with_retry(
            "certificate",
            lambda: F.certify(ws, line, rules, record, sealed=sealed, before=envelopes.failed),
            runs, line["binding"], cleanup)  # fmt: skip
        if failed:
            keep(sealed / "terminal_certificate.json", F.terminal(line, "certificate", failed))
            F.status(status, "completed")
            return 0
        if F.listing(Path(os.environ["HF_HOME"])) != hub:
            raise RuntimeError("the hub cache changed during the session's loads")
        keep(sealed / "certificate_blocks.json", result)
        F.status(status, "completed")
        return 0
    except Exception as error:  # noqa: BLE001 — the open status names the class only
        F.status(status, "failed", type(error).__name__)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
