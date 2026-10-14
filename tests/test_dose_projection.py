"""scripts/dose_projection.py: the next continuation dose's projection from the previous pass."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dose_projection.py"
spec = importlib.util.spec_from_file_location("dose_projection", SCRIPT)
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)

T0 = datetime(2026, 10, 13, 17, 16, 22, tzinfo=UTC)


def cell(runs: Path, name: str, start: datetime, minutes: float, status="completed") -> None:
    d = runs / name
    d.mkdir(parents=True)
    meta = {
        "status": status,
        "started_at": start.isoformat(),
        "finished_at": (start + timedelta(minutes=minutes)).isoformat(),
    }
    (d / "meta.json").write_text(json.dumps(meta))


def test_measured_hours_counts_only_the_passs_completed_cells(tmp_path) -> None:
    since = T0 + timedelta(hours=7.5)
    cell(tmp_path, "20261013-190000-relearn-aaaa-0001", since - timedelta(hours=1), 60)  # before
    cell(tmp_path, "20261014-004900-rates-bbbb-0002", since, 6)
    cell(tmp_path, "20261014-005500-relearn-cccc-0003", since + timedelta(minutes=6), 54)
    cell(tmp_path, "20261014-015000-tool_pd-dddd-0004", since + timedelta(hours=1), 30, "failed")
    cell(tmp_path, "20261014-004842-eeee-0005", since, 240)  # the pass's own run: no kind
    hours, cells, longest, span = dp.measured_hours(tmp_path, since)
    assert cells == 2 and hours == pytest.approx(1.0) and longest == pytest.approx(0.9)
    assert span == pytest.approx(1.0)  # the failed cell's later end does not count


def test_the_next_dose_starts_only_within_the_cap_and_in_time() -> None:
    common = dict(
        started_at=T0,
        prior_hours=0.0,
        cap_hours=17.54,
        margin=timedelta(minutes=25),
        export=timedelta(minutes=30),
        step=timedelta(minutes=25),
    )
    now = T0 + timedelta(hours=11.85)
    # 11.85 + 4.3 = 16.15 h fits the cap, but not the last start, 17.4 h − 25 − 30 − 25 min.
    late = dp.projection(measured=4.3, now=now, deadline=T0 + timedelta(hours=17.4), **common)
    assert late["within_cap"] and not late["in_time"] and not late["start"]
    ok = dp.projection(measured=4.3, now=now, deadline=T0 + timedelta(hours=17.5), **common)
    assert ok["start"]
    # No step starts after 17.5 h − 25 min − 30 min − 25 min = 16.17 h of the batch.
    assert ok["stop_fraction"] == pytest.approx((17.5 - 80 / 60) / 17.54)
    over = dp.projection(measured=6.0, now=now, deadline=T0 + timedelta(hours=17.5), **common)
    assert not over["within_cap"] and not over["start"]
    capped = dp.projection(measured=1.0, now=now, deadline=T0 + timedelta(hours=40), **common)
    assert capped["stop_fraction"] == 1.0


def test_api_committed_counts_spend_and_open_reservations(tmp_path) -> None:
    since = T0 + timedelta(hours=7.5)
    at = lambda h: (T0 + timedelta(hours=h)).isoformat()  # noqa: E731
    events = [
        {"event": "reserve", "id": "a", "usd": 0.03, "at": at(1)},
        {"event": "spend", "id": "a", "usd": 0.5, "at": at(1)},
        {"event": "reserve", "id": "b", "usd": 0.03, "at": at(8)},
        {"event": "spend", "id": "b", "usd": 0.25, "at": at(8)},
        {"event": "reserve", "id": "c", "usd": 0.03, "at": at(9)},
        {"event": "failed", "id": "c", "released": True, "at": at(9)},
        {"event": "reserve", "id": "d", "usd": 0.04, "at": at(9)},  # open: counted
        {"event": "reserve", "id": "e", "usd": 0.5, "at": at(7)},  # open before, settled after
        {"event": "spend", "id": "e", "usd": 0.01, "at": at(8)},
        {"event": "reserve", "id": "f", "usd": 0.06, "at": at(7)},  # still open from before
    ]
    ledger = tmp_path / "adl-api-pod-a.jsonl"
    ledger.write_text("".join(json.dumps(e) + "\n" for e in events))
    total, previous = dp.api_committed(ledger, since)
    # 0.5 + 0.25 + 0.04 + 0.01 + 0.06 now; the pass's part 0.25 + 0.04 + 0.01, not 0.01 − 0.5.
    assert total == pytest.approx(0.86) and previous == pytest.approx(0.30)


def test_main_projects_from_the_watcher_deadline(tmp_path, monkeypatch, capsys) -> None:
    now = datetime.now(UTC)
    since = now - timedelta(hours=4)
    runs = tmp_path / "runs"
    cell(runs, "20261014-004900-rates-bbbb-0002", since + timedelta(minutes=3), 6)
    cell(runs, "20261014-005500-relearn-cccc-0003", since + timedelta(minutes=10), 224)
    state = tmp_path / "state.json"
    argv = ["dose_projection.py", "--runs", str(runs), "--since", since.isoformat()]
    argv += ["--started-at", (now - timedelta(hours=8)).isoformat(), "--cap-hours", "17.54"]
    argv += ["--deadline-state", str(state)]
    for hours, status in ((9, 0), (8, 2)):
        state.write_text(json.dumps({"deadline": (now + timedelta(hours=hours)).timestamp()}))
        monkeypatch.setattr("sys.argv", argv)
        assert dp.main() == status
        record = json.loads(capsys.readouterr().out)
        # The span (3.9 h) is longer than the cells (3.83 h); the longest cell sets the step.
        assert record["measured_hours"] == pytest.approx(3.9, abs=1e-3)
        last = datetime.fromisoformat(record["last_step_start"])
        expected = now + timedelta(hours=hours, minutes=-25 - 30 - 224)
        assert abs((last - expected).total_seconds()) < 1
    # An ADL share that cannot cover the previous pass's API spend plus 25 % stops the dose.
    state.write_text(json.dumps({"deadline": (now + timedelta(hours=9)).timestamp()}))
    ledger = tmp_path / "adl-api-pod-a.jsonl"
    spend = {
        "event": "spend",
        "id": "x",
        "usd": 1.0,
        "at": (since + timedelta(hours=1)).isoformat(),
    }
    ledger.write_text(json.dumps(spend) + "\n")
    for allowance, status in ((2.3, 0), (2.2, 2)):
        monkeypatch.setattr(
            "sys.argv", argv + ["--ledger", str(ledger), "--allowance-usd", str(allowance)]
        )
        assert dp.main() == status
        assert json.loads(capsys.readouterr().out)["api_ok"] == (status == 0)
