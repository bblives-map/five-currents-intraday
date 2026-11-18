from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from calculate_five_currents_intraday_positions import active_intervals, planned_entry


AUDIT = Path(__file__).parent / "five_currents_frontmonth_20261025" / "monday_20261026_signal_audit.json"


def test_monday_blocks_reconcile_to_audited_frozen_legs():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    intervals = active_intervals(audit["positions"], pd.Timestamp(audit["session"]))
    assert len(intervals["CL"]) == 2
    assert len(intervals["NG"]) == 1
    assert len(intervals["GC"]) == 1
    assert len(intervals["HG"]) == 2
    first, second = intervals["HG"]
    assert first[0].strftime("%Y-%m-%d %H:%M") == "2026-10-26 17:00"
    assert first[1].strftime("%H:%M") == "19:00"
    assert first[2] == pytest.approx(0.112729287797, abs=1e-9)
    assert second[0] == first[1]
    assert second[2] == pytest.approx(0.000228828823, abs=1e-9)
    assert intervals["GC"][0][2] == pytest.approx(-0.2)


def test_regional_clocks_follow_their_own_dst_transitions():
    before = pd.Timestamp("2026-10-26")
    after = pd.Timestamp("2026-11-02")
    assert planned_entry(before, "NG", "us_open").tz_convert("Asia/Singapore").hour == 21
    assert planned_entry(after, "NG", "us_open").tz_convert("Asia/Singapore").hour == 22
    assert planned_entry(before, "CL", "london_close").tz_convert("Asia/Singapore").hour == 0
    assert planned_entry(after, "CL", "london_close").tz_convert("Asia/Singapore").hour == 0


def test_preview_intraday_uses_accepted_curve_and_keeps_alpha_legs_private(monkeypatch):
    import five_currents_daily.dashboard_worker as worker

    public = {"signal_date": "2026-10-26", "risk": {"ready": True},
              "input_coverage_ready": False, "execution_ready": False,
              "provisional": {"signal_date": "2026-10-26", "risk": {"ready": True},
                              "rank_limits": {"GC": {"normal": 6, "used": 5}},
                              "note": "GC ranks 1-5"}}
    admin = {}
    observed = []

    def build(replay, *, rank_limits):
        observed.append(rank_limits)
        return {"session": "2026-10-27", "signal_date": "2026-10-26", "intervals": []}, [{"family": "private"}]

    monkeypatch.setattr(worker, "build_exposure", build)
    worker.attach_intraday_exposure(public, admin, object(), expected_session="2026-10-26")
    assert observed == [{"CL": 6, "NG": 6, "GC": 5, "SI": 4, "HG": 4}]
    assert public["intraday_exposure"]["ready"]
    assert public["intraday_exposure"]["provisional"]
    assert not public["execution_ready"]
    assert "intraday_alpha_legs" not in public
    assert admin["intraday_alpha_legs"] == [{"family": "private"}]


def test_failed_build_retries_without_new_market_data(monkeypatch):
    import five_currents_daily.dashboard_worker as worker

    now = datetime(2026, 10, 27, 4, tzinfo=timezone.utc)
    public = {"signal_date": "2026-10-26", "risk": {"ready": True}, "input_coverage_ready": True}

    def fail(*args, **kwargs):
        raise ValueError("temporary calculation failure")

    monkeypatch.setattr(worker, "build_exposure", fail)
    worker.attach_intraday_exposure(public, {}, object(), expected_session="2026-10-26", now=now)
    assert not public["intraday_exposure"]["ready"]
    assert not worker.intraday_retry_due(public, now + timedelta(minutes=14))
    assert worker.intraday_retry_due(public, now + timedelta(minutes=15))


def test_stale_close_does_not_create_a_new_intraday_schedule(monkeypatch):
    import five_currents_daily.dashboard_worker as worker

    monkeypatch.setattr(worker, "build_exposure", lambda *args, **kwargs: pytest.fail("stale close built"))
    public = {"signal_date": "2026-10-23", "risk": {"ready": True}, "input_coverage_ready": True}
    worker.attach_intraday_exposure(public, {}, object(), expected_session="2026-10-26")
    assert public["intraday_exposure"]["state"] == "waiting_inputs"
    assert not worker.intraday_retry_due(public)
