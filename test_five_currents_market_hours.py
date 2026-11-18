from __future__ import annotations

import numpy as np
import pandas as pd

from run_five_currents_market_hours import (
    CLOCKS, clock_entry, clock_moves, common_clock_metrics, crosses_maintenance,
    closed_exit_reason,
)


def test_clock_entries_follow_local_daylight_saving() -> None:
    assert clock_entry("2026-01-15", "london_open") == pd.Timestamp("2026-01-15 08:00Z")
    assert clock_entry("2026-07-15", "london_open") == pd.Timestamp("2026-07-15 07:00Z")
    assert clock_entry("2026-01-15", "london_close") == pd.Timestamp("2026-01-15 16:30Z")
    assert clock_entry("2026-07-15", "london_close") == pd.Timestamp("2026-07-15 15:30Z")
    assert clock_entry("2026-01-15", "us_open") == pd.Timestamp("2026-01-15 14:30Z")
    assert clock_entry("2026-07-15", "us_open") == pd.Timestamp("2026-07-15 13:30Z")
    assert tuple(CLOCKS) == ("london_open", "london_close", "us_open")


def test_maintenance_crossing_is_flagged_without_changing_exit() -> None:
    us = clock_entry("2026-07-15", "us_open")
    assert not crosses_maintenance(us, us + pd.Timedelta(hours=4))
    assert crosses_maintenance(us, us + pd.Timedelta(hours=12))
    london = clock_entry("2026-07-15", "london_close")
    assert crosses_maintenance(london, london + pd.Timedelta(hours=12))
    assert closed_exit_reason(london + pd.Timedelta(hours=6)) == "exit during maintenance"
    assert closed_exit_reason(us + pd.Timedelta(hours=12)) is None
    friday = clock_entry("2026-07-17", "us_open")
    assert closed_exit_reason(friday + pd.Timedelta(hours=12)) == "exit after weekly close"


def test_exact_minute_exit_requires_same_instrument() -> None:
    session = pd.DatetimeIndex(["2026-07-15"])
    entry = clock_entry(session[0], "london_open")
    bars = pd.DataFrame([
        {"symbol": "MCL", "ts": entry, "open": 100., "volume": 10., "bar_count": 1.,
         "local_symbol": "MCLQ6", "expiry": 20260820},
        {"symbol": "MCL", "ts": entry + pd.Timedelta(hours=1), "open": 101., "volume": 10.,
         "bar_count": 1., "local_symbol": "MCLQ6", "expiry": 20260820},
        {"symbol": "MCL", "ts": entry + pd.Timedelta(hours=2), "open": 102., "volume": 10.,
         "bar_count": 1., "local_symbol": "MCLU6", "expiry": 20260920},
    ])
    moves = clock_moves(bars, session)
    one = moves.loc[moves.clock.eq("london_open") & moves.root.eq("CL") & moves.hours.eq(1)].iloc[0]
    two = moves.loc[moves.clock.eq("london_open") & moves.root.eq("CL") & moves.hours.eq(2)].iloc[0]
    assert one.status == "matched"
    assert np.isclose(one.price_return, .01)
    assert bool(one.entry_trade_confirmed) and bool(one.exit_trade_confirmed)
    assert two.status == "contract changed during hold"
    assert np.isnan(two.price_return)


def test_common_clock_metrics_exclude_missing_days_not_impute_zero() -> None:
    records = []
    for day, london, close, us in (
        ("2026-07-15", .01, .02, .03),
        ("2026-07-16", .01, np.nan, .03),
        ("2026-07-17", -.01, -.02, -.03),
    ):
        for clock, value in (("london_open", london), ("london_close", close), ("us_open", us)):
            records.append({"route": "front", "policy": "lagged_volume_oi", "signal": "portfolio_net",
                            "book": "four", "variant": "net_magnitude", "hours": 1,
                            "clock": clock, "session": pd.Timestamp(day), "net_return": value,
                            "slippage_net_return": value - .001 if np.isfinite(value) else np.nan})
    metrics = common_clock_metrics(pd.DataFrame(records))
    net = metrics.loc[metrics.cost_basis.eq("net_return")]
    assert len(net) == 3
    assert net.days.eq(2).all()
    assert np.isclose(net.loc[net.clock.eq("london_open"), "total_return"].iloc[0], -.0001)
