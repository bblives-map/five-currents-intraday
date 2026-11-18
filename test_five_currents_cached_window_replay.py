from types import SimpleNamespace

import pandas as pd

from run_five_currents_cached_window_replay import bar_index, choose_entry, choose_exit


def _bar(table, symbol, ts, local_symbol, open_price=100.0):
    return dict(table=table, symbol=symbol, ts=pd.Timestamp(ts),
                local_symbol=local_symbol, open=open_price, volume=1,
                bar_count=1, expiry=None, source="test")


def test_known_month_can_be_in_other_stored_route():
    stamp = pd.Timestamp("2026-10-09 06:00:00+00:00")
    bars = pd.DataFrame([
        _bar("bars_1m", "MHG", stamp, "MHGX6"),
        _bar("bars_1m_2m", "MHG", stamp + pd.Timedelta(minutes=2), "MHGZ6"),
    ])
    target = SimpleNamespace(root="HG", rank=1, month="2026-12")
    result = choose_entry(target, stamp, bar_index(bars))
    assert result.local_symbol == "MHGZ6"
    assert result.table == "bars_1m_2m"


def test_proxy_entry_does_not_look_at_exit():
    stamp = pd.Timestamp("2025-10-09 06:00:00+00:00")
    exit_stamp = stamp + pd.Timedelta(hours=1)
    bars = pd.DataFrame([
        _bar("bars_1m", "MGC", stamp + pd.Timedelta(minutes=1), "MGCZ5"),
        _bar("bars_1m", "MGC", stamp + pd.Timedelta(minutes=2), "MGCG6"),
        _bar("bars_1m", "MGC", exit_stamp, "MGCG6"),
    ])
    index = bar_index(bars)
    entry = choose_entry(SimpleNamespace(root="GC", rank=1, month=None), stamp, index)
    assert entry.local_symbol == "MGCZ5"
    assert choose_exit(entry, exit_stamp, index) is None


def test_exit_can_move_ten_minutes_but_not_change_contract():
    stamp = pd.Timestamp("2026-10-09 06:00:00+00:00")
    exit_stamp = stamp + pd.Timedelta(hours=1)
    bars = pd.DataFrame([
        _bar("bars_1m", "MGC", stamp, "MGCZ6"),
        _bar("bars_1m", "MGC", exit_stamp + pd.Timedelta(minutes=1), "MGCG7"),
        _bar("bars_1m", "MGC", exit_stamp + pd.Timedelta(minutes=10), "MGCZ6"),
        _bar("bars_1m", "MGC", exit_stamp + pd.Timedelta(minutes=11), "MGCZ6"),
    ])
    index = bar_index(bars)
    entry = choose_entry(SimpleNamespace(root="GC", rank=1, month="2026-12"), stamp, index)
    exit_bar = choose_exit(entry, exit_stamp, index)
    assert exit_bar.local_symbol == "MGCZ6"
    assert exit_bar.ts == exit_stamp + pd.Timedelta(minutes=10)


def test_stored_zero_volume_mark_is_retained_but_not_called_a_trade():
    stamp = pd.Timestamp("2026-10-09 06:00:00+00:00")
    bars = pd.DataFrame([_bar("bars_1m", "MGC", stamp, "MGCZ6")])
    bars.loc[0, ["volume", "bar_count"]] = 0
    entry = choose_entry(SimpleNamespace(root="GC", rank=1, month="2026-12"),
                         stamp, bar_index(bars))
    assert entry is not None
    assert entry.volume == 0
