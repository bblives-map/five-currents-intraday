import numpy as np
import pandas as pd
import pytest

from five_currents_front_intraday import (
    ROOTS, HOURS, daily_predictors, entry_time, front_month_moves, evaluate, summarize,
)


def weights():
    dates = pd.bdate_range("2026-10-01", periods=5)
    frame = pd.DataFrame(0., index=dates, columns=ROOTS)
    frame["CL"] = [.1, .2, -.1, .15, .3]
    return frame


def bars(session, missing=()):
    entry = entry_time(session)
    return pd.DataFrame([{"symbol": "MCL", "ts": entry+pd.Timedelta(hours=h),
                          "local_symbol": "MCLX6", "expiry": 202611,
                          "open": 70.+h, "volume": 10., "bar_count": 2}
                         for h in (0, *HOURS) if h not in missing])


def test_forecasts_are_strictly_prior_day_and_changes_are_not_forward_differences():
    frame = weights()
    p = daily_predictors(frame, "2026-10-05", "2026-10-05")
    row = p.loc[p.root.eq("CL")].iloc[0]
    assert row.signal_date == pd.Timestamp("2026-10-02")
    assert row.net_magnitude == .2
    assert row.change_magnitude == pytest.approx(.1)
    assert row.net_sign == .2
    assert row.change_sign == .2


def test_weekday_without_a_new_target_does_not_retrade_yesterdays_change():
    frame = weights().drop(pd.Timestamp("2026-10-05"))
    p = daily_predictors(frame, "2026-10-06", "2026-10-06")
    row = p.loc[p.root.eq("CL")].iloc[0]
    assert row.signal_date == pd.Timestamp("2026-10-02")
    assert row.change_magnitude == 0.


def test_entry_clock_handles_both_us_offsets():
    assert entry_time("2026-10-05").hour == 6
    assert entry_time("2026-12-07").hour == 7


def test_exit_is_same_day_and_does_not_require_any_daily_backtest_contract_identity():
    moves = front_month_moves(bars("2026-10-05"), pd.to_datetime(["2026-10-05"]))
    cl = moves.loc[moves.root.eq("CL")]
    assert cl.status.eq("matched").all()
    assert cl.exit_utc.dt.date.eq(cl.entry_utc.dt.date).all()
    np.testing.assert_allclose(cl.price_return, np.array(HOURS)/70.)


def test_missing_one_exit_does_not_disqualify_other_horizons():
    p = daily_predictors(weights(), "2026-10-05", "2026-10-05")
    moves = front_month_moves(bars("2026-10-05", missing=(3,)), pd.to_datetime(["2026-10-05"]))
    metrics, _, _ = summarize(evaluate(p, moves))
    selected = metrics.loc[metrics.book.eq("portfolio") & metrics.variant.eq("net_magnitude")].set_index("hours")
    assert selected.at[3, "days"] == 0
    assert selected.at[1, "days"] == 1
    assert selected.at[12, "days"] == 1


def test_reverse_negates_gross_not_net_and_keeps_same_micro_fee_without_extra_factor():
    frame = weights()
    frame["CL"] = 7000.  # One MCL lot per $1 capital at a $70 entry price.
    p = daily_predictors(frame, "2026-10-05", "2026-10-05")
    moves = front_month_moves(bars("2026-10-05"), pd.to_datetime(["2026-10-05"]))
    paths = evaluate(p, moves)
    pair = paths.loc[paths.root.eq("CL") & paths.hours.eq(1)].set_index("variant")
    original, reverse = pair.loc["net_magnitude"], pair.loc["reverse_net_magnitude"]
    assert original.commission_return == pytest.approx(2.16)
    assert original.commission_return == reverse.commission_return
    assert reverse.gross_return == -original.gross_return
    assert reverse.net_return == pytest.approx(-original.gross_return-2.16)


def test_front_series_roll_within_trade_is_not_a_price_return():
    b = bars("2026-10-05")
    b.loc[b.ts.eq(entry_time("2026-10-05")+pd.Timedelta(hours=1)), "local_symbol"] = "MCLZ6"
    moves = front_month_moves(b, pd.to_datetime(["2026-10-05"]))
    assert moves.loc[moves.root.eq("CL") & moves.hours.eq(1), "status"].iloc[0] == "front series rolled inside holding interval"


def test_entry_source_is_not_selected_based_on_future_exit_availability():
    micro = bars("2026-10-05")
    standard = micro.iloc[[0]].copy()
    standard["symbol"], standard["local_symbol"] = "CL", "CLX6"
    moves = front_month_moves(pd.concat([micro, standard]), pd.to_datetime(["2026-10-05"]))
    cl = moves.loc[moves.root.eq("CL")]
    assert cl.symbol.eq("CL").all()
    assert cl.status.eq("missing exit").all()


def test_price_prediction_does_not_drop_existing_price_marks_due_to_zero_volume():
    b = bars("2026-10-05")
    b["volume"], b["bar_count"] = 0., 0
    dates = pd.to_datetime(["2026-10-05"])
    primary = front_month_moves(b, dates)
    sensitivity = front_month_moves(b, dates, require_trade=True)
    assert primary.loc[primary.root.eq("CL"), "status"].eq("matched").all()
    assert not sensitivity.loc[sensitivity.root.eq("CL"), "status"].eq("matched").any()


def test_sign_does_not_turn_float_roundoff_into_a_full_sized_trade():
    frame = weights()
    frame["CL"] = [0., 1e-16, 0., 0., 0.]
    p = daily_predictors(frame, "2026-10-05", "2026-10-05")
    cl = p.loc[p.root.eq("CL")].iloc[0]
    assert cl.net_sign == 0. and cl.change_sign == 0.
