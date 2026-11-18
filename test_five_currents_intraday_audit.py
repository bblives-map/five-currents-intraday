import pandas as pd
import pytest

from audit_five_currents_intraday_reconciliation import chicago_close, held_predictors
from five_currents_front_intraday import ROOTS


def test_full_session_close_clock_handles_dst():
    assert chicago_close("2026-10-09") == pd.Timestamp("2026-10-09 21:00:00Z")
    assert chicago_close("2026-12-07") == pd.Timestamp("2026-12-07 22:00:00Z")


def test_held_positions_are_not_new_forecasts_from_previous_close():
    dates = pd.bdate_range("2026-10-01", periods=6)
    weights = pd.DataFrame(0., index=dates, columns=ROOTS)
    weights["CL"] = [.1, .2, .3, .4, .5, .6]
    frame = held_predictors(weights, dates, "2026-10-05", "2026-10-05")
    row = frame.loc[frame.root.eq("CL")].iloc[0]
    assert row.signal_date == pd.Timestamp("2026-10-01")
    assert row.net_magnitude == pytest.approx(.1)


def test_skipped_forecast_does_not_move_execution_earlier():
    dates = pd.bdate_range("2026-10-01", periods=6)
    weights = pd.DataFrame(.1, index=dates.delete(2), columns=ROOTS)
    frame = held_predictors(weights, dates, "2026-10-06", "2026-10-06")
    assert frame.signal_date.eq(pd.Timestamp("2026-10-02")).all()
    assert frame.change_magnitude.eq(0.).all()


def test_no_holding_is_invented_before_next_close_execution():
    dates = pd.bdate_range("2026-10-01", periods=6)
    weights = pd.DataFrame(.1, index=dates, columns=ROOTS)
    assert held_predictors(weights, dates, "2026-10-02", "2026-10-02").empty
