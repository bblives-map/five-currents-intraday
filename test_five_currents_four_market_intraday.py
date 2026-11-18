import numpy as np
import pandas as pd
import pytest

from run_five_currents_four_market_intraday import MARKETS, RETURNS, daily_portfolio


def sample():
    return pd.DataFrame([{"variant": "net_magnitude", "hours": 1,
                          "session": pd.Timestamp("2026-10-05"), "root": root,
                          "ready": True, **{column: .01 for column in RETURNS}}
                         for root in MARKETS])


def test_missing_active_price_does_not_become_cash():
    paths = sample()
    paths.loc[0, "ready"] = False
    paths.loc[0, list(RETURNS)] = np.nan
    result = daily_portfolio(paths).iloc[0]
    assert not result.ready
    assert pd.isna(result.net_return)


def test_four_contributions_are_added_without_reweighting():
    result = daily_portfolio(sample()).iloc[0]
    assert result.ready
    assert result.net_return == pytest.approx(.04)


def test_absent_market_is_not_silently_omitted():
    result = daily_portfolio(sample().iloc[:-1]).iloc[0]
    assert not result.ready
    assert pd.isna(result.net_return)


def test_duplicate_asset_fails_closed():
    paths = sample()
    with pytest.raises(ValueError, match="Duplicate"):
        daily_portfolio(pd.concat([paths, paths.iloc[[0]]], ignore_index=True))
