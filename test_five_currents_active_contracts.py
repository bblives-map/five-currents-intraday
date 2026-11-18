import numpy as np
import pandas as pd

from five_currents_active_contracts import PRODUCTS, daily_liquidity, select_active_contracts


def fixture_panel():
    dates = pd.bdate_range("2026-10-05", "2026-10-12")
    columns, values, mapping = [], {}, []
    for root in PRODUCTS:
        for rank, code in ((1, "X6"), (2, "Z6")):
            ticker = f"{root}{rank} Comdty"
            for field in ("PX_VOLUME", "OPEN_INT", "PX_SETTLE", "PX_LAST"):
                column = (ticker, field)
                columns.append(column)
                values[column] = np.full(len(dates), 100. if field != "PX_VOLUME" else rank * 10.)
                values[column][-1] = np.nan
            for date in dates:
                mapping.append({"date": date, "ticker": ticker, "contract": root + code,
                                "expiry": pd.Timestamp("2026-11-20" if rank == 1 else "2026-12-20")})
    wide = pd.DataFrame(values, index=dates)
    wide.columns = pd.MultiIndex.from_tuples(columns, names=("ticker", "field"))
    return wide, pd.DataFrame(mapping)


def test_all_products_use_same_liquidity_rule_and_skip_empty_holiday_row():
    wide, mapping = fixture_panel()
    for root in PRODUCTS:
        wide.loc[pd.Timestamp("2026-10-09"), (f"{root}1 Comdty", "PX_VOLUME")] = 1e9
    candidates = daily_liquidity(wide, mapping)
    chosen, _ = select_active_contracts(candidates, pd.to_datetime(["2026-10-13"]),
                                       pd.DatetimeIndex(candidates.date.unique()))
    assert set(chosen.root) == set(PRODUCTS)
    assert chosen["rank"].eq(2).all()
    assert chosen.liquidity_date.eq(pd.Timestamp("2026-10-09")).all()
    assert chosen.volume_asof.eq(pd.Timestamp("2026-10-08")).all()


def test_expiry_buffer_excludes_highest_volume_near_expiry_contract():
    wide, mapping = fixture_panel()
    mapping.loc[mapping.ticker.str.contains("2 Comdty"), "expiry"] = pd.Timestamp("2026-10-14")
    candidates = daily_liquidity(wide, mapping)
    chosen, _ = select_active_contracts(candidates, pd.to_datetime(["2026-10-13"]),
                                       pd.DatetimeIndex(candidates.date.unique()))
    assert chosen["rank"].eq(1).all()


def test_equal_volume_prefers_nearer_delivery_month():
    wide, mapping = fixture_panel()
    for root in PRODUCTS:
        wide[(f"{root}2 Comdty", "PX_VOLUME")] = wide[(f"{root}1 Comdty", "PX_VOLUME")]
    candidates = daily_liquidity(wide, mapping)
    chosen, _ = select_active_contracts(candidates, pd.to_datetime(["2026-10-13"]),
                                       pd.DatetimeIndex(candidates.date.unique()))
    assert chosen["rank"].eq(1).all()


def test_inferred_to_actual_code_change_keeps_same_month_volume_history():
    wide, mapping = fixture_panel()
    rows = mapping.ticker.eq("GC1 Comdty") & mapping.date.lt(pd.Timestamp("2026-10-09"))
    mapping.loc[rows, "contract"] = "GC_inferred_0101"
    mapping.loc[mapping.ticker.eq("GC1 Comdty") & ~rows, "contract"] = "GCV6"
    candidates = daily_liquidity(wide, mapping)
    current = candidates.loc[candidates.root.eq("GC") & candidates["rank"].eq(1)
                             & candidates.date.eq(pd.Timestamp("2026-10-09"))].iloc[0]
    assert current.month == "2026-10"
    assert current.adv_observations == 4
    assert current.adv == 10.
