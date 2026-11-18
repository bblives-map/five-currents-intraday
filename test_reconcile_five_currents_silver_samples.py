import pandas as pd
import pytest

from five_currents_front_intraday import entry_time, front_month_moves
from reconcile_five_currents_silver_samples import silver_proxy_endpoints


def bar(stamp, month="SILZ6", source="bars_1m_2m", price=30., symbol="SIL"):
    return {"symbol": symbol, "ts": stamp, "open": price, "local_symbol": month,
            "expiry": 20261229, "volume": 5., "bar_count": 1, "source_table": source}


def test_second_archive_is_explicit_proxy_and_other_markets_are_unchanged():
    days = pd.to_datetime(["2026-10-05"])
    stamp = entry_time(days[0])
    original = pd.DataFrame([bar(stamp, source="bars_1m"), bar(stamp, symbol="CL")])
    archive = pd.DataFrame([bar(stamp, month="SILF7")])
    result = silver_proxy_endpoints(original, archive, days)
    assert result.loc[result.symbol.eq("SIL"), "local_symbol"].item() == "SILF7"
    assert result.loc[result.symbol.eq("CL"), "open"].item() == 30.
    assert "proxy" in result.loc[result.symbol.eq("SIL"), "source_method"].item()
    assert original.loc[original.symbol.eq("SIL"), "local_symbol"].item() == "SILZ6"


def test_nearby_future_or_prior_prices_are_not_silently_used():
    days = pd.to_datetime(["2026-10-05"])
    stamp = entry_time(days[0])
    original = pd.DataFrame([bar(stamp)])
    archive = pd.DataFrame([bar(stamp-pd.Timedelta(minutes=1)), bar(stamp+pd.Timedelta(minutes=1))])
    assert silver_proxy_endpoints(original, archive, days).empty


def test_missing_proxy_exit_does_not_route_back_to_front_month():
    days = pd.to_datetime(["2026-10-05"])
    stamp = entry_time(days[0])
    original = pd.DataFrame([bar(stamp), bar(stamp+pd.Timedelta(hours=1))])
    archive = pd.DataFrame([bar(stamp)])
    result = front_month_moves(silver_proxy_endpoints(original, archive, days), days)
    assert result.loc[result.root.eq("SI") & result.hours.eq(1), "status"].item() == "missing exit"


def test_month_change_during_hold_is_rejected():
    days = pd.to_datetime(["2026-10-05"])
    stamp = entry_time(days[0])
    original = pd.DataFrame([bar(stamp)])
    archive = pd.DataFrame([bar(stamp), bar(stamp+pd.Timedelta(hours=1), month="SILF7")])
    result = front_month_moves(silver_proxy_endpoints(original, archive, days), days)
    assert result.loc[result.root.eq("SI") & result.hours.eq(1), "status"].item() == "front series rolled inside holding interval"


def test_ambiguous_archive_prices_fail_closed():
    days = pd.to_datetime(["2026-10-05"])
    stamp = entry_time(days[0])
    with pytest.raises(ValueError, match="Ambiguous"):
        silver_proxy_endpoints(pd.DataFrame([bar(stamp)]), pd.DataFrame([bar(stamp), bar(stamp)]), days)
