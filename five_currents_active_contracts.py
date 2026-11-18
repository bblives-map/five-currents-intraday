"""Select a daily-curve contract from prior liquidity, possibly at ranks 3-6.

This is not the front/second-nearby micro execution route used by the corrected
intraday replay in build_five_currents_frontmonth_prices.py.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from run_five_currents_masked_intraday import contract_month
from us_commodity_execution_overlays import (
    contract_sequence, delivery_period, expiry_session_calendar, last_trade_date,
)


PRODUCTS = ("CL", "NG", "GC", "SI", "HG")
ADV_OBSERVATIONS = 5
MIN_OBSERVATIONS = 3
EXPIRY_BUFFER_SESSIONS = 2
MAX_NEARBY_RANK = 6


def delivery_month(root: str, contract: object, date: pd.Timestamp) -> tuple[str | None, str]:
    actual = contract_month(contract, date)
    if actual is not None:
        return actual, "verified_month"
    if re.fullmatch(fr"{root}_inferred_\d+", str(contract)):
        sequence = contract_sequence(contract)
        if sequence is not None:
            return str(delivery_period(root, sequence)), "inferred_delivery_month"
    return None, "unidentified_contract"


def daily_liquidity(wide: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for root in PRODUCTS:
        for rank in range(1, MAX_NEARBY_RANK + 1):
            ticker = f"{root}{rank} Comdty"
            if ticker not in wide.columns.get_level_values("ticker"):
                continue
            prices = wide[ticker]
            frame = mapping.loc[mapping.ticker.eq(ticker)].copy()
            frame = frame.merge(prices.reset_index(names="date"), on="date", how="left",
                                validate="one_to_one")
            frame["root"], frame["rank"] = root, rank
            frame["close"] = frame.PX_SETTLE.where(frame.PX_SETTLE.gt(0), frame.PX_LAST)
            frames.append(frame[["date", "root", "rank", "contract", "expiry", "close",
                                 "PX_VOLUME", "OPEN_INT"]])
    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates.loc[candidates.contract.notna() & candidates.close.gt(0)].copy()
    if candidates.duplicated(["root", "date", "contract"]).any():
        raise ValueError("A listed contract occurs at multiple ranks on the same source date")
    months = [delivery_month(row.root, row.contract, row.date) for row in candidates.itertuples()]
    candidates["month"] = [month for month, _ in months]
    candidates["identity_basis"] = [basis for _, basis in months]
    candidates["liquidity_identity"] = candidates.month.fillna(candidates.contract)
    candidates = candidates.sort_values(["root", "liquidity_identity", "date"])
    if candidates.duplicated(["root", "date", "liquidity_identity"]).any():
        raise ValueError("Multiple candidates map to the same delivery month on one source date")
    grouped = candidates.groupby(["root", "liquidity_identity"], sort=False)
    candidates["adv"] = grouped.PX_VOLUME.transform(
        lambda volume: volume.shift(1).rolling(ADV_OBSERVATIONS, min_periods=MIN_OBSERVATIONS).mean())
    candidates["adv_observations"] = grouped.PX_VOLUME.transform(
        lambda volume: volume.shift(1).rolling(ADV_OBSERVATIONS, min_periods=1).count())
    candidates["adv_first_date"] = grouped.date.transform(
        lambda dates: dates.shift(ADV_OBSERVATIONS))
    candidates["volume_asof"] = grouped.date.shift(1)
    candidates["prior_oi"] = grouped.OPEN_INT.shift(1)
    calendar = expiry_session_calendar(pd.DatetimeIndex(candidates.date.unique()), boundary_days=500)
    derived = {}
    for root, month in candidates[["root", "month"]].dropna().drop_duplicates().itertuples(index=False):
        derived[(root, month)] = last_trade_date(root, pd.Period(month, freq="M"), calendar)
    candidates["expiry"] = pd.to_datetime(candidates.expiry, errors="coerce")
    fallback = pd.Series([derived.get((root, month), pd.NaT)
                          for root, month in zip(candidates.root, candidates.month)], index=candidates.index)
    candidates["expiry"] = candidates.expiry.fillna(fallback)
    return candidates.reset_index(drop=True)


def select_active_contracts(candidates: pd.DataFrame, sessions: pd.DatetimeIndex,
                            source_calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decide before each session; future price availability never enters selection."""
    calendar = expiry_session_calendar(source_calendar, boundary_days=500)
    source_dates = {root: pd.DatetimeIndex(group.date.unique()).sort_values()
                    for root, group in candidates.groupby("root", sort=False)}
    by_date = {key: group for key, group in candidates.groupby(["root", "date"], sort=False)}
    selected_rows, attempts = [], []
    for session in sessions:
        for root in PRODUCTS:
            pos = int(source_dates[root].searchsorted(session, side="left") - 1)
            if pos < 0:
                continue
            source_date = source_dates[root][pos]
            rows = by_date[(root, source_date)].copy()
            session_position = calendar.searchsorted(session)
            rows["sessions_to_expiry"] = [calendar.searchsorted(expiry) - session_position
                                            if pd.notna(expiry) else np.nan for expiry in rows.expiry]
            rows["eligible"] = (rows.month.notna() & rows.close.gt(0) & rows.adv.gt(0)
                                & rows.adv_observations.ge(MIN_OBSERVATIONS)
                                & rows.sessions_to_expiry.gt(EXPIRY_BUFFER_SESSIONS))
            rows["session"], rows["liquidity_date"] = session, source_date
            rows["selected"] = False
            choices = rows.loc[rows.eligible].sort_values(
                ["adv", "month", "prior_oi", "rank"], ascending=[False, True, False, True],
                na_position="last")
            if choices.empty:
                selected_rows.append({"session": session, "root": root, "liquidity_date": source_date,
                                      "rank": np.nan, "contract": None, "month": None,
                                      "identity_basis": "no_eligible_contract", "selection_status": "no eligible nearby contract"})
            else:
                winner = choices.iloc[0]
                rows.loc[winner.name, "selected"] = True
                selected_rows.append({"session": session, "root": root, "liquidity_date": source_date,
                                      "rank": float(winner["rank"]), "contract": winner.contract,
                                      "month": winner.month, "identity_basis": winner.identity_basis,
                                      "adv": float(winner.adv), "prior_oi": float(winner.prior_oi),
                                      "volume_asof": winner.volume_asof,
                                      "expiry": winner.expiry, "sessions_to_expiry": int(winner.sessions_to_expiry),
                                      "selection_status": "selected"})
            attempts.append(rows)
    selected = pd.DataFrame(selected_rows)
    if not selected.liquidity_date.lt(selected.session).all():
        raise AssertionError("Contract selection uses current-session daily data")
    return selected, pd.concat(attempts, ignore_index=True)
