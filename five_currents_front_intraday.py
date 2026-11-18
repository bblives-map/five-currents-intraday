"""Daily direction/size versus stored front-month intraday moves. No model fitting."""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOTS = ("CL", "NG", "GC", "SI", "HG")
HOURS = (1, 2, 3, 4, 6, 12)
MICROS = {"CL": "MCL", "NG": "MHNG", "GC": "MGC", "SI": "SIL", "HG": "MHG"}
MICRO_MULTIPLIERS = {"CL": 100., "NG": 1000., "GC": 10., "SI": 1000., "HG": 2500.}
SLIPPAGE_PER_SIDE = {"CL": .01, "NG": .001, "GC": .1, "SI": .005, "HG": .0005}
ROUND_TRIP_FEE = 2.16
BASE_VARIANTS = ("net_sign", "net_magnitude", "change_sign", "change_magnitude")
VARIANTS = (*BASE_VARIANTS, *("reverse_"+name for name in BASE_VARIANTS))
CT = ZoneInfo("America/Chicago")


def entry_time(session):
    stamp = datetime.combine(pd.Timestamp(session).date(), time(1), tzinfo=CT)
    return pd.Timestamp(stamp).tz_convert("UTC")


def daily_predictors(weights, start, end):
    """Use only prior dated targets; repeated targets do not repeat a change trade."""
    weights = weights.reindex(columns=ROOTS).sort_index()
    if weights.index.has_duplicates or weights.isna().any().any():
        raise ValueError("Daily targets must be unique and complete")
    calendar = pd.bdate_range(min(weights.index.min(), pd.Timestamp(start)), end)
    available = weights.reindex(weights.index.union(calendar)).sort_index().ffill().reindex(calendar)
    source_dates = pd.Series(weights.index, index=weights.index).reindex(
        weights.index.union(calendar)).sort_index().ffill().reindex(calendar).shift(1)
    targets = available.shift(1)
    changes = targets-targets.shift(1)
    selected = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    rows = []
    for session in selected:
        if targets.loc[session].isna().any() or changes.loc[session].isna().any():
            continue
        for root in ROOTS:
            value, change = float(targets.at[session, root]), float(changes.at[session, root])
            rows.append({"session": session, "signal_date": source_dates.at[session], "root": root,
                         "net_magnitude": value, "change_magnitude": change,
                         "net_sign": float(np.sign(value) if abs(value) > 1e-12 else 0.)/len(ROOTS),
                         "change_sign": float(np.sign(change) if abs(change) > 1e-12 else 0.)/len(ROOTS)})
    return pd.DataFrame(rows)


def observed(row):
    return bool(row is not None and pd.notna(row.open) and np.isfinite(row.open) and row.open > 0
                and ((pd.notna(row.volume) and row.volume > 0)
                     or (pd.notna(row.bar_count) and row.bar_count > 0)))


def valid_price(row):
    return bool(row is not None and pd.notna(row.open) and np.isfinite(row.open) and row.open > 0)


def same_intraday_instrument(entry, exit_bar):
    # This only prevents a roll inside one intraday trade, not a daily-book match.
    if pd.notna(entry.local_symbol) and pd.notna(exit_bar.local_symbol):
        return str(entry.local_symbol).strip() == str(exit_bar.local_symbol).strip()
    return bool(pd.notna(entry.expiry) and pd.notna(exit_bar.expiry) and entry.expiry == exit_bar.expiry)


def front_month_moves(bars, sessions, *, require_trade=False):
    lookup = {(row.symbol, row.ts): row for row in bars.itertuples()}
    eligible = observed if require_trade else valid_price
    rows = []
    for session in sessions:
        stamp = entry_time(session)
        for root in ROOTS:
            chosen = next((lookup.get((symbol, stamp)) for symbol in (root, MICROS[root])
                           if eligible(lookup.get((symbol, stamp)))), None)
            for hours in HOURS:
                end = stamp+pd.Timedelta(hours=hours)
                if end.date() != stamp.date():
                    raise ValueError("An intraday exit must remain on the same UTC date")
                record = {"session": session, "root": root, "hours": hours,
                          "entry_utc": stamp, "exit_utc": end, "status": "missing entry"}
                if chosen is not None:
                    record.update(symbol=chosen.symbol, local_symbol=chosen.local_symbol,
                                  expiry=chosen.expiry, entry_price=chosen.open)
                    exit_bar = lookup.get((chosen.symbol, end))
                    if not eligible(exit_bar):
                        record["status"] = "missing exit"
                    elif not same_intraday_instrument(chosen, exit_bar):
                        record["status"] = "front series rolled inside holding interval"
                    else:
                        record.update(status="matched", exit_price=exit_bar.open,
                                      price_return=exit_bar.open/chosen.open-1.,
                                      entry_trade_confirmed=observed(chosen), exit_trade_confirmed=observed(exit_bar))
                rows.append(record)
    result = pd.DataFrame(rows)
    for col in ("entry_price", "exit_price", "price_return"):
        if col not in result:
            result[col] = np.nan
    return result


def evaluate(predictors, moves):
    joined = moves.merge(predictors, on=["session", "root"], validate="many_to_one")
    outputs = []
    for variant in VARIANTS:
        reverse = variant.startswith("reverse_")
        base = variant.removeprefix("reverse_")
        weight = joined[base]*(-1. if reverse else 1.)
        inactive = weight.abs().le(1e-12)
        ready = joined.status.eq("matched") | inactive
        gross = (weight*joined.price_return).where(~inactive, 0.).where(ready)
        quantity = weight.abs()/(joined.entry_price*joined.root.map(MICRO_MULTIPLIERS))
        commission = (quantity*ROUND_TRIP_FEE).where(~inactive, 0.).where(ready)
        slippage = (weight.abs()*2.*joined.root.map(SLIPPAGE_PER_SIDE)/joined.entry_price).where(~inactive, 0.).where(ready)
        result = joined[["session", "signal_date", "root", "hours", "status"]].copy()
        result["variant"], result["weight"], result["ready"] = variant, weight, ready
        result["micro_lots_per_unit_capital"] = quantity.where(~inactive, 0.).where(ready)
        result["gross_return"], result["commission_return"] = gross, commission
        result["slippage_return"] = slippage
        result["net_return"] = gross-commission
        result["slippage_net_return"] = gross-commission-slippage
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def stats(values):
    values = pd.Series(values).dropna().astype(float)
    empty = {"days": 0, "sharpe": np.nan, "total_return": np.nan, "max_drawdown": np.nan,
             "annual_vol": np.nan, "win_rate": np.nan, "median_gain": np.nan, "median_loss": np.nan}
    if not len(values):
        return empty
    equity = (1.+values).cumprod()
    std = values.std(ddof=1)
    active = values.loc[values.ne(0.)]
    return {"days": len(values), "sharpe": values.mean()/std*np.sqrt(252.) if std > 0 else np.nan,
            "total_return": equity.iloc[-1]-1.,
            "max_drawdown": (equity/equity.cummax().clip(lower=1.)-1.).min(),
            "annual_vol": std*np.sqrt(252.), "win_rate": active.gt(0.).mean(),
            "median_gain": values.loc[values.gt(0.)].median(),
            "median_loss": values.loc[values.lt(0.)].median()}


def summarize(paths):
    metrics, curves, yearly = [], [], []
    returns = ("gross_return", "commission_return", "slippage_return", "net_return", "slippage_net_return")
    for (variant, hours), group in paths.groupby(["variant", "hours"]):
        for book in ("portfolio", "portfolio_ex_silver", *ROOTS):
            selected = (group if book == "portfolio" else group.loc[group.root.ne("SI")]
                        if book == "portfolio_ex_silver" else group.loc[group.root.eq(book)])
            ready = selected.groupby("session").ready.all()
            totals = selected.groupby("session")[list(returns)].sum(min_count=1)
            totals.loc[~ready, :] = np.nan
            net = stats(totals.net_return)
            metrics.append({"variant": variant, "hours": hours, "book": book,
                            "start": totals.index[ready].min(), "end": totals.index[ready].max(),
                            "possible_days": len(ready), "coverage_pct": ready.mean()*100.,
                            "gross_sharpe": stats(totals.gross_return)["sharpe"],
                            "slippage_net_sharpe": stats(totals.slippage_net_return)["sharpe"],
                            "mean_gross_exposure": selected.groupby("session").weight.apply(lambda v:v.abs().sum()).mean(),
                            **{("net_sharpe" if k == "sharpe" else k): v for k, v in net.items()}})
            totals["variant"], totals["hours"], totals["book"] = variant, hours, book
            totals["equity"] = (1.+totals.net_return).cumprod()*100.
            curves.append(totals.reset_index())
            for year, subset in totals.groupby(totals.index.year):
                yearly.append({"variant": variant, "hours": hours, "book": book, "year": year,
                               **stats(subset.net_return)})
    return pd.DataFrame(metrics), pd.concat(curves, ignore_index=True), pd.DataFrame(yearly)
