"""Archived grouped-clock research on the earlier sampled price cache.

For front/second-nearby execution, use build_five_currents_frontmonth_prices.py
and the five_currents_frontmonth_20261025 replay instead of build_prices here.
"""
from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from five_currents_execution_timing import (
    ROOTS, HOLDS, OFFSETS, PriceBook, within_session, asset_pnl, summarize_matrix,
)
from five_currents_front_intraday import MICROS


SOURCE = Path(__import__("os").environ.get("FIVE_CURRENTS_DATA_ROOT", str(Path(__file__).resolve().parent))) / "five_currents_execution_timing_20261023"
OUTPUT = Path(__import__("os").environ.get("FIVE_CURRENTS_DATA_ROOT", str(Path(__file__).resolve().parent))) / "five_currents_group_clock_20261024"
GROUPS = {"CL_NG": (0, 1), "GC_SI": (2, 3), "HG": (4,)}
CLOCKS = ("asia_open", "asia_close", "london_open", "london_close", "us_open", "us_close")
HOLD_MINUTES = HOLDS[:8]
DELAYS = (0, 5, 15, 30)
BASES = ("gross", "gross_change", "net", "net_change", "net_change_of_change")
STYLES = ("magnitude", "sign", "reverse_magnitude", "reverse_sign")
CLASSES = tuple(f"{base}__{style}" for base in BASES for style in STYLES)


def asia_time(session, hour_minute):
    local = datetime.combine(pd.Timestamp(session).date(), time.fromisoformat(hour_minute),
                             tzinfo=ZoneInfo("Asia/Singapore"))
    return pd.Timestamp(local).tz_convert("UTC")


def asia_pair(book, target, clock, hold, offset, asia_open, asia_close):
    planned = asia_time(target.session, asia_open if clock == "asia_open" else asia_close)
    planned += pd.Timedelta(minutes=offset)
    row = {"session": target.session, "root": target.root, "clock": clock,
           "hold": hold, "offset": offset, "month": target.month,
           "identity_basis": target.identity_basis, "planned_entry": planned,
           "status": "missing entry"}
    if not within_session(planned, target.session):
        row["status"] = "outside session"
        return row
    entry = book.find(target.root, target.month, planned)
    if entry is None:
        return row
    end = entry.ts+pd.Timedelta(minutes=hold)
    row.update(entry_ts=entry.ts, entry_price=entry.price, symbol=entry.symbol,
               planned_exit=end, entry_table=entry.table)
    if not within_session(end, target.session):
        row["status"] = "exit outside session"
        return row
    exit_fill = book.find(target.root, target.month, end, symbol=entry.symbol)
    if exit_fill is None:
        row["status"] = "missing same-contract exit"
    elif exit_fill.ts <= entry.ts or not within_session(exit_fill.ts, target.session):
        row["status"] = "invalid exit"
    else:
        row.update(status="matched", exit_ts=exit_fill.ts, exit_price=exit_fill.price,
                   exit_table=exit_fill.table, price_return=exit_fill.price/entry.price-1.)
    return row


def build_prices(source, output, *, asia_open="09:00", asia_close="17:00"):
    destination = output / "clock_prices.npz"
    if destination.exists():
        with np.load(destination) as saved:
            return {name: saved[name] for name in saved.files}
    sessions = pd.DatetimeIndex(pd.read_parquet(source / "sessions.parquet").session)
    targets = pd.read_parquet(source / "contract_targets.parquet")
    shape = (len(CLOCKS), len(HOLD_MINUTES), len(DELAYS), len(sessions), len(ROOTS))
    result = {name: np.full(shape, np.nan) for name in ("returns", "entry")}
    with np.load(source / "price_cube.npz") as old:
        old_clocks = (2, 3, 4, 6)
        old_offsets = [OFFSETS.index(x) for x in DELAYS]
        for name, key in (("returns", "returns"), ("entry", "entry")):
            result[name][2:] = old[key][list(old_clocks)][:, :len(HOLD_MINUTES)][:, :, old_offsets]
    by_session = {stamp: i for i, stamp in enumerate(sessions)}
    ledger = []
    for ai, root in enumerate(ROOTS):
        cache = source / "minute_cache"
        bars = pd.concat([pd.read_parquet(cache / f"{table}_{symbol}.parquet")
                          for table in ("bars_1m", "bars_1m_2m")
                          for symbol in (MICROS[root], root)], ignore_index=True)
        book = PriceBook(bars)
        del bars
        for target in targets.loc[targets.root.eq(root)].itertuples(index=False):
            di = by_session[target.session]
            for ci, clock in enumerate(CLOCKS[:2]):
                for oi, offset in enumerate(DELAYS):
                    for hi, hold in enumerate(HOLD_MINUTES):
                        row = asia_pair(book, target, clock, hold, offset, asia_open, asia_close)
                        ledger.append(row)
                        if row["status"] == "matched":
                            result["returns"][ci, hi, oi, di, ai] = row["price_return"]
                            result["entry"][ci, hi, oi, di, ai] = row["entry_price"]
        print(f"[cached Asia prices] {root}", flush=True)
    pd.DataFrame(ledger).to_parquet(output / "asia_fill_ledger.parquet", index=False)
    np.savez_compressed(destination, **result)
    return result


def weights_for_class(raw, allocated, dates, sessions, *, lag=0, family=0, signal_class="gross__magnitude"):
    """Return planned asset weights. T's vote is first tradable in session T+1."""
    base, style = signal_class.split("__")
    position = dates.searchsorted(sessions, side="left")-1-lag
    valid = position >= 2
    out = np.full((len(sessions), len(ROOTS)), np.nan)
    if valid.any():
        p = position[valid]
        gross, net = raw[family], allocated[family]
        if base == "gross":
            values = gross[p]
        elif base == "gross_change":
            values = gross[p]-gross[p-1]
        elif base == "net":
            values = net[p]
        elif base == "net_change":
            values = net[p]-net[p-1]
        elif base == "net_change_of_change":
            values = net[p]-2*net[p-1]+net[p-2]
        else:
            raise ValueError(signal_class)
        if style.endswith("sign"):
            values = np.where(np.abs(values)>1e-12, np.sign(values)/len(ROOTS), 0.)
        if style.startswith("reverse_"):
            values = -values
        values = values/np.maximum(1., np.abs(values).sum(axis=1))[:, None]
        out[valid] = values
    assert np.all(dates[position[valid]] < sessions[valid])
    return out


def candidate_pnl(weights, price_cube, indices):
    """Cell x delay x session returns, preserving unknown active trades as NaN."""
    moves = price_cube["returns"].reshape(-1, len(DELAYS), len(weights), len(ROOTS))
    entries = price_cube["entry"].reshape(moves.shape)
    gross, fee, slip = asset_pnl(weights[None, None], moves, entries)
    cost = fee+slip
    selected = np.array(indices)
    def combine(x):
        part = x[:, :, :, selected]
        return np.where(np.isfinite(part).all(axis=3), part.sum(axis=3), np.nan)
    return {"gross": combine(gross), "cost": combine(cost),
            "net": combine(gross-cost), "stressed": combine(gross-2*cost)}


def fit_scores(training, *, min_history=126):
    """Robust cost stressed score per candidate on its training prefix."""
    x = np.asarray(training, dtype=float)
    if x.ndim != 3:
        raise ValueError("Expected date x candidate x delay")
    if len(x) < min_history:
        return np.full(x.shape[1], np.nan), np.zeros(x.shape[1], dtype=bool)
    common = np.isfinite(x).all(axis=2)
    x = np.where(common[:, :, None], x, np.nan)
    count = common.sum(axis=0)
    half = len(x)//2
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.nansum(x, axis=0)/count[:, None]
        var = (np.nansum(x*x, axis=0)-count[:, None]*mean*mean)/(count[:, None]-1)
        sr = np.where(var>1e-28, mean/np.sqrt(var)*np.sqrt(252), np.nan)
        first = np.nansum(x[:half, :, 0], axis=0)/common[:half].sum(axis=0)
        last = np.nansum(x[half:, :, 0], axis=0)/common[half:].sum(axis=0)
    score = np.median(sr, axis=1)
    active = (np.isfinite(x[:, :, 0]) & (np.abs(x[:, :, 0])>1e-15)).sum(axis=0)
    eligible = ((count >= max(60, int(np.ceil(.55*len(x))))) & (active >= 15)
                & np.isfinite(score) & (score > 0) & (first > 0) & (last > 0))
    return score, eligible


def select_classes(training, *, families, classes, recipes):
    """Rank transformations on agreement across independently scored families."""
    score, eligible = fit_scores(training)
    score = score.reshape(families, classes, recipes)
    eligible = eligible.reshape(families, classes, recipes)
    family_best = np.max(np.where(eligible, score, -np.inf), axis=2)
    class_score = np.full(classes, -np.inf)
    support = np.isfinite(family_best).sum(axis=0)
    for i in range(classes):
        good = np.sort(family_best[np.isfinite(family_best[:, i]), i])[::-1]
        if len(good):
            class_score[i] = good[min(1, len(good)-1)]
    qualified = np.flatnonzero(support >= 2)
    if len(qualified) == 0:
        qualified = np.flatnonzero(support > 0)
    ranked = qualified[np.argsort(-class_score[qualified], kind="stable")]
    return ranked[:2], class_score, support


def row_stats(values):
    x = summarize_matrix(values)
    return {key: float(value[0]) if key != "days" and key != "active_days" else int(value[0])
            for key, value in x.items()}
