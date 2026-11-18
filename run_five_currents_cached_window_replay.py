"""Offline Five Currents intraday replay using cached one-minute prices.

The pre-2026-07-21 generic contract identities are inferred, so those sessions
are rank-route proxies. Later sessions require the actual listed contract month.
No database, broker, or market API is accessed by this module.
"""
from __future__ import annotations

import argparse
import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_front_intraday import MICROS, ROOTS, evaluate, observed
from five_currents_opening_study import daily_totals, stats
from run_five_currents_masked_intraday import (
    BOOKS, CLOCKS, CONCEPTS, END, HOURS, MASK, OUTPUT as OLD_OUTPUT,
    SNAPSHOT, START, contract_month, entry_time, load_bars, load_signals,
)
from run_five_currents_market_hours import closed_exit_reason, crosses_maintenance


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
OUTPUT = ROOT / "five_currents_cached_window_replay_20261013"
WINDOW_MINUTES = 10
FULL_FRONT = (
    "MCL_1m_IBKR_20250101_20260414.parquet",
    "MGC_1m_IBKR_20250101_20250414_pre0323.parquet",
    "MGC_1m_IBKR_20250414_20260414.parquet",
    "MHG_1m_IBKR_20250101_20250413.parquet",
    "MHG_1m_IBKR_20250414_20260414.parquet",
    "MHNG_1m_IBKR_20250101_20250413.parquet",
    "MHNG_1m_IBKR_20250414_20260414.parquet",
)
SMALL_ARCHIVES = (
    "five_currents_intraday_audit_20261011/audited_minute_anchors.parquet",
    "five_currents_silver_reconciliation_20261011/existing_silver_nearby_minutes.parquet",
    "five_currents_silver_reconciliation_20261011/front_endpoints_exact_repairs.parquet",
)
FIELDS = ("table", "symbol", "ts", "open", "volume", "bar_count", "local_symbol", "expiry")


def required_stamps(sessions: pd.DatetimeIndex) -> np.ndarray:
    stamps = {entry_time(day, clock) + pd.Timedelta(hours=hours)
              for day in sessions for clock in CLOCKS for hours in (0, *HOURS)}
    return pd.DatetimeIndex(sorted(stamps)).asi8


def near_required(ts: pd.Series, stamps: np.ndarray) -> np.ndarray:
    values = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).asi8
    positions = np.searchsorted(stamps, values)
    left = np.maximum(positions - 1, 0)
    right = np.minimum(positions, len(stamps) - 1)
    distance = np.minimum(np.abs(values - stamps[left]), np.abs(values - stamps[right]))
    return distance <= pd.Timedelta(minutes=WINDOW_MINUTES).value


def load_local_bars(sessions: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    base, original_conflicts = load_bars(sessions, cache_only=True)
    pieces = [base[list(FIELDS)].assign(source="archived_exact")]
    stamps = required_stamps(sessions)
    source_rows = {"archived_exact": len(base)}
    for name in SMALL_ARCHIVES:
        path = ROOT / name
        frame = pd.read_parquet(path)
        frame = frame.rename(columns={"source_table": "table"})
        frame = frame.loc[near_required(frame.ts, stamps), list(FIELDS)].copy()
        frame["source"] = name
        pieces.append(frame)
        source_rows[name] = len(frame)
    for name in FULL_FRONT:
        path = ROOT / name
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=[field for field in FIELDS if field != "table"])
        frame = frame.loc[near_required(frame.ts, stamps)].copy()
        frame["table"] = "local_micro_front"
        frame["source"] = name
        pieces.append(frame[list(FIELDS) + ["source"]])
        source_rows[name] = len(frame)
        del frame
    bars = pd.concat(pieces, ignore_index=True)
    bars["ts"] = pd.to_datetime(bars.ts, utc=True)
    bars["open"] = pd.to_numeric(bars.open, errors="coerce")
    bars["volume"] = pd.to_numeric(bars.volume, errors="coerce")
    bars["bar_count"] = pd.to_numeric(bars.bar_count, errors="coerce")
    bars = bars.loc[bars.open.gt(0) & bars.local_symbol.notna()].copy()
    key = ["table", "symbol", "ts"]
    identity_count = bars.groupby(key, sort=False).local_symbol.nunique(dropna=False)
    conflicting = identity_count[identity_count.gt(1)]
    if len(conflicting):
        bars = bars.loc[~pd.MultiIndex.from_frame(bars[key]).isin(conflicting.index)].copy()
    bars = bars.drop_duplicates(key, keep="first").sort_values(key).reset_index(drop=True)
    if bars.duplicated(key).any():
        raise AssertionError("Duplicate cached bar key")
    return bars, {"source_rows": source_rows, "identity_conflicts": int(len(conflicting)),
                  "original_cache_conflicts": original_conflicts, "valid_price_rows": len(bars),
                  "trade_confirmed_rows": int((bars.volume.gt(0) | bars.bar_count.gt(0)).sum())}


def contract_targets(predictors: pd.DataFrame, *, mask_name: str) -> pd.DataFrame:
    if mask_name not in ("silver_copper", "silver_copper_gold"):
        raise ValueError(mask_name)
    ranks = pd.read_parquet(MASK / f"{mask_name}_execution_ranks.parquet")
    mapping = pd.read_parquet(SNAPSHOT / "input_mapping.parquet")
    lookup = mapping.set_index(["date", "ticker"]).contract.to_dict()
    unique = predictors[["session", "signal_date", "root"]].drop_duplicates().copy()
    unique["rank"] = [ranks.at[row.signal_date, row.root] for row in unique.itertuples()]
    unique["contract"] = [lookup.get((row.signal_date, f"{row.root}{int(row.rank)} Comdty"))
                          if pd.notna(row.rank) else None for row in unique.itertuples()]
    unique["month"] = [contract_month(row.contract, row.signal_date)
                       for row in unique.itertuples()]
    unique["identity_basis"] = np.where(unique.month.notna(), "verified_month", "rank_route_proxy")
    return unique


def bar_index(bars: pd.DataFrame) -> dict:
    result = {}
    for key, frame in bars.groupby(["table", "symbol"], sort=False):
        records = list(frame.sort_values("ts").itertuples(index=False))
        result[key] = ([row.ts for row in records], records)
    return result


def window_rows(index: dict, table: str, symbol: str, stamp: pd.Timestamp) -> list:
    group = index.get((table, symbol))
    if group is None:
        return []
    stamps, rows = group
    tolerance = pd.Timedelta(minutes=WINDOW_MINUTES)
    return rows[bisect_left(stamps, stamp - tolerance):bisect_right(stamps, stamp + tolerance)]


def choose_entry(target, stamp: pd.Timestamp, index: dict):
    if target.month is not None:
        tables = ("bars_1m", "bars_1m_2m", "local_micro_front")
    elif pd.notna(target.rank) and int(target.rank) in (1, 2):
        tables = (("bars_1m", "local_micro_front") if int(target.rank) == 1
                  else ("bars_1m_2m",))
    else:
        return None
    candidates = []
    for table in tables:
        for symbol in (target.root, MICROS[target.root]):
            for bar in window_rows(index, table, symbol, stamp):
                if target.month is not None and contract_month(bar.local_symbol, bar.ts) != target.month:
                    continue
                offset = (bar.ts - stamp).total_seconds()
                candidates.append((abs(offset), not observed(bar), offset < 0, table != "bars_1m",
                                   symbol != target.root, bar.ts, bar))
    return min(candidates, default=None, key=lambda item: item[:-1])[-1] if candidates else None


def choose_exit(entry, stamp: pd.Timestamp, index: dict):
    same = [bar for bar in window_rows(index, entry.table, entry.symbol, stamp)
            if str(bar.local_symbol).strip().upper() == str(entry.local_symbol).strip().upper()
            and bar.ts > entry.ts and not closed_exit_reason(bar.ts)]
    return min(same, default=None,
               key=lambda bar: (abs((bar.ts - stamp).total_seconds()), not observed(bar),
                                bar.ts < stamp, bar.ts))


def price_moves(targets: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    index = bar_index(bars)
    records = []
    for target in targets.itertuples(index=False):
        for clock in CLOCKS:
            requested_entry = entry_time(target.session, clock)
            entry = choose_entry(target, requested_entry, index)
            for hours in HOURS:
                requested_exit = requested_entry + pd.Timedelta(hours=hours)
                crossed = crosses_maintenance(requested_entry, requested_exit)
                row = {"session": target.session, "signal_date": target.signal_date,
                       "root": target.root, "clock": clock, "hours": hours,
                       "rank": target.rank, "target_contract": target.contract,
                       "target_month": target.month, "identity_basis": target.identity_basis,
                       "entry_utc": requested_entry, "exit_utc": requested_exit,
                       "crosses_maintenance": crossed, "status": "missing entry"}
                if entry is None:
                    if pd.notna(target.rank) and int(target.rank) > 2 and target.month is None:
                        row["status"] = "unidentified rank above stored routes"
                else:
                    row.update(symbol=entry.symbol, local_symbol=entry.local_symbol,
                               source_table=entry.table, entry_ts=entry.ts,
                               entry_offset_min=(entry.ts-requested_entry).total_seconds()/60,
                               entry_price=float(entry.open),
                               month_matches_target=(contract_month(entry.local_symbol, entry.ts) == target.month)
                               if target.month is not None else np.nan)
                    if closed_exit_reason(requested_exit):
                        row["status"] = "scheduled exit in market break"
                    elif crossed:
                        row["status"] = "hold crosses maintenance"
                    else:
                        exit_bar = choose_exit(entry, requested_exit, index)
                        if exit_bar is None:
                            row["status"] = "missing same-contract exit"
                        elif crosses_maintenance(entry.ts, exit_bar.ts):
                            row["status"] = "actual hold crosses maintenance"
                        else:
                            row.update(status="matched", exit_ts=exit_bar.ts,
                                       exit_offset_min=(exit_bar.ts-requested_exit).total_seconds()/60,
                                       exit_price=float(exit_bar.open),
                                       price_return=float(exit_bar.open/entry.open-1),
                                       entry_trade_confirmed=observed(entry),
                                       exit_trade_confirmed=observed(exit_bar),
                                       trade_pair_confirmed=observed(entry) and observed(exit_bar))
                records.append(row)
    return pd.DataFrame(records)


def coverage(moves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (clock, hours, basis), group in moves.groupby(["clock", "hours", "identity_basis"]):
        for name, roots in BOOKS.items():
            ready = group.loc[group.root.isin(roots)].pivot(index="session", columns="root", values="status")
            matched = ready.reindex(columns=roots).eq("matched").all(axis=1)
            confirmed = (group.loc[group.root.isin(roots)]
                         .pivot(index="session", columns="root", values="trade_pair_confirmed")
                         .reindex(columns=roots).eq(True).all(axis=1))
            rows.append({"clock": clock, "hours": hours, "identity_basis": basis,
                         "book": name, "sessions": len(matched), "matched_days": int(matched.sum()),
                         "trade_confirmed_days": int(confirmed.sum())})
    details = moves.groupby(["clock", "hours", "root", "identity_basis", "status"],
                            dropna=False).size().rename("sessions").reset_index()
    return pd.DataFrame(rows), details


def score(predictors: pd.DataFrame, moves: pd.DataFrame, *, include_five_families: bool = False,
          persist_five_portfolio: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, daily_rows = [], []
    for clock in CLOCKS:
        clock_moves = moves.loc[moves.clock.eq(clock)].drop(columns=["clock", "signal_date"])
        for signal, frame in predictors.groupby("signal", sort=False):
            if signal.endswith("__finalized_reference"):
                continue
            paths = evaluate(frame.drop(columns="signal"), clock_moves)
            for (variant, hours), group in paths.groupby(["variant", "hours"], sort=False):
                for book, roots in BOOKS.items():
                    if book == "five" and signal != "portfolio_net" and not include_five_families:
                        continue
                    totals = daily_totals(group, roots)
                    for basis in ("net_return", "slippage_net_return"):
                        values = totals[basis]
                        for period, mask in (("full", np.ones(len(values), dtype=bool)),
                                             ("train", values.index < pd.Timestamp("2025-07-01")),
                                             ("test", values.index >= pd.Timestamp("2025-07-01"))):
                            current = values.iloc[np.flatnonzero(mask)]
                            metrics.append({"clock": clock, "signal": signal, "variant": variant,
                                            "hours": hours, "book": book, "basis": basis,
                                            "period": period, "possible_days": len(current),
                                            **stats(current)})
                    if signal == "portfolio_net" and (book == "four" or persist_five_portfolio):
                        daily_rows.append(totals.assign(clock=clock, signal=signal, variant=variant,
                                                         hours=hours, book=book).reset_index())
            print(f"[scored] {clock} / {signal}", flush=True)
    return pd.DataFrame(metrics), pd.concat(daily_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", choices=("silver_copper", "silver_copper_gold"),
                        default="silver_copper")
    parser.add_argument("--prices-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if args.prices_only and args.score_only:
        parser.error("--prices-only and --score-only are mutually exclusive")
    output = OUTPUT / args.mask
    output.mkdir(parents=True, exist_ok=True)
    sessions = pd.bdate_range(START, END)
    predictors, names = load_signals(sessions)
    predictors.to_parquet(output / "dated_predictors.parquet", index=False)
    if args.score_only:
        moves = pd.read_parquet(output / "price_moves.parquet")
    else:
        targets = contract_targets(predictors, mask_name=args.mask)
        bars, provenance = load_local_bars(sessions)
        print(f"[cached] {len(bars)} valid minute prices, {len(targets)} asset-days", flush=True)
        moves = price_moves(targets, bars)
        summary, reasons = coverage(moves)
        targets.to_parquet(output / "contract_targets.parquet", index=False)
        moves.to_parquet(output / "price_moves.parquet", index=False)
        summary.to_csv(output / "coverage.csv", index=False)
        reasons.to_csv(output / "coverage_reasons.csv", index=False)
        manifest = {"cached_only": True, "broker_calls": False, "database_calls": False,
                    "mask": args.mask, "entry_exit_window_minutes": WINDOW_MINUTES,
                    "price_basis": "nearest_cached_bar_open; zero-volume marks allowed and identified separately",
                    "signal_source": str(CONCEPTS), "daily_snapshot": str(SNAPSHOT),
                    "prior_invalid_result": str(OLD_OUTPUT), "sources": provenance,
                    "known_identity_sessions": int(targets.loc[targets.month.notna(), "session"].nunique()),
                    "signal_families": list(names)}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        print("[coverage]", summary.loc[summary.book.eq("four") & summary.hours.isin((1, 2, 4))].to_string(index=False), flush=True)
    if args.prices_only:
        return
    metrics, daily = score(predictors, moves)
    metrics.to_csv(output / "metrics.csv", index=False)
    daily.to_parquet(output / "portfolio_daily_returns.parquet", index=False)
    print("[portfolio]", metrics.loc[metrics.signal.eq("portfolio_net") & metrics.book.eq("four")
                                  & metrics.basis.eq("slippage_net_return") & metrics.period.eq("full")
                                  & metrics.variant.eq("net_magnitude")
                                  & metrics.hours.isin((1, 2, 4))][["clock", "hours", "days", "sharpe", "total_return"]]
          .to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
