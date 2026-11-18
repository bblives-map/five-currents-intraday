"""Replay Five Currents intraday sleeves and fit one causal ATR trailing exit.

The script is deliberately isolated from production. It performs read-only,
indexed PostgreSQL reads, never contacts IBKR or another API, and writes only to
its dated research directory.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg2 import sql
from threadpoolctl import threadpool_limits

from audit_five_currents_deployed_intraday import costs, independent_weights
from build_five_currents_frontmonth_prices import (
    FIELDS,
    TABLES,
    FrontRouteBook,
    fill_pair,
    prior_volume_route,
    session_calendar,
)
from calculate_five_currents_intraday_positions import BOOK, SOURCE as RULE_SOURCE
from five_currents_atr_trail import (
    causal_trailing_exit_index,
    daily_and_weekly_metrics,
    performance,
    true_range,
    weekly_returns,
    wilder_atr,
)
from five_currents_daily.curve_policy import CURVE_RANK_LIMITS
from five_currents_daily.db import readonly_connection
from five_currents_execution_timing import ROOTS
from five_currents_front_intraday import MICROS
from five_currents_group_clock_core import SOURCE
from five_currents_opening_study import lag_contract_fields, rebuild_concepts_variants
from run_five_currents_group_clock import bundle


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
FRONT = ROOT / "five_currents_frontmonth_20261025"
PRIOR_AUDIT = ROOT / "five_currents_intraday_direction_audit_20261027"
OUT = ROOT / "five_currents_intraday_atr_trail_20261111"
OHLC_CACHE = OUT / "minute_ohlc_cache"
PERIODS = (5, 10, 14, 20, 30, 45, 60, 90, 120)
MULTIPLES = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0,
             6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
CORE_GROUPS = ("Fixed four legs", "Volume six legs", "CL", "NG", "GC", "HG")
NOW = pd.Timestamp.now(tz="UTC")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rules() -> pd.DataFrame:
    rules = pd.read_csv(PRIOR_AUDIT / "frozen_rules.csv")
    expected = sum(len(families) for families in BOOK.values())
    if len(rules) != expected:
        raise AssertionError(f"Expected {expected} deployed rules, found {len(rules)}")
    rules["leg"] = rules.root + ":" + rules.family
    if rules.leg.duplicated().any():
        raise AssertionError("Deployed rules are not unique")
    return rules


def rebuild_current_votes():
    """Recreate the exact GC-four-rank daily votes used by the dashboard."""
    arrays, names, archived_sessions, archived_dates = bundle(SOURCE)
    replay_path = ROOT / "five_currents_dashboard" / "private_replay_cache.pkl"
    replay_bytes = replay_path.read_bytes()
    replay_digest = hashlib.sha256(replay_bytes).hexdigest()
    replay = pickle.loads(replay_bytes)
    dates = pd.DatetimeIndex(replay.combined.wide.index)
    if not dates[:len(archived_dates)].equals(archived_dates):
        raise AssertionError("Current daily panel no longer shares the archived date spine")
    lagged = lag_contract_fields(replay.combined.wide, replay.combined.mapping,
                                 replay.panel.metadata)
    with threadpool_limits(limits=2):
        baseline, gold_four = rebuild_concepts_variants(
            replay,
            replay,
            lagged,
            (CURVE_RANK_LIMITS, {**CURVE_RANK_LIMITS, "GC": 4}),
        )
    raw = baseline.copy()
    gold_index = ROOTS.index("GC")
    raw[:, :, gold_index] = gold_four[:, :, gold_index]
    if raw.shape[0] != len(names):
        raise AssertionError("Signal-family count changed")
    np.savez_compressed(OUT / "current_votes.npz", raw=raw,
                        dates=dates.to_numpy(dtype="datetime64[ns]"))
    return raw, names, dates, archived_sessions, replay_digest


def load_recent_route_bars(first: pd.Timestamp, last: pd.Timestamp) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    columns = tuple(FIELDS)
    with readonly_connection(statement_timeout_ms=45_000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET lock_timeout='1s'")
            cursor.execute("SET max_parallel_workers_per_gather=0")
            cursor.execute("SET application_name='five_currents_atr_route_readonly'")
            for root in BOOK:
                parts = []
                for table in TABLES:
                    query = sql.SQL(
                        "SELECT ts,open,close,volume,bar_count,local_symbol,expiry "
                        "FROM public.{} WHERE symbol=%s AND ts>=%s AND ts<%s ORDER BY ts"
                    ).format(sql.Identifier(table))
                    cursor.execute(query, (MICROS[root], first.to_pydatetime(), last.to_pydatetime()))
                    frame = pd.DataFrame(cursor.fetchall(), columns=columns)
                    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
                    frame["table"] = table
                    parts.append(frame)
                frames[root] = pd.concat(parts, ignore_index=True)
                print(f"[route bars] {root}: {len(frames[root]):,}", flush=True)
    return frames


def reconstruct_trade_ledger(rules: pd.DataFrame, raw: np.ndarray, names: list[str],
                             dates: pd.DatetimeIndex,
                             archived_sessions: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Retain audited historical fills and independently build later sessions."""
    extensions = dates[dates > archived_sessions[-1]]
    sessions = archived_sessions.union(extensions)
    calendar = session_calendar(sessions)
    first = (archived_sessions[-1] - pd.Timedelta(days=14)).tz_localize("UTC")
    route_frames = load_recent_route_bars(first, NOW + pd.Timedelta(seconds=1))
    route_books = {root: FrontRouteBook(frame) for root, frame in route_frames.items()}
    preferred = {}
    for root in BOOK:
        for session in extensions:
            previous = sessions[sessions < session][-5:]
            preferred[root, session], _ = prior_volume_route(route_books[root], session, previous)

    old = pd.read_parquet(PRIOR_AUDIT / "full_trade_audit.parquet")
    old = old.loc[old.session.le(archived_sessions[-1])].copy()
    pieces = []
    for rule in rules.itertuples(index=False):
        family_index = names.index(rule.family)
        asset_index = ROOTS.index(rule.root)
        weights = independent_weights(raw, dates, sessions, family_index,
                                      rule.signal_class)[:, asset_index]
        weight_map = pd.Series(weights, index=sessions)
        signal_indices = dates.searchsorted(sessions, side="left") - 1
        signal_map = pd.Series(dates[signal_indices], index=sessions)
        history = old.loc[old.root.eq(rule.root) & old.family.eq(rule.family)
                          & old.delay_minutes.isin([0, 15])].copy()
        history["weight"] = history.session.map(weight_map)
        history["signal_date"] = history.session.map(signal_map)
        additions = []
        for session in extensions:
            for delay in (0, 15):
                row = fill_pair(
                    route_books[rule.root], session, rule.root, rule.clock,
                    int(rule.hold_minutes), delay, calendar, preferred[root, session],
                )
                row.update(
                    family=rule.family,
                    signal_class=rule.signal_class,
                    book=rule.book,
                    weight=float(weight_map.loc[session]),
                    signal_date=signal_map.loc[session],
                )
                additions.append(row)
        combined = pd.concat([history, pd.DataFrame(additions)], ignore_index=True, sort=False)
        combined["hold_minutes"] = int(rule.hold_minutes)
        matched = combined.status.eq("matched")
        combined["gross"] = combined.weight * (combined.exit_price / combined.entry_price - 1.0)
        combined["cost"] = costs(combined.weight, combined.entry_price, rule.root)
        combined.loc[~matched, ["gross", "cost"]] = np.nan
        flat = combined.weight.abs().le(1e-12)
        combined.loc[flat, ["gross", "cost"]] = 0.0
        combined["net"] = combined.gross - combined.cost
        if not (pd.to_datetime(combined.signal_date) < pd.to_datetime(combined.session)).all():
            raise AssertionError(f"Non-causal signal mapping for {rule.leg}")
        pieces.append(combined)
        priced = int(combined.loc[combined.delay_minutes.eq(0), "status"].eq("matched").sum())
        print(f"[ledger] {rule.leg}: {priced}/{len(sessions)} matched sessions", flush=True)
    ledger = pd.concat(pieces, ignore_index=True, sort=False)
    ledger["leg"] = ledger.root + ":" + ledger.family
    expected = len(rules) * len(sessions) * 2
    if len(ledger) != expected:
        raise AssertionError(f"Ledger has {len(ledger)} rows, expected {expected}")
    duplicates = ledger.duplicated(["leg", "session", "delay_minutes"])
    if duplicates.any():
        raise AssertionError("Duplicate leg/session/delay rows")
    ledger.to_parquet(OUT / "trade_ledger.parquet", index=False, compression="zstd")
    return ledger, sessions


def ohlc_path(root: str, table: str, month: pd.Period) -> Path:
    return OHLC_CACHE / root / f"{table}_{month.strftime('%Y%m')}.parquet"


def cache_ohlc(sessions: pd.DatetimeIndex, required_end: pd.Timestamp) -> None:
    """Cache sequential indexed reads; never invoke a broker or production writer."""
    months = pd.period_range(sessions.min().to_period("M"),
                             required_end.tz_convert(None).to_period("M"), freq="M")
    columns = ("ts", "open", "high", "low", "close", "volume", "bar_count",
               "local_symbol", "expiry")
    records = []
    with readonly_connection(statement_timeout_ms=45_000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET lock_timeout='1s'")
            cursor.execute("SET max_parallel_workers_per_gather=0")
            cursor.execute("SET application_name='five_currents_atr_ohlc_readonly'")
            for root in BOOK:
                for table in TABLES:
                    query = sql.SQL(
                        "SELECT ts,open,high,low,close,volume,bar_count,local_symbol,expiry "
                        "FROM public.{} WHERE symbol=%s AND ts>=%s AND ts<%s ORDER BY ts"
                    ).format(sql.Identifier(table))
                    for month in months:
                        path = ohlc_path(root, table, month)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        refresh = not path.exists()
                        if path.exists() and month == months[-1]:
                            stamps = pd.read_parquet(path, columns=["ts"])
                            maximum = pd.to_datetime(stamps.ts, utc=True).max() if len(stamps) else pd.NaT
                            refresh = pd.isna(maximum) or maximum < required_end - pd.Timedelta(minutes=2)
                        if refresh:
                            begin = month.start_time.tz_localize("UTC")
                            end = (month + 1).start_time.tz_localize("UTC")
                            cursor.execute(query, (MICROS[root], begin.to_pydatetime(), end.to_pydatetime()))
                            frame = pd.DataFrame(cursor.fetchall(), columns=columns)
                            frame["ts"] = pd.to_datetime(frame.ts, utc=True)
                            if frame.ts.duplicated().any():
                                raise AssertionError(f"Duplicate {root}/{table}/{month} bars")
                            frame.to_parquet(path, index=False, compression="zstd")
                            time.sleep(0.12)
                        else:
                            frame = pd.read_parquet(path, columns=["ts"])
                        records.append({"root": root, "table": table, "month": str(month),
                                        "rows": len(frame), "path": str(path)})
                        print(f"[OHLC cache] {root} {table} {month}: {len(frame):,}", flush=True)
    pd.DataFrame(records).to_parquet(OUT / "ohlc_cache_ledger.parquet", index=False)


def load_contract_bars(root: str, month: pd.Period,
                       periods: tuple[int, ...]) -> dict[str, dict[str, np.ndarray]]:
    frames = []
    for offset in (-1, 0, 1):
        candidate_month = month + offset
        for table in TABLES:
            path = ohlc_path(root, table, candidate_month)
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            if len(frame):
                frame["table"] = table
                frames.append(frame)
    if not frames:
        return {}
    frame = pd.concat(frames, ignore_index=True)
    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
    for column in ("open", "high", "low", "close", "volume", "bar_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame.local_symbol.notna() & frame.close.gt(0)
                      & frame.high.gt(0) & frame.low.gt(0)].copy()
    frame["traded"] = frame.volume.gt(0)
    frame["source_priority"] = frame.table.ne("bars_1m").astype(int)
    frame["volume_priority"] = frame.volume.fillna(-1.0)
    frame = frame.sort_values(
        ["local_symbol", "ts", "traded", "volume_priority", "source_priority"],
        ascending=[True, True, False, False, True], kind="stable",
    ).drop_duplicates(["local_symbol", "ts"], keep="first")
    books = {}
    for contract, part in frame.groupby("local_symbol", sort=False):
        part = part.sort_values("ts", kind="stable")
        ranges = true_range(part.high.to_numpy(float), part.low.to_numpy(float),
                            part.close.to_numpy(float))
        books[str(contract)] = {
            "timestamps": part.ts.to_numpy(),
            "nanoseconds": part.ts.astype("int64").to_numpy(),
            "open": part.open.to_numpy(float),
            "high": part.high.to_numpy(float),
            "low": part.low.to_numpy(float),
            "close": part.close.to_numpy(float),
            "traded": part.traded.to_numpy(bool),
            "atr": {period: wilder_atr(ranges, period) for period in periods},
        }
    return books


def simulate_trails(ledger: pd.DataFrame,
                    pairs: list[tuple[int, float]]) -> dict[str, np.ndarray]:
    """Simulate all pairs on exact-contract paths, retaining baseline gaps."""
    periods = tuple(sorted({int(period) for period, _ in pairs}))
    candidates = len(pairs)
    rows = len(ledger)
    baseline_net = ledger.net.to_numpy(float)
    baseline_gross = ledger.gross.to_numpy(float)
    net = np.repeat(baseline_net[None, :], candidates, axis=0)
    gross = np.repeat(baseline_gross[None, :], candidates, axis=0)
    stopped = np.zeros((candidates, rows), dtype=bool)
    fallback = np.zeros((len(periods), rows), dtype=bool)
    initial_atr = np.full((len(periods), rows), np.nan, dtype=float)
    baseline_hold = (
        pd.to_datetime(ledger.exit_time, utc=True) - pd.to_datetime(ledger.entry_time, utc=True)
    ).dt.total_seconds().div(60).to_numpy(float)
    hold = np.repeat(baseline_hold[None, :], candidates, axis=0)
    exit_price = np.repeat(ledger.exit_price.to_numpy(float)[None, :], candidates, axis=0)
    baseline_exit_ns = pd.to_datetime(ledger.exit_time, utc=True).astype("int64").to_numpy()
    exit_ns = np.repeat(baseline_exit_ns[None, :], candidates, axis=0)
    pair_indices = {(period, multiple): i for i, (period, multiple) in enumerate(pairs)}
    period_indices = {period: i for i, period in enumerate(periods)}

    active = ledger.weight.abs().gt(1e-12)
    matched = ledger.status.eq("matched")
    targets = ledger.loc[active & matched].copy()
    targets["entry_month"] = pd.to_datetime(targets.entry_time, utc=True).dt.tz_localize(None).dt.to_period("M")
    unresolved = {"missing_contract": 0, "missing_entry_bar": 0, "missing_exit_bar": 0,
                  "price_mismatch": 0}
    for (root, month), group in targets.groupby(["root", "entry_month"], sort=True):
        books = load_contract_bars(root, month, periods)
        for row in group.itertuples():
            row_index = int(row.Index)
            book = books.get(str(row.entry_contract))
            if book is None:
                fallback[:, row_index] = True
                unresolved["missing_contract"] += 1
                continue
            entry_stamp = pd.Timestamp(row.entry_time) - pd.Timedelta(minutes=1)
            final_stamp = pd.Timestamp(row.exit_time) - pd.Timedelta(minutes=1)
            entry_index = int(np.searchsorted(book["nanoseconds"], entry_stamp.value, side="left"))
            final_index = int(np.searchsorted(book["nanoseconds"], final_stamp.value, side="left"))
            if entry_index >= len(book["nanoseconds"]) or book["nanoseconds"][entry_index] != entry_stamp.value:
                fallback[:, row_index] = True
                unresolved["missing_entry_bar"] += 1
                continue
            if final_index >= len(book["nanoseconds"]) or book["nanoseconds"][final_index] != final_stamp.value:
                fallback[:, row_index] = True
                unresolved["missing_exit_bar"] += 1
                continue
            if not np.isclose(float(book["close"][entry_index]), float(row.entry_price),
                              rtol=1e-8, atol=1e-8):
                unresolved["price_mismatch"] += 1
            for period in periods:
                pi = period_indices[period]
                atr = book["atr"][period]
                initial_atr[pi, row_index] = atr[entry_index]
                if not np.isfinite(atr[entry_index]) or atr[entry_index] <= 0:
                    fallback[pi, row_index] = True
                    continue
                for candidate_period, multiple in pairs:
                    if candidate_period != period:
                        continue
                    candidate = pair_indices[(candidate_period, multiple)]
                    result_price, stop_index, was_stopped = causal_trailing_exit_index(
                        book["open"], book["high"], book["low"], book["traded"], atr,
                        entry_index, final_index, float(row.entry_price), float(row.exit_price),
                        float(row.weight), float(multiple),
                    )
                    result_time = (pd.Timestamp(book["timestamps"][stop_index])
                                   + pd.Timedelta(minutes=1)
                                   if stop_index >= 0 else pd.Timestamp(row.exit_time))
                    candidate_gross = float(row.weight) * (result_price / float(row.entry_price) - 1.0)
                    gross[candidate, row_index] = candidate_gross
                    net[candidate, row_index] = candidate_gross - float(row.cost)
                    stopped[candidate, row_index] = was_stopped
                    exit_price[candidate, row_index] = result_price
                    exit_ns[candidate, row_index] = result_time.value
                    hold[candidate, row_index] = (
                        result_time - pd.Timestamp(row.entry_time)
                    ).total_seconds() / 60.0
        print(f"[trail] {root} {month}: {len(group):,} active trades", flush=True)
    print(f"[trail] fallback diagnostics: {unresolved}", flush=True)
    return {
        "net": net,
        "gross": gross,
        "stopped": stopped,
        "hold": hold,
        "exit_price": exit_price,
        "exit_ns": exit_ns,
        "initial_atr": initial_atr,
        "fallback": fallback,
        "periods": np.asarray(periods, dtype=int),
    }


def group_definitions(rules: pd.DataFrame) -> dict[str, list[str]]:
    legs = rules.leg.tolist()
    fixed = rules.loc[rules.book.eq("fixed"), "leg"].tolist()
    volume = rules.loc[rules.book.eq("volume"), "leg"].tolist()
    groups = {
        "Portfolio": legs,
        "Fixed four legs": fixed,
        "Volume six legs": volume,
    }
    for root in BOOK:
        groups[root] = rules.loc[rules.root.eq(root), "leg"].tolist()
    for leg in legs:
        groups[leg] = [leg]
    return groups


def aggregate_rows(row_values: np.ndarray, ledger: pd.DataFrame,
                   sessions: pd.DatetimeIndex, rules: pd.DataFrame) -> dict[str, np.ndarray]:
    values = np.asarray(row_values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    legs = rules.leg.tolist()
    session_lookup = {value: i for i, value in enumerate(sessions)}
    leg_lookup = {value: i for i, value in enumerate(legs)}
    cube = np.full((values.shape[0], len(sessions), len(legs)), np.nan, dtype=float)
    session_positions = np.fromiter((session_lookup[value] for value in ledger.session), int)
    leg_positions = np.fromiter((leg_lookup[value] for value in ledger.leg), int)
    cube[:, session_positions, leg_positions] = values
    output = {}
    for name, members in group_definitions(rules).items():
        indices = [leg_lookup[member] for member in members]
        part = cube[:, :, indices]
        valid = np.isfinite(part).all(axis=2)
        summed = np.nansum(part, axis=2)
        summed[~valid] = np.nan
        output[name] = summed
    return output


def series_metrics(values: np.ndarray, sessions: pd.DatetimeIndex,
                   mask: np.ndarray) -> dict[str, float]:
    series = pd.Series(values[mask], index=sessions[mask], dtype=float)
    return daily_and_weekly_metrics(series)


def parameter_features(candidate_groups: dict[str, np.ndarray],
                       baseline_groups: dict[str, np.ndarray],
                       sessions: pd.DatetimeIndex, fraction: float) -> tuple[pd.DataFrame, dict[str, float]]:
    cut = max(2, int(len(sessions) * fraction))
    mask = np.arange(len(sessions)) < cut
    baseline_portfolio = series_metrics(baseline_groups["Portfolio"][0], sessions, mask)
    baseline_core = [series_metrics(baseline_groups[name][0], sessions, mask)["daily_sharpe"]
                     for name in CORE_GROUPS]
    baseline = {
        "portfolio_daily_sharpe": baseline_portfolio["daily_sharpe"],
        "portfolio_weekly_sharpe": baseline_portfolio["weekly_sharpe"],
        "portfolio_daily_mean": baseline_portfolio["daily_mean"],
        "portfolio_weekly_mean": baseline_portfolio["weekly_mean"],
        "median_core_daily_sharpe": float(np.nanmedian(baseline_core)),
        "portfolio_max_drawdown": baseline_portfolio["daily_max_drawdown"],
    }
    rows = []
    for candidate, (period, multiple) in enumerate(product(PERIODS, MULTIPLES)):
        portfolio = series_metrics(candidate_groups["Portfolio"][candidate], sessions, mask)
        core = [series_metrics(candidate_groups[name][candidate], sessions, mask)["daily_sharpe"]
                for name in CORE_GROUPS]
        rows.append({
            "candidate": candidate,
            "atr_period": period,
            "atr_multiple": multiple,
            "portfolio_daily_sharpe": portfolio["daily_sharpe"],
            "portfolio_weekly_sharpe": portfolio["weekly_sharpe"],
            "portfolio_daily_mean": portfolio["daily_mean"],
            "portfolio_weekly_mean": portfolio["weekly_mean"],
            "median_core_daily_sharpe": float(np.nanmedian(core)),
            "portfolio_max_drawdown": portfolio["daily_max_drawdown"],
        })
    frame = pd.DataFrame(rows)
    criteria = list(baseline)
    percentile_columns = []
    for criterion in criteria:
        column = criterion + "_percentile"
        frame[column] = frame[criterion].rank(pct=True, method="average").fillna(0.0)
        percentile_columns.append(column)
    frame["rank_score"] = frame[percentile_columns].mean(axis=1)
    frame["improvement_count"] = sum(
        frame[criterion].ge(baseline[criterion]).astype(int) for criterion in criteria
    )
    score_lookup = frame.set_index(["atr_period", "atr_multiple"]).rank_score
    stable_scores = []
    for row in frame.itertuples(index=False):
        period_position = PERIODS.index(int(row.atr_period))
        multiple_position = MULTIPLES.index(float(row.atr_multiple))
        neighborhood = []
        for pi in range(max(0, period_position - 1), min(len(PERIODS), period_position + 2)):
            for mi in range(max(0, multiple_position - 1), min(len(MULTIPLES), multiple_position + 2)):
                neighborhood.append(score_lookup.loc[PERIODS[pi], MULTIPLES[mi]])
        stable_scores.append(float(np.nanmedian(neighborhood)))
    frame["stable_rank_score"] = stable_scores
    return frame, baseline


def choose_parameter(candidate_groups: dict[str, np.ndarray],
                     baseline_groups: dict[str, np.ndarray],
                     sessions: pd.DatetimeIndex, fraction: float):
    frame, baseline = parameter_features(candidate_groups, baseline_groups, sessions, fraction)
    required = 4
    eligible = frame.loc[frame.improvement_count.ge(required)]
    if eligible.empty:
        required = int(frame.improvement_count.max())
        eligible = frame.loc[frame.improvement_count.eq(required)]
    chosen = eligible.sort_values(
        ["stable_rank_score", "rank_score", "atr_multiple", "atr_period"],
        ascending=[False, False, False, True], kind="stable",
    ).iloc[0]
    return chosen, frame, baseline, required


def all_model_metrics(models: dict[str, dict[str, np.ndarray]],
                      sessions: pd.DatetimeIndex, cut: int) -> pd.DataFrame:
    rows = []
    masks = {
        "Full": np.ones(len(sessions), dtype=bool),
        "Calibration 30%": np.arange(len(sessions)) < cut,
        "Held-out 70%": np.arange(len(sessions)) >= cut,
    }
    for model, groups in models.items():
        for group, values in groups.items():
            if group not in ("Portfolio", *CORE_GROUPS) and ":" not in group:
                continue
            values = np.asarray(values)
            if values.ndim == 2:
                if len(values) != 1:
                    raise AssertionError(f"Expected one path for {model}/{group}")
                values = values[0]
            for sample, mask in masks.items():
                rows.append({"model": model, "series": group, "sample": sample,
                             **series_metrics(values, sessions, mask)})
    return pd.DataFrame(rows)


def split_stability(candidate_groups: dict[str, np.ndarray],
                    baseline_groups: dict[str, np.ndarray],
                    sessions: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for fraction in (0.3, 0.4, 0.5):
        chosen, _, _, required = choose_parameter(candidate_groups, baseline_groups,
                                                  sessions, fraction)
        candidate = int(chosen.candidate)
        cut = int(len(sessions) * fraction)
        mask = np.arange(len(sessions)) >= cut
        baseline = series_metrics(baseline_groups["Portfolio"][0], sessions, mask)
        overlaid = series_metrics(candidate_groups["Portfolio"][candidate], sessions, mask)
        rows.append({
            "calibration_fraction": fraction,
            "test_fraction": 1.0 - fraction,
            "calibration_end": sessions[cut - 1],
            "test_start": sessions[cut],
            "atr_period": int(chosen.atr_period),
            "atr_multiple": float(chosen.atr_multiple),
            "calibration_improvement_criteria": int(chosen.improvement_count),
            "minimum_required_criteria": required,
            "test_baseline_daily_sharpe": baseline["daily_sharpe"],
            "test_atr_daily_sharpe": overlaid["daily_sharpe"],
            "test_baseline_weekly_sharpe": baseline["weekly_sharpe"],
            "test_atr_weekly_sharpe": overlaid["weekly_sharpe"],
            "test_baseline_daily_mean": baseline["daily_mean"],
            "test_atr_daily_mean": overlaid["daily_mean"],
            "test_baseline_weekly_mean": baseline["weekly_mean"],
            "test_atr_weekly_mean": overlaid["weekly_mean"],
            "test_baseline_max_drawdown": baseline["daily_max_drawdown"],
            "test_atr_max_drawdown": overlaid["daily_max_drawdown"],
        })
    return pd.DataFrame(rows)


def yearly_metrics(models: dict[str, dict[str, np.ndarray]],
                   sessions: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for model, groups in models.items():
        for group, values in groups.items():
            if group not in ("Portfolio", *CORE_GROUPS):
                continue
            values = np.asarray(values)
            if values.ndim == 2:
                values = values[0]
            series = pd.Series(values, index=sessions)
            for year, part in series.groupby(series.index.year):
                rows.append({"model": model, "series": group, "year": int(year),
                             **daily_and_weekly_metrics(part)})
    return pd.DataFrame(rows)


def save_simulation(path: Path, simulation: dict[str, np.ndarray],
                    pairs: list[tuple[int, float]]) -> None:
    np.savez_compressed(
        path,
        **simulation,
        candidate_period=np.asarray([pair[0] for pair in pairs], dtype=int),
        candidate_multiple=np.asarray([pair[1] for pair in pairs], dtype=float),
    )


def load_simulation(path: Path) -> tuple[dict[str, np.ndarray], list[tuple[int, float]]]:
    with np.load(path) as saved:
        simulation = {name: saved[name] for name in saved.files
                      if name not in ("candidate_period", "candidate_multiple")}
        pairs = list(zip(saved["candidate_period"].astype(int).tolist(),
                         saved["candidate_multiple"].astype(float).tolist()))
    return simulation, pairs


def combine_simulations(first: dict[str, np.ndarray], first_pairs: list[tuple[int, float]],
                        second: dict[str, np.ndarray], second_pairs: list[tuple[int, float]],
                        desired_pairs: list[tuple[int, float]]) -> dict[str, np.ndarray]:
    if not np.array_equal(first["periods"], second["periods"]):
        raise AssertionError("Cannot combine simulations with different ATR-period grids")
    candidate_keys = ("net", "gross", "stopped", "hold", "exit_price", "exit_ns")
    locations = {pair: (first, i) for i, pair in enumerate(first_pairs)}
    locations.update({pair: (second, i) for i, pair in enumerate(second_pairs)})
    if any(pair not in locations for pair in desired_pairs):
        raise AssertionError("Combined simulation is missing requested parameter pairs")
    result = {key: np.stack([locations[pair][0][key][locations[pair][1]]
                             for pair in desired_pairs], axis=0)
              for key in candidate_keys}
    for key in ("initial_atr", "fallback", "periods"):
        result[key] = first[key]
    return result


def trail_behavior(ledger: pd.DataFrame, rules: pd.DataFrame,
                   simulation: dict[str, np.ndarray], candidate: int,
                   period: int, multiple: float) -> pd.DataFrame:
    groups = group_definitions(rules)
    active = ledger.weight.abs().gt(1e-12).to_numpy()
    matched = ledger.status.eq("matched").to_numpy()
    period_index = list(simulation["periods"]).index(period)
    rows = []
    for name, members in groups.items():
        member = ledger.leg.isin(members).to_numpy()
        eligible = member & active & matched
        stopped = simulation["stopped"][candidate] & eligible
        fallback = simulation["fallback"][period_index] & eligible
        baseline_holds = (
            pd.to_datetime(ledger.exit_time, utc=True) - pd.to_datetime(ledger.entry_time, utc=True)
        ).dt.total_seconds().div(60).to_numpy(float)
        rows.append({
            "series": name,
            "active_priced_trades": int(eligible.sum()),
            "stopped_trades": int(stopped.sum()),
            "stop_rate": float(stopped.sum() / eligible.sum()) if eligible.any() else np.nan,
            "atr_fallback_trades": int(fallback.sum()),
            "median_baseline_hold_minutes": float(np.nanmedian(baseline_holds[eligible]))
            if eligible.any() else np.nan,
            "median_atr_hold_minutes": float(np.nanmedian(simulation["hold"][candidate, eligible]))
            if eligible.any() else np.nan,
            "median_initial_stop_pct": float(np.nanmedian(
                multiple * simulation["initial_atr"][period_index, eligible]
                / ledger.entry_price.to_numpy(float)[eligible]
            )) if eligible.any() else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    rules.to_csv(OUT / "deployed_rules.csv", index=False)
    raw, names, dates, archived_sessions, replay_digest = rebuild_current_votes()
    ledger, sessions = reconstruct_trade_ledger(rules, raw, names, dates, archived_sessions)
    required_end = pd.to_datetime(ledger.loc[ledger.status.eq("matched"), "exit_time"], utc=True).max()
    cache_ohlc(sessions, required_end)

    leg_order = {leg: i for i, leg in enumerate(rules.leg)}
    ledgers = {}
    for delay in (0, 15):
        part = ledger.loc[ledger.delay_minutes.eq(delay)].copy()
        part["leg_order"] = part.leg.map(leg_order)
        part = part.sort_values(["session", "leg_order"], kind="stable").drop(columns="leg_order").reset_index(drop=True)
        if len(part) != len(sessions) * len(rules):
            raise AssertionError(f"Incomplete delay-{delay} ledger")
        ledgers[delay] = part

    pairs = [(period, multiple) for period, multiple in product(PERIODS, MULTIPLES)]
    grid_cache = OUT / "trail_grid_simulation.npz"
    if grid_cache.exists():
        simulation, cached_pairs = load_simulation(grid_cache)
        missing_pairs = [pair for pair in pairs if pair not in cached_pairs]
        if missing_pairs:
            print(f"[trail] extending cached grid by {len(missing_pairs)} wider pairs", flush=True)
            extension = simulate_trails(ledgers[0], missing_pairs)
            simulation = combine_simulations(simulation, cached_pairs, extension,
                                             missing_pairs, pairs)
            save_simulation(grid_cache, simulation, pairs)
        elif cached_pairs != pairs:
            locations = {pair: i for i, pair in enumerate(cached_pairs)}
            candidate_keys = ("net", "gross", "stopped", "hold", "exit_price", "exit_ns")
            simulation = {
                **{key: np.stack([simulation[key][locations[pair]] for pair in pairs])
                   for key in candidate_keys},
                **{key: simulation[key] for key in ("initial_atr", "fallback", "periods")},
            }
            save_simulation(grid_cache, simulation, pairs)
        print("[trail] loaded completed parameter simulation", flush=True)
    else:
        simulation = simulate_trails(ledgers[0], pairs)
        save_simulation(grid_cache, simulation, pairs)
    baseline_groups = aggregate_rows(ledgers[0].net.to_numpy(float), ledgers[0], sessions, rules)
    candidate_groups = aggregate_rows(simulation["net"], ledgers[0], sessions, rules)

    chosen, grid, baseline_calibration, required_criteria = choose_parameter(
        candidate_groups, baseline_groups, sessions, 0.3,
    )
    selected_index = int(chosen.candidate)
    selected_period = int(chosen.atr_period)
    selected_multiple = float(chosen.atr_multiple)
    cut = int(len(sessions) * 0.3)
    print(f"[selection] ATR({selected_period}) x {selected_multiple:g}; "
          f"{int(chosen.improvement_count)}/6 calibration criteria", flush=True)

    mask_full = np.ones(len(sessions), dtype=bool)
    mask_oos = np.arange(len(sessions)) >= cut
    for label, mask in (("full", mask_full), ("held_out_70", mask_oos)):
        for candidate in range(len(pairs)):
            values = series_metrics(candidate_groups["Portfolio"][candidate], sessions, mask)
            grid.loc[candidate, f"{label}_daily_sharpe"] = values["daily_sharpe"]
            grid.loc[candidate, f"{label}_weekly_sharpe"] = values["weekly_sharpe"]
            grid.loc[candidate, f"{label}_daily_mean"] = values["daily_mean"]
            grid.loc[candidate, f"{label}_weekly_mean"] = values["weekly_mean"]
            grid.loc[candidate, f"{label}_max_drawdown"] = values["daily_max_drawdown"]
    grid["selected_30_70"] = grid.candidate.eq(selected_index)
    grid.to_csv(OUT / "parameter_grid.csv", index=False)

    selected_groups = {name: values[selected_index:selected_index + 1]
                       for name, values in candidate_groups.items()}
    main_models = {"Fixed exit": baseline_groups, "ATR trail": selected_groups}
    metrics = all_model_metrics(main_models, sessions, cut)
    metrics.to_csv(OUT / "performance_metrics.csv", index=False)
    yearly_metrics(main_models, sessions).to_csv(OUT / "yearly_metrics.csv", index=False)
    splits = split_stability(candidate_groups, baseline_groups, sessions)
    splits.to_csv(OUT / "split_stability.csv", index=False)

    selected_trade = ledgers[0].copy()
    period_index = list(simulation["periods"]).index(selected_period)
    selected_trade["atr_period"] = selected_period
    selected_trade["atr_multiple"] = selected_multiple
    selected_trade["initial_atr"] = simulation["initial_atr"][period_index]
    selected_trade["atr_fallback"] = simulation["fallback"][period_index]
    selected_trade["atr_stopped"] = simulation["stopped"][selected_index]
    selected_trade["atr_exit_time"] = pd.to_datetime(simulation["exit_ns"][selected_index], utc=True)
    selected_trade["atr_exit_price"] = simulation["exit_price"][selected_index]
    selected_trade["atr_hold_minutes"] = simulation["hold"][selected_index]
    selected_trade["atr_gross"] = simulation["gross"][selected_index]
    selected_trade["atr_net"] = simulation["net"][selected_index]
    selected_trade.to_parquet(OUT / "selected_trade_audit.parquet", index=False, compression="zstd")
    behavior = trail_behavior(ledgers[0], rules, simulation, selected_index,
                              selected_period, selected_multiple)
    behavior.to_csv(OUT / "trail_behavior.csv", index=False)

    baseline_double_rows = ledgers[0].gross.to_numpy(float) - 2.0 * ledgers[0].cost.to_numpy(float)
    selected_double_rows = simulation["gross"][selected_index] - 2.0 * ledgers[0].cost.to_numpy(float)
    baseline_double = aggregate_rows(baseline_double_rows, ledgers[0], sessions, rules)
    selected_double = aggregate_rows(selected_double_rows, ledgers[0], sessions, rules)
    delayed_pairs = [(selected_period, selected_multiple)]
    delay_cache = OUT / "trail_delay15_simulation.npz"
    if delay_cache.exists():
        delayed_simulation, cached_delay_pairs = load_simulation(delay_cache)
        if cached_delay_pairs != delayed_pairs:
            delayed_simulation = simulate_trails(ledgers[15], delayed_pairs)
            save_simulation(delay_cache, delayed_simulation, delayed_pairs)
    else:
        delayed_simulation = simulate_trails(ledgers[15], delayed_pairs)
        save_simulation(delay_cache, delayed_simulation, delayed_pairs)
    baseline_delay = aggregate_rows(ledgers[15].net.to_numpy(float), ledgers[15], sessions, rules)
    selected_delay = aggregate_rows(delayed_simulation["net"], ledgers[15], sessions, rules)
    robustness_models = {
        "Fixed exit, doubled friction": baseline_double,
        "ATR trail, doubled friction": selected_double,
        "Fixed exit, +15m entry": baseline_delay,
        "ATR trail, +15m entry": selected_delay,
    }
    all_model_metrics(robustness_models, sessions, cut).to_csv(
        OUT / "robustness_metrics.csv", index=False,
    )

    daily_columns = {}
    for model, groups in {
        **main_models,
        **robustness_models,
    }.items():
        for group, values in groups.items():
            array = np.asarray(values)
            if array.ndim == 2:
                if len(array) != 1:
                    continue
                array = array[0]
            daily_columns[f"{model}|{group}"] = array
    daily = pd.DataFrame(daily_columns, index=sessions)
    daily.index.name = "session"
    daily.to_parquet(OUT / "daily_returns.parquet", compression="zstd")
    weekly = pd.DataFrame({column: weekly_returns(daily[column]) for column in daily})
    weekly.index.name = "week_end"
    weekly.to_parquet(OUT / "weekly_returns.parquet", compression="zstd")

    selection = {
        "atr_period_minutes": selected_period,
        "atr_multiple": selected_multiple,
        "candidate_index": selected_index,
        "calibration_fraction": 0.3,
        "calibration_start": str(sessions[0].date()),
        "calibration_end": str(sessions[cut - 1].date()),
        "held_out_start": str(sessions[cut].date()),
        "held_out_end": str(sessions[-1].date()),
        "improvement_criteria_met": int(chosen.improvement_count),
        "minimum_improvement_criteria_required": required_criteria,
        "stable_rank_score": float(chosen.stable_rank_score),
        "selection_features": list(baseline_calibration),
    }
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "history_start": str(sessions[0].date()),
        "history_end": str(sessions[-1].date()),
        "sessions": len(sessions),
        "rules": len(rules),
        "parameter_candidates": len(pairs),
        "atr_period_grid_minutes": PERIODS,
        "atr_multiple_grid": MULTIPLES,
        "selection": selection,
        "method": "One common ATR pair selected on the first 30% and frozen for the remaining 70%; 40/60 and 50/50 are independent stability checks.",
        "trail_causality": "Entry-bar ATR only; each stop is tested on the next bar before that bar can update the trail; gap-through exits use the observed open.",
        "execution": "Exact listed entry contract, positive-volume entry/exit within ten minutes, same modelled commission and slippage as the deployed replay.",
        "data_access": "Read-only indexed PostgreSQL OHLC reads. No broker/API calls, table writes, dashboard publications or collector changes.",
        "daily_replay_sha256": replay_digest,
        "prior_audit_sha256": file_digest(PRIOR_AUDIT / "full_trade_audit.parquet"),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    portfolio = metrics.loc[metrics.series.eq("Portfolio")
                            & metrics["sample"].isin(["Full", "Held-out 70%"]),
                            ["model", "sample", "daily_sharpe", "weekly_sharpe",
                             "daily_mean", "weekly_mean", "daily_max_drawdown"]]
    print(portfolio.to_string(index=False), flush=True)
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
