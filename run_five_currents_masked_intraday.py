"""Read-only clock study of the archived Five Currents liquidity-masked book.

The long-history price route is a rank-matched proxy, not a contract-identity match.
An exact-month check is reported separately where the daily mapping has real codes.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from five_currents_front_intraday import HOURS, MICROS, ROOTS, VARIANTS, evaluate, observed, valid_price
from five_currents_opening_study import allocated_sleeves, daily_totals, opening_predictors, session_open, stats
from run_five_currents_market_hours import closed_exit_reason, clock_entry as market_clock_entry, crosses_maintenance


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
SNAPSHOT = ROOT / "five_currents_dashboard/snapshots/000037_2026-10-12"
MASK = ROOT / "five_currents_liquidity_mask_20261013"
CONCEPTS = ROOT / "five_currents_opening_study_final_20261011/concept_weights.npz"
ARCHIVED_LAGGED = ROOT / "five_currents_opening_complete_20261011/lagged_volume_oi_targets.parquet"
OUTPUT = ROOT / "five_currents_masked_intraday_20261013"
START = pd.Timestamp("2024-06-20")
END = pd.Timestamp("2026-10-12")
SPLIT = pd.Timestamp("2025-07-01")
HALF = pd.Timestamp("2025-01-01")
CLOCKS = ("cme_open", "early_utc", "london_open", "london_close", "us_open", "us_close")
BOOKS = {"four": ("CL", "NG", "GC", "HG"), "five": ROOTS}
SYMBOLS = (*ROOTS, *MICROS.values())
TABLES = ("bars_1m", "bars_1m_2m")
MODES = ("rank_proxy_traded", "rank_proxy_mark", "exact_month_traded")
MONTH_CODES = "FGHJKMNQUVXZ"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def entry_time(session: pd.Timestamp, clock: str) -> pd.Timestamp:
    if clock == "cme_open":
        return session_open(session)
    if clock == "early_utc":
        date = pd.Timestamp(session).date()
        return pd.Timestamp(datetime(date.year, date.month, date.day, 1,
                                     tzinfo=ZoneInfo("America/Chicago"))).tz_convert("UTC")
    if clock == "us_close":
        date = pd.Timestamp(session).date()
        return pd.Timestamp(datetime(date.year, date.month, date.day, 16,
                                     tzinfo=ZoneInfo("America/New_York"))).tz_convert("UTC")
    return market_clock_entry(session, clock)


def load_signals(sessions: pd.DatetimeIndex) -> tuple[pd.DataFrame, tuple[str, ...]]:
    weights = pd.read_parquet(SNAPSHOT / "weights.parquet").reindex(columns=ROOTS)
    selection = pd.read_parquet(SNAPSHOT / "selection_audit.parquet")
    scaling = pd.read_parquet(SNAPSHOT / "product_scaling.parquet")
    names = tuple(selection.economic_family.drop_duplicates())
    concepts = np.load(CONCEPTS)
    original, original_sleeves, _ = allocated_sleeves(concepts["original"], names, weights.index[:-1], selection, scaling)
    np.testing.assert_allclose(original, weights.iloc[:-1].to_numpy(float), atol=0, rtol=0)
    lagged, sleeves, _ = allocated_sleeves(concepts["lagged"], names, weights.index[:-1], selection, scaling)
    archived = pd.read_parquet(ARCHIVED_LAGGED).reindex(columns=ROOTS)
    np.testing.assert_allclose(lagged, archived.to_numpy(float), atol=1e-12, rtol=0)
    sources = [("portfolio_net", pd.DataFrame(lagged, index=archived.index, columns=ROOTS)),
               ("portfolio_finalized_reference", weights.iloc[:-1])]
    sources.extend((name, pd.DataFrame(sleeves[i], index=archived.index, columns=ROOTS))
                   for i, name in enumerate(names))
    sources.extend((f"{name}__finalized_reference",
                    pd.DataFrame(original_sleeves[i], index=archived.index, columns=ROOTS))
                   for i, name in enumerate(names))
    frames = [opening_predictors(frame, sessions).assign(signal=name) for name, frame in sources]
    predictors = pd.concat(frames, ignore_index=True)
    if not predictors.signal_date.lt(predictors.session).all():
        raise AssertionError("A signal uses same-day information")
    return predictors, names


def cached_bars(table: str) -> pd.DataFrame:
    files = ([ROOT / "five_currents_market_hours_20261013_reconciled/front_minute_anchors.parquet",
              ROOT / "five_currents_opening_complete_20261011/opening_minute_anchors.parquet",
              ROOT / "five_currents_front_intraday_price_edge_20261010/front_minute_endpoints.parquet"]
             if table == "bars_1m" else
             [ROOT / "five_currents_market_hours_20261013_reconciled/nearby_proxy_minute_anchors.parquet"])
    frames = [pd.read_parquet(path) for path in files]
    bars = pd.concat(frames, ignore_index=True)
    bars["ts"] = pd.to_datetime(bars.ts, utc=True)
    bars = bars.loc[bars.symbol.isin(SYMBOLS)].copy()
    bars["table"] = table
    return bars


def required_stamps(sessions: pd.DatetimeIndex) -> list[datetime]:
    stamps = {entry_time(day, clock) + pd.Timedelta(hours=hours)
              for day in sessions for clock in CLOCKS for hours in (0, *HOURS)}
    return [stamp.to_pydatetime() for stamp in sorted(stamps)]


def fetch_missing(table: str, cached: pd.DataFrame, stamps: list[datetime]) -> pd.DataFrame:
    from psycopg2 import sql
    from five_currents_daily.db import readonly_connection

    request = pd.DatetimeIndex(stamps)
    rows = []
    fields = ("ts", "open", "close", "volume", "bar_count", "symbol", "local_symbol", "expiry")
    query = sql.SQL("SELECT ts,open,close,volume,bar_count,symbol,local_symbol,expiry "
                    "FROM public.{} WHERE symbol=%s AND ts=ANY(%s::timestamptz[])").format(sql.Identifier(table))
    with readonly_connection(statement_timeout_ms=10000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET max_parallel_workers_per_gather=0")
            cursor.execute("SET lock_timeout='1s'")
            cursor.execute("SET application_name='masked_intraday_readonly_endpoints'")
            for symbol in SYMBOLS:
                known = pd.DatetimeIndex(cached.loc[cached.symbol.eq(symbol), "ts"])
                missing = request.difference(known)
                found = 0
                for offset in range(0, len(missing), 256):
                    cursor.execute(query, (symbol, [ts.to_pydatetime() for ts in missing[offset:offset + 256]]))
                    batch = cursor.fetchall()
                    rows.extend(dict(zip(fields, values)) for values in batch)
                    found += len(batch)
                    time.sleep(.015)
                print(f"[read-only {table}] {symbol}: {found} additional indexed endpoints", flush=True)
    frame = pd.DataFrame(rows, columns=fields)
    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
    frame["table"] = table
    return frame


def load_bars(sessions: pd.DatetimeIndex, *, cache_only: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    stamps = required_stamps(sessions)
    frames, conflicts = [], {}
    for table in TABLES:
        cached = cached_bars(table)
        extra_path = OUTPUT / f"{table}_extra_endpoints.parquet"
        if extra_path.exists():
            extra = pd.read_parquet(extra_path)
        elif cache_only:
            extra = cached.iloc[:0].copy()
        else:
            extra = fetch_missing(table, cached, stamps)
            extra.to_parquet(extra_path, index=False)
        bars = pd.concat([cached, extra], ignore_index=True)
        grouped = bars.groupby(["symbol", "ts"], sort=False).local_symbol.nunique(dropna=False)
        ambiguous = grouped.loc[grouped.gt(1)]
        conflicts[table] = len(ambiguous)
        if len(ambiguous):
            pairs = set(ambiguous.index)
            bars = bars.loc[~pd.MultiIndex.from_frame(bars[["symbol", "ts"]]).isin(pairs)]
        bars = bars.drop_duplicates(["symbol", "ts"], keep="last")
        frames.append(bars)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["table", "symbol", "ts"]).any():
        raise AssertionError("Non-unique minute endpoint")
    return result, conflicts


def contract_month(code: object, when: pd.Timestamp) -> str | None:
    match = re.search(r"([FGHJKMNQUVXZ])(\d{1,2})$", str(code))
    if match is None:
        return None
    month = MONTH_CODES.index(match.group(1)) + 1
    short = int(match.group(2))
    if len(match.group(2)) == 2:
        year = 2000 + short
    else:
        base = when.year // 10 * 10
        year = min((base - 10 + short, base + short, base + 10 + short),
                   key=lambda value: abs(value - when.year))
    return f"{year:04d}-{month:02d}"


def contract_targets(predictors: pd.DataFrame) -> pd.DataFrame:
    rank_history = pd.read_parquet(MASK / "silver_copper_execution_ranks.parquet")
    mapping = pd.read_parquet(SNAPSHOT / "input_mapping.parquet")
    lookup = {(row.date, row.ticker): row.contract for row in mapping.itertuples()}
    unique = predictors[["session", "signal_date", "root"]].drop_duplicates().copy()
    unique["rank"] = [rank_history.at[row.signal_date, row.root]
                      for row in unique.itertuples()]
    unique["contract"] = [lookup.get((row.signal_date, f"{row.root}{int(row.rank)} Comdty"))
                          if pd.notna(row.rank) else None for row in unique.itertuples()]
    unique["month"] = [contract_month(row.contract, row.signal_date)
                       for row in unique.itertuples()]
    return unique


def price_moves(targets: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    lookup = {(row.table, row.symbol, row.ts): row for row in bars.itertuples()}
    rows = []
    for target in targets.itertuples(index=False):
        for clock in CLOCKS:
            stamp = entry_time(target.session, clock)
            for mode in MODES:
                table = TABLES[int(target.rank) - 1] if pd.notna(target.rank) and target.rank in (1, 2) else None
                choices = ([(table, target.root), (table, MICROS[target.root])]
                           if mode.startswith("rank_proxy") and table else
                           [(source, symbol) for symbol in (target.root, MICROS[target.root]) for source in TABLES]
                           if target.month else [])
                entry = next((bar for source, symbol in choices
                              if (bar := lookup.get((source, symbol, stamp))) is not None
                              and (observed(bar) if mode.endswith("traded") else valid_price(bar))
                              and (mode.startswith("rank_proxy") or
                                   contract_month(bar.local_symbol, stamp) == target.month)), None)
                for hours in HOURS:
                    exit_stamp = stamp + pd.Timedelta(hours=hours)
                    crossed = crosses_maintenance(stamp, exit_stamp)
                    record = {"session": target.session, "signal_date": target.signal_date,
                              "root": target.root, "clock": clock, "mode": mode,
                              "hours": hours, "rank": target.rank, "target_contract": target.contract,
                              "target_month": target.month, "entry_utc": stamp, "exit_utc": exit_stamp,
                              "crosses_maintenance": crossed, "status": "missing entry"}
                    if not mode.startswith("rank_proxy") and not target.month:
                        record["status"] = "historical contract identity unavailable"
                    elif not choices:
                        record["status"] = "selected rank unavailable in stored bar routes"
                    elif entry is not None:
                        record.update(symbol=entry.symbol, local_symbol=entry.local_symbol,
                                      expiry=entry.expiry, source_table=entry.table,
                                      entry_price=float(entry.open),
                                      proxy_month_matches_target=(contract_month(entry.local_symbol, stamp) == target.month)
                                      if target.month else np.nan)
                        exit_bar = lookup.get((entry.table, entry.symbol, exit_stamp))
                        closed = closed_exit_reason(exit_stamp)
                        if closed:
                            record["status"] = closed
                        elif not valid_price(exit_bar):
                            record["status"] = "missing exit"
                        elif str(entry.local_symbol) != str(exit_bar.local_symbol):
                            record["status"] = "series changed contract during hold"
                        elif mode.endswith("traded") and not (observed(entry) and observed(exit_bar)):
                            record["status"] = "endpoint not trade-confirmed"
                        else:
                            record.update(status="matched", exit_price=float(exit_bar.open),
                                          price_return=float(exit_bar.open / entry.open - 1),
                                          entry_trade_confirmed=observed(entry),
                                          exit_trade_confirmed=observed(exit_bar))
                    rows.append(record)
    result = pd.DataFrame(rows)
    for field in ("entry_price", "exit_price", "price_return"):
        if field not in result:
            result[field] = np.nan
    return result


def period_mask(index: pd.DatetimeIndex, name: str) -> np.ndarray:
    if name == "full":
        return np.ones(len(index), dtype=bool)
    if name == "train":
        return index < SPLIT
    if name == "test":
        return index >= SPLIT
    if name == "train_early":
        return index < HALF
    if name == "train_late":
        return (index >= HALF) & (index < SPLIT)
    raise ValueError(name)


def assess(predictors: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    records = []
    for mode in MODES:
        for clock in CLOCKS:
            selected_moves = moves.loc[moves["mode"].eq(mode) & moves.clock.eq(clock)].drop(columns=["mode", "clock", "signal_date"])
            for signal, group in predictors.groupby("signal", sort=False):
                if mode != "rank_proxy_traded" and signal != "portfolio_net":
                    continue
                if signal.endswith("__finalized_reference") and clock not in ("early_utc", "london_open", "us_open"):
                    continue
                paths = evaluate(group.drop(columns="signal"), selected_moves)
                for book, roots in BOOKS.items():
                    if mode != "rank_proxy_traded" and book != "four":
                        continue
                    selected = paths.loc[paths.root.isin(roots)].copy()
                    selected["abs_weight"] = selected.weight.abs()
                    grouped = selected.groupby(["variant", "hours", "session"], sort=False)
                    daily = grouped[["net_return", "slippage_net_return"]].sum(min_count=len(roots))
                    daily["ready"] = grouped.ready.all() & grouped.root.nunique().eq(len(roots))
                    daily["gross_exposure"] = grouped.abs_weight.sum()
                    daily.loc[~daily.ready, ["net_return", "slippage_net_return"]] = np.nan
                    for (variant, hours), trade in daily.groupby(level=[0, 1], sort=False):
                        if signal.endswith("__finalized_reference") and not (
                            variant == "reverse_change_magnitude" and
                            hours == {"early_utc": 2, "london_open": 1, "us_open": 4}[clock]
                        ):
                            continue
                        if mode == "exact_month_traded" and variant not in ("net_magnitude", "reverse_change_magnitude"):
                            continue
                        index = pd.DatetimeIndex(trade.index.get_level_values("session"))
                        periods = ["full"]
                        if mode == "rank_proxy_traded" and (signal in ("portfolio_net", "portfolio_finalized_reference") or
                                                              signal.endswith("__finalized_reference") or
                                                              (book == "four" and "magnitude" in variant)):
                            periods.extend(["train", "test"])
                            if signal not in ("portfolio_net", "portfolio_finalized_reference") and not signal.endswith("__finalized_reference"):
                                periods.extend(["train_early", "train_late"])
                        for period in periods:
                            current = trade.loc[period_mask(index, period)]
                            ready = current.ready
                            active = ready & current.gross_exposure.gt(1e-12)
                            bases = (("net_return", "slippage_net_return")
                                     if signal in ("portfolio_net", "portfolio_finalized_reference") and period == "full"
                                     else ("slippage_net_return",))
                            for basis in bases:
                                records.append({"mode": mode, "clock": clock, "signal": signal,
                                                "variant": variant, "hours": hours, "book": book,
                                                "period": period, "basis": basis,
                                                "possible_days": len(current), "active_days": int(active.sum()),
                                                "coverage": float(ready.mean()) if len(ready) else 0.,
                                                **stats(current[basis])})
            print(f"[scored] {mode} / {clock}", flush=True)
    return pd.DataFrame(records)


def candidate_returns(candidate: pd.Series, predictors: pd.DataFrame, moves: pd.DataFrame,
                      *, mode: str = "rank_proxy_traded", book: str = "four") -> pd.Series:
    signal = candidate.signal
    clock = candidate.clock
    group = predictors.loc[predictors.signal.eq(signal)].drop(columns="signal")
    selected = moves.loc[moves["mode"].eq(mode) & moves.clock.eq(clock)].drop(columns=["mode", "clock", "signal_date"])
    paths = evaluate(group, selected)
    paths = paths.loc[paths.variant.eq(candidate.variant) & paths.hours.eq(candidate.hours)]
    daily = daily_totals(paths, BOOKS[book])
    return daily.slippage_net_return.rename(signal)


def select_combination(metrics: pd.DataFrame, predictors: pd.DataFrame,
                       moves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wanted = metrics.loc[metrics["mode"].eq("rank_proxy_traded") & metrics.book.eq("four")
                         & metrics.basis.eq("slippage_net_return")
                         & ~metrics.signal.isin(("portfolio_net", "portfolio_finalized_reference"))
                         & ~metrics.signal.str.endswith("__finalized_reference")].copy()
    keys = ["signal", "clock", "variant", "hours"]
    wide = wanted.pivot(index=keys, columns="period", values=["sharpe", "days", "active_days", "coverage", "total_return"])
    wide.columns = [f"{item}_{period}" for item, period in wide.columns]
    wide = wide.reset_index()
    wide["robust_train_sharpe"] = wide[["sharpe_train_early", "sharpe_train_late"]].min(axis=1)
    eligible = wide.loc[(wide.days_train >= 140) & (wide.active_days_train >= 90)
                        & (wide.coverage_train >= .55) & (wide.days_train_early >= 50)
                        & (wide.days_train_late >= 50) & (wide.total_return_train_early > 0)
                        & (wide.total_return_train_late > 0)
                        & (wide.robust_train_sharpe > .3)].copy()
    # Holds that cross the exchange break are not candidates for an intraday book.
    break_share = moves.loc[moves["mode"].eq("rank_proxy_traded")].groupby(["clock", "hours"]).crosses_maintenance.mean()
    eligible["break_share"] = [break_share.get((row.clock, row.hours), 1.) for row in eligible.itertuples()]
    eligible = eligible.loc[eligible.break_share.le(.02)]
    eligible = eligible.sort_values(["robust_train_sharpe", "sharpe_train"], ascending=False)
    eligible = eligible.drop_duplicates("signal")
    chosen, series = [], []
    for row in eligible.itertuples(index=False):
        if len(chosen) >= 3:
            break
        spec = pd.Series(row._asdict())
        daily = candidate_returns(spec, predictors, moves)
        trial = [*series, daily]
        common = pd.concat(trial, axis=1).dropna()
        train = common.loc[common.index < SPLIT]
        if len(train) < 140:
            continue
        scaling = 1 / train.std(ddof=1).replace(0, np.nan)
        if scaling.isna().any():
            continue
        scaling /= scaling.sum()
        combined = common.mul(scaling, axis=1).sum(axis=1)
        if chosen:
            previous = pd.concat(series, axis=1).loc[train.index]
            old_scale = 1 / previous.std(ddof=1).replace(0, np.nan)
            old_scale /= old_scale.sum()
            previous_return = previous.mul(old_scale, axis=1).sum(axis=1)
            old_sr = stats(previous_return)["sharpe"]
            if stats(combined.loc[train.index])["sharpe"] < old_sr + .1:
                continue
        chosen.append(spec)
        series.append(daily)
    if not chosen:
        return eligible, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    common = pd.concat(series, axis=1).dropna()
    train = common.loc[common.index < SPLIT]
    scaling = 1 / train.std(ddof=1)
    scaling /= scaling.sum()
    combined = common.mul(scaling, axis=1).sum(axis=1).rename("combined")
    output = pd.DataFrame({"combined": combined})
    output["period"] = np.where(output.index < SPLIT, "train", "test")
    choice = pd.DataFrame(chosen)[keys].copy()
    choice["risk_weight"] = scaling.to_numpy(float)
    summary = []
    for period in ("full", "train", "test"):
        values = combined.loc[period_mask(pd.DatetimeIndex(combined.index), period)]
        summary.append({"period": period, **stats(values)})
    return eligible, choice, pd.DataFrame(summary).assign(common_days=len(combined)), output


def plot_combination(path: pd.DataFrame, destination: Path) -> None:
    if path.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4.5))
    equity = (1 + path.combined).cumprod()
    ax.plot(equity.index, equity, color="#126354", lw=1.6)
    ax.axvline(SPLIT, color="#ad6a32", lw=1.2, ls="--", label="selection frozen")
    ax.set(title="Masked intraday candidate | chronological training and replay", ylabel="Growth of 1")
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, moves: pd.DataFrame, eligible: pd.DataFrame,
                 choice: pd.DataFrame, summary: pd.DataFrame, path: pd.DataFrame,
                 names: tuple[str, ...], conflicts: dict[str, int]) -> None:
    coverage = (moves.groupby(["mode", "clock", "root", "status"]).size()
                .rename("asset_horizon_outcomes").reset_index())
    coverage.to_csv(OUTPUT / "coverage.csv", index=False)
    headline = metrics.loc[metrics["mode"].eq("rank_proxy_traded") & metrics.book.eq("four")
                           & metrics.signal.eq("portfolio_net") & metrics.basis.eq("slippage_net_return")
                           & metrics.period.isin(("full", "train", "test"))]
    headline = headline[["clock", "variant", "hours", "period", "days", "active_days", "coverage", "sharpe", "total_return", "max_drawdown"]]
    headline.to_csv(OUTPUT / "portfolio_clock_grid.csv", index=False)
    comparison = headline.loc[headline.hours.eq(4) & headline.variant.isin(("net_magnitude", "reverse_change_magnitude"))
                              & headline.period.isin(("full", "test"))].copy()
    comparison.to_csv(OUTPUT / "four_hour_clock_comparison.csv", index=False)
    reference = metrics.loc[metrics["mode"].eq("rank_proxy_traded") & metrics.book.eq("four")
                            & metrics.signal.eq("portfolio_finalized_reference")
                            & metrics.basis.eq("slippage_net_return") & metrics.hours.eq(4)
                            & metrics.variant.isin(("net_magnitude", "reverse_change_magnitude"))
                            & metrics.period.isin(("full", "test"))]
    reference = reference[["clock", "variant", "period", "days", "coverage", "sharpe", "total_return"]]
    reference.to_csv(OUTPUT / "finalized_flow_timing_reference.csv", index=False)
    family = metrics.loc[metrics["mode"].eq("rank_proxy_traded") & metrics.book.eq("four")
                         & ~metrics.signal.isin(("portfolio_net", "portfolio_finalized_reference"))
                         & ~metrics.signal.str.endswith("__finalized_reference")
                         & metrics.basis.eq("slippage_net_return") & metrics.clock.eq("us_open")
                         & metrics.hours.eq(4) & metrics.variant.eq("reverse_change_magnitude")
                         & metrics.period.isin(("train", "test"))]
    family = family[["signal", "period", "days", "active_days", "sharpe", "total_return"]]
    family.to_csv(OUTPUT / "us_open_family_check.csv", index=False)
    finalized_family = metrics.loc[
        metrics["mode"].eq("rank_proxy_traded") & metrics.book.eq("four")
        & metrics.signal.str.endswith("__finalized_reference")
        & metrics.basis.eq("slippage_net_return")
        & (((metrics.clock.eq("early_utc") & metrics.hours.eq(2))
            | (metrics.clock.eq("london_open") & metrics.hours.eq(1))
            | (metrics.clock.eq("us_open") & metrics.hours.eq(4))))
        & metrics.variant.eq("reverse_change_magnitude")
        & metrics.period.isin(("train", "test"))
    ][["signal", "clock", "period", "days", "active_days", "sharpe", "total_return"]].copy()
    finalized_family.to_csv(OUTPUT / "finalized_flow_family_sensitivity.csv", index=False)
    if not choice.empty:
        choice.to_csv(OUTPUT / "training_selected_combination.csv", index=False)
        summary.to_csv(OUTPUT / "combination_metrics.csv", index=False)
        path.to_csv(OUTPUT / "combination_daily_returns.csv", index_label="session")
        plot_combination(path, OUTPUT / "combination_equity.png")
    ranked = eligible.head(20)[["signal", "clock", "variant", "hours", "days_train", "robust_train_sharpe", "sharpe_test", "days_test"]]
    exact = metrics.loc[metrics["mode"].eq("exact_month_traded") & metrics.book.eq("four")
                        & metrics.signal.eq("portfolio_net") & metrics.basis.eq("slippage_net_return")
                        & metrics.period.eq("full") & metrics.variant.isin(("net_magnitude", "reverse_change_magnitude"))]
    exact = exact[["clock", "variant", "hours", "days", "coverage", "sharpe"]]
    matched = coverage.loc[coverage.status.eq("matched") & coverage["mode"].isin(("rank_proxy_traded", "exact_month_traded"))]
    totals = matched.groupby(["mode", "clock", "root"]).asset_horizon_outcomes.sum().rename("matched").reset_index()
    identity = moves.loc[moves["mode"].eq("rank_proxy_traded") & moves.hours.eq(4)
                         & moves.target_month.notna() & moves.status.eq("matched")].copy()
    identity["verified_same_month"] = identity.proxy_month_matches_target.eq(True)
    identity = identity.groupby(["clock", "root"]).agg(
        matched_days=("status", "size"), same_month_rate=("verified_same_month", "mean")
    ).reset_index()
    identity.to_csv(OUTPUT / "recent_proxy_identity_audit.csv", index=False)
    chart = '<img src="combination_equity.png" alt="Chronological combined candidate equity">' if not path.empty else ""
    exact_days = int(exact.days.max()) if len(exact) else 0
    test_sharpe = float(summary.loc[summary.period.eq("test"), "sharpe"].iloc[0]) if not summary.empty else np.nan
    deployable = len(choice) >= 2 and exact_days >= 100 and test_sharpe > 0
    conclusion = ("A multi-family candidate cleared the retrospective gates, but still requires prospective paper trading."
                  if deployable else
                  "No deployable combined intraday book passed: either the training-selected leg failed later, "
                  "or exact-month coverage and multi-family evidence are insufficient.")
    text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Five Currents | Masked Intraday Clock Study</title><style>
body{{font:15px/1.5 Georgia,serif;color:#173b35;background:#f5f2ea;max-width:1200px;margin:36px auto;padding:0 20px}}
h1{{font-size:34px}}h2{{border-top:1px solid #c9d3c8;padding-top:20px;margin-top:32px}}
table{{border-collapse:collapse;background:white;width:100%;font:12px/1.4 Consolas,monospace}}
td,th{{padding:6px 8px;border-bottom:1px solid #d9dfd5;text-align:right}}td:first-child,th:first-child{{text-align:left}}
.scroll{{overflow:auto}}.callout{{padding:16px;background:#e7ede5;border-left:4px solid #126354}}img{{width:100%}}
</style></head><body><h1>Five Currents | Intraday clock replay after the liquidity mask</h1>
<p class="callout"><b>Interpretation:</b> {html.escape(conclusion)} The long-history route is a rank-matched
front/nearby <b>proxy</b>, not a certified same-contract replay. Actual exchange-month identity becomes available
only in the recent daily archive; its small exact-match sample is a separate sensitivity.</p>
<p>Archived ten-family, lagged-volume/OI signals are the causal primary test. The finalized daily portfolio is
shown only as a timing reference because historical publication times were not certified. No intraday or daily refit.
Prices are stored one-minute bar opens.
The six clocks are CME Globex open (17:00 Chicago, previous calendar day), 01:00 Chicago (06:00/07:00 UTC),
08:00 and 16:30 London, and 09:30 and 16:00 New York. Holds are 1, 2, 3, 4, 6 or 12 elapsed hours;
closed-market exits are excluded. Both endpoints must report trading for the main result. Net results include
$2.16 per actual micro-lot round trip plus one tick of slippage per side. A price mark is not a guaranteed fill.</p>
<p>Training: {START.date()} to {(SPLIT-pd.Timedelta(days=1)).date()}; replay: {SPLIT.date()} to {END.date()}.
Family/clock/hold selection is confined to the training segment, requires positive results in both training halves,
and excludes holds crossing the Chicago maintenance break. The original strategy development has already seen
some of this history, so the replay is not a genuinely untouched holdout.</p>
<p><b>Release and clock caveat:</b> <a href="https://www.cmegroup.com/market-data/daily-bulletin.html">CME says</a>
its preliminary bulletin updates around midnight Chicago time the following business day and the final bulletin
around 10:00 Chicago time. Finalized prior-day volume/OI cannot be assumed available at the earlier entry clocks.
The US equity close is only an hour before the regular Chicago maintenance break; a one-hour exit is unavailable,
and longer holds that cross the break are not intraday candidates.</p>
<h2>Four-hour portfolio clock comparison</h2><p>Four-market complete-case slippage-net results, the same two
pre-existing position conventions at every clock. Full and later-period rows are shown; all horizons and sign variants
are in the adjacent CSV. Silver-inclusive coverage is in the full metrics file.</p>
<div class="scroll">{comparison.to_html(index=False, escape=True, float_format=lambda x:f'{x:.3f}')}</div>
<h2>Training-selected family candidates</h2><div class="scroll">{ranked.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}')}</div>
<h2>US-open family check, reversed change, four hours</h2><div class="scroll">{family.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}')}</div>
<h2>Training selection and later replay</h2>{chart}<div class="scroll">{choice.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}') if not choice.empty else '<p>None passed.</p>'}</div>
<div class="scroll">{summary.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}') if not summary.empty else ''}</div>
<h2>Finalized-flow timing reference</h2><p>Not a timestamp-certified entry signal, and not eligible for candidate selection.</p>
<div class="scroll">{reference.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}')}</div>
<h3>Finalized-weight family sensitivity</h3><p>Early UTC two hours, London open one hour and US open four hours, reversed change;
these family rows are not eligible for the causal selection above.</p>
<div class="scroll">{finalized_family.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}')}</div>
<h2>Exact-month check</h2><p>Counts are small because historical daily contract identifiers before July 2026 were inferred.
No older inferred identity is treated as a verified month.</p><div class="scroll">{exact.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}')}</div>
<h2>Proxy month identity, recent known-contract days</h2><p>A routed front/nearby bar is not automatically the
most-liquid contract. These rates are the share that actually matched the masked daily contract month.</p>
<div class="scroll">{identity.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}')}</div>
<h2>Coverage by clock and asset</h2><div class="scroll">{totals.to_html(index=False,escape=True)}</div>
<p>Archive conflicts excluded by table: {html.escape(str(conflicts))}. Missing prices are not imputed as zero profit.
No broker API, market-data API, collector or production table was changed. Public clock references:
<a href="https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.contractSpecs.html">CME</a>,
<a href="https://www.londonstockexchange.com/personal-investing/faqs?lang=en">LSE</a>,
<a href="https://www.nyse.com/trade/trading-information">NYSE</a>.</p></body></html>'''
    (OUTPUT / "report.html").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-only", action="store_true", help="Never query the stored bar tables")
    parser.add_argument("--score-only", action="store_true", help="Reuse saved predictors and price outcomes")
    parser.add_argument("--render-only", action="store_true", help="Refresh report from completed local outputs")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    if args.render_only:
        moves = pd.read_parquet(OUTPUT / "price_moves.parquet")
        metrics = pd.read_csv(OUTPUT / "all_metrics.csv")
        eligible = pd.read_csv(OUTPUT / "training_eligible_candidates.csv")
        choice_path = OUTPUT / "training_selected_combination.csv"
        choice = pd.read_csv(choice_path) if choice_path.exists() else pd.DataFrame()
        summary_path = OUTPUT / "combination_metrics.csv"
        summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        path_file = OUTPUT / "combination_daily_returns.csv"
        path = pd.read_csv(path_file, parse_dates=["session"], index_col="session") if path_file.exists() else pd.DataFrame()
        names = tuple(pd.read_parquet(SNAPSHOT / "selection_audit.parquet").economic_family.drop_duplicates())
        conflicts = {table: 0 for table in TABLES}
        write_report(metrics, moves, eligible, choice, summary, path, names, conflicts)
    elif args.score_only:
        predictors, names = load_signals(pd.bdate_range(START, END))
        predictors.to_parquet(OUTPUT / "dated_predictors.parquet", index=False)
        moves = pd.read_parquet(OUTPUT / "price_moves.parquet")
        conflicts = {}
    else:
        sessions = pd.bdate_range(START, END)
        print("[signals] validating current model and ten-family allocation", flush=True)
        predictors, names = load_signals(sessions)
        targets = contract_targets(predictors)
        print("[bars] loading saved endpoints and narrowly querying uncached minutes", flush=True)
        bars, conflicts = load_bars(sessions, cache_only=args.cache_only)
        print(f"[bars] {len(bars):,} unique stored one-minute endpoints", flush=True)
        moves = price_moves(targets, bars)
        moves.to_parquet(OUTPUT / "price_moves.parquet", index=False)
        predictors.to_parquet(OUTPUT / "dated_predictors.parquet", index=False)
        targets.to_parquet(OUTPUT / "contract_targets.parquet", index=False)
    if not args.render_only:
        print(f"[moves] {len(moves):,} contract/clock/hold outcomes", flush=True)
        metrics = assess(predictors, moves)
        metrics.to_csv(OUTPUT / "all_metrics.csv", index=False)
        eligible, choice, summary, path = select_combination(metrics, predictors, moves)
        eligible.to_csv(OUTPUT / "training_eligible_candidates.csv", index=False)
        write_report(metrics, moves, eligible, choice, summary, path, names, conflicts)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "archive": str(SNAPSHOT), "masked_rank_source": str(MASK),
                "model_retrained": False, "collector_touched": False, "broker_api_calls": False,
                "database_writes": False, "database_reads_this_run": not (args.cache_only or args.score_only or args.render_only),
                "endpoint_provenance": "archived parquet caches plus read-only bars_1m and bars_1m_2m indexed endpoints",
                "additional_stored_endpoints": {table: len(pd.read_parquet(OUTPUT / f"{table}_extra_endpoints.parquet"))
                                                if (OUTPUT / f"{table}_extra_endpoints.parquet").exists() else 0
                                                for table in TABLES},
                "source_hashes": {str(path): sha256(path) for path in
                                  (SNAPSHOT / "weights.parquet", SNAPSHOT / "input_mapping.parquet",
                                   MASK / "silver_copper_execution_ranks.parquet", CONCEPTS, Path(__file__))},
                "output_hashes": {path.name: sha256(path) for path in OUTPUT.iterdir()
                                  if path.is_file() and path.name != "manifest.json"}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[combination]", summary.to_string(index=False) if not summary.empty else "none passed", flush=True)
    print("[output]", OUTPUT / "report.html", flush=True)


if __name__ == "__main__":
    main()
