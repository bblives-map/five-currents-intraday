"""Read-only market-clock test of archived Five Currents daily positions.

No daily model is refit. Minute prices come only from stored PostgreSQL bar tables
or a previously exported endpoint cache; neither a broker nor a market API is used.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from five_currents_front_intraday import (
    HOURS, MICROS, ROOTS, VARIANTS, evaluate, observed, same_intraday_instrument,
    valid_price,
)
from five_currents_opening_metrics import collect_metrics
from five_currents_opening_study import allocated_sleeves, opening_predictors, stats


SOURCE = Path("five_currents_opening_complete_20261011")
SIGNALS = Path("five_currents_opening_comparison_20261011/all_opening_predictors.parquet")
SELECTION = Path("five_currents_dashboard/snapshots/000032_2026-10-09")
CONCEPTS = Path("five_currents_opening_study_final_20261011/concept_weights.npz")
CLOCKS = {
    "london_open": ("Europe/London", 8, 0),
    "london_close": ("Europe/London", 16, 30),
    "us_open": ("America/New_York", 9, 30),
}
ROUTES = {"front": "bars_1m", "nearby_proxy": "bars_1m_2m"}
FOUR = ("CL", "NG", "GC", "HG")
BASE_POLICY = "lagged_volume_oi"
REFERENCE_POLICY = "original_final_flows_reference"
VALUE_COLUMNS = ("gross_return", "net_return", "slippage_net_return")


def clock_entry(session: object, clock: str) -> pd.Timestamp:
    zone, hour, minute = CLOCKS[clock]
    day = pd.Timestamp(session).date()
    stamp = datetime(day.year, day.month, day.day, hour, minute, tzinfo=ZoneInfo(zone))
    return pd.Timestamp(stamp).tz_convert("UTC")


def crosses_maintenance(entry: pd.Timestamp, exit_stamp: pd.Timestamp) -> bool:
    """Flag a hold crossing the standard 16:00-17:00 Chicago Globex break."""
    chicago = ZoneInfo("America/Chicago")
    first = entry.tz_convert(chicago).date()
    last = exit_stamp.tz_convert(chicago).date()
    for offset in range((last - first).days + 1):
        day = first + timedelta(days=offset)
        start = pd.Timestamp(datetime(day.year, day.month, day.day, 16, tzinfo=chicago))
        end = start + pd.Timedelta(hours=1)
        if entry < end.tz_convert("UTC") and exit_stamp > start.tz_convert("UTC"):
            return True
    return False


def closed_exit_reason(exit_stamp: pd.Timestamp) -> str | None:
    """Do not treat a carried/stale price mark during the standard break as a fill."""
    local = exit_stamp.tz_convert("America/Chicago")
    weekday = local.weekday()
    clock = (local.hour, local.minute)
    if weekday == 5 or (weekday == 6 and clock < (17, 0)) or (weekday == 4 and clock >= (16, 0)):
        return "exit after weekly close"
    if (16, 0) <= clock < (17, 0):
        return "exit during maintenance"
    return None


def load_predictors() -> tuple[pd.DataFrame, tuple[str, ...], dict[str, str]]:
    archived = pd.read_parquet(SIGNALS)
    archived = archived.loc[
        archived.signal.eq("portfolio_net") & archived.policy.isin((BASE_POLICY, REFERENCE_POLICY))
    ].copy()
    sessions = pd.DatetimeIndex(archived.session.unique()).sort_values()
    if not archived.signal_date.lt(archived.session).all():
        raise AssertionError("A daily forecast is dated on or after its trading session")

    selection = pd.read_parquet(SELECTION / "selection_audit.parquet")
    scaling = pd.read_parquet(SELECTION / "product_scaling.parquet")
    names = tuple(selection.economic_family.drop_duplicates())
    targets = pd.read_parquet(SOURCE / "lagged_volume_oi_targets.parquet")
    concepts = np.load(CONCEPTS)
    final, sleeves, _ = allocated_sleeves(concepts["lagged"], names, targets.index, selection, scaling)
    np.testing.assert_allclose(final, targets.reindex(columns=ROOTS).to_numpy(), atol=1e-12)

    independently_built = opening_predictors(targets, sessions)
    recorded = archived.loc[archived.policy.eq(BASE_POLICY)].sort_values(["session", "root"])
    independently_built = independently_built.sort_values(["session", "root"])
    for column in ("net_magnitude", "change_magnitude", "net_sign", "change_sign"):
        np.testing.assert_allclose(
            independently_built[column].to_numpy(), recorded[column].to_numpy(), atol=1e-12
        )
    family_frames = []
    for position, name in enumerate(names):
        weights = pd.DataFrame(sleeves[position], index=targets.index, columns=ROOTS)
        frame = opening_predictors(weights, sessions)
        family_frames.append(frame.assign(policy=BASE_POLICY, signal=name))
    predictors = pd.concat([archived, *family_frames], ignore_index=True)
    if predictors.duplicated(["policy", "signal", "session", "root"]).any():
        raise AssertionError("Duplicate signal/asset/session predictors")
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        SIGNALS, SELECTION / "selection_audit.parquet", SELECTION / "product_scaling.parquet",
        CONCEPTS, SOURCE / "lagged_volume_oi_targets.parquet",
    )}
    return predictors, names, hashes


def requested_stamps(sessions: pd.DatetimeIndex) -> list[datetime]:
    stamps = {clock_entry(day, clock) + pd.Timedelta(hours=hour)
              for day in sessions for clock in CLOCKS for hour in (0, *HOURS)}
    return [stamp.to_pydatetime() for stamp in sorted(stamps)]


def read_stored_bars(table: str, stamps: list[datetime]) -> pd.DataFrame:
    from psycopg2 import sql
    from five_currents_daily.db import readonly_connection

    if table not in ROUTES.values():
        raise ValueError("Only archived one-minute bar tables may be read")
    fields = ("ts", "open", "close", "volume", "bar_count", "symbol", "local_symbol", "expiry")
    rows = []
    query = sql.SQL(
        "SELECT ts,open,close,volume,bar_count,symbol,local_symbol,expiry "
        "FROM public.{} WHERE symbol=%s AND ts=ANY(%s::timestamptz[])"
    ).format(sql.Identifier(table))
    with readonly_connection(statement_timeout_ms=10000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET max_parallel_workers_per_gather=0")
            cursor.execute("SET lock_timeout='1s'")
            cursor.execute("SET application_name='market_hours_intraday_readonly'")
            for symbol in (*ROOTS, *MICROS.values()):
                count = 0
                for offset in range(0, len(stamps), 256):
                    cursor.execute(query, (symbol, stamps[offset:offset + 256]))
                    batch = cursor.fetchall()
                    rows.extend(dict(zip(fields, values)) for values in batch)
                    count += len(batch)
                    time.sleep(.03)
                print(f"[stored {table}] {symbol}: {count:,} exact-minute endpoints", flush=True)
    bars = pd.DataFrame(rows, columns=fields)
    bars["ts"] = pd.to_datetime(bars.ts, utc=True)
    if bars.duplicated(["symbol", "ts"]).any():
        raise AssertionError(f"Duplicate {table} bar endpoints")
    return bars


def clock_moves(bars: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    lookup = {(row.symbol, row.ts): row for row in bars.itertuples()}
    rows = []
    for session in sessions:
        for clock in CLOCKS:
            entry_stamp = clock_entry(session, clock)
            for root in ROOTS:
                entry = next((lookup.get((symbol, entry_stamp)) for symbol in (root, MICROS[root])
                              if valid_price(lookup.get((symbol, entry_stamp)))), None)
                for hour in HOURS:
                    exit_stamp = entry_stamp + pd.Timedelta(hours=hour)
                    result = {"session": session, "clock": clock, "root": root, "hours": hour,
                              "entry_utc": entry_stamp, "exit_utc": exit_stamp,
                              "crosses_maintenance": crosses_maintenance(entry_stamp, exit_stamp),
                              "status": "missing entry"}
                    if entry is not None:
                        result.update(symbol=entry.symbol, local_symbol=entry.local_symbol,
                                      expiry=entry.expiry, entry_price=float(entry.open))
                        exit_bar = lookup.get((entry.symbol, exit_stamp))
                        closed = closed_exit_reason(exit_stamp)
                        if closed:
                            result["status"] = closed
                        elif not valid_price(exit_bar):
                            result["status"] = "missing exit"
                        elif not same_intraday_instrument(entry, exit_bar):
                            result["status"] = "contract changed during hold"
                        else:
                            result.update(status="matched", exit_price=float(exit_bar.open),
                                          price_return=float(exit_bar.open / entry.open - 1),
                                          entry_trade_confirmed=observed(entry),
                                          exit_trade_confirmed=observed(exit_bar))
                    rows.append(result)
    result = pd.DataFrame(rows)
    for column in ("entry_price", "exit_price", "price_return"):
        if column not in result:
            result[column] = np.nan
    return result


def common_clock_metrics(curves: pd.DataFrame) -> pd.DataFrame:
    results = []
    keys = ("route", "policy", "signal", "book", "variant", "hours")
    subset = curves.loc[curves.signal.eq("portfolio_net") & curves.policy.eq(BASE_POLICY)]
    for key, group in subset.groupby(list(keys)):
        for basis in ("net_return", "slippage_net_return"):
            shared = group.pivot(index="session", columns="clock", values=basis).reindex(columns=CLOCKS)
            shared = shared.dropna(how="any")
            for clock in CLOCKS:
                results.append(dict(zip(keys, key)) | {"clock": clock, "cost_basis": basis,
                                  "start": shared.index.min() if len(shared) else None,
                                  "end": shared.index.max() if len(shared) else None,
                                  **stats(shared[clock])})
    return pd.DataFrame(results)


def perform_study(predictors: pd.DataFrame, anchors: dict[str, pd.DataFrame], output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions = pd.DatetimeIndex(predictors.session.unique()).sort_values()
    metrics, curves, coverage = [], [], []
    for route, bars in anchors.items():
        moves = clock_moves(bars, sessions)
        moves.to_parquet(output / f"{route}_price_moves.parquet", index=False)
        counts = moves.groupby(["clock", "root", "hours", "status"]).size().rename("sessions").reset_index()
        coverage.append(counts.assign(route=route))
        print(f"[prices] {route}: {len(moves):,} candidate asset/clock/horizon outcomes", flush=True)
        for clock in CLOCKS:
            selected_moves = moves.loc[moves.clock.eq(clock)].drop(columns="clock")
            for (policy, signal), group in predictors.groupby(["policy", "signal"], sort=False):
                paths = evaluate(group.drop(columns=["policy", "signal"]), selected_moves)
                first_metric, first_curve = len(metrics), len(curves)
                collect_metrics(paths, policy, signal, metrics, curves, cost_only=signal != "portfolio_net")
                for record in metrics[first_metric:]:
                    record.update(route=route, clock=clock)
                for record in curves[first_curve:]:
                    record["route"], record["clock"] = route, clock
            print(f"[signals] {route} / {clock}: portfolio plus 10 selected families", flush=True)
    metric_frame = pd.DataFrame(metrics)
    curve_frame = pd.concat(curves, ignore_index=True)
    coverage_frame = pd.concat(coverage, ignore_index=True)
    shared = common_clock_metrics(curve_frame)
    metric_frame.to_csv(output / "all_metrics.csv", index=False)
    curve_frame.to_parquet(output / "portfolio_daily_returns.parquet", index=False)
    coverage_frame.to_csv(output / "price_coverage.csv", index=False)
    shared.to_csv(output / "common_clock_metrics.csv", index=False)
    return metric_frame, curve_frame, coverage_frame, shared


def _matrix(metrics: pd.DataFrame, *, route: str, clock: str, book: str, signal: str,
            variants: tuple[str, ...], min_days: int = 60) -> str:
    selected = metrics.loc[
        metrics.route.eq(route) & metrics.clock.eq(clock) & metrics.book.eq(book)
        & metrics.policy.eq(BASE_POLICY) & metrics.signal.eq(signal)
        & metrics.cost_basis.eq("net_return")
        & metrics.variant.isin(variants)
    ].copy()
    selected["display"] = selected.apply(
        lambda row: (f"{row.sharpe:.2f} ({int(row.days)})"
                     if row.days >= min_days and np.isfinite(row.sharpe)
                     else f"insufficient ({int(row.days)})"), axis=1
    )
    pivot = selected.pivot(index="variant", columns="hours", values="display")
    pivot = pivot.reindex(index=variants, columns=HOURS)
    return pivot.to_html(border=0, escape=True, na_rep="n/a")


def _family_matrix(metrics: pd.DataFrame, names: tuple[str, ...], *, route: str, clock: str,
                   book: str, variant: str, min_days: int = 60) -> str:
    selected = metrics.loc[
        metrics.route.eq(route) & metrics.clock.eq(clock) & metrics.book.eq(book)
        & metrics.policy.eq(BASE_POLICY) & metrics.signal.isin(names)
        & metrics.variant.eq(variant)
        & metrics.cost_basis.eq("net_return")
    ].copy()
    selected["display"] = selected.apply(
        lambda row: (f"{row.sharpe:.2f} ({int(row.active_days)}/{int(row.days)})"
                     if row.days >= min_days and row.active_days >= min_days and np.isfinite(row.sharpe)
                     else f"insufficient ({int(row.active_days)}/{int(row.days)})"), axis=1
    )
    pivot = selected.pivot(index="signal", columns="hours", values="display")
    return pivot.reindex(index=names, columns=HOURS).to_html(border=0, escape=True, na_rep="n/a")


def _common_table(shared: pd.DataFrame, route: str, book: str) -> str:
    selected = shared.loc[
        shared.route.eq(route) & shared.book.eq(book) & shared.cost_basis.eq("net_return")
        & shared.variant.isin(("net_magnitude", "change_magnitude",
                             "reverse_net_magnitude", "reverse_change_magnitude"))
    ].copy()
    selected = selected.loc[selected.hours.isin((1, 3, 6, 12))]
    selected["Sharpe"] = selected.apply(
        lambda row: f"{row.sharpe:.2f}" if row.days >= 60 and np.isfinite(row.sharpe) else "insufficient", axis=1
    )
    selected = selected.rename(columns={"variant": "Position", "hours": "Hours", "clock": "Entry", "days": "Days"})
    return selected[["Position", "Hours", "Entry", "Days", "Sharpe"]].to_html(index=False, border=0, escape=True)


def yearly_metrics(curves: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = ("route", "clock", "policy", "signal", "book", "variant", "hours")
    for key, group in curves.groupby(list(keys)):
        for year, period in group.groupby(group.session.dt.year):
            for basis in ("net_return", "slippage_net_return"):
                records.append(dict(zip(keys, key)) | {"year": int(year), "cost_basis": basis,
                                                        **stats(period[basis])})
    return pd.DataFrame(records)


def trade_confirmed_four_hour(predictors: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Require both endpoint bars to report trading, not just a price mark."""
    predictor = predictors.loc[
        predictors.policy.eq(BASE_POLICY) & predictors.signal.eq("portfolio_net")
    ].drop(columns=["policy", "signal"])
    records = []
    for route in ROUTES:
        moves = pd.read_parquet(output / f"{route}_price_moves.parquet")
        moves = moves.loc[moves.hours.eq(4)].copy()
        unconfirmed = moves.status.eq("matched") & ~(
            moves.entry_trade_confirmed.eq(True) & moves.exit_trade_confirmed.eq(True)
        )
        moves.loc[unconfirmed, "status"] = "endpoint not trade-confirmed"
        for clock in CLOCKS:
            selected = moves.loc[moves.clock.eq(clock)].drop(columns="clock")
            paths = evaluate(predictor, selected)
            first = len(records)
            collect_metrics(paths, BASE_POLICY, "portfolio_net", records, [], cost_only=True)
            for record in records[first:]:
                record.update(route=route, clock=clock)
    return pd.DataFrame(records)


def render_report(output: Path, metrics: pd.DataFrame, curves: pd.DataFrame,
                  coverage: pd.DataFrame, shared: pd.DataFrame, names: tuple[str, ...],
                  annual: pd.DataFrame, confirmed: pd.DataFrame) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axis, (route, variant) in zip(axes.flat, (
        ("front", "net_magnitude"), ("front", "reverse_change_magnitude"),
        ("nearby_proxy", "net_magnitude"), ("nearby_proxy", "reverse_change_magnitude"),
    )):
        group = curves.loc[
            curves.route.eq(route) & curves.policy.eq(BASE_POLICY)
            & curves.signal.eq("portfolio_net") & curves.book.eq("four")
            & curves.variant.eq(variant) & curves.hours.eq(4)
        ]
        pivot = group.pivot(index="session", columns="clock", values="net_return").reindex(columns=CLOCKS)
        common = pivot.dropna().index
        for clock, color in zip(CLOCKS, ("#1a7063", "#bb7937", "#284e76")):
            values = pivot[clock].where(pivot.index.isin(common))
            axis.plot(values.index, (1 + values).cumprod() * 100, label=clock.replace("_", " "),
                      color=color, lw=1.3)
        axis.set_title(f"{route.replace('_', ' ')} / {variant.replace('_', ' ')} / 4h / n={len(common)}")
        axis.grid(alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    chart = output / "common_date_four_hour_equity.png"
    fig.savefig(chart, dpi=150, bbox_inches="tight")
    plt.close(fig)

    sections = []
    for route in ROUTES:
        sections.append(f"<h2>{html.escape(route.replace('_', ' ').title())} Price Series</h2>")
        for book in ("four", "five"):
            sections.append(f"<h3>{book.title()}-market portfolio, commission-net Sharpe (matched days)</h3>")
            for clock in CLOCKS:
                sections.append(f"<h4>{html.escape(clock.replace('_', ' ').title())}</h4>")
                sections.append(_matrix(metrics, route=route, clock=clock, book=book,
                                        signal="portfolio_net", variants=VARIANTS))
        sections.append("<h3>Common-date comparison, four markets</h3>")
        sections.append(_common_table(shared, route, "four"))
    family_sections = []
    for route in ROUTES:
        family_sections.append(f"<h3>{html.escape(route.replace('_', ' ').title())} price series</h3>")
        for clock in CLOCKS:
            family_sections.append(f"<h4>{html.escape(clock.replace('_', ' ').title())}</h4>")
            for variant in VARIANTS:
                family_sections.append(
                    f"<details><summary>{html.escape(variant.replace('_', ' '))}</summary>"
                    + _family_matrix(metrics, names, route=route, clock=clock,
                                     book="four", variant=variant) + "</details>"
                )
    covers = coverage.loc[coverage.status.eq("matched") & coverage.hours.isin((1, 4, 6, 12))]
    cover_table = covers.pivot_table(index=["route", "clock", "root"], columns="hours",
                                    values="sessions", aggfunc="sum", fill_value=0)
    reference = metrics.loc[
        metrics.policy.eq(REFERENCE_POLICY) & metrics.signal.eq("portfolio_net")
        & metrics.book.eq("four") & metrics.variant.eq("net_magnitude")
        & metrics.cost_basis.eq("net_return") & metrics.hours.isin((1, 3, 6, 12))
    ][["route", "clock", "hours", "days", "sharpe", "total_return"]].copy()
    reference["sharpe"] = reference.apply(
        lambda row: f"{row.sharpe:.2f}" if row.days >= 60 and np.isfinite(row.sharpe) else "insufficient", axis=1
    )
    clock_control = shared.loc[
        shared.book.eq("four") & shared.variant.eq("reverse_change_magnitude")
        & shared.hours.eq(4) & shared.cost_basis.eq("slippage_net_return")
    ][["route", "clock", "days", "sharpe", "total_return", "max_drawdown"]]
    annual_control = annual.loc[
        annual.policy.eq(BASE_POLICY) & annual.signal.eq("portfolio_net")
        & annual.book.eq("four") & annual.clock.eq("us_open")
        & annual.variant.eq("reverse_change_magnitude") & annual.hours.eq(4)
        & annual.cost_basis.eq("slippage_net_return")
    ][["route", "year", "days", "sharpe", "total_return", "max_drawdown"]]
    asset_control = metrics.loc[
        metrics.policy.eq(BASE_POLICY) & metrics.signal.eq("portfolio_net")
        & metrics.book.isin(ROOTS) & metrics.clock.eq("us_open")
        & metrics.variant.eq("reverse_change_magnitude") & metrics.hours.eq(4)
        & metrics.cost_basis.eq("slippage_net_return")
    ][["route", "book", "days", "sharpe", "total_return"]]
    confirmed_control = confirmed.loc[
        confirmed.book.isin(("four", "five")) & confirmed.clock.eq("us_open")
        & confirmed.variant.eq("reverse_change_magnitude")
        & confirmed.cost_basis.eq("slippage_net_return")
    ][["route", "book", "days", "active_days", "sharpe"]].copy()
    confirmed_control["sharpe"] = confirmed_control.apply(
        lambda row: f"{row.sharpe:.2f}" if row.days >= 60 and np.isfinite(row.sharpe) else "insufficient", axis=1
    )
    html_text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Five Currents | Three Market Clocks</title><style>
:root{{--ink:#193b35;--accent:#146a61;--paper:#f6f4eb;--line:#d2d9ce}}
body{{background:var(--paper);color:var(--ink);font:15px/1.55 Georgia,serif;margin:0}}
main{{max-width:1260px;margin:auto;padding:40px 28px 100px}}h1{{font-size:40px;line-height:1.1}}
h2{{border-top:1px solid var(--line);padding-top:24px;margin-top:40px}}h4{{margin-bottom:5px}}
.lead{{font-size:18px;max-width:960px}}.callout{{background:#e8ede4;padding:16px 20px;border-left:4px solid var(--accent)}}
table{{border-collapse:collapse;font:12px/1.4 Consolas,monospace;margin:10px 0 22px;min-width:630px}}
td,th{{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
td:first-child,th:first-child{{text-align:left}}th{{background:#e7ebe2}}.scroll{{overflow:auto}}
img{{width:100%;height:auto}}details{{margin:8px 0;padding:8px 12px;background:white;border:1px solid var(--line)}}
summary{{cursor:pointer;font-weight:bold}}.small{{font-size:13px;color:#566c63}}
@media print{{body{{background:white}}h2,h3,h4{{break-after:avoid}}table,img,details{{break-inside:avoid}}}}
</style></head><body><main><p>FIVE CURRENTS / INTRADAY CLOCK STUDY</p>
<h1>Three Clocks, One Frozen Daily Model</h1>
<p class="lead">We entered the archived daily signal at London open, London close and the US equity open, then exited to flat after 1, 2, 3, 4, 6 or 12 hours. All existing causal family admission, coefficients and risk scaling were frozen.</p>
<div class="callout"><b>Primary interpretation:</b> prior-known price and one-session-lagged volume/OI. The full finalized-flow signal is shown only as a publication-time sensitivity. These are commodity futures traded at clock anchors, not London Stock Exchange or NYSE instruments. A weak or short matched sample is not an investable track record.</div>
<p><b>Clock definitions:</b> 08:00 and 16:30 Europe/London (<a href="https://www.londonstockexchange.com/personal-investing/faqs?lang=en">LSE hours</a>); 09:30 America/New_York (<a href="https://www.nyse.com/trade/trading-information">NYSE hours</a>). Historical DST is applied separately in each zone. Exact one-minute opening marks are required at both ends; the same listed contract must be present. No interpolation, missing-day cash imputation or hindsight route selection.</p>
<p><b>Signal variants:</b> net position and day-on-day position change, each by signed magnitude or equal-sign allocation, plus their exact reversals. Sign and magnitude have different exposure levels. Positions use the latest completed daily observation before the session; family rows are the actually admitted, allocated concept sleeves. No intraday retuning or best-horizon selection.</p>
<p><b>Costs:</b> $2.16 per actual micro-lot round trip. The headline is commission-net; separate gross and one-tick-per-side slippage sensitivities are in the CSV. All active contracts are closed at the specified exit, rather than held to the daily model's close. Minute-bar price marks are not guaranteed fills.</p>
<h2>Comparable Four-Hour Paths</h2><p class="small">Four-market paths use the same priced dates across all three clocks within each panel. Missing days remain gaps. The nearby series is a proxy, not a repaired front-month contract.</p>
<img src="{chart.name}" alt="Four-hour net and reversed-change equity by market clock">
<h2>What the Clock Comparison Shows</h2><p>The most coherent exploratory pattern is a reversal of the new daily position change at US open over adjacent 2-6 hour holds. This was one of the pre-specified variants, but it was not selected on a separate holdout. The four-hour same-date comparison below includes an additional one-tick-per-side slippage allowance.</p>
<div class="scroll">{clock_control.to_html(index=False,border=0,na_rep='n/a',float_format=lambda x:f'{x:.3f}')}</div>
<h3>Annual and asset checks, US open / reversed change / four hours</h3>
<p class="small">These are diagnostics, not another optimization step. A good pooled result can conceal a weak year or a losing commodity.</p>
<div class="scroll">{annual_control.to_html(index=False,border=0,na_rep='n/a',float_format=lambda x:f'{x:.3f}')}</div>
<div class="scroll">{asset_control.to_html(index=False,border=0,na_rep='n/a',float_format=lambda x:f'{x:.3f}')}</div>
<h3>Trade-confirmed endpoint sensitivity</h3><p>Both stored one-minute bars must report a trade count or volume at the entry and exit. This is a stricter coverage test, not proof of fillability; unconfirmed marks are excluded, never treated as zero return. Low sample counts limit inference.</p>
<div class="scroll">{confirmed_control.to_html(index=False,border=0,escape=True)}</div>
<h2>Portfolio Results</h2><p class="small">Each cell is annualized daily Sharpe followed by matched sessions in parentheses; fewer than 60 sessions are labelled insufficient. Different horizons can have different samples. The same-date table below each price route is the safer clock comparison.</p>
<div class="scroll">{''.join(sections)}</div>
<h2>Selected Signal Families</h2><p class="small">Each family is the selected, allocated historical sleeve, not a raw always-on signal. Cells show Sharpe (active sessions / total priced sessions), with fewer than 60 active sessions labelled insufficient. The ten-family tables are expandable; the full asset and variant grid is in all_metrics.csv.</p>
<div class="scroll">{''.join(family_sections)}</div>
<h2>Coverage and Availability</h2><p>Front-month five-market samples must include silver when it is active. The nearby-series control improves observability but changes the instrument. A six-hour London-close exit usually falls inside the <a href="https://www.cmegroup.com/markets/metals/precious/gold.timeAndSales.html">regular Chicago maintenance break</a> and is unavailable; a few London/US DST-mismatch dates do not make an interpretable sample. Twelve-hour London-close and US-open holds can cross that break; the positions remain open and incur gap risk.</p>
<div class="scroll">{cover_table.to_html(border=0)}</div>
<h3>Finalized-flow reference, not a timestamp-certified entry signal</h3>
<div class="scroll">{reference.to_html(index=False,border=0,na_rep='n/a',float_format=lambda x:f'{x:.3f}')}</div>
<p class="small">The daily-signal archive ends 2026-10-09. It does not establish historical publication timestamps of volume and OI. No live broker, tick/L2 collector, market API, dashboard or production database state was changed.</p>
</main></body></html>'''
    path = output / "Five_Currents_Three_Market_Clocks.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def finalize_report(output: Path, predictors: pd.DataFrame, names: tuple[str, ...],
                    metrics: pd.DataFrame, curves: pd.DataFrame,
                    coverage: pd.DataFrame, shared: pd.DataFrame) -> Path:
    annual = yearly_metrics(curves)
    confirmed = trade_confirmed_four_hour(predictors, output)
    annual.to_csv(output / "yearly_metrics.csv", index=False)
    confirmed.to_csv(output / "trade_confirmed_four_hour_metrics.csv", index=False)
    return render_report(output, metrics, curves, coverage, shared, names, annual, confirmed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--front-cache", type=Path)
    parser.add_argument("--nearby-cache", type=Path)
    parser.add_argument("--report-only", action="store_true", help="Refresh diagnostics from completed local outputs")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.report_only:
        if not output.is_dir():
            raise ValueError("A completed output directory is required for --report-only")
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        predictors = pd.read_parquet(output / "dated_predictors.parquet")
        metrics = pd.read_csv(output / "all_metrics.csv")
        curves = pd.read_parquet(output / "portfolio_daily_returns.parquet")
        coverage = pd.read_csv(output / "price_coverage.csv")
        shared = pd.read_csv(output / "common_clock_metrics.csv")
        report = finalize_report(output, predictors, tuple(manifest["families"]),
                                 metrics, curves, coverage, shared)
        manifest["report_refreshed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["source_code_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                          for path in (Path(__file__), Path("five_currents_front_intraday.py"),
                                                       Path("five_currents_opening_study.py"))}
        manifest["output_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in output.iterdir() if path.is_file() and path != manifest_path}
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        print(f"[report refreshed from local outputs] {report}", flush=True)
        return
    output.mkdir(exist_ok=False)
    predictors, names, input_hashes = load_predictors()
    predictors.to_parquet(output / "dated_predictors.parquet", index=False)
    sessions = pd.DatetimeIndex(predictors.session.unique()).sort_values()
    stamps = requested_stamps(sessions)
    print(f"[signals] {sessions.min().date()} to {sessions.max().date()}, {len(sessions)} sessions; {len(stamps):,} distinct price timestamps", flush=True)
    anchors = {}
    for route, table in ROUTES.items():
        cache = args.front_cache if route == "front" else args.nearby_cache
        anchors[route] = pd.read_parquet(cache) if cache else read_stored_bars(table, stamps)
        anchors[route].to_parquet(output / f"{route}_minute_anchors.parquet", index=False)
        if anchors[route].empty:
            raise ValueError(f"No archived minute bars found for {route}")
    metrics, curves, coverage, shared = perform_study(predictors, anchors, output)
    report = finalize_report(output, predictors, names, metrics, curves, coverage, shared)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "signal_sessions": [str(sessions.min().date()), str(sessions.max().date())],
        "clock_local_times": CLOCKS, "hours": HOURS, "variants": VARIANTS,
        "primary_policy": BASE_POLICY, "reference_policy": REFERENCE_POLICY,
        "markets": ROOTS, "families": names, "bar_tables": ROUTES,
        "market_api_calls": False, "database_writes": False, "model_refits": False,
        "database_access": "read-only indexed minute endpoints" if not (args.front_cache and args.nearby_cache) else "none",
        "input_sha256": input_hashes,
        "source_code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (Path(__file__), Path("five_currents_front_intraday.py"),
                                            Path("five_currents_opening_study.py"))},
        "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in output.iterdir() if path.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    summary = shared.loc[shared.route.eq("front") & shared.book.eq("four")
                         & shared.variant.eq("net_magnitude") & shared.hours.eq(4)
                         & shared.cost_basis.eq("net_return")]
    print("[common-date 4h net position / front four]\n" + summary[["clock", "days", "sharpe", "total_return"]].to_string(index=False), flush=True)
    print(f"[report] {report}", flush=True)


if __name__ == "__main__":
    main()
