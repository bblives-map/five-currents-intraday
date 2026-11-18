"""Fresh daily-signal intraday study, using only existing front-month minute bars."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_front_intraday import (
    ROOTS, HOURS, MICROS, VARIANTS, ROUND_TRIP_FEE, daily_predictors,
    entry_time, front_month_moves, evaluate, summarize,
)


def extract_existing_bars(sessions, output):
    from five_currents_daily.db import readonly_connection
    timestamps = sorted({(entry_time(day)+pd.Timedelta(hours=h)).to_pydatetime()
                         for day in sessions for h in (0, *HOURS)})
    rows, profiles = [], []
    with readonly_connection(statement_timeout_ms=7500) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET max_parallel_workers_per_gather = 0")
            cursor.execute("SET lock_timeout = '1s'")
            cursor.execute("SET application_name = 'front_intraday_research_readonly'")
            for symbol in (*ROOTS, *MICROS.values()):
                cursor.execute("SELECT ts FROM public.bars_1m WHERE symbol=%s ORDER BY ts LIMIT 1", (symbol,))
                first = cursor.fetchone()
                cursor.execute("SELECT ts FROM public.bars_1m WHERE symbol=%s ORDER BY ts DESC LIMIT 1", (symbol,))
                last = cursor.fetchone()
                if not first or not last:
                    continue
                first, last = pd.Timestamp(first[0]), pd.Timestamp(last[0])
                profiles.append({"symbol": symbol, "first": first, "last": last})
                wanted = [stamp for stamp in timestamps if first <= pd.Timestamp(stamp) <= last]
                count = 0
                for offset in range(0, len(wanted), 384):
                    batch = wanted[offset:offset+384]
                    cursor.execute("SELECT ts,open,volume,bar_count,symbol,local_symbol,expiry "
                                   "FROM public.bars_1m WHERE symbol=%s AND ts=ANY(%s::timestamptz[])",
                                   (symbol, batch))
                    fetched = cursor.fetchall()
                    rows.extend(dict(zip(("ts", "open", "volume", "bar_count", "symbol", "local_symbol", "expiry"), row))
                                for row in fetched)
                    count += len(fetched)
                    time.sleep(.05)
                print(f"[stored front bars] {symbol}: {count:,} endpoints; {first.date()} to {last.date()}", flush=True)
    if not rows:
        raise ValueError("No stored front-month minute prices were found")
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
    if frame.duplicated(["symbol", "ts"]).any():
        raise ValueError("The front-month minute series has duplicate timestamps")
    frame.to_parquet(output / "front_minute_endpoints.parquet", index=False)
    pd.DataFrame(profiles).to_csv(output / "minute_source_coverage.csv", index=False)
    return frame


def report(output, metrics, curves, yearly, moves, predictors, confirmed_metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = metrics.loc[metrics.book.eq("portfolio")].copy()
    grid = primary.pivot(index="variant", columns="hours", values="net_sharpe").reindex(VARIANTS)
    gross = primary.pivot(index="variant", columns="hours", values="gross_sharpe").reindex(VARIANTS)
    samples = primary.pivot(index="variant", columns="hours", values="days").reindex(VARIANTS)
    confirmed = confirmed_metrics.loc[confirmed_metrics.book.eq("portfolio")]
    confirmed_samples = confirmed.pivot(index="variant", columns="hours", values="days").reindex(VARIANTS)
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True)
    colors = plt.cm.viridis(np.linspace(.05, .95, len(HOURS)))
    for ax, variant in zip(axes.flat, VARIANTS):
        for hour, color in zip(HOURS, colors):
            selected = curves.loc[curves.book.eq("portfolio") & curves.variant.eq(variant) & curves.hours.eq(hour)]
            ax.plot(selected.session, selected.equity, label=f"{hour}h", color=color, lw=1.)
        ax.set_title(variant.replace("_", " "), loc="left")
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    image = "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()
    coverage = moves.groupby(["root", "hours", "status"]).size().rename("days").reset_index()
    fmt = lambda value: f"{value:.3f}"
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Five Currents / Front-Month Intraday Edge</title>
<style>body{{background:#faf9f4;color:#193c36;font:15px/1.6 Georgia,serif;margin:0}}main{{max-width:1180px;padding:40px;margin:auto}}table{{border-collapse:collapse;font:12px/1.5 Consolas,monospace;width:100%}}td,th{{padding:8px;border-bottom:1px solid #d3dbce;text-align:right}}th{{background:#e6ece0}}img{{width:100%}}.scroll{{overflow:auto}}</style></head><body><main>
<h1>Does the Daily Signal Have Intraday Edge?</h1>
<p>CL, NG, GC, SI and HG. Stored front-month minute prices, independent of the daily backtest's maturity choices. Entry at 01:00 Chicago (06:00 UTC in summer, 07:00 UTC in winter); exit to flat after 1, 2, 3, 4, 6 or 12 hours on the same UTC date.</p>
<p>Daily targets are frozen. Net position uses the last target dated strictly before entry day. Position change is the change versus the preceding morning's target, not an old-contract exit/new-contract entry pair. Sign uses +20%, -20% or cash per commodity; magnitude preserves the cached economic weights or their differences. Reversed cases negate exposure before costs. No horizon, alpha, parameter or allocation is fitted.</p>
<h2>Portfolio Sharpe / After Commission</h2>{grid.to_html(border=0,na_rep='n/a',float_format=fmt)}
<p>Commission is $2.16 per actual micro lot for the complete round trip, with no extra x10. These headline net results deduct commission only. Separate price-slippage sensitivity is retained in the detailed metrics. Magnitude and sign have different exposure levels; annual volatility and mean gross exposure are provided alongside total returns.</p>
<h2>Portfolio Sharpe / Before Costs</h2>{gross.to_html(border=0,na_rep='n/a',float_format=fmt)}
<h2>Independent Matched Days</h2>{samples.to_html(border=0)}
<p>Each holding period stands alone. No date is rejected because another horizon is missing. Standard front-month bars are preferred at entry, otherwise the stored micro front-month stream is used. Entry source is chosen without looking at the exit. Both prices use the same stream and intraday instrument; no daily-book maturity matching, second-month routing or roll optimization is applied.</p>
<p>The main question is price prediction: exact stored minute-open price marks are used without a volume or trade-count filter. Some marks have zero trade volume/count; they measure a stored price move, not a confirmed fill opportunity. A separate sensitivity requires a trade at both endpoints. Missing prices remain unknown, not flat, and are never redistributed to other assets. Complete portfolio observations require every active commodity. The asset tables retain each commodity's longer available history. Equity compounds available marked outcomes, with missing sessions left as chart gaps; it is not a complete deployable track record where coverage is missing.</p>
<h2>Equity / All Eight Variants</h2><img src="{image}" alt="All intraday strategy equity curves">
<h2>Portfolio Statistics</h2><div class="scroll">{primary.to_html(index=False,border=0,na_rep='n/a',float_format=fmt)}</div>
<h2>Asset Statistics</h2><div class="scroll">{metrics.loc[metrics.book.isin(ROOTS)].to_html(index=False,border=0,na_rep='n/a',float_format=fmt)}</div>
<h2>Four-Market Coverage Control / No Silver</h2><p>Original CL/NG/GC/HG weights are retained with no redistribution. This exposes the impact of silver's missing price history, not an optimized replacement portfolio.</p><div class="scroll">{metrics.loc[metrics.book.eq('portfolio_ex_silver')].to_html(index=False,border=0,na_rep='n/a',float_format=fmt)}</div>
<h2>Trade-Confirmed Endpoint Sensitivity</h2><p>This is not a filter on the main edge test. Its substantially smaller sample cannot validate executable performance across the full price history. Raw, unsuppressed figures are retained in trade_confirmed_summary.csv for audit; Sharpe estimates from fewer than 20 days should not be interpreted.</p>{confirmed_samples.to_html(border=0)}
<h2>Year-by-Year Portfolio</h2><div class="scroll">{yearly.loc[yearly.book.eq('portfolio')].to_html(index=False,border=0,na_rep='n/a',float_format=fmt)}</div>
<h2>Endpoint Coverage</h2>{coverage.to_html(index=False,border=0)}
<p>Signal-date range: {predictors.signal_date.min().date()} to {predictors.signal_date.max().date()}. Prior-day signal availability by entry time is assumed; historical OI publication vintages are not independently audited. All 48 portfolio combinations are reported, not a hindsight-selected winner. No new market data is requested and no collector, broker, scheduler or live dashboard is changed.</p>
</main></body></html>'''
    path = output / "Five_Currents_Front_Month_Intraday_Report.html"
    path.write_text(html, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("five_currents_intraday_exit_20261009_matched/frozen_weights.parquet"))
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bar-cache", type=Path, help="Reuse a front endpoint parquet without database access")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=False)
    weights = pd.read_parquet(args.weights)
    predictors = daily_predictors(weights, args.start, args.end)
    sessions = pd.DatetimeIndex(predictors.session.unique()).sort_values()
    print(f"[signals] {sessions.min().date()} to {sessions.max().date()}; prior-day cached targets; no refits", flush=True)
    if args.bar_cache:
        bars = pd.read_parquet(args.bar_cache)
        bars.to_parquet(output / "front_minute_endpoints.parquet", index=False)
    else:
        bars = extract_existing_bars(sessions, output)
    first_price_day = bars.ts.min().tz_convert("UTC").normalize().tz_localize(None)
    last_price_day = bars.ts.max().tz_convert("UTC").normalize().tz_localize(None)
    sessions = sessions[(sessions >= first_price_day) & (sessions <= last_price_day)]
    predictors = predictors.loc[predictors.session.isin(sessions)].copy()
    predictors.to_parquet(output / "daily_predictors.parquet", index=False)
    print(f"[analysis window] {sessions.min().date()} to {sessions.max().date()}; all available front price marks", flush=True)
    moves = front_month_moves(bars, sessions)
    moves.to_parquet(output / "front_price_moves.parquet", index=False)
    paths = evaluate(predictors, moves)
    paths.to_parquet(output / "strategy_asset_returns.parquet", index=False)
    metrics, curves, yearly = summarize(paths)
    metrics.to_csv(output / "summary_metrics.csv", index=False)
    curves.to_parquet(output / "equity_curves.parquet", index=False)
    yearly.to_csv(output / "yearly_metrics.csv", index=False)
    moves.groupby(["root", "hours", "status"]).size().rename("days").to_csv(output / "coverage.csv")
    confirmed_moves = front_month_moves(bars, sessions, require_trade=True)
    confirmed_metrics, _, _ = summarize(evaluate(predictors, confirmed_moves))
    confirmed_metrics.to_csv(output / "trade_confirmed_summary.csv", index=False)
    confirmed_moves.to_parquet(output / "trade_confirmed_price_moves.parquet", index=False)
    path = report(output, metrics, curves, yearly, moves, predictors, confirmed_metrics)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "weights_source": str(args.weights.resolve()),
                "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                "start": args.start, "end": args.end, "entry": "01:00 America/Chicago", "holding_hours": HOURS,
                "variants": VARIANTS, "market_api_calls": False, "model_refits": False,
                "production_changes": False, "database_access": "none" if args.bar_cache else "read-only indexed endpoint reads",
                "execution_route": "stored bars_1m front series; never match the daily backtest maturity",
                "price_policy": "primary: all finite positive stored price marks; sensitivity: trade-confirmed endpoints",
                "commission_usd_per_micro_round_trip_lot": ROUND_TRIP_FEE,
                "source_code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                       for name in ("five_currents_front_intraday.py", "run_five_currents_front_intraday.py")},
                "outputs_sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                   for file in output.iterdir() if file.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    primary = metrics.loc[metrics.book.eq("portfolio")]
    for column in ("gross_sharpe", "net_sharpe", "slippage_net_sharpe", "days"):
        print("\n[portfolio] "+column, flush=True)
        print(primary.pivot(index="variant", columns="hours", values=column).reindex(VARIANTS).round(3).to_string(), flush=True)
    print("[report] "+str(path), flush=True)


if __name__ == "__main__":
    main()
