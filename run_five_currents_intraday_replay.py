from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psycopg2 import sql

from five_currents_daily.dashboard_core import json_bytes
from five_currents_daily.db import readonly_connection
from five_currents_intraday_replay import (
    ROOTS, MICROS, HORIZONS, entry_time, identity_key, signal_legs, match_bars,
    daily_paths, summarize, performance, frozen_execution_routes, recost_matches,
)


def cached_routes(replay):
    return frozen_execution_routes(replay)


def extract_endpoints(legs, output):
    timestamps = sorted({(x + pd.Timedelta(minutes=h)).to_pydatetime()
                         for x in legs.entry_utc.unique() for h in (0, *HORIZONS)})
    results, profiles = [], []
    with readonly_connection(statement_timeout_ms=20000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET max_parallel_workers_per_gather = 0")
            cursor.execute("SET lock_timeout = '1s'")
            for table in ("bars_1m", "bars_1m_2m"):
                for symbol in (*ROOTS, *MICROS.values()):
                    cursor.execute(sql.SQL("SELECT ts FROM public.{} WHERE symbol=%s ORDER BY ts LIMIT 1").format(sql.Identifier(table)), (symbol,))
                    first = cursor.fetchone()
                    cursor.execute(sql.SQL("SELECT ts FROM public.{} WHERE symbol=%s ORDER BY ts DESC LIMIT 1").format(sql.Identifier(table)), (symbol,))
                    last = cursor.fetchone()
                    if not first or not last:
                        continue
                    first_ts, last_ts = pd.Timestamp(first[0]), pd.Timestamp(last[0])
                    wanted = [x for x in timestamps if first_ts <= pd.Timestamp(x) <= last_ts]
                    profiles.append({"table": table, "symbol": symbol, "first": first_ts, "last": last_ts})
                    retrieved = 0
                    for offset in range(0, len(wanted), 256):
                        batch = wanted[offset:offset+256]
                        cursor.execute(sql.SQL("SELECT ts,open,volume,bar_count,local_symbol,expiry FROM public.{} WHERE symbol=%s AND ts=ANY(%s::timestamptz[])").format(sql.Identifier(table)), (symbol, batch))
                        chunk = cursor.fetchall()
                        retrieved += len(chunk)
                        results.extend({"table": table, "symbol": symbol, "ts": r[0], "open": r[1],
                                        "volume": r[2], "bar_count": r[3], "local_symbol": r[4], "expiry": r[5]}
                                       for r in chunk)
                    print(f"[cached bars] {table} {symbol}: {retrieved:,} endpoint bars; {first_ts.date()} to {last_ts.date()}", flush=True)
    bars = pd.DataFrame(results)
    bars["ts"] = pd.to_datetime(bars.ts, utc=True)
    root_for = {**{x:x for x in ROOTS}, **{v:k for k, v in MICROS.items()}}
    bars["root"] = bars.symbol.map(root_for)
    bars["contract_key"] = [identity_key(row.root, row.local_symbol, row.ts) for row in bars.itertuples()]
    bars.to_parquet(output / "cached_minute_endpoints.parquet", index=False)
    pd.DataFrame(profiles).to_csv(output / "minute_source_coverage.csv", index=False)
    return bars


def figure_image(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="#faf9f4")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def make_report(output, metrics, curves, paths, matches, manifest):
    diagnostic = metrics.loc[metrics.book.eq("portfolio_ex_silver")]
    cl = metrics.loc[metrics.book.eq("CL") & metrics["mode"].eq("target") & metrics.horizon.eq(360)].iloc[0]
    silver_days = int(metrics.loc[metrics.book.eq("SI") & metrics["mode"].eq("target"), "days"].max())
    book_days = int(metrics.loc[metrics.book.eq("portfolio") & metrics["mode"].eq("target"), "days"].max())
    if diagnostic.net_sharpe.lt(0).all():
        conclusion = "Neither the full daily target nor its incremental change earns a positive net Sharpe at any tested horizon in the complete four-asset sample. Most results are already weak before costs: friction worsens the outcome but does not fully explain it."
    else:
        conclusion = "Results are exploratory comparisons of unchanged daily signals, not a new horizon-selection fit; evaluate both pre-cost and after-cost results alongside coverage."
    conclusion += (f" CL's full-target 360-minute case has net Sharpe {cl.net_sharpe:.2f} "
                   f"({cl.stressed_sharpe:.2f} with doubled friction) across {int(cl.days)} matched days on its separate, larger sample. "
                   "This modest asset-specific result does not establish a five-market intraday edge.")
    figures = []
    colors = plt.cm.viridis(np.linspace(.1, .9, len(HORIZONS)))
    for mode in ("target", "change"):
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        for color, horizon in zip(colors, HORIZONS):
            selected = curves.loc[(curves["mode"] == mode) & (curves.book == "portfolio_ex_silver") & (curves.horizon == horizon)]
            if len(selected):
                axes[0].plot(selected.session, selected.equity, color=color, label=f"{horizon} min")
                axes[1].plot(selected.session, selected.drawdown*100., color=color)
        axes[0].set_title(f"{mode.title()} / CL-NG-GC-HG: matched-observation equity after costs", loc="left")
        axes[0].set_ylabel("Growth of 100")
        axes[1].set_ylabel("Drawdown %")
        if len(curves.loc[(curves["mode"] == mode) & (curves.book == "portfolio_ex_silver")]):
            axes[0].legend(ncol=7, fontsize=8, frameon=False)
        for ax in axes:
            ax.grid(alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
        figures.append(f'<img src="{figure_image(fig)}" alt="{mode} equity curves">')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for mode, ax in zip(("target", "change"), axes):
        data = metrics.loc[(metrics["mode"] == mode) & metrics.book.isin(ROOTS)].copy()
        data.loc[data.days.lt(20), "net_sharpe"] = np.nan
        data = data.pivot(index="book", columns="horizon", values="net_sharpe").reindex(ROOTS)
        im = ax.imshow(data.to_numpy(float), aspect="auto", cmap="RdYlGn", vmin=-2., vmax=2.)
        ax.set_xticks(range(len(HORIZONS)), HORIZONS)
        ax.set_yticks(range(len(ROOTS)), ROOTS)
        ax.set_title(mode.title()+" / net Sharpe")
        ax.set_xlabel("Holding period (minutes)")
        for i in range(len(ROOTS)):
            for j in range(len(HORIZONS)):
                value = data.iloc[i,j]
                if pd.notna(value):
                    ax.text(j,i,f"{value:.2f}",ha="center",va="center",fontsize=8)
    fig.colorbar(im, ax=axes.tolist(), shrink=.7)
    figures.append(f'<img src="{figure_image(fig)}" alt="Asset Sharpe comparisons">')
    display = metrics.copy()
    for column in ("gross_sharpe", "net_sharpe", "stressed_sharpe"):
        display.loc[display.days.lt(20), column] = np.nan
    for col in ("cagr", "total_return", "max_drawdown", "win_rate", "median_gain", "median_loss"):
        display[col] = display[col].map(lambda x:f"{x*100:.3f}%" if pd.notna(x) else "n/a")
    keep = ["mode","book","horizon","days","coverage_pct","gross_sharpe","net_sharpe","stressed_sharpe",
            "total_return","max_drawdown","win_rate","median_gain","median_loss"]
    primary = display.loc[display.book.eq("portfolio_ex_silver"), keep].to_html(index=False, float_format=lambda x:f"{x:.2f}", border=0, na_rep="n/a")
    full = display.loc[display.book.eq("portfolio"), keep].to_html(index=False, float_format=lambda x:f"{x:.2f}", border=0, na_rep="n/a")
    assets = display.loc[display.book.isin(ROOTS), keep].to_html(index=False, float_format=lambda x:f"{x:.2f}", border=0, na_rep="n/a")
    coverage = matches.groupby(["root","horizon","status"]).size().rename("legs").reset_index().to_html(index=False,border=0)
    ready = paths.groupby(["mode","session","root"]).ready.all().reset_index()
    counts = ready.groupby(["mode","root"]).ready.agg(["sum","count"]).reset_index()
    counts["coverage_pct"] = 100.*counts["sum"]/counts["count"]
    monthly_rows = []
    yearly_rows = []
    for keys, group in curves.groupby(["mode","book","horizon"]):
        series = group.set_index("session").net_return
        for period, by in (("month", "M"), ("year", "Y")):
            for label, returns in series.groupby(series.index.to_period(by)):
                row = dict(zip(("mode","book","horizon"),keys))
                row.update(period=str(label), **performance(returns))
                (monthly_rows if period == "month" else yearly_rows).append(row)
    pd.DataFrame(monthly_rows).to_csv(output / "monthly_metrics.csv", index=False)
    annual = pd.DataFrame(yearly_rows)
    annual.to_csv(output / "yearly_metrics.csv", index=False)
    annual.loc[annual.days.lt(20), "net_sharpe"] = np.nan
    annual_table = annual.loc[annual.book.eq("portfolio_ex_silver"), ["mode","horizon","period","days","net_sharpe","total_return","max_drawdown"]].to_html(index=False,float_format=lambda x:f"{x:.4f}",border=0,na_rep="n/a") if len(annual) else "No complete portfolio observations."
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Five Currents / Early Daily Signal, Intraday Exit</title>
<style>body{{font:15px/1.65 Georgia,serif;color:#193c36;background:#faf9f4;margin:0}}main{{max-width:1160px;margin:auto;padding:45px}}h1{{font-size:42px;line-height:1.1}}h2{{margin-top:42px;font-size:25px}}.eyebrow{{font:12px Consolas,monospace;letter-spacing:2px}}.note{{border-left:3px solid #a88950;padding:13px 20px;background:#f0ecdf}}img{{max-width:100%;display:block;margin:24px 0}}table{{width:100%;border-collapse:collapse;font:11px/1.5 Consolas,monospace}}th,td{{padding:8px;border-bottom:1px solid #d9ded1;text-align:right}}th{{background:#e7ede3}}.scroll{{overflow:auto}}code{{font-family:Consolas,monospace}}a{{color:#176b5c}}</style></head>
<body><main><div class="eyebrow">FIVE CURRENTS / CACHED-DATA TIMING STUDY</div><h1>Early Daily Signal,<br>Intraday Exit</h1>
<p>Entry: 01:00 America/Chicago, equivalent to 06:00 UTC during US daylight saving and 07:00 UTC in winter. Exit after exactly 30, 60, 90, 120, 180, 240 or 360 minutes.</p>
<p class="note"><b>Conditional timing experiment, not a new alpha fit.</b> The daily weights, selections, refit cadence and volatility scaling are frozen from publication {manifest['source_publication']}. Historical daily input publication timestamps are not complete: this assumes the prior session's signals were ready by the proposed entry time. Historical final/revised OI could have become available later. It does not prove this schedule was executable in real time.</p>
<h2>Findings</h2><p>{conclusion}</p>
<h2>Construction</h2><p><b>Target</b> takes the full prior-close daily target from flat, then exits to flat. <b>Change</b> trades the difference versus the previous daily target, then exits that incremental exposure to flat. Reducing a long is a short incremental trade. A contract switch has two exact-month legs; it is not treated as a same-contract weight change.</p>
<p>Cached economic weights and the daily backtest's exact contract path are unchanged. No expiry rule, liquidity crossover or overnight holding lock is recomputed. Weights are not renormalized to fill the intraday risk budget. This is not a reconstruction of the dashboard's capital-dependent lot/capacity constraints or a $1m VaR simulation. No new daily or intraday model is trained.</p>
<p>Standard minute bars are preferred where available; otherwise the micro must match the same delivery month. Both entry and exit use that same instrument and table. Source choice uses entry information only. Prices are the exact timestamp's minute-open, with adverse slippage and fees, not that minute's close. Execution endpoints must have positive volume and bar count: forward-filled/stale bars are excluded.</p>
<p>Each holding period is evaluated independently; a missing exit at another horizon cannot exclude its observation. Portfolio comparisons require every non-zero asset/roll leg at that horizon. Missing outcomes are not zero-filled, not substituted across expiries and not leveraged away. Equity follows matched observations only; absent dates represent unknown outcomes, not a continuous deployable track record. Sharpe is annualized with 252 sessions; medians are positive/negative matched daily returns. Drawdown is on matched end-of-trade observations, not worst intratrade drawdown. Different asset samples must not be compared as identical histories.</p>
<h2>Results / Four-Asset Coverage Diagnostic</h2><p>CL, NG, GC and HG retain their original cached weights. Silver is excluded solely to expose its limited minute-bar coverage; its weight is not redistributed. Each horizon uses its own complete four-asset dates. This is not a replacement five-asset production strategy.</p><div class="scroll">{primary}</div>{figures[0]}{figures[1]}
<h2>Full Five-Asset Book / Insufficient Coverage</h2><p>The complete five-asset target has only {book_days} matched days; silver alone has {silver_days}. No standard SI minute history is stored, and cached SIL front/second-calendar-month bars rarely cover the more distant liquid silver delivery month selected by the daily book. Observed trades are also required at every endpoint. This is an exact-contract coverage problem, not evidence of a silver signal's failure. Sharpe is suppressed below 20 matched days; raw calculations remain in the research CSV for audit, not as meaningful estimates.</p><div class="scroll">{full}</div>
<h2>Asset Results</h2>{figures[2]}<div class="scroll">{assets}</div><h2>Coverage</h2><div class="scroll">{counts.to_html(index=False,border=0,float_format=lambda x:f'{x:.1f}')}</div>
<p>The overlapping cache begins no earlier than 20 June 2024 and ends with the completed 8 October 2026 session. Earlier daily signals exist, but corresponding minute bars are absent; no pre-2024 intraday results are invented. Thin complete-book samples and a sweep of seven correlated horizons make apparent winners exploratory.</p>
<h2>Cost Sensitivity</h2><p>Commission is $2.16 per actual micro lot for the complete round trip, without an additional standard-contract conversion. Slippage per side is separate: CL $0.01; NG $0.001; GC $0.10; HG $0.0005; SI $0.005. Standard bars, where used, are same-month price proxies for the micro exposure. Stress doubles total modeled friction. Minute trade prices cannot establish actual spread, depth, order impact or fund-scale capacity.</p>
<p>Reference checks: <a href="https://www.interactivebrokers.com/en/pricing/commissions-futures.php">IBKR commissions</a>, <a href="https://www.interactivebrokers.com/en/accounts/fees/COMEX.php">COMEX fees</a>, <a href="https://www.interactivebrokers.com/en/accounts/fees/NYMEX.php">NYMEX fees</a>, <a href="https://www.cmegroup.com/content/dam/cmegroup/rulebook/COMEX/1a/121.pdf">Micro Silver rulebook</a>. Historical fees are modeled consistently rather than reconstructed invoice by invoice.</p>
<h2>Year-by-Year / Four-Asset Diagnostic</h2><div class="scroll">{annual_table}</div><h2>Execution Exclusions</h2><div class="scroll">{coverage}</div>
<p class="eyebrow">LOCAL RESEARCH ONLY / NO COLLECTOR, SCHEDULE, DASHBOARD OR BROKER CHANGES</p></main></body></html>'''
    (output / "Five_Currents_Intraday_Exit_Report.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Read-only cached-signal intraday exit experiment; no APIs or retuning")
    parser.add_argument("--dashboard-dir", type=Path, default=Path("five_currents_dashboard"))
    parser.add_argument("--start", default="2024-06-20")
    parser.add_argument("--end", default="2026-10-08")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bar-cache", type=Path, help="Reuse a previously extracted local endpoint parquet; no DB reads")
    args = parser.parse_args()
    output = args.output_dir or Path("five_currents_intraday_exit_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    output.mkdir(exist_ok=False)
    root = args.dashboard_dir.resolve()
    with sqlite3.connect((root / "publications.sqlite3").as_uri()+"?mode=ro", uri=True) as connection:
        number, date = connection.execute("SELECT id,signal_date FROM publications ORDER BY id DESC LIMIT 1").fetchone()
    snapshot = root / "snapshots" / f"{number:06d}_{date}"
    weights = pd.read_parquet(snapshot / "weights.parquet")
    cache = root / "private_replay_cache.pkl"
    info = json.loads(cache.with_suffix(".sha256.json").read_text())
    cache_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    if cache_hash != info["sha256"]:
        raise ValueError("Local replay cache checksum mismatch")
    with cache.open("rb") as handle:
        replay = pickle.load(handle)
    print(f"[signals] publication {number}; frozen cached weights; no model fits", flush=True)
    identities = cached_routes(replay).reindex(weights.index)
    identities.to_parquet(output / "cached_execution_routes.parquet")
    weights.to_parquet(output / "frozen_weights.parquet")
    legs = signal_legs(weights, identities, start=args.start, end=args.end)
    sessions = weights.index[(weights.index >= pd.Timestamp(args.start)) & (weights.index <= pd.Timestamp(args.end))]
    legs.to_parquet(output / "signal_legs.parquet", index=False)
    if args.bar_cache:
        bars = pd.read_parquet(args.bar_cache)
        bars.to_parquet(output / "cached_minute_endpoints.parquet", index=False)
        profile = args.bar_cache.parent / "minute_source_coverage.csv"
        if profile.exists():
            pd.read_csv(profile).to_csv(output / profile.name, index=False)
        print(f"[cached bars] reusing {len(bars):,} local endpoint rows; no database reads", flush=True)
    else:
        bars = extract_endpoints(legs, output)
    matches = recost_matches(match_bars(legs, bars), execution_basis="micro")
    matches.to_parquet(output / "execution_audit.parquet", index=False)
    paths = daily_paths(matches, sessions)
    paths.to_parquet(output / "asset_daily_returns.parquet", index=False)
    metrics, curves = summarize(paths)
    metrics.to_csv(output / "summary_metrics.csv", index=False)
    curves.to_parquet(output / "equity_curves.parquet", index=False)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_publication": number,
                "source_snapshot": str(snapshot), "source_cache": str(cache), "cache_sha256": cache_hash,
                "start": args.start, "end": args.end, "entry": "01:00 America/Chicago",
                "holding_minutes": HORIZONS, "model_refit": False, "new_market_api_calls": False,
                "execution_contracts": "micro lots on exact frozen daily-backtest delivery months",
                "execution_route_policy": "frozen replay.execution.identity; no rerouting",
                "commission_usd_per_micro_round_trip_lot": 2.16,
                "sample_policy": "independent horizon",
                "database_access": "local endpoint parquet; no PVX reads" if args.bar_cache else "read-only indexed endpoint reads", "signal_availability": "conditional, not independently historically verified",
                "weights_sha256": hashlib.sha256((snapshot / "weights.parquet").read_bytes()).hexdigest()}
    if args.bar_cache:
        manifest["minute_cache_source"] = str(args.bar_cache.resolve())
        manifest["minute_cache_source_sha256"] = hashlib.sha256(args.bar_cache.read_bytes()).hexdigest()
    make_report(output, metrics, curves, paths, matches, manifest)
    manifest["output_sha256"] = {file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in output.iterdir() if file.is_file()}
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    print(metrics.loc[metrics.book.eq("portfolio_ex_silver"), ["mode","horizon","days","coverage_pct","gross_sharpe","net_sharpe","stressed_sharpe","total_return","max_drawdown"]].to_string(index=False), flush=True)
    print("[report] " + str((output / "Five_Currents_Intraday_Exit_Report.html").resolve()), flush=True)


if __name__ == "__main__":
    main()
