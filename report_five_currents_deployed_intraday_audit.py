"""Render the completed intraday audit without rebuilding signals or reading PVX."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__import__("os").environ.get("FIVE_CURRENTS_DATA_ROOT", str(Path(__file__).resolve().parent))) / "five_currents_intraday_direction_audit_20261027"
COLORS = {"Portfolio": "#176b5c", "Fixed four legs": "#ac7535", "Volume six legs": "#6583a4",
          "CL": "#ac643e", "NG": "#52805b", "GC": "#a98a31", "HG": "#376b91"}


def equity(values):
    valid = values.notna()
    growth = (1+values.fillna(0)).cumprod()
    drawdown = growth/np.maximum(1., growth.cummax())-1
    return (100*growth).where(valid), (100*drawdown).where(valid)


def figure(frame, assets=False):
    if assets:
        fig, axes = plt.subplots(4, 1, figsize=(12.8, 11.5), sharex=True)
        for ax, root in zip(axes, ("CL", "NG", "GC", "HG")):
            growth, _ = equity(frame[root])
            ax.plot(growth.index, growth, color=COLORS[root], lw=1.5)
            ax.set_title(root, loc="left", weight="bold", fontsize=12)
            ax.set_ylabel("Growth of 100")
            ax.axhline(100, color="#b6bcb6", lw=.6)
        fig.suptitle("Intraday sleeve by commodity | each asset's own priced sessions", weight="bold", fontsize=15)
        filename = "asset_equity.png"
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.3), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        for name in ("Portfolio", "Fixed four legs", "Volume six legs"):
            growth, dd = equity(frame[name])
            axes[0].plot(growth.index, growth, color=COLORS[name], lw=1.6,
                         label=f"{name} ({frame[name].notna().sum()} priced sessions)")
            if name == "Portfolio":
                axes[1].plot(dd.index, dd, color=COLORS[name], lw=1.1)
                axes[1].fill_between(dd.index, dd, 0, color=COLORS[name], alpha=.1)
        axes[0].legend(frameon=False, fontsize=9)
        axes[0].set_ylabel("Growth of 100")
        axes[1].set_ylabel("Drawdown %")
        fig.suptitle("The displayed intraday book | ten frozen rules, after modelled costs", weight="bold", fontsize=15)
        filename = "portfolio_equity.png"
    for ax in axes:
        ax.grid(alpha=.16)
        for edge in ("top", "right"):
            ax.spines[edge].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/filename, dpi=165, facecolor="white")
    plt.close(fig)
    return filename


def table(frame):
    return '<div class="scroll">'+frame.to_html(index=False, border=0, float_format=lambda x:f"{x:.3f}")+'</div>'


def main():
    manifest = json.loads((OUT/"manifest.json").read_text())
    frame = pd.read_parquet(OUT/"portfolio_daily_returns.parquet")
    metrics = pd.read_csv(OUT/"performance_metrics.csv")
    rules = pd.read_csv(OUT/"frozen_rules.csv")
    years = pd.read_csv(OUT/"yearly_metrics.csv")
    today = pd.read_json(OUT/"today_legs.json")
    reconciled = pd.read_csv(OUT/"today_issued_weight_reconciliation.csv")
    maximum_error = reconciled.difference.abs().max()
    assert maximum_error < 1e-9
    images = [figure(frame), figure(frame, assets=True)]
    order = ["Portfolio", "CL", "NG", "GC", "HG", "Fixed four legs", "Volume six legs"]
    shown = metrics.set_index("series").loc[order].reset_index()
    columns = ["series", "observed_days", "net_sharpe", "total_return", "max_drawdown", "win_rate", "median_gain", "median_loss", "delay15_sharpe", "sr_without_top5"]
    shown = shown[columns].copy()
    for col in ("total_return", "max_drawdown", "win_rate", "median_gain", "median_loss"):
        shown[col] *= 100
    shown = shown.rename(columns={"series":"Book", "observed_days":"Priced days", "net_sharpe":"Net SR",
                                  "total_return":"Observed return %", "max_drawdown":"Max DD %", "win_rate":"Win %",
                                  "median_gain":"Median up day %", "median_loss":"Median down day %",
                                  "delay15_sharpe":"+15m SR", "sr_without_top5":"SR excluding top 5 days"})
    gc = today.loc[today.root.eq("GC")].copy()
    gc_mtm = gc.gross_mtm.sum()
    definitions = rules[["root", "family", "signal_class", "clock", "hold_minutes"]].copy()
    definitions["series"] = definitions.root+":"+definitions.family
    legs = definitions.merge(metrics[["series", "net_sharpe", "delay15_sharpe", "observed_days"]],
                              on="series", how="left", validate="one_to_one")
    leg_display = legs[["root","family","signal_class","clock","hold_minutes","net_sharpe","delay15_sharpe","observed_days"]]
    annual = years.loc[years.series.isin(order[:5])].pivot(index="series", columns="year", values="net_sharpe").reindex(order[:5]).reset_index()
    utc = pd.to_datetime(gc.planned_entry, utc=True)
    gc["Entry UTC"] = utc.dt.strftime("%d %b %H:%M")
    gc["Entry SGT"] = utc.dt.tz_convert("Asia/Singapore").dt.strftime("%d %b %H:%M")
    end = pd.to_datetime(gc.planned_exit_nominal, utc=True)
    gc["Exit UTC"] = end.dt.strftime("%d %b %H:%M")
    gc["Exit SGT"] = end.dt.tz_convert("Asia/Singapore").dt.strftime("%d %b %H:%M")
    gc["Weight %"] = gc.weight*100
    gc["Model gross MTM %"] = gc.gross_mtm*100
    gc["Mark UTC"] = pd.to_datetime(gc.mark_time, utc=True).dt.strftime("%d %b %H:%M")
    gold_table = gc[["family","Weight %","Entry UTC","Exit UTC","Entry SGT","Exit SGT",
                     "status_now","entry_price","mark_price","Mark UTC","Model gross MTM %"]]
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Five Currents | Intraday Replay Audit</title>
<style>:root{{--ink:#203c3b;--paper:#f5f2eb;--line:#d9ded3}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 Georgia,serif}}header{{padding:38px max(5vw,24px);background:#173e38;color:#fffdf7}}header h1{{font-size:38px;margin:5px 0}}header p{{margin:0;color:#d0dfd3}}main{{max-width:1240px;margin:auto;padding:18px 28px 65px}}h2{{font-size:27px;border-bottom:1px solid var(--line);padding-bottom:7px;margin-top:34px}}p{{max-width:1100px}}.note{{background:#fffdf7;border-left:4px solid #b38d40;padding:15px 20px;margin:22px 0}}img{{display:block;width:100%;background:white;margin:22px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;background:#fffdf8;font:11px/1.5 Verdana,sans-serif;width:100%}}td,th{{padding:8px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}}td:first-child,th:first-child{{text-align:left}}.eyebrow{{font:11px Verdana,sans-serif;letter-spacing:.15em}}small{{font:12px Verdana,sans-serif}}a{{color:#176b5c}}@media print{{header{{padding:20px}}main{{padding:0}}img{{break-inside:avoid}}}}</style></head><body>
<header><div class="eyebrow">FIVE CURRENTS / RECONCILIATION</div><h1>Intraday Direction and Performance</h1><p>{manifest['history_start']} to {manifest['history_end']} | Today's open trades marked separately.</p></header><main>
<h2>What the audit establishes</h2>
<p>All ten deployed rules reproduce their archived Sharpes. Independent computations matched the signs, prior-close signal dates and regional clocks. The audit verified {manifest['independently_verified_minute_endpoints']:,} entry/exit endpoints against positive-volume minute bars of the same listed contract. Today's published weights have zero discrepancies; gold's preview and full-curve publications agree.</p>
<div class="note">The rule roster, treatment, clock and holding period were selected with hindsight using this history. The displayed ten-rule sum had not previously been tested as an allocated portfolio. The replay below now measures that exact sum, retaining the published unit weights. It does not apply the daily book's $1m dollar-VaR sizing.</div>
<h2>Today's gold direction</h2><p>The OI-direction leg applies an inverse-sign treatment to a positive raw vote: +0.039743 becomes -0.200000. The volume-spike change contributes +0.093019 for the first hour; the dry-up change is zero. This produces -10.698% net model exposure for the first hour and -20.000% for the next three hours. No publication changed those gold weights today.</p>
{table(gold_table)}
<p>At the saved mark cutoff {html.escape(manifest['mark_cutoff_utc'])}, gold's two active/closed components contribute approximately <b>{gc_mtm*100:+.3f}%</b> gross at the displayed model scale. This includes the completed long spike leg and the still-open short OI leg. The OI trade is entered on the traded 13:30 UTC minute's close (13:31 actual fill). Nominal exit is 17:30 UTC; the archived bar-close convention generally fills at 17:32 UTC.</p>
<h2>Portfolio replay</h2>"""
    def embed(name):
        return '<img src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">'
    body += embed(images[0])+table(shown)
    body += f"""<p>The portfolio has {frame.Portfolio.notna().sum()} fully priced sessions from {len(frame)} scheduled completed sessions. Each commodity uses its own available observations; an unavailable active component leaves the portfolio outcome unknown. Gaps are retained on the plots. Growth and drawdown summarize observed outcomes only. Returns and risk differ by asset because these are the actual published weights, without newly imposed equal-risk allocation.</p>
<p>Costs retain $2.16 per micro contract round trip, plus one modelled price tick per side (two ticks for silver, which is excluded here). Commissions and slippage are charged per leg before any savings from execution netting. Timing robustness shifts every entry by +15 minutes and retains the same holding period. The top-five diagnostic removes the five most profitable observed days.</p>
<h2>Each commodity</h2>"""+embed(images[1])+"<h2>Year-by-year net Sharpe</h2>"+table(annual)
    body += "<h2>Every frozen leg</h2>"+table(leg_display)
    body += """<h2>Timing and reproducibility</h2><p>Signals from close T are first used in session T+1. The deployed intraday construction retains the archived one-session lag on same-contract volume and OI. London and New York clocks use their own local daylight-saving rules. Fills use the first positive-volume minute at or after the clock within ten minutes; the fill price is that minute's close. The holding period starts at the entry fill, followed by the first eligible exit bar's close. This normally adds one minute to entry and two minutes to the nominal schedule's end.</p>
<p>The audited price history starts in June 2024. The original 593-session archive ends on 22 October 2026; stored bars extend the completed replay through 26 October. The 27 October trades remain outside historical performance statistics while positions are open. The daily-vote reconstruction differs on one previously revised input date, 22 October, and every archived traded-rule path still reconciles.</p>
<p><a href="performance_metrics.csv">Full performance metrics</a> | <a href="frozen_rules.csv">Frozen rules</a> | <a href="today_issued_weight_reconciliation.csv">Issued-position reconciliation</a> | <a href="full_trade_audit.parquet">Complete trade audit</a> | <a href="manifest.json">Audit manifest</a></p></main></body></html>"""
    (OUT/"intraday_replay_report.html").write_text(body, encoding="utf-8")
    print(OUT/"intraday_replay_report.html")


if __name__ == "__main__":
    main()
