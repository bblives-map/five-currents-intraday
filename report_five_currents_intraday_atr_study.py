"""Render the Five Currents joint intraday and causal ATR-trail study."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from five_currents_atr_trail import rolling_sharpe
from run_five_currents_intraday_atr_study import MULTIPLES, OUT, PERIODS


COLORS = {
    "Fixed exit": "#173f39",
    "ATR trail": "#c06a3b",
    "Expanding monthly ATR": "#3e7594",
    "Expanding quarterly ATR": "#ad8a35",
}
ASSET_COLORS = {"CL": "#a95b3b", "NG": "#527b58", "GC": "#b28a2f", "HG": "#376b91"}


def observed_equity(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    observed = values.dropna()
    equity = (1.0 + observed).cumprod() * 100.0
    drawdown = (equity / equity.cummax().clip(lower=100.0) - 1.0) * 100.0
    return equity, drawdown


def series(frame: pd.DataFrame, model: str, name: str) -> pd.Series:
    return frame[f"{model}|{name}"]


def save_portfolio_figure(daily: pd.DataFrame, cut: pd.Timestamp) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [3.2, 1.1]})
    for model in ("Fixed exit", "ATR trail"):
        equity, drawdown = observed_equity(series(daily, model, "Portfolio"))
        axes[0].plot(equity.index, equity, lw=1.75, color=COLORS[model], label=model)
        axes[1].plot(drawdown.index, drawdown, lw=1.2, color=COLORS[model])
    for axis in axes:
        axis.axvspan(daily.index.min(), cut, color="#d8ccb5", alpha=0.28)
        axis.axvline(cut, color="#846f50", lw=0.8, ls="--")
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[0].set_ylabel("Growth of 100")
    axes[1].set_ylabel("Drawdown %")
    axes[1].set_xlabel("Shaded: first 30% calibration; unshaded: frozen holdout")
    fig.suptitle("Joint intraday book: original exits versus the causal common trail",
                 fontsize=15, weight="bold")
    fig.tight_layout()
    name = "portfolio_equity_and_drawdown.png"
    fig.savefig(OUT / name, dpi=175, facecolor="white")
    plt.close(fig)
    return name


def save_component_figure(daily: pd.DataFrame) -> str:
    names = ("Fixed four legs", "Volume six legs", "CL", "NG", "GC", "HG")
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 10.3), sharex=True)
    for axis, name in zip(axes.flat, names):
        for model in ("Fixed exit", "ATR trail"):
            equity, _ = observed_equity(series(daily, model, name))
            axis.plot(equity.index, equity, color=COLORS[model], lw=1.35,
                      label=model if name == names[0] else None)
        axis.set_title(name, loc="left", fontsize=11, weight="bold")
        axis.set_ylabel("Growth of 100")
        axis.grid(alpha=0.14)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=9)
    fig.suptitle("Component books and commodities", fontsize=15, weight="bold")
    fig.tight_layout()
    name = "component_equity.png"
    fig.savefig(OUT / name, dpi=175, facecolor="white")
    plt.close(fig)
    return name


def save_leg_figure(daily: pd.DataFrame, rules: pd.DataFrame) -> str:
    legs = rules.leg.tolist()
    fig, axes = plt.subplots(5, 2, figsize=(13.2, 14.8), sharex=True)
    for axis, leg in zip(axes.flat, legs):
        root = leg.split(":", 1)[0]
        fixed, _ = observed_equity(series(daily, "Fixed exit", leg))
        trailed, _ = observed_equity(series(daily, "ATR trail", leg))
        axis.plot(fixed.index, fixed, color=ASSET_COLORS[root], lw=1.35, label="Original")
        axis.plot(trailed.index, trailed, color="#c06a3b", lw=1.05, ls="--", label="ATR")
        axis.set_title(leg.replace("_", " "), loc="left", fontsize=10.5, weight="bold")
        axis.grid(alpha=0.14)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel("Growth of 100")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Every deployed intraday alpha", fontsize=15, weight="bold")
    fig.tight_layout()
    name = "leg_equity.png"
    fig.savefig(OUT / name, dpi=170, facecolor="white")
    plt.close(fig)
    return name


def save_parameter_figure(grid: pd.DataFrame, baseline_oos_sharpe: float,
                          selected_period: int, selected_multiple: float) -> str:
    calibration = grid.pivot(index="atr_period", columns="atr_multiple",
                             values="stable_rank_score").reindex(index=PERIODS, columns=MULTIPLES)
    oos = grid.pivot(index="atr_period", columns="atr_multiple",
                     values="held_out_70_daily_sharpe").reindex(index=PERIODS, columns=MULTIPLES)
    oos = oos - baseline_oos_sharpe
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4))
    downside = max(-3.0, float(np.nanmin(oos.to_numpy())))
    upside = max(0.1, float(np.nanmax(oos.to_numpy())))
    images = [axes[0].imshow(calibration.to_numpy(), aspect="auto", cmap="YlGnBu", origin="lower"),
              axes[1].imshow(oos.to_numpy(), aspect="auto", cmap="RdYlGn", origin="lower",
                             norm=TwoSlopeNorm(vmin=downside, vcenter=0.0, vmax=upside))]
    titles = ("Calibration: stable multi-objective rank",
              "Held-out daily Sharpe change (display floor: -3)")
    for axis, title, image in zip(axes, titles, images):
        axis.set_title(title, fontsize=11.5, weight="bold")
        axis.set_xticks(range(len(MULTIPLES)), [f"{value:g}" for value in MULTIPLES], rotation=60)
        axis.set_yticks(range(len(PERIODS)), [str(value) for value in PERIODS])
        axis.set_xlabel("ATR multiple k")
        axis.set_ylabel("ATR period (one-minute bars)")
        fig.colorbar(image, ax=axis, shrink=0.82)
    pi = PERIODS.index(selected_period)
    mi = MULTIPLES.index(selected_multiple)
    for axis in axes:
        axis.scatter(mi, pi, marker="*", s=170, facecolor="white", edgecolor="#222", lw=0.8)
    fig.suptitle("The selected region was fixed before the holdout", fontsize=15, weight="bold")
    fig.tight_layout()
    name = "parameter_stability.png"
    fig.savefig(OUT / name, dpi=175, facecolor="white")
    plt.close(fig)
    return name


def save_walkforward_figure(walk: pd.DataFrame, start: pd.Timestamp) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1.2]})
    models = ("Fixed exit", "Expanding monthly ATR", "Expanding quarterly ATR")
    for model in models:
        values = series(walk, model, "Portfolio").loc[start:]
        equity, _ = observed_equity(values)
        axes[0].plot(equity.index, equity, color=COLORS[model], lw=1.55, label=model)
        axes[1].plot(values.index, rolling_sharpe(values, 63), color=COLORS[model], lw=1.1)
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    axes[0].set_ylabel("Growth of 100")
    axes[1].set_ylabel("63-session SR")
    axes[1].set_xlabel("Held-out period only")
    for axis in axes:
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Causal parameter refits do not recover the original exit edge",
                 fontsize=15, weight="bold")
    fig.tight_layout()
    name = "walkforward_comparison.png"
    fig.savefig(OUT / name, dpi=175, facecolor="white")
    plt.close(fig)
    return name


def table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    shown = frame.copy()
    for column, spec in (formats or {}).items():
        if column in shown:
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else format(value, spec))
    return '<div class="scroll">' + shown.to_html(index=False, border=0, escape=True) + "</div>"


def embed(name: str) -> str:
    encoded = base64.b64encode((OUT / name).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(name)}">'


def comparison_table(metrics: pd.DataFrame, sample: str, series_name: str = "Portfolio") -> pd.DataFrame:
    shown = metrics.loc[metrics["sample"].eq(sample) & metrics.series.eq(series_name)].copy()
    shown["Mean day (bp)"] = shown.daily_mean * 10000
    shown["Mean week (bp)"] = shown.weekly_mean * 10000
    shown["Return %"] = shown.daily_total_return * 100
    shown["Max DD %"] = shown.daily_max_drawdown * 100
    shown["Win %"] = shown.daily_win_rate * 100
    shown["Median up %"] = shown.daily_median_gain * 100
    shown["Median down %"] = shown.daily_median_loss * 100
    return shown.rename(columns={"model": "Exit", "daily_observations": "Days",
                                 "daily_sharpe": "Daily SR", "weekly_sharpe": "Weekly SR"})[
        ["Exit", "Days", "Daily SR", "Weekly SR", "Mean day (bp)", "Mean week (bp)",
         "Return %", "Max DD %", "Win %", "Median up %", "Median down %"]
    ]


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    daily = pd.read_parquet(OUT / "daily_returns.parquet")
    walk = pd.read_parquet(OUT / "walkforward_daily_returns.parquet")
    metrics = pd.read_csv(OUT / "performance_metrics.csv")
    robustness = pd.read_csv(OUT / "robustness_metrics.csv")
    walk_metrics = pd.read_csv(OUT / "walkforward_metrics.csv")
    rules = pd.read_csv(OUT / "deployed_rules.csv")
    grid = pd.read_csv(OUT / "parameter_grid.csv")
    splits = pd.read_csv(OUT / "split_stability.csv")
    behavior = pd.read_csv(OUT / "trail_behavior.csv")
    years = pd.read_csv(OUT / "yearly_metrics.csv")
    refits = pd.read_csv(OUT / "walkforward_refits.csv")
    daily.index = pd.to_datetime(daily.index)
    walk.index = pd.to_datetime(walk.index)
    cut = pd.Timestamp(selection["held_out_start"])

    images = [
        save_portfolio_figure(daily, cut),
        save_component_figure(daily),
        save_leg_figure(daily, rules),
        save_parameter_figure(
            grid,
            metrics.loc[metrics.model.eq("Fixed exit") & metrics.series.eq("Portfolio")
                        & metrics["sample"].eq("Held-out 70%"), "daily_sharpe"].iloc[0],
            selection["atr_period_minutes"], selection["atr_multiple"],
        ),
        save_walkforward_figure(walk, cut),
    ]

    full = comparison_table(metrics, "Full")
    oos = comparison_table(metrics, "Held-out 70%")
    component = metrics.loc[metrics["sample"].eq("Held-out 70%")
                            & metrics.series.isin(["Fixed four legs", "Volume six legs",
                                                  "CL", "NG", "GC", "HG"])].copy()
    component["Mean day bp"] = component.daily_mean * 10000
    component["Mean week bp"] = component.weekly_mean * 10000
    component["Return %"] = component.daily_total_return * 100
    component["Max DD %"] = component.daily_max_drawdown * 100
    component = component.rename(columns={"model": "Exit", "series": "Book",
                                          "daily_sharpe": "Daily SR",
                                          "weekly_sharpe": "Weekly SR"})[
        ["Exit", "Book", "Daily SR", "Weekly SR", "Mean day bp", "Mean week bp",
         "Return %", "Max DD %"]
    ]
    leg_metrics = metrics.loc[metrics["sample"].eq("Held-out 70%")
                              & metrics.series.str.contains(":", regex=False)].copy()
    leg_metrics = leg_metrics.merge(rules[["leg", "clock", "hold_minutes"]],
                                    left_on="series", right_on="leg", validate="many_to_one")
    leg_metrics["Mean day bp"] = leg_metrics.daily_mean * 10000
    leg_metrics["Return %"] = leg_metrics.daily_total_return * 100
    leg_metrics = leg_metrics.rename(columns={"model": "Exit", "series": "Alpha",
                                              "clock": "Clock", "hold_minutes": "Hold min",
                                              "daily_sharpe": "Daily SR",
                                              "weekly_sharpe": "Weekly SR"})[
        ["Exit", "Alpha", "Clock", "Hold min", "Daily SR", "Weekly SR",
         "Mean day bp", "Return %"]
    ]
    behavior_shown = behavior.loc[behavior.series.isin(
        ["Portfolio", "Fixed four legs", "Volume six legs", "CL", "NG", "GC", "HG"]
    )].copy()
    behavior_shown["Stop %"] = behavior_shown.stop_rate * 100
    behavior_shown["Initial distance %"] = behavior_shown.median_initial_stop_pct * 100
    behavior_shown = behavior_shown.rename(columns={"series": "Book",
        "active_priced_trades": "Active trades", "stopped_trades": "Stopped",
        "median_baseline_hold_minutes": "Original median min",
        "median_atr_hold_minutes": "ATR median min"})[
        ["Book", "Active trades", "Stopped", "Stop %", "Original median min",
         "ATR median min", "Initial distance %"]
    ]
    split_shown = splits.copy()
    split_shown["Test baseline SR"] = split_shown.test_baseline_daily_sharpe
    split_shown["Test ATR SR"] = split_shown.test_atr_daily_sharpe
    split_shown["Test baseline weekly SR"] = split_shown.test_baseline_weekly_sharpe
    split_shown["Test ATR weekly SR"] = split_shown.test_atr_weekly_sharpe
    split_shown["Baseline mean week bp"] = split_shown.test_baseline_weekly_mean * 10000
    split_shown["ATR mean week bp"] = split_shown.test_atr_weekly_mean * 10000
    split_shown = split_shown.rename(columns={"calibration_fraction": "Train",
        "test_fraction": "Holdout", "atr_period": "ATR min", "atr_multiple": "k"})[
        ["Train", "Holdout", "ATR min", "k", "Test baseline SR", "Test ATR SR",
         "Test baseline weekly SR", "Test ATR weekly SR", "Baseline mean week bp",
         "ATR mean week bp"]
    ]
    robust_portfolio = robustness.loc[robustness.series.eq("Portfolio")
                                      & robustness["sample"].eq("Held-out 70%")].copy()
    robust_portfolio["Mean day bp"] = robust_portfolio.daily_mean * 10000
    robust_portfolio["Mean week bp"] = robust_portfolio.weekly_mean * 10000
    robust_portfolio["Return %"] = robust_portfolio.daily_total_return * 100
    robust_portfolio["Max DD %"] = robust_portfolio.daily_max_drawdown * 100
    robust_portfolio = robust_portfolio.rename(columns={"model": "Test",
        "daily_sharpe": "Daily SR", "weekly_sharpe": "Weekly SR"})[
        ["Test", "Daily SR", "Weekly SR", "Mean day bp", "Mean week bp", "Return %", "Max DD %"]
    ]
    walk_portfolio = walk_metrics.loc[walk_metrics.series.eq("Portfolio")].copy()
    walk_portfolio["Mean day bp"] = walk_portfolio.daily_mean * 10000
    walk_portfolio["Mean week bp"] = walk_portfolio.weekly_mean * 10000
    walk_portfolio["Return %"] = walk_portfolio.daily_total_return * 100
    walk_portfolio["Max DD %"] = walk_portfolio.daily_max_drawdown * 100
    walk_portfolio = walk_portfolio.rename(columns={"model": "Exit",
        "daily_sharpe": "Daily SR", "weekly_sharpe": "Weekly SR"})[
        ["Exit", "Daily SR", "Weekly SR", "Mean day bp", "Mean week bp", "Return %", "Max DD %"]
    ]
    annual = years.loc[years.series.eq("Portfolio")].copy()
    annual["Return %"] = annual.daily_total_return * 100
    annual["Max DD %"] = annual.daily_max_drawdown * 100
    annual = annual.rename(columns={"model": "Exit", "year": "Year",
        "daily_sharpe": "Daily SR", "weekly_sharpe": "Weekly SR"})[
        ["Exit", "Year", "Daily SR", "Weekly SR", "Return %", "Max DD %"]
    ]

    fixed_oos = oos.loc[oos.Exit.eq("Fixed exit")].iloc[0]
    atr_oos = oos.loc[oos.Exit.eq("ATR trail")].iloc[0]
    drawdown_improvement = 1.0 - abs(atr_oos["Max DD %"]) / abs(fixed_oos["Max DD %"])
    latest_refits = refits.groupby("cadence_sessions").tail(1)
    latest_text = "; ".join(
        f"{int(row.cadence_sessions)}-session refit: ATR({int(row.atr_period)}) x {row.atr_multiple:g}"
        for row in latest_refits.itertuples()
    )

    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Five Currents | Joint Intraday Alpha and ATR Trail Study</title>
<style>
:root{{--ink:#173b37;--paper:#f4f0e7;--white:#fffdf8;--line:#d7d8cd;--rust:#b55f38;--gold:#b29148;--blue:#3e7594}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.58 Georgia,serif}}
header{{padding:42px max(5vw,28px);background:linear-gradient(115deg,#143c37,#27584d);color:#fffdf7}}
.eyebrow{{font:11px/1.4 Verdana,sans-serif;letter-spacing:.18em;text-transform:uppercase;color:#d2dfd6}}
h1{{font-size:39px;line-height:1.08;margin:8px 0 10px}}header p{{margin:0;max-width:920px;color:#dbe6df}}
main{{max-width:1260px;margin:auto;padding:20px 30px 70px}}h2{{font-size:27px;margin:38px 0 13px;border-bottom:1px solid var(--line);padding-bottom:7px}}
h3{{font-size:20px;margin:27px 0 8px}}p{{max-width:1100px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:22px 0}}
.card{{background:var(--white);border:1px solid var(--line);padding:16px 17px;border-radius:9px}}.card b{{display:block;font:24px/1.2 Georgia,serif;color:#143c37}}
.card span{{font:11px/1.4 Verdana,sans-serif;color:#68736c}}.verdict{{background:#fff8eb;border-left:5px solid var(--gold);padding:17px 21px;margin:24px 0}}
.method{{background:#e9efeb;border-left:5px solid #477b6d;padding:17px 21px;margin:24px 0}}img{{display:block;width:100%;background:white;margin:20px 0 28px;border:1px solid #e0dfd6}}
.scroll{{overflow:auto;margin:14px 0 25px}}table{{border-collapse:collapse;width:100%;background:var(--white);font:11px/1.45 Verdana,sans-serif}}
th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{background:#e7ece7;color:#234a43}}th:first-child,td:first-child{{text-align:left}}
.formula{{font:14px/1.6 'Cambria Math',Georgia,serif;background:var(--white);border:1px solid var(--line);padding:13px 17px;margin:9px 0}}
.small{{font:12px/1.55 Verdana,sans-serif;color:#5f6c65}}footer{{margin-top:42px;padding-top:16px;border-top:1px solid var(--line);font:11px Verdana,sans-serif;color:#657069}}
@media(max-width:850px){{.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:31px}}main{{padding:15px}}}}@media print{{header{{padding:25px}}img,.card,table{{break-inside:avoid}}}}
</style></head><body><header><div class="eyebrow">Five Currents / Execution Research</div>
<h1>Joint Intraday Alpha and ATR Trail Study</h1><p>{manifest['history_start']} through {manifest['history_end']} | ten deployed rules, four commodities, exact-contract one-minute paths, after modelled costs</p></header><main>
<h2>Executive finding</h2>
<div class="cards"><div class="card"><b>{fixed_oos['Daily SR']:.2f}</b><span>Original held-out daily Sharpe</span></div>
<div class="card"><b>{fixed_oos['Weekly SR']:.2f}</b><span>Original held-out weekly Sharpe</span></div>
<div class="card"><b>{fixed_oos['Mean week (bp)']:.1f} bp</b><span>Original mean held-out week</span></div>
<div class="card"><b>{fixed_oos['Max DD %']:.2f}%</b><span>Original held-out max drawdown</span></div></div>
<p>The joint intraday book remains strong after extending the dashboard replay through 10 November 2026. Its original family-specific holding periods produce a full-history net daily Sharpe of {full.loc[full.Exit.eq('Fixed exit'),'Daily SR'].iloc[0]:.2f} and a held-out Sharpe of {fixed_oos['Daily SR']:.2f}. The result is broad enough to remain positive in every commodity and in both the fixed and volume books.</p>
<div class="verdict"><b>No common ATR trail qualifies as an alpha enhancement.</b> The causally selected ATR({selection['atr_period_minutes']}) x {selection['atr_multiple']:g} overlay cuts held-out maximum drawdown by {drawdown_improvement:.0%}, but lowers daily Sharpe to {atr_oos['Daily SR']:.2f}, weekly Sharpe to {atr_oos['Weekly SR']:.2f}, mean daily gain from {fixed_oos['Mean day (bp)']:.2f} to {atr_oos['Mean day (bp)']:.2f} bp and mean weekly gain from {fixed_oos['Mean week (bp)']:.1f} to {atr_oos['Mean week (bp)']:.1f} bp. It is a defensible optional risk overlay, not a replacement for the current exits.</div>
{embed(images[0])}
<h3>Full replay</h3>{table(full, {'Daily SR':'.3f','Weekly SR':'.3f','Mean day (bp)':'.2f','Mean week (bp)':'.2f','Return %':'.2f','Max DD %':'.2f','Win %':'.1f','Median up %':'.3f','Median down %':'.3f'})}
<h3>Frozen 70% holdout</h3>{table(oos, {'Daily SR':'.3f','Weekly SR':'.3f','Mean day (bp)':'.2f','Mean week (bp)':'.2f','Return %':'.2f','Max DD %':'.2f','Win %':'.1f','Median up %':'.3f','Median down %':'.3f'})}

<h2>What is being tested</h2>
<p>The source book is the exact ten-leg intraday schedule published by the dashboard: prior-close daily signals, the deployed signal treatments, regional daylight-saving clocks, family-specific holding periods, prior-session liquidity routing and the same listed contract from entry to exit. The replay contains 606 scheduled sessions and 477 completely priced portfolio sessions. Missing active outcomes remain unknown; they are never replaced with zero.</p>
<div class="method"><b>Causal design.</b> Ninety tight candidates first showed that conventional small multipliers stopped almost every trade. The multiplier axis was therefore extended into a non-binding region without adding another strategy variable. One common pair across every asset and alpha was selected on the first 30% of sessions using daily Sharpe, weekly Sharpe, daily gain, weekly gain, median component Sharpe and drawdown. A stable neighboring region, not the best isolated cell, determined the choice. The final 70% was not used in selection.</div>
<div class="formula"><b>True range:</b> TR(t) = max[ H(t)-L(t), |H(t)-C(t-1)|, |L(t)-C(t-1)| ]</div>
<div class="formula"><b>Wilder ATR:</b> ATR(p,t) = ((p-1) ATR(p,t-1) + TR(t)) / p</div>
<div class="formula"><b>Long trail:</b> stop(t+1) = max[ stop(t), highest_high_since_entry(t) - k ATR(p,t) ]<br><b>Short trail:</b> stop(t+1) = min[ stop(t), lowest_low_since_entry(t) + k ATR(p,t) ]</div>
<p class="small">The entry-bar ATR is known at the close used for entry. Each later bar is tested against the stop known before that bar; only a surviving bar may update the next stop. Gap-through fills use the observed open. This prevents intrabar look-ahead.</p>

<h2>Where the intraday edge lives</h2>{embed(images[1])}
{table(component, {'Daily SR':'.3f','Weekly SR':'.3f','Mean day bp':'.2f','Mean week bp':'.2f','Return %':'.2f','Max DD %':'.2f'})}
<p>The trail is least intrusive in gold, where it stops only 3.7% of active trades, and most intrusive in the volume book, CL and NG. That asymmetry explains the trade-off: it removes some tails but also truncates the long-duration volume-family payoffs that contribute materially to the joint book.</p>
{table(behavior_shown, {'Stop %':'.1f','Original median min':'.1f','ATR median min':'.1f','Initial distance %':'.2f'})}

<h2>Every deployed alpha</h2>{embed(images[2])}
{table(leg_metrics, {'Daily SR':'.3f','Weekly SR':'.3f','Mean day bp':'.2f','Return %':'.2f'})}

<h2>Parameter and split stability</h2>{embed(images[3])}
<p>The star marks the causally selected ATR(30) x 40 region. Several k=20 cells look stronger after seeing the holdout, but they met only one to three of six early calibration criteria. Treating them as the answer would be hindsight selection. Independent 40/60 and 50/50 fits also favor a wide k=40 region, but neither beats the original exits consistently in its own holdout.</p>
{table(split_shown, {'Train':'.0%','Holdout':'.0%','k':'.1f','Test baseline SR':'.3f','Test ATR SR':'.3f','Test baseline weekly SR':'.3f','Test ATR weekly SR':'.3f','Baseline mean week bp':'.2f','ATR mean week bp':'.2f'})}

<h2>Causal refits and execution stress</h2>{embed(images[4])}
<p>Expanding monthly and quarterly refits were tested because the preferred multiplier shifts over time. They still underperform the original exits: adaptation does not recover the lost daily or weekly gain. The latest choices are {html.escape(latest_text)}.</p>
{table(walk_portfolio, {'Daily SR':'.3f','Weekly SR':'.3f','Mean day bp':'.2f','Mean week bp':'.2f','Return %':'.2f','Max DD %':'.2f'})}
<p>A 15-minute delayed entry makes the original schedule weaker; the trail then improves the delayed variant, but both remain below the original on-time fixed exit. Under doubled friction, the original exit also remains superior. This supports using the stop as delay protection only if operational execution cannot reliably meet the intended clocks.</p>
{table(robust_portfolio, {'Daily SR':'.3f','Weekly SR':'.3f','Mean day bp':'.2f','Mean week bp':'.2f','Return %':'.2f','Max DD %':'.2f'})}

<h2>Year-by-year portfolio</h2>{table(annual, {'Daily SR':'.3f','Weekly SR':'.3f','Return %':'.2f','Max DD %':'.2f'})}

<h2>Recommendation</h2>
<p><b>Retain the deployed family-specific fixed holding periods as the production baseline.</b> They have the highest causal daily Sharpe and gain, the highest held-out weekly gain, and positive contributions across all four commodities. Do not replace them with a universal ATR trail.</p>
<p>If a discretionary risk cap is required, ATR(30) x 40 is the least objectionable common overlay: it fires on 16.2% of active trades and improves the observed drawdown. It should be paper-tracked as a separate risk sleeve, with explicit acceptance that expected return and Sharpe are lower. The attractive k=20 holdout cells should remain research diagnostics unless a later, genuinely unseen sample validates them.</p>
<footer>Read-only study. No live API calls, database writes, dashboard publications or collector changes. Costs retain the deployed $2.16 micro-contract round trip plus the existing slippage model. Connected chart segments join observed outcomes; missing active sessions are excluded rather than imputed. Reproducible artifacts: performance_metrics.csv, parameter_grid.csv, selected_trade_audit.parquet, split_stability.csv, walkforward_refits.csv and manifest.json.</footer>
</main></body></html>"""
    destination = OUT / "five_currents_joint_intraday_atr_trail_report.html"
    destination.write_text(body, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
