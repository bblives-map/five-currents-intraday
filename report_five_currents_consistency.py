"""Self-contained research report for the fixed consistency-objective comparison."""
from __future__ import annotations

import base64

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from five_currents_consistency_objectives import LABELS, OBJECTIVES
from five_currents_daily.report import CONCEPT_LABELS
from us_commodity_research_core import PRODUCT_ORDER


COLORS = {"sharpe": "#23485b", "upside_capped": "#167f70", "rolling_sharpe": "#c18438",
          "weekly": "#806756", "monthly": "#a24e60"}


def table(frame):
    frame = frame.copy()
    percentages = {"annualized_return", "annualized_volatility", "max_drawdown", "cash_fraction", "total_return",
                   "weekly_win_rate", "monthly_win_rate", "positive_quarterly_fraction", "top_1pct_positive_pnl_share"}
    for name in frame:
        if name in percentages:
            frame[name] = frame[name].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
        elif pd.api.types.is_float_dtype(frame[name]):
            frame[name] = frame[name].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "-")
    return frame.to_html(index=False, border=0, escape=True)


def render_report(output, metrics, annual, paths, products, audit, fits, mature_start):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fbfcf8",
                         "axes.facecolor": "#fbfcf8", "text.color": "#233e39"})
    base = paths.loc[paths.scenario.eq("base")]
    dates = pd.DatetimeIndex(base.realization_session.unique()).sort_values()
    maturity = paths.loc[paths.session.ge(mature_start), "realization_session"].min()
    for mode in ("admission", "pruned", "portfolio_only"):
        fig, axes = plt.subplots(2, 1, figsize=(12.4, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2.5, 1]})
        for objective in OBJECTIVES:
            model = "sharpe_admission" if mode == "portfolio_only" and objective == "sharpe" else f"{objective}_{mode}"
            values = base.loc[base.model.eq(model)].set_index("realization_session").net_return
            equity = 100 * (1 + values).cumprod()
            row = metrics.loc[metrics.model.eq(model) & metrics.scenario.eq("base") & metrics.period.eq("full_common")].iloc[0]
            axes[0].plot(equity.index, equity, color=COLORS[objective], lw=1.5,
                         label=f"{LABELS[objective]} | SR {row.sharpe:.2f}")
            axes[1].plot(equity.index, 100 * (equity / equity.cummax().clip(lower=100) - 1),
                         color=COLORS[objective], lw=1)
        for ax in axes:
            ax.axvspan(dates.min(), maturity, color="#d7e1d3", alpha=.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
        title = {"admission": "Recalibrated admission | equal concept votes",
                 "pruned": "Recalibrated admission + optional greedy pruning",
                 "portfolio_only": "Original admission retained | change portfolio fitness only"}[mode]
        axes[0].set_title(title, loc="left", fontsize=16, weight="bold", pad=17)
        axes[0].set_ylabel("Growth of 100, after costs")
        axes[0].legend(frameon=False, fontsize=9, ncol=1, loc="upper left")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].xaxis.set_major_locator(mdates.YearLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[1].set_xlim(dates.min(), dates.max())
        fig.text(.08, .025, "Actual uncapped returns. Same 504/21 schedule, contract migration, costs and risk caps.\n"
                 "Shaded: expanding training. Horizontal axis is the realized close, two observed sessions after the forecast.",
                 fontsize=9, color="#53645e", linespacing=1.5)
        fig.tight_layout(rect=(.02, .08, .99, .98))
        fig.savefig(output / f"consistency_{mode}_equity.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    for objective in OBJECTIVES:
        values = base.loc[base.model.eq(f"{objective}_admission")].set_index("realization_session").no_windfall_return
        ax.plot(values.index, 100 * (1 + values).cumprod(), color=COLORS[objective], lw=1.5, label=LABELS[objective])
    ax.set_title("Dependence on the baseline's biggest gain days", loc="left", fontsize=16, weight="bold", pad=17)
    ax.set_ylabel("Growth of 100, windfall ablation")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(.08, .025, "Shared dates: baseline's best 1% of common marked sessions. Positive P&L is removed on those dates; losses remain.\n"
             "Hindsight sensitivity diagnostic, not an executable strategy or a selection input. Other dates are untouched.",
             fontsize=9, color="#53645e", linespacing=1.5)
    fig.tight_layout(rect=(.02, .12, .99, .98))
    fig.savefig(output / "consistency_windfall_ablation.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12.4, 10.4), sharex=True)
    common = base.loc[base.model.eq("sharpe_admission")].set_index("session").common_marked
    realized_lookup = base.loc[base.model.eq("sharpe_admission")].set_index("session").realization_session
    for ax, product in zip(axes.flat, PRODUCT_ORDER, strict=False):
        for objective in OBJECTIVES:
            model = f"{objective}_admission"
            values = products.loc[products.model.eq(model) & products.scenario.eq("base")
                                  & products["product"].eq(product)].set_index("session").net_contribution.where(common)
            ax.plot(pd.DatetimeIndex(realized_lookup.reindex(values.index)), 100 * values.cumsum(),
                    color=COLORS[objective], lw=1.15, label=LABELS[objective])
        ax.set_title(product, loc="left", weight="bold")
        ax.set_ylabel("Cumulative contribution (pp)")
        ax.axhline(0, color="#8a918a", lw=.6)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False, fontsize=9)
    fig.suptitle("Commodity contributions | admission variants", x=.07, ha="left", weight="bold", fontsize=17)
    fig.tight_layout(rect=(.02, .02, .99, .96))
    fig.savefig(output / "consistency_by_commodity.png", dpi=150)
    plt.close(fig)

    def picture(name):
        data = base64.b64encode((output / name).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" alt="{name}">'

    def readable(frame):
        frame = frame.copy()
        if "objective" in frame:
            frame["objective"] = frame.objective.map(LABELS)
        return frame

    actual = metrics.loc[metrics.scenario.eq("base") & metrics.period.eq("full_common")].copy()
    primary = actual.loc[actual["mode"].eq("admission")].set_index("objective").reindex(OBJECTIVES).reset_index()
    secondary = actual.loc[actual["mode"].eq("pruned")].set_index("objective").reindex(OBJECTIVES).reset_index()
    portfolio_only = actual.loc[actual["mode"].eq("portfolio_only") | actual.model.eq("sharpe_admission")]
    portfolio_only = portfolio_only.set_index("objective").reindex(OBJECTIVES).reset_index()
    main_cols = ["objective", "sharpe", "annualized_return", "annualized_volatility", "max_drawdown",
                 "weekly_win_rate", "monthly_win_rate", "median_quarterly_sharpe", "cash_fraction", "mean_admitted"]
    extra_cols = ["objective", "mode", "positive_quarterly_fraction", "top_1pct_positive_pnl_share",
                  "weekly_observations", "monthly_observations", "mean_daily_turnover", "mean_gross_exposure"]
    stress = metrics.loc[metrics.scenario.eq("stressed") & metrics.period.eq("full_common"),
                         ["objective", "mode", "sharpe", "annualized_return", "max_drawdown"]]
    ablation = metrics.loc[metrics.scenario.eq("base") & metrics.period.eq("shared_windfall_ablation"),
                           ["objective", "mode", "sharpe", "annualized_return", "max_drawdown"]]
    native = metrics.loc[metrics.scenario.eq("base") & metrics.period.eq("full_native"),
                         ["objective", "mode", "sharpe", "annualized_return", "observations"]]
    mature = metrics.loc[metrics.scenario.eq("base") & metrics.period.eq("trailing504_common"), main_cols + ["mode"]]
    annual_tables = []
    for mode in ("admission", "pruned", "portfolio_only"):
        subset = annual.loc[annual.model.str.endswith("_" + mode) | ((mode == "portfolio_only") & annual.model.eq("sharpe_admission"))].copy()
        keys = subset.model.str.removesuffix("_" + mode).replace({"sharpe_admission": "sharpe"})
        subset["objective"] = keys.map(LABELS)
        sr = subset.pivot(index="year", columns="objective", values="sharpe").reindex(columns=[LABELS[k] for k in OBJECTIVES]).reset_index()
        ret = subset.pivot(index="year", columns="objective", values="year_return").reindex(columns=[LABELS[k] for k in OBJECTIVES])
        ret = ret.map(lambda x: f"{x:.1%}" if pd.notna(x) else "-").reset_index()
        annual_tables.append(f"<h3>{mode.title()}: net Sharpe</h3>{table(sr)}<h3>Compounded returns</h3>{table(ret)}")
    frequency = audit.loc[audit["mode"].eq("admission")].groupby(["economic_family", "objective"]).selected.mean().unstack("objective")
    frequency = frequency.reindex(columns=OBJECTIVES).map(lambda x: f"{x:.0%}").rename(columns=LABELS).reset_index()
    frequency["economic_family"] = frequency.economic_family.map(CONCEPT_LABELS)
    latest = audit.loc[audit.refit_ordinal.eq(audit.refit_ordinal.max()) & audit["mode"].eq("admission")]
    latest = latest.pivot(index="economic_family", columns="objective", values="selected").reindex(columns=OBJECTIVES)
    latest = latest.map(lambda x: "Admitted" if x else "Excluded").rename(columns=LABELS).reset_index()
    diagnostic = primary.set_index("objective")
    original = diagnostic.loc["sharpe"]
    weekly_book = portfolio_only.set_index("objective").loc["weekly"]
    windfall_metrics = metrics.loc[metrics.scenario.eq("base") & metrics.period.eq("shared_windfall_ablation")].set_index("model")
    windfall_original = windfall_metrics.loc["sharpe_admission"]
    windfall_weekly = windfall_metrics.loc["weekly_portfolio_only"]
    weekly_observations = int(original.weekly_observations)
    original_week_wins = int(round(original.weekly_win_rate * weekly_observations))
    challenger_week_wins = int(round(weekly_book.weekly_win_rate * weekly_observations))
    removal_frequency = audit.loc[audit["mode"].eq("portfolio_only")].copy()
    removal_frequency["removed"] = removal_frequency.original_admitted & ~removal_frequency.selected
    removal_frequency = removal_frequency.groupby(["economic_family", "objective"]).removed.sum().unstack("objective")
    removal_frequency = removal_frequency.reindex(columns=OBJECTIVES[1:]).rename(columns=LABELS).reset_index()
    removal_frequency["economic_family"] = removal_frequency.economic_family.map(CONCEPT_LABELS)
    comparisons = []
    for objective in OBJECTIVES[1:]:
        row = diagnostic.loc[objective]
        comparisons.append(f"<li><b>{LABELS[objective]}:</b> net Sharpe {row.sharpe:.2f}, "
                           f"CAGR {row.annualized_return:.1%}, profitable weeks {row.weekly_win_rate:.1%}, "
                           f"profitable months {row.monthly_win_rate:.1%}; "
                           f"cash on {row.cash_fraction:.1%} of marked sessions.</li>")
    period_note = f"{dates.min():%d %b %Y} to {dates.max():%d %b %Y}"
    text = f"""<!doctype html><html><head><meta charset="utf-8"><title>Five Currents | Consistency Objectives</title>
<style>body{{margin:0;background:#f4f5ef;color:#253f39;font-family:Georgia,serif}}main{{max-width:1220px;margin:auto;padding:36px}}
h1{{font-size:34px;margin-bottom:8px}}h2{{margin-top:6px}}p,li{{line-height:1.6}}.lede{{font-size:19px;color:#53665e}}
section{{background:white;padding:25px;margin:22px 0;border:1px solid #d9dfd5;border-radius:9px}}img{{width:100%;height:auto}}
table{{width:100%;border-collapse:collapse;font:12px Calibri,sans-serif}}td,th{{padding:8px;text-align:right;border-bottom:1px solid #e0e5dc}}
td:first-child,th:first-child{{text-align:left}}th{{background:#edf2e9}}.scroll{{overflow-x:auto}}.note{{font-size:14px;color:#53665e}}</style></head><body><main>
<h1>Five Currents: Consistency Before Windfalls</h1><p class="lede">A fixed, causal comparison of selection objectives</p>
<p>CL, NG, GC, SI and HG. Actual realized-close history: {period_note}. Ten existing economic concepts;
504-session trailing learning with 21-session refits, preceded by the same expanding calibration.</p>
<section><h2>Findings</h2><p>The original admission has net Sharpe {original.sharpe:.2f}, CAGR {original.annualized_return:.1%},
profitable weeks {original.weekly_win_rate:.1%} and profitable months {original.monthly_win_rate:.1%} on the common marked dates.</p>
<ul>{''.join(comparisons)}</ul><p>These comparisons recalibrate admission and, separately, optional subset pruning.
They do not tune new alpha feature parameters or select a winning objective for production using the whole replay.</p>
<p><b>The more promising result is at portfolio level:</b> retain the original admission pool, then prune only when weekly consistency improves.
Net Sharpe rises from {original.sharpe:.2f} to {weekly_book.sharpe:.2f}, and CAGR from {original.annualized_return:.1%} to {weekly_book.annualized_return:.1%}.
Profitable weeks rise from {original_week_wins} to {challenger_week_wins} out of {weekly_observations}; monthly wins remain unchanged.
Maximum drawdown worsens from {abs(original.max_drawdown):.2%} to {abs(weekly_book.max_drawdown):.2%}, while median quarterly Sharpe
falls from {original.median_quarterly_sharpe:.2f} to {weekly_book.median_quarterly_sharpe:.2f}. This is a modest, mixed improvement, not universal dominance.</p>
<p>After removing gains on the same baseline-selected best 1% of sessions, Sharpe is {windfall_original.sharpe:.2f} for the baseline
and {windfall_weekly.sharpe:.2f} for the weekly portfolio criterion; CAGR is {windfall_original.annualized_return:.1%} and {windfall_weekly.annualized_return:.1%}.
The weekly variant is a reasonable shadow challenger, not sufficient evidence for an automatic production switch.</p></section>
<section><h2>Primary test: admission objective</h2>{picture('consistency_admission_equity.png')}
<div class="scroll">{table(readable(primary[main_cols]))}</div>
<p class="note">Actual uncapped P&amp;L, same complete five-product marked dates for every version, with genuine cash retained.
Reported weekly/monthly win rates count flat complete periods as non-wins; active-only rates are also saved in the CSV.
Incomplete boundaries and any calendar period containing an unpriced session are excluded, not imputed.
Win rates are not probabilities of future profitability.</p></section>
<section><h2>Secondary test: portfolio pruning</h2>{picture('consistency_pruned_equity.png')}
<div class="scroll">{table(readable(secondary[main_cols]))}</div><p>Each objective starts with its own newly admitted roster.
Greedy removal must improve that objective in the full training window and both chronological halves under both base and doubled costs.
    Survivors receive equal concept votes. There is no optimized K, return-weight regression or extra leverage.
Strict improvement makes pruning naturally conservative for coarse weekly/monthly hit-rate scores.</p></section>
<section><h2>Control: keep the original admission pool</h2>{picture('consistency_portfolio_only_equity.png')}
<div class="scroll">{table(readable(portfolio_only[main_cols]))}</div><p>These four controls keep the original Sharpe-admitted pool at every refit,
then change only the combined-book pruning objective. This distinguishes a portfolio-level consistency criterion from rejecting useful diversifiers on standalone consistency.
The original equal-vote portfolio is the reference curve; the original Sharpe-pruning control is reported in the preceding section.</p></section>
<section><h2>Which concepts are removed conditionally?</h2><div class="scroll">{table(removal_frequency)}</div>
<p>Counts are refits at which a concept admitted by the original rule is removed by portfolio-only pruning.
This is conditional subset selection, not a permanent blacklist or evidence that a family is universally unprofitable.
The remaining concepts continue to be reassessed at every scheduled refit.</p></section>
<section><h2>Dependence on exceptional days</h2>{picture('consistency_windfall_ablation.png')}
<div class="scroll">{table(readable(ablation))}</div><p>The same dates are used for all versions: the baseline's best 1% of marked sessions.
Only positive P&amp;L is removed on those dates; all losses are retained. This intentionally hindsight-selected ablation is a sensitivity diagnostic,
never a trading rule, training input or revised actual return series. No war dates or other event windows were hand-selected.</p>
<h3>Additional consistency and exposure checks</h3><div class="scroll">{table(readable(actual[extra_cols]))}</div>
<p class="note">The best-1% share is each model's own largest positive days divided by its total positive daily P&amp;L, not its net cumulative return.
Overlapping 63-session Sharpe observations are not independent statistical trials.</p></section>
<section><h2>Commodity contributions</h2>{picture('consistency_by_commodity.png')}
<p class="note">Additive contributions to the same portfolio, not separately compounded asset investments.</p></section>
<section><h2>What was actually recalibrated?</h2><ul>
<li><b>Original:</b> the archived activity, positive half-window/stressed Sharpe and composite-score admission gates, reproduced exactly.</li>
<li><b>Upside-capped Sharpe:</b> cap positive daily training returns at max(0, the training slice's 95th percentile), retain every loss unchanged,
then calculate Sharpe. Caps are relearned inside each refit and each half, not from future returns. Displayed realized returns are never capped.</li>
<li><b>Rolling Sharpe:</b> median Sharpe across all fully elapsed 63-session windows inside the training slice,
with at least 50 marked observations per window. Each half is evaluated separately; no earlier or future return enters its window.</li>
<li><b>Weekly/monthly:</b> profitable fraction of complete, nonflat calendar periods, minus 50%.
Admission requires positive scores in both halves, positive stressed score and composite fitness, plus positive full-window stressed expectancy.
At least two nonflat periods per evaluated half are required. Monthly evidence is coarse and particularly thin during early expanding fits.</li>
<li><b>Shared:</b> the old performance score is replaced by the chosen metric, using the unchanged 35/25/25/15 full/stressed/worst-half/mean-half structure.
Activity and observation safeguards remain. The metric is evaluated on each existing concept's counterfactual training path; realized portfolio selection is held until the next refit.</li>
<li><b>Unchanged:</b> alpha definitions and neighbourhood averages, differential roll/liquidity rules, product inverse-volatility normalization,
gross cap 1.0, product cap 0.4, base commissions/slippage and doubled-friction control.
Each changed roster relearns the same product-risk overlay from its available training history.</li>
</ul><h3>Admission frequency across refits</h3><div class="scroll">{table(frequency)}</div>
<h3>Latest fit: {fits.forecast_start.max():%d %b %Y}</h3><div class="scroll">{table(latest)}</div></section>
<section><h2>Year-by-year</h2>{''.join(annual_tables)}</section>
<section><h2>Controls</h2><h3>Doubled execution costs</h3><div class="scroll">{table(readable(stress))}</div>
<h3>After expanding training reaches 504 sessions</h3><div class="scroll">{table(readable(mature))}</div>
<p class="note">Secondary time slice only, not an untouched holdout. Full-history curves always include the earlier realized calibration returns.</p>
<h3>Each book's native marked-date statistics</h3><div class="scroll">{table(readable(native))}</div>
<p class="note">The original admission and original Sharpe-pruning weights and P&amp;L are checked against the preceding study.
Native baseline statistics may differ from the primary comparison because the latter deliberately imposes a shared complete five-product mark mask.</p></section>
<section><h2>Interpretation</h2><p>A smoother objective is useful only if forward realized consistency improves without simply suppressing exposure or hiding losses.
High win rate is insufficient when losing periods are larger; positive expectancy is therefore required and every realized loss remains in the replay.
Likewise, large trend gains are not automatically spurious: research finds that large market moves can be a genuine driver of trend-following returns.
<a href="https://www.aqr.com/Insights/Research/Journal-Article/You-Cant-Always-Trend-When-You-Want">AQR research</a>.</p>
<p>The roster and its feature neighbourhoods were previously studied on this history. New objective comparisons add researcher choice;
they do not erase that hindsight selection. No newly unseen data, event-specific filters or extensive objective hyperparameter grid were introduced.</p>
<p class="note">Local cached files only. No market API calls, database writes, live dashboard publications or orders.
Saved raw concept training returns, netted training-book returns, admission/pruning decisions and independent execution checks make this replay auditable.</p></section>
</main></body></html>"""
    (output / "Five_Currents_Consistency_Objectives.html").write_text(text, encoding="utf-8")
