"""Offline report for the uniform active-contract intraday replay."""
from __future__ import annotations

import base64
import json
from html import escape
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from five_currents_front_intraday import evaluate
from five_currents_opening_study import stats
from report_five_currents_cached_window_replay import (
    CLOCK_LABELS, HORIZONS, VARIANT_LABELS, _metric_cell, complete_breakdown, fmt_table,
)
from run_five_currents_uniform_active_intraday import OUT, BOOKS


def figure_data(fig) -> str:
    data = BytesIO()
    fig.savefig(data, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(data.getvalue()).decode("ascii")


def matrix_html(wide: pd.DataFrame, book: str, signal: str, title: str, opened=False) -> str:
    frame = wide.loc[wide.book.eq(book) & wide.signal.eq(signal)].set_index(["clock", "variant", "hours"])
    if len(frame) != 288:
        raise AssertionError(f"Incomplete report matrix: {book}/{signal}")
    rows = []
    for clock in CLOCK_LABELS:
        for variant in VARIANT_LABELS:
            cells = "".join(f"<td>{_metric_cell(frame.loc[(clock, variant, hours)])}</td>" for hours in HORIZONS)
            rows.append(f"<tr><th>{escape(CLOCK_LABELS[clock])}</th><td>{escape(VARIANT_LABELS[variant])}</td>{cells}</tr>")
    headings = "".join(f"<th>{hours}h</th>" for hours in HORIZONS)
    return (f'<details class="matrix-section"{" open" if opened else ""}><summary>{escape(title)}</summary>'
            f'<div class="scroll"><table class="matrix"><thead><tr><th>Clock</th><th>Position type</th>{headings}</tr></thead>'
            '<tbody>' + "".join(rows) + "</tbody></table></div></details>")


def all_horizon_chart(daily: pd.DataFrame) -> str:
    colors = ("#d5a73d", "#628e89", "#c47653", "#166b5a", "#344a69", "#8d7862")
    chosen = daily.loc[daily.book.eq("five") & daily.variant.eq("reverse_change_magnitude")]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    fig.patch.set_facecolor("#f6f2e8")
    for ax, clock in zip(axes.flat, CLOCK_LABELS):
        ax.set_facecolor("#fffdf8")
        for hours, color in zip(HORIZONS, colors):
            path = chosen.loc[chosen.clock.eq(clock) & chosen.hours.eq(hours)].sort_values("session")
            if path.slippage_net_return.notna().any():
                equity = 100 * (1 + path.set_index("session").slippage_net_return).cumprod()
                ax.plot(equity.index, equity, label=f"{hours}h", color=color, lw=1.3)
        ax.set_title(CLOCK_LABELS[clock], loc="left")
        ax.set_xlim(daily.session.min(), daily.session.max())
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        if ax.lines:
            ax.legend(frameon=False, ncol=3, fontsize=8)
        else:
            ax.text(.5, .5, "No valid same-session exits", transform=ax.transAxes, ha="center")
    fig.suptitle("Five commodities | reverse position-change size | every hold", x=.08, ha="left", fontsize=16)
    fig.supylabel("Growth of 100 across observed trades, modelled costs")
    fig.tight_layout()
    return figure_data(fig)


def attribution(predictors: pd.DataFrame, moves: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    selected_moves = moves.loc[moves.clock.eq("us_open") & moves.hours.eq(4)].drop(columns=["clock", "signal_date"])
    path = daily.loc[daily.book.eq("five") & daily.clock.eq("us_open") & daily.hours.eq(4)
                     & daily.variant.eq("reverse_change_magnitude")].set_index("session")
    common = path.index[path.ready]
    portfolio_legs = evaluate(predictors.loc[predictors.signal.eq("portfolio_net")].drop(columns="signal"), selected_moves)
    portfolio_legs = portfolio_legs.loc[portfolio_legs.variant.eq("reverse_change_magnitude")
                                       & portfolio_legs.session.isin(common)]
    asset_returns = portfolio_legs.pivot(index="session", columns="root", values="slippage_net_return")
    np.testing.assert_allclose(asset_returns.sum(axis=1, min_count=5), path.loc[asset_returns.index, "slippage_net_return"],
                               atol=1e-12, rtol=0)
    asset_returns.to_parquet(OUT / "us_open_4h_common_asset_returns.parquet")
    asset_rows = [{"Asset": root, "Days": len(values), "Net contribution": float(values.sum()),
                   "Net Sharpe": stats(values)["sharpe"]} for root, values in asset_returns.items()]
    family_rows, family_daily = [], []
    families = sorted(set(predictors.signal) - {"portfolio_net"})
    for signal in families:
        legs = evaluate(predictors.loc[predictors.signal.eq(signal)].drop(columns="signal"), selected_moves)
        legs = legs.loc[legs.variant.eq("reverse_change_magnitude") & legs.session.isin(common)]
        gross = legs.groupby("session").gross_return.sum(min_count=5)
        family_daily.append(gross.rename(signal))
        for root, part in legs.groupby("root"):
            family_rows.append({"Signal family": signal, "Asset": root,
                                "Gross contribution": float(part.gross_return.sum())})
    component = pd.concat(family_daily, axis=1)
    np.testing.assert_allclose(component.sum(axis=1), portfolio_legs.groupby("session").gross_return.sum(), atol=1e-12, rtol=0)
    pd.DataFrame(family_rows).to_csv(OUT / "us_open_4h_family_asset_gross_attribution.csv", index=False)
    family_matrix = pd.DataFrame(family_rows).pivot(index="Signal family", columns="Asset", values="Gross contribution").reset_index()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.patch.set_facecolor("#f6f2e8")
    ax.set_facecolor("#fffdf8")
    colors = {"CL": "#bd6a35", "NG": "#658261", "GC": "#a88732", "SI": "#7d8587", "HG": "#3c7c88"}
    for root, values in asset_returns.items():
        ax.plot(values.index, 100 * values.cumsum(), color=colors[root], label=root, lw=1.4)
    ax.axhline(0, color="#67756d", lw=.8)
    ax.set_title(f"US open, four-hour reverse change | contributions on {len(common)} common dates", loc="left")
    ax.set_ylabel("Additive portfolio contribution, percentage points")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return pd.DataFrame(asset_rows), family_matrix, figure_data(fig)


def main() -> None:
    metrics = pd.read_csv(OUT / "metrics.csv")
    daily = pd.read_parquet(OUT / "portfolio_daily_returns.parquet")
    predictors = pd.read_parquet(OUT / "dated_predictors.parquet")
    moves = pd.read_parquet(OUT / "price_moves.parquet")
    targets = pd.read_parquet(OUT / "contract_targets.parquet")
    ledger = pd.read_parquet(OUT / "liquidity_candidates.parquet")
    maximum = ledger.loc[ledger.eligible].groupby(["session", "root"]).adv.max().rename("maximum_adv")
    checked = targets.merge(maximum, on=["session", "root"], validate="one_to_one")
    np.testing.assert_allclose(checked.adv, checked.maximum_adv, rtol=0, atol=0)
    if len(checked) != 5 * targets.session.nunique() or targets.duplicated(["session", "root"]).any():
        raise AssertionError("Each commodity/session must have exactly one selected contract")
    if not targets.volume_asof.lt(targets.liquidity_date).all() or not targets.liquidity_date.lt(targets.session).all():
        raise AssertionError("Selection does not use strictly prior liquidity observations")
    if targets.sessions_to_expiry.le(2).any():
        raise AssertionError("Selected contract is inside expiry buffer")
    matched = moves.loc[moves.status.eq("matched")]
    if not matched.month_matches_target.eq(True).all() or not matched.exit_ts.gt(matched.entry_ts).all():
        raise AssertionError("Matched bars do not preserve selected delivery month and ordering")
    np.testing.assert_allclose(matched.price_return, matched.exit_price / matched.entry_price - 1,
                               rtol=0, atol=1e-12)
    (OUT / "selection_audit.json").write_text(json.dumps({"selected_asset_sessions": len(targets),
        "highest_eligible_adv_error": float((checked.adv - checked.maximum_adv).abs().max()),
        "causal_liquidity_dates": True, "minimum_sessions_to_expiry": int(targets.sessions_to_expiry.min()),
        "matched_selected_month_pairs": len(matched), "month_matches_target": True,
        "maximum_rank_by_asset": targets.groupby("root")["rank"].max().to_dict()}, indent=2), encoding="utf-8")
    audit = json.loads((OUT / "portfolio_audit.json").read_text(encoding="utf-8"))
    counts = targets.groupby(["root", "rank"]).size().unstack(fill_value=0).reset_index()
    counts.columns = ["Asset", *(f"Rank {int(rank)}" for rank in counts.columns[1:])]
    portfolio_coverage = pd.read_csv(OUT / "portfolio_price_coverage.csv")
    coverage = portfolio_coverage.loc[portfolio_coverage.book.eq("five")].drop(columns="book")
    coverage.columns = ["Clock", "Hours", "Weekdays", "Matched months", "Active bars", "Verified months"]
    asset_coverage = pd.read_csv(OUT / "asset_price_coverage.csv")
    asset_coverage = asset_coverage.loc[asset_coverage.clock.eq("us_open") & asset_coverage.hours.isin((4, 6))].drop(columns="clock")
    asset_coverage.columns = ["Hours", "Asset", "Weekdays", "Matched months", "Active bars", "Verified months"]
    headline = metrics.loc[metrics.book.eq("five") & metrics.signal.eq("portfolio_net")
                           & metrics.clock.eq("us_open") & metrics.variant.eq("reverse_change_magnitude")
                           & metrics.basis.eq("slippage_net_return") & metrics.period.eq("full"),
                           ["hours", "days", "sharpe", "total_return", "max_drawdown"]]
    headline.columns = ["Hours", "Days", "Net Sharpe", "Compounded return", "Max drawdown"]
    subsets = pd.read_csv(OUT / "portfolio_subsets.csv")
    subsets = subsets.loc[subsets.book.eq("five") & subsets.clock.eq("us_open")
                           & subsets.hours.isin((4, 6)) & subsets.variant.eq("reverse_change_magnitude"),
                           ["hours", "subset", "days", "sharpe", "total_return"]]
    subsets.columns = ["Hours", "Subset", "Days", "Net Sharpe", "Compounded return"]
    subsets["Net Sharpe"] = [f"{value:.2f}" if count >= 20 and pd.notna(value) else "Too few dates"
                              for value, count in zip(subsets["Net Sharpe"], subsets["Days"])]
    comparison = pd.read_csv(OUT / "uniform_mask_comparison.csv")
    comparison = comparison.loc[comparison.book.eq("five") & comparison.clock.eq("us_open")
                                 & comparison.variant.eq("reverse_change_magnitude")].drop(columns=["book", "clock", "variant"])
    comparison.columns = ["Hours", "Common days", "Previous mask Sharpe", "Uniform mask Sharpe", "Previous return", "Uniform return"]
    wide = complete_breakdown(metrics)
    wide.to_csv(OUT / "complete_signal_breakdown.csv", index=False)
    sections = [matrix_html(wide, "five", "portfolio_net", "Five-commodity combined portfolio", True),
                matrix_html(wide, "four", "portfolio_net", "Four-commodity comparison, excluding silver")]
    sections.extend(matrix_html(wide, "five", name, name.replace("_", " ").title())
                    for name in sorted(set(predictors.signal) - {"portfolio_net"}))
    chart = all_horizon_chart(daily)
    asset, family, asset_chart = attribution(predictors, moves, daily)
    reference = daily.loc[daily.book.eq("five") & daily.clock.eq("us_open") & daily.hours.eq(4)
                           & daily.variant.eq("reverse_change_magnitude") & daily.ready]
    observed_start, observed_end = reference.session.min(), reference.session.max()
    observed_years = reference.groupby(reference.session.dt.year).size().rename("Matched dates").reset_index()
    observed_years.columns = ["Year", "Matched dates"]
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Five Currents | One Active Contract</title>
<style>body{{font:16px/1.5 Georgia,serif;background:#f6f2e8;color:#153f39;max-width:1100px;margin:auto;padding:32px}}
h1{{font-size:36px;line-height:1.1}}h2{{font-size:24px;margin:32px 0 10px}}.small{{font-size:13px;color:#65736a}}
.callout{{background:#e4ece3;padding:16px;border-left:5px solid #166b5a}}img{{width:100%;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font:13px/1.4 sans-serif;background:#fffdf8}}td,th{{padding:8px;border-bottom:1px solid #dfdbd1;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#e7ece4}}
.scroll{{overflow-x:auto}}.matrix-section{{margin:12px 0;border:1px solid #ded7ca;border-radius:8px;background:#fffdf8}}
summary{{cursor:pointer;padding:13px;font-weight:bold}}.matrix{{min-width:1520px}}.matrix td,.matrix th{{text-align:left;vertical-align:top}}
.matrix td{{min-width:190px;font-size:12px;white-space:nowrap}}.matrix small{{display:block;font-size:11px;color:#65736a}}
.positive{{color:#166b5a}}.negative{{color:#a64c3c}}.no-data{{color:#849087}}</style>
<p class="small">Internal cached-data replay | 14 October 2026</p><h1>One Active Contract, Five Commodities</h1>
<p>The same liquidity rule now selects one eligible nearby contract for CL, NG, GC, SI and HG. The frozen daily signals and allocator are retained. This replay covers the available intraday period through 12 October 2026.</p>
<div class="callout">The first six nearby generic ranks supply candidates. The highest five-observation average volume wins; the observation is lagged once, at least three volume observations are required, and the selected contract must be more than two trading sessions from expiry at entry. Equal volume favours the nearer delivery month, then prior OI. The selected month stays fixed through each intraday exit. Price-cache availability never chooses the winner.</div>
<h2>Where the Selector Trades</h2><div class="scroll">{fmt_table(counts)}</div>
<p class="small">All actual winners are ranks 1–2 except gold, which also uses rank 3. Empty holiday rows are skipped. Volume history follows the same delivery month across a generic-rank or identifier change. The conservative volume lag preserves the previous mask's timing rather than assuming settled volume is published at every entry clock.</p>
<h2>Selected-Month Coverage</h2><div class="scroll">{fmt_table(coverage)}</div>
<div class="scroll">{fmt_table(asset_coverage)}</div>
<p class="small">Historical inferred delivery months are estimates from the existing mapping calendar. They are matched to that month in the stored minute bars; known source contracts use their actual delivery month. Both entry and exit must remain in the same cached listed contract, within ten minutes of the scheduled times. An absent selected month remains missing. Active bars means recorded volume or trade count at both endpoints, and is a price-activity diagnostic.</p>
<h2>US-Open Reference Result</h2><p>The pre-existing reverse position-change-size rule is shown for continuity; the complete grid below contains every direction, family, clock and hold.</p>
<p class="small">The matched four-hour five-commodity dates span {observed_start.date()} to {observed_end.date()}. Selection runs through 12 October, but unavailable selected-month bars prevent a complete five-commodity return after that last matched date. The two verified-month observations do not support a meaningful Sharpe estimate.</p><div class="scroll">{fmt_table(observed_years)}</div>
<div class="scroll">{fmt_table(headline, ("Compounded return", "Max drawdown"))}</div>
<div class="scroll">{fmt_table(subsets, ("Compounded return",))}</div>
<h2>Previous and Uniform Selection</h2><div class="scroll">{fmt_table(comparison, ("Previous return", "Uniform return"))}</div>
<p class="small">Both masks use identical dates, the same frozen daily targets, the same costs, and the same matching of delivery months to minute prices. The previous mask retains its SI/HG overlay and incumbent CL/NG/GC policy; the uniform mask applies one nearby-liquidity rule across all five. A separate file, common_date_comparison.csv, also records comparison with the old route-based pricing assumption.</p>
<h2>All Holding Periods</h2><img alt="Five-commodity equity for all clocks and holds" src="data:image/png;base64,{chart}">
<p class="small">Cumulative performance uses observed trades; missing observations remain gaps and are excluded from statistics. Costs use $2.16 per micro lot per round trip plus one tick per side. Each holding period is a separate entry and complete exit. No forecast, refit cadence, or alpha parameter was tuned in this study.</p>
<h2>Asset and Signal Attribution</h2><div class="scroll">{fmt_table(asset, ("Net contribution",))}</div>
<img alt="Common-date commodity contributions" src="data:image/png;base64,{asset_chart}">
<div class="scroll">{fmt_table(family, BOOKS['five'])}</div>
<p class="small">These four-hour US-open contributions use the same complete five-commodity dates. Asset net contributions sum to portfolio daily net returns. Family-by-asset contributions are gross; standalone family costs do not add to portfolio costs because signals can offset.</p>
<h2>Full Five-Commodity Signal Breakdown</h2><p>Every matrix contains six clocks, eight position types and six holds. Cells show full-period net Sharpe, compounded return and count, followed by earlier/later Sharpe and counts. These inspected splits are descriptive.</p>
{''.join(sections)}
<p><a href="complete_signal_breakdown.csv">All portfolio and family metrics</a> &middot; <a href="signal_asset_metrics.csv">All signal-by-asset results</a> &middot; <a href="contract_targets.parquet">Daily selected contracts</a></p>
<p class="small">Independent calculation reconciled {audit['rows']:,} combined-portfolio day rows, including four- and five-commodity books. The selected contracts, candidate liquidity values, matched prices, costs, source hashes and missing-data reasons are saved beside this report.</p></html>'''
    (OUT / "report.html").write_text(html, encoding="utf-8")
    print("Report:", OUT / "report.html")


if __name__ == "__main__":
    main()
