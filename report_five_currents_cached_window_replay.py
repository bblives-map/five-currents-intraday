"""Build a transparent offline report from the cached-window replay outputs."""
from __future__ import annotations

import base64
import json
from html import escape
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from five_currents_front_intraday import evaluate
from five_currents_opening_study import daily_totals, stats
from run_five_currents_cached_window_replay import OLD_OUTPUT, OUTPUT, load_signals
from run_five_currents_masked_intraday import BOOKS, END, START


OUT = OUTPUT / "silver_copper"
AUDIT = OUT / "independent_audit"
SPLIT = pd.Timestamp("2025-07-01")
IDENTITY_BOUNDARY = pd.Timestamp("2026-07-21")


def png(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def series_for(daily: pd.DataFrame, *, clock: str, hours: int, variant: str) -> pd.Series:
    chosen = daily.loc[daily.clock.eq(clock) & daily.hours.eq(hours)
                       & daily.variant.eq(variant)].set_index("session")
    return chosen.slippage_net_return.sort_index()


def performance_table(metrics: pd.DataFrame) -> pd.DataFrame:
    wanted = metrics.loc[metrics.signal.eq("portfolio_net") & metrics.book.eq("four")
                         & metrics.basis.eq("slippage_net_return")
                         & metrics.clock.isin(("early_utc", "london_open", "london_close", "us_open"))
                         & metrics.hours.isin((2, 4, 6, 12))
                         & metrics.variant.isin(("net_magnitude", "change_magnitude",
                                                "reverse_change_magnitude"))].copy()
    wide = wanted.pivot(index=["clock", "hours", "variant"], columns="period",
                        values=["days", "sharpe", "total_return", "max_drawdown"])
    wide.columns = ["_".join(col) for col in wide.columns]
    return wide.reset_index()


HORIZONS = (1, 2, 3, 4, 6, 12)
CLOCK_LABELS = {
    "cme_open": "CME open", "early_utc": "01:00 Chicago",
    "london_open": "London open", "london_close": "London close",
    "us_open": "US open", "us_close": "US close",
}
VARIANT_LABELS = {
    "net_sign": "Net position: sign", "net_magnitude": "Net position: size",
    "change_sign": "Position change: sign", "change_magnitude": "Position change: size",
    "reverse_net_sign": "Reverse net: sign", "reverse_net_magnitude": "Reverse net: size",
    "reverse_change_sign": "Reverse change: sign",
    "reverse_change_magnitude": "Reverse change: size",
}


def complete_breakdown(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per tested signal/clock/position/horizon, including empty trials."""
    selected = metrics.loc[metrics.basis.eq("slippage_net_return")].copy()
    keys = ["book", "signal", "clock", "variant", "hours"]
    if selected.duplicated(keys + ["period"]).any():
        raise AssertionError("Duplicate replay metric keys")
    if set(selected.period) != {"full", "train", "test"}:
        raise AssertionError("Missing diagnostic period")
    wide = selected.pivot(index=keys, columns="period",
                          values=["days", "sharpe", "total_return", "max_drawdown"])
    wide.columns = ["_".join(pair) for pair in wide.columns]
    wide = wide.reset_index()
    expected = len(selected) // 3
    if len(wide) != expected or not wide.groupby(["book", "signal", "clock", "variant"]).hours.apply(
            lambda values: set(values) == set(HORIZONS)).all():
        raise AssertionError("Report does not cover every scored horizon")
    return wide


def _metric_cell(row: pd.Series) -> str:
    n = int(row.days_full)
    if not n:
        return '<span class="no-data">No complete-book sample</span>'

    def sr(value: float) -> str:
        return f"{value:+.2f}" if pd.notna(value) else "n/a"

    def pct(value: float) -> str:
        return f"{value:+.2%}" if pd.notna(value) else "n/a"

    tone = "positive" if pd.notna(row.sharpe_full) and row.sharpe_full > 0 else "negative"
    return (f'<span class="{tone}"><b>SR {sr(row.sharpe_full)}</b></span>'
            f' &middot; {pct(row.total_return_full)} &middot; {n}d'
            f'<small>Earlier SR {sr(row.sharpe_train)} ({int(row.days_train)}d)'
            f' / later SR {sr(row.sharpe_test)} ({int(row.days_test)}d)</small>')


def complete_breakdown_html(wide: pd.DataFrame) -> str:
    groups = [("four", "portfolio_net", "Four-market portfolio: CL, NG, GC, HG"),
              ("five", "portfolio_net", "Five-market portfolio: CL, NG, GC, SI, HG"),
              ("four", "portfolio_finalized_reference", "Finalized-flow reference (not entry-timed)")]
    families = sorted(set(wide.loc[wide.book.eq("four"), "signal"])
                      - {"portfolio_net", "portfolio_finalized_reference"})
    groups.extend(("four", name, name.replace("_", " ").title()) for name in families)
    sections = []
    for book, signal, label in groups:
        group = wide.loc[wide.book.eq(book) & wide.signal.eq(signal)]
        if len(group) != len(CLOCK_LABELS) * len(VARIANT_LABELS) * len(HORIZONS):
            raise AssertionError(f"Incomplete matrix for {book}/{signal}: {len(group)}")
        by_key = group.set_index(["clock", "variant", "hours"])
        rows = []
        for clock in CLOCK_LABELS:
            for variant in VARIANT_LABELS:
                cells = [f"<td>{_metric_cell(by_key.loc[(clock, variant, hours)])}</td>"
                         for hours in HORIZONS]
                rows.append("<tr><th scope='row'>" + escape(CLOCK_LABELS[clock])
                            + "</th><td>" + escape(VARIANT_LABELS[variant]) + "</td>"
                            + "".join(cells) + "</tr>")
        headers = "".join(f"<th scope='col'>{hours}h</th>" for hours in HORIZONS)
        opened = " open" if book == "four" and signal == "portfolio_net" else ""
        sections.append(f'<details class="matrix-section"{opened}><summary>{escape(label)}'
                        f' <span class="small">48 signal/clock rows, all six holds</span></summary>'
                        f'<div class="scroll"><table class="matrix"><thead><tr><th>Clock</th>'
                        f'<th>Position type</th>{headers}</tr></thead><tbody>'
                        + "".join(rows) + "</tbody></table></div></details>")
    return "\n".join(sections)


def sensitivity(moves: pd.DataFrame, daily: pd.DataFrame, hours: int) -> pd.DataFrame:
    selected = moves.loc[moves.clock.eq("us_open") & moves.hours.eq(hours)
                         & moves.root.isin(BOOKS["four"])]
    marked = selected.pivot(index="session", columns="root", values="status").eq("matched").all(axis=1)
    traded = selected.pivot(index="session", columns="root", values="trade_pair_confirmed").eq(True).all(axis=1)
    values = series_for(daily, clock="us_open", hours=hours, variant="reverse_change_magnitude")
    cases = {
        "All matched cached marks": marked,
        "Both endpoints trade-confirmed": traded,
        "Historical rank-route proxy": marked & (marked.index < IDENTITY_BOUNDARY),
        "Recent verified month": marked & (marked.index >= IDENTITY_BOUNDARY),
    }
    rows = []
    for label, mask in cases.items():
        result = stats(values.reindex(mask.index)[mask])
        rows.append({"Holding hours": hours, "Price/identity subset": label,
                     "Days": result["days"], "Net Sharpe": result["sharpe"],
                     "Cumulative return": result["total_return"]})
    return pd.DataFrame(rows)


def yearly(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hours in (4, 6):
        series = series_for(daily, clock="us_open", hours=hours,
                            variant="reverse_change_magnitude")
        for year, values in series.groupby(series.index.year):
            result = stats(values)
            rows.append({"Year": year, "Holding hours": hours,
                         "Days": result["days"], "Net Sharpe": result["sharpe"],
                         "Cumulative return": result["total_return"]})
    return pd.DataFrame(rows)


def equity_chart(daily: pd.DataFrame) -> str:
    palette = {"4h reverse change": "#166b5a", "6h reverse change": "#b47927",
               "4h original change": "#a64c3c"}
    cases = {
        "4h reverse change": (4, "reverse_change_magnitude"),
        "6h reverse change": (6, "reverse_change_magnitude"),
        "4h original change": (4, "change_magnitude"),
    }
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.patch.set_facecolor("#f6f2e8")
    ax.set_facecolor("#fffdf8")
    for label, (hours, variant) in cases.items():
        values = series_for(daily, clock="us_open", hours=hours, variant=variant).dropna()
        equity = 100 * (1 + values).cumprod()
        ax.plot(equity.index, equity, label=label, color=palette[label], lw=1.8)
    ax.axvline(SPLIT, color="#62716a", ls="--", lw=1, label="Diagnostic split")
    ax.axvspan(IDENTITY_BOUNDARY, pd.Timestamp(END), color="#d5e5dc", alpha=.35,
               label="Exact-month identity available")
    ax.set_ylabel("Growth of 100, modelled costs")
    ax.set_title("US open | full cached-sample equity path", loc="left", fontsize=15)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return png(fig)


def asset_chart(moves: pd.DataFrame, daily: pd.DataFrame) -> str:
    sessions = pd.bdate_range(START, END)
    predictors, _ = load_signals(sessions)
    predictors.to_parquet(OUT / "dated_predictors.parquet", index=False)
    predictors = predictors.loc[predictors.signal.eq("portfolio_net")].drop(columns="signal")
    selected_moves = moves.loc[moves.clock.eq("us_open") & moves.hours.eq(4)]
    selected_moves = selected_moves.drop(columns=["clock", "signal_date"])
    paths = evaluate(predictors, selected_moves)
    selected = paths.loc[paths.variant.eq("reverse_change_magnitude")
                         & paths.root.isin(BOOKS["four"])]
    asset = selected.pivot(index="session", columns="root", values="slippage_net_return")
    asset = asset.reindex(columns=BOOKS["four"])
    common = asset.dropna(how="any")
    portfolio = series_for(daily, clock="us_open", hours=4,
                           variant="reverse_change_magnitude").reindex(common.index)
    np.testing.assert_allclose(common.sum(axis=1), portfolio, atol=1e-12, rtol=0)
    asset.to_parquet(OUT / "us_open_4h_asset_returns.parquet")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor("#f6f2e8")
    ax.set_facecolor("#fffdf8")
    colors = {"CL": "#bd6a35", "NG": "#8a9c68", "GC": "#a88732", "HG": "#3c7c88"}
    for root in BOOKS["four"]:
        ax.plot(common.index, 100 * common[root].cumsum(), label=root,
                color=colors[root], lw=1.6)
    ax.axhline(0, color="#62716a", lw=.8)
    ax.axvspan(IDENTITY_BOUNDARY, pd.Timestamp(END), color="#d5e5dc", alpha=.35)
    ax.set_ylabel("Cumulative contribution, % of unit book")
    ax.set_title("US open, four-hour reverse change | asset contributions", loc="left", fontsize=15)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return png(fig)


def fmt_table(frame: pd.DataFrame, percentages=()) -> str:
    shown = frame.copy()
    for column in shown:
        if column in percentages:
            shown[column] = shown[column].map(lambda x: f"{x:.2%}" if pd.notna(x) else "n/a")
        elif pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")
    return shown.to_html(index=False, escape=True, border=0)


def reconcile_old_exact_prices(moves: pd.DataFrame) -> dict:
    old = pd.read_parquet(OLD_OUTPUT / "price_moves.parquet")
    old = old.loc[old["mode"].eq("rank_proxy_traded") & old.status.eq("matched"),
                  ["session", "root", "clock", "hours", "local_symbol", "entry_utc",
                   "exit_utc", "price_return"]]
    new = moves.loc[moves.status.eq("matched"),
                    ["session", "root", "clock", "hours", "local_symbol", "entry_ts",
                     "exit_ts", "price_return"]]
    common = old.merge(new, on=["session", "root", "clock", "hours", "local_symbol"],
                       suffixes=("_old", "_new"))
    same_stamps = common.loc[common.entry_utc.eq(common.entry_ts)
                             & common.exit_utc.eq(common.exit_ts)]
    error = (same_stamps.price_return_old - same_stamps.price_return_new).abs()
    maximum = float(error.max()) if len(error) else None
    if maximum is None or maximum > 1e-12:
        raise AssertionError(f"Exact-endpoint price reconciliation failed: {maximum}")
    result = {"same_contract_same_timestamp_pairs": int(len(same_stamps)),
              "max_absolute_return_error": maximum,
              "reconciliation_basis": "same session, root, clock, horizon, local symbol and both bar timestamps"}
    (OUT / "reconciliation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    moves = pd.read_parquet(OUT / "price_moves.parquet")
    daily = pd.read_parquet(OUT / "portfolio_daily_returns.parquet")
    metrics = pd.read_csv(OUT / "metrics.csv")
    coverage = pd.read_csv(OUT / "coverage.csv")
    audit = json.loads((AUDIT / "audit_summary.json").read_text(encoding="utf-8"))
    if audit["portfolio_daily"]["rows"] != len(daily) or audit["all_metrics"]["rows"] != len(metrics):
        raise AssertionError("Independent audit does not match current replay row counts")
    asset_audit = pd.read_csv(AUDIT / "candidate_asset_attribution.csv")
    asset_audit = asset_audit[["hours", "root", "days", "net_contribution", "sharpe",
                               "trade_confirmed_days"]]
    asset_audit.columns = ["Hold", "Asset", "Days", "Net contribution", "Net Sharpe",
                           "Trade-confirmed legs"]
    family_audit = pd.read_csv(AUDIT / "candidate_family_attribution.csv")
    family_audit = family_audit.pivot(index="signal", columns="hours",
                                    values=["gross_contribution", "standalone_net_sharpe"])
    family_audit.columns = [f"{hours}h {('gross contribution' if metric == 'gross_contribution' else 'standalone net Sharpe')}"
                            for metric, hours in family_audit.columns]
    family_audit = family_audit.reset_index().rename(columns={"signal": "Signal family"})
    family_asset_audit = pd.read_csv(AUDIT / "candidate_family_asset_attribution.csv")
    family_asset_audit = family_asset_audit.pivot(index=["hours", "signal"], columns="root",
                                                  values="gross_contribution").reset_index()
    family_asset_audit = family_asset_audit.rename(columns={"hours": "Hold", "signal": "Signal family"})
    portfolio_bridge = pd.DataFrame([
        {"Hold": int(hours), "Gross contribution": values["portfolio_gross_contribution"],
         "Modelled costs": values["portfolio_cost_drag"],
         "Net contribution": values["portfolio_net_contribution"]}
        for hours, values in audit["candidate_attribution"].items()
    ]).sort_values("Hold")
    subset_audit = pd.read_csv(AUDIT / "portfolio_subsets.csv")
    subset_audit = subset_audit.loc[subset_audit.hours.isin((4, 6))
                                    & subset_audit.subset.isin(("all matched", "all four trade-confirmed",
                                                                "historical rank-route proxy",
                                                                "recent verified month")),
                                    ["hours", "subset", "days", "sharpe", "total_return"]]
    subset_audit.columns = ["Hold", "Price/identity subset", "Days", "Net Sharpe", "Cumulative return"]
    loo_audit = pd.read_csv(AUDIT / "candidate_leave_one_out.csv")
    loo_audit = loo_audit[["hours", "omitted_asset", "days", "sharpe", "total_return"]]
    loo_audit.columns = ["Hold", "Omitted asset", "Days", "Net Sharpe", "Cumulative return"]
    reconciliation = reconcile_old_exact_prices(moves)
    gold = pd.read_parquet(OUTPUT / "silver_copper_gold" / "price_moves.parquet")
    gold = gold.loc[gold.root.eq("GC") & gold.clock.eq("early_utc") & gold.hours.eq(2)
                    & gold["rank"].eq(3)]
    if len(gold) != 51 or gold.status.eq("matched").any():
        raise AssertionError("Unexpected local rank-3 gold coverage")
    full = complete_breakdown(metrics)
    full.to_csv(OUT / "complete_signal_breakdown.csv", index=False)
    full_html = complete_breakdown_html(full)
    perf = performance_table(metrics)
    perf = perf.loc[perf.clock.eq("us_open") & perf.hours.isin((4, 6))
                    & perf.variant.isin(("net_magnitude", "change_magnitude",
                                         "reverse_change_magnitude"))]
    perf = perf[["clock", "hours", "variant", "days_full", "sharpe_train", "sharpe_test",
                 "sharpe_full", "total_return_full", "max_drawdown_full"]]
    perf.columns = ["Clock", "Hours", "Position", "Days", "Train Sharpe", "Later Sharpe",
                    "Full Sharpe", "Cumulative return", "Max drawdown"]
    sens = pd.concat([sensitivity(moves, daily, 4), sensitivity(moves, daily, 6)],
                     ignore_index=True)
    year = yearly(daily)
    family = metrics.loc[metrics.clock.eq("us_open") & metrics.hours.eq(4)
                         & metrics.variant.eq("reverse_change_magnitude")
                         & metrics.book.eq("four") & metrics.basis.eq("slippage_net_return")
                         & ~metrics.signal.isin(("portfolio_net", "portfolio_finalized_reference"))]
    family = family.pivot(index="signal", columns="period", values="sharpe")
    family = family.rename(columns={"train": "Earlier Sharpe", "test": "Later Sharpe"})
    family.index.name = "Signal family"
    family = family.reset_index().sort_values("Earlier Sharpe", ascending=False)
    cov = coverage.loc[coverage.book.eq("four")].copy()
    cov = cov.groupby(["clock", "hours"], as_index=False)[["sessions", "matched_days",
                                                             "trade_confirmed_days"]].sum()
    cov.columns = ["Clock", "Hours", "Weekdays", "Matched marks", "Trade-confirmed pairs"]
    silver_coverage = coverage.loc[coverage.book.eq("five") & coverage.clock.eq("us_open")
                                   & coverage.hours.eq(4)]
    five_days = int(silver_coverage.matched_days.sum())
    five_recent = int(silver_coverage.loc[silver_coverage.identity_basis.eq("verified_month"),
                                          "matched_days"].sum())
    equity = equity_chart(daily)
    assets = asset_chart(moves, daily)
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Five Currents | Cached-Window Intraday Audit</title>
<style>
body{{font:16px/1.5 Georgia,serif;background:#f6f2e8;color:#153f39;max-width:1100px;margin:0 auto;padding:34px}}
h1{{font-size:38px;line-height:1.08;margin:10px 0 14px}}h2{{font-size:24px;margin:36px 0 10px}}
p{{max-width:950px}}.eyebrow{{font:700 12px Arial,sans-serif;letter-spacing:.16em;text-transform:uppercase;color:#b47927}}
.lead{{font-size:20px;color:#34534b}}.callout{{background:#e4ece3;border-left:5px solid #166b5a;padding:16px 20px}}
.warning{{background:#f4e5d5;border-left-color:#b47927}}img{{width:100%;border:1px solid #ded7ca;border-radius:8px;background:#fffdf8}}
table{{border-collapse:collapse;width:100%;font:13px/1.35 Arial,sans-serif;background:#fffdf8}}
th,td{{text-align:right;padding:9px;border-bottom:1px solid #e0ded5}}th:first-child,td:first-child{{text-align:left}}
th{{background:#e7ece4;color:#153f39}}.scroll{{overflow-x:auto}}.small{{font-size:13px;color:#5b665f}}
.matrix-section{{margin:14px 0;background:#fffdf8;border:1px solid #ded7ca;border-radius:8px}}
.matrix-section summary{{cursor:pointer;padding:13px 16px;font:700 15px/1.4 Arial,sans-serif}}
.matrix-section summary span{{font-weight:400;margin-left:12px}}.matrix{{min-width:1520px}}
.matrix th,.matrix td{{text-align:left;vertical-align:top;padding:7px 9px;border-right:1px solid #eee9df}}
.matrix thead th{{position:sticky;top:0}}.matrix tbody tr:hover{{background:#edf4ed}}
.matrix td{{font-size:12px;min-width:190px;white-space:nowrap}}.matrix td small{{display:block;color:#68746d;font-size:11px}}
.matrix .positive{{color:#166b5a}}.matrix .negative{{color:#a64c3c}}.no-data{{color:#849087}}
</style><div class="eyebrow">Internal research audit | 13 October 2026</div>
<h1>Five Currents, Re-timed</h1>
<p class="lead">A cached-data replay of daily signals at market clocks. Entries and exits use the nearest locally stored one-minute price within ten minutes of the scheduled time; the position is closed in the same listed contract.</p>
<div class="callout"><strong>Coverage correction.</strong> The earlier 245-day count reflected stricter trade-confirmation and narrower cached-price coverage, not a loss of daily forecasts. At the 01:00 Chicago clock with a two-hour hold, this replay has 548 four-market price-mark days out of 603 weekdays. The expanded ten-minute window does not create these early-Chicago or US-open observations: their matched entries and exits are at the exact scheduled minute. Zero-volume marks are indicative, not verified fills.</div>
<h2>What Was Held Fixed</h2><p>The archived daily signals, lagged-volume/OI policy, constituent family weights, and modelled micro round-trip commission plus one tick per side were not retuned. The headline four-market book is CL, NG, GC, and HG. In this mask every selected rank is 1 or 2. The signal comes from the prior completed daily observation. No broker, database, or market API was accessed for this replay.</p>
<p class="small">Before 21 July 2026, the archived daily panel identifies generic ranks but not exchange contract months: those prices are front/nearby route proxies. Later rows require the daily selected month to match the stored listed contract. The optional gold-inclusive mask selects rank 3 on 51 intraday-window dates; none of those exact contracts is available in the local front/nearby caches, so that mask is not scored here.</p>
<h2>Coverage by Clock and Hold</h2><div class="scroll">{fmt_table(cov)}</div>
<p class="small">Every tested 1, 2, 3, 4, 6 and 12-hour hold is shown, including combinations with zero valid price pairs. US close plus one to twelve hours falls in or crosses the normal maintenance break, so this same-session test has no valid US-close exit. Four-market days require a valid price pair in each market; trade-confirmed pairs are a stricter subset. Adding SI leaves {five_days} five-market US-open/four-hour matched days, only {five_recent} of them in the recent exact-month segment, so SI is not part of the headline comparison.</p>
<h2>Independent Replay Audit</h2><p>An independent calculation matched all {audit['portfolio_daily']['rows']:,} saved portfolio-day rows and all {audit['all_metrics']['rows']:,} metric rows to numerical precision. Every position and position change matches the archived lagged daily target and its prior observation. The ten family forecasts sum to the portfolio, and every matched entry and exit agrees with its local cached contract bar. This verifies the arithmetic and timestamp/contract matching; it does not validate the older inferred contract route as the actual historical trade.</p>
<div class="scroll">{fmt_table(portfolio_bridge, ("Gross contribution", "Modelled costs", "Net contribution"))}</div>
<p class="small">Gross minus modelled commission and slippage equals net on the same complete-book dates. These are additive daily-return contributions; the compounded equity gain differs slightly.</p>
<div class="scroll">{fmt_table(subset_audit, ("Cumulative return",))}</div>
<p class="small">The four-hour result is positive on 209 days with trades at both endpoints, but 519 of its 551 complete-book days use historical rank-route proxies. Only 32 days have exact selected contract months; the six-hour result is negative on those 32 days. These are differing samples, not interchangeable out-of-sample trials.</p>
<h2>Asset Attribution on Identical Dates</h2><div class="scroll">{fmt_table(asset_audit, ("Net contribution",))}</div>
<p class="small">Net contribution is the sum of each asset's daily portfolio return on the same complete-book dates; it adds to the portfolio's simple daily-return sum, not exactly to compounded growth. HG supplies most of the positive return, while NG detracts. The table below removes one asset at a time without rescaling the remaining risk; it is a fragility diagnostic, not a revised strategy.</p>
<div class="scroll">{fmt_table(loo_audit, ("Cumulative return",))}</div>
<h2>Signal Attribution on Identical Dates</h2><div class="scroll">{fmt_table(family_audit, ("4h gross contribution", "6h gross contribution"))}</div>
<p class="small">Family gross contributions add exactly to portfolio gross P&amp;L. Standalone net Sharpe includes the costs of trading each family separately and therefore must not be added across families: offsetting sleeves can trade against one another. The largest four-hour gross contribution is curve-quality reversal, mainly in HG.</p>
<details class="matrix-section"><summary>Signal by asset gross contribution <span class="small">Both four- and six-hour holds</span></summary><div class="scroll">{fmt_table(family_asset_audit, ("CL", "NG", "GC", "HG"))}</div></details>
<p class="small">GC's four-hour net contribution is +0.78 percentage points, but about +0.76 points comes from bars without trades at both endpoints. These marks may be valid prices but are not executable fill evidence. Together with HG concentration and the short verified-month sample, this limits confidence in a diversified intraday edge.</p>
<h2>Complete Signal and Holding-Period Breakdown</h2>
<p>Expand each portfolio or signal family below. Each matrix contains all six clocks, all eight position types, and all six holding periods. A cell shows full-period, after-cost Sharpe, compounded return and matched-day count; the second line gives earlier and later Sharpe with their own day counts. The split is descriptive and was inspected during this research, not an untouched test. <a href="complete_signal_breakdown.csv">Download the complete numerical table</a>.</p>
<p class="small">Net position is the level of the previous completed daily forecast; position change is its day-on-day change. Sign trades direction only, while size uses forecast magnitude. Reverse flips the trade direction. All ten economic families use the lagged-volume/OI version. The finalized-flow portfolio is shown only as a timing-uncertain reference, not as an executable early signal. A family can have more usable days than the portfolio because an inactive commodity needs no price mark. Sharpe is n/a for zero-variance returns, including inactive sleeves; zero complete-book days are called out separately.</p>
{full_html}
<p><strong>Across horizons:</strong> The four-market reverse-change-size portfolio is negative at the US open after one hour, then positive over two, three, four and six hours; the twelve-hour US-open exit has no complete-book sample. Twelve-hour holds at CME open, 01:00 Chicago and London open are positive for this same position type, whereas their shorter holds are mostly negative. Net-position size is weak over short US-open holds; sign-only trades do not reproduce the positive reverse-change-size result. These are descriptive observations across inspected alternatives, not a rule chosen on untouched data.</p>
<h2>US-Open Directional Diagnostic</h2><p class="small">This compact comparison is not the full search: every clock, position type, signal family, and hold appears in the matrices above.</p><div class="scroll">{fmt_table(perf, ("Cumulative return", "Max drawdown"))}</div>
<p>On this sample the unreversed short-horizon net position performs poorly. The aggregate reverse position-change direction at the US open is positive at both four and six hours in the earlier and later split. The six-hour horizon ranks highest in the earlier split, but the four-hour result has the stronger exact-month and trade-confirmed sensitivity. The later split was inspected during this research and is not untouched out-of-sample evidence. Neither observation justifies live deployment.</p>
<img alt="Full cached-sample equity curves" src="data:image/png;base64,{equity}"><p class="small">Lines connect observed sessions; missing sessions are omitted from statistics and are not counted as zero-return days. The shading marks the period with archived exact-month identities, not an out-of-sample boundary.</p>
<h2>Year-by-Year and Source Sensitivity</h2><div class="scroll">{fmt_table(year, ("Cumulative return",))}</div>
<div class="scroll">{fmt_table(sens, ("Cumulative return",))}</div>
<p class="small">The recent exact-month subset is only about 32 matched four-market days at the US open. The four-hour result also survives on 209 days with trades confirmed at both endpoints; six hours is weaker on its confirmed subset. These subsets have different calendars, so their Sharpe ratios are diagnostics, not directly comparable portfolio trials.</p>
<h2>Where the Result Comes From</h2><img alt="Cumulative asset contributions" src="data:image/png;base64,{assets}">
<h2>Family Stability</h2><p>The earlier-versus-later result for <em>every</em> family, clock, position type and hold appears in its matrix above. As one illustration, the four-hour US-open reverse-change slice is below; it is not a selected family portfolio.</p><div class="scroll">{fmt_table(family)}</div>
<p>Many apparently strong individual families in the earlier sample turn negative later. I would not promote a hindsight-selected family combination from this search. The four-hour reverse-change rule is at most a provisional paper-trading candidate; the six-hour rule is weaker on trade-confirmed days and negative in the short exact-month period. Record actual contract, spread, and fill time rather than treating marked prices as executable. The cached-data result remains vulnerable to proxy-month error and stale marked bars.</p>
<p class="small">On {reconciliation['same_contract_same_timestamp_pairs']:,} price pairs with identical contract and exact timestamps, the new and prior ledgers agree to a maximum absolute return error of {reconciliation['max_absolute_return_error']:.1g}. Files alongside this report contain the dated forecasts, per-asset return ledger, every attempted price pair and its status, coverage, metrics, and source manifest. This is a research replay, not a verified executable P&amp;L record.</p></html>"""
    (OUT / "report.html").write_text(html, encoding="utf-8")
    print("Report:", OUT / "report.html")


if __name__ == "__main__":
    main()
