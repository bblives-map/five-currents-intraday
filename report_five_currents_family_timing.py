"""Render the cross-family walk-forward repeat, including every declared configuration."""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from five_currents_family_timing import FITS
from five_currents_execution_timing import ROOTS, summarize_matrix, block_interval
from prepare_five_currents_execution_timing import save_json, digest
from run_five_currents_family_timing import OUTPUT


TITLE = "Five Currents | Learning Which Families to Trade"
COLORS = ["#176b5c", "#b87543", "#5b7896", "#a39340", "#9b6778"]
METHOD = {"fixed_ten": "Fixed ten-family budgets", "best_family": "One selected family",
          "greedy_three": "Greedy diversified / at most three"}


def fmt(value, kind="num"):
    if value is None or pd.isna(value):
        return "--"
    if kind == "pct":
        return f"{value*100:.2f}%"
    if kind == "int":
        return f"{value:,.0f}"
    return f"{value:.2f}" if isinstance(value, (float, np.floating)) else html.escape(str(value))


def table(frame, columns):
    headings = "".join(f"<th>{label}</th>" for key, label, kind in columns)
    body = "".join("<tr>"+"".join("<td>"+fmt(row[key], kind)+"</td>" for key, _, kind in columns)+"</tr>"
                   for row in frame.to_dict("records"))
    return f'<div class="table"><table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div>'


def decorate(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(axis="y", alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)


def plot_path(ax, frame, label, color=None):
    # cumprod(skipna) preserves visible gaps; no price or P&L interpolation.
    ax.plot(frame.session, (1+frame.net).cumprod()*100, label=label, lw=1.4, color=color)
    if not getattr(ax, "_shaded", False) and frame.calibrating.any():
        ax.axvspan(frame.session.min(), frame.loc[frame.calibrating, "session"].max(), alpha=.08, color="#5b7896")
        ax._shaded = True
    decorate(ax)


def report(output):
    manifest = json.loads((output / "manifest.json").read_text())
    paths = pd.read_parquet(output / "paths.parquet")
    paths["session"] = pd.to_datetime(paths.session)
    metrics = pd.read_csv(output / "metrics.csv")
    post = metrics.loc[metrics.period.eq("post_calibration")].copy()
    stress = pd.read_csv(output / "stress.csv")
    attribution = pd.read_csv(output / "attribution_common_dates.csv")
    asset = attribution.loc[attribution.kind.eq("asset")].rename(columns={"component": "root"})
    family = attribution.loc[attribution.kind.eq("family")].rename(columns={"component": "family"})
    concentration = pd.read_csv(output / "concentration.csv")
    controls = pd.read_csv(output / "direction_controls.csv")
    ledger = pd.read_parquet(output / "refit_ledger.parquet")
    groups = {key: frame.sort_values("session").reset_index(drop=True) for key, frame in paths.groupby("id")}
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#faf8f2",
        "axes.facecolor": "#faf8f2", "font.size": 9, "text.color": "#203a35", "axes.labelcolor": "#203a35"})
    images = []

    def save(name, title, caption, fig):
        path = figures / (name+".png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        images.append((name, title, caption))

    def sid(grid="regional", timeline="older_settled", book="five", scope="shared", fit="252_monthly", method="greedy_three"):
        return "__".join((grid, timeline, book, scope, fit, method))

    def primary(frame):
        return frame.loc[frame.grid.eq("regional") & frame.book.eq("five") & frame.fit.eq("252_monthly")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    for row, timeline in enumerate(("fresh_lagged", "older_settled")):
        for col, scope in enumerate(("shared", "assetwise")):
            ax = axes[row, col]
            for j, method in enumerate(METHOD):
                plot_path(ax, groups[sid(timeline=timeline, scope=scope, method=method)], METHOD[method], COLORS[j])
            ax.set_title(f'{timeline.replace("_", " ").title()} | {scope} family selection', loc="left", fontweight="bold")
            ax.set_ylabel("Observed growth of 100")
            ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save("main", "Cross-family learning versus fixed family budgets",
         "Five markets, 252-session monthly refits; four regional windows. Shading is initial calibration cash, not suppressed returns. Gaps remain unpriced active trades. This configuration was specified before reading this repeat's results.", fig)

    for grid in ("regional", "fixed_hold"):
        frame = post.loc[post.grid.eq(grid) & post.method.eq("greedy_three") & post.fit.isin(FITS)]
        matrix = frame.pivot(index=["timeline", "book", "scope"], columns="fit", values="sharpe").reindex(columns=FITS)
        fig, ax = plt.subplots(figsize=(11, 5.4))
        data = matrix.to_numpy()
        im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap="RdBu", vmin=-2, vmax=2)
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(matrix)), [" / ".join(key) for key in matrix.index])
        ax.spines[:].set_visible(False)
        for i, j in np.ndindex(data.shape):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", color="white" if abs(data[i,j]) > 1.4 else "#233731")
        fig.colorbar(im, ax=ax, label="Net OOS Sharpe")
        fig.tight_layout()
        save(grid+"_fits", grid.replace("_", " ").title()+" | all rolling fits",
             "Weekly = five observed sessions; monthly = 21. Windows expand from 126 observations to their stated size. Four markets excludes silver without levering the remaining budgets up. Expanding fits never discard earlier history.", fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4))
    for ax, timeline in zip(axes, ("fresh_lagged", "older_settled")):
        frame = groups[sid(timeline=timeline)]
        complete = np.isfinite(frame.net.to_numpy())
        for i, root in enumerate(ROOTS):
            values = frame[root].where(complete)
            ax.plot(frame.session, values.cumsum()*100, label=root, lw=1.3, color=COLORS[i])
        ax.plot(frame.session, frame.net.cumsum()*100, label="Portfolio", color="#172c29", lw=1.8)
        ax.set_title(timeline.replace("_", " ").title()+" | shared family admission", loc="left", fontweight="bold")
        ax.set_ylabel("Cumulative return contribution (pp)")
        ax.legend(ncol=6, fontsize=8)
        decorate(ax)
    fig.tight_layout()
    save("assets", "Where portfolio P&L is coming from",
         "Additive asset contributions on complete portfolio dates only, not independently rescaled single-market strategies. Missing portfolio outcomes are not counted as flat returns.", fig)

    with np.load(output / "schedules.npz") as saved:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6.4))
        for ax, timeline in zip(axes, ("fresh_lagged", "older_settled")):
            alpha = saved[sid(timeline=timeline)+"__allocations"].mean(axis=1).T
            extent = [mdates.date2num(paths.session.min()), mdates.date2num(paths.session.max()), len(alpha)-.5, -.5]
            im = ax.imshow(alpha, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1, extent=extent)
            ax.set_yticks(range(len(alpha)), [name.replace("_", " ") for name in manifest["families"]], fontsize=8)
            ax.set_title(timeline.replace("_", " ").title()+" | shared / 252 monthly", loc="left", fontweight="bold")
            decorate(ax)
        fig.tight_layout()
        save("stability", "Family admission over time",
             "Nonnegative family weights fixed until the next refit. Selection does not guarantee a nonzero daily vote. Grey-free pale early history is calibration cash; no family is retrospectively removed.", fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8))
    for ax, timeline in zip(axes, ("fresh_lagged", "older_settled")):
        for i, (grid, book) in enumerate((("regional", "five"), ("regional", "four"), ("fixed_hold", "five"), ("fixed_hold", "four"))):
            plot_path(ax, groups[sid(grid=grid, timeline=timeline, book=book)], f"{grid} / {book}", COLORS[i])
        ax.set_title(timeline.replace("_", " ").title()+" | same 252-monthly greedy rule", loc="left", fontweight="bold")
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    save("grid_comparison", "Does broad clock/holding-period tuning help?",
         "Regional grid: 16 recipes per family. Fixed-duration grid: 224 recipes per family, covering all seven requested clocks and 30/60/90/120/180/240/360/720-minute holds where the trade fits within the session. Every recipe can use direct/reversed position or position change.", fig)

    # Show the full unchanged daily core as context, not as a spliced intraday result.
    source = Path(manifest["source"])
    history = json.loads((source / "inputs/daily_dashboard_history.json").read_text())
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(pd.to_datetime(history["dates"]), history["portfolio"], color=COLORS[0], lw=1.6)
    ax.axvspan(pd.Timestamp(manifest["start"]), pd.Timestamp(manifest["end"]), alpha=.09, color=COLORS[2])
    ax.set_ylabel("Growth of 100")
    decorate(ax)
    save("daily_core", "The interday core is not changed by this experiment",
         "Full cached daily history from 2020. Highlighted section is the available intraday sample; no 2020-2023 minute bars are invented, and no intraday profit is added to the daily core.", fig)

    paired_rows = []
    for (grid, timeline, book, scope, fit), batch in post.groupby(["grid", "timeline", "book", "scope", "fit"]):
        frames = [groups[row.id] for row in batch.itertuples()]
        common = np.logical_and.reduce([np.isfinite(f.net.to_numpy()) & ~f.calibrating.to_numpy() for f in frames])
        for frame in frames:
            row = frame.iloc[0]
            planned = (frame.exposure.to_numpy() > 1e-12) & ~frame.calibrating.to_numpy()
            paired_rows.append({"id": row.id, "common_days": int(common.sum()),
                "paired_net_sharpe": summarize_matrix(frame.net.to_numpy()[common])["sharpe"][0],
                "planned_trade_days": int(planned.sum()),
                "priced_trade_coverage": float(np.isfinite(frame.net.to_numpy()[planned]).mean()) if planned.any() else np.nan})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "paired_method_comparison.csv", index=False)
    post = post.merge(paired, on="id", how="left")
    intervals = []
    for timeline in ("fresh_lagged", "older_settled"):
        for scope in ("shared", "assetwise"):
            frame = groups[sid(timeline=timeline, scope=scope)]
            intervals.append({"timeline": timeline, "scope": scope,
                              **block_interval(frame.loc[~frame.calibrating, "net"].to_numpy())})
    pd.DataFrame(intervals).to_csv(output / "conditional_intervals.csv", index=False)
    main = primary(post).sort_values(["timeline", "scope", "method"])
    m = primary(metrics).loc[lambda f: f.method.eq("greedy_three") & f.scope.eq("shared")]
    main.to_csv(output / "primary_comparison.csv", index=False)
    recent = ledger.loc[ledger.grid.eq("regional") & ledger.fit.eq("252_monthly")
                        & ledger.method.eq("greedy_three") & ledger.pool.ne("four")].copy()
    recent = recent.loc[recent.start.eq(recent.groupby(["timeline", "pool"]).start.transform("max"))
                        & recent.allocation.gt(0)]
    recent.to_csv(output / "last_simulated_refit.csv", index=False)
    secondary = post.loc[post.fit.isin(FITS) & post.method.ne("fixed_ten")].sort_values("sharpe", ascending=False).iloc[0]
    secondary_stress = stress.loc[stress.id.eq(secondary.id)]
    secondary_concentration = concentration.loc[concentration.id.eq(secondary.id)]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for j, method in enumerate(METHOD):
        key = sid(grid=secondary.grid, timeline=secondary.timeline, book=secondary.book,
                  scope=secondary.scope, fit=secondary.fit, method=method)
        plot_path(ax, groups[key], METHOD[method], COLORS[j])
    ax.legend(fontsize=8)
    ax.set_ylabel("Observed growth of 100")
    save("secondary", "Strongest new-selection row | descriptive, not a fresh holdout",
         f"{secondary.grid} / {secondary.timeline} / {secondary.book} / {secondary.scope} / {secondary.fit}. "
         "Chosen after comparing the reported rows; it is not independent validation or an automatic production choice.", fig)
    summary = {"primary": main.replace({np.nan: None}).to_dict("records"),
               "secondary_ex_post_not_holdout": secondary.replace({np.nan: None}).to_dict(),
               "conditional_intervals": intervals, "not_a_pristine_holdout": True}
    save_json(output / "report_summary.json", summary)

    columns = [("timeline", "Information", "num"), ("scope", "Scope", "num"), ("method", "Rule", "num"),
        ("sharpe", "Net SR", "num"), ("stressed_sharpe", "2x-cost SR", "num"),
        ("independent_fee_sharpe", "Separate-fee SR", "num"),
        ("observed_total_return", "Observed return", "pct"), ("observed_max_drawdown", "Max DD", "pct"),
        ("coverage", "Coverage", "pct"), ("active_days", "Active days", "int"),
        ("priced_trade_coverage", "Priced trade days", "pct"), ("paired_net_sharpe", "Paired SR", "num")]
    headline = []
    for timeline in ("fresh_lagged", "older_settled"):
        part = main.loc[main.timeline.eq(timeline) & main.scope.eq("shared")].set_index("method")
        g, fixed, one = (part.loc[method] for method in ("greedy_three", "fixed_ten", "best_family"))
        weekly = post.loc[post.grid.eq("regional") & post.timeline.eq(timeline) & post.book.eq("five")
                          & post.scope.eq("shared") & post.fit.eq("252_weekly") & post.method.eq("greedy_three")].iloc[0]
        dropped = primary(concentration).loc[lambda f: f.timeline.eq(timeline) & f.scope.eq("shared")
                    & f.method.eq("greedy_three") & f.drop_best_days.eq(5)].iloc[0]
        headline.append(f"<li><b>{timeline.replace('_', ' ').title()}:</b> monthly diversified net SR {fmt(g.sharpe)} "
            f"versus {fmt(fixed.sharpe)} for fixed family budgets and {fmt(one.sharpe)} for one selected family. "
            f"Weekly diversified SR {fmt(weekly.sharpe)}; monthly SR with the best five days removed {fmt(dropped.sharpe)}.</li>")
    content = [f"<h1>{TITLE}</h1><p class='lead'>Periodic walk-forward selection across economic families, execution times and trade direction. Cached data only; live collection and the dashboard are untouched.</p>",
        f"<p>{manifest['start']} to {manifest['end']} | {manifest['sessions']} sessions | ten existing alpha families | four- and five-market books</p>",
        "<h2>Results in brief</h2><ul>"+"".join(headline)+"</ul><p>These are standalone intraday replays, not changes to the daily core's Sharpe or profits added on top of it. The robustness tables and all alternative fits below are retained regardless of performance.</p>",
        "<p><b>Interpretation:</b> periodic family selection has not produced a consistently stronger broad intraday book. Positive variants remain sensitive to cadence, friction, coverage and a few large days. The experiment does not justify replacing the interday core with this allocator.</p>",
        "<h2>What changed</h2><p>The prior experiment learned execution timing inside each family, then retained fixed one-tenth family budgets. This repeat also learns which families to retain at each refit. The economic alpha formulas and their parameters are unchanged. This is cross-family admission and allocation, not a new search over raw alpha parameters.</p>",
        "<p><b>Three rules:</b> the previous fixed ten-family budgets; a single strongest admitted family; and a nonnegative equal-weight basket, greedily adding at most two diversifying families when they improve the doubled-cost trailing Sharpe on matching dates. The last rule holds at most three families per selection pool; assetwise fitting may hold more than three across the whole portfolio.</p>",
        "<p><b>Periodic fitting:</b> 252-session, 504-session and expanding lookbacks, each refitted every five or 21 observed sessions. Training begins with 126 observations and excludes the immediately preceding completed session. Frozen 30:70, 40:60 and 50:50 splits are controls, not additional untouched data. With only 593 intraday sessions, the 504-session and expanding fits coincide for much of the sample; they are not independent confirmations.</p>",
        "<h2>Declared primary comparison</h2><p>Four regional blocks, five markets, 252-session monthly refits. Shared selection chooses families across the basket; assetwise selection fits each market independently. Both information timelines are shown rather than selecting one with hindsight. All metrics below exclude initial calibration cash; the chart still shows it.</p>",
        table(main, columns),
        "<p>Coverage includes genuine cash days. Priced trade days measures coverage only when the model planned a nonzero trade; a high cash allocation cannot disguise missing active outcomes. Paired SR compares the three rules on their common priced, post-calibration dates.</p>"]
    for name, title, caption in images:
        data = base64.b64encode((figures / (name+".png")).read_bytes()).decode()
        content.append(f'<figure><h2>{title}</h2><img alt="{title}" src="data:image/png;base64,{data}"><figcaption>{caption}</figcaption></figure>')
    content.extend([
        "<h2>Promising secondary row, with its weaknesses</h2><p>The strongest rolling result among the two new family-selection rules was identified after comparing this study's results. It is shown transparently, not presented as a preselected validation success.</p>",
        table(pd.DataFrame([secondary]), [("grid", "Grid", "num"), ("timeline", "Information", "num"),
              ("book", "Markets", "num"), ("scope", "Scope", "num"), ("fit", "Fit", "num"),
              ("method", "Rule", "num"), ("sharpe", "Net SR", "num"), ("stressed_sharpe", "2x-cost SR", "num"),
              ("observed_total_return", "Observed return", "pct"), ("priced_trade_coverage", "Priced trade days", "pct")]),
        table(secondary_stress, [("delay_minutes", "Delay", "int"), ("lag", "Extra signal age", "int"),
              ("sharpe", "Net SR", "num"), ("paired_sharpe", "Paired SR", "num"), ("paired_base_sharpe", "Paired base SR", "num")]),
        table(secondary_concentration, [("drop_best_days", "Best days removed", "int"),
              ("sharpe", "Net SR", "num"), ("observed_total_return", "Observed return", "pct")]),
        "<h2>Full window versus calibrated period</h2><p>Full window includes the actual initial no-trade period. Post-calibration starts on the first permissible fit. Common late window is the same calendar tail used to compare all split controls.</p>",
        table(m, [("timeline", "Information", "num"), ("period", "Period", "num"),
                  ("sharpe", "Net SR", "num"), ("observed_total_return", "Observed return", "pct"),
                  ("observed_max_drawdown", "Max DD", "pct"), ("coverage", "Coverage", "pct")]),
        "<h2>Delay, signal-age and concentration checks</h2><p>Original out-of-sample decisions are frozen before perturbing fills by five, 15 or 30 minutes, or making the signal one/two observed sessions older. They are not retuned to pass each stress. Removing the best days is a concentration diagnostic, not another strategy.</p>",
        table(primary(stress).loc[lambda f: f.method.eq("greedy_three")],
              [("timeline", "Information", "num"), ("scope", "Scope", "num"), ("delay_minutes", "Delay", "int"),
               ("lag", "Extra age", "int"), ("sharpe", "Net SR", "num"), ("paired_sharpe", "Paired SR", "num"),
               ("paired_base_sharpe", "Paired base", "num"), ("coverage", "Coverage", "pct")]),
        table(primary(concentration).loc[lambda f: f.method.eq("greedy_three") & f.scope.eq("shared")],
              [("timeline", "Information", "num"), ("drop_best_days", "Best days removed", "int"),
               ("sharpe", "Net SR", "num"), ("observed_total_return", "Observed return", "pct")]),
        "<h2>Asset and family attribution</h2><p>Each family receives its executed gross contribution and a pro-rata share of netted execution costs. Completely opposing orders in an identical window are not executed. All component statistics below use exactly the same complete dates as their portfolio; observed wealth-weighted P&amp;L contributions add to the portfolio's observed total return. Contribution Sharpes are not standalone tradable Sharpe ratios.</p>",
        table(primary(asset).loc[lambda f: f.method.eq("greedy_three")],
              [("timeline", "Information", "num"), ("scope", "Scope", "num"), ("root", "Market", "num"),
               ("sharpe", "Contribution SR", "num"), ("observed_pnl_contribution", "Observed P&L contribution", "pct"),
               ("active_days", "Active days", "int")]),
        table(primary(family).loc[lambda f: f.method.eq("greedy_three") & f.scope.eq("shared")],
              [("timeline", "Information", "num"), ("family", "Family", "num"), ("sharpe", "Contribution SR", "num"),
               ("observed_pnl_contribution", "Observed P&L contribution", "pct"),
               ("allocation_sessions", "Selected sessions", "int"), ("active_days", "Nonzero P&L days", "int")]),
        "<h2>Last simulated scheduled refit</h2><p>These are decisions already contained in the pinned replay, not refreshed live positions or a production recommendation. The five-market pool is the shared selector; individual symbols are independently fitted asset pools.</p>",
        table(recent, [("timeline", "Information", "num"), ("pool", "Pool", "num"),
              ("last_training_session", "Last training session", "num"), ("family", "Family", "num"),
              ("allocation", "Weight", "pct"), ("variant", "Signal class", "num"), ("window", "Window", "num")]),
        "<h2>All fits, not just the strongest row</h2>"])
    for grid in ("regional", "fixed_hold"):
        for timeline in ("fresh_lagged", "older_settled"):
            batch = post.loc[post.grid.eq(grid) & post.timeline.eq(timeline)].sort_values(["book", "scope", "fit", "method"])
            content.append(f"<details><summary>{grid} / {timeline}: rolling and all three frozen splits</summary>"+
                table(batch, [("book", "Markets", "num"), ("scope", "Scope", "num"), ("fit", "Fit", "num"),
                ("method", "Rule", "num"), ("sharpe", "Net SR", "num"), ("stressed_sharpe", "2x cost", "num"),
                ("observed_total_return", "Observed return", "pct"), ("coverage", "Coverage", "pct")])+"</details>")
    content.extend([
        "<h2>Implementation and limits</h2><p>All outcomes use the same pre-session liquid-nearby contract mask and cached bars as the prior study. No contract is selected because its later exit happens to be available. Fills are forward-only within ten minutes; matched exits retain delivery month and price instrument. Regional clocks follow their local DST calendars.</p>",
        "<p>Fresh means prior-close prices with previous-observed-session contract volume/OI. Older settled uses the forecast already held under the original daily lag. Neither assumes finalized current-session flows were published at CME reopen. These are information-policy proxies without historical publication-vintage verification.</p>",
        "<p>Fees are $2.16 per micro round trip, plus the original slippage allowances, with no standard-contract multiplier applied to commissions. Identical asset/recipe orders are netted. Different windows retain separate round trips, so adjacent-boundary cost savings are not claimed. Selection itself uses conservative unnetted doubled-cost family outcomes. Planned gross exposure is capped at one, with scaling down only; silver's absence does not cause an automatic leverage increase.</p>",
        "<p>The ten-family roster and economic parameter regions were identified in earlier hindsight-informed research. Refit decisions here are past-only, but this repeat is not a pristine external holdout. Broader recipe grids and the many reported configurations create further selection risk: do not sell the highest row as independent validation. Thin SIL coverage, historically inferred contract months and the lack of quote-depth execution evidence remain material limitations. Observed-return curves compound only priced sessions; missing active outcomes are gaps, not cash.</p>",
        f"<p><b>Checks:</b> {manifest['baseline_reconciliation_checks']} exact prior-baseline reconciliations; {manifest['attribution_checks']} additive portfolio checks; {manifest['prefix_invariance_checks']} actual-data future-mutation checks; {manifest['scalar_checks']} independent micro-lot P&amp;L/fee comparisons; frozen source hashes unchanged.</p>",
        "<h2>Reproducible files</h2><p>"+" | ".join(f'<a href="{name}">{label}</a>' for name, label in [
            ("metrics.csv", "All metrics"), ("paths.parquet", "All daily paths"), ("refit_ledger.parquet", "Refit decisions"),
            ("stress.csv", "Frozen-decision stress"), ("yearly.csv", "Yearly results"), ("direction_controls.csv", "Direction controls"),
            ("attribution_common_dates.csv", "Common-date asset/family attribution"),
            ("schedules.npz", "Exact schedules"), ("manifest.json", "Audit manifest")])+"</p>"])
    css = """body{max-width:1180px;margin:40px auto;padding:0 25px;background:#faf8f2;color:#203a35;font:15px/1.6 Georgia,serif}h1{font-size:36px;line-height:1.15;max-width:900px}h2{font-size:24px;margin-top:32px}.lead{font-size:19px;max-width:1000px}img{width:100%;height:auto}figure{margin:28px 0;break-inside:avoid}figcaption{font-size:13px;color:#53665e}.table{overflow:auto}table{border-collapse:collapse;width:100%;font:12px/1.4 'DejaVu Sans',sans-serif}th{text-align:left;background:#e7ece4;white-space:nowrap}th,td{padding:8px;border-bottom:1px solid #d9dfd3}details{margin:18px 0}summary{cursor:pointer;font-weight:bold}a{color:#176b5c}@media print{body{margin:0;max-width:none;padding:0}details{display:block}h2{break-after:avoid}.table{overflow:visible}figure{page-break-inside:avoid}}"""
    document = '<!doctype html><html><head><meta charset="utf-8"><title>'+TITLE+'</title><style>'+css+'</style></head><body>'+"\n".join(content)+"</body></html>"
    report_path = output / "Five_Currents_Family_Walkforward_Report.html"
    report_path.write_text(document, encoding="utf-8")
    save_json(output / "report_verification.json", {
        "embedded_pngs": len(images), "nonempty_paths": len(groups),
        "source_sha256_matches": all(digest(source/name) == checksum for name, checksum in manifest["source_sha256"].items()),
        "report_sha256": digest(report_path), "report_bytes": report_path.stat().st_size,
        "code_sha256": {name: digest(Path(__file__).parent/name) for name in
                         ("five_currents_family_timing.py", "run_five_currents_family_timing.py", "report_five_currents_family_timing.py")}})
    print(report_path, flush=True)
    print(main[["timeline", "scope", "method", "sharpe", "stressed_sharpe", "coverage"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    report(parser.parse_args().output.resolve())
