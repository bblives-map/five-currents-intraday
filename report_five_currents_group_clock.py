"""Summarize the offline family-by-clock experiment without promoting grid winners."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from five_currents_group_clock_core import OUTPUT, GROUPS, CLOCKS, HOLD_MINUTES, row_stats
from five_currents_execution_timing import ROOTS


LABEL = {"CL_NG": "Energy | CL and NG", "GC_SI": "Precious metals | GC and SI",
         "HG": "Copper | HG"}
TIMELINE = {"fresh_lagged": "Earlier price + lagged flows",
            "older_settled": "Older settled vote"}
COLORS = {"CL_NG": "#bc6f35", "GC_SI": "#b39841", "HG": "#216d67"}


def val(x, style="number"):
    if pd.isna(x):
        return "n/a"
    if style == "pct":
        return f"{100*x:.1f}%"
    if style == "int":
        return f"{int(x):,}"
    return f"{x:.2f}"


def table(frame, columns):
    head = "".join(f"<th>{html.escape(label)}</th>" for key, label, style in columns)
    body = "".join("<tr>" + "".join(
        f"<td>{html.escape(str(row[key])) if style == 'text' else val(row[key], style)}</td>"
        for key, label, style in columns) + "</tr>" for row in frame.to_dict("records"))
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def plot_equity(frame, ax, label, color, *, ls="-"):
    frame = frame.sort_values("session")
    equity = (1+frame.net).cumprod()*100
    ax.plot(frame.session, equity, label=label, color=color, lw=1.6, ls=ls)
    ax.grid(axis="y", alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)
    loc = mdates.MonthLocator(bymonth=(1, 7))
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def path_image(paths, output):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.3), sharex=True)
    for row, group in enumerate(GROUPS):
        for col, timeline in enumerate(TIMELINE):
            ax = axes[row, col]
            view = paths.loc[paths.group.eq(group) & paths.timeline.eq(timeline)
                             & paths.cut.eq("cut30") & paths.fit.eq("252_monthly")]
            for count, pattern in ((1, "-"), (2, "--")):
                item = view.loc[view.class_count.eq(count)]
                if len(item):
                    plot_equity(item, ax, f"{count} class", COLORS[group], ls=pattern)
            if len(view):
                cal = view.loc[view.calibrating, "session"]
                if len(cal):
                    ax.axvspan(view.session.min(), cal.max(), color="#9dada6", alpha=.13)
            ax.set_title(f"{LABEL[group]} | {TIMELINE[timeline]}", loc="left", fontsize=10, weight="bold")
            ax.set_ylabel("Growth of 100")
            ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = output / "figures" / "forward_paths.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fit_image(metrics, output):
    view = metrics.loc[metrics.period.eq("post_calibration") & metrics.cut.eq("cut30")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    order = [f"{group} / {timeline}" for group in GROUPS for timeline in TIMELINE]
    for ax, count in zip(axes, (1, 2)):
        part = view.loc[view.class_count.eq(count)].copy()
        part["row"] = part.group+" / "+part.timeline
        matrix = part.pivot_table(index="row", columns="fit", values="sharpe", aggfunc="first")
        matrix = matrix.reindex(order).reindex(columns=("252_weekly", "252_monthly", "504_weekly",
            "504_monthly", "expanding_weekly", "expanding_monthly"))
        data = matrix.to_numpy()
        im = ax.imshow(np.ma.masked_invalid(data), cmap="RdBu", vmin=-2, vmax=2, aspect="auto")
        ax.set_yticks(range(len(order)), [x.replace("fresh_lagged", "fresh").replace("older_settled", "settled") for x in order])
        ax.set_xticks(range(len(matrix.columns)), [x.replace("_", "\n") for x in matrix.columns], fontsize=7)
        ax.set_title(f"{count} class{'es' if count == 2 else ''}: net OOS Sharpe")
        for i, j in np.ndindex(data.shape):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(data[i,j]) > 1.1 else "#20332e")
        ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=axes, label="Net Sharpe", shrink=.82)
    path = output / "figures" / "fit_stability.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def asset_image(paths, output):
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.2), sharex=True)
    palette = {"CL": "#b96530", "NG": "#755d9d", "GC": "#a88d30", "SI": "#687d97", "HG": "#216d67"}
    for row, (group, indices) in enumerate(GROUPS.items()):
        for col, timeline in enumerate(TIMELINE):
            ax = axes[row, col]
            part = paths.loc[paths.group.eq(group) & paths.timeline.eq(timeline)
                             & paths.cut.eq("cut30") & paths.fit.eq("252_monthly")
                             & paths.class_count.eq(2)].sort_values("session")
            if len(part):
                priced = np.isfinite(part.net.to_numpy()) & ~part.calibrating.to_numpy()
                for root in [ROOTS[i] for i in indices]:
                    component = part[root].where(priced)
                    ax.plot(part.session, component.cumsum()*100, label=root, color=palette[root], lw=1.4)
                ax.plot(part.session, part.net.where(priced).cumsum()*100,
                        label="Group", color="#263b36", lw=1.5, ls="--")
            ax.set_title(f"{LABEL[group]} | {TIMELINE[timeline]}", loc="left", fontsize=10, weight="bold")
            ax.set_ylabel("Cumulative return (pp)")
            ax.grid(axis="y", alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="upper left", fontsize=8)
            loc = mdates.MonthLocator(bymonth=(1, 7))
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    path = output / "figures" / "asset_contributions.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def coverage_image(output):
    with np.load(output / "clock_prices.npz") as saved:
        prices = saved["returns"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), sharex=True, sharey=True)
    for ax, (group, indices) in zip(axes, GROUPS.items()):
        complete = np.isfinite(prices[:, :, 0, :, list(indices)]).all(axis=-1)
        # Advanced indexing places the asset axis first on some NumPy versions.
        if complete.shape[:2] != (len(CLOCKS), len(HOLD_MINUTES)):
            complete = np.isfinite(prices[:, :, 0][:, :, :, list(indices)]).all(axis=-1)
        coverage = complete.mean(axis=2)
        im = ax.imshow(coverage, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(HOLD_MINUTES)), [str(v) for v in HOLD_MINUTES], fontsize=8)
        ax.set_yticks(range(len(CLOCKS)), [x.replace("_", " ") for x in CLOCKS], fontsize=8)
        ax.set_title(LABEL[group], fontsize=10)
        ax.set_xlabel("Holding minutes")
        for i, j in np.ndindex(coverage.shape):
            ax.text(j, i, f"{coverage[i,j]*100:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if coverage[i,j]>.65 else "#25312e")
        ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=axes, label="Matched day share", shrink=.8)
    path = output / "figures" / "clock_coverage.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def family_clock_table(grid, group, timeline):
    part = grid.loc[grid.group.eq(group) & grid.timeline.eq(timeline) & grid.period.eq("post_30_pct")].copy()
    part = part.loc[part.priced_trade_coverage.ge(.55) & part.active_days.ge(15) & part.days.ge(60)]
    if part.empty:
        return "<p>No sufficiently covered family/clock cells.</p>"
    top = part.sort_values("stressed_sharpe", ascending=False).drop_duplicates(["family", "clock"])
    rows = []
    for family, chunk in top.groupby("family", sort=True):
        record = {"family": family.replace("_", " ")}
        for clock in CLOCKS:
            item = chunk.loc[chunk.clock.eq(clock)]
            if len(item):
                best = item.iloc[0]
                record[clock] = f"{best.sharpe:+.2f} | {int(best.hold_minutes)}m | {best.signal_class.replace('__', ' / ')}"
            else:
                record[clock] = "insufficient coverage"
        rows.append(record)
    return table(pd.DataFrame(rows), [("family", "Family", "text")]
        + [(clock, clock.replace("_", " ").title(), "text") for clock in CLOCKS])


def individual_asset_tables(grid):
    sections = []
    for root in ROOTS:
        for timeline in TIMELINE:
            part = grid.loc[grid.group.eq(root) & grid.timeline.eq(timeline)
                            & grid.period.eq("post_30_pct")]
            part = part.loc[part.priced_trade_coverage.ge(.55) & part.active_days.ge(15)
                            & part.days.ge(60)]
            top = part.sort_values("stressed_sharpe", ascending=False).drop_duplicates("family")
            rows = []
            for family in sorted(grid.family.unique()):
                match = top.loc[top.family.eq(family)]
                if len(match):
                    best = match.iloc[0]
                    rows.append({"family": family.replace("_", " "),
                                 "clock": best.clock.replace("_", " "),
                                 "hold": int(best.hold_minutes),
                                 "transform": best.signal_class.replace("__", " / "),
                                 "sr": best.sharpe,
                                 "coverage": best.priced_trade_coverage})
                else:
                    rows.append({"family": family.replace("_", " "),
                                 "clock": "insufficient coverage", "hold": np.nan,
                                 "transform": "n/a", "sr": np.nan, "coverage": np.nan})
            label = f"{root} | {TIMELINE[timeline]}"
            sections.append(f"<details><summary>{html.escape(label)}</summary>"
                + table(pd.DataFrame(rows), [("family", "Family", "text"),
                    ("clock", "Best clock", "text"), ("hold", "Hold (min)", "int"),
                    ("transform", "Vote transform", "text"), ("sr", "Net SR", "number"),
                    ("coverage", "Trade coverage", "pct")]) + "</details>")
    return "".join(sections)


def concentration_table(paths):
    part = paths.loc[paths.cut.eq("cut30") & paths.fit.eq("252_monthly")
                     & paths.class_count.eq(2) & ~paths.calibrating].copy()
    part["session"] = pd.to_datetime(part.session)
    rows = []
    for (group, timeline), frame in part.groupby(["group", "timeline"]):
        valid = frame.loc[frame.net.notna()].copy()
        if valid.empty:
            continue
        best_days = valid.nlargest(3, "net")
        without_best = valid.drop(best_days.index)
        monthly = valid.groupby(valid.session.dt.to_period("M")).net.sum().sort_values(ascending=False)
        best_month = str(monthly.index[0])
        without_month = valid.loc[valid.session.dt.to_period("M").astype(str).ne(best_month)]
        total = float(valid.net.sum())
        rows.append({"market": LABEL[group], "information": TIMELINE[timeline],
                     "best_month": best_month,
                     "best_month_share": float(monthly.iloc[0]/total) if total > 0 else np.nan,
                     "top3_share": float(best_days.net.sum()/total) if total > 0 else np.nan,
                     "without_top3_sharpe": row_stats(without_best.net.to_numpy())["sharpe"],
                     "without_best_month_sharpe": row_stats(without_month.net.to_numpy())["sharpe"]})
    return pd.DataFrame(rows)


def report(output=OUTPUT):
    output = Path(output)
    output.joinpath("figures").mkdir(exist_ok=True)
    manifest = json.loads((output / "manifest.json").read_text())
    paths = pd.read_parquet(output / "paths.parquet")
    paths["session"] = pd.to_datetime(paths.session)
    metrics = pd.read_csv(output / "metrics.csv")
    attribution = pd.read_csv(output / "asset_attribution.csv")
    yearly = pd.read_csv(output / "yearly.csv")
    selection = pd.read_csv(output / "class_selection.csv")
    ledger = pd.read_parquet(output / "refit_ledger.parquet")
    stress = pd.read_csv(output / "stress.csv")
    grid = pd.read_parquet(output / "full_family_clock_grid.parquet")
    asset_grid_path = output / "asset_family_clock_grid.parquet"
    asset_grid = pd.read_parquet(asset_grid_path) if asset_grid_path.exists() else None
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "figure.facecolor": "#faf8f1", "axes.facecolor": "#faf8f1", "text.color": "#263b36"})
    path_image(paths, output)
    fit_image(metrics, output)
    asset_image(paths, output)
    coverage_image(output)
    post = metrics.loc[metrics.period.eq("post_calibration") & metrics.cut.eq("cut30")
                   & metrics.fit.eq("252_monthly")].copy()
    post["market"] = post.group.map(LABEL)
    post["information"] = post.timeline.map(TIMELINE)
    post = post.sort_values(["group", "timeline", "class_count"])
    cols = [("market", "Market", "text"), ("information", "Signal information", "text"),
            ("class_count", "Classes", "int"), ("days", "Observed days incl. cash", "int"),
            ("priced_trade_coverage", "Trade coverage", "pct"), ("sharpe", "Net SR", "number"),
            ("stressed_sharpe", "2x-cost SR", "number"),
            ("observed_total_return", "Observed return", "pct"),
            ("observed_max_drawdown", "Max DD", "pct")]
    attribution = attribution.loc[attribution.cut.eq("cut30") & attribution.fit.eq("252_monthly")
                                  & attribution.class_count.eq(2)].copy()
    attribution["market"] = attribution.group.map(LABEL)
    attribution["information"] = attribution.timeline.map(TIMELINE)
    attribution = attribution.sort_values(["group", "timeline", "root"])
    concentration = concentration_table(paths)
    concentration.to_csv(output / "concentration.csv", index=False)
    yearly = yearly.loc[yearly.cut.eq("cut30") & yearly.fit.eq("252_monthly")
                         & yearly.class_count.eq(2)].copy()
    yearly["market"] = yearly.group.map(LABEL)
    yearly["information"] = yearly.timeline.map(TIMELINE)
    yearly = yearly.sort_values(["group", "timeline", "year"])
    sections = []
    for group in GROUPS:
        head = f"<h2>{html.escape(LABEL[group])}</h2>"
        sc = selection.loc[selection.group.eq(group) & selection.cut_fraction.eq(.3)]
        parts = []
        for timeline in TIMELINE:
            chosen = sc.loc[sc.timeline.eq(timeline)]
            text = ", ".join(f"{r.signal_class.replace('__', ' / ')} (supported by {r.supporting_families} families)"
                             for r in chosen.itertuples()) or "No class passed the early-sample rule"
            base = ledger.loc[ledger.group.eq(group) & ledger.timeline.eq(timeline)
                              & ledger.cut.eq("cut30") & ledger.fit.eq("252_monthly")
                              & ledger.class_count.eq(2) & ledger.family.notna()].copy()
            if len(base):
                base["sessions"] = base.end-base.start
                leaders = base.groupby(["family", "clock", "hold_minutes"], observed=True).sessions.sum()
                leaders = leaders.sort_values(ascending=False).head(5)
                summary = "; ".join(f"{f.replace('_',' ')} at {c.replace('_',' ')} for {int(h)}m ({int(days)} refit sessions)"
                                    for (f,c,h), days in leaders.items())
            else:
                summary = "No eligible live recipe"
            parts.append(f"<h3>{html.escape(TIMELINE[timeline])}</h3>"
                         f"<p><strong>Early class selection:</strong> {html.escape(text)}. "
                         f"<strong>Most-used forward recipes:</strong> {html.escape(summary)}.</p>"
                         + family_clock_table(grid, group, timeline))
        sections.append(head+"".join(parts))
    controls = metrics.loc[metrics.period.eq("common_late_window") & metrics.fit.eq("252_monthly")
                           & metrics.class_count.eq(2)].copy()
    controls["market"] = controls.group.map(LABEL)
    controls["information"] = controls.timeline.map(TIMELINE)
    controls = controls.sort_values(["group", "timeline", "cut"])
    paired = stress.loc[stress.cut.eq("cut30") & stress.fit.eq("252_monthly")
                        & stress.class_count.eq(2)].copy()
    paired["market"] = paired.group.map(LABEL)
    paired["information"] = paired.timeline.map(TIMELINE)
    paired["stress"] = np.where(paired.extra_signal_sessions.eq(1), "one more session of signal age",
                                "+"+paired.delay_minutes.astype(str)+" minute fill")
    pairing_cols = [("market", "Market", "text"), ("information", "Signal information", "text"),
                    ("stress", "Perturbation", "text"), ("paired_days", "Paired days", "int"),
                    ("paired_sharpe", "Perturbed SR", "number"),
                    ("paired_base_sharpe", "Base on same days", "number")]
    report_path = output / "Five_Currents_Grouped_Intraday_Clocks.html"
    css = """
    :root{--ink:#243a35;--teal:#216d67;--cream:#faf8f1;--border:#dcded6}
    *{box-sizing:border-box}body{margin:0;background:#e8e9e1;color:var(--ink);font:15px/1.6 'Trebuchet MS',sans-serif}
    main{max-width:1160px;margin:30px auto;background:var(--cream);padding:40px 46px;box-shadow:0 9px 40px #1f34301a}
    header{border-bottom:4px solid var(--teal);padding-bottom:18px}h1,h2,h3{font-family:Georgia,serif;line-height:1.2}
    h1{font-size:37px;margin:8px 0}h2{font-size:27px;margin:40px 0 13px;border-top:1px solid var(--border);padding-top:24px}
    h3{font-size:19px;margin:30px 0 9px}p{max-width:90ch}.eyebrow{text-transform:uppercase;letter-spacing:.14em;color:var(--teal);font-size:11px;font-weight:bold}
    .lead{font-size:18px}.note{border-left:3px solid var(--teal);background:#eaf0e9;padding:11px 16px;margin:18px 0}
    .figure{margin:24px 0}.figure img{display:block;width:100%;height:auto;border:1px solid var(--border);border-radius:7px}
    figcaption{color:#62706a;font-size:12px;margin-top:7px}.table-wrap{overflow-x:auto;margin:14px 0 25px}
    table{border-collapse:collapse;width:100%;font-size:12px;min-width:720px}th{text-align:left;background:#e6eee8;color:#24473f;white-space:nowrap}
    th,td{padding:8px 10px;border-bottom:1px solid var(--border);vertical-align:top}tr:hover td{background:#eef2ec}
    .grid table{min-width:1040px}.grid td:not(:first-child){font-size:10px;min-width:130px}
    details{border:1px solid var(--border);border-radius:6px;margin:8px 0;padding:0 12px}summary{cursor:pointer;padding:9px 3px;font-weight:600;color:var(--teal)}
    a{color:#126a61}small{color:#687670}footer{border-top:1px solid var(--border);margin-top:42px;padding-top:15px}
    @media(max-width:650px){main{padding:22px;margin:0}h1{font-size:30px}}
    """
    page = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>Five Currents | Regional Clock and Family Test</title><style>{css}</style></head><body><main>
    <header><div class='eyebrow'>Offline research | {manifest['start']} to {manifest['end']}</div>
    <h1>Where the daily vote meets the clock</h1><p class='lead'>Ten daily signal families tested independently on CL/NG, GC/SI, and HG across the Asia, London, and New York opens and closes.</p></header>
    <div class='note'><strong>Reading rule:</strong> the family-by-clock matrices below are descriptive hindsight screens across 160 transformations and holds per cell. Only the refit paths and controls are causal forward selection; neither validates the original daily alpha roster independently.</div>
    <h2>Forward-selected portfolios</h2><p>Each market group chooses one or two signal classes from its early calibration sample, then selects family, clock, and holding period using prior outcomes only. The chart spans the full available minute-bar period; shaded history is calibration, not a claimed trading record.</p>
    <figure class='figure'><img src='figures/forward_paths.png' alt='Grouped forward equity paths'><figcaption>Growth of 100 from available intraday data. Missing active trades are left as breaks; no missing-price return is imputed. Solid line uses one class, dashed uses two. These are not combined with the daily close-to-close sleeve.</figcaption></figure>
    {table(post, cols)}
    <div class='note'><strong>Interpretation:</strong> none of these grouped intraday timing paths is ready to promote as a trading sleeve. GC/SI has the strongest headline fit but sparse silver coverage and concentrated gains; the positive older-settled CL/NG and HG paths are modest and sensitive to a few sessions or a different calibration cut. The fresh-lagged HG path loses money.</div>
    <h3>How concentrated are the positive outcomes?</h3>
    <p>The table deliberately removes the three best observed days or the best calendar month <em>after seeing the outcomes</em>. This is a stress diagnostic, not an executable selection rule. Shares above 100% mean the remaining days lost money overall.</p>
    {table(concentration, [('market','Market','text'),('information','Signal information','text'),('best_month','Best month','text'),('top3_share','Top 3 / net P&L','pct'),('best_month_share','Best month / net P&L','pct'),('without_top3_sharpe','SR excluding top 3','number'),('without_best_month_sharpe','SR excluding best month','number')])}
    <h3>Year-by-year, two-class monthly path</h3>
    {table(yearly, [('market','Market','text'),('information','Signal information','text'),('year','Year','int'),('days','Observed days incl. cash','int'),('sharpe','Net SR','number'),('observed_total_return','Observed return','pct')])}
    <h3>Which contract contributes?</h3>
    <figure class='figure'><img src='figures/asset_contributions.png' alt='Per-contract return contributions'><figcaption>Additive return contributions on the same observed post-calibration days as each group's two-class portfolio. A gap is missing price coverage, not flat P&amp;L.</figcaption></figure>
    {table(attribution, [('market','Market','text'),('information','Signal information','text'),('root','Contract','text'),('days','Observed days incl. cash','int'),('sharpe','Standalone contribution SR','number'),('observed_pnl_contribution','Compounded portfolio contribution','pct')])}
    <h2>Selection stability</h2><p>Weekly means five observed sessions; monthly means 21. Fits use 252 sessions, 504 sessions, or an expanding history, always with a one-session training embargo. The early class choice is held fixed within each calibration-cut scenario.</p>
    <figure class='figure'><img src='figures/fit_stability.png' alt='Refit stability matrix'><figcaption>Net Sharpe after the initial 30% class calibration. White cells mean a path was not available, not zero return.</figcaption></figure>
    <h3>Calibration-cut controls on the same late window</h3>
    {table(controls, [('market','Market','text'),('information','Signal information','text'),('cut','Early cut','text'),('days','Priced days','int'),('sharpe','Net SR','number'),('stressed_sharpe','2x-cost SR','number'),('observed_max_drawdown','Max DD','pct')])}
    <h2>Price coverage and timing</h2><p>A matched trade needs an observed active bar after entry and exit on the same listed contract, within ten minutes of each planned time, and inside the exchange session. The six clocks are 09:00 and 17:00 Singapore; 08:00 and 16:30 London; 09:30 and 16:00 New York, with local daylight saving applied.</p>
    <figure class='figure'><img src='figures/clock_coverage.png' alt='Same-contract fill coverage'><figcaption>Share of 593 sessions with a matched same-contract price for every asset in a group, without requiring a nonzero signal. SI is the main thin-history constraint. Longer New York-close holds often leave the exchange session.</figcaption></figure>
    <h3>Fill delay and older signal test</h3>
    {table(paired, pairing_cols)}
    <h2>Family-by-clock screens</h2><p>Each cell shows the highest <em>post-30% descriptive</em> net Sharpe, hold in minutes, and vote transform among the 160 tested for that family and clock, subject to basic price coverage. This is intentionally <strong>not</strong> a deployable forecast. The forward tables above decide whether any pattern survived re-selection.</p>
    <div class='grid'>{''.join(sections)}</div>
    <h2>Single-contract screens</h2><p>The grouped books above are the decision unit. For transparency, the same descriptive test is also run separately for every family in each of CL, NG, GC, SI, and HG. Open a contract to see its most favorable sufficiently covered clock, transform, and hold. These maxima were chosen with hindsight; the <a href='asset_family_clock_grid.parquet'>complete contract-level rule grid</a> contains all tested cells.</p>
    {individual_asset_tables(asset_grid) if asset_grid is not None else '<p>Contract-level grid not yet generated.</p>'}
    <h2>Method and caveats</h2><p>Gross means the raw daily family vote; net means that family's contribution after the pinned daily allocator. Changes are first differences, or a second difference for net-change-of-change. Every base is tested in magnitude, sign-only, reversed magnitude, and reversed sign form. The early selection is based on the second-strongest qualifying family when at least two qualify, not the single best isolated backtest. Refit recipes use the median doubled-cost Sharpe over 0, +5, and +15-minute fills and require positive outcomes in both halves of their training window.</p>
    <p>One round trip is charged at $2.16 per micro contract plus the source slippage allowance; no x10 commission scaling. Source daily signals, contract choices, minute prices, and prior daily allocator are pinned from the earlier replay. No new data were requested from IBKR, Databento, CME, Bloomberg, or Postgres. Results are research, not proof of executable depth or a clean untouched holdout: the original family roster was influenced by earlier research, and historical data-publication vintages are unavailable.</p>
    <footer><strong>Reproducibility:</strong> <a href='full_family_clock_grid.parquet'>full rule grid</a> | <a href='metrics.csv'>portfolio metrics</a> | <a href='refit_ledger.parquet'>refit ledger</a> | <a href='paths.parquet'>paths</a> | <a href='manifest.json'>manifest and limitations</a>.</footer>
    </main></body></html>"""
    report_path.write_text(page, encoding="utf-8")
    return report_path


if __name__ == "__main__":
    print(report())
