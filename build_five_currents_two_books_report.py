"""Offline, hindsight-labelled execution-book comparison and observed-only curves."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from five_currents_execution_timing import ROOTS
from five_currents_group_clock_core import (
    CLOCKS, HOLD_MINUTES, DELAYS, SOURCE, candidate_pnl, row_stats, weights_for_class,
)
from run_five_currents_group_clock import bundle


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
OUT = ROOT / "five_currents_two_books_20261024"
AUDIT = ROOT / "five_currents_gc4_pure_alpha_20261024"
PRICE_SOURCE = ROOT / "five_currents_group_clock_20261024" / "clock_prices.npz"
DATA_ONLY = False
BOOKS = {
    "two_books_carry_oi": {"CL": "carry_oi_reversal", "NG": "carry_oi_reversal",
                           "GC": "oi_direction_reversion", "HG": "curve_quality_reversal"},
    "energy_spike": {"CL": "volume_spike_reversion", "NG": "volume_spike_reversion"},
    "metals_spike_curve": {"GC": "volume_spike_reversion", "HG": "curve_quality_reversal"},
    "two_books_equal": {"CL": "volume_spike_reversion", "NG": "volume_spike_reversion",
                        "GC": "volume_spike_reversion", "HG": "curve_quality_reversal"},
    "energy_carry_oi": {"CL": "carry_oi_reversal", "NG": "carry_oi_reversal"},
    "four_spike": {root: "volume_spike_reversion" for root in ("CL", "NG", "GC", "HG")},
    "four_curve": {root: "curve_quality_reversal" for root in ("CL", "NG", "GC", "HG")},
    "metals_volume_curve": {"GC": "aggregate_volume_carry", "HG": "curve_quality_reversal"},
    "metals_oi_curve": {"GC": "oi_direction_reversion", "HG": "curve_quality_reversal"},
    "metals_curve": {"GC": "curve_quality_reversal", "HG": "curve_quality_reversal"},
}
CARRY_MOMENTUM_FAMILIES = (
    "carry_oi_reversal", "carry_basis_reversal", "aggregate_volume_carry",
    "curve_quality_reversal", "carry_confirmed_momentum",
    "volume_dryup_continuation", "volume_spike_continuation",
)
GROUP_LABELS = {"energy": "Energy: CL + NG", "gold": "Gold: GC",
                "copper": "Copper: HG", "precious": "Precious: GC + SI",
                "overall_four": "Four assets: CL + NG + GC + HG"}


def sharpe(values):
    return row_stats(values)["sharpe"]


def drop_five(values):
    copy = np.asarray(values, float).copy()
    positive = np.flatnonzero(np.isfinite(copy) & (copy > 0))
    if len(positive):
        copy[positive[np.argsort(copy[positive])[-5:]]] = np.nan
    return sharpe(copy)


def metrics(gross, cost, planned, sessions, name):
    net = gross - cost
    matched = planned & np.isfinite(net)
    observed = row_stats(net)
    result = {"book": name, "net_sr": observed["sharpe"],
              "after_top5_sr": drop_five(net), "stress_sr": sharpe(gross-2*cost),
              "observed_growth": observed["observed_total_return"],
              "observed_max_dd": observed["observed_max_drawdown"],
              "annual_vol": observed["annual_vol"],
              "planned_days": int(planned.sum()), "matched_days": int(matched.sum()),
              "priced_trade_coverage": float(matched.sum()/planned.sum()) if planned.any() else np.nan}
    result.update({f"sr_{year}": sharpe(net[sessions.year == year])
                   for year in (2024, 2025, 2026)})
    return result


def leg_path(row, root, family, raw, dates, sessions, prices, names):
    weights = weights_for_class(raw, raw, dates, sessions, lag=0,
                                family=names.index(family), signal_class=row.signal_class)
    ai = ROOTS.index(root)
    pnl = candidate_pnl(weights, prices, (ai,))
    cell = CLOCKS.index(row.clock)*len(HOLD_MINUTES)+HOLD_MINUTES.index(row.hold_minutes)
    return pnl["gross"][cell], pnl["cost"][cell], np.abs(weights[:, ai]) > 1e-12


def figure(book_paths, sessions, names):
    fig, axes = plt.subplots(len(names), 2, figsize=(12.4, 3.2*len(names)),
                             gridspec_kw={"width_ratios": [4.5, 1]}, squeeze=False)
    for i, name in enumerate(names):
        path = book_paths[name]
        net = path["gross"][0]-path["cost"][0]
        stressed = path["gross"][0]-2*path["cost"][0]
        valid = np.isfinite(net)
        equity = np.cumprod(1+np.where(valid, net, 0.))
        stress_equity = np.cumprod(1+np.where(valid, stressed, 0.))
        # Break the drawn line at unpriced active trades; no missing return is asserted to be zero.
        equity[~valid] = np.nan
        stress_equity[~valid] = np.nan
        ax = axes[i, 0]
        ax.plot(sessions, equity, color="#0d625e", lw=1.65, label="Net, priced days")
        ax.plot(sessions, stress_equity, color="#c17836", lw=1.1, ls="--",
                label="Doubled costs, priced days")
        ax.axhline(1., color="#9ca9a2", lw=.6)
        ax.set_title(name.replace("_", " ").title(), loc="left", fontweight="bold", fontsize=11)
        ax.set_ylabel("Growth of 1, observed-only")
        ax.grid(alpha=.15)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        state = np.where(~path["planned"], 0, np.where(valid, 1, 2))
        ax2 = axes[i, 1]
        counts = [(state == value).sum() for value in (1, 2, 0)]
        ax2.barh(["Priced", "Unpriced", "No position"], counts,
                 color=["#0d625e", "#b75b46", "#b8c0b5"])
        ax2.set_xlim(0, len(sessions))
        for j, value in enumerate(counts):
            ax2.text(value+4, j, str(value), va="center", fontsize=7)
        ax2.set_title("Session coverage", fontsize=9)
        ax2.tick_params(axis="both", labelsize=7)
        ax2.invert_yaxis()
    fig.suptitle("Observed-trade equity, not a complete investable history", y=1.005,
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def asset_figure(leg_paths, books, sessions):
    fig, axes = plt.subplots(len(books), 1, figsize=(12.4, 3.3*len(books)), squeeze=False)
    colors = {"CL": "#a85d2a", "NG": "#568460", "GC": "#b59942", "HG": "#3d6891"}
    for i, book in enumerate(books):
        ax = axes[i, 0]
        for root, family in BOOKS[book].items():
            gross, cost, _ = leg_paths[(root, family)]
            net = gross[0]-cost[0]
            valid = np.isfinite(net)
            equity = np.cumprod(1+np.where(valid, net, 0.))
            equity[~valid] = np.nan
            ax.plot(sessions, equity, lw=1.25, color=colors[root], label=f"{root}: {family}")
        ax.axhline(1., color="#9ca9a2", lw=.6)
        ax.set_title(book.replace("_", " ").title() + " | component legs", loc="left", fontsize=11)
        ax.set_ylabel("Growth of 1, observed-only")
        ax.grid(alpha=.15)
        ax.legend(frameon=False, ncol=2, fontsize=7)
    fig.tight_layout()
    return fig


def family_comparison_figure(frame):
    groups = tuple(GROUP_LABELS)
    values = np.full((len(groups), len(CARRY_MOMENTUM_FAMILIES)), np.nan)
    coverages = np.full_like(values, np.nan)
    for row in frame.itertuples(index=False):
        i = groups.index(row.group)
        j = CARRY_MOMENTUM_FAMILIES.index(row.family)
        values[i, j] = row.net_sr
        coverages[i, j] = row.priced_trade_coverage * 100
    fig, ax = plt.subplots(figsize=(13.4, 5.2))
    colors = plt.get_cmap("RdYlGn").copy()
    colors.set_bad("#e4e1d8")
    im = ax.imshow(np.ma.masked_invalid(values), cmap=colors, vmin=-0.5, vmax=3.0,
                   aspect="auto")
    labels = [name.replace("_", " ").title() for name in CARRY_MOMENTUM_FAMILIES]
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(groups)), [GROUP_LABELS[group] for group in groups], fontsize=9)
    for i in range(len(groups)):
        for j in range(len(labels)):
            note = (f"{values[i, j]:.2f} SR\n{coverages[i, j]:.0f}% priced"
                    if np.isfinite(values[i, j]) else "No eligible rule")
            color = "white" if np.isfinite(values[i, j]) and values[i, j] >= 2.6 else "#17312e"
            ax.text(j, i, note, ha="center", va="center", fontsize=8, color=color)
    ax.set_title("Carry / curve and momentum: net Sharpe with priced-trade coverage",
                 loc="left", fontweight="bold", fontsize=12)
    fig.colorbar(im, ax=ax, label="Observed-day net Sharpe", shrink=.83, pad=.015)
    fig.tight_layout()
    return fig


def png(fig):
    stream = BytesIO()
    fig.savefig(stream, format="png", dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stream.getvalue()


def table(frame, columns):
    labels = {"book": "Book", "root": "Asset", "family": "Alpha family",
              "signal_class": "Vote treatment", "clock": "Entry clock",
              "hold_minutes": "Hold (min)", "net_sr": "Net SR",
              "after_top5_sr": "SR less five wins", "stress_sr": "2x-cost SR",
              "delay15_sr": "+15m SR", "sr_2024": "2024 SR",
              "sr_2025": "2025 SR", "sr_2026": "2026 SR",
              "priced_trade_coverage": "Priced fraction", "coverage_pct": "Priced %",
              "matched_days": "Matched days", "observed_growth_pct": "Observed growth %",
              "matched_active_days": "Matched active days",
              "observed_max_dd_pct": "Observed max DD %", "annual_vol_pct": "Ann. vol %"}
    return frame[columns].rename(columns=labels).to_html(index=False, classes="tbl", border=0,
                                  float_format=lambda value: f"{value:.2f}", escape=True)


def run():
    OUT.mkdir(exist_ok=True)
    raw = np.load(AUDIT / "recomputed_votes.npz")["gc4_lagged"]
    arrays, names, sessions, dates = bundle(SOURCE)
    with np.load(PRICE_SOURCE) as saved:
        prices = {key: saved[key] for key in saved.files}
    winners = pd.read_parquet(AUDIT / "individual_hindsight_winners.parquet")
    winners = winners.loc[winners.state.eq("gc4")].set_index(["root", "family"])
    choices = sorted({pair for book in BOOKS.values() for pair in book.items()})
    leg_paths = {}
    leg_rows = []
    for root, family in choices:
        row = winners.loc[(root, family)]
        gross, cost, planned = leg_path(row, root, family, raw, dates, sessions, prices, names)
        leg_paths[(root, family)] = gross, cost, planned
        leg_rows.append({**metrics(gross[0], cost[0], planned, sessions, f"{root}:{family}"),
                         "root": root, "family": family, "signal_class": row.signal_class,
                         "clock": row.clock, "hold_minutes": row.hold_minutes,
                         "delay15_sr": sharpe(gross[2]-cost[2])})
    book_paths = {}
    rows = []
    for name, book in BOOKS.items():
        legs = [leg_paths[pair] for pair in book.items()]
        gross = np.mean([leg[0] for leg in legs], axis=0)
        cost = np.mean([leg[1] for leg in legs], axis=0)
        planned = np.any([leg[2] for leg in legs], axis=0)
        book_paths[name] = {"gross": gross, "cost": cost, "planned": planned}
        row = metrics(gross[0], cost[0], planned, sessions, name)
        row["delay15_sr"] = sharpe(gross[2]-cost[2])
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("net_sr", ascending=False)
    summary.to_parquet(OUT / "book_comparison.parquet", index=False)
    pd.DataFrame(leg_rows).to_parquet(OUT / "leg_metrics.parquet", index=False)
    daily = pd.DataFrame({"session": sessions})
    for name, path in book_paths.items():
        daily[name] = path["gross"][0]-path["cost"][0]
        daily[name+"__planned"] = path["planned"]
    daily.to_parquet(OUT / "book_daily.parquet", index=False)
    names_to_plot = ("energy_carry_oi", "metals_oi_curve", "two_books_carry_oi")
    equity_bytes = png(figure(book_paths, sessions, names_to_plot))
    asset_bytes = png(asset_figure(leg_paths, ("energy_carry_oi", "metals_oi_curve"), sessions))
    family_groups = pd.read_parquet(AUDIT / "hindsight_group_metrics.parquet")
    family_groups = family_groups.loc[
        family_groups.group.isin(GROUP_LABELS)
        & family_groups.family.isin(CARRY_MOMENTUM_FAMILIES)
    ].copy()
    gold = pd.read_parquet(AUDIT / "individual_hindsight_winners.parquet")
    gold = gold.loc[gold.state.eq("gc4") & gold.root.eq("GC")
                    & gold.family.isin(CARRY_MOMENTUM_FAMILIES)].copy()
    gold["group"] = "gold"
    family_groups = pd.concat([family_groups, gold.reindex(columns=family_groups.columns)],
                              ignore_index=True)
    family_groups.to_parquet(OUT / "carry_momentum_group_metrics.parquet", index=False)
    comparison_bytes = png(family_comparison_figure(family_groups))
    (OUT / "observed_equity.png").write_bytes(equity_bytes)
    (OUT / "asset_legs.png").write_bytes(asset_bytes)
    (OUT / "carry_momentum_group_heatmap.png").write_bytes(comparison_bytes)
    common = daily[["energy_carry_oi", "metals_oi_curve", "energy_spike",
                    "metals_spike_curve", "metals_volume_curve",
                    "four_spike", "four_curve"]].corr()
    common.to_parquet(OUT / "book_correlations.parquet")
    if DATA_ONLY:
        print("[books] Corrected front-month paths and metrics saved without old-price prose",
              OUT, flush=True)
        return
    equity_b64 = base64.b64encode(equity_bytes).decode("ascii")
    asset_b64 = base64.b64encode(asset_bytes).decode("ascii")
    comparison_b64 = base64.b64encode(comparison_bytes).decode("ascii")
    display = summary.assign(
        book=summary.book.str.replace("_", " ").str.title(),
        coverage_pct=summary.priced_trade_coverage*100,
        observed_growth_pct=summary.observed_growth*100,
        observed_max_dd_pct=summary.observed_max_dd*100,
        annual_vol_pct=summary.annual_vol*100,
    )
    chosen_order = ["Energy Carry Oi", "Metals Oi Curve", "Two Books Carry Oi"]
    chosen = display.set_index("book").loc[chosen_order].reset_index()
    selected_pairs = set(BOOKS["two_books_carry_oi"].items())
    selected_legs = pd.DataFrame(leg_rows)
    selected_legs = selected_legs.loc[selected_legs.apply(
        lambda row: (row.root, row.family) in selected_pairs, axis=1)]
    comparison_display = family_groups.assign(
        group=family_groups.group.map(GROUP_LABELS),
        family=family_groups.family.str.replace("_", " ").str.title(),
        coverage_pct=family_groups.priced_trade_coverage * 100,
    ).sort_values(["group", "family"])
    def planned_months(name):
        planned = daily.loc[daily[name + "__planned"], "session"].dt.to_period("M")
        return ", ".join(str(int((planned == month).sum()))
                         for month in ("2026-08", "2026-09", "2026-10"))
    selected_corr = common.loc["energy_carry_oi", "metals_oi_curve"]
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Five Currents | Two Shadow Books</title>
<style>:root{{--ink:#163a3b;--teal:#0d625e;--ochre:#a56c2f;--paper:#f6f3eb;--panel:#fffdf8;--line:#d7d4c8}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.52 Georgia,serif}}
header{{background:linear-gradient(115deg,#123f43,#24665d 66%,#967040);color:white;padding:58px max(6vw,32px)}}
h1{{font-size:48px;line-height:1.04;margin:7px 0 15px}}header p{{max-width:850px;font-size:19px}}
main{{max-width:1200px;margin:auto;padding:28px 30px 70px}}section{{margin:34px 0}}h2{{font-size:28px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.15em;font:700 11px Arial,sans-serif;opacity:.8}}
.card{{background:var(--panel);border:1px solid var(--line);padding:18px 22px;margin:14px 0;border-radius:6px}}
.warning{{border-left:4px solid #b75b46}}figure{{margin:22px 0;background:white;border:1px solid var(--line);padding:14px}}
figure img{{display:block;width:100%;height:auto}}figcaption{{font:12px/1.5 Arial,sans-serif;color:#5d6b68;margin-top:8px}}
.scroll{{overflow-x:auto}}.tbl{{border-collapse:collapse;width:100%;font:12px/1.45 Arial,sans-serif}}
.tbl th{{background:#18494a;color:white;padding:8px;text-align:right;white-space:nowrap}}.tbl td{{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child{{text-align:left}}.cols{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
@media(max-width:800px){{.cols{{grid-template-columns:1fr}}h1{{font-size:35px}}main{{padding:20px}}}}</style></head>
<body><header><div class="eyebrow">Five Currents / offline research / 24 October 2026</div><h1>Two Ways to Trade the Currents</h1>
<p>Asset-specific intraday execution of frozen daily alpha families. The curves below show only exact-contract matched trades; unpriced active days are unknown, not cash days.</p></header>
<main><section><h2>Decision Boundary</h2><div class="card warning"><strong>Hindsight exercise, not a live execution claim.</strong>
The alpha treatment and trading clock for each commodity were selected on this same 2024-2026 sample. The two-book choice also uses knowledge that some volume-spike votes fell silent in late 2026. No intraday retuning occurs inside the replay. Missing exact-contract minute bars prevent a complete investable equity history, particularly for silver; it is excluded from the proposed books.</div></section>
<section><h2>Two Shadow Books</h2><div class="cols"><div class="card"><strong>Book A | Energy carry and open-interest reversal.</strong>
CL: reverse the signed magnitude of the current carry/OI-reversal vote; enter at 16:30 London local time, exit 3 hours later.
NG: follow the signed magnitude of the vote's change; enter at 09:30 New York local time, exit 6 hours later.
The legs are independent: do not wait for one market's signal or clock to trade the other.</div>
<div class="card"><strong>Book B | Metals open-interest and curve quality.</strong>
HG: follow the signed magnitude of the change in the curve-quality-reversal vote; enter at 09:00 Singapore time (01:00 UTC), exit 3 hours later.
GC: reverse the sign of the current OI-direction-reversion vote; enter at 09:30 New York local time, exit 2 hours later.
The metal legs are likewise independent; do not force simultaneous entry.</div></div>
<p>Compute each frozen vote after the preceding completed daily panel, with the archived one-session timing alignment. Trade only non-zero signed votes using the contract chosen by the existing pre-session liquidity/roll rule. The combination uses equal book allocations and, within each book, equal commodity allocations before applying the frozen vote magnitude. This report tests normalized weights, <em>not</em> executable lot counts or equal dollar-risk positions.</p>
<div class="scroll"><table class="tbl"><thead><tr><th>Book</th><th>Contract root</th><th>Alpha treatment</th><th>Enter</th><th>Exit</th></tr></thead><tbody>
<tr><td>A / Energy</td><td>CL</td><td>Reverse signed carry/OI vote</td><td>16:30 London local, 15:30/16:30 UTC</td><td>3h later</td></tr>
<tr><td>A / Energy</td><td>NG</td><td>Follow signed carry/OI vote change</td><td>09:30 New York local, 13:30/14:30 UTC</td><td>6h later</td></tr>
<tr><td>B / Metals</td><td>HG</td><td>Follow signed magnitude of vote change</td><td>09:00 SGT, 01:00 UTC</td><td>3h later</td></tr>
<tr><td>B / Metals</td><td>GC</td><td>Reverse sign of current OI-direction vote</td><td>09:30 New York local, 13:30/14:30 UTC</td><td>2h later</td></tr>
</tbody></table></div><p><strong>Scheduling:</strong> the books may hold positions concurrently, but the four entries are not synchronized or conditioned on one another. London and New York clocks follow their respective daylight-saving rules.</p></section>
<section><h2>Selected Book Results</h2><p>Equal fixed budget per commodity leg. Net of micro-future USD 2.16 round trip plus the source tick-slippage allowance; the stress doubles both. Sharpe is computed only on observed priced days. Return and drawdown below are on the <em>normalized unit-weight</em> series, not a dollar-VaR implementation.</p>
<div class="scroll">{table(chosen, ['book','net_sr','after_top5_sr','stress_sr','delay15_sr','sr_2024','sr_2025','sr_2026','coverage_pct','matched_days','observed_growth_pct','observed_max_dd_pct','annual_vol_pct'])}</div>
<p>Book A versus Book B daily P&amp;L correlation is {selected_corr:.2f} on common priced sessions. This is a descriptive correlation, not an independent validation sample.</p>
<p><strong>Recent planned activity, August / September / October 2026:</strong> the selected energy book has {planned_months('energy_carry_oi')} sessions and the selected metals book has {planned_months('metals_oi_curve')}. The previous volume-spike alternatives have {planned_months('energy_spike')} and {planned_months('metals_spike_curve')}, respectively. October is through the 22nd.</p></section>
<section><h2>Observed-Day Equity</h2><figure><img src="data:image/png;base64,{equity_b64}"><figcaption>Growth starts at 1. Curves compound known matched daily returns; the line is broken at missing active trades, and the right panel counts those missing sessions. No missing return is assigned a value. 2024 begins in June; 2026 ends 22 October.</figcaption></figure></section>
<section><h2>Commodity Legs</h2><div class="scroll">{table(selected_legs, ['root','family','signal_class','clock','hold_minutes','net_sr','after_top5_sr','stress_sr','delay15_sr','priced_trade_coverage'])}</div>
<figure><img src="data:image/png;base64,{asset_b64}"><figcaption>Each line is the exact-contract observed-only P&amp;L for a component; curves break where its active trade cannot be priced.</figcaption></figure></section>
<section><h2>Carry and Momentum by Group</h2><p>This is a comparison, not an addition to the two books. Each asset/family keeps its own same-history best vote treatment, clock and hold; group returns average those independently costed legs. Carry/curve includes OI reversal, basis reversal, aggregate-volume carry and curve-quality reversal. Momentum includes carry-confirmed momentum and two volume-continuation families. A blank cell means no eligible rule passed the original minimum priced-coverage and sample-count screens.</p>
<figure><img src="data:image/png;base64,{comparison_b64}"><figcaption>Net observed-day Sharpe and the fraction of intended active days with an exact-contract entry and exit. The gold-only row is more useful than GC + SI for assessing a deployable metal sleeve: GC + SI has low priced coverage, chiefly due to silver.</figcaption></figure>
<div class="scroll">{table(comparison_display, ['group','family','net_sr','after_top5_sr','stress_sr','sr_2024','sr_2025','sr_2026','coverage_pct','matched_active_days'])}</div>
<p>These family winners are hindsight-selected on the same sample as their performance; none is a forward-validated replacement for the proposed books.</p></section>
<section><h2>Other Configurations Checked</h2><p>Volume-spike alternatives retain attractive historical statistics but have become nearly inactive; that is why they are comparison rows, not the first shadow-execution choice. The higher-headline metals volume/curve alternative prices fewer intended trades and has a weak 2024 segment. The all-four volume-spike book largely duplicates its energy sleeve rather than providing a second independent book.</p><div class="scroll">{table(display, ['book','net_sr','after_top5_sr','stress_sr','coverage_pct','sr_2024','sr_2025','sr_2026'])}</div></section>
<section><h2>How to Read This</h2><div class="cols"><div class="card"><strong>Signal construction.</strong> GC uses first four curve ranks; HG and SI use first four; CL and NG use first six. The exact dashboard baseline rebuilds numerically before the GC-only change. The current dashboard policy is not altered. The same-history search covered 384 vote/clock/hold variants per alpha and commodity; the top-five ablation does not erase that selection bias.</div>
<div class="card"><strong>Price boundary.</strong> Daily GC/HG curves are nearly complete, but selected GC rank-3 and HG rank-2 contracts often lack cached minute entry prices. This test does not substitute another contract or invent an intraday mark.</div></div></section></main></body></html>"""
    (OUT / "two_books_research_report.html").write_text(html, encoding="utf-8")
    print(summary[['book','net_sr','after_top5_sr','stress_sr','delay15_sr','sr_2024',
                   'sr_2025','sr_2026','priced_trade_coverage','matched_days']].to_string(index=False), flush=True)
    print("\nDaily common-date correlations:\n" + common.to_string(), flush=True)


if __name__ == "__main__":
    run()
