"""Offline replay and single-file report of the archived Five Currents alpha audit."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_five_currents_two_books_report import (
    AUDIT, BOOKS, PRICE_SOURCE, ROOT, SOURCE, asset_figure, figure, leg_path,
    metrics, png, sharpe,
)
from five_currents_execution_timing import ROOTS
from run_five_currents_group_clock import bundle


OUT = ROOT / "five_currents_all_promising_20261024"
ROOT_ORDER = ("CL", "NG", "GC", "HG", "SI")
GROUPS = {
    "energy": ("CL", "NG"), "gold": ("GC",), "copper": ("HG",),
    "precious": ("GC", "SI"), "four": ("CL", "NG", "GC", "HG"),
    "silver_diagnostic": ("SI",),
}
GROUP_NAMES = {
    "energy": "Energy: CL + NG", "gold": "Gold: GC", "copper": "Copper: HG",
    "precious": "Precious: GC + SI", "four": "Four commodities: CL + NG + GC + HG",
    "silver_diagnostic": "Silver: SI (diagnostic only)",
}
SCREEN = {"net_sr": 1.0, "after_top5_sr": 0.4, "stress_sr": 0.5,
          "delay15_sr": 0.5, "priced_trade_coverage": 0.75,
          "matched_days": 120}
CLOCK_LABELS = {
    "asia_open": "09:00 Singapore", "asia_close": "17:00 Singapore",
    "london_open": "08:00 London", "london_close": "16:30 London",
    "us_open": "09:30 New York", "us_close": "16:00 New York",
}


def reviewed(row):
    return row["root"] != "SI" and all(row[key] >= value for key, value in SCREEN.items())


def save_png(fig, path):
    data = png(fig)
    path.write_bytes(data)
    return base64.b64encode(data).decode("ascii")


def report_table(frame, columns):
    labels = {
        "book": "Book", "root": "Asset", "family": "Alpha family",
        "group": "Group", "scope": "Status", "signal_class": "Vote treatment",
        "entry": "Entry", "hold_minutes": "Hold (min)", "net_sr": "Net SR",
        "after_top5_sr": "SR less top five", "stress_sr": "2x-cost SR",
        "delay15_sr": "+15m SR", "sr_2024": "2024 SR",
        "sr_2025": "2025 SR", "sr_2026": "2026 SR",
        "coverage_pct": "Priced %", "matched_days": "Priced days",
        "planned_2026-08": "Aug votes", "planned_2026-09": "Sep votes",
        "planned_2026-10": "Oct votes",
    }
    return frame[columns].rename(columns=labels).to_html(
        index=False, classes="tbl", border=0,
        float_format=lambda value: f"{value:.2f}", escape=True)


def plotted_equity(ax, sessions, gross, cost, *, label):
    net = gross[0] - cost[0]
    stressed = gross[0] - 2 * cost[0]
    observed = np.isfinite(net)
    curve = np.cumprod(1 + np.where(observed, net, 0.0))
    stress_curve = np.cumprod(1 + np.where(observed, stressed, 0.0))
    curve[~observed] = np.nan
    stress_curve[~observed] = np.nan
    ax.plot(sessions, curve, color="#0e6861", lw=1.3, label="Net")
    ax.plot(sessions, stress_curve, color="#b47b3e", lw=0.9, ls="--", label="2x costs")
    ax.axhline(1, color="#aeb9b4", lw=0.6)
    ax.set_title(label, loc="left", fontsize=9.5, fontweight="bold")
    ax.grid(alpha=0.15)
    ax.tick_params(axis="both", labelsize=7)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")


def small_multiples(items, sessions, title):
    rows = int(np.ceil(len(items) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, 2.8 * rows), squeeze=False)
    for ax, (label, gross, cost) in zip(axes.flat, items):
        plotted_equity(ax, sessions, gross, cost, label=label)
    for ax in axes.flat[len(items):]:
        ax.axis("off")
    fig.suptitle(title, y=1.005, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def heatmap(frame):
    families = sorted(frame.family.unique())
    groups = list(GROUPS)
    sr = np.full((len(groups), len(families)), np.nan)
    coverage = np.full_like(sr, np.nan)
    for row in frame.itertuples(index=False):
        i, j = groups.index(row.group), families.index(row.family)
        sr[i, j] = row.net_sr
        coverage[i, j] = 100 * row.priced_trade_coverage
    fig, ax = plt.subplots(figsize=(16, 6.5))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#e3e4de")
    im = ax.imshow(np.ma.masked_invalid(sr), cmap=cmap, vmin=-0.5, vmax=3.0,
                   aspect="auto")
    ax.set_xticks(range(len(families)),
                  [name.replace("_", " ").title() for name in families],
                  rotation=38, ha="right", fontsize=8)
    ax.set_yticks(range(len(groups)), [GROUP_NAMES[group] for group in groups],
                  fontsize=9)
    for i in range(len(groups)):
        for j in range(len(families)):
            note = (f"{sr[i, j]:.1f} SR\n{coverage[i, j]:.0f}% priced"
                    if np.isfinite(sr[i, j]) else "No eligible rule")
            color = "white" if np.isfinite(sr[i, j]) and sr[i, j] >= 2.6 else "#1c3532"
            ax.text(j, i, note, ha="center", va="center", fontsize=7.2, color=color)
    ax.set_title("All ten families by commodity group | observed-day net Sharpe and coverage",
                 fontsize=13, fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax, label="Net Sharpe", shrink=0.8, pad=0.015)
    fig.tight_layout()
    return fig


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recent_activity(planned, net, sessions):
    result = {}
    for month in ("2026-08", "2026-09", "2026-10"):
        period = sessions.to_period("M") == month
        result[f"planned_{month}"] = int(np.sum(planned & period))
        result[f"priced_{month}"] = int(np.sum(planned & period & np.isfinite(net)))
    return result


def run():
    OUT.mkdir(exist_ok=True)
    source = np.load(AUDIT / "recomputed_votes.npz")
    raw = source["gc4_lagged"]
    arrays, names, sessions, dates = bundle(SOURCE)
    assert len(sessions) == 593 and sessions[0] == pd.Timestamp("2024-06-20")
    assert sessions[-1] == pd.Timestamp("2026-10-22")
    with np.load(PRICE_SOURCE) as saved:
        prices = {key: saved[key] for key in saved.files}
    winners = pd.read_parquet(AUDIT / "individual_hindsight_winners.parquet")
    winners = winners.loc[winners.state.eq("gc4")].copy()
    indexed = winners.set_index(["root", "family"])
    paths = {}
    individual = []
    for row in winners.itertuples(index=False):
        gross, cost, planned = leg_path(row, row.root, row.family, raw, dates,
                                        sessions, prices, names)
        paths[(row.root, row.family)] = gross, cost, planned
        result = {**metrics(gross[0], cost[0], planned, sessions,
                            f"{row.root} | {row.family}"),
                  "root": row.root, "family": row.family,
                  "signal_class": row.signal_class, "clock": row.clock,
                  "hold_minutes": row.hold_minutes,
                  "delay15_sr": sharpe(gross[2] - cost[2]),
                  **recent_activity(planned, gross[0] - cost[0], sessions)}
        for name in ("net_sr", "after_top5_sr", "stress_sr", "delay15_sr"):
            np.testing.assert_allclose(result[name], getattr(row, name), rtol=0,
                                       atol=1e-10, equal_nan=True)
        result["review_shortlist"] = reviewed(result)
        result["selected_book_leg"] = ((row.root, row.family)
                                       in BOOKS["two_books_carry_oi"].items())
        individual.append(result)
    individual = pd.DataFrame(individual)
    individual.to_parquet(OUT / "individual_replay.parquet", index=False)
    shortlist = individual.loc[individual.review_shortlist |
                               individual.selected_book_leg].copy()
    shortlist.to_parquet(OUT / "review_shortlist.parquet", index=False)

    groups = []
    group_paths = {}
    for group, roots in GROUPS.items():
        for family in names:
            keys = [(root, family) for root in roots]
            if not all(key in paths for key in keys):
                continue
            legs = [paths[key] for key in keys]
            gross = np.mean([leg[0] for leg in legs], axis=0)
            cost = np.mean([leg[1] for leg in legs], axis=0)
            planned = np.any([leg[2] for leg in legs], axis=0)
            group_paths[(group, family)] = gross, cost, planned
            result = {**metrics(gross[0], cost[0], planned, sessions,
                                f"{group} | {family}"),
                      "group": group, "family": family,
                      "delay15_sr": sharpe(gross[2] - cost[2]),
                      **recent_activity(planned, gross[0] - cost[0], sessions)}
            result["review_shortlist"] = (group not in ("precious", "silver_diagnostic")
                                          and all(result[key] >= value
                                                  for key, value in SCREEN.items()))
            groups.append(result)
    groups = pd.DataFrame(groups)
    groups.to_parquet(OUT / "group_replay.parquet", index=False)
    archived = pd.read_parquet(AUDIT / "hindsight_group_metrics.parquet")
    archived = archived.loc[archived.state.eq("gc4")]
    compare = groups.loc[groups.group.isin(("energy", "copper", "precious", "four"))].copy()
    compare["old_group"] = compare.group.replace({"four": "overall_four"})
    compare = compare.merge(archived, left_on=["old_group", "family"],
                            right_on=["group", "family"], suffixes=("_new", "_old"),
                            validate="one_to_one")
    for field in ("net_sr", "after_top5_sr", "stress_sr", "priced_trade_coverage"):
        np.testing.assert_allclose(compare[field + "_new"], compare[field + "_old"],
                                   rtol=0, atol=1e-10, equal_nan=True)

    books = []
    book_paths = {}
    for book, members in BOOKS.items():
        legs = [paths[key] for key in members.items()]
        gross = np.mean([leg[0] for leg in legs], axis=0)
        cost = np.mean([leg[1] for leg in legs], axis=0)
        planned = np.any([leg[2] for leg in legs], axis=0)
        book_paths[book] = {"gross": gross, "cost": cost, "planned": planned}
        books.append({**metrics(gross[0], cost[0], planned, sessions, book),
                      "delay15_sr": sharpe(gross[2] - cost[2]),
                      **recent_activity(planned, gross[0] - cost[0], sessions)})
    books = pd.DataFrame(books)
    books.to_parquet(OUT / "book_replay.parquet", index=False)
    previous = pd.read_parquet(ROOT / "five_currents_two_books_20261024"
                              / "book_comparison.parquet")
    check = books.merge(previous, on="book", suffixes=("_new", "_old"),
                        validate="one_to_one")
    for field in ("net_sr", "after_top5_sr", "stress_sr", "delay15_sr",
                  "priced_trade_coverage"):
        np.testing.assert_allclose(check[field + "_new"], check[field + "_old"],
                                   rtol=0, atol=1e-10, equal_nan=True)

    selected = ("energy_carry_oi", "metals_oi_curve", "two_books_carry_oi")
    book_b64 = save_png(figure(book_paths, sessions, selected), OUT / "book_equity.png")
    legs_b64 = save_png(asset_figure(paths, selected[:2], sessions),
                        OUT / "book_asset_legs.png")
    heat_b64 = save_png(heatmap(groups), OUT / "all_family_groups.png")
    root_figures = {}
    all_root_figures = {}
    for root in ROOT_ORDER:
        picked = shortlist.loc[shortlist.root.eq(root)].sort_values("net_sr", ascending=False)
        if picked.empty:
            continue
        items = []
        for row in picked.itertuples(index=False):
            gross, cost, _ = paths[(root, row.family)]
            label = (f"{row.family.replace('_', ' ').title()} | SR {row.net_sr:.2f}"
                     f" | {row.priced_trade_coverage:.0%} priced")
            items.append((label, gross, cost))
        root_figures[root] = save_png(
            small_multiples(items, sessions, f"{root}: review shortlist and selected book legs"),
            OUT / f"{root}_alpha_equity.png")
    for root in ROOT_ORDER:
        picked = individual.loc[individual.root.eq(root)].sort_values("net_sr", ascending=False)
        items = []
        for row in picked.itertuples(index=False):
            gross, cost, _ = paths[(root, row.family)]
            label = (f"{row.family.replace('_', ' ').title()} | SR {row.net_sr:.2f}"
                     f" | {row.priced_trade_coverage:.0%} priced")
            items.append((label, gross, cost))
        all_root_figures[root] = save_png(
            small_multiples(items, sessions, f"{root}: every eligible archived family"),
            OUT / f"{root}_all_family_equity.png")
    group_picks = groups.loc[groups.review_shortlist &
                             groups.group.isin(("energy", "four"))].sort_values(
        ["group", "net_sr"], ascending=[True, False])
    group_items = []
    for row in group_picks.itertuples(index=False):
        gross, cost, _ = group_paths[(row.group, row.family)]
        label = (f"{GROUP_NAMES[row.group]} | {row.family.replace('_', ' ').title()}"
                 f" | SR {row.net_sr:.2f}, {row.priced_trade_coverage:.0%} priced")
        group_items.append((label, gross, cost))
    group_b64 = save_png(small_multiples(group_items, sessions,
                                        "Group-family review shortlist"),
                         OUT / "group_alpha_equity.png")

    individual_view = individual.assign(
        family=individual.family.str.replace("_", " ").str.title(),
        coverage_pct=individual.priced_trade_coverage * 100,
        scope=np.where(individual.review_shortlist, "Review shortlist",
                       np.where(individual.selected_book_leg, "Book leg", "Context")),
        entry=individual.clock.map(CLOCK_LABELS),
    )
    short_view = individual_view.loc[individual_view.scope.ne("Context")].sort_values(
        ["root", "net_sr"], ascending=[True, False])
    all_view = individual_view.sort_values(["root", "net_sr"], ascending=[True, False])
    group_view = groups.assign(
        group=groups.group.map(GROUP_NAMES),
        family=groups.family.str.replace("_", " ").str.title(),
        coverage_pct=groups.priced_trade_coverage * 100,
    ).sort_values(["group", "net_sr"], ascending=[True, False])
    book_view = books.loc[books.book.isin(selected)].copy()
    book_view["book"] = book_view.book.str.replace("_", " ").str.title()
    book_view["coverage_pct"] = book_view.priced_trade_coverage * 100
    book_view = book_view.set_index("book").loc[
        [name.replace("_", " ").title() for name in selected]].reset_index()
    book_legs = individual_view.loc[individual_view.selected_book_leg].sort_values("root")
    root_sections = "".join(
        f'<details><summary>{root} | {len(shortlist.loc[shortlist.root.eq(root)])} rules'
        f'</summary><figure><img src="data:image/png;base64,{root_figures[root]}">'
        f'<figcaption>Observed exact-contract sessions only; lines break at missing '
        f'active trades. Titles show net Sharpe and priced-trade coverage.</figcaption>'
        f'</figure></details>' for root in ROOT_ORDER if root in root_figures)
    all_root_sections = "".join(
        f'<details><summary>{root} | {len(individual.loc[individual.root.eq(root)])} '
        f'eligible families</summary><figure><img src="data:image/png;base64,'
        f'{all_root_figures[root]}"><figcaption>Full individual audit, including weak and '
        f'low-coverage candidates; SI is diagnostic only.</figcaption></figure></details>'
        for root in ROOT_ORDER)
    energy = (book_paths["energy_carry_oi"]["gross"][0]
              - book_paths["energy_carry_oi"]["cost"][0])
    metals = (book_paths["metals_oi_curve"]["gross"][0]
              - book_paths["metals_oi_curve"]["cost"][0])
    common = np.isfinite(energy) & np.isfinite(metals)
    selected_corr = np.corrcoef(energy[common], metals[common])[0, 1]
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Five Currents | All Promising Intraday Alphas</title><style>
:root{{--ink:#183c3d;--teal:#0e6861;--paper:#f5f3ed;--panel:#fffdf8;--line:#d9d8cf;--warn:#b45845}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 Georgia,serif}}
header{{background:linear-gradient(115deg,#153c40,#317369 70%,#ad8c53);color:white;padding:56px max(5vw,24px)}}
header h1{{font-size:46px;line-height:1.08;max-width:950px;margin:8px 0 15px}}header p{{max-width:880px;font-size:19px}}
main{{max-width:1360px;margin:auto;padding:28px 30px 70px}}h2{{font-size:29px;border-bottom:1px solid var(--line);padding-bottom:7px}}
h3{{font-size:21px}}section{{margin:35px 0}}.eyebrow{{text-transform:uppercase;letter-spacing:.14em;font:700 11px Arial,sans-serif}}
.card{{background:var(--panel);border:1px solid var(--line);padding:18px 23px;border-radius:5px;margin:13px 0}}
.warning{{border-left:4px solid var(--warn)}}.cols{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
figure{{margin:18px 0;background:white;border:1px solid var(--line);padding:12px}}figure img{{display:block;width:100%;height:auto}}
figcaption{{font:12px/1.45 Arial,sans-serif;color:#526562;margin-top:9px}}.scroll{{overflow-x:auto}}
.tbl{{border-collapse:collapse;width:100%;font:12px/1.45 Arial,sans-serif}}.tbl th{{background:#194f4f;color:white;padding:7px;text-align:right;white-space:nowrap}}
.tbl td{{padding:6px 7px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child{{text-align:left}}details{{background:var(--panel);border:1px solid var(--line);padding:13px 16px;margin:9px 0}}
summary{{cursor:pointer;font:bold 15px Arial,sans-serif}}a{{color:#0d675d}}@media(max-width:800px){{.cols{{grid-template-columns:1fr}}header h1{{font-size:35px}}main{{padding:18px}}}}
</style></head><body><header><div class="eyebrow">Five Currents / offline audit / 24 October 2026</div>
<h1>The Intraday Alpha Map</h1><p>One report for the complete cached commodity/family replay, its review shortlist, the two proposed shadow books, and the carry-versus-momentum group comparison.</p></header><main>
<section><h2>Read This First</h2><div class="card warning"><strong>Hypothetical, hindsight-selected research.</strong>
The original search tested eight vote treatments, six clocks and eight holding periods per eligible commodity/family. Each displayed rule is its same-history winner, not a causally selected clock or an out-of-sample result. This report replays all 48 eligible winners against the pinned daily votes and exact-contract minute cache. Missing active trades remain unknown, not zero. The two books are candidates for <em>shadow execution only</em> until market-data coverage, source-field freshness, live spreads and lot sizing are independently validated.</div>
<p>Period: 20 June 2024–22 October 2026, 593 sessions. Price/volume/OI votes are frozen from the archive. GC/HG/SI use curve ranks 1–4; CL/NG use 1–6. Micro-futures costs: USD 2.16 round trip per lot plus the original one-tick-per-side allowance, with a separate 2x-cost stress. London and New York clocks follow their local daylight-saving rules. All reported Sharpe ratios use observed priced sessions; cumulative lines break at missing active trades.</p></section>
<section><h2>Two Shadow Books</h2><div class="cols"><div class="card"><h3>A | Energy carry and OI</h3><p>CL: reverse the current carry/OI-reversal vote at 16:30 London, exit three hours later. NG: follow the change in that vote at 09:30 New York, exit six hours later.</p></div>
<div class="card"><h3>B | Gold and copper</h3><p>GC: reverse the current OI-direction-reversion sign at 09:30 New York, exit two hours later. HG: follow the change in the curve-quality-reversal vote at 09:00 Singapore, exit three hours later.</p></div></div>
<p>Legs enter independently; they are not conditional on agreement across commodities. The displayed combination is 50/50 between books and equal allocation to the two legs inside each book before vote magnitude. It is a normalized research allocation, not a USD VaR target or executable lot schedule. Book A/B common-date correlation is approximately {selected_corr:.2f}, descriptive only.</p>
<div class="scroll">{report_table(book_view, ['book','net_sr','after_top5_sr','stress_sr','delay15_sr','sr_2024','sr_2025','sr_2026','coverage_pct','matched_days','planned_2026-08','planned_2026-09','planned_2026-10'])}</div>
<figure><img src="data:image/png;base64,{book_b64}"><figcaption>Full observed-only equity history, net and doubled costs. Coverage bars distinguish priced, unpriced and no-position sessions.</figcaption></figure>
<h3>Constituent Legs</h3><div class="scroll">{report_table(book_legs, ['root','family','signal_class','entry','hold_minutes','net_sr','after_top5_sr','stress_sr','delay15_sr','coverage_pct','planned_2026-08','planned_2026-09','planned_2026-10'])}</div>
<figure><img src="data:image/png;base64,{legs_b64}"><figcaption>Component equity curves. These are not equal-dollar-risk or $1m VaR-sized results.</figcaption></figure></section>
<section><h2>All Candidate Families by Group</h2><p>The heatmap includes every eligible archived family in energy, gold, copper, the four-commodity combination, and two silver diagnostics. Each group averages independently costed asset legs; no rule is forced to share a clock or vote treatment across assets. A high precious-metals SR with 25–50% coverage should not outrank a lower but well-priced energy or gold result.</p>
<figure><img src="data:image/png;base64,{heat_b64}"><figcaption>Each cell shows net observed-day Sharpe and priced fraction of intended active days. Grey means a commodity in that group did not have an eligible archived winner for the family.</figcaption></figure>
<details><summary>Full group table: all family combinations</summary><div class="scroll">{report_table(group_view, ['group','family','net_sr','after_top5_sr','stress_sr','delay15_sr','sr_2024','sr_2025','sr_2026','coverage_pct','matched_days','planned_2026-08','planned_2026-09','planned_2026-10'])}</div></details></section>
<section><h2>What the Families Say</h2><div class="cols"><div class="card"><h3>Energy</h3><p>CL carry/OI reversal is the strongest robust individual energy leg (1.63 net SR, 97% priced). NG is weaker individually: its carry/OI leg is positive but fails the doubled-cost shortlist threshold; it remains visible because the CL+NG group improves to 1.72 net SR. The volume-spike-reversion group has 1.55 SR and 97% priced coverage, but its votes disappear in the latest months.</p></div>
<div class="card"><h3>Metals</h3><p>Gold has the broadest apparent family support, including volume continuation, aggregate-volume carry and OI-direction reversal. Copper's most credible rule is curve-quality reversal (1.35 net SR, 88% priced); its aggregate-volume carry is weak. Silver's apparent 2–3 Sharpe rules price only about a third of intended trades, so they cannot justify an executable silver sleeve.</p></div></div>
<p>Across CL, NG, GC and HG, curve-quality reversal (1.94 net SR, 78% priced) and volume-spike reversion (1.91, 93% priced) look attractive, but their family-wide curves also inherit the same-history clock search. The latter has poor current vote activity. None of these comparisons is an independent forward test.</p></section>
<section><h2>Review Shortlist</h2><p>A transparent descriptive screen, <em>not</em> a p-value or a forward selection rule: non-silver individual rules with net SR ≥1.0, SR after removing their five best priced days ≥0.4, doubled-cost SR ≥0.5, +15-minute entry-delay SR ≥0.5, priced-trade coverage ≥75%, and at least 120 priced active days. It flags {int(individual.review_shortlist.sum())} rules. Any proposed book leg that misses this screen stays visible as a book leg, not mislabelled as a pass. Because clocks and treatments were chosen on the full sample, passing this screen does not remove selection bias.</p>
<div class="scroll">{report_table(short_view, ['scope','root','family','signal_class','entry','hold_minutes','net_sr','after_top5_sr','stress_sr','delay15_sr','sr_2024','sr_2025','sr_2026','coverage_pct','matched_days','planned_2026-08','planned_2026-09','planned_2026-10'])}</div>
<h3>Individual Equity Curves</h3>{root_sections}</section>
<section><h2>Promising Group Equity</h2><p>The same descriptive screen produces {len(group_items)} multi-asset energy or four-commodity group/family rows, plotted below. Single-asset gold and copper are already shown above; silver-containing groups are excluded because exact-contract minute pricing is too incomplete. All numerical outcomes remain in the full group table.</p>
<figure><img src="data:image/png;base64,{group_b64}"><figcaption>Observed-only group P&amp;L, never imputed through missing active trades. Daily series are equal-weight averages of independently timed commodity legs.</figcaption></figure></section>
<section><h2>What Did Not Make the First Shadow Cut</h2><p>The original energy volume-spike book planned only 3, 0 and 0 sessions in August, September and October 2026; the original metals spike/curve book planned 10, 1 and 3. This is a genuine lack of votes in the pinned archive, not merely a plotting gap. Their prior statistics remain in the comparison table; switching away from them is itself a hindsight decision. Silver has striking apparent alphas but only roughly one-third of intended exact-contract trades can be priced. The combined two-book curve has missing active sessions too, so its Sharpe is conditional on what we can observe rather than a complete investable record.</p>
<p>Before capital deployment: run these rules forward unchanged, repair exact-contract endpoint coverage, verify OI and volume freshness at each decision time, and calibrate micro-lot sizing and execution slippage from actual tradable depth. The <a href="https://www.nfa.futures.org/rulebooksql/rules.aspx?RuleID=9025&amp;Section=9">NFA discussion of hypothetical performance</a> describes why hindsight and unmodelled liquidity can make simulated outcomes misleading.</p></section>
<section><h2>Complete Individual Audit</h2><p>All 48 eligible archived commodity/family winners are shown, including weak or unstable rules and silver diagnostics. No row has been deleted because it weakens the narrative.</p><details><summary>Show complete 48-rule table</summary><div class="scroll">{report_table(all_view, ['root','family','signal_class','entry','hold_minutes','net_sr','after_top5_sr','stress_sr','delay15_sr','sr_2024','sr_2025','sr_2026','coverage_pct','matched_days','planned_2026-08','planned_2026-09','planned_2026-10'])}</div></details>{all_root_sections}</section>
</main></body></html>"""
    report = OUT / "all_promising_intraday_report.html"
    report.write_text(html, encoding="utf-8")
    inputs = (AUDIT / "recomputed_votes.npz", AUDIT / "individual_hindsight_winners.parquet",
              SOURCE / "sessions.parquet", PRICE_SOURCE)
    (OUT / "manifest.json").write_text(json.dumps({
        "source_sha256": {str(path): sha256(path) for path in inputs},
        "sessions": len(sessions), "start": str(sessions[0].date()),
        "end": str(sessions[-1].date()), "eligible_winners": len(individual),
        "review_shortlist": int(individual.review_shortlist.sum()),
        "book_exception_legs": int((individual.selected_book_leg &
                                    ~individual.review_shortlist).sum()),
        "screen": SCREEN, "database_access": False, "market_api_access": False,
        "selection_warning": "Original grid winners and the review screen are same-history, descriptive research, not out-of-sample validation.",
    }, indent=2), encoding="utf-8")
    print(f"[complete] {report} | {len(individual)} winners | "
          f"{int(individual.review_shortlist.sum())} review passes | "
          f"{len(groups)} group cells")
    print(books.loc[books.book.isin(selected), ["book", "net_sr", "after_top5_sr",
                                            "stress_sr", "delay15_sr",
                                            "priced_trade_coverage", "matched_days"]].to_string(index=False))


if __name__ == "__main__":
    run()
