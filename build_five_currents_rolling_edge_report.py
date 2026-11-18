"""Self-contained report for the offline rolling intraday allocator sweep."""
from __future__ import annotations

import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_five_currents_two_books_report import BOOKS
from five_currents_group_clock_core import row_stats
from run_five_currents_rolling_edge import (
    ASSETS, ASSET_CADENCES, ASSET_WINDOWS, FAMILY_CADENCES, FAMILY_WINDOWS,
    OUT, WARMUP,
)


COLORS = {"fixed_carry_oi": "#0c625e", "fixed_volume_spike": "#8f9d98",
          "fixed_energy": "#0c625e", "fixed_energy_spike": "#8f9d98",
          "family_L63_C5_K2_equal": "#bb782e",
          "family_L252_C21_K2_equal": "#2d5e84",
          "one_time_fit_K2_equal": "#ac5950"}
LABELS = {"fixed_carry_oi": "Fixed carry/OI books",
          "fixed_volume_spike": "Fixed volume-spike books",
          "family_L63_C5_K2_equal": "Rolling 63 / 5",
          "family_L252_C21_K2_equal": "Rolling 252 / 21",
          "one_time_fit_K2_equal": "One-time fit",
          "fixed_energy": "Fixed energy carry/OI",
          "fixed_energy_spike": "Fixed energy volume spike",
          "family_L63_C5_K2_equal__energy": "Energy: rolling 63 / 5",
          "family_L252_C21_K2_equal__energy": "Energy: rolling 252 / 21"}


def image_bytes(fig, filename):
    stream = BytesIO()
    fig.savefig(stream, format="png", dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data = stream.getvalue()
    (OUT / filename).write_bytes(data)
    return base64.b64encode(data).decode("ascii")


def equity(values):
    x = np.asarray(values, float)
    observed = np.isfinite(x)
    curve = np.cumprod(1 + np.where(observed, x, 0))
    curve[~observed] = np.nan
    return curve


def equity_figure(daily):
    sessions = daily.session
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.1), sharex=True)
    portfolio = ("fixed_carry_oi", "fixed_volume_spike",
                 "family_L63_C5_K2_equal", "family_L252_C21_K2_equal",
                 "one_time_fit_K2_equal")
    energy = ("fixed_energy", "fixed_energy_spike",
              "family_L63_C5_K2_equal__energy",
              "family_L252_C21_K2_equal__energy")
    for ax, keys, title in zip(axes, (portfolio, energy),
                               ("Four-asset portfolio", "CL + NG only")):
        for key in keys:
            color = COLORS.get(key.replace("__energy", ""), "#6d7572")
            ax.plot(sessions, equity(daily[key]), label=LABELS.get(key, key),
                    color=color, lw=1.5 if "fixed_carry" in key or key == "fixed_energy" else 1.1,
                    ls="--" if "one_time" in key else "-")
        ax.axhline(1, color="#9caba5", lw=.7)
        ax.axvspan(sessions.iloc[0], sessions.iloc[WARMUP], color="#c3d3ce",
                   alpha=.22, label="63-session common warmup")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("Growth of 1, observed only")
        ax.grid(alpha=.15)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.suptitle("A moving edge? Full-history normalized equity", fontsize=16,
                 fontweight="bold", y=1.005)
    fig.tight_layout()
    return fig


def recent_energy_figure(daily):
    recent = daily.loc[daily.session >= pd.Timestamp("2026-05-01")]
    fig, ax = plt.subplots(figsize=(12.7, 4.3))
    keys = ("fixed_energy", "fixed_energy_spike",
            "family_L63_C5_K2_equal__energy",
            "family_L252_C21_K2_equal__energy")
    for key in keys:
        ax.plot(recent.session, equity(recent[key]), label=LABELS[key],
                color=COLORS.get(key.replace("__energy", ""), "#0c625e"), lw=1.5)
    ax.axhline(1, color="#9caba5", lw=.7)
    ax.set_title("Energy detail | May to October 2026", loc="left",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of 1, observed only")
    ax.grid(alpha=.15)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def family_heatmaps(metrics, energy_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    for ax, frame, label in zip(axes, (metrics, energy_metrics),
                                ("Four-asset portfolio", "Energy: CL + NG")):
        matrix = np.full((len(FAMILY_WINDOWS), len(FAMILY_CADENCES)), np.nan)
        coverage = np.full_like(matrix, np.nan)
        for i, window in enumerate(FAMILY_WINDOWS):
            for j, cadence in enumerate(FAMILY_CADENCES):
                key = f"family_L{window}_C{cadence}_K2_equal"
                row = frame.loc[frame.strategy.eq(key)].iloc[0]
                matrix[i, j], coverage[i, j] = row.net_sr, row.coverage * 100
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-.5, vmax=2.5, aspect="auto")
        ax.set_xticks(range(len(FAMILY_CADENCES)),
                      [f"{cadence} sessions" for cadence in FAMILY_CADENCES], fontsize=8)
        ax.set_yticks(range(len(FAMILY_WINDOWS)),
                      [f"{window} sessions" for window in FAMILY_WINDOWS], fontsize=8)
        ax.set_xlabel("Family refit cadence")
        ax.set_ylabel("Trailing history")
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i,j]:.2f} SR\n{coverage[i,j]:.0f}% priced",
                        ha="center", va="center", fontsize=8,
                        color="white" if matrix[i,j] > 2.2 else "#173b39")
    fig.subplots_adjust(left=.07, right=.86, wspace=.32, top=.79, bottom=.15)
    colorbar_ax = fig.add_axes([.89, .18, .018, .56])
    fig.colorbar(im, cax=colorbar_ax, label="Observed-day net Sharpe")
    fig.suptitle("Rolling family selection | net Sharpe and exact-contract coverage",
                 fontsize=14, fontweight="bold", y=.96)
    return fig


def asset_heatmaps(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4))
    for ax, family, label in zip(axes, ("L63_C5", "L252_C21"),
                                ("Fast weekly families", "Slow monthly families")):
        base = metrics.loc[metrics.strategy.eq(f"family_{family}_K2_equal")].iloc[0].net_sr
        matrix = np.full((len(ASSET_WINDOWS), len(ASSET_CADENCES)), np.nan)
        for i, window in enumerate(ASSET_WINDOWS):
            for j, cadence in enumerate(ASSET_CADENCES):
                key = f"family_{family}_K2_asset_L{window}_C{cadence}"
                matrix[i, j] = metrics.loc[metrics.strategy.eq(key)].iloc[0].net_sr - base
        im = ax.imshow(matrix, cmap="BrBG", vmin=-.25, vmax=.25, aspect="auto")
        ax.set_xticks(range(len(ASSET_CADENCES)), ASSET_CADENCES, fontsize=8)
        ax.set_yticks(range(len(ASSET_WINDOWS)), ASSET_WINDOWS, fontsize=8)
        ax.set_xlabel("Asset refit cadence (sessions)")
        ax.set_ylabel("Asset lookback (sessions)")
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center",
                        fontsize=9, color="#173b39")
    fig.subplots_adjust(left=.08, right=.86, wspace=.34, top=.79, bottom=.15)
    colorbar_ax = fig.add_axes([.89, .18, .018, .56])
    fig.colorbar(im, cax=colorbar_ax, label="Change in net Sharpe vs equal assets")
    fig.suptitle("Asset-edge tilts add little to the family fit", fontsize=14,
                 fontweight="bold", y=.96)
    return fig


def selection_figure(selections):
    frame = selections.loc[selections.strategy.eq("family_L63_C5_K2_equal")
                           & selections.root.isin(("CL", "NG"))]
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.1))
    for ax, root in zip(axes, ("CL", "NG")):
        part = (frame.loc[frame.root.eq(root)]
                .sort_values("selected_aug_oct_2026", ascending=False).head(6))
        labels = part.family.str.replace("_", " ").str.title()
        ax.barh(labels.iloc[::-1], part.selected_aug_oct_2026.iloc[::-1],
                color="#2d6b65")
        ax.set_title(root + " | Aug-Oct 2026", loc="left", fontsize=11,
                     fontweight="bold")
        ax.set_xlabel("Sessions selected for a known pre-entry vote")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Fast rolling model changes the energy family mix",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig


def formatted_table(frame, columns, rename=None):
    display = frame[columns].rename(columns=rename or {})
    return display.to_html(index=False, classes="tbl", border=0,
                           na_rep="n/a", float_format=lambda x: f"{x:.3f}", escape=True)


def monthly_table(daily):
    rows = []
    names = ("fixed_carry_oi", "family_L63_C5_K2_equal",
             "family_L252_C21_K2_equal", "fixed_energy",
             "family_L63_C5_K2_equal__energy",
             "family_L252_C21_K2_equal__energy")
    for period, frame in daily.loc[daily.session >= pd.Timestamp("2026-07-01")].groupby(
            daily.session.dt.to_period("M")):
        for name in names:
            x = frame[name].dropna()
            rows.append({"Month": str(period), "Sleeve": "Energy" if
                         name.endswith("__energy") or name == "fixed_energy" else "Portfolio",
                         "Model": LABELS[name], "Observed return %":
                         100 * (np.prod(1 + x) - 1) if len(x) else np.nan,
                         "Priced sessions": int(len(x)), "Calendar sessions": len(frame)})
    return pd.DataFrame(rows)


def asset_table(daily):
    winners = pd.read_parquet(OUT / "candidate_metadata.parquet")
    with np.load(OUT / "candidate_paths.npz") as saved:
        fixed_paths = saved["gross"] - saved["cost"]
        fixed_planned = saved["planned"]
    with np.load(OUT / "weekly_63_weights.npz") as saved:
        fast_weights = saved["family_weights"]
    with np.load(OUT / "primary_weights.npz") as saved:
        slow_weights = saved["family_weights"]
    sessions = daily.session
    late = sessions >= pd.Timestamp("2026-08-01")
    rows = []
    for root in ASSETS:
        indices = np.flatnonzero(winners.root.to_numpy() == root)
        fixed_family = BOOKS["two_books_carry_oi"][root]
        fixed_index = int(winners.index[(winners.root == root)
                                        & (winners.family == fixed_family)][0])
        variants = (("Fixed", fixed_paths[:, fixed_index], fixed_planned[:, fixed_index]),
                    ("Rolling 63 / 5", daily[f"family_L63_C5_K2_equal__{root}"].to_numpy(),
                     (fast_weights[:, indices] > 0).any(axis=1)),
                    ("Rolling 252 / 21", daily[f"family_L252_C21_K2_equal__{root}"].to_numpy(),
                     (slow_weights[:, indices] > 0).any(axis=1)))
        for label, pnl, planned in variants:
            priced = planned & np.isfinite(pnl)
            late_values = pnl[late & priced]
            rows.append({"Asset": root, "Model": label,
                         "Full net SR": row_stats(pnl)["sharpe"],
                         "2025 SR": row_stats(pnl[sessions.dt.year.eq(2025)])["sharpe"],
                         "2026 SR": row_stats(pnl[sessions.dt.year.eq(2026)])["sharpe"],
                         "Aug-Oct observed return %": 100 * (np.prod(1 + late_values) - 1)
                         if len(late_values) else np.nan,
                         "Aug-Oct planned trades": int((planned & late).sum()),
                         "Aug-Oct priced sessions": int((priced & late).sum()),
                         "Active fill coverage %": 100 * priced.sum() / planned.sum()
                         if planned.any() else np.nan})
    return pd.DataFrame(rows)


def run():
    OUT.mkdir(exist_ok=True)
    metrics = pd.read_parquet(OUT / "grid_metrics.parquet")
    energy = pd.read_parquet(OUT / "energy_metrics.parquet")
    daily = pd.read_parquet(OUT / "daily_replay.parquet")
    selections = pd.read_parquet(OUT / "family_selection_counts.parquet")
    with np.load(OUT / "weekly_63_weights.npz") as saved:
        fast_asset_weights = saved["asset_weights"]
    fixed = metrics.loc[metrics.strategy.eq("fixed_carry_oi")].iloc[0]
    fast = metrics.loc[metrics.strategy.eq("family_L63_C5_K2_equal")].iloc[0]
    slow = metrics.loc[metrics.strategy.eq("family_L252_C21_K2_equal")].iloc[0]
    fast_tilt = metrics.loc[metrics.strategy.eq("family_L63_C5_K2_asset_L252_C21")].iloc[0]
    slow_tilt = metrics.loc[metrics.strategy.eq("family_L252_C21_K2_asset_L252_C21")].iloc[0]
    fixed_energy = energy.loc[energy.strategy.eq("fixed_energy")].iloc[0]
    fast_energy = energy.loc[energy.strategy.eq("family_L63_C5_K2_equal")].iloc[0]
    slow_energy = energy.loc[energy.strategy.eq("family_L252_C21_K2_equal")].iloc[0]
    early = daily.loc[daily.session.dt.year.eq(2026)]
    pair = early[["fixed_energy", "family_L63_C5_K2_equal__energy"]].dropna()
    sr = lambda x: row_stats(x.to_numpy())["sharpe"]
    common_energy_days = len(pair)
    common_fixed_energy_sr, common_fast_energy_sr = sr(pair.iloc[:, 0]), sr(pair.iloc[:, 1])
    one = daily["one_time_fit_K2_equal"]
    weekly = daily["family_L63_C5_K2_equal"]
    common = one.notna() & weekly.notna()
    one_common_sr, weekly_common_sr = sr(one[common]), sr(weekly[common])

    images = {
        "equity": image_bytes(equity_figure(daily), "equity_full.png"),
        "recent": image_bytes(recent_energy_figure(daily), "energy_recent.png"),
        "families": image_bytes(family_heatmaps(metrics, energy), "family_grid.png"),
        "assets": image_bytes(asset_heatmaps(metrics), "asset_tilt_grid.png"),
        "selection": image_bytes(selection_figure(selections), "energy_family_selection.png"),
    }
    headline_names = ("fixed_carry_oi", "one_time_fit_K2_equal",
                      "family_L63_C5_K2_equal", "family_L252_C21_K2_equal",
                      "family_L63_C5_K2_asset_L252_C21",
                      "family_L252_C21_K2_asset_L252_C21")
    headline = metrics.loc[metrics.strategy.isin(headline_names)].copy()
    headline["Model"] = headline.strategy.map(LABELS).fillna(headline.strategy).replace({
        "family_L63_C5_K2_asset_L252_C21": "Fast families + 252/21 asset tilt",
        "family_L252_C21_K2_asset_L252_C21": "Slow families + 252/21 asset tilt"})
    headline["Coverage %"] = 100 * headline.coverage
    headline["Aug-Oct observed return %"] = 100 * headline.late_growth
    headline = headline.set_index("strategy").loc[list(headline_names)].reset_index()
    family_grid = metrics.loc[metrics.strategy.str.match(r"^family_L\d+_C\d+_K2_equal$")].copy()
    family_grid["Coverage %"] = 100 * family_grid.coverage
    family_grid["Aug-Oct observed return %"] = 100 * family_grid.late_growth
    family_grid["Model"] = family_grid.strategy
    asset_grid = metrics.loc[metrics.strategy.str.contains("_asset_L")].copy()
    asset_grid["Coverage %"] = 100 * asset_grid.coverage
    asset_grid["Model"] = asset_grid.strategy
    energy_grid = energy.loc[energy.strategy.str.match(r"^family_L\d+_C\d+_K2_equal$")].copy()
    energy_grid["Coverage %"] = 100 * energy_grid.coverage
    energy_grid["Aug-Oct observed return %"] = 100 * energy_grid.late_growth
    energy_grid["Model"] = energy_grid.strategy
    months = monthly_table(daily)
    assets = asset_table(daily)
    chosen_late = selections.loc[selections.strategy.eq("family_L63_C5_K2_equal")
                                 & selections.root.isin(("CL", "NG"))].copy()
    chosen_late = chosen_late.loc[chosen_late.selected_aug_oct_2026.gt(0)]
    chosen_late["Family"] = chosen_late.family.str.replace("_", " ").str.title()
    chosen_late = chosen_late.sort_values(["root", "selected_aug_oct_2026"],
                                          ascending=[True, False])

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Five Currents | Rolling Intraday Edge</title>
<style>:root{{--ink:#173c3e;--teal:#0c625e;--ochre:#bb782e;--paper:#f5f3ed;--panel:#fffdf8;--line:#d8d9d2;--warn:#ae5546}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.56 Georgia,serif}}
header{{background:linear-gradient(115deg,#153c40,#34736b 72%,#aa8550);padding:52px max(5vw,24px);color:white}}
header h1{{font-size:45px;line-height:1.07;margin:9px 0 16px;max-width:900px}}header p{{max-width:880px;font-size:19px}}
.eyebrow{{font:700 11px Arial,sans-serif;text-transform:uppercase;letter-spacing:.15em}}main{{max-width:1300px;margin:auto;padding:28px 28px 65px}}
section{{margin:35px 0}}h2{{font-size:29px;border-bottom:1px solid var(--line);padding-bottom:8px}}h3{{font-size:21px}}
.card{{background:var(--panel);border:1px solid var(--line);padding:17px 21px;border-radius:5px;margin:14px 0}}
.warn{{border-left:4px solid var(--warn)}}.cols{{display:grid;grid-template-columns:1fr 1fr;gap:17px}}
figure{{margin:19px 0;background:white;border:1px solid var(--line);padding:12px}}figure img{{display:block;width:100%;height:auto}}
figcaption{{font:12px/1.45 Arial,sans-serif;color:#526562;margin-top:8px}}.scroll{{overflow-x:auto}}
.tbl{{border-collapse:collapse;width:100%;font:12px/1.45 Arial,sans-serif}}.tbl th{{background:#194f4f;color:white;text-align:right;padding:7px;white-space:nowrap}}
.tbl td{{text-align:right;padding:6px 7px;border-bottom:1px solid var(--line);white-space:nowrap}}.tbl th:first-child,.tbl td:first-child{{text-align:left}}
details{{background:var(--panel);border:1px solid var(--line);padding:12px 16px;margin:10px 0}}summary{{cursor:pointer;font:bold 15px Arial,sans-serif}}
a{{color:#0c625e}}@media(max-width:800px){{.cols{{grid-template-columns:1fr}}header h1{{font-size:35px}}main{{padding:17px}}}}</style></head>
<body><header><div class="eyebrow">Five Currents / offline rolling replay / 24 October 2026</div>
<h1>Can a Moving Intraday Edge Revive Energy?</h1><p>Past-only family and commodity weights, tested across trailing horizons and refit cadences against the frozen daily-alpha execution archive.</p></header><main>
<section><h2>Answer</h2><div class="card warn"><strong>Not robustly yet.</strong> The fast 63-session / five-session family refit improves the <em>2026 energy sleeve</em>, but underperforms the fixed energy reference over the full replay, loses in 2025, and degrades at doubled costs. Slower fits do not rescue it. A separate asset-edge tilt has little incremental effect. The four-asset portfolio is especially hard to judge recently because exact-contract minute fills are missing on many intended trades; August 2026 has no fully priced dynamic portfolio session.</div>
<div class="scroll">{formatted_table(headline, ['Model','net_sr','stress_sr','sr_2025','sr_2026','Coverage %','priced_days','Aug-Oct observed return %','late_priced_days','common_sr_delta'], {'net_sr':'Net SR','stress_sr':'2x-cost SR','sr_2025':'2025 SR','sr_2026':'2026 SR','priced_days':'Priced days','late_priced_days':'Aug-Oct priced days','common_sr_delta':'SR difference on days also priced by fixed book'})}</div>
<p>Fixed carry/OI: {fixed.net_sr:.2f} net SR, {100*fixed.coverage:.0f}% priced. Fast weekly family fit: {fast.net_sr:.2f} net SR, {100*fast.coverage:.0f}% priced. The predeclared slower 252/21 fit: {slow.net_sr:.2f} net SR, {100*slow.coverage:.0f}% priced. The 252/21 asset tilt changes the fast fit from {fast.net_sr:.2f} to {fast_tilt.net_sr:.2f} and the slow fit from {slow.net_sr:.2f} to {slow_tilt.net_sr:.2f}. The best fast-family asset tilt in the full grid is an ex-post observation, not a deployable setting.</p>
<figure><img src="data:image/png;base64,{images['equity']}"><figcaption>Full 20 June 2024 to 22 October 2026 history. The shaded first 63 sessions use a common equal-active-family warmup; after that each rolling line uses only previous returns. Lines break at unpriced active days. The displayed path compounds observed returns only and is not a complete investable history.</figcaption></figure></section>
<section><h2>Energy Diagnosis</h2><div class="cols"><div class="card"><h3>What improved</h3><p>On {common_energy_days} jointly priced 2026 energy sessions, the fast model's net SR is {common_fast_energy_sr:.2f} versus {common_fixed_energy_sr:.2f} for the fixed carry/OI energy sleeve. Across August-October, both energy series price 52 sessions: fast gains {100*fast_energy.late_growth:.3f}% versus {100*fixed_energy.late_growth:.3f}% fixed. The improvement is mainly NG; CL's fast fit trails its fixed leg. The fast fit has shifted CL principally toward carry-confirmed momentum and NG toward carry-basis reversal and volume-spike continuation rather than the now-silent volume-spike-reversion vote.</p></div>
<div class="card"><h3>What did not</h3><p>Full-period energy net SR is {fast_energy.net_sr:.2f} fast versus {fixed_energy.net_sr:.2f} fixed; the fast strategy's doubled-cost SR is {fast_energy.stress_sr:.2f} versus {fixed_energy.stress_sr:.2f}. Its 2025 energy SR is {fast_energy.sr_2025:.2f}, while the fixed sleeve is {fixed_energy.sr_2025:.2f}. The slow 252/21 energy fit is weaker still ({slow_energy.net_sr:.2f} net SR). Thus the recent recovery cannot be presented as a stable remedy.</p></div></div>
<figure><img src="data:image/png;base64,{images['recent']}"><figcaption>Recent energy view only. The volume-spike line is flat chiefly because its archived votes became inactive, not because missing returns were filled with zero.</figcaption></figure>
<figure><img src="data:image/png;base64,{images['selection']}"><figcaption>Selection counts are pre-entry decisions on sessions with a known current daily vote; two families can appear on one session. They do not show profitable trades or confirm causal alpha discovery.</figcaption></figure>
<details><summary>Asset-by-asset return and fill breakdown</summary><p>Every row uses the same archived exact-contract intraday path. Observed-only returns are not comparable without allowing for differences in priced sessions and active-trade coverage. Fast GC has 20 intended active August-October sessions and zero complete fills; its return is unknown, not zero.</p><div class="scroll">{formatted_table(assets, ['Asset','Model','Full net SR','2025 SR','2026 SR','Aug-Oct observed return %','Aug-Oct planned trades','Aug-Oct priced sessions','Active fill coverage %'])}</div></details>
<details><summary>Family selection counts for CL and NG</summary><div class="scroll">{formatted_table(chosen_late, ['root','Family','selected_2025','selected_2026','selected_aug_oct_2026','mean_weight_2026'], {'root':'Asset','selected_2025':'2025 selected sessions','selected_2026':'2026 selected sessions','selected_aug_oct_2026':'Aug-Oct selected sessions','mean_weight_2026':'Average 2026 family weight'})}</div></details></section>
<section><h2>How the Rolling Test Works</h2><p>Each of 39 non-silver asset/family candidates retains its archived vote treatment, entry clock, and hold. Its candidate-level trade P&amp;L includes the original micro-futures commission (USD 2.16 per round trip) and one-tick-per-side slippage. At each family refit, only prior realized returns within the requested lookback are scored. A candidate must have at least eight priced active days, at least 75% recent intended-trade coverage, positive full- and recent-half net mean, and a positive doubled-cost mean. Positive net Sharpe is shrunk for sparse activity. On the trading day, up to two eligible families <em>with an already-known vote</em> share each asset's budget in proportion to their past scores; if none qualifies, the asset holds cash. The first 63 sessions use equal weights across active family votes for every model.</p>
<p>The primary sweep fixes 25% notional budget per commodity and changes family lookback/refit timing only. A second sweep tilts the commodity budgets using prior asset P&amp;L, but keeps half of the budget equally allocated, sums weights to one, and never adds leverage. It does not instantly reallocate a silent asset's budget to another commodity. Missing exact-contract prices are never treated as zero P&amp;L. Future-P&amp;L perturbation tests confirm that both family and asset weights before the perturbation date remain identical. Importantly, the frozen price cache queried only five-minute timestamp samples from the underlying one-minute tables; its forward search accepts the next positive-volume sampled bar within ten minutes, but cannot see off-grid minutes.</p>
<div class="card warn"><strong>Selection boundary:</strong> this is causal <em>allocation conditional on an earlier hindsight-selected candidate set</em>. The candidate vote treatment, regional clock and holding period were selected on the same 2024-2026 history in the earlier audit. Scanning this lookback/cadence grid for its best result would add another layer of hindsight. Sharpe ratios are conditional on priced sessions; the normalized equity curves are not dollar-VaR or executable lot forecasts.</div></section>
<section><h2>Family Lookback and Cadence</h2><p>The full 4 x 4 sweep varies a 63-, 126-, 252- or 504-session trailing window and 5-, 10-, 21- or 63-session refits. Longer windows expand from available history until full length. Every strategy uses the same 63-session warmup and up to two families per commodity.</p>
<figure><img src="data:image/png;base64,{images['families']}"><figcaption>Cells show observed-day net SR and the percentage of intended active portfolio/energy sessions that could be priced. Higher SR paired with lower coverage is not automatically better.</figcaption></figure>
<details><summary>All 16 four-asset family configurations</summary><div class="scroll">{formatted_table(family_grid, ['Model','net_sr','stress_sr','sr_2025','sr_2026','Coverage %','common_days','common_sr_delta','Aug-Oct observed return %','late_priced_days'], {'net_sr':'Net SR','stress_sr':'2x-cost SR','sr_2025':'2025 SR','sr_2026':'2026 SR','common_days':'Matched with fixed','common_sr_delta':'Paired SR difference','late_priced_days':'Aug-Oct priced days'})}</div></details>
<details><summary>All 16 energy-only family configurations</summary><div class="scroll">{formatted_table(energy_grid, ['Model','net_sr','stress_sr','sr_2025','sr_2026','Coverage %','common_days','common_sr_delta','Aug-Oct observed return %','late_priced_days'], {'net_sr':'Net SR','stress_sr':'2x-cost SR','sr_2025':'2025 SR','sr_2026':'2026 SR','common_days':'Matched with fixed energy','common_sr_delta':'Paired SR difference','late_priced_days':'Aug-Oct priced days'})}</div></details></section>
<section><h2>Asset Reweighting</h2><p>Applying 63-, 126-, 252- and 504-session asset lookbacks at 5-, 21- and 63-session refits to both the fast and slow family models barely changes net Sharpe. In particular, it cannot recover the lost coverage from unpriced GC/HG contracts, and it does not create an energy edge by itself. The plot shows changes relative to equal commodity allocations, not a comparison with the fixed books.</p>
<figure><img src="data:image/png;base64,{images['assets']}"><figcaption>Change in net SR from asset tilting after the family fit. The displayed 252/21 asset-tilt examples were predeclared; the best cell in this matrix is not selected for deployment.</figcaption></figure>
<details><summary>All 24 asset-tilt configurations</summary><div class="scroll">{formatted_table(asset_grid, ['Model','net_sr','stress_sr','sr_2025','sr_2026','Coverage %','common_days','common_sr_delta'], {'net_sr':'Net SR','stress_sr':'2x-cost SR','sr_2025':'2025 SR','sr_2026':'2026 SR','common_days':'Matched with fixed','common_sr_delta':'Paired SR difference'})}</div></details></section>
<section><h2>Recent Pricing Boundary</h2><p>Observed-only monthly returns are omitted when zero sessions can be priced. The portfolio's August 2026 row illustrates the problem: the fast and slow rolling books each have no fully priced session, despite 21 session dates. By contrast, both energy-only variants price all 21 August sessions. Portfolio-level apparent flatness or recovery during these gaps cannot be inferred from the chart.</p>
<p>The month-level mismatch is concrete in this frozen snapshot: the daily GC target switches to December 2026 on 3 August, but sampled MGCZ6 bars first appear on 28 August at 05:05 UTC; HG switches to December 2026 on 31 August, while sampled MHGZ6 bars first appear on 29 September at 05:05 UTC. This demonstrates an execution-cache coverage problem, not that the daily OI or curve vote was absent. It does not establish whether the full one-minute database or a later patch contains those bars.</p>
<div class="scroll">{formatted_table(months, ['Month','Sleeve','Model','Observed return %','Priced sessions','Calendar sessions'])}</div>
<p>On the same 387 priced days, the fast weekly allocator has SR {weekly_common_sr:.2f} versus {one_common_sr:.2f} for a one-time fit at the end of warmup. That supports the usefulness of updating <em>relative to no update</em>, but the fast allocator still trails the fixed hindsight-selected carry/OI reference on its common priced days. The result therefore answers a narrower question than whether a deployable, fully causal strategy has been found.</p></section>
<section><h2>Decision</h2><p>Do not select or shadow-deploy the 63/5 rule from this replay yet. First reconcile the selected delivery month against the full one-minute bar archive, rebuild the fill cube without five-minute subsampling, and rerun the fixed and rolling books with the same fill policy. Keep asset-level P&amp;L and incomplete portfolio days visible rather than deleting all other independently timed trades. Only then assess whether the energy improvement persists; no increase in live risk is supported here. This respects the broader limitations of hypothetical futures performance discussed by the <a href="https://www.nfa.futures.org/rulebooksql/rules.aspx?RuleID=9025&amp;Section=9">NFA</a>.</p></section>
</main></body></html>"""
    (OUT / "rolling_intraday_edge_report.html").write_text(html, encoding="utf-8")
    print(f"[report] {OUT / 'rolling_intraday_edge_report.html'}")
    print(f"[check] fixed {fixed.net_sr:.3f}, fast {fast.net_sr:.3f}, "
          f"slow {slow.net_sr:.3f}; energy fast {fast_energy.net_sr:.3f}")


if __name__ == "__main__":
    run()
