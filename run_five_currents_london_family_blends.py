"""London-open blends of three frozen signal families, using cached contract bars."""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from audit_five_currents_cached_replay import MULTIPLIER, TICK_COST, independent_stats, leg_values
from five_currents_opening_study import daily_totals
from report_five_currents_cached_window_replay import fmt_table


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
SOURCE = ROOT / "five_currents_uniform_active_intraday_20261014"
OUT = ROOT / "five_currents_london_family_blends_20261014"
BOOKS = {"four": ("CL", "NG", "GC", "HG"), "five": ("CL", "NG", "GC", "SI", "HG")}
HOURS = (1, 2, 3, 4, 6, 12)
SPLIT = pd.Timestamp("2025-07-01")
RETURNS = ("gross_return", "commission_return", "slippage_return", "net_return", "slippage_net_return")
LABELS = {"A_two": "Carry/basis + curve quality", "B_three": "Core pair + reversed dryup change",
          "B_mixed": "Core 12h + reversed dryup change 6h"}
COLORS = {"CL": "#bd6a35", "NG": "#658261", "GC": "#a88732", "HG": "#3c7c88", "SI": "#7d8587"}


def signal_components(predictors: pd.DataFrame) -> pd.DataFrame:
    keys = ["session", "signal_date", "root"]
    names = {"carry_basis_reversal": ("net_magnitude", "basis"),
             "curve_quality_reversal": ("net_magnitude", "quality"),
             "volume_dryup_continuation": ("change_magnitude", "dryup_change")}
    frames = [predictors.loc[predictors.signal.eq(name), keys + [field]].rename(columns={field: label})
              .set_index(keys) for name, (field, label) in names.items()]
    result = pd.concat(frames, axis=1).reset_index()
    if result[["basis", "quality", "dryup_change"]].isna().any().any() or result.duplicated(keys).any():
        raise AssertionError("Family forecasts are missing, duplicated, or differently dated")
    if not result.signal_date.lt(result.session).all():
        raise AssertionError("A family forecast uses current-session information")
    result["A_two"] = (result.basis + result.quality) / 2
    result["B_three"] = (result.basis + result.quality - result.dryup_change) / 3
    result["mixed_core"] = (result.basis + result.quality) / 3
    result["mixed_dryup"] = -result.dryup_change / 3
    return result


def fixed_exit_legs(components: pd.DataFrame, moves: pd.DataFrame, portfolio: str) -> pd.DataFrame:
    joined = moves.merge(components, on=["session", "root"], suffixes=("_price", "_signal"), validate="many_to_one")
    if not joined.signal_date_price.eq(joined.signal_date_signal).all():
        raise AssertionError("Price and forecast source dates differ")
    weight = joined[portfolio].to_numpy(float)
    inactive = np.abs(weight) <= 1e-12
    ready = joined.status.eq("matched").to_numpy() | inactive
    entry = joined.entry_price.to_numpy(float)
    price_return = joined.exit_price.to_numpy(float) / entry - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        gross = np.where(inactive, 0., weight * price_return)
        contracts = np.abs(weight) / (entry * joined.root.map(MULTIPLIER).to_numpy(float))
        commission = np.where(inactive, 0., contracts * 2.16)
        slippage = np.where(inactive, 0., np.abs(weight) * 2 * joined.root.map(TICK_COST).to_numpy(float) / entry)
    legs = joined[["session", "root", "hours", "signal_date_price", "identity_basis", "status",
                   "trade_pair_confirmed"]].rename(columns={"signal_date_price": "signal_date"})
    legs["portfolio"], legs["weight"], legs["ready"] = portfolio, weight, ready
    for name, values in (("gross_return", gross), ("commission_return", commission), ("slippage_return", slippage)):
        legs[name] = np.where(ready, values, np.nan)
    legs["net_return"] = legs.gross_return - legs.commission_return
    legs["slippage_net_return"] = legs.net_return - legs.slippage_return
    # An independent evaluator checks the blend's reversal and single netted round trip.
    reference_input = joined.copy()
    reference_input["net_magnitude"] = weight
    reference_input["clock"] = "london_open"
    reference = leg_values(reference_input, "net_magnitude")
    for field in RETURNS:
        np.testing.assert_allclose(legs[field], reference[field], equal_nan=True, rtol=0, atol=1e-12)
    return legs


def mixed_exit_legs(components: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    fields = ["session", "root", "entry_price", "exit_price", "status", "entry_ts", "local_symbol",
              "trade_pair_confirmed", "identity_basis"]
    six = moves.loc[moves.hours.eq(6), fields]
    twelve = moves.loc[moves.hours.eq(12), fields]
    joined = components.merge(six, on=["session", "root"], validate="one_to_one").merge(
        twelve, on=["session", "root"], suffixes=("_6", "_12"), validate="one_to_one")
    both_entry = joined.entry_price_6.notna() & joined.entry_price_12.notna()
    if not joined.loc[both_entry, "entry_ts_6"].eq(joined.loc[both_entry, "entry_ts_12"]).all():
        raise AssertionError("Mixed horizons have different entry timestamps")
    if not joined.loc[both_entry, "local_symbol_6"].eq(joined.loc[both_entry, "local_symbol_12"]).all():
        raise AssertionError("Mixed horizons change selected contract")
    np.testing.assert_allclose(joined.loc[both_entry, "entry_price_6"], joined.loc[both_entry, "entry_price_12"], rtol=0, atol=0)
    core, dryup = joined.mixed_core.to_numpy(float), joined.mixed_dryup.to_numpy(float)
    entry = joined.entry_price_12.fillna(joined.entry_price_6).to_numpy(float)
    active_core, active_dryup = np.abs(core) > 1e-12, np.abs(dryup) > 1e-12
    ready = ((joined.status_12.eq("matched").to_numpy() | ~active_core)
             & (joined.status_6.eq("matched").to_numpy() | ~active_dryup))
    with np.errstate(divide="ignore", invalid="ignore"):
        gross = (np.where(active_core, core * (joined.exit_price_12.to_numpy(float) / entry - 1), 0.)
                 + np.where(active_dryup, dryup * (joined.exit_price_6.to_numpy(float) / entry - 1), 0.))
        traded_notional = np.abs(core + dryup) + np.abs(dryup) + np.abs(core)
        commission = traded_notional * 1.08 / (entry * joined.root.map(MULTIPLIER).to_numpy(float))
        slippage = traded_notional * joined.root.map(TICK_COST).to_numpy(float) / entry
    idle = ~active_core & ~active_dryup
    legs = joined[["session", "signal_date", "root"]].copy()
    legs["hours"], legs["portfolio"] = 12, "B_mixed"
    legs["weight"] = core + dryup
    legs["peak_component_exposure"] = np.abs(core) + np.abs(dryup)
    legs["ready"] = ready
    legs["identity_basis"] = joined.identity_basis_12
    legs["status"] = np.where(ready, "matched or inactive", "missing required timed exit")
    legs["trade_pair_confirmed"] = ((joined.trade_pair_confirmed_12.eq(True) | ~active_core)
                                     & (joined.trade_pair_confirmed_6.eq(True) | ~active_dryup))
    for name, values in (("gross_return", gross), ("commission_return", commission), ("slippage_return", slippage)):
        legs[name] = np.where(ready, np.where(idle, 0., values), np.nan)
    legs["net_return"] = legs.gross_return - legs.commission_return
    legs["slippage_net_return"] = legs.net_return - legs.slippage_return
    return legs


def portfolio_paths(legs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (portfolio, hours), group in legs.groupby(["portfolio", "hours"], sort=False):
        for book, roots in BOOKS.items():
            values = daily_totals(group, roots)
            if portfolio == "B_mixed":
                values["gross_exposure"] = group.loc[group.root.isin(roots)].groupby("session").peak_component_exposure.sum()
            values = values.reset_index().assign(portfolio=portfolio, hours=int(hours), book=book)
            frames.append(values)
    return pd.concat(frames, ignore_index=True)


def metrics(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (portfolio, book, hours), group in paths.groupby(["portfolio", "book", "hours"]):
        group = group.sort_values("session")
        for period, mask in (("full", np.ones(len(group), dtype=bool)),
                             ("earlier", group.session.lt(SPLIT).to_numpy()),
                             ("later", group.session.ge(SPLIT).to_numpy())):
            values = group.loc[mask]
            for basis in ("gross_return", "slippage_net_return"):
                rows.append({"portfolio": portfolio, "book": book, "hours": int(hours), "period": period,
                             "basis": basis, "possible_days": len(values),
                             "active_days": int((values.ready & values.gross_exposure.gt(1e-12)).sum()),
                             **independent_stats(values[basis])})
    return pd.DataFrame(rows)


def pair_paths(paths: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["session", "book", "hours"]
    a = paths.loc[paths.portfolio.eq("A_two")]
    b = paths.loc[paths.portfolio.eq("B_three")]
    paired = a[keys + ["slippage_net_return"]].merge(b[keys + ["slippage_net_return"]],
                on=keys, suffixes=("_A", "_B"), validate="one_to_one")
    valid = paired.slippage_net_return_A.notna() & paired.slippage_net_return_B.notna()
    paired.loc[~valid, ["slippage_net_return_A", "slippage_net_return_B"]] = np.nan
    rows = []
    for (book, hours), group in paired.groupby(["book", "hours"]):
        for period, mask in (("full", np.ones(len(group), dtype=bool)),
                             ("earlier", group.session.lt(SPLIT).to_numpy()),
                             ("later", group.session.ge(SPLIT).to_numpy())):
            part = group.loc[mask].sort_values("session")
            for name in ("A", "B"):
                rows.append({"portfolio": "A_two" if name == "A" else "B_three", "book": book,
                             "hours": int(hours), "period": period,
                             **independent_stats(part[f"slippage_net_return_{name}"])})
    return paired, pd.DataFrame(rows)


def timing_comparison(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for book in BOOKS:
        part = paths.loc[paths.book.eq(book) & paths.hours.eq(12)]
        wide = part.pivot(index="session", columns="portfolio", values="slippage_net_return").dropna()
        for period, frame in (("full", wide), ("earlier", wide.loc[wide.index < SPLIT]),
                               ("later", wide.loc[wide.index >= SPLIT])):
            for name in ("A_two", "B_three", "B_mixed"):
                rows.append({"book": book, "period": period, "portfolio": name,
                             **independent_stats(frame[name])})
            rows.append({"book": book, "period": period, "portfolio": "core_two_thirds",
                         **independent_stats(frame.A_two * (2 / 3))})
    return pd.DataFrame(rows)


def asset_metrics(legs: pd.DataFrame, paths: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (portfolio, hours), group in legs.groupby(["portfolio", "hours"], sort=False):
        for book, roots in BOOKS.items():
            if portfolio == "B_mixed":
                valid_dates = paths.loc[paths.portfolio.eq(portfolio) & paths.book.eq(book) & paths.ready, "session"]
            else:
                valid_dates = paired.loc[paired.book.eq(book) & paired.hours.eq(hours)
                                         & paired.slippage_net_return_A.notna(), "session"]
            selected = group.loc[group.root.isin(roots) & group.session.isin(valid_dates)]
            pivot = selected.pivot(index="session", columns="root", values="slippage_net_return").reindex(columns=roots)
            source = paths.loc[paths.portfolio.eq(portfolio) & paths.book.eq(book) & paths.hours.eq(hours)].set_index("session")
            np.testing.assert_allclose(pivot.sum(axis=1, min_count=len(roots)),
                                       source.reindex(pivot.index).slippage_net_return, atol=1e-12, rtol=0)
            for root, values in pivot.items():
                data = selected.loc[selected.root.eq(root)]
                rows.append({"portfolio": portfolio, "book": book, "hours": int(hours), "root": root,
                             "net_contribution": float(values.sum()),
                             "gross_contribution": float(data.gross_return.sum()),
                             "commission_drag": float(data.commission_return.sum()),
                             "slippage_drag": float(data.slippage_return.sum()),
                             **independent_stats(values)})
    return pd.DataFrame(rows)


def png(fig, path: Path) -> str:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def equity_chart(paired: pd.DataFrame, paths: pd.DataFrame, portfolio: str, book: str) -> str:
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    fig.patch.set_facecolor("#f6f2e8")
    ax.set_facecolor("#fffdf8")
    colors = ("#b9902f", "#8d7862", "#56897e", "#c47653", "#344a69", "#166b5a")
    value_column = "slippage_net_return_A" if portfolio == "A_two" else "slippage_net_return_B"
    for hours, color in zip(HOURS, colors):
        part = paired.loc[paired.book.eq(book) & paired.hours.eq(hours)].sort_values("session").set_index("session")
        returns = part[value_column]
        ax.plot(returns.index, 100 * (1 + returns).cumprod(), color=color, lw=1.6,
                label=f"{hours}h ({int(returns.notna().sum())} dates)")
    if portfolio == "B_three":
        mixed = paths.loc[paths.book.eq(book) & paths.portfolio.eq("B_mixed")].sort_values("session").set_index("session")
        ax.plot(mixed.index, 100 * (1 + mixed.slippage_net_return).cumprod(), color="#282f2b", ls="--", lw=1.5,
                label=f"Core 12h / dryup 6h ({int(mixed.ready.sum())} dates)")
    ax.axvline(SPLIT, color="#8a968b", ls=":", lw=1)
    ax.set_xlim(paired.session.min(), paired.session.max())
    ax.set_ylabel("Growth of 100, modelled costs")
    suffix = "CL, NG, GC, HG" if book == "four" else "CL, NG, GC, SI, HG"
    ax.set_title(f"London open | {LABELS[portfolio]}\n{suffix}", loc="left", fontsize=14)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="best")
    fig.tight_layout()
    return png(fig, OUT / f"equity_{portfolio}_{book}.png")


def asset_chart(legs: pd.DataFrame, paired: pd.DataFrame, book: str) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8), sharex=True)
    fig.patch.set_facecolor("#f6f2e8")
    dates = paired.loc[paired.book.eq(book) & paired.hours.eq(12)].set_index("session")
    for ax, portfolio in zip(axes, ("A_two", "B_three")):
        ax.set_facecolor("#fffdf8")
        selected = legs.loc[legs.portfolio.eq(portfolio) & legs.hours.eq(12) & legs.root.isin(BOOKS[book])]
        values = selected.pivot(index="session", columns="root", values="slippage_net_return").reindex(dates.index)
        values.loc[dates.slippage_net_return_A.isna(), :] = np.nan
        for root in BOOKS[book]:
            ax.plot(values.index, 100 * values[root].cumsum(), color=COLORS[root], label=root, lw=1.5)
        ax.set_title(f"{LABELS[portfolio]} | 12-hour hold", loc="left", fontsize=13)
        ax.axhline(0, color="#8a968b", lw=.8)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=5, fontsize=9)
    axes[-1].set_xlim(paired.session.min(), paired.session.max())
    fig.supylabel("Cumulative asset contribution, percentage points")
    fig.tight_layout()
    return png(fig, OUT / f"asset_contributions_{book}.png")


def asset_horizon_chart(assets: pd.DataFrame, book: str) -> str:
    roots = BOOKS[book]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.7), sharex=True, layout="constrained")
    fig.patch.set_facecolor("#f6f2e8")
    cmap = LinearSegmentedColormap.from_list("returns", ["#a85238", "#fffdf8", "#166b5a"])
    for ax, portfolio in zip(axes, ("A_two", "B_three")):
        data = assets.loc[assets.book.eq(book) & assets.portfolio.eq(portfolio)].pivot(
            index="root", columns="hours", values="sharpe").reindex(index=roots, columns=HOURS)
        plot = ax.imshow(data.to_numpy(float), cmap=cmap, vmin=-3, vmax=3, aspect="auto")
        ax.set_yticks(range(len(roots)), roots)
        ax.set_xticks(range(len(HOURS)), [f"{hours}h" for hours in HOURS])
        ax.tick_params(axis="both", length=0, labelsize=11, labelbottom=True)
        ax.set_title(LABELS[portfolio], loc="left", fontsize=13)
        for row in range(len(roots)):
            for column in range(len(HOURS)):
                value = data.iloc[row, column]
                ax.text(column, row, f"{value:.2f}" if pd.notna(value) else "n/a", ha="center", va="center",
                        fontsize=12, color="white" if abs(value) > 2 else "#153f39")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(plot, ax=axes, shrink=.8, label="Net Sharpe on each portfolio's common dates")
    return png(fig, OUT / f"asset_horizon_sharpe_{book}.png")


def quality_metrics(legs: pd.DataFrame, paths: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (portfolio, hours), group in legs.groupby(["portfolio", "hours"], sort=False):
        for book, roots in BOOKS.items():
            selected = group.loc[group.root.isin(roots)].copy()
            exposure = selected.peak_component_exposure if portfolio == "B_mixed" else selected.weight.abs()
            selected["bar_ok"] = exposure.le(1e-12) | selected.trade_pair_confirmed.eq(True)
            selected["month_known"] = exposure.le(1e-12) | selected.identity_basis.eq("verified_month")
            flags = selected.groupby("session")[["bar_ok", "month_known"]].all()
            flags["active"] = exposure.groupby(selected.session).sum().gt(1e-12)
            source = paths.loc[paths.portfolio.eq(portfolio) & paths.hours.eq(hours)
                               & paths.book.eq(book)].set_index("session")
            if portfolio == "B_mixed":
                eligible = source.ready.copy()
            else:
                shared = paired.loc[paired.book.eq(book) & paired.hours.eq(hours)].set_index("session")
                eligible = shared.slippage_net_return_A.notna().reindex(source.index).fillna(False)
            for subset, condition in (("common comparison dates", eligible),
                                       ("active bars for held positions", eligible & flags.bar_ok),
                                       ("verified months for held positions", eligible & flags.month_known)):
                rows.append({"portfolio": portfolio, "book": book, "hours": int(hours), "subset": subset,
                             "trading_days": int((condition & flags.active).sum()),
                             **independent_stats(source.loc[condition, "slippage_net_return"])})
    return pd.DataFrame(rows)


def performance_table(common_metrics: pd.DataFrame, paths_metrics: pd.DataFrame, book: str) -> pd.DataFrame:
    primary = common_metrics.loc[common_metrics.book.eq(book) & common_metrics.period.eq("full"),
                                 ["portfolio", "hours", "days", "sharpe", "total_return", "max_drawdown"]]
    extra = paths_metrics.loc[paths_metrics.book.eq(book) & paths_metrics.portfolio.eq("B_mixed")
                              & paths_metrics.period.eq("full") & paths_metrics.basis.eq("slippage_net_return"),
                              ["portfolio", "hours", "days", "sharpe", "total_return", "max_drawdown"]]
    table = pd.concat([primary, extra], ignore_index=True)
    table.portfolio = table.portfolio.map(LABELS)
    table.columns = ["Portfolio", "Hours", "Days", "Net Sharpe", "Compounded return", "Max drawdown"]
    return table


def main() -> None:
    OUT.mkdir(exist_ok=True)
    predictors = pd.read_parquet(SOURCE / "dated_predictors.parquet")
    moves = pd.read_parquet(SOURCE / "price_moves.parquet")
    moves = moves.loc[moves.clock.eq("london_open")].drop(columns="clock")
    components = signal_components(predictors)
    components.to_parquet(OUT / "combined_forecasts.parquet", index=False)
    legs = pd.concat([fixed_exit_legs(components, moves, portfolio) for portfolio in ("A_two", "B_three")]
                     + [mixed_exit_legs(components, moves)], ignore_index=True)
    paths = portfolio_paths(legs)
    portfolio_metrics = metrics(paths)
    paired, common_metrics = pair_paths(paths)
    timed = timing_comparison(paths)
    assets = asset_metrics(legs, paths, paired)
    quality = quality_metrics(legs, paths, paired)
    legs.to_parquet(OUT / "asset_return_ledger.parquet", index=False)
    paths.to_parquet(OUT / "portfolio_daily_returns.parquet", index=False)
    paired.to_parquet(OUT / "matched_comparison_returns.parquet", index=False)
    portfolio_metrics.to_csv(OUT / "portfolio_metrics_available_dates.csv", index=False)
    common_metrics.to_csv(OUT / "portfolio_metrics_common_dates.csv", index=False)
    timed.to_csv(OUT / "timed_exit_comparison_common_dates.csv", index=False)
    incremental = []
    for (book, hours), frame in paired.groupby(["book", "hours"]):
        difference = frame.slippage_net_return_B - frame.slippage_net_return_A * (2 / 3)
        incremental.append({"book": book, "hours": int(hours), **independent_stats(difference)})
    pd.DataFrame(incremental).to_csv(OUT / "incremental_dryup_metrics.csv", index=False)
    assets.to_csv(OUT / "asset_metrics_common_dates.csv", index=False)
    quality.to_csv(OUT / "price_quality_metrics.csv", index=False)
    images = {book: {portfolio: equity_chart(paired, paths, portfolio, book) for portfolio in ("A_two", "B_three")}
              for book in BOOKS}
    contributions = {book: asset_chart(legs, paired, book) for book in BOOKS}
    horizon_assets = {book: asset_horizon_chart(assets, book) for book in BOOKS}
    sections = []
    for book, roots in BOOKS.items():
        table = performance_table(common_metrics, portfolio_metrics, book)
        asset_table = assets.loc[assets.book.eq(book) & assets.portfolio.isin(("A_two", "B_three")),
                                 ["portfolio", "hours", "root", "days", "sharpe", "net_contribution"]].copy()
        asset_table.portfolio = asset_table.portfolio.map(LABELS)
        asset_table.columns = ["Portfolio", "Hours", "Commodity", "Days", "Net Sharpe", "Net contribution"]
        split = common_metrics.loc[common_metrics.book.eq(book)].pivot(index=["portfolio", "hours"],
                               columns="period", values=["days", "sharpe"])
        split.columns = ["_".join(pair) for pair in split.columns]
        split = split.reset_index()
        title = "Four commodities, excluding silver" if book == "four" else "Five-commodity comparison"
        timed_table = timed.loc[timed.book.eq(book) & timed.period.eq("full"),
                                ["portfolio", "days", "sharpe", "total_return", "max_drawdown"]].copy()
        timed_table.portfolio = timed_table.portfolio.map({**LABELS, "core_two_thirds": "Core pair at two-thirds allocation"})
        timed_table.columns = ["Portfolio", "Days", "Net Sharpe", "Compounded return", "Max drawdown"]
        quality_table = quality.loc[quality.book.eq(book) & quality.hours.eq(12),
                                   ["portfolio", "subset", "days", "trading_days", "sharpe"]].copy()
        quality_table["sharpe"] = [f"{value:.2f}" if trades >= 20 and pd.notna(value) else "Too few trades"
                                    for value, trades in zip(quality_table.sharpe, quality_table.trading_days)]
        quality_table.portfolio = quality_table.portfolio.map(LABELS)
        quality_table.columns = ["Portfolio", "Price subset", "Days", "Trading days", "Net Sharpe"]
        sections.append(f"<h2>{title}</h2><div class='scroll'>{fmt_table(table, ('Compounded return', 'Max drawdown'))}</div>"
                        f"<img alt='Two-signal equity across holding periods' src='data:image/png;base64,{images[book]['A_two']}'>"
                        f"<img alt='Three-signal equity across holding periods' src='data:image/png;base64,{images[book]['B_three']}'>"
                        f"<h3>Commodity contributions at twelve hours</h3><img alt='Commodity contribution curves' src='data:image/png;base64,{contributions[book]}'>"
                        f"<h3>Commodity performance at every holding period</h3><img alt='Net Sharpe by commodity and hold' src='data:image/png;base64,{horizon_assets[book]}'>"
                        f"<h3>Same-date comparison of component exit timing</h3><div class='scroll'>{fmt_table(timed_table, ('Compounded return', 'Max drawdown'))}</div>"
                        f"<details><summary>Twelve-hour price activity and contract mapping</summary><div class='scroll'>{fmt_table(quality_table)}</div></details>"
                        f"<details><summary>Commodity statistics for every holding period</summary><div class='scroll'>{fmt_table(asset_table, ('Net contribution',))}</div></details>"
                        f"<details><summary>Earlier and later descriptive results</summary><div class='scroll'>{fmt_table(split)}</div></details>")
        print(title, table.to_string(index=False), flush=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "cached_only": True,
                "entry": "08:00 Europe/London, DST-aware", "source": str(SOURCE),
                "source_dates": {"first": str(moves.session.min().date()), "last": str(moves.session.max().date())},
                "signals_retuned": False, "portfolio_volatility_rescaled": False,
                "A_two": "0.5 * carry_basis_reversal net + 0.5 * curve_quality_reversal net",
                "B_three": "(carry_basis_reversal net + curve_quality_reversal net - volume_dryup_continuation change) / 3",
                "B_mixed": "same three weights; core pair exits 12h; reversed dryup change exits 6h",
                "fees": "$2.16 round trip per micro lot; $1.08 each side for timed partial exits",
                "slippage": "one tick per executed side, charged on netted position changes",
                "comparison_calendar": "A and B share available dates separately for each holding period",
                "missing_price_policy": "No substitution to another contract and no missing-return imputation",
                "audit": "Fixed exits match independent evaluator; commodity daily returns sum to portfolio",
                "input_hashes": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                                 for name in ("dated_predictors.parquet", "price_moves.parquet")}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Five Currents | London Signal Blends</title>
<style>body{{font:16px/1.5 Georgia,serif;background:#f6f2e8;color:#153f39;max-width:1100px;margin:auto;padding:32px}}h1{{font-size:36px;line-height:1.1}}h2{{margin-top:32px}}img{{width:100%;margin:14px 0}}table{{border-collapse:collapse;width:100%;font:13px/1.4 sans-serif;background:#fffdf8}}th,td{{padding:8px;border-bottom:1px solid #dfdbd1;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#e7ece4}}.scroll{{overflow-x:auto}}details{{padding:12px;border:1px solid #dfdbd1;margin:12px 0}}summary{{cursor:pointer;font-weight:bold}}.small{{font-size:13px;color:#65736a}}</style>
<p class="small">Internal cached-data study | 14 October 2026</p><h1>London Open: Two Signals, Then Three</h1>
<p>Enter at 08:00 London local time. Portfolio A averages the archived carry/basis-reversal and curve-quality-reversal position sizes. Portfolio B averages those two and the negative day-on-day position change of volume-dryup continuation. The existing asset scaling and causal family admission are retained; no alpha or allocation parameter is refit.</p>
<p class="small">These are equal shares of archived allocated positions, not equal standalone risk allocations. Combined positions are netted before commission and slippage. Holdings of 1, 2, 3, 4, 6 and 12 hours are independent tests, each with its own valid dates. A and B use the same dates within each holding period. The dashed three-signal curve also keeps the originally discussed 12h/12h/6h component exits, with costs on entry, the partial exit and the final exit.</p>
<p class="small">Execution uses the uniform nearby-liquidity mask and its selected contract month. Historical inferred delivery months remain estimates. Missing prices create chart gaps and are excluded from statistics. Raw model sizing is unchanged; no added leverage or portfolio volatility target is applied.</p>
<p class="small">Usable dates include genuine idle days when the archived families have no position. The price-quality tables distinguish those dates from days with actual exposure; the verified-month slices contain too few trading days for a reliable Sharpe estimate.</p>
{''.join(sections)}
<p><a href="portfolio_metrics_common_dates.csv">Portfolio statistics</a> &middot; <a href="asset_metrics_common_dates.csv">Commodity statistics for all holds</a> &middot; <a href="asset_return_ledger.parquet">Auditable commodity return ledger</a></p></html>'''
    (OUT / "report.html").write_text(html, encoding="utf-8")
    print("Report:", OUT / "report.html", flush=True)


if __name__ == "__main__":
    main()
