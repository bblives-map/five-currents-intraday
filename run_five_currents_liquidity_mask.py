"""Offline execution-only liquidity mask for the frozen Five Currents book.

Reads one archived dashboard snapshot. No database, broker, or market API access.
"""
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
import numpy as np
import pandas as pd

from five_currents_daily.dashboard_core import dollar_cost_execution
from five_currents_daily.data import coverage_from_wide
from five_currents_daily.reconciliation import (
    MarkBook, calendar_backtest, repair_forward_marks, with_stored_expiries,
)
from five_currents_daily.report import _series_metrics
from run_us_commodity_liquidity_led_roll import (
    build_product_inputs, causal_product_specific_ranks,
)
from us_commodity_asset_family_walkforward_core import product_net_returns
from us_commodity_disciplined_core import fast_product_backtest
from us_commodity_execution_overlays import (
    contract_sequence, delivery_period, expiry_session_calendar, last_trade_date,
)
from us_commodity_front_roll_core import (
    build_front_roll_inputs, build_listed_contract_execution, install_execution_ranks,
)
from us_commodity_research_core import (
    PRODUCT_ORDER, _field_matrix, _mapping_matrix, build_panel,
)


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
SNAPSHOT = ROOT / "five_currents_dashboard/snapshots/000037_2026-10-12"
ARCHIVED_RANKS = ROOT / "us_commodity_liquidity_led_roll_20260814/liquidity_execution_ranks.parquet"
OUTPUT = ROOT / "five_currents_liquidity_mask_20261013"
CUTOFF = pd.Timestamp("2026-07-20")
ADV_SESSIONS = 5
MIN_ADV_OBSERVATIONS = 3
EXPIRY_BUFFER_SESSIONS = 2
VARIANTS = {"incumbent": (), "silver_copper": ("SI", "HG"),
            "silver_copper_gold": ("SI", "HG", "GC")}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_liquidity(panel, wide, mapping_long, root: str) -> pd.DataFrame:
    index = panel.index
    mapping = _mapping_matrix(mapping_long, panel.metadata, root, index)
    volume = _field_matrix(wide, root, "PX_VOLUME").reindex(index)
    oi = _field_matrix(wide, root, "OPEN_INT").reindex(index)
    settle = _field_matrix(wide, root, "PX_SETTLE").reindex(index)
    last = _field_matrix(wide, root, "PX_LAST").reindex(index)
    price = settle.where(settle.gt(0), last)
    parts = []
    for rank in range(1, 7):
        if rank not in mapping:
            continue
        parts.append(pd.DataFrame({"date": index, "root": root, "rank": rank,
                                   "contract": mapping[rank].to_numpy(),
                                   "volume": volume[rank].to_numpy(float),
                                   "oi": oi[rank].to_numpy(float),
                                   "price": price[rank].to_numpy(float)}))
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.loc[frame.contract.notna()].copy()
    if frame.duplicated(["date", "contract"]).any():
        raise AssertionError(f"A listed {root} contract appears at two ranks on the same day")
    frame = frame.sort_values(["contract", "date", "rank"])
    frame["lagged_adv"] = frame.groupby("contract", sort=False).volume.transform(
        lambda values: values.shift(1).rolling(ADV_SESSIONS, min_periods=MIN_ADV_OBSERVATIONS).mean()
    )
    frame["lagged_oi"] = frame.groupby("contract", sort=False).oi.shift(1)

    sessions = expiry_session_calendar(index, boundary_days=500)
    session_pos = pd.Series(np.arange(len(sessions)), index=sessions)
    records = mapping_long.loc[mapping_long.ticker.str.match(fr"^{root}[1-6] Comdty$")].copy()
    records["expiry"] = pd.to_datetime(records.expiry, errors="coerce")
    expiry_lookup = records.set_index(["date", "ticker"]).expiry
    official_cache = {}
    days = []
    for row in frame.itertuples():
        stamped_expiry = expiry_lookup.get((row.date, f"{root}{row.rank} Comdty"), pd.NaT)
        if pd.notna(stamped_expiry):
            expiry = pd.Timestamp(stamped_expiry).normalize()
        else:
            key = row.contract
            if key not in official_cache:
                sequence = contract_sequence(key)
                official_cache[key] = (
                    last_trade_date(root, delivery_period(root, sequence), sessions)
                    if sequence is not None else pd.NaT
                )
            expiry = official_cache[key]
        date = pd.Timestamp(row.date).normalize()
        if pd.isna(expiry) or date not in session_pos:
            days.append(np.nan)
        else:
            days.append(float(sessions.searchsorted(expiry) - session_pos[date]))
    frame["sessions_to_expiry"] = days
    frame["eligible"] = (frame.price.gt(0) & frame.lagged_adv.gt(0) &
                         frame.sessions_to_expiry.gt(EXPIRY_BUFFER_SESSIONS))
    return frame


def selected_ranks(panel, wide, mapping, incumbent):
    eligible_frames = [contract_liquidity(panel, wide, mapping, root)
                       for root in ("SI", "HG", "GC")]
    ledger = pd.concat(eligible_frames, ignore_index=True)
    chosen = ledger.loc[ledger.eligible].sort_values(
        ["root", "date", "lagged_adv", "lagged_oi", "rank"],
        ascending=[True, True, False, False, True],
    ).drop_duplicates(["root", "date"])
    selected = incumbent.copy()
    for root in ("SI", "HG", "GC"):
        current = chosen.loc[chosen.root.eq(root)].set_index("date")
        selected.loc[current.index, root] = current["rank"].astype(float)
    return selected, ledger, chosen


def realised_path(panel, wide, mapping, marks, ranks, weights, first_causal):
    execution_panel = install_execution_ranks(panel, wide, mapping, ranks)
    execution = dollar_cost_execution(build_listed_contract_execution(execution_panel))
    array = weights.to_numpy(float)
    paths = {scenario: fast_product_backtest(array, execution_panel, execution, scenario)
             for scenario in ("base", "stressed")}
    products = pd.DataFrame(product_net_returns(array, execution, scenario="base"),
                            index=panel.index, columns=PRODUCT_ORDER)
    calendar, calendar_products = calendar_backtest(array, panel.index, execution, marks, "base")
    calendar_stress, _ = calendar_backtest(array, panel.index, execution, marks, "stressed")
    dates = calendar.loc[first_causal:].index

    def aligned(values):
        source = values.reindex(panel.index).copy()
        source.loc[source.index < first_causal] = np.nan
        return source.shift(2).reindex(dates)

    base = calendar.reindex(dates)
    stressed = calendar_stress.reindex(dates)
    contributions = calendar_products.reindex(dates)
    frozen = dates <= CUTOFF
    base.loc[frozen] = aligned(paths["base"].net_return).loc[frozen]
    stressed.loc[frozen] = aligned(paths["stressed"].net_return).loc[frozen]
    contributions.loc[frozen, PRODUCT_ORDER] = aligned(products).loc[frozen, PRODUCT_ORDER]
    if not np.allclose(contributions.sum(axis=1, min_count=1).loc[base.notna()],
                       base.loc[base.notna()], atol=1e-10, rtol=0):
        raise AssertionError("Asset contributions do not sum to portfolio net returns")
    return pd.DataFrame({"base": base, "stressed": stressed}), contributions, paths


def metrics(values: pd.Series) -> dict:
    result = _series_metrics(values)
    return {key: (str(value.date()) if isinstance(value, pd.Timestamp) else value)
            for key, value in result.items()}


def plot_equity(returns: pd.DataFrame, destination: Path) -> None:
    palette = {"incumbent": "#557467", "silver_copper": "#b67634",
               "silver_copper_gold": "#184c61"}
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#f5f2e9")
    for ax in axes:
        ax.set_facecolor("#fffdf7")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    for name in VARIANTS:
        values = returns[name]
        equity = (1 + values.fillna(0)).cumprod()
        axes[0].plot(equity.index, equity.where(values.notna()), label=name.replace("_", " "),
                     color=palette[name], lw=1.6)
        dd = equity / equity.cummax() - 1
        axes[1].plot(dd.index, 100 * dd.where(values.notna()), color=palette[name], lw=1.2)
    axes[0].set_title("Five Currents | frozen signals, alternative execution contract", loc="left", fontsize=15)
    axes[0].set_ylabel("Growth of 1, net")
    axes[1].set_ylabel("Drawdown, %")
    axes[0].legend(ncol=3, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(destination, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    wide = pd.read_parquet(SNAPSHOT / "input_panel.parquet")
    mapping = pd.read_parquet(SNAPSHOT / "input_mapping.parquet")
    marks = pd.read_parquet(SNAPSHOT / "exact_valuation_marks.parquet")
    weights = pd.read_parquet(SNAPSHOT / "weights.parquet").reindex(columns=PRODUCT_ORDER)
    client = json.loads((SNAPSHOT / "client.json").read_text(encoding="utf-8"))
    first_causal = pd.Timestamp(client["first_causal_date"])

    print("[panel] building archived six-rank product panel", flush=True)
    panel = build_panel(wide, mapping, coverage_from_wide(wide))
    repaired = repair_forward_marks(panel, mapping, marks, CUTOFF)
    front = build_front_roll_inputs(panel, wide, mapping)
    front = with_stored_expiries(front, mapping, MarkBook(marks, panel.index).calendar)
    print("[incumbent] reconstructing causal liquidity ranks", flush=True)
    baseline, _, _ = causal_product_specific_ranks(
        panel, front, build_product_inputs(panel, wide, mapping)
    )
    archived = pd.read_parquet(ARCHIVED_RANKS)
    common = baseline.index.intersection(archived.index)
    comparable = baseline.loc[common, PRODUCT_ORDER].to_numpy(float)
    reference = archived.loc[common, PRODUCT_ORDER].to_numpy(float)
    if not np.array_equal(np.isfinite(comparable), np.isfinite(reference)) or not np.allclose(
        comparable, reference, equal_nan=True, atol=0, rtol=0
    ):
        raise AssertionError("Incumbent ranks do not match archived causal ranks")

    print("[mask] selecting prior-known most-liquid metal contracts", flush=True)
    masked, ledger, chosen = selected_ranks(panel, wide, mapping, baseline)
    rank_paths = {"incumbent": baseline,
                  "silver_copper": baseline.assign(SI=masked.SI, HG=masked.HG),
                  "silver_copper_gold": baseline.assign(SI=masked.SI, HG=masked.HG, GC=masked.GC)}
    for name, ranks in rank_paths.items():
        ranks.to_parquet(OUTPUT / f"{name}_execution_ranks.parquet")
    returns, stress, contributions, paths = {}, {}, {}, {}
    for name, ranks in rank_paths.items():
        print(f"[replay] {name}", flush=True)
        result, asset, path = realised_path(panel, wide, mapping, marks, ranks, weights, first_causal)
        returns[name], stress[name], contributions[name], paths[name] = result.base, result.stressed, asset, path
    portfolio = pd.DataFrame(returns)
    stressed = pd.DataFrame(stress)
    dashboard = pd.Series(client["history"]["returns"],
                          index=pd.to_datetime(client["history"]["dates"]), dtype=float)
    overlap = dashboard.index.intersection(portfolio.index)
    discrepancies = (dashboard.reindex(overlap) - portfolio.incumbent.reindex(overlap)).abs()
    baseline_error = float(discrepancies.max())
    if baseline_error > 1e-9:
        raise AssertionError(f"Replay does not match published dashboard: {baseline_error:.4g}")
    print(f"[validation] published dashboard max absolute return error {baseline_error:.3g}", flush=True)

    valid = portfolio.notna().all(axis=1)
    common_returns = portfolio.loc[valid]
    common_stress = stressed.loc[valid]
    rows = []
    for name in VARIANTS:
        for basis, frame in (("net", common_returns), ("double_cost", common_stress)):
            rows.append({"case": name, "basis": basis, **metrics(frame[name]),
                         "missing_sessions": int(portfolio[name].isna().sum()),
                         "common_sessions": int(valid.sum())})
    metrics_frame = pd.DataFrame(rows)
    year_rows = []
    for year, frame in common_returns.groupby(common_returns.index.year):
        for name in VARIANTS:
            year_rows.append({"year": year, "case": name, **metrics(frame[name])})
    asset_rows = []
    for name, frame in contributions.items():
        for root in PRODUCT_ORDER:
            asset_rows.append({"case": name, "asset": root,
                               "sum_net_contribution": float(frame.loc[valid, root].sum()),
                               "missing_sessions": int(frame[root].isna().sum())})
    rank_rows = []
    for root in ("SI", "HG", "GC"):
        selected = rank_paths["silver_copper_gold"][root]
        old = baseline[root]
        active = weights[root].abs().gt(1e-12)
        rank_rows.append({"asset": root, "signal_days_changed": int(selected.ne(old).sum()),
                          "active_signal_days_changed": int((selected.ne(old) & active).sum()),
                          "selected_rank_1": int(selected.eq(1).sum()),
                          "selected_rank_2": int(selected.eq(2).sum()),
                          "selected_rank_3_plus": int(selected.ge(3).sum())})

    portfolio.to_csv(OUTPUT / "daily_net_returns.csv", index_label="realised_date")
    stressed.to_csv(OUTPUT / "daily_double_cost_returns.csv", index_label="realised_date")
    metrics_frame.to_csv(OUTPUT / "metrics_common_dates.csv", index=False)
    pd.DataFrame(year_rows).to_csv(OUTPUT / "yearly_common_dates.csv", index=False)
    pd.DataFrame(asset_rows).to_csv(OUTPUT / "asset_contributions.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(OUTPUT / "rank_changes.csv", index=False)
    chosen.to_csv(OUTPUT / "chosen_contracts.csv", index=False)
    forecast_rows = []
    for name, scenarios in paths.items():
        for scenario, path in scenarios.items():
            valid_forecasts = path.loc[path.index >= first_causal].dropna(subset=["net_return"])
            forecast_rows.append({"case": name, "basis": scenario,
                                  "gross_return_sum": float(valid_forecasts.gross_return.sum()),
                                  "cost_sum": float(valid_forecasts.cost.sum()),
                                  "turnover_sum": float(valid_forecasts.turnover.sum()),
                                  "median_turnover": float(valid_forecasts.turnover.median()),
                                  "forecast_observations": len(valid_forecasts)})
    forecast_frame = pd.DataFrame(forecast_rows)
    forecast_frame.to_csv(OUTPUT / "forecast_cost_turnover.csv", index=False)
    plot_equity(common_returns, OUTPUT / "equity_and_drawdown.png")
    chart = base64.b64encode((OUTPUT / "equity_and_drawdown.png").read_bytes()).decode("ascii")
    table = metrics_frame[["case", "basis", "observations", "annualized_return", "sharpe", "max_drawdown", "total_return", "missing_sessions"]].copy()
    for column in ("annualized_return", "max_drawdown", "total_return"):
        table[column] = table[column].map(lambda value: f"{value:.2%}")
    table["sharpe"] = table.sharpe.map(lambda value: f"{value:.3f}")
    yearly_table = pd.DataFrame(year_rows)[["year", "case", "annualized_return", "sharpe"]].copy()
    yearly_table["annualized_return"] = yearly_table.annualized_return.map(lambda value: f"{value:.2%}")
    yearly_table["sharpe"] = yearly_table.sharpe.map(lambda value: f"{value:.2f}")
    asset_table = pd.DataFrame(asset_rows).pivot(index="asset", columns="case", values="sum_net_contribution")
    asset_table = asset_table.map(lambda value: f"{value:+.2%}").reset_index()
    (OUTPUT / "report.html").write_text(
        "<html><meta charset='utf-8'><title>Five Currents | Liquidity Mask</title>"
        "<style>body{font:16px Georgia,sans-serif;max-width:1100px;margin:40px auto;color:#173d3a;background:#f5f2e9}"
        "h1{font-size:36px}p{line-height:1.5}img{width:100%}table{border-collapse:collapse;background:#fffdf7;width:100%}"
        "td,th{padding:9px;border-bottom:1px solid #d8d5ca;text-align:right}td:first-child,th:first-child{text-align:left}</style>"
        "<h1>Five Currents | Execution-contract liquidity mask</h1>"
        "<p>Offline frozen-signal counterfactual, 2020-2026. SI/HG and optional GC select the highest "
        "prior-five-observation average volume among eligible archived ranks 1-6. OI breaks ties. "
        "Two-session expiry buffer, listed-contract returns, exact roll legs, incumbent portfolio weights and "
        "dollar-corrected model costs are retained. All metrics use the dates observed in every case; "
        "missing returns are never filled as zero.</p>"
        + table.to_html(index=False, escape=True)
        + f"<img alt='Full history net equity and drawdown' src='data:image/png;base64,{chart}'>"
        + "<h2>Year by year</h2>" + yearly_table.to_html(index=False, escape=True)
        + "<h2>Net asset contribution</h2>" + asset_table.to_html(index=False, escape=True)
        + "<h2>Gross return, model cost and turnover</h2>"
        + forecast_frame.to_html(index=False, escape=True)
        + "<p>Contract choices and all daily returns are in the adjacent CSV files. "
        "Historical simulations are not proof of executable liquidity; bid-ask depth was not tested.</p></html>",
        encoding="utf-8",
    )
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "snapshot": str(SNAPSHOT), "snapshot_client_sha256": file_hash(SNAPSHOT / "client.json"),
                "database_writes": False, "database_reads": False, "market_api_calls": False,
                "signals_recomputed": False, "weights_retrained": False,
                "policy": {"adv_sessions": ADV_SESSIONS, "min_prior_observations": MIN_ADV_OBSERVATIONS,
                           "expiry_buffer_sessions": EXPIRY_BUFFER_SESSIONS, "oi_tie_break_only": True},
                "baseline_dashboard_max_abs_error": baseline_error,
                "repaired_forward_marks": repaired,
                "output_hashes": {p.name: file_hash(p) for p in OUTPUT.iterdir() if p.is_file()}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nMETRICS (common observed dates):\n", metrics_frame.to_string(index=False), flush=True)
    print("\nRANK CHANGES:\n", pd.DataFrame(rank_rows).to_string(index=False), flush=True)
    print("\nCOST AND TURNOVER:\n", forecast_frame.to_string(index=False), flush=True)
    print("\nOUTPUT", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
