"""Cached-only four-market ablation of the frozen daily-signal intraday test."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_front_intraday import evaluate, front_month_moves, stats

MARKETS = ("CL", "NG", "GC", "HG")
RETURNS = ("gross_return", "commission_return", "slippage_return", "net_return", "slippage_net_return")


def daily_portfolio(paths):
    keys = ["variant", "hours", "session"]
    if paths.duplicated(keys + ["root"]).any():
        raise ValueError("Duplicate asset outcomes")
    grouped = paths.groupby(keys)
    result = grouped[list(RETURNS)].sum(min_count=1)
    result["ready"] = grouped.ready.all() & grouped.root.nunique().eq(len(MARKETS))
    result.loc[~result.ready, list(RETURNS)] = np.nan
    return result.reset_index()


def run(source, comparison, output):
    output.mkdir(exist_ok=False)
    bars = pd.read_parquet(source / "front_minute_endpoints.parquet")
    predictors = pd.read_parquet(source / "daily_predictors.parquet")
    predictors = predictors.loc[predictors.root.isin(MARKETS)].copy()
    assert predictors.signal_date.lt(predictors.session).all()
    sessions = pd.DatetimeIndex(predictors.session.unique()).sort_values()
    moves = front_month_moves(bars.loc[~bars.symbol.isin(["SI", "SIL"])], sessions)
    moves = moves.loc[moves.root.isin(MARKETS)].copy()
    paths = evaluate(predictors, moves)
    original_paths = pd.read_parquet(source / "strategy_asset_returns.parquet")
    original_paths = original_paths.loc[original_paths.root.isin(MARKETS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(paths.reset_index(drop=True), original_paths)
    totals = daily_portfolio(paths)
    metrics, yearly = [], []
    for (variant, hours), group in totals.groupby(["variant", "hours"]):
        good = group.loc[group.ready]
        metrics.append({"variant": variant, "hours": hours,
                        "start": good.session.min(), "end": good.session.max(),
                        "possible_days": len(group), "coverage_pct": group.ready.mean()*100.,
                        "gross_sharpe": stats(group.gross_return)["sharpe"],
                        "slippage_net_sharpe": stats(group.slippage_net_return)["sharpe"],
                        **{("net_sharpe" if k == "sharpe" else k): v
                           for k, v in stats(group.net_return).items()}})
        for year, subset in group.groupby(group.session.dt.year):
            yearly.append({"variant": variant, "hours": hours, "year": year, **stats(subset.net_return)})
    metrics = pd.DataFrame(metrics)
    reference = pd.read_csv(source / "summary_metrics.csv")
    reference = reference.loc[reference.book.eq("portfolio_ex_silver")].sort_values(["variant", "hours"])
    ordered = metrics.sort_values(["variant", "hours"])
    for column in ("days", "net_sharpe", "total_return", "max_drawdown", "slippage_net_sharpe"):
        np.testing.assert_allclose(reference[column], ordered[column], equal_nan=True, rtol=1e-10, atol=1e-12)
    totals["equity"] = totals.groupby(["variant", "hours"]).net_return.transform(lambda r: (1.+r).cumprod()*100.)
    metrics.to_csv(output / "summary_metrics.csv", index=False)
    pd.DataFrame(yearly).to_csv(output / "yearly_metrics.csv", index=False)
    totals.to_parquet(output / "daily_portfolio_and_equity.parquet", index=False)
    paths.to_parquet(output / "strategy_asset_returns.parquet", index=False)
    moves.to_parquet(output / "price_moves.parquet", index=False)

    common = []
    five = pd.read_parquet(comparison / "silver_second_month_proxy" / "equity_curves.parquet")
    five = five.loc[five.book.eq("portfolio")]
    for (variant, hours), group in totals.groupby(["variant", "hours"]):
        control = five.loc[five.variant.eq(variant) & five.hours.eq(hours)]
        shared = group.set_index("session").net_return.to_frame("four").join(
            control.set_index("session").net_return.rename("five"), how="inner").dropna()
        for name in ("four", "five"):
            common.append({"book": name, "variant": variant, "hours": hours, **stats(shared[name])})
    pd.DataFrame(common).to_csv(output / "same_date_four_vs_five.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "markets": MARKETS,
        "market_api_calls": False, "database_access": False, "model_refits": False,
        "production_changes": False, "original_four_asset_paths_verified": True,
        "weight_policy": "Drop SI only; no redistribution. Sign remains +/-0.2 per asset; magnitude unchanged.",
        "entry": "01:00 America/Chicago (06:00 UTC summer, 07:00 UTC winter)",
        "exit_hours": [1, 2, 3, 4, 6, 12], "commission_usd_per_micro_round_trip": 2.16,
        "missing_policy": "Each horizon independent; active missing prices remain unknown, not cash.",
        "input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [source / "front_minute_endpoints.parquet", source / "daily_predictors.parquet",
                                   source / "strategy_asset_returns.parquet", source / "summary_metrics.csv",
                                   comparison / "silver_second_month_proxy" / "equity_curves.parquet", Path(__file__)]},
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in output.iterdir() if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[four markets] net Sharpe after commissions\n" + metrics.pivot(index="variant", columns="hours", values="net_sharpe").round(3).to_string())
    print("[observations]\n" + metrics.pivot(index="variant", columns="hours", values="days").to_string())
    print("[output] " + str(output.resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("five_currents_front_intraday_price_edge_20261010"))
    parser.add_argument("--comparison", type=Path, default=Path("five_currents_silver_reconciled_replay_20261011"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.comparison, args.output_dir)
