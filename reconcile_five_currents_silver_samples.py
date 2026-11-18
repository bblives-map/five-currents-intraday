"""Replay frozen intraday signals with separately identified cached silver sources.

This never edits a database, obtains market data, or fits a model. A different
silver delivery month is a price-prediction proxy, not an exact SIL repair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_front_intraday import (
    HOURS, entry_time, evaluate, front_month_moves, stats, summarize,
)


def silver_proxy_endpoints(original, archive, sessions):
    """Use a fixed archive, never choose a month from future returns or exits."""
    requested = {entry_time(day) + pd.Timedelta(hours=h)
                 for day in sessions for h in (0, *HOURS)}
    silver = archive.loc[
        archive.source_table.eq("bars_1m_2m") & archive.symbol.eq("SIL")
        & archive.ts.isin(requested)
    ].copy()
    if silver.duplicated(["symbol", "ts"]).any():
        raise ValueError("Ambiguous silver archive timestamp")
    silver["source_method"] = "fixed_second_month_archive_price_proxy"
    other = original.loc[~original.symbol.isin(["SI", "SIL"])].copy()
    other["source_method"] = "unchanged_original_front_series"
    result = pd.concat([other, silver], ignore_index=True)
    if result.duplicated(["symbol", "ts"]).any():
        raise ValueError("Endpoint identity must remain unique")
    return result


def common_date_comparison(paths_by_case):
    rows = []
    original = paths_by_case["original"]
    for case, paths in paths_by_case.items():
        if case == "original":
            continue
        for (variant, hours), base in original.groupby(["variant", "hours"]):
            changed = paths.loc[paths.variant.eq(variant) & paths.hours.eq(hours)]
            for book in ("portfolio", "SI"):
                left = base if book == "portfolio" else base.loc[base.root.eq(book)]
                right = changed if book == "portfolio" else changed.loc[changed.root.eq(book)]
                left_ready = left.groupby("session").ready.all()
                right_ready = right.groupby("session").ready.all()
                dates = left_ready.index[left_ready & right_ready.reindex(left_ready.index).fillna(False)]
                for label, data in (("original", left), (case, right)):
                    pnl = data.groupby("session").net_return.sum(min_count=1).reindex(dates)
                    rows.append({"comparison_case": case, "case": label, "variant": variant,
                                 "hours": hours, "book": book, **stats(pnl)})
    return pd.DataFrame(rows)


def run(source, audit, output):
    output.mkdir(exist_ok=False)
    original = pd.read_parquet(source / "front_minute_endpoints.parquet")
    predictors = pd.read_parquet(source / "daily_predictors.parquet")
    sessions = pd.DatetimeIndex(predictors.session.unique()).sort_values()
    if not predictors.signal_date.lt(predictors.session).all():
        raise ValueError("Predictors must be available before the entry date")
    archive = pd.read_parquet(audit / "existing_silver_nearby_minutes.parquet")
    proxy = silver_proxy_endpoints(original, archive, sessions)
    proxy.to_parquet(output / "silver_second_month_proxy_endpoints.parquet", index=False)
    cases = {
        "original": original,
        "same_month_exact_repair": pd.read_parquet(audit / "front_endpoints_exact_repairs.parquet"),
        "silver_second_month_proxy": proxy,
    }
    paths_by_case, all_metrics, all_coverage, all_moves = {}, [], [], []
    for case, endpoints in cases.items():
        moves = front_month_moves(endpoints, sessions)
        paths = evaluate(predictors, moves)
        metrics, curves, yearly = summarize(paths)
        paths_by_case[case] = paths
        case_dir = output / case
        case_dir.mkdir()
        paths.to_parquet(case_dir / "strategy_asset_returns.parquet", index=False)
        curves.to_parquet(case_dir / "equity_curves.parquet", index=False)
        yearly.to_csv(case_dir / "yearly_metrics.csv", index=False)
        metrics["case"] = case
        all_metrics.append(metrics)
        silver = moves.loc[moves.root.eq("SI")]
        for hours, group in silver.groupby("hours"):
            matched = group.loc[group.status.eq("matched")]
            all_coverage.append({"case": case, "hours": hours, "calendar_weekdays": len(group),
                                 "matched_silver_days": len(matched),
                                 "both_endpoints_trade_confirmed": int(
                                     (matched.entry_trade_confirmed & matched.exit_trade_confirmed).sum()),
                                 "first_match": matched.session.min(), "last_match": matched.session.max()})
        moves["case"] = case
        all_moves.append(moves)
        print(f"[replay] {case}: complete, with original signals and micro fees", flush=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(output / "summary_metrics.csv", index=False)
    coverage = pd.DataFrame(all_coverage)
    coverage.to_csv(output / "silver_coverage.csv", index=False)
    moves = pd.concat(all_moves, ignore_index=True)
    moves.to_parquet(output / "price_moves_with_source_case.parquet", index=False)
    common_date_comparison(paths_by_case).to_csv(output / "common_date_metrics.csv", index=False)

    baseline = paths_by_case["original"].loc[lambda x: x.root.ne("SI")].reset_index(drop=True)
    for case, paths in paths_by_case.items():
        pd.testing.assert_frame_equal(baseline, paths.loc[paths.root.ne("SI")].reset_index(drop=True))
    old = pd.read_csv(source / "summary_metrics.csv").sort_values(["variant", "hours", "book"])
    replay = metrics.loc[metrics.case.eq("original")].sort_values(["variant", "hours", "book"])
    for column in ("days", "net_sharpe", "total_return", "max_drawdown"):
        np.testing.assert_allclose(old[column], replay[column], equal_nan=True, rtol=1e-10, atol=1e-12)

    inputs = [source / "front_minute_endpoints.parquet", source / "daily_predictors.parquet",
              source / "summary_metrics.csv", audit / "existing_silver_nearby_minutes.parquet",
              audit / "front_endpoints_exact_repairs.parquet", Path(__file__)]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "market_api_calls": False, "database_writes": False, "model_refits": False,
        "collector_or_dashboard_changes": False, "original_replay_verified": True,
        "other_four_markets_unchanged_verified": True,
        "proxy_policy": "Fixed bars_1m_2m SIL series, chosen independently of P&L and exit availability; "
                        "exact timestamp opens only, same actual contract within each trade. "
                        "Different month is a research price proxy, never an exact first-month repair.",
        "missing_policy": "No interpolation, forward/back fill, cash substitution, or future exit-source selection.",
        "fee_policy": "$2.16 per micro-lot round trip, unchanged; proxy is not proof of executable SIL fills.",
        "calendar_policy": "Original weekday axis retained; holidays and absent prices remain missing, not zero.",
        "input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "output_sha256": {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in output.rglob("*") if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(coverage.to_string(index=False), flush=True)
    selected = metrics.loc[metrics.book.eq("portfolio") & metrics.case.isin(["original", "silver_second_month_proxy"])]
    for case, group in selected.groupby("case"):
        print(f"\n[{case}] net Sharpe\n" + group.pivot(index="variant", columns="hours", values="net_sharpe").round(3).to_string(), flush=True)
        print(f"[{case}] portfolio observations\n" + group.pivot(index="variant", columns="hours", values="days").to_string(), flush=True)
    print("[output] " + str(output.resolve()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("five_currents_front_intraday_price_edge_20261010"))
    parser.add_argument("--audit", type=Path, default=Path("five_currents_silver_reconciliation_20261011"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.audit, args.output_dir)


if __name__ == "__main__":
    main()
