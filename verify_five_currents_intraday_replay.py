"""Reconcile an existing local intraday study without API or database access."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_intraday_replay import HORIZONS, MULTIPLIERS, ROOTS, SLIPPAGE, entry_time
from run_five_currents_intraday_replay import make_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("--refresh-report", action="store_true")
    args = parser.parse_args()
    folder = args.study.resolve()
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
    source = Path(manifest["source_snapshot"]) / "weights.parquet"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["weights_sha256"]
    frozen = pd.read_parquet(folder / "frozen_weights.parquet")
    pd.testing.assert_frame_equal(frozen, pd.read_parquet(source))

    audit = pd.read_parquet(folder / "execution_audit.parquet")
    paths = pd.read_parquet(folder / "asset_daily_returns.parquet")
    metrics = pd.read_csv(folder / "summary_metrics.csv")
    curves = pd.read_parquet(folder / "equity_curves.parquet")
    observed = audit.loc[audit.status.eq("matched")]
    gross = observed.weight * (observed.exit_price - observed.entry_price) / observed.entry_price
    # Historical issued studies retain their original accounting for audit.
    execution_symbol = observed.execution_symbol if "execution_symbol" in observed else observed.symbol
    fees = execution_symbol.map(lambda symbol: 2.75 if symbol in ROOTS else 1.60)
    costs = observed.weight.abs() * (
        2 * execution_symbol.map(SLIPPAGE) + 2 * fees / execution_symbol.map(MULTIPLIERS)
    ) / observed.entry_price
    for actual, expected in ((observed.gross_return, gross), (observed.cost_return, costs),
                             (observed.net_return, gross - costs),
                             (observed.stressed_return, gross - 2 * costs)):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-14)
    assert audit.signal_date.lt(audit.session).all()
    assert audit.entry_utc.eq(audit.session.map(entry_time)).all()
    assert audit.exit_utc.eq(audit.entry_utc + pd.to_timedelta(audit.horizon, unit="m")).all()
    assert observed.entry_volume.gt(0).all() and observed.exit_volume.gt(0).all()
    choice = observed.groupby(["mode", "session", "root", "contract_key"])[["symbol", "table"]].nunique()
    assert choice.le(1).all().all()
    missing = paths.loc[paths.active & ~paths.ready]
    assert missing[["gross_return", "cost_return", "net_return", "stressed_return"]].isna().all().all()
    aggregate = observed.groupby(["mode", "session", "root", "horizon"])[
        ["gross_return", "cost_return", "net_return", "stressed_return"]
    ].sum()
    complete = paths.loc[paths.active & paths.ready].set_index(aggregate.index.names)
    np.testing.assert_allclose(complete[aggregate.columns], aggregate.reindex(complete.index), atol=1e-14)
    require_shared_dates = metrics["sample"].eq("all-horizons matched").all()
    for (_, book), group in curves.groupby(["mode", "book"]):
        dates = [tuple(one.session) for _, one in group.groupby("horizon")]
        assert len(dates) == len(HORIZONS)
        if require_shared_dates:
            assert all(day_set == dates[0] for day_set in dates)
        for _, one in group.groupby("horizon"):
            np.testing.assert_allclose(one.equity, (1 + one.net_return).cumprod() * 100)
            roots = ROOTS if book == "portfolio" else tuple(root for root in ROOTS if root != "SI") if book == "portfolio_ex_silver" else (book,)
            expected = paths.loc[paths["mode"].eq(one["mode"].iloc[0]) & paths.root.isin(roots)
                                 & paths.horizon.eq(one.horizon.iloc[0]) & paths.session.isin(one.session)]
            daily = expected.groupby("session").net_return.sum()
            np.testing.assert_allclose(one.net_return, daily.reindex(one.session), atol=1e-14)

    if args.refresh_report:
        manifest["database_access"] = "local endpoint parquet; no PVX reads"
        make_report(folder, metrics, curves, paths, audit, manifest)
    html = (folder / "Five_Currents_Intraday_Exit_Report.html").read_text(encoding="utf-8")
    assert html.count("data:image/png;base64,") == 3
    verification = {"verified_utc": datetime.now(timezone.utc).isoformat(),
                    "frozen_weights_unchanged": True, "independent_trade_accounting": True,
                    "strict_prior_close_signals": True, "us_dst_timing": True,
                    "missing_returns_not_imputed": True, "same_dates_all_horizons": require_shared_dates,
                    "matched_legs": len(observed), "embedded_charts": 3,
                    "new_api_calls": False, "database_access": False}
    (folder / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    manifest["output_sha256"] = {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                                 for item in folder.iterdir() if item.is_file() and item.name != "manifest.json"}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))
    print(metrics.loc[metrics.book.eq("portfolio_ex_silver"),
                      ["mode", "horizon", "days", "net_sharpe", "total_return"]].to_string(index=False))


if __name__ == "__main__":
    main()
