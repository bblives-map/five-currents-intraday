"""Replay each exit horizon independently with an explicit execution-lot cost basis."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_intraday_replay import (
    HORIZONS, ROOTS, MICROS, MICRO_COMMISSION_ROUND_TRIP, MULTIPLIERS, SLIPPAGE, daily_paths, recost_matches,
    reverse_paths, summarize, usable_bar,
)


def coverage_reasons(audit, bars):
    exact, product = {}, set()
    for row in bars.itertuples():
        exact.setdefault((row.root, row.contract_key, row.ts), []).append(row)
        product.add((row.root, row.ts))
    rows = []
    for row in audit.loc[audit["mode"].eq("target")].itertuples():
        entries = exact.get((row.root, row.contract_key, row.entry_utc), [])
        if row.status == "matched":
            reason = "matched"
        elif pd.isna(row.contract_key):
            reason = "unknown signal contract identity"
        elif not entries:
            reason = ("required delivery month not stored at entry" if (row.root, row.entry_utc) in product
                      else "no stored root bar at entry")
        elif not any(usable_bar(entry) for entry in entries):
            reason = "entry bar has no observed trade"
        else:
            exits = exact.get((row.root, row.contract_key, row.exit_utc), [])
            own = [bar for bar in exits if bar.symbol == row.symbol and bar.table == row.table]
            reason = "exit bar has no observed trade" if own else "selected source has no exit bar"
        rows.append({"session": row.session, "root": row.root, "horizon": row.horizon, "reason": reason})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("five_currents_intraday_exit_20261009_matched"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-basis", choices=("micro", "standard"), default="micro")
    args = parser.parse_args()
    source = args.source.resolve()
    old_manifest = json.loads((source / "manifest.json").read_text())
    inputs = ("execution_audit.parquet", "cached_minute_endpoints.parquet", "asset_daily_returns.parquet")
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in inputs}
    if any(hashes[name] != old_manifest["output_sha256"][name] for name in inputs):
        raise ValueError("Saved source audit checksum mismatch")
    old = pd.read_parquet(source / "execution_audit.parquet")
    bars = pd.read_parquet(source / "cached_minute_endpoints.parquet")
    old_paths = pd.read_parquet(source / "asset_daily_returns.parquet")
    audit = recost_matches(old, execution_basis=args.execution_basis)
    assert audit.status.equals(old.status)
    np.testing.assert_allclose(audit.gross_return, old.gross_return, equal_nan=True)
    matched = audit.loc[audit.status.eq("matched")]
    old_matched = old.loc[old.status.eq("matched")]
    fee = MICRO_COMMISSION_ROUND_TRIP if args.execution_basis == "micro" else 5.50
    quantities_per_capital = matched.weight.abs() / (matched.entry_price * matched.execution_multiplier)
    expected = quantities_per_capital * (fee + 2 * matched.root.map(SLIPPAGE) * matched.execution_multiplier)
    np.testing.assert_allclose(matched.cost_return, expected, atol=1e-14)
    np.testing.assert_allclose(matched.commission_return, quantities_per_capital * fee, atol=1e-14)
    quantity_pnl = (matched.weight / (matched.entry_price * matched.execution_multiplier)
                    * matched.execution_multiplier * (matched.exit_price - matched.entry_price))
    np.testing.assert_allclose(matched.gross_return, quantity_pnl, atol=1e-14)
    assert matched.commission_usd_per_round_trip_lot.eq(fee).all()
    paths = daily_paths(audit, sorted(old_paths.session.unique()))
    np.testing.assert_allclose(paths.gross_return, old_paths.gross_return, equal_nan=True)
    assert paths.ready.equals(old_paths.ready)
    original_metrics, original_curves = summarize(paths)
    inverse = reverse_paths(paths)
    opposite_metrics, opposite_curves = summarize(inverse)
    for frame, direction in ((original_metrics, "original"), (opposite_metrics, "opposite"),
                             (original_curves, "original"), (opposite_curves, "opposite")):
        frame["direction"] = direction
    metrics = pd.concat([original_metrics, opposite_metrics], ignore_index=True)
    curves = pd.concat([original_curves, opposite_curves], ignore_index=True)
    assert original_metrics.days.equals(opposite_metrics.days)
    np.testing.assert_allclose(original_metrics.gross_sharpe, -opposite_metrics.gross_sharpe, equal_nan=True)
    original_dates = original_curves[["mode", "book", "horizon", "session"]]
    pd.testing.assert_frame_equal(original_dates, opposite_curves[list(original_dates.columns)])

    out = args.output_dir.resolve()
    out.mkdir(exist_ok=False)
    audit.to_parquet(out / "execution_audit.parquet", index=False)
    paths.to_parquet(out / "asset_daily_returns.parquet", index=False)
    metrics.to_csv(out / "summary_metrics.csv", index=False)
    curves.to_parquet(out / "equity_curves.parquet", index=False)
    calendar = paths.copy()
    calendar["opposite_net_return"] = -calendar.gross_return - calendar.cost_return
    calendar["commission_only_net_return"] = calendar.gross_return - calendar.commission_return
    calendar["opposite_commission_only_net_return"] = -calendar.gross_return - calendar.commission_return
    calendar.to_csv(out / "full_calendar_asset_returns.csv", index=False)
    reasons = coverage_reasons(audit, bars)
    reasons.to_csv(out / "coverage_by_day.csv", index=False)
    reason_counts = reasons.groupby(["root", "horizon", "reason"]).size().rename("days").reset_index()
    reason_counts.to_csv(out / "coverage_reasons.csv", index=False)
    cost_legs = matched[["session", "mode", "root", "horizon", "symbol", "entry_price", "weight",
                         "execution_symbol", "execution_multiplier", "commission_usd_per_round_trip_lot",
                         "quantity_per_unit_capital", "slippage_usd_per_round_trip_lot", "commission_return",
                         "slippage_return", "cost_return"]].copy()
    cost_legs["previous_cost_return"] = old_matched.cost_return
    cost_legs.to_csv(out / "cost_ledger.csv", index=False)

    target = metrics.loc[metrics["mode"].eq("target")]
    comparison = target.loc[target.book.eq("portfolio_ex_silver")].pivot(
        index=["horizon", "days"], columns="direction",
        values=["gross_sharpe", "commission_only_sharpe", "net_sharpe", "stressed_sharpe"])
    comparison.columns = [f"{kind}_{direction}" for kind, direction in comparison.columns]
    comparison = comparison.reset_index()
    print(f"INDEPENDENT HOLDS / {args.execution_basis.upper()} LOTS / ${fee:.2f} COMMISSION ROUND TRIP", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print("INDEPENDENT MATCHED DAYS / ORIGINAL NET POSITIONS", flush=True)
    counts = target.loc[target.direction.eq("original")].pivot(index="book", columns="horizon", values="days")
    print(counts.to_string(), flush=True)
    print("COVERAGE REASONS AT 120 MINUTES", flush=True)
    print(reason_counts.loc[reason_counts.horizon.eq(120)].to_string(index=False), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    calendar_dates = pd.DatetimeIndex(sorted(paths.session.unique()))
    for direction, axis in zip(("original", "opposite"), axes):
        for color, horizon in zip(plt.cm.viridis(np.linspace(.1, .9, len(HORIZONS))), HORIZONS):
            one = curves.loc[curves["mode"].eq("target") & curves.book.eq("portfolio_ex_silver")
                             & curves.direction.eq(direction) & curves.horizon.eq(horizon)]
            # A missing execution remains a visible gap, never an invented flat return.
            values = one.set_index("session").equity.reindex(calendar_dates)
            axis.plot(calendar_dates, values, color=color, label=f"{horizon} min")
        axis.set_title(direction.title() + " net positions / observed-trade equity", loc="left")
        axis.set_ylabel("Growth of 100")
        axis.grid(alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=7, frameon=False, fontsize=8)
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=150, bbox_inches="tight", facecolor="#faf9f4")
    plt.close(fig)
    image = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()
    asset_display = target.loc[target.book.isin(ROOTS)].copy()
    asset_display.loc[asset_display.days.lt(20), ["gross_sharpe", "commission_only_sharpe", "net_sharpe", "stressed_sharpe"]] = np.nan
    symbols = {root: MICROS[root] if args.execution_basis == "micro" else root for root in ROOTS}
    dollars = pd.DataFrame([{"product": root, "execution_symbol": symbols[root], "commission_round_trip": fee,
                            "slippage_round_trip": 2 * SLIPPAGE[root] * MULTIPLIERS[symbols[root]],
                            "total_round_trip_per_lot": fee + 2 * SLIPPAGE[root] * MULTIPLIERS[symbols[root]]}
                           for root in ROOTS])
    lot_examples = pd.DataFrame([{"execution_symbol": symbols[root], "lots": lots,
                                 "commission_round_trip_per_lot_usd": fee,
                                 "total_commission_usd": lots * fee}
                                for root in ROOTS for lots in (1, 3)])
    lot_examples.to_csv(out / "lot_commission_examples.csv", index=False)
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Five Currents / Corrected Intraday Audit</title>
<style>body{{font:15px/1.6 Georgia,serif;color:#193c36;background:#faf9f4;margin:0}}main{{max-width:1150px;margin:auto;padding:40px}}img{{width:100%}}table{{width:100%;border-collapse:collapse;font:11px/1.5 Consolas,monospace}}th,td{{padding:8px;text-align:right;border-bottom:1px solid #d9ded1}}th{{background:#e7ede3}}.scroll{{overflow:auto}}</style></head><body><main>
<h1>Independent Intraday Holds / Corrected Costs</h1>
<p>Execution basis: <b>{args.execution_basis} lots</b>. Commission is <b>${fee:.2f} per lot for the entire round trip</b>, charged once. No standard-equivalent fee conversion and no extra multiplier of 10 is applied. Each holding period is an independent entry and exit.</p>
<p>Signals, economic exposure weights, exact delivery months, observed entry/exit prices and US DST timing remain unchanged. Lot quantities and cash P&amp;L use the same {args.execution_basis} contract multiplier. The complete cached calendar is retained; no refits, market API calls, database access or production changes.</p>
<h2>Cost Per {args.execution_basis.title()} Lot / USD</h2>{dollars.to_html(index=False, border=0, float_format=lambda value: f'{value:.2f}')}
<p>For one lot, commission is ${fee:.2f}; for three lots it is ${3*fee:.2f}. The per-lot fee is not multiplied by the standard/micro size ratio. The existing adverse price allowances remain separate from commission and unchanged in price units. The table distinguishes commission-only Sharpe from net Sharpe including that additional modeled slippage. These allowances are assumptions, not measured fills.</p>
<h2>Full Net Positions / Four-Asset Diagnostic</h2><div class="scroll">{comparison.to_html(index=False, border=0, float_format=lambda value: f'{value:.3f}')}</div><img src="{image}" alt="Original and opposite net positions on the full cached calendar">
<p>CL, NG, GC and HG retain their original weights; excluded silver weight is not redistributed. Each row uses every complete date available for that horizon only. Unknown outcomes remain missing, so the lines show compounded observed outcomes with gaps rather than a claimed continuous executable track record. Comparisons between horizons can use different dates. Annualized Sharpe uses 252 observations.</p>
<h2>Dates Available / Each Independent Holding Period</h2>{counts.to_html(border=0)}
<p>The cached calendar contains {paths.session.nunique()} sessions from {calendar_dates.min().date()} through {calendar_dates.max().date()}. Historical minute coverage for a root does not imply that all the delivery months selected by the daily model were stored. No wrong-month prices or unobserved synthetic execution bars are substituted to manufacture full coverage. Silver's exact-month coverage remains inadequate.</p>
<h2>Asset Results / Both Directions</h2><div class="scroll">{asset_display[['direction','book','horizon','days','start','end','gross_sharpe','commission_only_sharpe','net_sharpe','stressed_sharpe','total_return','max_drawdown']].to_html(index=False, border=0, na_rep='n/a', float_format=lambda value: f'{value:.4f}')}</div>
<h2>Missingness / 120-Minute Example</h2>{reason_counts.loc[reason_counts.horizon.eq(120)].to_html(index=False, border=0)}
<p>All dates, costs and exclusion reasons are retained in accompanying CSV/parquet files. Full target and target-change experiments are separate, not combined. Opposite-direction testing follows observation of the original results; it is exploratory, not independent validation. Historical publication times are incomplete, so prior-close signal availability by 06:00/07:00 UTC remains conditional.</p>
</main></body></html>'''
    report = out / "Five_Currents_Corrected_Intraday_Report.html"
    report.write_text(html, encoding="utf-8")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
                "source_sha256": hashes, "sample": "independent per holding period",
                "execution_contracts": args.execution_basis, "commission_per_lot_round_trip_usd": fee,
                "fee_standard_equivalent_conversion": False,
                "slippage_price_allowance_per_side": {root: SLIPPAGE[root] for root in ROOTS},
                "price_source_changes": False,
                "signal_changes": False, "refits": False, "database_access": False, "api_calls": False,
                "quantity_dollar_cost_reconciliation": True, "missing_outcomes_imputed": False,
                "output_sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                  for file in out.iterdir() if file.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[report] " + str(report), flush=True)


if __name__ == "__main__":
    main()
