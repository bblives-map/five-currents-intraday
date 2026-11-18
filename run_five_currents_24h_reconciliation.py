"""Reconcile the published daily ledger, then test cached morning-to-morning bars."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_daily_reconciliation_audit import (
    independent_daily_ledger, compare_series, exact_daily_price_pair, settlement_window_controls,
)
from five_currents_intraday_replay import (
    ROOTS, MICROS, MULTIPLIERS, SLIPPAGE, HORIZONS, identity_key, signal_legs, match_bars,
    recost_matches, daily_paths, summarize, performance, reverse_paths,
)


def controls_from_daily_marks(audit, weights, execution, daily_ledger, marks, keys, first_causal_date):
    calendar = weights.index.union(pd.DatetimeIndex(marks.date.unique())).sort_values()
    lookup = {(pd.Timestamp(row.date), row.product, row.contract_key, row.basis):
              float(row.price)*(.01 if row.product == "HG" else 1.) for row in marks.itertuples()}
    cached_costs = daily_ledger.set_index(["signal_date", "root"]).cost_return
    rows, price_checks = [], []
    for i, day in enumerate(weights.index):
        for j, root in enumerate(ROOTS):
            pair = exact_daily_price_pair(lookup, day, root, keys.at[day, root], calendar)
            manual = pair[3]/pair[2]-1. if pair else np.nan
            cached = execution.target_return[i, j]
            price_checks.append({"signal_date": day, "root": root, "cached_target": cached,
                                 "independent_target": manual,
                                 "weight": weights.at[day, root],
                                 "causal_period": day >= first_causal_date,
                                 "difference": manual-cached if np.isfinite(manual) and np.isfinite(cached) else np.nan})
    price_checks = pd.DataFrame(price_checks)
    selected = audit.loc[audit.variant.isin(("exact_daily_contract", "exact_daily_contract_after_entry_close")) & audit["mode"].eq("target")
                         & audit.status.eq("matched")]
    index_position = {day: i for i, day in enumerate(weights.index)}
    for row in selected.itertuples():
        pair = exact_daily_price_pair(lookup, row.signal_date, row.root, row.contract_key, calendar)
        if not pair:
            continue
        entry_date, exit_date, entry, exit_price, basis = pair
        i, j = index_position[row.signal_date], ROOTS.index(row.root)
        cached_return = execution.target_return[i, j]
        if not np.isfinite(cached_return):
            continue
        daily_gross = row.weight*cached_return
        daily_commission = abs(row.weight)*2.16 / (entry*MULTIPLIERS[MICROS[row.root]])
        daily_slippage = abs(row.weight)*2.*SLIPPAGE[row.root]/entry
        rows.append({"variant": row.variant, "signal_date": row.signal_date, "morning_entry_session": row.session,
                     "root": row.root, "contract": row.contract_key, "weight": row.weight,
                     "daily_entry_date": entry_date, "daily_exit_date": exit_date, "daily_basis": basis,
                     "daily_entry_price": entry, "daily_exit_price": exit_price,
                     "minute_entry_price": row.entry_price, "minute_exit_price": row.exit_price,
                     "morning_gross": row.gross_return, "morning_commission_only": row.gross_return-row.commission_return,
                     "morning_net": row.net_return, "daily_gross": daily_gross,
                     "daily_independent_gross": row.weight*(exit_price/entry-1.),
                     "daily_round_trip_commission_only": daily_gross-daily_commission,
                     "daily_round_trip_net": daily_gross-daily_commission-daily_slippage,
                     "daily_original_net": daily_gross-cached_costs.at[(row.signal_date, row.root)],
                     "entry_price_difference_pct": 100.*(row.entry_price/entry-1.),
                     "exit_price_difference_pct": 100.*(row.exit_price/exit_price-1.)})
    return pd.DataFrame(rows), price_checks


def paired_control_metrics(controls):
    columns = ["morning_gross", "morning_commission_only", "morning_net", "daily_gross",
               "daily_round_trip_commission_only", "daily_round_trip_net", "daily_original_net"]
    metrics, curves = [], []
    for variant, variant_rows in controls.groupby("variant"):
        for book in ("portfolio", "portfolio_ex_silver", *ROOTS):
            roots = ROOTS if book == "portfolio" else tuple(root for root in ROOTS if root != "SI") if book == "portfolio_ex_silver" else (book,)
            selected = variant_rows.loc[variant_rows.root.isin(roots)]
            complete = selected.groupby("signal_date").root.nunique().eq(len(roots))
            values = selected.loc[selected.signal_date.isin(complete.index[complete])].groupby("signal_date")[columns].sum()
            correlation = values.morning_gross.corr(values.daily_gross) if len(values) >= 2 else np.nan
            for column in columns:
                metrics.append({"variant": variant, "book": book, "case": column, "return_correlation": correlation,
                                "start_signal": values.index.min(), "end_signal": values.index.max(),
                                **performance(values[column])})
                if len(values):
                    curve = values[[column]].rename(columns={column: "return"}).reset_index()
                    curve["variant"], curve["book"], curve["case"] = variant, book, column
                    curve["equity"] = (1.+curve["return"]).cumprod()*100.
                    curves.append(curve)
    return pd.DataFrame(metrics), pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def report(out, checks, metrics, paired, paired_curves, route_counts, baseline, window_correlations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {"daily_round_trip_net": "#166b5d", "morning_net": "#b35831"}
    for axis, book in zip(axes, ("portfolio", "portfolio_ex_silver")):
        for case in colors:
            selected = paired_curves.loc[paired_curves.book.eq(book) & paired_curves["case"].eq(case)
                                         & paired_curves.variant.eq("exact_daily_contract")]
            axis.plot(selected.signal_date, selected.equity, color=colors[case],
                      label="Daily close interval" if case.startswith("daily") else "06/07 UTC + 24 hours")
        axis.set_title(book.replace("_", " ").title()+" / identical matched signals and contracts", loc="left")
        axis.set_ylabel("Growth of 100")
        axis.legend(frameon=False)
        axis.grid(alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=150, bbox_inches="tight", facecolor="#faf9f4")
    plt.close(fig)
    embedded = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()
    display = paired.copy()
    display.loc[display.days.lt(20), "net_sharpe"] = np.nan
    main = display.loc[display.book.isin(("portfolio", "portfolio_ex_silver"))
                       & display.variant.eq("exact_daily_contract")]
    observed = metrics.loc[metrics["mode"].eq("target")].copy()
    observed.loc[observed.days.lt(20), ["gross_sharpe", "commission_only_sharpe", "net_sharpe"]] = np.nan
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Five Currents / 24-Hour Reconciliation</title>
<style>body{{font:15px/1.6 Georgia,serif;background:#faf9f4;color:#193c36;margin:0}}main{{max-width:1150px;margin:auto;padding:40px}}img{{width:100%}}table{{font:11px/1.5 Consolas,monospace;border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #d9ded1;text-align:right}}th{{background:#e7ede3}}.scroll{{overflow:auto}}</style></head><body><main>
<h1>24-Hour Hold / Daily Backtest Reconciliation</h1>
<h2>Accounting Identity Check</h2><pre>{json.dumps(checks, indent=2)}</pre>
<p>The published daily return ledger is reproduced with the frozen weights and original cost convention. Its forecast at T uses the close-to-close interval T+1 to T+2. Archive returns are displayed on their actual T+2 mark date; the maintained extension uses the stored exact-contract calendar marks.</p>
<h2>Execution Mismatch Found</h2>{route_counts.to_html(border=0)}
<p>The earlier intraday experiment reconstructed an expiry/crossover execution route instead of retaining the actual daily-backtest contract identity. That changed the traded maturity on many dates. Both paths are retained below for diagnosis, not as alternatives selected by performance. The exact daily contract path is the reconciliation control.</p>
<p><b>The earlier short-horizon conclusions are superseded.</b> The corrected independent-horizon results, including original and reversed target/change exposures, are saved in corrected_short_horizon_summary.csv. Each horizon has its own matching dates. No parameters or weights are re-estimated.</p>
<h2>Same-Sample Bridge</h2><div class="scroll">{main[['book','case','days','net_sharpe','total_return','max_drawdown','return_correlation']].to_html(index=False,border=0,na_rep='n/a',float_format=lambda value:f'{value:.4f}')}</div><img src="{embedded}" alt="Same-sample daily and 24-hour equity comparison">
<p>For each book, these comparisons use identical signal dates, unchanged weights and the identical held delivery month. Daily and morning round-trip cases both charge $2.16 per micro lot, plus the same separate price-slippage allowances. Daily-original net retains the original standard-contract turnover-only costing and is not an apples-to-apples cost comparison.</p>
<p>A 06:00/07:00 UTC entry plus exactly 24 elapsed hours is a different interval from the daily close-to-close target. It also cannot silently become a 72-hour weekend hold. Cached endpoints lacking a literal 24-hour exit remain missing. Missing outcomes are not assumed flat. Equity compounds the observed matched subset, not an uninterrupted full-calendar track record.</p>
<h2>Signal-to-Execution Timing Control</h2>
<p>The requested morning entry uses signal T on the morning of T+1, before the daily backtest's T+1 close entry. The additional after-entry-close control waits until the morning of T+2 with exactly the same T signal and contract. Neither morning interval is identical to the original daily interval. This explicitly exposes the timing difference; it is not a search for a more profitable delay.</p>
<div class="scroll">{display.loc[display.book.isin(('portfolio','portfolio_ex_silver')) & display.variant.eq('exact_daily_contract_after_entry_close'),['book','case','days','net_sharpe','return_correlation']].to_html(index=False,border=0,na_rep='n/a',float_format=lambda value:f'{value:.4f}')}</div>
<h2>Price-Window Audit</h2>
<p>These correlations use unweighted price moves with the identical contract month. The requested morning-to-morning window spans the tail of the previous close interval and the start of the next one. Its correlations therefore need not match the daily target's close-to-close return. Detailed dated price pairs remain in same_signal_contract_price_bridge.csv.</p>
<div class="scroll">{window_correlations.loc[window_correlations.variant.eq('exact_daily_contract')].to_html(index=False,border=0,float_format=lambda value:f'{value:.4f}')}</div>
<p>One raw archive mark is absent: the negative May-2020 WTI settlement on 20 April 2020 is filtered from the positive-price valuation-mark table. The corresponding forecast is 16 April 2020, before this strategy begins, with zero held CL weight. It does not affect any causal portfolio return. The all-history raw-mark check records this exception rather than silently treating it as a pass.</p>
<h2>Published Daily Benchmark</h2>{baseline.to_html(index=False,border=0,float_format=lambda value:f'{value:.4f}')}
<h2>24-Hour Results / Both Contract Paths</h2><div class="scroll">{observed[['variant','book','days','gross_sharpe','commission_only_sharpe','net_sharpe','total_return','max_drawdown']].to_html(index=False,border=0,na_rep='n/a',float_format=lambda value:f'{value:.4f}')}</div>
<h2>Asset Same-Sample Controls</h2><div class="scroll">{display.loc[display.book.isin(ROOTS),['variant','book','case','days','net_sharpe','return_correlation']].to_html(index=False,border=0,na_rep='n/a',float_format=lambda value:f'{value:.4f}')}</div>
<p>All trade identities, prices, dates and costs are retained in CSV/parquet audit ledgers. No alpha selection, parameter refitting, new market-data pull, database access or production change occurs in this study. Historical signal publication vintages are incomplete, so morning signal availability remains a conditional assumption.</p></main></body></html>'''
    path = out / "Five_Currents_24h_Reconciliation_Report.html"
    path.write_text(html, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("five_currents_intraday_exit_20261009_matched"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-24h", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    snapshot = Path(manifest["source_snapshot"])
    cache = Path(manifest["source_cache"])
    cache_bytes = cache.read_bytes()
    if hashlib.sha256(cache_bytes).hexdigest() != manifest["cache_sha256"]:
        raise ValueError("Daily cache changed since the original experiment; do not mix vintages")
    if hashlib.sha256((source / "frozen_weights.parquet").read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Frozen weights checksum mismatch")
    replay = pickle.loads(cache_bytes)
    weights = pd.read_parquet(source / "frozen_weights.parquet")
    pd.testing.assert_frame_equal(weights, pd.read_parquet(snapshot / "weights.parquet"))
    client = json.loads((snapshot / "client.json").read_text(encoding="utf-8"))
    history = client["history"]
    expected = pd.Series(history["returns"], index=pd.to_datetime(history["dates"]), dtype=float)
    out = args.output_dir.resolve()
    out.mkdir(exist_ok=False)
    print("[1/4] Frozen publication weights loaded; no fitting or data pulls", flush=True)

    from five_currents_daily.dashboard_core import dollar_cost_execution
    from five_currents_daily.reconciliation import calendar_backtest
    from us_commodity_disciplined_core import fast_product_backtest
    execution = dollar_cost_execution(replay.execution)
    ledger = independent_daily_ledger(weights, execution.target_return, execution.identity, execution.base_cost_rate)
    ledger.to_parquet(out / "daily_accounting_ledger.parquet", index=False)
    product = ledger.pivot(index="signal_date", columns="root", values="net_return").reindex(columns=ROOTS)
    actual = product.sum(axis=1, min_count=len(ROOTS))
    production = fast_product_backtest(weights.to_numpy(), replay.panel, execution, "base")
    checks = {"independent_accounting_vs_daily_engine": compare_series(actual, production.net_return)}
    archived = actual.where(actual.index >= replay.first_causal_date).shift(2)
    marks = replay.combined.valuation_marks
    calendar_net, calendar_products = calendar_backtest(weights.to_numpy(), weights.index, execution, marks, "base")
    reproduced = calendar_net.reindex(expected.index)
    frozen = reproduced.index <= replay.combined.cutoff
    reproduced.loc[frozen] = archived.reindex(reproduced.index).loc[frozen]
    checks["reproduced_vs_published_daily_history"] = compare_series(reproduced, expected)
    published = pd.DataFrame({"published_net": expected, "reproduced_net": reproduced,
                              "difference": reproduced-expected})
    published.to_csv(out / "published_daily_reconciliation.csv", index_label="mark_date")
    print(json.dumps(checks, indent=2), flush=True)

    original_identity = pd.DataFrame(execution.identity, index=weights.index, columns=ROOTS)
    intraday_identity = pd.read_parquet(source / "cached_execution_routes.parquet")
    keys = original_identity.copy()
    prior_keys = intraday_identity.copy()
    for root in ROOTS:
        keys[root] = [identity_key(root, value, day) for day, value in original_identity[root].items()]
        prior_keys[root] = [identity_key(root, value, day) for day, value in intraday_identity[root].items()]
    window = weights.index[(weights.index >= pd.Timestamp(manifest["start"])) & (weights.index <= pd.Timestamp(manifest["end"]))]
    route_rows = []
    for day in window:
        for root in ROOTS:
            route_rows.append({"signal_date": day, "root": root, "backtest_contract": keys.at[day, root],
                               "previous_intraday_contract": prior_keys.at[day, root],
                               "same_contract": keys.at[day, root] == prior_keys.at[day, root]})
    routes = pd.DataFrame(route_rows)
    routes.to_csv(out / "execution_route_reconciliation.csv", index=False)
    print("[2/4] Intraday route vs frozen daily backtest", flush=True)
    print(routes.groupby("root").same_contract.agg(["sum", "count", "mean"]).to_string(), flush=True)

    bars = pd.read_parquet(source / "cached_minute_endpoints.parquet")
    if args.reuse_24h:
        metrics = pd.read_csv(args.reuse_24h / "24h_summary_metrics.csv")
        curves = pd.read_parquet(args.reuse_24h / "24h_equity_curves.parquet")
        audit = pd.read_parquet(args.reuse_24h / "24h_execution_audit.parquet")
    else:
        variants, all_curves, all_audits = [], [], []
        for label, identities in (("exact_daily_contract", original_identity), ("previous_intraday_route", intraday_identity)):
            legs = signal_legs(weights, identities, start=manifest["start"], end=manifest["end"])
            audit = recost_matches(match_bars(legs, bars, horizons=(1440,)), execution_basis="micro")
            paths = daily_paths(audit, window, horizons=(1440,))
            metrics, curves = summarize(paths, horizons=(1440,))
            for frame in (audit, metrics, curves):
                frame["variant"] = label
            variants.append(metrics)
            all_curves.append(curves)
            all_audits.append(audit)
        metrics = pd.concat(variants, ignore_index=True)
        curves = pd.concat(all_curves, ignore_index=True)
        audit = pd.concat(all_audits, ignore_index=True)
    # A diagnostic delay aligns the morning decision with the daily entry close;
    # it must never replace the user's requested schedule based on its outcome.
    late_legs = signal_legs(weights, original_identity, start=manifest["start"], end=manifest["end"],
                            execution_lag_sessions=2)
    late_audit = recost_matches(match_bars(late_legs, bars, horizons=(1440,)), execution_basis="micro")
    late_paths = daily_paths(late_audit, window, horizons=(1440,))
    late_metrics, late_curves = summarize(late_paths, horizons=(1440,))
    for frame in (late_audit, late_metrics, late_curves):
        frame["variant"] = "exact_daily_contract_after_entry_close"
    metrics = pd.concat([metrics, late_metrics], ignore_index=True)
    curves = pd.concat([curves, late_curves], ignore_index=True)
    audit = pd.concat([audit, late_audit], ignore_index=True)
    metrics.to_csv(out / "24h_summary_metrics.csv", index=False)
    curves.to_parquet(out / "24h_equity_curves.parquet", index=False)
    audit.to_parquet(out / "24h_execution_audit.parquet", index=False)
    print("[3/4] Literal 24-hour holds / same cached prices / $2.16 per micro round trip", flush=True)
    print(metrics.loc[metrics["mode"].eq("target"), ["variant", "book", "days", "gross_sharpe", "commission_only_sharpe", "net_sharpe"]].to_string(index=False), flush=True)

    print("[corrected replay] Independent original horizons, frozen daily contract path", flush=True)
    original_identity.to_parquet(out / "frozen_daily_execution_routes.parquet")
    short_legs = signal_legs(weights, original_identity, start=manifest["start"], end=manifest["end"])
    short_audit = recost_matches(match_bars(short_legs, bars), execution_basis="micro")
    short_audit.to_parquet(out / "corrected_short_horizon_execution_audit.parquet", index=False)
    short_paths = daily_paths(short_audit, window)
    short_metrics, short_curves = [], []
    for direction, paths in (("original", short_paths), ("reversed", reverse_paths(short_paths))):
        summary, curve = summarize(paths)
        summary["direction"], curve["direction"] = direction, direction
        short_metrics.append(summary)
        short_curves.append(curve)
    pd.concat(short_metrics, ignore_index=True).to_csv(out / "corrected_short_horizon_summary.csv", index=False)
    pd.concat(short_curves, ignore_index=True).to_parquet(out / "corrected_short_horizon_equity.parquet", index=False)

    controls, price_checks = controls_from_daily_marks(audit, weights, execution, ledger, marks, keys, replay.first_causal_date)
    controls.to_csv(out / "same_signal_contract_price_bridge.csv", index=False)
    price_checks.to_csv(out / "independent_daily_price_check.csv", index=False)
    valid_prices = price_checks.dropna(subset=["cached_target", "independent_target"])
    price_missing = price_checks.cached_target.isna().ne(price_checks.independent_target.isna())
    checks["independent_daily_price_reconstruction_all_raw_targets"] = {
        "compared": len(valid_prices), "max_absolute_error": float(valid_prices.difference.abs().max()),
        "mismatched_values": int(valid_prices.difference.abs().gt(1e-10).sum()),
        "missing_mismatches": int(price_missing.sum()),
        "passed": bool(valid_prices.difference.abs().le(1e-10).all() and not price_missing.any()),
        "exception": "May-2020 WTI negative settlement is absent from positive-price mark table; forecast 2020-04-16 precedes first causal trade and has zero CL weight"}
    active_prices = price_checks.loc[price_checks.causal_period & price_checks.weight.abs().gt(1e-10)].copy()
    checks["independent_daily_price_reconstruction_active_causal_targets"] = compare_series(
        active_prices.independent_target, active_prices.cached_target)
    price_checks.loc[price_missing].to_csv(out / "raw_price_reconstruction_exceptions.csv", index=False)
    window_audit, window_correlations = settlement_window_controls(controls, marks)
    window_audit.to_csv(out / "unweighted_price_window_audit.csv", index=False)
    window_correlations.to_csv(out / "unweighted_price_window_correlations.csv", index=False)
    paired, paired_curves = paired_control_metrics(controls)
    paired.to_csv(out / "same_sample_comparison.csv", index=False)
    paired_curves.to_parquet(out / "same_sample_equity_curves.parquet", index=False)
    print("[bridge] Exact contract daily marks vs cached morning bars", flush=True)
    print(json.dumps({k: v for k, v in checks.items() if k.startswith("independent_daily_price")}, indent=2), flush=True)
    print(paired.loc[paired.book.isin(("portfolio", "portfolio_ex_silver")),
                     ["variant", "book", "case", "days", "net_sharpe", "return_correlation"]].to_string(index=False), flush=True)

    baseline_rows = []
    for name, series in (("published_all_five_full_history", expected),
                         ("published_all_five_minute_overlap", expected.loc[window.min():window.max()])):
        baseline_rows.append({"book": name, **performance(series)})
    pd.DataFrame(baseline_rows).to_csv(out / "published_daily_summary.csv", index=False)
    checks["frozen_weights_unchanged"] = {"passed": True}
    checks["no_refits_or_new_market_data"] = {"passed": True}
    (out / "reconciliation_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print("[4/4] Daily benchmark", flush=True)
    print(pd.DataFrame(baseline_rows).to_string(index=False), flush=True)
    path = report(out, checks, metrics, paired, paired_curves,
                  routes.groupby("root").same_contract.agg(["sum", "count", "mean"]), pd.DataFrame(baseline_rows),
                  window_correlations)
    issuance = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_snapshot": str(snapshot),
                "source_cache_sha256": manifest["cache_sha256"], "frozen_weights_sha256": manifest["weights_sha256"],
                "model_refits": False, "new_market_data": False, "database_access": False,
                "reconciliation_checks": checks,
                "output_sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                  for file in out.iterdir() if file.is_file()}}
    (out / "manifest.json").write_text(json.dumps(issuance, indent=2), encoding="utf-8")
    print("[report] " + str(path), flush=True)
    print("[output] " + str(out), flush=True)


if __name__ == "__main__":
    main()
