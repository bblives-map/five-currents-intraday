"""Opening results, with a separately labelled stored second-series price control."""
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

from five_currents_close_only_features import close_only_concepts
from five_currents_front_intraday import HOURS, ROOTS, VARIANTS, evaluate
from five_currents_opening_study import (
    FOUR, PRICE_CONCEPTS, allocated_sleeves, daily_totals,
    opening_moves, opening_predictors, stats,
)
from five_currents_opening_metrics import collect_metrics

PRIMARY_POLICIES = (
    "original_final_flows_reference", "lagged_volume_oi", "price_only", "frozen_flow_indicators",
)


def common_date_metrics(daily):
    records = []
    selected = daily.loc[daily.signal.eq("portfolio_net") & daily.policy.isin(PRIMARY_POLICIES)]
    for (route, book, variant, hours), group in selected.groupby(["price_route", "book", "variant", "hours"]):
        pivot = group.pivot(index="session", columns="policy", values="net_return")
        if set(pivot.columns) != set(PRIMARY_POLICIES):
            raise AssertionError("Missing a policy in the common-date comparison")
        dates = pivot.dropna().index
        for policy, subset in group.groupby("policy"):
            subset = subset.set_index("session").loc[dates]
            for basis in ("gross_return", "net_return", "slippage_net_return"):
                records.append({"price_route": route, "book": book, "variant": variant,
                                "hours": hours, "policy": policy, "cost_basis": basis,
                                "start": dates.min(), "end": dates.max(), **stats(subset[basis])})
    return pd.DataFrame(records)


def render(output, metrics, common, daily, coverage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    labels = {"original_final_flows_reference": "Final-flow reference (not opening-available)",
              "lagged_volume_oi": "One-session-lagged volume / OI",
              "price_only": "Price / curve concepts only",
              "frozen_flow_indicators": "Previous flow indicators + latest close"}
    colors = ("#a59b88", "#176960", "#c27236", "#315b85")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, route in enumerate(("front", "second_series_proxy")):
        for col, variant in enumerate(("net_magnitude", "change_magnitude")):
            group = daily.loc[daily.price_route.eq(route) & daily.book.eq("four")
                              & daily.signal.eq("portfolio_net") & daily.variant.eq(variant)
                              & daily.hours.eq(6) & daily.policy.isin(PRIMARY_POLICIES)]
            pivot = group.pivot(index="session", columns="policy", values="net_return")
            common_dates = pivot.dropna().index
            for policy, color in zip(PRIMARY_POLICIES, colors):
                values = pivot[policy].where(pivot.index.isin(common_dates))
                axes[row, col].plot(values.index, 100*(1+values).cumprod(), lw=1.2,
                                    color=color, label=labels[policy])
            route_label = "Front series" if route == "front" else "Nearby-contract proxy"
            signal_label = "Net position" if variant == "net_magnitude" else "Position change"
            axes[row, col].set_title(f"{route_label} / {signal_label} / 6h", fontsize=11)
            locator = mdates.MonthLocator(interval=6)
            axes[row, col].xaxis.set_major_locator(locator)
            axes[row, col].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
            axes[row, col].tick_params(labelsize=9)
            axes[row, col].grid(alpha=.15)
            axes[row, col].spines[["top", "right"]].set_visible(False)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, 1))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    (output / "opening_comparison.png").write_bytes(buffer.getvalue())
    image = "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()
    sections = []
    for route in ("front", "second_series_proxy"):
        for book in ("four", "five"):
            subset = common.loc[common.price_route.eq(route) & common.book.eq(book)
                                & common.variant.isin(("net_magnitude", "change_magnitude",
                                                       "reverse_net_magnitude", "reverse_change_magnitude"))]
            for basis in ("gross_return", "net_return", "slippage_net_return"):
                view = subset.loc[subset.cost_basis.eq(basis)].pivot(
                    index=["policy", "variant"], columns="hours", values="sharpe")
                sections.append(f"<h3>{route} / {book} markets / {basis}</h3>"+
                                view.to_html(border=0, float_format=lambda x:f"{x:.2f}", na_rep="n/a"))
            counts = subset.loc[subset.cost_basis.eq("net_return") & subset.variant.eq("net_magnitude")].pivot(
                index="policy", columns="hours", values="days")
            sections.append("<p>Common observations:</p>"+counts.to_html(border=0))
    alpha = metrics.loc[metrics.price_route.eq("second_series_proxy") & metrics.book.eq("four")
                        & ~metrics.signal.str.startswith("portfolio_")
                        & metrics.policy.isin(("lagged_volume_oi", "frozen_flow_indicators"))
                        & metrics.variant.eq("net_magnitude") & metrics.cost_basis.eq("net_return")]
    alpha_table = alpha.pivot(index=["policy", "signal"], columns="hours", values="sharpe")
    assets = metrics.loc[metrics.price_route.eq("second_series_proxy") & metrics.book.isin(ROOTS)
                         & metrics.signal.eq("portfolio_net") & metrics.policy.isin(PRIMARY_POLICIES)
                         & metrics.variant.eq("net_magnitude") & metrics.cost_basis.eq("net_return")]
    asset_table = assets.pivot(index=["policy", "book"], columns="hours", values="sharpe")
    path = output / "Five_Currents_Opening_Availability_Study.html"
    text = f'''<!doctype html><html><head><meta charset="utf-8"><title>Five Currents / Opening Availability</title>
<style>body{{margin:0;background:#f7f5ee;color:#183b36;font:15px/1.6 Georgia,serif}}main{{max-width:1180px;margin:auto;padding:36px}}h1,h2,h3{{line-height:1.25}}table{{border-collapse:collapse;font:12px/1.5 Consolas,monospace}}td,th{{padding:7px 10px;border-bottom:1px solid #d7ddd2;text-align:right}}th{{background:#e8ece2}}img{{width:100%}}.scroll{{overflow:auto}}a{{color:#176960}}</style></head><body><main>
<h1>Five Currents: Signals at the Session Reopen</h1>
<p>Latest completed daily prices; no new alpha parameters, refits, signal selection or risk coefficients.
Entry is 17:00 Chicago on the calendar evening before the trading session: 06:00 Singapore during US daylight time,
07:00 in winter. Exit completely after 1, 2, 3, 4, 6 or 12 hours, within the same Singapore date.
This is not the earlier 06:00-07:00 UTC entry. The clock follows the
<a href="https://www.cmegroup.com/market-regulation/files/gold-futures-and-options-fact-card.pdf">CME session convention</a>.</p>
<h2>What Changes</h2><p><b>Price only:</b> curve quality reversal, carry-confirmed momentum and carry/basis reversal, retaining only concepts admitted by the existing historical selector. Original prior-session ADV gates remain in place.
<b>Lagged volume/OI:</b> rebuild all ten concepts after replacing each contract's volume and open interest with its previous observed session's same-contract values. No forward filling across unknown values.
<b>Frozen flow indicators:</b> carry forward the previously known OBV/volume gates, OI scores and curve participation indicators, while updating price returns and curve slopes using the latest completed close. OI-only and aggregate-volume-only components receive no fresh flow information.
<b>Final-flow reference:</b> original finalized current-session inputs, shown only as an availability-optimistic reference, not a tradable opening forecast.</p>
<p>Historical admission decisions, local parameter neighbourhoods, daily holding/stagger logic, risk tilts and refit cadence are fixed. Removing concepts averages only the remaining admitted concepts. The three price concepts are unchanged by the input-lag reconstruction. The separate close-refresh delta compares the new preview with the preceding finalized target; the main change test compares consecutive previews under the same policy.</p>
<h2>Positions and Costs</h2><p>Each policy tests net position and position change, sign and magnitude, plus all four reversals. Gross means before-cost P&amp;L.
Net deducts $2.16 per actual micro lot for the complete round trip, without an extra x10; a further sensitivity includes the previously specified per-product tick slippage.
Standalone alpha results retain each concept's existing economic position sizing. Separately traded, unnetted selected sleeves are costed before offsetting; their magnitude-weighted gross P&amp;L reconciles to the net book, but their costs are higher. Sign-weighted sleeve aggregation is a different exposure construction and is shown only in the CSV.</p>
<h2>Price Coverage</h2><p>Primary results use the stored front-series prices. The fixed second-series control always uses the existing bars_1m_2m archive, independently of exits; it is a nearby-contract proxy, not a repaired first-contract history. Both legs of every trade use the same stored stream and contract. Missing active-asset prices stay missing, never zero. Four-market controls exclude silver without redistributing weights. Tables below use common dates across the four main availability policies for each horizon; different horizons need not have identical dates. No fills are inferred from the chart gaps.</p>
<p><b>The five-market front-series common sample has only 6-13 days, because silver opening prices are sparse. Its Sharpe estimates are not interpretable. The separate nearby-contract control has substantially better coverage but is not the same execution contract.</b></p>
<div class="scroll">{coverage.to_html(index=False,border=0)}</div>
<h2>Common-Date Equity / Four Markets</h2><p>Six hours is a pre-specified illustration, not a selected winner. Growth of 100 after commission; gaps remain visible.</p><img src="{image}" alt="Opening net and change strategies on common dates">
<h2>All Holding Periods</h2><div class="scroll">{''.join(sections)}</div>
<h2>Individual Alphas / Commission-Net Sharpe</h2><p>Four-market second-series diagnostic, all existing concepts, no performance-based filtering. Dates can differ where an individual alpha is inactive or unpriced.</p><div class="scroll">{alpha_table.to_html(border=0,float_format=lambda x:f'{x:.2f}',na_rep='n/a')}</div>
<h2>Individual Assets / Commission-Net Sharpe</h2><div class="scroll">{asset_table.to_html(border=0,float_format=lambda x:f'{x:.2f}',na_rep='n/a')}</div>
<p>Full sign/magnitude/reversal, gross/net/slippage, per-asset and unnetted-sleeve metrics, dated predictions and daily P&amp;L are retained alongside this report. The stored intraday sample begins in June 2024; this is not a 2020-onward intraday record. Historical daily publication vintages are unavailable, so a one-session lag is an availability assumption, not a timestamp-certified live reconstruction. No broker/data APIs, live processes or production publications were changed.</p>
</main></body></html>'''
    path.write_text(text, encoding="utf-8")
    return path


def run(args):
    source = args.source.resolve()
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=False)
    manifest = json.loads((source / "manifest.json").read_text())
    cache = Path("five_currents_dashboard/private_replay_cache.pkl")
    cache_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    assert cache_hash == manifest["source_cache_sha256"]
    with cache.open("rb") as handle:
        original = pickle.load(handle)
    snapshot = Path("five_currents_dashboard/snapshots/000032_2026-10-09")
    selection = pd.read_parquet(snapshot / "selection_audit.parquet")
    coefficients = pd.read_parquet(snapshot / "product_scaling.parquet")
    concepts = np.load(source / "concept_weights.npz")
    print("[close-only] Factoring existing alphas into price and flow components; verifying the control", flush=True)
    frozen = close_only_concepts(original, original, concepts["original"])
    print("[close-only] Unchanged-flow factorization agrees with original alphas to 1e-10", flush=True)
    final, _, _ = allocated_sleeves(frozen, original.concept_names, original.weights.index, selection, coefficients)
    weights = pd.DataFrame(final, index=original.weights.index, columns=ROOTS)
    weights.to_parquet(output / "frozen_flow_indicators_targets.parquet")
    np.savez_compressed(output / "frozen_flow_concepts.npz", weights=frozen)
    old = pd.read_parquet(source / "opening_predictors.parquet")
    sessions = pd.DatetimeIndex(old.session.unique()).sort_values()
    extra = [opening_predictors(weights, sessions).assign(policy="frozen_flow_indicators", signal="portfolio_net")]
    for position, name in enumerate(original.concept_names):
        values = pd.DataFrame(frozen[position], index=weights.index, columns=ROOTS)
        extra.append(opening_predictors(values, sessions).assign(policy="frozen_flow_indicators", signal=name))
    finalized = pd.read_parquet(source / "original_final_flows_reference_targets.parquet")
    extra.append(opening_predictors(weights, sessions, previous_weights=finalized).assign(
        policy="frozen_flow_refresh_vs_previous_final", signal="portfolio_net"))
    predictors = pd.concat([old, *extra], ignore_index=True)
    predictors.to_parquet(output / "all_opening_predictors.parquet", index=False)
    metrics, curves, coverage, paths_to_save = [], [], [], []
    concept_sets = {"original_final_flows_reference": concepts["original"],
                    "lagged_volume_oi": concepts["lagged"], "price_only": concepts["original"],
                    "frozen_flow_indicators": frozen}
    net_weights = {policy: pd.read_parquet(source / f"{policy}_targets.parquet")
                   for policy in PRIMARY_POLICIES if policy != "frozen_flow_indicators"}
    net_weights["frozen_flow_indicators"] = weights
    for route, path in (("front", source / "opening_minute_anchors.parquet"),
                        ("second_series_proxy", args.second_cache)):
        bars = pd.read_parquet(path)
        moves = opening_moves(bars, sessions)
        moves.to_parquet(output / f"{route}_price_moves.parquet", index=False)
        summary = moves.assign(matched=moves.status.eq("matched")).groupby(["root", "hours"]).agg(
            candidate_days=("matched", "size"), matched_days=("matched", "sum")).reset_index()
        coverage.append(summary.assign(price_route=route))
        for (policy, signal), predictor in predictors.groupby(["policy", "signal"], sort=False):
            paths = evaluate(predictor, moves)
            if "vs_previous_final" in policy:
                paths = paths.loc[paths.variant.str.contains("change")]
            first = len(metrics)
            collect_metrics(paths, policy, signal, metrics, curves,
                            cost_only=signal != "portfolio_net")
            for record in metrics[first:]:
                record["price_route"] = route
            if signal == "portfolio_net":
                paths_to_save.append(paths.assign(policy=policy, price_route=route))
        print(f"[results] {route}: individual alphas, net/change/sign/reversals complete", flush=True)
        for policy, values in concept_sets.items():
            _, sleeves, _ = allocated_sleeves(values, original.concept_names, original.weights.index,
                selection, coefficients, PRICE_CONCEPTS if policy == "price_only" else None)
            independent = []
            for position in range(len(original.concept_names)):
                frame = pd.DataFrame(sleeves[position], index=weights.index, columns=ROOTS)
                independent.append(evaluate(opening_predictors(frame, sessions), moves))
            independent = pd.concat(independent, ignore_index=True)
            group = independent.groupby(["session", "root", "hours", "variant"])
            columns = ["gross_return", "commission_return", "slippage_return", "net_return", "slippage_net_return"]
            gross = group[columns].sum(min_count=1)
            ready = group.ready.all()
            gross.loc[~ready] = np.nan
            gross["ready"] = ready
            gross["weight"] = independent.weight.abs().groupby(
                [independent[key] for key in ("session", "root", "hours", "variant")]).sum()
            first = len(metrics)
            collect_metrics(gross.reset_index(), policy, "portfolio_unnetted_sleeves", metrics, curves)
            for record in metrics[first:]:
                record["price_route"] = route
        # Tag the curves emitted for this price route without modifying earlier rows.
        for curve in curves:
            if "price_route" not in curve:
                curve["price_route"] = route
    metrics = pd.DataFrame(metrics)
    daily = pd.concat(curves, ignore_index=True)
    coverage = pd.concat(coverage, ignore_index=True)
    common = common_date_metrics(daily)
    metrics.to_csv(output / "all_signal_metrics.csv", index=False)
    common.to_csv(output / "common_date_policy_metrics.csv", index=False)
    daily.to_parquet(output / "portfolio_daily_returns.parquet", index=False)
    pd.concat(paths_to_save, ignore_index=True).to_parquet(output / "portfolio_asset_returns.parquet", index=False)
    coverage.to_csv(output / "price_coverage.csv", index=False)
    yearly = []
    for key, group in daily.groupby(["price_route", "policy", "signal", "book", "variant", "hours"]):
        for year, subset in group.groupby(group.session.dt.year):
            yearly.append(dict(zip(["price_route", "policy", "signal", "book", "variant", "hours"], key))
                          | {"year": int(year), **stats(subset.net_return)})
    pd.DataFrame(yearly).to_csv(output / "yearly_metrics.csv", index=False)
    report = render(output, metrics, common, daily, coverage)
    assert cache_hash == hashlib.sha256(cache.read_bytes()).hexdigest()
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_study": str(source),
             "market_api_calls": False, "database_access": False, "production_changes": False,
             "refitting": False, "close_only_factorization_reconciles": True,
             "original_source_cache_sha256": cache_hash,
             "price_sources_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (source / "opening_minute_anchors.parquet", args.second_cache)},
             "source_code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in ("compare_five_currents_opening_policies.py", "five_currents_close_only_features.py", "five_currents_opening_study.py")},
             "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    selected = common.loc[common.book.eq("four") & common.cost_basis.eq("net_return")
                          & common.variant.isin(("net_magnitude", "change_magnitude"))]
    for route, group in selected.groupby("price_route"):
        print("\n[common-date commission-net Sharpe] " + route, flush=True)
        print(group.pivot(index=["policy", "variant"], columns="hours", values="sharpe").round(3).to_string(), flush=True)
    print("[report] " + str(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="five_currents_opening_study_20261011", type=Path)
    parser.add_argument("--second-cache", default="five_currents_opening_second_series_cache_20261011/opening_minute_anchors.parquet", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())
