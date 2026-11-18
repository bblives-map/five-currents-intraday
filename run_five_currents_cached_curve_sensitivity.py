"""Small offline input-ablation replay; never refit or touch the live dashboard."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import pickle
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from five_currents_daily.engine import FROZEN_CUTOFF, DEFAULT_SOURCE_DIR
from five_currents_daily.curve_policy import CURVE_RANK_LIMITS as MIXED_RANK_LIMITS
from five_currents_daily.curve_signals import (
    CURVE_CONCEPTS, requested_curve_features, frozen_allocation, replace_curve_concepts,
)
from run_us_commodity_10_concept_504_21_allocator_study import MAXIMUM_GROSS, PRODUCT_CAP
from run_us_commodity_fixed_vintage_robustness import candidate_weights_from_row
from us_commodity_carry_tmv_expansion_core import (
    _dynamic_curve_statistics, _dynamic_curve_sum, _safe_zscore,
)
from us_commodity_disciplined_core import _dynamic_rank_value, fast_product_backtest
from us_commodity_execution_overlays import build_execution_overlays
from us_commodity_front_roll_core import install_execution_ranks
from us_commodity_locked_three_sleeve_core import path_metrics
from us_commodity_pooled_harmonization_core import restore_gross_after_product_tilt
from us_commodity_research_core import (
    PRODUCT_ORDER, _cap_and_normalize_rows, _field_matrix, _mapping_matrix,
    _same_contract_log_change_matrix,
)

POLICIES = ("near6", "near5", "near4", "near3", "liquid4")


def exact_contract_adv(volume, mapping, window=20, minimum=10):
    """Use previous sessions of the same listed contract, including rank migration."""
    result = pd.DataFrame(np.nan, index=volume.index, columns=volume.columns)
    contracts = pd.unique(mapping.to_numpy().ravel())
    for contract in contracts:
        if pd.isna(contract):
            continue
        selected = mapping.eq(contract)
        if selected.sum(axis=1).gt(1).any():
            raise ValueError("A physical contract occupies multiple curve ranks")
        history = volume.where(selected).sum(axis=1, min_count=1)
        average = history.shift(1).rolling(window, min_periods=minimum).mean()
        for rank in volume.columns:
            result.loc[selected[rank], rank] = average.loc[selected[rank]]
    return result


def curve_mask(policy, volume, mapping, *, root=None):
    if policy == "mixed":
        if root not in MIXED_RANK_LIMITS:
            raise ValueError("Mixed curve policy requires a supported commodity root")
        policy = "near" + str(MIXED_RANK_LIMITS[root])
    keep = pd.DataFrame(False, index=volume.index, columns=volume.columns)
    if policy.startswith("near"):
        keep.loc[:, keep.columns <= int(policy[4:])] = True
        return keep
    if policy != "liquid4":
        raise ValueError(policy)
    # Keep the two original signal/execution anchors. Never compress rank gaps.
    keep.loc[:, [1, 2]] = True
    adv = exact_contract_adv(volume.loc[:, 1:6], mapping.loc[:, 1:6])
    for date, row in adv.loc[:, 3:6].iterrows():
        chosen = row.dropna().sort_values(ascending=False, kind="stable").head(2).index
        keep.loc[date, chosen] = True
    return keep




def primary_only_inputs(replay, inputs):
    primary = {root: {field: values.copy() for field, values in current.items()}
               for root, current in inputs.items()}
    fields = {"settlement": "price", "cleared_volume": "volume", "open_interest": "oi"}
    for row in replay.combined.database_rows.itertuples(index=False):
        provenance = json.loads(row.failsafe_provenance) if isinstance(row.failsafe_provenance, str) else {}
        for field in provenance:
            target = fields.get(field)
            if target:
                primary[row.root_symbol][target].at[pd.Timestamp(row.session_date), int(row.contract_rank)] = np.nan
    return primary


def coverage_records(policy, inputs, masks, start):
    records = []
    for root, fields in inputs.items():
        mask = masks[root].loc[masks[root].index > start]
        valid = (fields["price"].gt(0) & fields["volume"].ge(0) & fields["oi"].ge(0)).reindex_like(mask)
        wanted = MIXED_RANK_LIMITS[root] if policy == "mixed" else 4 if policy == "liquid4" else int(policy[4:])
        complete = (valid | ~mask).all(axis=1) & mask.sum(axis=1).eq(wanted)
        records.append({"policy": policy, "product": root, "sessions": len(complete),
                        "complete_sessions": int(complete.sum()), "complete_fraction": float(complete.mean()),
                        "selection_warmup_sessions": int(mask.sum(axis=1).lt(wanted).sum()),
                        "latest_complete": bool(complete.iloc[-1])})
    return records


def save_equity_chart(output, paths, start, policies=POLICIES):
    colors = ("#12685d", "#33769e", "#bc8430", "#ae5750", "#536338")
    labels = {"near6": "All markets: ranks 1-6", "mixed": "SI/HG: 1-4; CL/NG/GC: 1-6"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for color, policy in zip(colors, policies):
        series = paths[(policy, "base")].net_return.loc[start:]
        wealth = (1 + series).cumprod()
        axes[0].plot(series.index, 100 * wealth, color=color, label=labels.get(policy, policy))
        axes[1].plot(series.index, 100 * (wealth / wealth.cummax() - 1), color=color)
    axes[0].set_ylabel("Growth of 100, after costs")
    axes[1].set_ylabel("Drawdown %")
    axes[0].legend(ncol=min(len(policies), 5), frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    fig.tight_layout()
    chart = output / "curve_sensitivity.png"
    fig.savefig(chart, dpi=150)
    plt.close(fig)
    return chart


def report(output, metrics, coverage, paths, diagnostics, replay, common):
    chart = save_equity_chart(output, paths, replay.first_causal_date)
    embedded = "data:image/png;base64," + base64.b64encode(chart.read_bytes()).decode()
    table = metrics.loc[metrics.period.eq("full_common") & metrics.scenario.eq("base"),
                        ["policy", "net_sharpe", "cagr", "max_drawdown", "annual_turnover", "baseline_return_correlation"]].copy()
    for field in ("cagr", "max_drawdown"):
        table[field] = table[field].map(lambda v: f"{v:.2%}")
    for field in ("net_sharpe", "annual_turnover", "baseline_return_correlation"):
        table[field] = table[field].map(lambda v: f"{v:.3f}")
    cov = coverage.pivot(index="product", columns="policy", values="complete_fraction").reindex(columns=POLICIES)
    cov = cov.map(lambda v: f"{v:.1%}")
    liquid_coverage = coverage.loc[coverage.policy.eq("liquid4")]
    warmup = int(liquid_coverage.selection_warmup_sessions.max())
    after_warmup = ("After that warm-up its selected inputs are complete in this cache."
                    if (liquid_coverage.sessions - liquid_coverage.selection_warmup_sessions).eq(liquid_coverage.complete_sessions).all()
                    else "Remaining incompleteness after warm-up represents selected-input gaps.")
    recent = metrics.loc[metrics.period.eq("recent_common") & metrics.scenario.eq("base"),
                         ["policy", "total_return", "max_drawdown"]].copy()
    for field in ("total_return", "max_drawdown"):
        recent[field] = recent[field].map(lambda v: f"{v:.2%}")
    latest = pd.DataFrame(diagnostics).set_index("policy")
    latest = latest[["weight_direction_agreement", "active_curve_votes_fraction", "latest_primary_complete_products"]]
    latest["weight_direction_agreement"] = latest.weight_direction_agreement.map(lambda v: f"{v:.1%}")
    latest["active_curve_votes_fraction"] = latest.active_curve_votes_fraction.map(lambda v: f"{v:.1%}")
    text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Five Currents | Cached Curve Sensitivity</title>
<style>body{{font:16px Georgia,serif;color:#203b37;background:#f4f3ec;margin:0}}main{{max-width:1080px;margin:auto;padding:40px}}h1{{font-size:36px}}h2{{margin-top:32px}}p{{line-height:1.6}}img{{width:100%;background:white;border-radius:10px}}table{{border-collapse:collapse;width:100%;background:white;font:14px sans-serif}}th,td{{padding:12px;border-bottom:1px solid #dedfd5;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{padding:18px;border-left:4px solid #bc8430;background:white}}@media(max-width:700px){{main{{padding:18px}}table{{font-size:11px}}th,td{{padding:6px}}}}</style></head><body><main>
<p>FIVE CURRENTS / OFFLINE SENSITIVITY REPLAY</p><h1>How Much Curve Do We Need?</h1>
<p>{replay.first_causal_date.date()} to {replay.last_realised_date.date()} on the original forecast-date axis. No new data, retuning, new candidates or production changes.</p>
<p class="note">Historical concept admissions, their refit transitions, product-risk coefficients, signal parameters, holding periods, execution contracts, roll rules and costs are frozen. Only curve inputs change. This is a conditional sensitivity test, not an independently trained reduced-curve strategy.</p>
<h2>Full-History Performance</h2><p>Statistics use the same {int(common.sum())} marked sessions across all five variants. Missing valuation marks remain unresolved and break the displayed curves; they are never replaced with zero returns. CAGR uses observed-session annualization, as in the existing replay.</p>{table.to_html(index=False, border=0)}
<img src="{embedded}" alt="Full history equity curves and drawdown">
<h2>Cached Primary-Source Completeness</h2><p>July 21, 2026 onwards: all selected curve prices, volume and OI must be present without supplementary Databento/CME fields. These are completeness rates in the existing cache, not proof of historical publication-time availability.</p>{cov.to_html(border=0)}
<p>The liquidity-selected variant has up to {warmup} cached sessions without sufficient exact-contract ADV history to select all four contracts. These are selection warm-up, not evidence of a missing market feed. {after_warmup}</p>
<h2>Recent Two Months</h2><p>Smaller curves reduce input dependence but do not resolve the recent performance decline. These results use the same marked sessions as the full-period comparison; the latest still-unrealised forecasts are not treated as returns.</p>{recent.to_html(index=False, border=0)}
<h2>What Changes?</h2>{latest.to_html(border=0)}
<p>near6/5/4/3 retain their original ordered ranks. liquid4 keeps ranks 1 and 2, plus the two highest-ADV remaining contracts within ranks 3–6, using the prior 20 sessions of each exact contract (minimum 10 observations). Original rank coordinates and offsets are retained: gaps are not compressed, far prices are not substituted, and a pair needing an excluded contract remains unavailable. Technical/front-contract concepts are copied unchanged. Curve regressions and participation sums retain their original minimum-three-observation rules; smaller curves can therefore weaken or disable some components.</p>
<p>Detailed base/doubled-cost results, common-period and recent-period statistics, per-asset contributions, frozen selection/scaling audits and selected-rank masks are saved beside this report. No variant is automatically selected for deployment.</p>
</main></body></html>'''
    (output / "curve_sensitivity_report.html").write_text(text, encoding="utf-8")


def run(args):
    cache = args.cache
    content = cache.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    expected = json.loads(cache.with_suffix(".sha256.json").read_text())["sha256"]
    if digest != expected:
        raise ValueError("Replay cache checksum changed; retry after the worker finishes")
    replay = pickle.loads(content)  # Only the checksum-verified, locally generated private cache.
    output = args.output or Path("us_commodity_cached_curve_sensitivity_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    output.mkdir(exist_ok=False)
    library = pd.read_csv(DEFAULT_SOURCE_DIR / "neighborhood_library.csv")
    library = library.loc[library.effective_source.eq("carry_expanded")].copy()
    signal_panel = install_execution_ranks(replay.panel, replay.combined.wide, replay.combined.mapping, replay.signal_ranks)
    overlays = build_execution_overlays(signal_panel, replay.combined.wide, replay.combined.mapping)
    inputs = {}
    for root in PRODUCT_ORDER:
        price = _field_matrix(replay.combined.wide, root, "PX_SETTLE")
        price = price.where(price.gt(0), _field_matrix(replay.combined.wide, root, "PX_LAST"))
        inputs[root] = {"price": price, "volume": _field_matrix(replay.combined.wide, root, "PX_VOLUME"),
                        "oi": _field_matrix(replay.combined.wide, root, "OPEN_INT"),
                        "mapping": _mapping_matrix(replay.combined.mapping, replay.panel.metadata, root, replay.panel.index)}
    primary = primary_only_inputs(replay, inputs)
    policies = tuple(args.policies)
    metrics, coverage, diagnostics, paths, weights_by_policy = [], [], [], {}, {}
    mask_history = []
    for policy in policies:
        print(f"[{policy}] recomputing {library.signal_id.nunique()} cached curve definitions; frozen selection", flush=True)
        masks = {root: curve_mask(policy, values["volume"], values["mapping"], root=root)
                 for root, values in inputs.items()}
        reduced = {root: {field: value if field == "mapping" else value.where(masks[root])
                          for field, value in fields.items()} for root, fields in inputs.items()}
        features = requested_curve_features(reduced, replay.signal_ranks, sorted(library.signal_id.unique()))
        if policy == "near6":
            baseline_features = features
        elif policy == "mixed":
            for name, values in features.items():
                for root in ("CL", "NG", "GC"):
                    np.testing.assert_allclose(values[root], baseline_features[name][root], atol=0, rtol=0, equal_nan=True)
        concepts = replace_curve_concepts(replay, signal_panel, overlays, library, features)
        weights = frozen_allocation(concepts, replay.concept_names, replay.weights.index,
                                    replay.selection_audit, replay.product_coefficients)
        if policy == "near6":
            concept_error = float(np.max(np.abs(concepts - replay.concept_weights)))
            weight_error = float(np.max(np.abs(weights - replay.weights.to_numpy())))
            if max(concept_error, weight_error) > 1e-10:
                raise AssertionError(f"Baseline does not reconcile: concepts={concept_error}, weights={weight_error}")
        unchanged = [i for i, name in enumerate(replay.concept_names) if name not in CURVE_CONCEPTS]
        np.testing.assert_array_equal(concepts[unchanged], replay.concept_weights[unchanged])
        weights_by_policy[policy] = weights
        for scenario in ("base", "stressed"):
            paths[(policy, scenario)] = fast_product_backtest(weights, replay.panel, replay.execution, scenario)
            if policy == "near6":
                np.testing.assert_allclose(paths[(policy, scenario)].net_return,
                                           replay.paths[scenario].net_return, atol=1e-12, equal_nan=True)
            paths[(policy, scenario)].to_parquet(output / f"{policy}_{scenario}_path.parquet")
        pd.DataFrame(weights, index=replay.weights.index, columns=PRODUCT_ORDER).to_parquet(output / f"{policy}_weights.parquet")
        np.save(output / f"{policy}_concept_weights.npy", concepts)
        for root, mask in masks.items():
            current = mask.loc[:, 1:6].copy()
            current.columns = ["rank_" + str(rank) for rank in current.columns]
            current["product"] = root
            current["policy"] = policy
            mask_history.append(current)
        current_coverage = coverage_records(policy, primary, masks, FROZEN_CUTOFF)
        coverage.extend(current_coverage)
        curve_ids = [replay.concept_names.index(name) for name in CURVE_CONCEPTS]
        active = concepts[curve_ids][:, replay.weights.index >= replay.first_causal_date]
        reference = replay.weights.loc[replay.first_causal_date:].to_numpy()
        diagnostics.append({"policy": policy,
                            "weight_direction_agreement": float((np.sign(weights[replay.weights.index >= replay.first_causal_date]) == np.sign(reference)).mean()),
                            "active_curve_votes_fraction": float((np.abs(active) > 1e-12).mean()),
                            "latest_primary_complete_products": sum(row["latest_complete"] for row in current_coverage)})
    common = pd.Series(True, index=replay.weights.index)
    for path in paths.values():
        common &= path.net_return.notna()
    common &= replay.weights.index >= replay.first_causal_date
    reference = paths[("near6", "base")].net_return
    for (policy, scenario), path in paths.items():
        for period, start, shared in (("full_common", replay.first_causal_date, True),
                                      ("full_available", replay.first_causal_date, False),
                                      ("recent_common", replay.last_realised_date - pd.DateOffset(months=2), True)):
            sample = path.copy()
            if shared:
                sample.loc[~common, ["net_return", "gross_return"]] = np.nan
            result = path_metrics(sample, start=start, end=replay.last_realised_date)
            metrics.append({"policy": policy, "scenario": scenario, "period": period, **result,
                            "baseline_return_correlation": float(sample.net_return.loc[start:].corr(reference.loc[start:]))})
    metrics, coverage = pd.DataFrame(metrics), pd.DataFrame(coverage)
    metrics.to_csv(output / "metrics.csv", index=False)
    coverage.to_csv(output / "primary_coverage.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "diagnostics.csv", index=False)
    pd.concat(mask_history).to_parquet(output / "curve_masks.parquet")
    replay.selection_audit.to_parquet(output / "frozen_selection_audit.parquet")
    replay.product_coefficients.to_parquet(output / "frozen_product_coefficients.parquet")
    from us_commodity_asset_family_walkforward_core import product_net_returns
    for policy, weights in weights_by_policy.items():
        contributions = pd.DataFrame(product_net_returns(weights, replay.execution, scenario="base"),
                                     index=replay.weights.index, columns=PRODUCT_ORDER)
        contributions = contributions.where(paths[(policy, "base")].net_return.notna(), np.nan)
        np.testing.assert_allclose(contributions.sum(axis=1, min_count=1), paths[(policy, "base")].net_return,
                                   atol=1e-12, equal_nan=True)
        contributions.to_parquet(output / f"{policy}_product_returns.parquet")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "cache": str(cache.resolve()),
                "cache_sha256": digest, "policies": list(policies), "network_disabled": True,
                "new_data_requests": 0, "retuning": False, "curve_candidate_rows": len(library),
                "curve_definitions": int(library.signal_id.nunique()), "selection_and_scaling": "Archived intervals and coefficients, unchanged",
                "baseline_concept_maximum_error": concept_error, "baseline_weight_maximum_error": weight_error,
                "mixed_curve_rank_limits": MIXED_RANK_LIMITS if "mixed" in policies else None,
                "mixed_full_depth_feature_checks_passed": "mixed" in policies,
                "first_causal_date": str(replay.first_causal_date.date()), "last_signal_date": str(replay.last_signal_date.date()),
                "last_realised_forecast_date": str(replay.last_realised_date.date()),
                "limitations": ["Original concept admissions were learned on the full curve; this is conditional sensitivity, not independent retuning.",
                                "Absent original pair endpoints are unavailable; there is no remapping, filling or new alpha.",
                                "Cache completeness does not establish historical release-time availability."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.no_report:
        chart = save_equity_chart(output, paths, replay.first_causal_date, policies)
    else:
        report(output, metrics, coverage, paths, diagnostics, replay, common)
    print(metrics.loc[(metrics.period == "full_common") & (metrics.scenario == "base"),
                      ["policy", "net_sharpe", "cagr", "max_drawdown", "baseline_return_correlation"]].to_string(index=False), flush=True)
    print(("Equity curve: " + str(chart.resolve())) if args.no_report else
          "Report: " + str((output / "curve_sensitivity_report.html").resolve()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("five_currents_dashboard/private_replay_cache.pkl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policies", nargs="+", choices=(*POLICIES, "mixed"), default=list(POLICIES))
    parser.add_argument("--no-report", action="store_true", help="Save statistics and equity image without creating HTML")
    args = parser.parse_args()
    if args.policies[0] != "near6" or len(set(args.policies)) != len(args.policies):
        parser.error("Start with the near6 reconciliation control and use unique policies")
    if tuple(args.policies) != POLICIES and not args.no_report:
        parser.error("Custom policy comparisons require --no-report")
    # Fail closed if an imported helper unexpectedly attempts any network access.
    with patch.object(socket.socket, "connect", side_effect=RuntimeError("Network disabled in cached replay")), \
         patch.object(socket.socket, "connect_ex", side_effect=RuntimeError("Network disabled in cached replay")):
        run(args)


if __name__ == "__main__":
    main()
