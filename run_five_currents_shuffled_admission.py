"""Replay extra shuffled-Sharpe gates on cached daily data, without live side effects."""
from __future__ import annotations

import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import base64
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from five_currents_daily.curve_activation import mixed_concepts
from five_currents_daily.curve_signals import frozen_allocation
from five_currents_daily.dashboard_core import dollar_cost_execution
from five_currents_daily.report import _series_metrics
from five_currents_shuffled_admission import (
    apply_admission_gate, cash_before_first_trade, pooled_null_sharpes, training_slice,
)
from run_us_commodity_pooled_harmonization import family_path_matrices
from shfe_rolling_walkforward_core import candidate_training_scores
from us_commodity_asset_family_walkforward_core import _inverse_volatility_coefficients, product_net_returns
from us_commodity_disciplined_core import fast_product_backtest
from us_commodity_research_core import PRODUCT_ORDER


LABELS = {"baseline": "Existing admission", "p80": "Above shuffled P80",
          "p90": "Above shuffled P90", "p95": "Above shuffled P95"}
COLORS = {"baseline": "#264a62", "p80": "#178476", "p90": "#c18536", "p95": "#a44b58"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recalibrate_coefficients(selection, replay, operational, execution, anchor_ordinal):
    """Relearn only the established product-risk tilt for each admitted concept mix."""
    index, names = replay.weights.index, replay.concept_names
    rows = []
    for ordinal, segment in selection.groupby("refit_ordinal", sort=True):
        first = segment.iloc[0]
        positions = training_slice(index, first)
        concepts = replay.concept_weights if ordinal <= anchor_ordinal else operational
        admitted = [names.index(name) for name in segment.loc[segment.selected, "economic_family"]]
        if admitted:
            historical = np.nanmean(concepts[admitted], axis=0)
            returns = product_net_returns(historical, execution, scenario="stressed")[positions]
            coefficients, annual_volatility = _inverse_volatility_coefficients(returns)
            observations = np.isfinite(returns).sum(axis=0)
        else:
            coefficients = np.zeros(len(PRODUCT_ORDER))
            annual_volatility = np.full(len(PRODUCT_ORDER), np.nan)
            observations = np.zeros(len(PRODUCT_ORDER), dtype=int)
        for j, product in enumerate(PRODUCT_ORDER):
            rows.append({"refit_ordinal": ordinal, "forecast_start": first.forecast_start,
                         "forecast_end": first.forecast_end, "train_start": first.train_start,
                         "train_end": first.train_end, "selected_family_count": len(admitted),
                         "product": product, "inverse_volatility_coefficient": coefficients[j],
                         "training_annual_volatility": annual_volatility[j],
                         "training_observations": observations[j], "causal": True})
    return pd.DataFrame(rows)


def formatted_table(frame):
    result = frame.copy()
    for column in result:
        if column in ("annualized_return", "total_return", "annualized_volatility", "max_drawdown",
                      "cash_fraction", "mean_gross_exposure", "admission_fraction", "year_return"):
            result[column] = result[column].map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
        elif pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "-")
    return result.to_html(index=False, border=0, escape=True)


def render(output, metrics, annual, paths, thresholds, admissions, first_date, last_date, draws):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fbfcf9",
                         "axes.facecolor": "#fbfcf9", "text.color": "#213f3a"})
    calibration_end = admissions.loc[admissions.training_sessions.eq(504), "forecast_start"].min()
    available = pd.concat({name: group.set_index("session").net_return
                           for name, group in paths.loc[paths.scenario.eq("base")].groupby("model")}, axis=1)
    common = available.notna().all(axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.1), sharex=True,
                             gridspec_kw={"height_ratios": [2.3, 1]})
    for name in LABELS:
        values = available[name].where(common)
        equity = 100 * (1 + values).cumprod()
        drawdown = 100 * (equity / equity.cummax() - 1)
        record = metrics.loc[metrics.model.eq(name) & metrics.scenario.eq("base")
                              & metrics.period.eq("full_common")].iloc[0]
        axes[0].plot(equity.index, equity, lw=1.55, color=COLORS[name],
                     label=f"{LABELS[name]} | SR {record.sharpe:.2f}")
        axes[1].plot(drawdown.index, drawdown, lw=1.2, color=COLORS[name])
    for ax in axes:
        ax.axvspan(first_date, calibration_end, color="#d5dfd5", alpha=.35)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[0].set_title("Daily portfolio | shuffled-Sharpe admission at every refit", loc="left",
                       fontsize=16, weight="bold", pad=18)
    axes[0].set_ylabel("Growth of 100, after base costs")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlim(first_date, last_date)
    fig.text(.08, .035,
             "Identical signals, execution assumptions and 504/21 cadence; an extra training-only admission gate.\n"
             "Shaded: 126-to-504-session expanding calibration. Actual early returns are retained. "
             "Missing marked returns remain gaps.", fontsize=9, color="#52645d", linespacing=1.5)
    fig.tight_layout(rect=(.025, .1, .985, .97))
    fig.savefig(output / "shuffled_admission_equity.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.4), sharex=True)
    for p in (80, 90, 95):
        axes[0].plot(thresholds.forecast_start, thresholds[f"p{p}"], lw=1.4,
                     color=COLORS[f"p{p}"], label=f"P{p} pooled null Sharpe")
    for name in LABELS:
        counts = admissions.loc[admissions.model.eq(name)].groupby("forecast_start").selected.sum()
        axes[1].step(counts.index, counts, where="post", lw=1.4, color=COLORS[name], label=LABELS[name])
    axes[0].set_title("What changes at each refit?", loc="left", fontsize=15, weight="bold", pad=15)
    axes[0].set_ylabel("Required raw training Sharpe")
    axes[1].set_ylabel("Admitted concepts / 10")
    axes[1].set_ylim(-.3, 10.5)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "shuffled_admission_thresholds.png", dpi=150)
    plt.close(fig)

    def picture(name):
        encoded = base64.b64encode((output / name).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{encoded}" alt="{name}">'

    columns = ["model", "sharpe", "annualized_return", "annualized_volatility", "total_return",
               "max_drawdown", "mean_admitted", "cash_fraction", "mean_daily_turnover", "observations"]
    base = metrics.loc[metrics.scenario.eq("base") & metrics.period.eq("full_common"), columns]
    stress = metrics.loc[metrics.scenario.eq("stressed") & metrics.period.eq("full_common"),
                         ["model", "sharpe", "annualized_return", "max_drawdown"]]
    mature = metrics.loc[metrics.scenario.eq("base") & metrics.period.eq("trailing504_common"), columns[:7] + ["observations"]]
    yearly = annual.pivot(index="year", columns="model", values="sharpe").reindex(columns=LABELS).reset_index()
    yearly_returns = annual.pivot(index="year", columns="model", values="year_return").reindex(columns=LABELS)
    yearly_returns = yearly_returns.map(lambda x: f"{x:.2%}" if pd.notna(x) else "-").reset_index()
    fractions = admissions.groupby(["model", "economic_family"]).selected.mean().unstack("model").reindex(columns=LABELS)
    fractions = fractions.map(lambda x: f"{x:.0%}").reset_index()
    text = f"""<!doctype html><html><head><meta charset="utf-8"><title>Five Currents | Shuffled Sharpe Admission</title>
<style>body{{font-family:Georgia,serif;background:#f5f5ef;color:#253f39;margin:0}}main{{max-width:1160px;margin:auto;padding:38px}}
h1{{font-size:34px;margin-bottom:8px}}h2{{margin-top:35px}}p,li{{line-height:1.55}}.lede{{font-size:19px;color:#52645d}}
section{{background:white;padding:25px;margin:22px 0;border:1px solid #d9ded5;border-radius:9px}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-family:Calibri,sans-serif;font-size:13px}}th,td{{padding:9px;text-align:right;border-bottom:1px solid #e0e4dc}}
th:first-child,td:first-child{{text-align:left}}th{{background:#eef3eb}}.scroll{{overflow-x:auto}}.note{{color:#52645d;font-size:14px}}</style></head><body><main>
<h1>Five Currents: Shuffled-Sharpe Admission</h1><p class="lede">An additional causal gate, not a new alpha search</p>
<p>Daily close-to-close portfolio across CL, NG, GC, SI and HG. Full signal-date history: {first_date:%d %b %Y} to {last_date:%d %b %Y}.
The displayed return for signal date T belongs to the established close T+1 to close T+2 execution convention.</p>
<section><h2>Full-history comparison</h2>{picture('shuffled_admission_equity.png')}
<div class="scroll">{formatted_table(base)}</div><p class="note">Statistics use common valid dates across all four versions, including genuine cash periods.
No leverage or volatility retargeting was added. CAGR is compounded annualized return, not arithmetic mean return.
Cash fraction means zero model exposure; marked-data gaps are not counted as cash.</p>
<h3>Doubled-friction control</h3>{formatted_table(stress)}
<h3>After the expanding window reaches 504 sessions</h3><p class="note">This is a secondary time slice, not a fresh untouched holdout.</p>{formatted_table(mature)}</section>
<section><h2>Training-only shuffled null</h2><ul>
<li>Every existing refit is retained: 21-session intervals, 126-session expanding warmup to 504-session trailing training, and the same two-session realization embargo.</li>
<li>Each concept must still pass the existing positive stressed-cost and half-window Sharpe gates and the positive robustness-score/activity requirements.</li>
<li>At each refit, {draws:,} five-session-block permutations jointly shuffle the five underlying forward-return series against fixed historical positions.
Prices are not downloaded, finished P&amp;L is not shuffled, and actual historical turnover/roll costs stay fixed.</li>
<li>Return vectors are permuted within their priced-asset availability pattern. Every date's missingness mask is preserved; no missing price is invented.</li>
<li>The ten concepts receive the same number of null draws. Their finite base-cost Sharpe values form one pooled distribution.
Admission additionally requires ordinary base-cost net Sharpe to exceed P80, P90 or P95 of that distribution, respectively.</li>
<li>Surviving concepts keep equal initial votes. The established inverse-volatility product scaling is relearned from the same training slice for that new mix, with unchanged gross/product caps.</li>
</ul>{picture('shuffled_admission_thresholds.png')}
<h3>Concept admission frequency</h3><div class="scroll">{formatted_table(fractions)}</div></section>
<section><h2>Year-by-year results</h2><h3>After-cost Sharpe</h3>{formatted_table(yearly)}
<h3>Compounded returns</h3>{formatted_table(yearly_returns)}</section>
<section><h2>Interpretation and controls</h2><p>The baseline is reproduced exactly against the saved dashboard weights.
All historical admission metrics and risk coefficients are checked against the archived refits before adding a gate.
The operational curve configuration is unchanged: SI/HG use ranks 1-4, CL/NG/GC use ranks 1-6.
The dashboard's already-anchored historical fits retain their original training definitions; this experiment does not quietly recalibrate that baseline.</p>
<p>These are additional research variants on already-studied history and a preselected ten-concept roster, not new out-of-sample evidence.
A pooled null percentile is not a family-wise maximum-Sharpe significance test. Five-session blocks preserve short-run return dependence but not all longer-term structure.
The same execution marks, contract migration, delayed implementation and base/doubled-cost assumptions are retained.</p>
<p class="note">Cached files only. No market API or database access, no live publications, and no order submission.
CSV/Parquet outputs, full per-refit decisions, shuffled samples and SHA-256 provenance accompany this report.</p></section>
</main></body></html>"""
    (output / "Five_Currents_Shuffled_Sharpe_Admission.html").write_text(text, encoding="utf-8")


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=False)
    cache = Path("five_currents_dashboard/private_replay_cache.pkl")
    cache_hash = sha256(cache)
    with cache.open("rb") as handle:
        replay = pickle.load(handle)
    snapshot = args.snapshot.resolve()
    selection = pd.read_parquet(snapshot / "selection_audit.parquet")
    archived_coefficients = pd.read_parquet(snapshot / "product_scaling.parquet")
    archived_weights = pd.read_parquet(snapshot / "weights.parquet")
    activation = Path("five_currents_dashboard/curve_policy_activation")
    anchor = pd.read_parquet(activation / "selection_audit.parquet")
    anchor_ordinal = int(anchor.refit_ordinal.max())
    index = replay.weights.index
    pd.testing.assert_index_equal(index, archived_weights.index)
    print("[baseline] Rebuilding only the cached operational curve features", flush=True)
    operational = mixed_concepts(replay)
    execution = dollar_cost_execution(replay.execution)
    calibrations = {"legacy": replay.concept_weights, "mixed": operational}
    families, family_costs = {}, {}
    for key, concepts in calibrations.items():
        families[key] = family_path_matrices(concepts, replay.panel, execution)
        family_costs[key] = np.stack([fast_product_backtest(w, replay.panel, execution, "base").cost.to_numpy()
                                     for w in concepts])
    score_columns = ("train_base_sharpe", "train_stressed_sharpe", "train_first_half_sharpe",
                     "train_second_half_sharpe", "train_active_days", "train_score")
    for ordinal, segment in selection.groupby("refit_ordinal", sort=True):
        key = "legacy" if ordinal <= anchor_ordinal else "mixed"
        positions = training_slice(index, segment.iloc[0])
        scored = candidate_training_scores(*families[key], positions)
        original = segment.set_index("economic_family").reindex(replay.concept_names)
        for column in score_columns:
            np.testing.assert_allclose(scored[column], original[column], atol=1e-10, rtol=1e-10)
        eligible = (np.isfinite(scored.train_score) & scored.train_score.gt(0)
                    & scored.train_stressed_sharpe.gt(0) & scored.train_first_half_sharpe.gt(0)
                    & scored.train_second_half_sharpe.gt(0))
        np.testing.assert_array_equal(eligible.to_numpy(), original.selected.to_numpy())
    recomputed = recalibrate_coefficients(selection, replay, operational, execution, anchor_ordinal)
    coeff_keys = ["refit_ordinal", "product"]
    np.testing.assert_allclose(
        recomputed.set_index(coeff_keys).inverse_volatility_coefficient.sort_index(),
        archived_coefficients.set_index(coeff_keys).inverse_volatility_coefficient.sort_index(),
        atol=1e-10, rtol=1e-10,
    )
    baseline = frozen_allocation(operational, replay.concept_names, index, selection, recomputed)
    np.testing.assert_allclose(baseline, archived_weights.to_numpy(), atol=1e-10, rtol=1e-10)
    print("[baseline] All archived admission scores, decisions, risk coefficients and weights reproduced", flush=True)

    threshold_rows, null_rows = [], []
    grouped = list(selection.groupby("refit_ordinal", sort=True))
    for number, (ordinal, segment) in enumerate(grouped, 1):
        first = segment.iloc[0]
        positions = training_slice(index, first)
        key = "legacy" if ordinal <= anchor_ordinal else "mixed"
        base, stress, exposure = families[key]
        null = pooled_null_sharpes(calibrations[key][:, positions], execution.target_return[positions],
                                  family_costs[key][:, positions], np.isfinite(base[:, positions]),
                                  draws=args.draws, seed=args.seed + int(ordinal), block_size=5)
        pooled = null[np.isfinite(null)]
        if not len(pooled):
            raise AssertionError("Empty pooled null")
        cuts = np.quantile(pooled, [.8, .9, .95])
        threshold_rows.append({"refit_ordinal": ordinal, "forecast_start": first.forecast_start,
                               "train_start": first.train_start, "train_end": first.train_end,
                               "training_sessions": len(positions), "p80": cuts[0], "p90": cuts[1],
                               "p95": cuts[2], "null_observations": len(pooled),
                               "draws_per_concept": args.draws, "calibration_definition": key})
        null_rows.append(pd.DataFrame(null, columns=replay.concept_names).assign(refit_ordinal=ordinal, draw=np.arange(args.draws)))
        print(f"[refit {number:02d}/{len(grouped)}] {first.forecast_start:%Y-%m-%d} | "
              f"train {first.train_start:%Y-%m-%d} to {first.train_end:%Y-%m-%d} | "
              f"null P80/P90/P95 = {cuts[0]:.3f}/{cuts[1]:.3f}/{cuts[2]:.3f}", flush=True)
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(output / "refit_null_thresholds.csv", index=False)
    pd.concat(null_rows, ignore_index=True).to_parquet(output / "null_sharpe_samples.parquet", index=False)
    threshold_map = {p: apply_admission_gate(selection, thresholds, p) for p in (80, 90, 95)}
    for p, stricter in ((90, 80), (95, 90)):
        assert (threshold_map[p].selected <= threshold_map[stricter].selected).all()

    admissions, all_paths, all_weights, all_coefficients = [], [], [], []
    for model in LABELS:
        selected = selection.copy() if model == "baseline" else threshold_map[int(model[1:])]
        coefficients = recalibrate_coefficients(selected, replay, operational, execution, anchor_ordinal)
        weights = frozen_allocation(operational, replay.concept_names, index, selected, coefficients)
        assert (np.abs(weights).sum(axis=1) <= 1 + 1e-10).all()
        assert (np.abs(weights) <= .4 + 1e-10).all()
        admissions.append(selected.assign(model=model))
        all_coefficients.append(coefficients.assign(model=model))
        all_weights.append(pd.DataFrame(weights, index=index, columns=PRODUCT_ORDER).rename_axis("session")
                           .reset_index().assign(model=model))
        for scenario in ("base", "stressed"):
            path = fast_product_backtest(weights, replay.panel, execution, scenario)
            path = cash_before_first_trade(path, replay.first_causal_date)
            all_paths.append(path.rename_axis("session").reset_index().assign(model=model, scenario=scenario))
    admissions = pd.concat(admissions, ignore_index=True)
    paths = pd.concat(all_paths, ignore_index=True)
    first_date, last_date = replay.first_causal_date, replay.last_realised_date
    paths = paths.loc[paths.session.between(first_date, last_date)].copy()
    admissions.to_parquet(output / "admission_audit.parquet", index=False)
    paths.to_parquet(output / "portfolio_daily_paths.parquet", index=False)
    pd.concat(all_weights, ignore_index=True).to_parquet(output / "model_weights.parquet", index=False)
    pd.concat(all_coefficients, ignore_index=True).to_parquet(output / "product_scaling.parquet", index=False)

    metric_rows, annual_rows = [], []
    mature_start = selection.loc[selection.training_sessions.eq(504), "forecast_start"].min()
    for scenario in ("base", "stressed"):
        subset = paths.loc[paths.scenario.eq(scenario)]
        pivot = subset.pivot(index="session", columns="model", values="net_return")
        common = pivot.dropna().index
        for model, group in subset.groupby("model", sort=False):
            group = group.set_index("session")
            selected = admissions.loc[admissions.model.eq(model)]
            mean_count = selected.groupby("refit_ordinal").selected.sum().mean()
            for period, dates in (("full_common", common), ("full_native", group.net_return.dropna().index),
                                  ("trailing504_common", common[common >= mature_start])):
                sample = group.loc[dates]
                metric_rows.append({"model": model, "scenario": scenario, "period": period,
                                    **_series_metrics(sample.net_return), "mean_admitted": mean_count,
                                    "cash_fraction": float(sample.gross_exposure.le(1e-12).mean()),
                                    "mean_gross_exposure": float(sample.gross_exposure.mean()),
                                    "mean_daily_turnover": float(sample.turnover.mean()),
                                    "excluded_marked_sessions": len(pivot) - len(dates)})
            if scenario == "base":
                for year, values in group.loc[common, "net_return"].groupby(common.year):
                    summary = _series_metrics(values)
                    annual_rows.append({"model": model, "year": year, "sharpe": summary["sharpe"],
                                        "year_return": summary["total_return"], "observations": len(values)})
    metrics, annual = pd.DataFrame(metric_rows), pd.DataFrame(annual_rows)
    metrics.to_csv(output / "performance_summary.csv", index=False)
    annual.to_csv(output / "yearly_performance.csv", index=False)
    render(output, metrics, annual, paths, thresholds, admissions, first_date, last_date, args.draws)
    verification = {"baseline_scores_and_admissions_match": True, "baseline_risk_coefficients_match": True,
                    "baseline_weights_match": True, "gates_nested_and_additional": True,
                    "maximum_gross": 1., "product_cap": .4, "embargo_sessions": 2,
                    "no_future_inputs_to_null": True, "cache_unchanged": cache_hash == sha256(cache)}
    source_files = [cache, snapshot / "selection_audit.parquet", snapshot / "product_scaling.parquet",
                    snapshot / "weights.parquet", activation / "manifest.json", Path(__file__),
                    Path("five_currents_shuffled_admission.py"), Path("test_five_currents_shuffled_admission.py")]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "snapshot": str(snapshot),
                "source_sha256": {str(p.resolve()): (cache_hash if p == cache else sha256(p)) for p in source_files},
                "null_method": "Five-session return block permutations jointly across assets, within availability patterns; fixed signal weights and costs",
                "null_draws_per_concept_per_refit": args.draws, "seed": args.seed,
                "thresholds": [80, 90, 95], "gate_basis": "Base-cost net Sharpe before robustness-score weighting",
                "additional_gate_only": True, "parameter_search": False, "retune_cadence_sessions": 21,
                "lookback_sessions": 504, "expanding_minimum_sessions": 126,
                "market_api_calls": False, "database_access": False, "production_changes": False,
                "verification": verification,
                "outputs_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nFull-period, common-date base-cost results:", flush=True)
    print(metrics.loc[metrics.scenario.eq("base") & metrics.period.eq("full_common"),
                      ["model", "sharpe", "annualized_return", "total_return", "max_drawdown", "mean_admitted", "cash_fraction"]]
          .to_string(index=False), flush=True)
    print(f"Report: {output / 'Five_Currents_Shuffled_Sharpe_Admission.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path,
                        default=Path("five_currents_dashboard/snapshots/000033_2026-10-09"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("five_currents_shuffled_admission_20261012"))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    run(args)
