"""Recalibrate concept admission and optional pruning using cached inputs only."""
from __future__ import annotations

import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_consistency_objectives import (
    OBJECTIVES, LABELS, objective_admission, objective_pruning, performance_consistency,
)
from five_currents_daily.curve_activation import mixed_concepts
from five_currents_daily.dashboard_core import dollar_cost_execution
from five_currents_daily.report import _series_metrics
from five_currents_lightweight_combination import TrainingBook, allocate_refits, normalize_votes
from five_currents_shuffled_admission import cash_before_first_trade, training_slice
from report_five_currents_consistency import render_report
from run_five_currents_lightweight_combination import sha256
from run_us_commodity_pooled_harmonization import family_path_matrices
from shfe_rolling_walkforward_core import candidate_training_scores
from us_commodity_asset_family_walkforward_core import product_net_returns
from us_commodity_disciplined_core import fast_product_backtest
from us_commodity_research_core import PRODUCT_ORDER


def book_key(votes):
    return "-".join(map(str, np.flatnonzero(votes))) or "cash"


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=False)
    cache = Path("five_currents_dashboard/private_replay_cache.pkl")
    content = cache.read_bytes()
    cache_hash = hashlib.sha256(content).hexdigest()
    replay = pickle.loads(content)
    del content
    snapshot = args.snapshot.resolve()
    selection = pd.read_parquet(snapshot / "selection_audit.parquet")
    archived_weights = pd.read_parquet(snapshot / "weights.parquet")
    archived_factors = pd.read_parquet(snapshot / "product_scaling.parquet")
    activation = Path("five_currents_dashboard/curve_policy_activation")
    anchor_ordinal = int(pd.read_parquet(activation / "selection_audit.parquet").refit_ordinal.max())
    index, names = replay.weights.index, tuple(replay.concept_names)
    pd.testing.assert_index_equal(index, archived_weights.index)
    outcome_dates = pd.Series(index, index=index).shift(-2)
    reference_dates = index[2:]
    print("[baseline] Rebuilding cached curves; no database or market API access", flush=True)
    operational = mixed_concepts(replay)
    execution = dollar_cost_execution(replay.execution)
    calibrations = {"legacy": replay.concept_weights, "mixed": operational}
    families = {key: family_path_matrices(values, replay.panel, execution) for key, values in calibrations.items()}
    audit_rows, coefficient_rows, fit_rows, pruning_rows = [], [], [], []
    concept_training, book_training = [], []
    grouped = list(selection.groupby("refit_ordinal", sort=True))
    for number, (ordinal, group) in enumerate(grouped, 1):
        first = group.iloc[0]
        positions = training_slice(index, first)
        dates = pd.DatetimeIndex(outcome_dates.iloc[positions])
        assert dates.max() == first.forecast_start
        calibration = "legacy" if ordinal <= anchor_ordinal else "mixed"
        base, stressed, exposure = [values[:, positions] for values in families[calibration]]
        original = group.set_index("economic_family").reindex(names)
        old_scores = candidate_training_scores(*families[calibration], positions)
        for column in ("train_base_sharpe", "train_stressed_sharpe", "train_first_half_sharpe",
                       "train_second_half_sharpe", "train_active_days", "train_score"):
            np.testing.assert_allclose(old_scores[column], original[column], atol=1e-10, rtol=1e-10)
        evaluator = TrainingBook(calibrations[calibration], index, execution, positions)
        for i, name in enumerate(names):
            concept_training.append(pd.DataFrame({"refit_ordinal": ordinal, "economic_family": name,
                                                  "session": index[positions], "realization_session": dates,
                                                  "base": base[i], "stressed": stressed[i], "exposure": exposure[i]}))
        admission_counts = []
        original_eligible = np.flatnonzero(original.selected.to_numpy()).tolist()
        for objective in OBJECTIVES:
            admission = objective_admission(base, stressed, exposure, dates, reference_dates,
                                            first.forecast_start, objective)
            if objective == "sharpe":
                np.testing.assert_array_equal(admission.selected, original.selected)
                np.testing.assert_allclose(admission.base_full, old_scores.train_base_sharpe, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(admission.fitness.where(admission.activity_eligible, -np.inf),
                                           old_scores.train_score, atol=1e-10, rtol=1e-10)
            eligible = np.flatnonzero(admission.selected).tolist()
            equal = np.zeros(len(names))
            equal[eligible] = 1
            equal = normalize_votes(equal)
            pruned, trials = objective_pruning(eligible, len(names), evaluator, dates,
                                               reference_dates, first.forecast_start, objective)
            for trial in trials:
                pruning_rows.append({"objective": objective, "refit_ordinal": ordinal,
                                     "mode": "pruned",
                                     "forecast_start": first.forecast_start,
                                     "economic_family": names[trial["removed_index"]], **trial})
            admission_counts.append(f"{objective}:{len(eligible)}/{(pruned > 0).sum()}")
            modes = [("admission", equal), ("pruned", pruned)]
            if objective != "sharpe":
                portfolio_votes, portfolio_trials = objective_pruning(
                    original_eligible, len(names), evaluator, dates, reference_dates, first.forecast_start, objective)
                modes.append(("portfolio_only", portfolio_votes))
                for trial in portfolio_trials:
                    pruning_rows.append({"objective": objective, "mode": "portfolio_only", "refit_ordinal": ordinal,
                                         "forecast_start": first.forecast_start,
                                         "economic_family": names[trial["removed_index"]], **trial})
            for mode, votes in modes:
                model = f"{objective}_{mode}"
                book = evaluator.evaluate(votes)
                if model == "sharpe_admission":
                    old = archived_factors.loc[archived_factors.refit_ordinal.eq(ordinal)].set_index("product")
                    np.testing.assert_allclose(book.coefficients, old.reindex(PRODUCT_ORDER).inverse_volatility_coefficient,
                                               atol=1e-10, rtol=1e-10)
                for i, name in enumerate(names):
                    audit_rows.append({**original.loc[name].to_dict(), **admission.iloc[i].to_dict(),
                                       "model": model, "mode": mode, "economic_family": name,
                                       "original_admitted": bool(original.selected.iloc[i]),
                                       "raw_objective_admitted": bool(admission.selected.iloc[i]),
                                       "objective_admitted": bool(original.selected.iloc[i]) if mode == "portfolio_only" else bool(admission.selected.iloc[i]),
                                       "admission_basis": "original_sharpe" if mode == "portfolio_only" else objective,
                                       "selected": bool(votes[i] > 0), "concept_vote": votes[i],
                                       "selected_family_count": int((votes > 0).sum()), "cash_fallback": not votes.any()})
                for i, product in enumerate(PRODUCT_ORDER):
                    coefficient_rows.append({"model": model, "refit_ordinal": ordinal, "product": product,
                                             "inverse_volatility_coefficient": book.coefficients[i],
                                             "training_annual_volatility": book.annual_volatility[i],
                                             "training_observations": book.observations[i]})
                fit_rows.append({"model": model, "objective": objective, "mode": mode,
                                 "refit_ordinal": ordinal, "forecast_start": first.forecast_start,
                                 "forecast_end": first.forecast_end, "train_start": first.train_start,
                                 "train_end": first.train_end, "last_training_outcome": dates.max(),
                                 "training_sessions": len(positions), "selected_concepts": int((votes > 0).sum()),
                                 "eligible_concepts": len(original_eligible) if mode == "portfolio_only" else len(eligible),
                                 "book_key": book_key(votes)})
        for book in evaluator.cache.values():
            book_training.append(pd.DataFrame({"refit_ordinal": ordinal, "book_key": book_key(book.votes),
                                               "session": index[positions], "realization_session": dates,
                                               "base": book.base, "stressed": book.stressed}))
        print(f"[refit {number:02d}/{len(grouped)}] {first.forecast_start:%Y-%m-%d} | "
              + " | ".join(admission_counts), flush=True)
    audit, factors, fits = pd.DataFrame(audit_rows), pd.DataFrame(coefficient_rows), pd.DataFrame(fit_rows)
    all_paths, all_products, all_weights = [], [], []
    previous_dir = Path("five_currents_lightweight_combination_20261012_final")
    previous_paths = pd.read_parquet(previous_dir / "portfolio_daily_paths.parquet")
    previous_weights = pd.read_parquet(previous_dir / "model_weights.parquet")
    for model in fits.model.unique():
        weights = allocate_refits(operational, names, index, audit.loc[audit.model.eq(model)], factors.loc[factors.model.eq(model)])
        if model in ("sharpe_admission", "sharpe_pruned"):
            old_model = "baseline" if model == "sharpe_admission" else "greedy"
            prior = previous_weights.loc[previous_weights.model.eq(old_model)].set_index("session").reindex(index)
            np.testing.assert_allclose(weights, prior[list(PRODUCT_ORDER)], atol=1e-10, rtol=1e-10)
        assert np.abs(weights).sum(axis=1).max() <= 1 + 1e-10
        assert np.abs(weights).max() <= .4 + 1e-10
        all_weights.append(pd.DataFrame(weights, index=index, columns=PRODUCT_ORDER).rename_axis("session").reset_index().assign(model=model))
        for scenario in ("base", "stressed"):
            path = cash_before_first_trade(fast_product_backtest(weights, replay.panel, execution, scenario), replay.first_causal_date)
            if model in ("sharpe_admission", "sharpe_pruned"):
                prior = previous_paths.loc[previous_paths.model.eq(old_model) & previous_paths.scenario.eq(scenario)].set_index("session")
                common = path.index.intersection(prior.index)
                np.testing.assert_allclose(path.loc[common, "net_return"], prior.loc[common, "net_return"],
                                           atol=1e-10, rtol=1e-10, equal_nan=True)
            saved = path.rename_axis("session").reset_index().assign(model=model, scenario=scenario)
            saved["realization_session"] = saved.session.map(outcome_dates)
            all_paths.append(saved)
            products = product_net_returns(weights, execution, scenario=scenario)
            started = np.maximum.accumulate(np.abs(weights) > 1e-12, axis=0)
            products[~started] = 0
            causal = index >= replay.first_causal_date
            np.testing.assert_allclose(products.sum(axis=1)[causal], path.net_return.to_numpy()[causal],
                                       atol=1e-12, rtol=1e-10, equal_nan=True)
            all_products.append(pd.DataFrame(products, index=index, columns=PRODUCT_ORDER).rename_axis("session").reset_index()
                                .melt(id_vars="session", var_name="product", value_name="net_contribution").assign(model=model, scenario=scenario))
    paths, products = pd.concat(all_paths, ignore_index=True), pd.concat(all_products, ignore_index=True)
    first_date, last_date = replay.first_causal_date, replay.last_realised_date
    paths = paths.loc[paths.session.between(first_date, last_date)].copy()
    products = products.loc[products.session.between(first_date, last_date)].copy()
    base_panel = paths.loc[paths.scenario.eq("base")].pivot(index="session", columns="model", values="net_return")
    complete_marks = pd.Series(np.isfinite(execution.target_return).all(axis=1), index=index)
    fixed_common = base_panel.notna().all(axis=1) & complete_marks.reindex(base_panel.index)
    baseline = base_panel.loc[fixed_common, "sharpe_admission"]
    k = max(1, math.ceil(.01 * len(baseline)))
    windfall_dates = baseline[baseline > 0].nlargest(k).index
    metric_rows, yearly_rows, monthly_rows = [], [], []
    mature_start = selection.loc[selection.training_sessions.eq(504), "forecast_start"].min()
    common_paths = []
    for scenario in ("base", "stressed"):
        for model, group in paths.loc[paths.scenario.eq(scenario)].groupby("model", sort=False):
            group = group.set_index("session")
            dates = group.index[fixed_common.reindex(group.index)]
            sample_fits = fits.loc[fits.model.eq(model)]
            for period, selected_dates in (("full_common", dates), ("full_native", group.net_return.dropna().index),
                                          ("trailing504_common", dates[dates >= mature_start])):
                sample = group.loc[selected_dates]
                values = sample.net_return
                summary = _series_metrics(values)
                equity = (1 + values).cumprod()
                summary["max_drawdown"] = float((equity / equity.cummax().clip(lower=1) - 1).min())
                axis = group.index[group.index >= (mature_start if period == "trailing504_common" else first_date)]
                consistency_values = group.net_return.reindex(axis)
                if period != "full_native":
                    consistency_values = consistency_values.where(fixed_common.reindex(axis))
                realized = pd.DatetimeIndex(outcome_dates.reindex(axis))
                consistency = performance_consistency(consistency_values.to_numpy(), realized, reference_dates,
                                                       realized.max())
                metric_rows.append({"model": model, "objective": sample_fits.iloc[0].objective, "mode": sample_fits.iloc[0]["mode"],
                                    "scenario": scenario, "period": period, **summary, **consistency,
                                    "mean_admitted": sample_fits.selected_concepts.mean(),
                                    "cash_fraction": sample.gross_exposure.le(1e-12).mean(),
                                    "mean_gross_exposure": sample.gross_exposure.mean(),
                                    "mean_daily_turnover": sample.turnover.mean(), "excluded_marked_sessions": len(axis) - len(sample)})
            common_series = group.net_return.where(fixed_common)
            no_windfall = common_series.copy()
            no_windfall.loc[windfall_dates] = np.minimum(no_windfall.loc[windfall_dates], 0.0)
            stress_summary = _series_metrics(no_windfall)
            stress_equity = (1 + no_windfall.dropna()).cumprod()
            stress_summary["max_drawdown"] = float((stress_equity / stress_equity.cummax().clip(lower=1) - 1).min())
            metric_rows.append({"model": model, "objective": sample_fits.iloc[0].objective, "mode": sample_fits.iloc[0]["mode"],
                                "scenario": scenario, "period": "shared_windfall_ablation", **stress_summary})
            common_paths.append(pd.DataFrame({"session": group.index, "realization_session": group.realization_session,
                                              "model": model, "scenario": scenario, "net_return": common_series,
                                              "no_windfall_return": no_windfall, "common_marked": fixed_common,
                                              "windfall_date": group.index.isin(windfall_dates)}))
            if scenario == "base":
                year_axis = pd.DatetimeIndex(group.realization_session)
                for year in sorted(set(year_axis.year)):
                    values = common_series.loc[year_axis.year == year]
                    summary = _series_metrics(values)
                    yearly_rows.append({"model": model, "year": year, "sharpe": summary["sharpe"],
                                        "year_return": summary["total_return"], "observations": summary["observations"]})
                from five_currents_consistency_objectives import completed_period_returns
                month_values = completed_period_returns(common_series.to_numpy(), year_axis, reference_dates,
                                                        year_axis.max(), "M")
                monthly_rows.extend({"model": model, "month": date, "net_return": value} for date, value in month_values.items())
    metrics, annual = pd.DataFrame(metric_rows), pd.DataFrame(yearly_rows)
    common_paths = pd.concat(common_paths, ignore_index=True)
    metrics.to_csv(output / "performance_summary.csv", index=False)
    annual.to_csv(output / "yearly_performance.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(output / "monthly_returns.csv", index=False)
    audit.to_parquet(output / "selection_audit.parquet", index=False)
    factors.to_parquet(output / "product_scaling.parquet", index=False)
    fits.to_csv(output / "refit_summary.csv", index=False)
    pd.DataFrame(pruning_rows).to_csv(output / "pruning_comparisons.csv", index=False)
    pd.concat(concept_training, ignore_index=True).to_parquet(output / "concept_training_paths.parquet", index=False)
    pd.concat(book_training, ignore_index=True).drop_duplicates(["refit_ordinal", "book_key", "session"]).to_parquet(
        output / "book_training_paths.parquet", index=False)
    pd.DataFrame({"realization_session": reference_dates}).to_parquet(output / "reference_calendar.parquet", index=False)
    paths.to_parquet(output / "portfolio_daily_paths.parquet", index=False)
    common_paths.to_parquet(output / "common_daily_paths.parquet", index=False)
    products.to_parquet(output / "product_daily_contributions.parquet", index=False)
    pd.concat(all_weights, ignore_index=True).to_parquet(output / "model_weights.parquet", index=False)
    pd.DataFrame({"session": windfall_dates, "realization_session": outcome_dates.reindex(windfall_dates).to_numpy(),
                  "baseline_net_return": baseline.reindex(windfall_dates).to_numpy()}).sort_values("session").to_csv(
        output / "windfall_dates.csv", index=False)
    render_report(output, metrics, annual, common_paths, products, audit, fits, mature_start)
    sources = [cache, snapshot / "selection_audit.parquet", snapshot / "product_scaling.parquet", snapshot / "weights.parquet",
               activation / "manifest.json", Path(__file__), Path("five_currents_consistency_objectives.py"),
               Path("report_five_currents_consistency.py"), Path("test_five_currents_consistency_objectives.py"),
               Path("five_currents_lightweight_combination.py"), previous_dir / "manifest.json"]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": {
                    str(path.resolve()): cache_hash if path == cache else sha256(path) for path in sources},
                "snapshot": str(snapshot), "objectives": LABELS, "selection_modes": ["admission", "pruned", "portfolio_only"],
                "portfolio_only_control": "Original Sharpe admission pool retained; only combined-book pruning fitness changes",
                "concepts": names, "refits": len(grouped), "lookback_sessions": 504, "refit_sessions": 21,
                "minimum_training_sessions": 126, "embargo_sessions": 2, "upside_cap_quantile": .95,
                "rolling_sharpe_sessions": 63, "rolling_minimum_marked_observations": 50,
                "win_rate_gate": "positive-period fraction > 0.5; >=2 nonflat complete periods per half; positive full stressed expectancy",
                "alpha_feature_parameters_unchanged": True, "original_performance_gates_replaced": True,
                "objective_grid_search": False, "execution_convention": "forecast T; enter close T+1; realize close T+2",
                "calendar_returns_grouped_by": "realization close date; incomplete/unpriced periods excluded",
                "windfall_ablation": "positive returns set to zero on baseline's best 1% common marked dates; all losses retained; diagnostic only",
                "market_api_calls": False, "database_access": False, "production_changes": False,
                "verification": {"baseline_admission_scores_match": True, "baseline_weights_and_returns_match": True,
                                 "previous_sharpe_pruning_reproduced": True, "all_training_outcomes_available_at_refit": True,
                                 "all_product_contributions_reconcile": True, "cache_unchanged": sha256(cache) == cache_hash},
                "outputs_sha256": {path.name: sha256(path) for path in output.iterdir() if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nFull common-date results (actual, uncapped returns):", flush=True)
    print(metrics.loc[metrics.scenario.eq("base") & metrics.period.eq("full_common"),
                      ["model", "sharpe", "annualized_return", "max_drawdown", "weekly_win_rate", "monthly_win_rate",
                       "median_quarterly_sharpe", "mean_admitted"]].to_string(index=False), flush=True)
    print(f"Report: {output / 'Five_Currents_Consistency_Objectives.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path("five_currents_dashboard/snapshots/000033_2026-10-09"))
    parser.add_argument("--output-dir", type=Path, default=Path("five_currents_consistency_objectives_20261012"))
    run(parser.parse_args())
