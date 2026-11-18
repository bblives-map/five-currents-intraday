"""Independent checks of cached-data randomization, admissions and daily accounting."""
from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_daily.dashboard_core import dollar_cost_execution
from five_currents_daily_reconciliation_audit import independent_daily_ledger
from five_currents_shuffled_admission import sharpes_from_permutations, training_slice
from us_commodity_disciplined_core import fast_product_backtest


def run(output):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["outputs_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected, name
    cache = Path("five_currents_dashboard/private_replay_cache.pkl")
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == manifest["source_sha256"][str(cache.resolve())]
    with cache.open("rb") as handle:
        replay = pickle.load(handle)
    execution = dollar_cost_execution(replay.execution)
    admissions = pd.read_parquet(output / "admission_audit.parquet")
    null = pd.read_parquet(output / "null_sharpe_samples.parquet")
    thresholds = pd.read_csv(output / "refit_null_thresholds.csv")
    baseline = admissions.loc[admissions.model.eq("baseline")]
    paths = pd.read_parquet(output / "portfolio_daily_paths.parquet")
    weights = pd.read_parquet(output / "model_weights.parquet")
    metrics = pd.read_csv(output / "performance_summary.csv")
    families = [fast_product_backtest(w, replay.panel, execution, "base") for w in replay.concept_weights]
    costs = np.stack([p.cost for p in families])
    valid = np.stack([p.net_return.notna() for p in families])
    max_identity_error = 0.
    for ordinal, group in baseline.groupby("refit_ordinal", sort=True):
        positions = training_slice(replay.weights.index, group.iloc[0])
        original_order = np.arange(len(positions))[None, :]
        actual = sharpes_from_permutations(replay.concept_weights[:, positions],
                                          execution.target_return[positions], costs[:, positions],
                                          valid[:, positions], original_order)[0]
        expected = group.set_index("economic_family").reindex(replay.concept_names).train_base_sharpe.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True)
        errors = abs(actual - expected)
        if np.isfinite(errors).any():
            max_identity_error = max(max_identity_error, float(errors[np.isfinite(errors)].max()))
        values = null.loc[null.refit_ordinal.eq(ordinal), list(replay.concept_names)].to_numpy()
        values = values[np.isfinite(values)]
        cuts = np.quantile(values, [.8, .9, .95])
        recorded = thresholds.loc[thresholds.refit_ordinal.eq(ordinal)].iloc[0]
        np.testing.assert_allclose(cuts, recorded[["p80", "p90", "p95"]].to_numpy(float), atol=1e-12)
    for p in (80, 90, 95):
        group = admissions.loc[admissions.model.eq(f"p{p}")]
        cuts = group.refit_ordinal.map(thresholds.set_index("refit_ordinal")[f"p{p}"])
        expected = group.baseline_selected & group.train_base_sharpe.gt(cuts)
        np.testing.assert_array_equal(group.selected, expected)
        assert group.cash_fallback.eq(group.selected_family_count.eq(0)).all()

    max_accounting_error = 0.
    for model, frame in weights.groupby("model", sort=False):
        w = frame.set_index("session")[list(replay.weights.columns)]
        for scenario, rates in (("base", execution.base_cost_rate), ("stressed", execution.stressed_cost_rate)):
            ledger = independent_daily_ledger(w, execution.target_return, execution.identity, rates)
            group = ledger.groupby("signal_date")
            gross = group.gross_return.sum(min_count=len(w.columns))
            cost = group.cost_return.sum()
            net = gross - cost
            pre_entry_cash = ~w.abs().sum(axis=1).gt(1e-12).cummax()
            net.loc[pre_entry_cash & net.index.to_series().ge(replay.first_causal_date)] = 0.
            actual = paths.loc[paths.model.eq(model) & paths.scenario.eq(scenario)].set_index("session").net_return
            expected = net.reindex(actual.index)
            np.testing.assert_array_equal(actual.isna(), expected.isna())
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True)
            error = (actual - expected).abs().max()
            max_accounting_error = max(max_accounting_error, float(error))
    result = {"unit_tests_passed": 10, "refits_checked": int(baseline.refit_ordinal.nunique()),
              "all_output_hashes_verified": True, "live_cache_unchanged": True,
              "identity_permutation_matches_observed_training_sharpes": True,
              "maximum_identity_sharpe_error": max_identity_error,
              "saved_samples_reproduce_all_null_quantiles": True,
              "every_admission_matches_additional_gate": True,
              "independent_accounting_models_checked": 4, "independent_accounting_cost_scenarios": 2,
              "maximum_independent_accounting_error": max_accounting_error,
              "daily_missingness_matches_independent_accounting": True}
    path = output / "independent_verification.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest["independent_verification"] = result
    manifest["outputs_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    script = Path(__file__).resolve()
    manifest["source_sha256"][str(script)] = hashlib.sha256(script.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(metrics.loc[metrics.scenario.eq("base") & metrics.period.eq("full_common")].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("five_currents_shuffled_admission_20261012"))
    run(parser.parse_args().output_dir.resolve())
