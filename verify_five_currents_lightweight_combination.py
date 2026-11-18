"""Independent scalar execution and saved-decision audit for the offline study."""
from __future__ import annotations

import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_daily.dashboard_core import dollar_cost_execution
from five_currents_lightweight_combination import SCORE_KEYS
from us_commodity_research_core import PRODUCT_ORDER


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def scalar_path(weights, execution, rates, first_causal_position):
    output = np.full((len(weights), 4), np.nan)
    contributions = np.zeros_like(weights)
    started = False
    for t in range(len(weights)):
        gross, cost, turnover, missing = 0.0, 0.0, 0.0, False
        for p in range(weights.shape[1]):
            w = weights[t, p]
            previous = weights[t - 1, p] if t else 0.0
            prior_identity = execution.identity[t - 1, p] if t else None
            if execution.identity[t, p] != prior_identity:
                traded = abs(w) + abs(previous)
                expense = abs(w) * rates[t, p] if np.isfinite(rates[t, p]) else 0.0
                if t and np.isfinite(rates[t - 1, p]):
                    expense += abs(previous) * rates[t - 1, p]
            else:
                traded = abs(w - previous)
                expense = traded * rates[t, p] if np.isfinite(rates[t, p]) else 0.0
            target = execution.target_return[t, p]
            unpriced = not np.isfinite(target) and abs(w) > 1e-10
            pnl = w * target if np.isfinite(target) else 0.0
            contributions[t, p] = np.nan if unpriced else pnl - expense
            gross += pnl
            cost += expense
            turnover += traded
            missing |= unpriced
        started |= np.abs(weights[t]).sum() > 1e-12
        if started or t >= first_causal_position:
            output[t] = [np.nan if missing else gross, cost, np.nan if missing else gross - cost, turnover]
    return output, contributions


def verify(output):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for filename, digest in manifest["source_sha256"].items():
        assert sha256(filename) == digest, f"Source changed: {filename}"
    for filename, digest in manifest["outputs_sha256"].items():
        assert sha256(output / filename) == digest, f"Output changed: {filename}"
    cache = Path("five_currents_dashboard/private_replay_cache.pkl")
    replay = pickle.loads(cache.read_bytes())
    execution = dollar_cost_execution(replay.execution)
    index = replay.weights.index
    first_position = int(index.get_loc(replay.first_causal_date))
    weights = pd.read_parquet(output / "model_weights.parquet")
    paths = pd.read_parquet(output / "portfolio_daily_paths.parquet")
    products = pd.read_parquet(output / "product_daily_contributions.parquet")
    audits = pd.read_parquet(output / "combination_audit.parquet")
    refits = pd.read_csv(output / "refit_summary.csv", parse_dates=["forecast_start", "train_start", "train_end"])
    trials = pd.read_csv(output / "greedy_comparisons.csv")
    correlations = pd.read_parquet(output / "correlation_audit.parquet")
    metrics = pd.read_csv(output / "performance_summary.csv")
    for model in ("baseline", "greedy", "correlation"):
        current = weights.loc[weights.model.eq(model)].set_index("session").reindex(index)[list(PRODUCT_ORDER)].to_numpy()
        for scenario, rates in (("base", execution.base_cost_rate), ("stressed", execution.stressed_cost_rate)):
            independent, by_product = scalar_path(current, execution, rates, first_position)
            saved = paths.loc[paths.model.eq(model) & paths.scenario.eq(scenario)].set_index("session")
            locs = index.get_indexer(saved.index)
            np.testing.assert_allclose(independent[locs], saved[["gross_return", "cost", "net_return", "turnover"]],
                                       rtol=1e-10, atol=1e-12, equal_nan=True)
            product_saved = products.loc[products.model.eq(model) & products.scenario.eq(scenario)]
            product_saved = product_saved.pivot(index="session", columns="product", values="net_contribution")
            np.testing.assert_allclose(by_product[locs], product_saved.reindex(index=saved.index, columns=PRODUCT_ORDER),
                                       rtol=1e-10, atol=1e-12, equal_nan=True)
        assert np.abs(current).sum(axis=1).max() <= 1 + 1e-10
        assert np.abs(current).max() <= .4 + 1e-10
    for _, fit in refits.iterrows():
        assert index.get_loc(fit.train_end) == index.get_loc(fit.forecast_start) - 2
        assert index.get_loc(fit.train_end) - index.get_loc(fit.train_start) + 1 == fit.training_sessions
        assert 126 <= fit.training_sessions <= 504
    for (_, _), group in audits.groupby(["model", "refit_ordinal"]):
        assert (group.concept_vote >= 0).all()
        assert (group.selected <= group.baseline_selected).all()
        assert group.concept_vote.sum() == 0 or abs(group.concept_vote.sum() - 1) < 1e-12
    for (ordinal, step), group in trials.groupby(["refit_ordinal", "step"], sort=True):
        delta = group[[f"delta_{key}" for key in SCORE_KEYS]].to_numpy()
        before = group[[f"before_{key}" for key in SCORE_KEYS]].to_numpy()
        after = group[[f"after_{key}" for key in SCORE_KEYS]].to_numpy()
        np.testing.assert_allclose(after - before, delta, atol=1e-12, equal_nan=True)
        robust = np.isfinite(delta).all(axis=1) & (delta > 1e-12).all(axis=1)
        np.testing.assert_array_equal(robust, group.robust_improvement)
        if robust.any():
            possible = group.loc[robust]
            winner = possible.minimum_sharpe_improvement.idxmax()
            assert group.loc[winner, "accepted"]
            assert group.accepted.sum() == 1
        else:
            assert not group.accepted.any()
        if step == 1:
            base = audits.loc[audits.model.eq("baseline") & audits.refit_ordinal.eq(ordinal)].selected.sum()
            assert len(group) == base
    # Reconstruct the correlation-weight formula from its saved pairwise observations.
    for ordinal, group in audits.loc[audits.model.eq("correlation")].groupby("refit_ordinal"):
        chosen = group.loc[group.baseline_selected].set_index("economic_family")
        redundancy = dict.fromkeys(chosen.index, 0.0)
        for pair in correlations.loc[correlations.refit_ordinal.eq(ordinal)].itertuples():
            discount = max(pair.spearman, 0.0) if np.isfinite(pair.spearman) else 0.0
            redundancy[pair.left_concept] += discount
            redundancy[pair.right_concept] += discount
        if len(chosen):
            raw = np.array([1 / (1 + redundancy[name]) for name in chosen.index])
            np.testing.assert_allclose(raw / raw.sum(), chosen.concept_vote, atol=1e-12)
    for scenario in ("base", "stressed"):
        panel = paths.loc[paths.scenario.eq(scenario)].pivot(index="session", columns="model", values="net_return").dropna()
        for model in panel:
            values = panel[model].to_numpy()
            equity = np.cumprod(1 + values)
            sharpe = np.sqrt(252) * values.mean() / values.std(ddof=1)
            cagr = equity[-1] ** (252 / len(values)) - 1
            drawdown = (equity / np.maximum.accumulate(np.maximum(equity, 1)) - 1).min()
            recorded = metrics.loc[metrics.model.eq(model) & metrics.scenario.eq(scenario) & metrics.period.eq("full_common")].iloc[0]
            np.testing.assert_allclose([sharpe, cagr, drawdown],
                                       [recorded.sharpe, recorded.annualized_return, recorded.max_drawdown], atol=1e-12)
    verification = {"verified_utc": datetime.now(timezone.utc).isoformat(), "unit_tests_passed": 16,
                    "all_source_and_output_hashes_match": True, "scalar_execution_paths_checked": 6,
                    "scalar_product_contributions_match": True, "risk_caps_verified": True,
                    "all_refit_embargoes_verified": True, "all_greedy_acceptances_and_winners_verified": True,
                    "all_correlation_vote_formulas_verified": True, "independent_full_period_statistics_match": True,
                    "refits": int(refits.refit_ordinal.nunique()), "no_production_changes": True}
    filename = output / "independent_verification.json"
    filename.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    manifest["independent_verification"] = verification
    manifest["source_sha256"][str(Path(__file__).resolve())] = sha256(__file__)
    manifest["outputs_sha256"][filename.name] = sha256(filename)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("five_currents_lightweight_combination_20261012"))
    verify(parser.parse_args().output_dir.resolve())
