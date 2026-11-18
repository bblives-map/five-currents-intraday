"""Cached-only causal comparison; never publishes to the production dashboard."""
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
from five_currents_daily.dashboard_core import dollar_cost_execution
from five_currents_daily.report import CONCEPT_LABELS, _series_metrics
from five_currents_lightweight_combination import (
    SCORE_KEYS, TrainingBook, allocate_refits, correlation_discount,
    greedy_backward, normalize_votes, residual_diagnostic,
)
from five_currents_shuffled_admission import cash_before_first_trade, training_slice
from run_five_currents_shuffled_admission import formatted_table, recalibrate_coefficients
from run_us_commodity_pooled_harmonization import family_path_matrices
from shfe_rolling_walkforward_core import candidate_training_scores
from us_commodity_asset_family_walkforward_core import product_net_returns
from us_commodity_disciplined_core import fast_product_backtest
from us_commodity_research_core import PRODUCT_ORDER


LABELS = {"baseline": "Existing equal votes", "greedy": "Greedy backward pruning",
          "correlation": "Correlation-discounted votes"}
COLORS = {"baseline": "#23495c", "greedy": "#178475", "correlation": "#bc833b"}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_report(output, metrics, annual, paths, products, audits, greedy, residuals,
                names, first_date, last_date, mature_start):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fafbf7",
                         "axes.facecolor": "#fafbf7", "text.color": "#233f3b"})
    panel = paths.loc[paths.scenario.eq("base")].pivot(index="session", columns="model", values="net_return")
    common = panel.notna().all(axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 8.1), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1]})
    for model in LABELS:
        equity = 100 * (1 + panel[model].where(common)).cumprod()
        # Drawdown includes starting capital, rather than starting after the first loss.
        peak = equity.cummax().clip(lower=100)
        record = metrics.loc[metrics.model.eq(model) & metrics.scenario.eq("base")
                              & metrics.period.eq("full_common")].iloc[0]
        axes[0].plot(equity.index, equity, color=COLORS[model], lw=1.6,
                     label=f"{LABELS[model]} | SR {record.sharpe:.2f}")
        axes[1].plot(equity.index, 100 * (equity / peak - 1), color=COLORS[model], lw=1.1)
    for ax in axes:
        ax.axvspan(first_date, mature_start, color="#d6e1d5", alpha=.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_title("Five Currents | simpler portfolio combination", loc="left", fontsize=17, weight="bold", pad=18)
    axes[0].set_ylabel("Growth of 100, after costs")
    axes[0].legend(frameon=False, fontsize=10)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlim(first_date, last_date)
    fig.text(.08, .025, "Same ten concepts, 504/21 cadence, product risk scaling, execution and costs. No extra leverage.\n"
             "Shaded: expanding 126-to-504-session training. Realized early performance is retained; unavailable marks remain gaps.",
             fontsize=9, color="#52645d", linespacing=1.5)
    fig.tight_layout(rect=(.02, .085, .99, .98))
    fig.savefig(output / "combination_equity.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12.4, 10.4), sharex=True)
    for ax, product in zip(axes.flat, PRODUCT_ORDER, strict=False):
        for model in LABELS:
            sample = products.loc[products.scenario.eq("base") & products.model.eq(model)
                                  & products["product"].eq(product)].set_index("session").net_contribution
            sample = sample.reindex(panel.index).where(common)
            ax.plot(sample.index, 100 * sample.cumsum(), color=COLORS[model], lw=1.3, label=LABELS[model])
        ax.set_title(product, loc="left", weight="bold")
        ax.axhline(0, lw=.6, color="#79837d")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_ylabel("Cumulative contribution (pp)")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle("Which commodities explain the difference?", x=.07, ha="left", fontsize=17, weight="bold")
    fig.tight_layout(rect=(.02, .02, .99, .96))
    fig.savefig(output / "combination_by_commodity.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12.4, 9), sharex=True)
    for ax, model in zip(axes, LABELS, strict=True):
        grid = audits.loc[audits.model.eq(model)].pivot(index="economic_family", columns="forecast_start", values="concept_vote")
        grid = grid.reindex(names)
        dates = mdates.date2num(pd.DatetimeIndex(grid.columns))
        ax.imshow(grid.to_numpy(), aspect="auto", interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=.5,
                  extent=[dates[0], dates[-1] + 21, len(names) - .5, -.5])
        ax.set_yticks(range(len(names)), [CONCEPT_LABELS.get(n, n) for n in names], fontsize=8)
        ax.set_title(LABELS[model], loc="left", fontsize=12, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].xaxis_date()
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Concept votes fixed at each refit", x=.28, ha="left", fontsize=16, weight="bold")
    fig.text(.29, .02, "Blank: not admitted. Darker: larger concept vote before the unchanged product-risk overlay.", fontsize=9)
    fig.tight_layout(rect=(.015, .05, .99, .96))
    fig.savefig(output / "combination_votes.png", dpi=150)
    plt.close(fig)

    def picture(filename):
        data = base64.b64encode((output / filename).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" alt="{filename}">'

    full = metrics.loc[metrics.period.eq("full_common") & metrics.scenario.eq("base")].copy()
    cols = ["model", "sharpe", "annualized_return", "annualized_volatility", "total_return", "max_drawdown",
            "mean_admitted", "mean_effective_concepts", "mean_daily_turnover", "observations"]
    baseline = full.set_index("model").loc["baseline"]
    commentary = []
    for model in ("greedy", "correlation"):
        row = full.set_index("model").loc[model]
        direction = "improves" if row.sharpe > baseline.sharpe else "reduces"
        commentary.append(f"{LABELS[model]} {direction} full-period net Sharpe from {baseline.sharpe:.2f} to {row.sharpe:.2f}; "
                          f"CAGR is {row.annualized_return:.2%}, versus {baseline.annualized_return:.2%} for the baseline.")
    changed = int(greedy.loc[greedy.accepted, "refit_ordinal"].nunique()) if len(greedy) else 0
    nrefits = audits.refit_ordinal.nunique()
    fractions = audits.groupby(["economic_family", "model"]).selected.mean().unstack("model").reindex(names)
    fractions = fractions.reindex(columns=LABELS).map(lambda x: f"{x:.0%}").reset_index()
    diagnostic = residuals.groupby("economic_family").agg(
        refits=("residual_sharpe", "count"), median_residual_sharpe=("residual_sharpe", "median"),
        fraction_positive=("residual_sharpe", lambda x: x.dropna().gt(0).mean())).reset_index()
    diagnostic.fraction_positive = diagnostic.fraction_positive.map(lambda x: f"{x:.0%}")
    asset_summary = []
    common_dates = panel.index[common]
    for (model, product), group in products.loc[products.scenario.eq("base")].groupby(["model", "product"]):
        values = group.set_index("session").net_contribution.reindex(common_dates)
        asset_summary.append({"model": model, "product": product, "net_contribution_pp": 100 * values.sum(),
                              "standalone_contribution_sharpe": _series_metrics(values)["sharpe"]})
    pd.DataFrame(asset_summary).to_csv(output / "asset_contribution_summary.csv", index=False)
    annual_sr = annual.pivot(index="year", columns="model", values="sharpe").reindex(columns=LABELS).reset_index()
    annual_ret = annual.pivot(index="year", columns="model", values="year_return").reindex(columns=LABELS)
    annual_ret = annual_ret.map(lambda x: f"{x:.2%}").reset_index()
    stress = metrics.loc[metrics.period.eq("full_common") & metrics.scenario.eq("stressed"),
                         ["model", "sharpe", "annualized_return", "max_drawdown", "observations"]]
    mature = metrics.loc[metrics.period.eq("trailing504_common") & metrics.scenario.eq("base"), cols]
    narrative = " ".join(commentary)
    text = f"""<!doctype html><html><head><meta charset="utf-8"><title>Five Currents | Lightweight Combination</title>
<style>body{{margin:0;background:#f4f5ef;color:#253f39;font-family:Georgia,serif}}main{{max-width:1160px;margin:auto;padding:38px}}
h1{{font-size:35px;margin-bottom:8px}}h2{{margin-top:8px}}p,li{{line-height:1.6}}.lede{{font-size:19px;color:#53665f}}
section{{background:white;padding:26px;margin:23px 0;border:1px solid #d9dfd5;border-radius:9px}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font:13px Calibri,sans-serif}}td,th{{padding:9px;text-align:right;border-bottom:1px solid #e1e6dc}}
td:first-child,th:first-child{{text-align:left}}th{{background:#edf2e9}}.scroll{{overflow-x:auto}}.note{{font-size:14px;color:#53665f}}
@media print{{body{{background:white}}main{{padding:10px}}section{{break-inside:avoid}}}}</style></head><body><main>
<h1>Five Currents: Lightweight Combination</h1><p class="lede">Portfolio contribution, without a continuous weight optimizer</p>
<p>CL, NG, GC, SI and HG. Full signal-date performance: {first_date:%d %b %Y} to {last_date:%d %b %Y}.
Signal date T earns the established close T+1 to close T+2 return. Only the combination rule changes.</p>
<section><h2>Findings</h2><p>{narrative}</p><p>Greedy pruning changes the admitted roster at {changed} of {nrefits} refits.
All versions retain the existing standalone eligibility rules; this is not an unrestricted new alpha search.</p>
{picture('combination_equity.png')}<div class="scroll">{formatted_table(full[cols])}</div>
<p class="note">Common valid dates across the three books, including real cash. CAGR is compounded annualized return.
Effective concepts = 1 / sum of squared concept votes, before product scaling; reported means are across refits.
No leverage increase or portfolio-volatility retargeting is added. Unpriced active positions are not replaced with zero returns.</p>
<h3>Doubled costs</h3>{formatted_table(stress)}<h3>After the expanding training reaches 504 sessions</h3>
<p class="note">Secondary period beginning {mature_start:%d %b %Y}, not a new untouched holdout.</p><div class="scroll">{formatted_table(mature)}</div></section>
<section><h2>Commodity contributions</h2>{picture('combination_by_commodity.png')}
<p class="note">Additive P&amp;L contributions to the common portfolio, in percentage points; these are not independently compounded asset portfolios.</p>
<div class="scroll">{formatted_table(pd.DataFrame(asset_summary))}</div></section>
<section><h2>Selection and combination</h2><ul>
<li><b>Baseline:</b> original admission gates and equal admitted-concept votes, reproduced against archived weights and after-cost P&amp;L.</li>
<li><b>Greedy:</b> start with that eligible roster. Try removing each concept, keeping equal votes among survivors.
A removal must strictly improve net Sharpe in the full training window and both chronological halves under both base and doubled costs.
Choose the removal with the largest worst-case improvement across these six comparisons; repeat until no removal passes.
There is no fitted K or tuned improvement hurdle, and the final remaining eligible concept is not removed into cash.</li>
<li><b>Correlation discount:</b> for each eligible concept, vote is proportional to 1 / (1 + sum of its positive Spearman correlations with the other eligible concepts).
Correlations use training-only base-cost concept returns, with at least 20 paired observations. Unknown correlations receive zero discount and are logged.
No return forecast, correlation cutoff, negative weight or covariance inverse is fitted.</li>
<li><b>Shared:</b> 21-session refits; 126-session expanding warmup to a 504-session trailing window; archived two-session realization embargo.
Votes and product-risk coefficients remain frozen until the next scheduled refit, while signal positions update daily.
Each changed mix relearns the same inverse-volatility product overlay from its training slice, with gross cap 1.0 and product cap 0.4.</li>
<li><b>Training comparisons:</b> reconstructed netted books, including turnover and contract-roll costs, not averages of individual signal Sharpes.
All candidate comparisons use the same complete five-product marked training dates. The half split uses session order before removing missing marks.
Product scaling is fitted within that training window; its halves are stability diagnostics, not separate out-of-sample tests.</li>
</ul>{picture('combination_votes.png')}<h3>Admission frequency</h3>{formatted_table(fractions)}</section>
<section><h2>Year-by-year</h2><h3>Net Sharpe</h3>{formatted_table(annual_sr)}
<h3>Compounded returns</h3>{formatted_table(annual_ret)}</section>
<section><h2>Residual Sharpe diagnostic</h2><p>For each eligible concept at each refit, compare its risk-scaled net returns with the equal-vote book of the other eligible concepts.
Subtract the fitted beta exposure but retain the intercept contribution: residual mean is the estimated alpha, not forced to zero.
These figures never determine admission or weights. Finite equal-weight additions can differ from the residual diagnostic because of caps and netted costs.</p>
{formatted_table(diagnostic)}</section>
<section><h2>Controls and limitations</h2><p>The ten concepts and their internal neighbourhoods were previously researched on this history.
This is a causal replay of two new combination rules, not newly unseen evidence or a guarantee against selection overfitting.
Greedy subset choice remains adaptive and order-dependent. Correlation discounting need not reward profitable independent alpha.</p>
<p>All archived admission scores and risk coefficients are reproduced before the comparison. Operational curves remain SI/HG ranks 1-4 and CL/NG/GC ranks 1-6;
the historical calibration definitions anchored by the prior curve-policy activation remain unchanged. Base commissions, slippage, differential contract migration,
execution delay and doubled-friction controls are identical across versions.</p>
<p class="note">Cached local files only. No database connections, market-data requests, live dashboard changes or order submission.
Per-refit votes, every greedy comparison, correlations, residual diagnostics, daily positions and P&amp;L are saved alongside SHA-256 provenance.</p></section>
</main></body></html>"""
    (output / "Five_Currents_Lightweight_Combination.html").write_text(text, encoding="utf-8")


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
    archived_coefficients = pd.read_parquet(snapshot / "product_scaling.parquet")
    archived_weights = pd.read_parquet(snapshot / "weights.parquet")
    activation = Path("five_currents_dashboard/curve_policy_activation")
    anchor_ordinal = int(pd.read_parquet(activation / "selection_audit.parquet").refit_ordinal.max())
    index, names = replay.weights.index, tuple(replay.concept_names)
    pd.testing.assert_index_equal(index, archived_weights.index)
    print("[baseline] Rebuilding cached operational curves; no API/database access", flush=True)
    operational = mixed_concepts(replay)
    execution = dollar_cost_execution(replay.execution)
    calibrations = {"legacy": replay.concept_weights, "mixed": operational}
    families = {key: family_path_matrices(values, replay.panel, execution) for key, values in calibrations.items()}
    recomputed = recalibrate_coefficients(selection, replay, operational, execution, anchor_ordinal)
    keys = ["refit_ordinal", "product"]
    np.testing.assert_allclose(recomputed.set_index(keys).inverse_volatility_coefficient.sort_index(),
                               archived_coefficients.set_index(keys).inverse_volatility_coefficient.sort_index(),
                               atol=1e-10, rtol=1e-10)
    audit_rows, coefficients, greedy_rows, correlations, residuals, refit_rows = [], [], [], [], [], []
    grouped = list(selection.groupby("refit_ordinal", sort=True))
    for number, (ordinal, group) in enumerate(grouped, 1):
        first = group.iloc[0]
        positions = training_slice(index, first)
        key = "legacy" if ordinal <= anchor_ordinal else "mixed"
        scored = candidate_training_scores(*families[key], positions)
        ordered = group.set_index("economic_family").reindex(names)
        score_columns = ("train_base_sharpe", "train_stressed_sharpe", "train_first_half_sharpe",
                         "train_second_half_sharpe", "train_active_days", "train_score")
        for column in score_columns:
            np.testing.assert_allclose(scored[column], ordered[column], atol=1e-10, rtol=1e-10)
        eligible_mask = (np.isfinite(scored.train_score) & scored.train_score.gt(0)
                         & scored.train_stressed_sharpe.gt(0) & scored.train_first_half_sharpe.gt(0)
                         & scored.train_second_half_sharpe.gt(0))
        np.testing.assert_array_equal(eligible_mask, ordered.selected)
        eligible = np.flatnonzero(eligible_mask).tolist()
        evaluator = TrainingBook(calibrations[key], index, execution, positions)
        equal = np.zeros(len(names))
        equal[eligible] = 1
        equal = normalize_votes(equal)
        pruned, trials = greedy_backward(eligible, len(names), evaluator.evaluate)
        base_training = families[key][0][:, positions].copy()
        base_training[:, ~evaluator.common] = np.nan
        discounted, redundancy, pairs = correlation_discount(base_training, eligible)
        for trial in trials:
            greedy_rows.append({"refit_ordinal": ordinal, "forecast_start": first.forecast_start,
                                "economic_family": names[trial["removed_index"]], **trial})
        for pair in pairs:
            correlations.append({"refit_ordinal": ordinal, "forecast_start": first.forecast_start,
                                 "left_concept": names[pair["left_index"]],
                                 "right_concept": names[pair["right_index"]], **pair})
        for concept in eligible:
            own = np.zeros(len(names))
            own[concept] = 1
            other = equal.copy()
            other[concept] = 0
            diagnostic = residual_diagnostic(evaluator.evaluate(own).base, evaluator.evaluate(other).base)
            residuals.append({"refit_ordinal": ordinal, "forecast_start": first.forecast_start,
                              "economic_family": names[concept], **diagnostic})
        votes_by_model = {"baseline": equal, "greedy": pruned, "correlation": discounted}
        for model, votes in votes_by_model.items():
            evaluation = evaluator.evaluate(votes)
            if model == "baseline":
                original = archived_coefficients.loc[archived_coefficients.refit_ordinal.eq(ordinal)].set_index("product")
                np.testing.assert_allclose(evaluation.coefficients, original.reindex(PRODUCT_ORDER).inverse_volatility_coefficient,
                                           atol=1e-10, rtol=1e-10)
            for j, name in enumerate(names):
                audit_rows.append({**ordered.loc[name].to_dict(), "model": model, "economic_family": name,
                                   "baseline_selected": bool(eligible_mask.iloc[j]), "selected": bool(votes[j] > 0),
                                   "concept_vote": votes[j], "positive_spearman_sum": redundancy[j],
                                   "selected_family_count": int((votes > 0).sum()), "cash_fallback": not votes.any()})
            for j, product in enumerate(PRODUCT_ORDER):
                coefficients.append({"model": model, "refit_ordinal": ordinal, "forecast_start": first.forecast_start,
                                     "forecast_end": first.forecast_end, "train_start": first.train_start, "train_end": first.train_end,
                                     "product": product, "inverse_volatility_coefficient": evaluation.coefficients[j],
                                     "training_annual_volatility": evaluation.annual_volatility[j],
                                     "training_observations": evaluation.observations[j]})
            refit_rows.append({"model": model, "refit_ordinal": ordinal, "forecast_start": first.forecast_start,
                               "train_start": first.train_start, "train_end": first.train_end,
                               "training_sessions": len(positions), "complete_marked_training_sessions": int(evaluator.common.sum()),
                               "selected_concepts": int((votes > 0).sum()),
                               "effective_concepts": float(1 / np.square(votes).sum()) if votes.any() else 0,
                               **evaluation.scores})
        print(f"[refit {number:02d}/{len(grouped)}] {first.forecast_start:%Y-%m-%d} | "
              f"train {first.train_start:%Y-%m-%d}..{first.train_end:%Y-%m-%d} | "
              f"eligible {len(eligible)}, greedy {(pruned > 0).sum()}, correlation effective "
              f"{1 / np.square(discounted).sum() if discounted.any() else 0:.2f}", flush=True)
    audits, factors = pd.DataFrame(audit_rows), pd.DataFrame(coefficients)
    greedy, correlation_audit = pd.DataFrame(greedy_rows), pd.DataFrame(correlations)
    residuals, refits = pd.DataFrame(residuals), pd.DataFrame(refit_rows)
    paths, product_paths, weights_rows = [], [], []
    for model in LABELS:
        weights = allocate_refits(operational, names, index, audits.loc[audits.model.eq(model)], factors.loc[factors.model.eq(model)])
        if model == "baseline":
            np.testing.assert_allclose(weights, archived_weights.to_numpy(), atol=1e-10, rtol=1e-10)
        assert np.abs(weights).sum(axis=1).max() <= 1 + 1e-10
        assert np.abs(weights).max() <= .4 + 1e-10
        weights_rows.append(pd.DataFrame(weights, index=index, columns=PRODUCT_ORDER).rename_axis("session").reset_index().assign(model=model))
        for scenario in ("base", "stressed"):
            path = cash_before_first_trade(fast_product_backtest(weights, replay.panel, execution, scenario), replay.first_causal_date)
            paths.append(path.rename_axis("session").reset_index().assign(model=model, scenario=scenario))
            returns = product_net_returns(weights, execution, scenario=scenario)
            # A not-yet-traded product contributes zero, without zeroing missing active marks.
            started = np.maximum.accumulate(np.abs(weights) > 1e-12, axis=0)
            returns[~started] = 0
            causal_rows = index >= replay.first_causal_date
            np.testing.assert_allclose(returns.sum(axis=1)[causal_rows], path.net_return.to_numpy()[causal_rows],
                                       atol=1e-12, rtol=1e-10, equal_nan=True)
            product_paths.append(pd.DataFrame(returns, index=index, columns=PRODUCT_ORDER).rename_axis("session").reset_index()
                                 .melt(id_vars="session", var_name="product", value_name="net_contribution").assign(model=model, scenario=scenario))
    paths, products = pd.concat(paths, ignore_index=True), pd.concat(product_paths, ignore_index=True)
    first_date, last_date = replay.first_causal_date, replay.last_realised_date
    paths = paths.loc[paths.session.between(first_date, last_date)].copy()
    products = products.loc[products.session.between(first_date, last_date)].copy()
    previous = Path("five_currents_shuffled_admission_20261012/portfolio_daily_paths.parquet")
    if previous.exists():
        old = pd.read_parquet(previous)
        for scenario in ("base", "stressed"):
            left = paths.loc[paths.model.eq("baseline") & paths.scenario.eq(scenario)].set_index("session")
            right = old.loc[old.model.eq("baseline") & old.scenario.eq(scenario)].set_index("session")
            for column in ("gross_return", "cost", "net_return", "turnover", "gross_exposure"):
                np.testing.assert_allclose(left[column], right.reindex(left.index)[column], atol=1e-10, rtol=1e-10, equal_nan=True)
    metric_rows, annual_rows = [], []
    mature_start = selection.loc[selection.training_sessions.eq(504), "forecast_start"].min()
    for scenario in ("base", "stressed"):
        subset = paths.loc[paths.scenario.eq(scenario)]
        panel = subset.pivot(index="session", columns="model", values="net_return")
        common = panel.dropna().index
        for model, group in subset.groupby("model", sort=False):
            group = group.set_index("session")
            fits = refits.loc[refits.model.eq(model)]
            for period, dates in (("full_common", common), ("full_native", group.net_return.dropna().index),
                                  ("trailing504_common", common[common >= mature_start])):
                sample = group.loc[dates]
                summary = _series_metrics(sample.net_return)
                eq = (1 + sample.net_return).cumprod()
                summary["max_drawdown"] = float((eq / eq.cummax().clip(lower=1) - 1).min())
                metric_rows.append({"model": model, "scenario": scenario, "period": period, **summary,
                                    "mean_admitted": fits.selected_concepts.mean(),
                                    "mean_effective_concepts": fits.effective_concepts.mean(),
                                    "cash_fraction": sample.gross_exposure.le(1e-12).mean(),
                                    "mean_daily_turnover": sample.turnover.mean(),
                                    "mean_gross_exposure": sample.gross_exposure.mean(),
                                    "excluded_marked_sessions": len(panel) - len(dates)})
            if scenario == "base":
                for year, values in group.loc[common, "net_return"].groupby(common.year):
                    summary = _series_metrics(values)
                    annual_rows.append({"model": model, "year": year, "sharpe": summary["sharpe"],
                                        "year_return": summary["total_return"], "observations": len(values)})
    metrics, annual = pd.DataFrame(metric_rows), pd.DataFrame(annual_rows)
    metrics.to_csv(output / "performance_summary.csv", index=False)
    annual.to_csv(output / "yearly_performance.csv", index=False)
    audits.to_parquet(output / "combination_audit.parquet", index=False)
    factors.to_parquet(output / "product_scaling.parquet", index=False)
    greedy.to_csv(output / "greedy_comparisons.csv", index=False)
    correlation_audit.to_parquet(output / "correlation_audit.parquet", index=False)
    residuals.to_csv(output / "residual_diagnostics.csv", index=False)
    refits.to_csv(output / "refit_summary.csv", index=False)
    paths.to_parquet(output / "portfolio_daily_paths.parquet", index=False)
    products.to_parquet(output / "product_daily_contributions.parquet", index=False)
    pd.concat(weights_rows, ignore_index=True).to_parquet(output / "model_weights.parquet", index=False)
    save_report(output, metrics, annual, paths, products, audits, greedy, residuals, names, first_date, last_date, mature_start)
    source_files = [cache, snapshot / "selection_audit.parquet", snapshot / "product_scaling.parquet", snapshot / "weights.parquet",
                    activation / "manifest.json", Path(__file__), Path("five_currents_lightweight_combination.py"),
                    Path("test_five_currents_lightweight_combination.py")]
    source_hashes = {str(p.resolve()): cache_hash if p == cache else sha256(p) for p in source_files}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hashes,
                "snapshot": str(snapshot), "models": LABELS, "refits": len(grouped), "concept_count": len(names),
                "lookback_sessions": 504, "refit_sessions": 21, "minimum_training_sessions": 126, "embargo_sessions": 2,
                "greedy_checks": SCORE_KEYS, "greedy_ranking": "maximum of minimum six Sharpe improvements; fixed concept-order ties",
                "correlation_rule": "1 / (1 + sum positive paired Spearman correlations), normalized across eligible concepts",
                "existing_admission_preserved": True, "parameter_search": False, "residual_diagnostic_only": True,
                "market_api_calls": False, "database_access": False, "production_changes": False,
                "verification": {"baseline_admission_scores_match": True, "baseline_coefficients_match": True,
                                 "baseline_weights_match": True, "baseline_previous_study_paths_match": previous.exists(),
                                 "product_contributions_sum_to_portfolio": True, "cache_unchanged": sha256(cache) == cache_hash},
                "outputs_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nFull-period common-date base-cost results:", flush=True)
    print(metrics.loc[metrics.period.eq("full_common") & metrics.scenario.eq("base"),
                      ["model", "sharpe", "annualized_return", "total_return", "max_drawdown", "mean_admitted"]].to_string(index=False), flush=True)
    print(f"Report: {output / 'Five_Currents_Lightweight_Combination.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path("five_currents_dashboard/snapshots/000033_2026-10-09"))
    parser.add_argument("--output-dir", type=Path, default=Path("five_currents_lightweight_combination_20261012"))
    run(parser.parse_args())
