"""Causal expanding-window refits of the common Five Currents ATR trail."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_atr_trail import daily_and_weekly_metrics
from run_five_currents_intraday_atr_study import (
    CORE_GROUPS,
    MULTIPLES,
    OUT,
    PERIODS,
    aggregate_rows,
    group_definitions,
    load_rules,
    load_simulation,
)


def matrix_statistics(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    total = np.nansum(values, axis=1)
    mean = np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)
    sum_squares = np.nansum(values * values, axis=1)
    variance = np.divide(sum_squares - count * mean * mean, count - 1,
                         out=np.full(len(values), np.nan), where=count > 1)
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    sharpe = np.divide(mean, standard_deviation, out=np.full(len(values), np.nan),
                       where=standard_deviation > 0) * np.sqrt(252.0)
    equity = np.cumprod(np.where(finite, 1.0 + values, 1.0), axis=1)
    peak = np.maximum(1.0, np.maximum.accumulate(equity, axis=1))
    drawdown = np.min(equity / peak - 1.0, axis=1)
    return {"sharpe": sharpe, "mean": mean, "max_drawdown": drawdown}


def weekly_matrix(values: np.ndarray, sessions: pd.DatetimeIndex) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    labels = sessions.to_period("W-FRI")
    unique = labels.unique().sort_values()
    output = np.full((len(values), len(unique)), np.nan, dtype=float)
    for column, label in enumerate(unique):
        part = values[:, labels == label]
        valid = np.isfinite(part).all(axis=1)
        output[valid, column] = np.prod(1.0 + part[valid], axis=1) - 1.0
    return output


def score_candidates(candidate_groups: dict[str, np.ndarray],
                     baseline_groups: dict[str, np.ndarray],
                     sessions: pd.DatetimeIndex, end: int):
    candidate_portfolio = candidate_groups["Portfolio"][:, :end]
    baseline_portfolio = baseline_groups["Portfolio"][:, :end]
    candidate_daily = matrix_statistics(candidate_portfolio)
    baseline_daily = matrix_statistics(baseline_portfolio)
    candidate_weekly = matrix_statistics(weekly_matrix(candidate_portfolio, sessions[:end]))
    baseline_weekly = matrix_statistics(weekly_matrix(baseline_portfolio, sessions[:end]))
    candidate_core = np.nanmedian(np.vstack([
        matrix_statistics(candidate_groups[name][:, :end])["sharpe"] for name in CORE_GROUPS
    ]), axis=0)
    baseline_core = float(np.nanmedian([
        matrix_statistics(baseline_groups[name][:, :end])["sharpe"][0] for name in CORE_GROUPS
    ]))
    frame = pd.DataFrame({
        "candidate": np.arange(len(PERIODS) * len(MULTIPLES)),
        "atr_period": np.repeat(PERIODS, len(MULTIPLES)),
        "atr_multiple": np.tile(MULTIPLES, len(PERIODS)),
        "portfolio_daily_sharpe": candidate_daily["sharpe"],
        "portfolio_weekly_sharpe": candidate_weekly["sharpe"],
        "portfolio_daily_mean": candidate_daily["mean"],
        "portfolio_weekly_mean": candidate_weekly["mean"],
        "median_core_daily_sharpe": candidate_core,
        "portfolio_max_drawdown": candidate_daily["max_drawdown"],
    })
    baseline = {
        "portfolio_daily_sharpe": baseline_daily["sharpe"][0],
        "portfolio_weekly_sharpe": baseline_weekly["sharpe"][0],
        "portfolio_daily_mean": baseline_daily["mean"][0],
        "portfolio_weekly_mean": baseline_weekly["mean"][0],
        "median_core_daily_sharpe": baseline_core,
        "portfolio_max_drawdown": baseline_daily["max_drawdown"][0],
    }
    criteria = list(baseline)
    percentile_columns = []
    for criterion in criteria:
        column = criterion + "_percentile"
        frame[column] = frame[criterion].rank(pct=True, method="average").fillna(0.0)
        percentile_columns.append(column)
    frame["rank_score"] = frame[percentile_columns].mean(axis=1)
    frame["improvement_count"] = sum(
        frame[criterion].ge(baseline[criterion]).astype(int) for criterion in criteria
    )
    scores = frame.set_index(["atr_period", "atr_multiple"]).rank_score
    stable = []
    for row in frame.itertuples(index=False):
        pi = PERIODS.index(int(row.atr_period))
        mi = MULTIPLES.index(float(row.atr_multiple))
        neighborhood = [scores.loc[PERIODS[p], MULTIPLES[m]]
                        for p in range(max(0, pi - 1), min(len(PERIODS), pi + 2))
                        for m in range(max(0, mi - 1), min(len(MULTIPLES), mi + 2))]
        stable.append(float(np.nanmedian(neighborhood)))
    frame["stable_rank_score"] = stable
    eligible = frame.loc[frame.improvement_count.ge(4)]
    required = 4
    if eligible.empty:
        required = int(frame.improvement_count.max())
        eligible = frame.loc[frame.improvement_count.eq(required)]
    chosen = eligible.sort_values(
        ["stable_rank_score", "rank_score", "atr_multiple", "atr_period"],
        ascending=[False, False, False, True], kind="stable",
    ).iloc[0]
    return chosen, required


def run_path(candidate_groups: dict[str, np.ndarray],
             baseline_groups: dict[str, np.ndarray], sessions: pd.DatetimeIndex,
             start: int, cadence: int):
    paths = {name: np.full(len(sessions), np.nan, dtype=float) for name in candidate_groups}
    rows = []
    for begin in range(start, len(sessions), cadence):
        chosen, required = score_candidates(candidate_groups, baseline_groups, sessions, begin)
        end = min(len(sessions), begin + cadence)
        candidate = int(chosen.candidate)
        for name in paths:
            paths[name][begin:end] = candidate_groups[name][candidate, begin:end]
        rows.append({
            "cadence_sessions": cadence,
            "fit_through": sessions[begin - 1],
            "effective_start": sessions[begin],
            "effective_end": sessions[end - 1],
            "history_sessions": begin,
            "candidate": candidate,
            "atr_period": int(chosen.atr_period),
            "atr_multiple": float(chosen.atr_multiple),
            "improvement_criteria": int(chosen.improvement_count),
            "minimum_required_criteria": required,
            "stable_rank_score": float(chosen.stable_rank_score),
        })
    return paths, rows


def main() -> None:
    rules = load_rules()
    ledger = pd.read_parquet(OUT / "trade_ledger.parquet")
    ledger = ledger.loc[ledger.delay_minutes.eq(0)].copy()
    leg_order = {leg: i for i, leg in enumerate(rules.leg)}
    ledger["leg_order"] = ledger.leg.map(leg_order)
    ledger = ledger.sort_values(["session", "leg_order"], kind="stable").drop(columns="leg_order").reset_index(drop=True)
    sessions = pd.DatetimeIndex(ledger.session.unique()).sort_values()
    simulation, pairs = load_simulation(OUT / "trail_grid_simulation.npz")
    expected = [(period, multiple) for period in PERIODS for multiple in MULTIPLES]
    if pairs != expected:
        raise AssertionError("Cached simulation does not match the walk-forward grid")
    baseline_groups = aggregate_rows(ledger.net.to_numpy(float), ledger, sessions, rules)
    candidate_groups = aggregate_rows(simulation["net"], ledger, sessions, rules)
    start = int(len(sessions) * 0.3)
    models = {"Fixed exit": {name: values[0] for name, values in baseline_groups.items()}}
    refits = []
    for cadence, name in ((21, "Expanding monthly ATR"), (63, "Expanding quarterly ATR")):
        paths, rows = run_path(candidate_groups, baseline_groups, sessions, start, cadence)
        models[name] = paths
        refits.extend(rows)
    pd.DataFrame(refits).to_csv(OUT / "walkforward_refits.csv", index=False)

    columns = {}
    metric_rows = []
    for model, groups in models.items():
        for series, values in groups.items():
            columns[f"{model}|{series}"] = values
            if series not in ("Portfolio", *CORE_GROUPS):
                continue
            oos = pd.Series(values[start:], index=sessions[start:])
            metric_rows.append({"model": model, "series": series, "sample": "Held-out 70%",
                                **daily_and_weekly_metrics(oos)})
    frame = pd.DataFrame(columns, index=sessions)
    frame.index.name = "session"
    frame.to_parquet(OUT / "walkforward_daily_returns.parquet", compression="zstd")
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "walkforward_metrics.csv", index=False)
    print(metrics.loc[metrics.series.eq("Portfolio"),
                      ["model", "daily_sharpe", "weekly_sharpe", "daily_mean",
                       "weekly_mean", "daily_total_return", "daily_max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
