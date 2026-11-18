"""Training-only fitness alternatives that do not reward a few exceptional gains."""
from __future__ import annotations

import math
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd

from five_currents_lightweight_combination import SCORE_KEYS, greedy_backward, training_sharpe


OBJECTIVES = ("sharpe", "upside_capped", "rolling_sharpe", "weekly", "monthly")
LABELS = {"sharpe": "Original Sharpe", "upside_capped": "Upside-capped Sharpe",
          "rolling_sharpe": "Median rolling quarterly Sharpe", "weekly": "Profitable-week frequency",
          "monthly": "Profitable-month frequency"}
ROLLING_SESSIONS = 63
ROLLING_MINIMUM_OBSERVATIONS = 50
UPSIDE_QUANTILE = .95


def cap_upside(values):
    values = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(values)
    if not finite.any():
        return values, np.nan
    cap = max(0.0, float(np.quantile(values[finite], UPSIDE_QUANTILE)))
    values[finite] = np.minimum(values[finite], cap)
    return values, cap


def rolling_sharpes(values):
    series = pd.Series(np.asarray(values, dtype=float))
    rolling = series.rolling(ROLLING_SESSIONS, min_periods=ROLLING_MINIMUM_OBSERVATIONS)
    mean, volatility = rolling.mean(), rolling.std(ddof=0)
    scores = np.sqrt(252) * mean / volatility.where(volatility > 1e-12)
    # A 50-observation prefix is not a full quarterly window.
    scores.iloc[:ROLLING_SESSIONS - 1] = np.nan
    return scores.to_numpy()


@lru_cache(maxsize=2048)
def _period_plan(date_bytes, reference_bytes, cutoff_ns, frequency):
    dates = pd.DatetimeIndex(np.frombuffer(date_bytes, dtype="datetime64[ns]"))
    reference = pd.DatetimeIndex(np.frombuffer(reference_bytes, dtype="datetime64[ns]"))
    cutoff = pd.Timestamp(cutoff_ns)
    reference = reference[reference.normalize() <= cutoff]
    if not len(reference):
        return ()
    periods = dates.to_period(frequency)
    reference_periods = reference.to_period(frequency)
    expected = pd.Series(1, index=reference_periods).groupby(level=0).sum()
    rows = []
    for period in periods.unique():
        if period.end_time.normalize() > cutoff or period not in expected.index:
            continue
        lower = period.start_time.normalize()
        if frequency == "W-FRI":
            lower += pd.Timedelta(days=2)  # W-FRI starts Saturday; futures daily labels start Monday.
        if lower < reference.min().normalize():
            continue
        positions = np.flatnonzero(periods == period)
        if len(positions) != int(expected.loc[period]):
            continue
        rows.append((period.end_time.normalize(), positions))
    return tuple(rows)


def completed_period_returns(values, dates, reference_dates, cutoff, frequency):
    """Exclude incomplete boundaries and every period with an unpriced session."""
    if frequency not in ("W-FRI", "M"):
        raise ValueError("Only weekly and monthly calendar aggregation is supported")
    values = np.asarray(values, dtype=float)
    dates, reference = pd.DatetimeIndex(dates), pd.DatetimeIndex(reference_dates)
    if len(values) != len(dates) or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Expected aligned, ordered, unique realization dates")
    cutoff = pd.Timestamp(cutoff).normalize()
    if len(dates) and dates.max().normalize() > cutoff:
        raise ValueError("Period aggregation received a future outcome")
    plan = _period_plan(dates.to_numpy(dtype="datetime64[ns]").tobytes(),
                        reference.to_numpy(dtype="datetime64[ns]").tobytes(), cutoff.value, frequency)
    rows = []
    for end, positions in plan:
        subset = values[positions]
        if np.isfinite(subset).all():
            rows.append((end, float(np.prod(1 + subset) - 1)))
    if not rows:
        return pd.Series(dtype=float, name="period_return")
    return pd.Series(dict(rows), dtype=float, name="period_return").sort_index()


def win_rate(period_returns):
    values = np.asarray(period_returns, dtype=float)
    values = values[np.isfinite(values) & (np.abs(values) > 1e-12)]
    return float((values > 0).mean()) if len(values) >= 2 else np.nan


def objective_value(values, dates, reference_dates, cutoff, objective):
    if objective == "sharpe":
        return training_sharpe(values)
    if objective == "upside_capped":
        return training_sharpe(cap_upside(values)[0])
    if objective == "rolling_sharpe":
        scores = rolling_sharpes(values)
        finite = scores[np.isfinite(scores)]
        return float(np.median(finite)) if len(finite) else np.nan
    if objective in ("weekly", "monthly"):
        frequency = "W-FRI" if objective == "weekly" else "M"
        periods = completed_period_returns(values, dates, reference_dates, cutoff, frequency)
        hit_rate = win_rate(periods)
        return hit_rate - .5 if np.isfinite(hit_rate) else np.nan
    raise ValueError(f"Unknown objective: {objective}")


def fitness_scores(base, stressed, dates, reference_dates, cutoff, objective):
    base, stressed = np.asarray(base, dtype=float), np.asarray(stressed, dtype=float)
    dates = pd.DatetimeIndex(dates)
    if len(base) != len(stressed) or len(base) != len(dates):
        raise ValueError("Misaligned training returns")
    midpoint = len(base) // 2
    result = {}
    for scenario, values in (("base", base), ("stressed", stressed)):
        for period, rows in (("full", slice(None)), ("first_half", slice(None, midpoint)),
                             ("second_half", slice(midpoint, None))):
            selected_dates = dates[rows]
            selected_cutoff = min(pd.Timestamp(cutoff), selected_dates.max())
            result[f"{scenario}_{period}"] = objective_value(
                values[rows], selected_dates, reference_dates, selected_cutoff, objective)
    return result


def activity_eligibility(base, stressed, exposure):
    base, stressed, exposure = map(lambda x: np.asarray(x, dtype=float), (base, stressed, exposure))
    if base.shape != stressed.shape or base.shape != exposure.shape or base.ndim != 2:
        raise ValueError("Expected matching concept x training-date matrices")
    midpoint = base.shape[1] // 2
    active = exposure > 1e-10
    eligible = (active.sum(axis=1) >= max(16, math.ceil(.06 * base.shape[1])))
    eligible &= (active[:, :midpoint].sum(axis=1) >= 5) & (active[:, midpoint:].sum(axis=1) >= 5)
    for values in (base, stressed):
        eligible &= np.isfinite(values).sum(axis=1) >= 20
        eligible &= np.isfinite(values[:, :midpoint]).sum(axis=1) >= 20
        eligible &= np.isfinite(values[:, midpoint:]).sum(axis=1) >= 20
    return eligible


def objective_admission(base, stressed, exposure, dates, reference_dates, cutoff, objective):
    eligible = activity_eligibility(base, stressed, exposure)
    rows = []
    midpoint = len(dates) // 2
    for i in range(len(base)):
        scores = fitness_scores(base[i], stressed[i], dates, reference_dates, cutoff, objective)
        a, b = scores["base_first_half"], scores["base_second_half"]
        composite = (.35 * scores["base_full"] + .25 * scores["stressed_full"]
                     + .25 * min(a, b) + .15 * .5 * (a + b))
        finite = stressed[i][np.isfinite(stressed[i])]
        positive_expectancy = bool(len(finite) and finite.mean() > 0)
        # Replaces, rather than supplements, the original Sharpe performance gate.
        selected = (eligible[i] and np.isfinite(composite) and composite > 0
                    and scores["stressed_full"] > 0 and a > 0 and b > 0
                    and positive_expectancy)
        observations = {}
        if objective in ("weekly", "monthly"):
            frequency = "W-FRI" if objective == "weekly" else "M"
            for name, subset in (("full", slice(None)), ("first_half", slice(None, midpoint)),
                                 ("second_half", slice(midpoint, None))):
                date_subset = dates[subset]
                periods = completed_period_returns(base[i, subset], date_subset, reference_dates,
                                                   min(pd.Timestamp(cutoff), date_subset.max()), frequency)
                observations[f"{name}_nonflat_periods"] = int((np.abs(periods) > 1e-12).sum())
        rows.append({"concept_index": i, "objective": objective, "activity_eligible": bool(eligible[i]),
                     "positive_stressed_expectancy": positive_expectancy, "selected": bool(selected),
                     "fitness": composite, **scores, **observations})
    return pd.DataFrame(rows)


def objective_pruning(eligible, count, evaluator, dates, reference_dates, cutoff, objective):
    score_cache = {}

    def evaluate(votes):
        key = tuple(votes)
        if key not in score_cache:
            book = evaluator.evaluate(votes)
            scores = fitness_scores(book.base, book.stressed, dates, reference_dates, cutoff, objective)
            score_cache[key] = SimpleNamespace(scores=scores)
        return score_cache[key]

    return greedy_backward(eligible, count, evaluate)


def performance_consistency(values, dates, reference_dates, cutoff):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    rolling = rolling_sharpes(values)
    rolling = rolling[np.isfinite(rolling)]
    result = {"median_quarterly_sharpe": float(np.median(rolling)) if len(rolling) else np.nan,
              "positive_quarterly_fraction": float((rolling > 0).mean()) if len(rolling) else np.nan}
    for name, frequency in (("weekly", "W-FRI"), ("monthly", "M")):
        periods = completed_period_returns(values, dates, reference_dates, cutoff, frequency)
        result[f"{name}_win_rate"] = float((periods > 1e-12).mean()) if len(periods) >= 2 else np.nan
        result[f"{name}_active_win_rate"] = win_rate(periods)
        result[f"{name}_observations"] = len(periods)
        result[f"{name}_nonflat_observations"] = int((np.abs(periods) > 1e-12).sum())
    positive = np.sort(finite[finite > 0])[::-1]
    k = max(1, math.ceil(.01 * len(finite)))
    result["top_1pct_positive_pnl_share"] = float(positive[:k].sum() / positive.sum()) if len(positive) else np.nan
    return result
