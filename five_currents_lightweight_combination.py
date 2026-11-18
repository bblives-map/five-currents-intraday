"""Small, training-only combination rules for the existing concept portfolios."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from us_commodity_asset_family_walkforward_core import (
    _inverse_volatility_coefficients, product_net_returns,
)
from us_commodity_disciplined_core import ProductExecution, fast_product_backtest
from us_commodity_research_core import PRODUCT_ORDER, _cap_and_normalize_rows


SCORE_KEYS = tuple(f"{scenario}_{period}" for scenario in ("base", "stressed")
                   for period in ("full", "first_half", "second_half"))


def normalize_votes(votes):
    values = np.asarray(votes, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Concept votes must be finite and nonnegative")
    return values / values.sum() if values.sum() > 0 else values.copy()


def blend_concepts(concepts, votes):
    values = np.asarray(concepts, dtype=float)
    votes = normalize_votes(votes)
    if values.ndim != 3 or len(values) != len(votes):
        raise ValueError("Expected aligned concept x date x product arrays")
    finite = np.isfinite(values)
    denominator = np.einsum("c,ctp->tp", votes, finite)
    numerator = np.einsum("c,ctp->tp", votes, np.where(finite, values, 0.0))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def restore_gross_batched(unscaled, tilted):
    """The established 80-step bisection, batched over dates instead of Python rows."""
    targets = np.minimum(np.abs(unscaled).sum(axis=1), 1.0)
    magnitude = np.abs(tilted)
    active = targets > 1e-15
    if (((magnitude > 1e-15).sum(axis=1) * .4 + 1e-12 < targets) & active).any():
        raise ValueError("Product tilt cannot preserve gross under the active-set cap")
    low, high = np.zeros(len(targets)), np.ones(len(targets))
    insufficient = (np.minimum(high[:, None] * magnitude, .4).sum(axis=1) < targets) & active
    while insufficient.any():
        high[insufficient] *= 2
        insufficient = (np.minimum(high[:, None] * magnitude, .4).sum(axis=1) < targets) & active
    for _ in range(80):
        middle = .5 * (low + high)
        below = np.minimum(middle[:, None] * magnitude, .4).sum(axis=1) < targets
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    output = np.sign(tilted) * np.minimum(high[:, None] * magnitude, .4)
    output[~active] = 0
    np.testing.assert_allclose(np.abs(output).sum(axis=1)[active], targets[active], atol=1e-10)
    if np.max(np.abs(output), initial=0.0) > .4 + 1e-12:
        raise AssertionError("Product cap violated")
    return output


def apply_product_scaling(raw, coefficients):
    unscaled = _cap_and_normalize_rows(raw, maximum_gross=1.0, product_cap=0.4)
    tilted = _cap_and_normalize_rows(raw * coefficients, maximum_gross=1.0, product_cap=0.4)
    return restore_gross_batched(unscaled, tilted)


def training_sharpe(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 20 or np.std(values, ddof=0) <= 1e-12:
        return np.nan
    return float(np.sqrt(252) * values.mean() / np.std(values, ddof=0))


@dataclass
class BookEvaluation:
    votes: np.ndarray
    coefficients: np.ndarray
    annual_volatility: np.ndarray
    observations: np.ndarray
    base: np.ndarray
    stressed: np.ndarray
    scores: dict[str, float]


class TrainingBook:
    """Cut off every input at train_end before evaluating any candidate book."""

    def __init__(self, concepts, index, execution, positions):
        self.positions = np.asarray(positions, dtype=int)
        if not len(self.positions) or np.any(np.diff(self.positions) != 1):
            raise ValueError("Training positions must be nonempty and contiguous")
        end = int(self.positions[-1]) + 1
        self.concepts = np.asarray(concepts, dtype=float)[:, :end].copy()
        self.panel = SimpleNamespace(index=index[:end], products=PRODUCT_ORDER)
        self.execution = ProductExecution(*(np.asarray(getattr(execution, name))[:end].copy()
                                            for name in ("target_return", "base_cost_rate",
                                                         "stressed_cost_rate", "identity")))
        self.common = np.isfinite(self.execution.target_return[self.positions]).all(axis=1)
        self.cache = {}

    def evaluate(self, votes):
        votes = normalize_votes(votes)
        key = tuple(votes)
        if key in self.cache:
            return self.cache[key]
        raw = blend_concepts(self.concepts, votes)
        product_returns = product_net_returns(raw, self.execution, scenario="stressed")[self.positions]
        if votes.sum():
            coefficients, annual_volatility = _inverse_volatility_coefficients(product_returns)
        else:
            coefficients = np.zeros(len(PRODUCT_ORDER))
            annual_volatility = np.full(len(PRODUCT_ORDER), np.nan)
        weights = apply_product_scaling(raw, coefficients)
        returns = {}
        for scenario in ("base", "stressed"):
            path = fast_product_backtest(weights, self.panel, self.execution, scenario)
            before_first_trade = ~path.gross_exposure.gt(1e-12).cummax()
            path.loc[before_first_trade, "net_return"] = 0.0
            values = path.net_return.to_numpy()[self.positions].copy()
            values[~self.common] = np.nan
            returns[scenario] = values
        midpoint = len(self.positions) // 2
        scores = {}
        for scenario, values in returns.items():
            for period, sliced in (("full", values), ("first_half", values[:midpoint]),
                                   ("second_half", values[midpoint:])):
                scores[f"{scenario}_{period}"] = training_sharpe(sliced)
        result = BookEvaluation(votes, coefficients, annual_volatility,
                                np.isfinite(product_returns).sum(axis=0),
                                returns["base"], returns["stressed"], scores)
        self.cache[key] = result
        return result


def greedy_backward(eligible, concept_count, evaluate):
    """Equal votes; remove only when all six fixed training comparisons improve."""
    retained = list(eligible)
    audit = []

    def votes_for(members):
        votes = np.zeros(concept_count)
        votes[members] = 1.0
        return normalize_votes(votes)

    step = 0
    while len(retained) > 1:
        step += 1
        current = evaluate(votes_for(retained))
        trials = []
        for removed in retained:
            remaining = [i for i in retained if i != removed]
            proposed = evaluate(votes_for(remaining))
            delta = np.asarray([proposed.scores[key] - current.scores[key] for key in SCORE_KEYS])
            robust = bool(np.isfinite(delta).all() and (delta > 1e-12).all())
            row = {"step": step, "removed_index": removed, "size_before": len(retained),
                   "robust_improvement": robust, "accepted": False,
                   "minimum_sharpe_improvement": float(delta.min()) if np.isfinite(delta).all() else np.nan}
            for key, change in zip(SCORE_KEYS, delta, strict=True):
                row[f"before_{key}"] = current.scores[key]
                row[f"after_{key}"] = proposed.scores[key]
                row[f"delta_{key}"] = change
            trials.append(row)
        possible = [row for row in trials if row["robust_improvement"]]
        chosen = max(possible, key=lambda row: row["minimum_sharpe_improvement"]) if possible else None
        if chosen is not None:
            chosen["accepted"] = True
            retained.remove(chosen["removed_index"])
        audit.extend(trials)
        if chosen is None:
            break
    return votes_for(retained), audit


def correlation_discount(concept_returns, eligible, *, minimum_observations=20):
    """Discount positive Spearman duplication without fitting returns or a cutoff."""
    values = np.asarray(concept_returns, dtype=float)
    votes = np.zeros(len(values))
    redundancy = np.zeros(len(values))
    rows = []
    if not len(eligible):
        return votes, redundancy, rows
    subset = values[np.asarray(eligible)].T
    frame = pd.DataFrame(subset, columns=eligible)
    correlation = frame.corr(method="spearman", min_periods=minimum_observations).to_numpy()
    positive = np.maximum(np.nan_to_num(correlation, nan=0.0), 0.0)
    np.fill_diagonal(positive, 0.0)
    redundancy[eligible] = positive.sum(axis=1)
    votes[eligible] = 1 / (1 + redundancy[eligible])
    for left in range(len(eligible)):
        for right in range(left + 1, len(eligible)):
            common = np.isfinite(subset[:, left]) & np.isfinite(subset[:, right])
            rows.append({"left_index": eligible[left], "right_index": eligible[right],
                         "spearman": correlation[left, right], "observations": int(common.sum()),
                         "unknown_correlation": not np.isfinite(correlation[left, right])})
    return normalize_votes(votes), redundancy, rows


def residual_diagnostic(candidate, rest):
    """Remove beta exposure, not the intercept: the residual mean is the alpha."""
    candidate, rest = np.asarray(candidate, dtype=float), np.asarray(rest, dtype=float)
    valid = np.isfinite(candidate) & np.isfinite(rest)
    y, x = candidate[valid], rest[valid]
    result = {"observations": len(y), "beta": np.nan, "annualized_alpha": np.nan,
              "residual_sharpe": np.nan, "correlation": np.nan}
    if len(y) < 20 or np.var(x) <= 1e-20:
        return result
    beta = float(np.mean((y - y.mean()) * (x - x.mean())) / np.var(x))
    orthogonal = y - beta * x
    result.update(beta=beta, annualized_alpha=float(252 * orthogonal.mean()),
                  residual_sharpe=training_sharpe(orthogonal),
                  correlation=float(np.corrcoef(y, x)[0, 1]) if np.std(y) > 0 else np.nan)
    return result


def allocate_refits(concepts, names, index, audit, coefficients):
    raw = np.zeros(concepts.shape[1:])
    tilt = np.zeros_like(raw)
    covered = np.zeros(len(index), dtype=bool)
    for ordinal, group in audit.groupby("refit_ordinal", sort=True):
        first = group.iloc[0]
        rows = (index >= first.forecast_start) & (index <= first.forecast_end)
        if covered[rows].any():
            raise AssertionError("Overlapping refits")
        covered[rows] = True
        votes = group.set_index("economic_family").reindex(names).concept_vote.to_numpy(float)
        raw[rows] = blend_concepts(concepts[:, rows], votes)
        factors = coefficients.loc[coefficients.refit_ordinal.eq(ordinal)].set_index("product")
        factors = factors.reindex(PRODUCT_ORDER).inverse_volatility_coefficient.to_numpy(float)
        if not np.isfinite(factors).all():
            raise AssertionError("Missing product risk coefficients")
        tilt[rows] = raw[rows] * factors
    unscaled = _cap_and_normalize_rows(raw, maximum_gross=1.0, product_cap=0.4)
    tilted = _cap_and_normalize_rows(tilt, maximum_gross=1.0, product_cap=0.4)
    return restore_gross_batched(unscaled, tilted)
