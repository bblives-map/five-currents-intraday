"""Training-only block randomization for an additional pooled Sharpe admission gate."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def block_permutation_indices(target_returns, rng, *, block_size=5):
    """Jointly permute return blocks without changing any date's availability mask."""
    target = np.asarray(target_returns, dtype=float)
    if target.ndim != 2 or block_size < 1:
        raise ValueError("Expected dates x assets and a positive block size")
    available = np.isfinite(target)
    patterns = np.packbits(available, axis=1)
    _, groups = np.unique(patterns, axis=0, return_inverse=True)
    permutation = np.arange(len(target))
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        blocks = [positions[start:start + block_size]
                  for start in range(0, len(positions), block_size)]
        if len(blocks) > 1:
            permutation[positions] = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
    np.testing.assert_array_equal(available[permutation], available)
    return permutation


def sharpes_from_permutations(weights, targets, costs, valid, permutations):
    """Use fixed positions/costs and randomized asset returns, not shuffled P&L."""
    weights = np.asarray(weights, dtype=float)
    target = np.asarray(targets, dtype=float)
    costs = np.asarray(costs, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if weights.shape[1:] != target.shape or weights.shape[:2] != costs.shape or valid.shape != costs.shape:
        raise ValueError("Misaligned null-model arrays")
    shuffled = np.nan_to_num(target[permutations], nan=0.0)
    gross = np.einsum("ctp,btp->bct", weights, shuffled, optimize=True)
    net = gross - costs[None, :, :]
    count = valid.sum(axis=1)
    total = np.where(valid[None, :, :], net, 0.0).sum(axis=2)
    mean = np.divide(total, count[None, :], out=np.full_like(total, np.nan), where=count[None, :] > 0)
    centered = np.where(valid[None, :, :], net - mean[:, :, None], 0.0)
    variance = np.divide(np.square(centered).sum(axis=2), count[None, :],
                         out=np.full_like(total, np.nan), where=count[None, :] > 0)
    volatility = np.sqrt(variance)
    sharpes = np.divide(math.sqrt(252) * mean, volatility, out=np.full_like(mean, np.nan),
                        where=volatility > 0)
    sharpes[:, count < 20] = np.nan
    return sharpes


def pooled_null_sharpes(weights, targets, costs, valid, *, draws=2000, seed=20261012,
                        block_size=5, batch_size=100):
    if draws < 1 or batch_size < 1:
        raise ValueError("Draw and batch counts must be positive")
    rng = np.random.default_rng(seed)
    blocks = []
    for first in range(0, draws, batch_size):
        permutations = np.stack([block_permutation_indices(targets, rng, block_size=block_size)
                                 for _ in range(min(batch_size, draws - first))])
        blocks.append(sharpes_from_permutations(weights, targets, costs, valid, permutations))
    return np.concatenate(blocks, axis=0)


def apply_admission_gate(selection, thresholds, percentile):
    """Keep every original gate, then require raw base-cost Sharpe above the null cut."""
    result = selection.copy()
    cuts = thresholds.set_index("refit_ordinal")[f"p{percentile}"]
    result["baseline_selected"] = result.selected
    result["null_sharpe_threshold"] = result.refit_ordinal.map(cuts)
    if result.null_sharpe_threshold.isna().any():
        raise ValueError("Missing a training-only threshold")
    result["null_percentile"] = percentile
    result["null_gate_passed"] = result.train_base_sharpe.gt(result.null_sharpe_threshold)
    result["selected"] = result.baseline_selected & result.null_gate_passed
    counts = result.groupby("refit_ordinal").selected.transform("sum")
    result["selected_family_count"] = counts
    result["cash_fallback"] = counts.eq(0)
    return result


def training_slice(index, row):
    positions = np.flatnonzero((index >= row.train_start) & (index <= row.train_end))
    first_forecast = int(index.get_loc(row.forecast_start))
    if len(positions) != int(row.training_sessions) or positions[-1] != first_forecast - 2:
        raise AssertionError("Training slice violated the archived two-session embargo")
    return positions


def cash_before_first_trade(path, first_forecast):
    result = path.copy()
    lead = ~result.gross_exposure.gt(1e-12).cummax() & result.index.to_series().ge(first_forecast)
    result.loc[lead, ["gross_return", "cost", "net_return"]] = 0.0
    return result
