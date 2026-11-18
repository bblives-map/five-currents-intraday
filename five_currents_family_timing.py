"""Offline, prefix-only family admission and execution-recipe research helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from five_currents_execution_timing import (
    ROOTS, PRIMARY_VARIANTS, forecast_matrix, asset_pnl, summarize_matrix,
)


METHODS = ("fixed_ten", "best_family", "greedy_three")
FITS = {"252_weekly": (252, 5), "252_monthly": (252, 21),
        "504_weekly": (504, 5), "504_monthly": (504, 21),
        "expanding_weekly": (None, 5), "expanding_monthly": (None, 21)}


def fitness(values, *, minimum_history=126):
    """Median doubled-cost Sharpe across three fill delays, with coverage gates."""
    x = np.asarray(values, float)
    if x.ndim == 2:
        x = x[:, None, :]
    common = np.isfinite(x).all(axis=2)
    x = np.where(common[:, :, None], x, np.nan)
    n = common.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.nansum(x, axis=0) / n[:, None]
        var = (np.nansum(x*x, axis=0)-n[:, None]*mean*mean)/(n[:, None]-1)
        sd = np.sqrt(np.maximum(var, 0))
        sr = np.where(sd > 1e-14, mean/sd*np.sqrt(252), np.nan)
        half = len(x)//2
        before = np.nansum(x[:half, :, 0], axis=0)/common[:half].sum(axis=0)
        after = np.nansum(x[half:, :, 0], axis=0)/common[half:].sum(axis=0)
    scores = np.median(sr, axis=1)
    active = np.sum(np.isfinite(x[:, :, 0]) & (np.abs(x[:, :, 0]) > 1e-15), axis=0)
    eligible = ((len(x) >= minimum_history) & (n >= max(60, int(np.ceil(.55*len(x)))))
                & (active >= 15) & np.isfinite(scores) & (scores > 0)
                & (before > 0) & (after > 0))
    return scores, eligible


def select_recipes(training):
    """One recipe per family; every family receives exactly the same recipe grid."""
    days, families, recipes, delays = training.shape
    scores, eligible = fitness(training.reshape(days, families*recipes, delays))
    scores, eligible = scores.reshape(families, recipes), eligible.reshape(families, recipes)
    chosen = np.argmax(np.where(eligible, scores, -np.inf), axis=1)
    chosen[~eligible.any(axis=1)] = -1
    family_scores = np.array([scores[f, c] if c >= 0 else np.nan for f, c in enumerate(chosen)])
    paths = np.full((days, families, delays), np.nan)
    for f, c in enumerate(chosen):
        if c >= 0:
            paths[:, f] = training[:, f, c]
    return chosen, family_scores, paths


def family_allocation(paths, scores, method):
    """Nonnegative equal weights; no fitted covariance matrix or leverage increase."""
    available = np.flatnonzero(np.isfinite(scores))
    result = np.zeros(len(scores))
    if not len(available):
        return result
    if method == "fixed_ten":
        result[available] = 1/len(scores)
        return result
    first = int(available[np.argmax(scores[available])])
    selected = [first]
    if method == "greedy_three":
        # Compare each addition against the incumbent on identical training dates.
        for _ in range(2):
            best, gain = None, 0.
            base = paths[:, selected].mean(axis=1)
            for candidate in available:
                if candidate in selected:
                    continue
                trial = paths[:, selected+[int(candidate)]].mean(axis=1)
                paired = np.isfinite(base).all(axis=1) & np.isfinite(trial).all(axis=1)
                b = np.where(paired[:, None], base, np.nan)
                t = np.where(paired[:, None], trial, np.nan)
                sb, eb = fitness(b)
                st, et = fitness(t)
                improvement = st[0]-sb[0]
                if eb[0] and et[0] and improvement > gain+1e-12:
                    best, gain = int(candidate), float(improvement)
            if best is None:
                break
            selected.append(best)
    elif method != "best_family":
        raise ValueError(method)
    result[selected] = 1/len(selected)
    return result


def fit_schedule(neighbors, *, lookback=252, refit=21, frozen_fraction=None,
                 minimum_history=126, embargo=1):
    """Training excludes the immediately preceding complete trading session."""
    n, families = neighbors.shape[:2]
    choices = np.full((n, families), -1, dtype=np.int16)
    allocations = {method: np.zeros((n, families)) for method in METHODS}
    records = []
    first = int(n*frozen_fraction) if frozen_fraction is not None else minimum_history+embargo
    starts = [first] if frozen_fraction is not None else range(first, n, refit)
    for start in starts:
        stop = start-embargo
        begin = max(0, stop-lookback) if lookback and frozen_fraction is None else 0
        chosen, scores, paths = select_recipes(neighbors[begin:stop])
        end = n if frozen_fraction is not None else min(n, start+refit)
        choices[start:end] = chosen
        for method in METHODS:
            alpha = family_allocation(paths, scores, method)
            allocations[method][start:end] = alpha
            for family in range(families):
                records.append({"start": start, "end": end, "train_start": begin,
                    "train_end_exclusive": stop, "method": method, "family_index": family,
                    "candidate": int(chosen[family]), "allocation": alpha[family],
                    "fitness": scores[family]})
    return choices, allocations, pd.DataFrame(records), first


def mapped_weights(source, timeline, *, extra_lag=0):
    names = source.manifest["signal_names"]
    arrays = []
    for name in names:
        data = source.settled[name] if timeline == "older_settled" else source.values[name]
        arrays.append(np.stack([forecast_matrix(data, source.dates, source.sessions,
            lag=extra_lag+int(timeline == "older_settled"), variant=v)[0] for v in PRIMARY_VARIANTS]))
    return np.stack(arrays)  # family, style, session, asset


def training_pools(weights, cube, books):
    """Produce stressed return candidates without using future price availability."""
    families, styles, days, assets = weights.shape
    recipes = cube["returns"].shape[0]
    pools = {name: np.full((days, families, styles*recipes, 3), np.nan) for name in books}
    for f in range(families):
        for v in range(styles):
            for delay in range(3):
                gross, fee, slip = asset_pnl(weights[f, v][None], cube["returns"][:, delay], cube["entry"][:, delay])
                stressed = gross-2*(fee+slip)
                for name, indices in books.items():
                    part = stressed[:, :, indices]
                    values = np.where(np.isfinite(part).all(axis=2), part.sum(axis=2), np.nan).T
                    pools[name][:, f, v*recipes:(v+1)*recipes, delay] = values
    return pools


def simulate(weights, cube, choices, allocations, *, delay=0, direction=None):
    """Net identical asset/recipe orders before fees; no netting across windows."""
    families, _, days, assets = weights.shape
    cells = cube["returns"].shape[0]
    positions = np.zeros((days, cells, assets))
    abs_orders = positions.copy()
    components = []
    for f in range(families):
        d, a = np.where((choices[:, :, f] >= 0) & (allocations[:, :, f] > 0))
        recipe = choices[d, a, f]
        variant, cell = np.divmod(recipe, cells)
        amount = allocations[d, a, f]*weights[f, variant, d, a]
        np.add.at(positions, (d, cell, a), amount)
        np.add.at(abs_orders, (d, cell, a), np.abs(amount))
        components.append((d, cell, a, amount))
    # Exposure cap is based on planned orders, not on whether later prices exist.
    cap = np.maximum(np.abs(positions).sum(axis=(1, 2)), 1.)
    positions /= cap[:, None, None]
    abs_orders /= cap[:, None, None]
    if direction is not None:
        positions = np.abs(positions)*direction
    moves = cube["returns"][:, delay].transpose(1, 0, 2)
    entries = cube["entry"][:, delay].transpose(1, 0, 2)
    gross, fee, slip = asset_pnl(positions, moves, entries)
    cost = fee+slip
    _, ifee, islip = asset_pnl(abs_orders, moves, entries)
    independent_cost = ifee+islip
    by_asset = {"gross": gross.sum(axis=1), "cost": cost.sum(axis=1),
                "independent_cost": independent_cost.sum(axis=1)}
    by_asset["net"] = by_asset["gross"]-by_asset["cost"]
    by_family = np.zeros((days, families))
    if direction is None:
        for f, (d, c, a, amount) in enumerate(components):
            amount = amount/cap[d]
            live = np.abs(positions[d, c, a]) > 1e-12
            g = np.where(live, amount*moves[d, c, a], 0.)
            fraction = np.divide(np.abs(amount), abs_orders[d, c, a],
                                 out=np.zeros_like(amount), where=abs_orders[d, c, a] > 0)
            debit = np.where(live, fraction*cost[d, c, a], 0.)
            np.add.at(by_family[:, f], d, g-debit)
    values = {name: matrix.sum(axis=1) for name, matrix in by_asset.items()}
    values["net"] = values["gross"]-values["cost"]
    values["stressed"] = values["gross"]-2*values["cost"]
    values["independent_net"] = values["gross"]-values["independent_cost"]
    values.update(by_asset=by_asset, by_family=by_family, positions=positions,
                  exposure=np.abs(positions).sum(axis=(1, 2)), cap=cap)
    return values


def concentration(values):
    rows = []
    for number in (0, 1, 3, 5):
        x = values.copy()
        if number:
            locations = np.flatnonzero(np.isfinite(x))
            x[locations[np.argsort(x[locations])[-number:]]] = np.nan
        row = {k: v[0] for k, v in summarize_matrix(x).items()}
        rows.append({"drop_best_days": number, **row})
    return rows
