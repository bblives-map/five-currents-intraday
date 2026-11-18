"""Causal ATR trailing exits and performance helpers for Five Currents research."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit


TRADING_DAYS = 252
TRADING_WEEKS = 52


@dataclass(frozen=True)
class TrailResult:
    exit_price: float
    exit_time: pd.Timestamp
    stopped: bool
    initial_atr: float
    minimum_stop_distance: float


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Return bar true ranges using only the preceding observed close."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    previous = np.r_[np.nan, close[:-1]]
    ranges = np.column_stack((high - low, np.abs(high - previous), np.abs(low - previous)))
    output = np.nanmax(ranges, axis=1)
    output[~(np.isfinite(high) & np.isfinite(low) & np.isfinite(close))] = np.nan
    return output


@njit(cache=True)
def _wilder_atr_compiled(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    seed = -1
    for end in range(period - 1, len(values)):
        total = 0.0
        valid = True
        for i in range(end - period + 1, end + 1):
            if not np.isfinite(values[i]):
                valid = False
                break
            total += values[i]
        if valid:
            result[end] = total / period
            seed = end
            break
    if seed < 0:
        return result
    for i in range(seed + 1, len(values)):
        if np.isfinite(values[i]):
            result[i] = (result[i - 1] * (period - 1) + values[i]) / period
        else:
            result[i] = result[i - 1]
    return result


def wilder_atr(ranges: np.ndarray, period: int) -> np.ndarray:
    """Wilder ATR with an arithmetic seed and no use of future observations."""
    if period < 2:
        raise ValueError("ATR period must be at least two bars")
    return _wilder_atr_compiled(np.asarray(ranges, dtype=np.float64), int(period))


@njit(cache=True)
def causal_trailing_exit_index(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    traded: np.ndarray,
    atr: np.ndarray,
    entry_index: int,
    final_index: int,
    entry_price: float,
    final_price: float,
    direction: float,
    multiple: float,
) -> tuple[float, int, bool]:
    """Compiled price-path kernel; stop index -1 denotes the fixed exit."""
    initial_atr = atr[entry_index]
    if not np.isfinite(initial_atr) or initial_atr <= 0 or direction == 0:
        return final_price, -1, False
    sign = 1.0 if direction > 0 else -1.0
    extreme = entry_price
    stop = entry_price - sign * multiple * initial_atr
    for i in range(entry_index + 1, final_index + 1):
        if not traded[i]:
            continue
        opening = open_[i]
        upper = high[i]
        lower = low[i]
        if sign > 0:
            if np.isfinite(opening) and opening <= stop:
                return opening, i, True
            if np.isfinite(lower) and lower <= stop:
                return stop, i, True
            if np.isfinite(upper) and upper > extreme:
                extreme = upper
            if np.isfinite(atr[i]) and atr[i] > 0:
                candidate = extreme - multiple * atr[i]
                if candidate > stop:
                    stop = candidate
        else:
            if np.isfinite(opening) and opening >= stop:
                return opening, i, True
            if np.isfinite(upper) and upper >= stop:
                return stop, i, True
            if np.isfinite(lower) and lower < extreme:
                extreme = lower
            if np.isfinite(atr[i]) and atr[i] > 0:
                candidate = extreme + multiple * atr[i]
                if candidate < stop:
                    stop = candidate
    return final_price, -1, False


def causal_trailing_exit(
    *,
    timestamps: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    traded: np.ndarray,
    atr: np.ndarray,
    entry_index: int,
    final_index: int,
    entry_price: float,
    final_price: float,
    final_time: pd.Timestamp,
    direction: float,
    multiple: float,
) -> TrailResult:
    """Apply a stop known before each bar, then update it after that bar closes.

    The entry bar's range is allowed in the initial ATR because execution is at that
    bar's close. A later bar can trigger the stop set after the previous bar, but its
    own high, low and ATR can only affect the following bar. Gap-through fills use the
    observed open rather than the stale stop level.
    """
    initial_atr = float(atr[entry_index])
    if not np.isfinite(initial_atr) or initial_atr <= 0 or direction == 0:
        return TrailResult(float(final_price), pd.Timestamp(final_time), False,
                           initial_atr, np.nan)
    sign = 1.0 if direction > 0 else -1.0
    extreme = float(entry_price)
    stop = entry_price - sign * multiple * initial_atr
    initial_distance = abs(entry_price - stop)
    for i in range(entry_index + 1, final_index + 1):
        if not bool(traded[i]):
            continue
        opening = float(open_[i])
        upper = float(high[i])
        lower = float(low[i])
        if sign > 0:
            if np.isfinite(opening) and opening <= stop:
                return TrailResult(opening, pd.Timestamp(timestamps[i]) + pd.Timedelta(minutes=1),
                                   True, initial_atr, initial_distance)
            if np.isfinite(lower) and lower <= stop:
                return TrailResult(stop, pd.Timestamp(timestamps[i]) + pd.Timedelta(minutes=1),
                                   True, initial_atr, initial_distance)
            if np.isfinite(upper):
                extreme = max(extreme, upper)
            if np.isfinite(atr[i]) and atr[i] > 0:
                stop = max(stop, extreme - multiple * float(atr[i]))
        else:
            if np.isfinite(opening) and opening >= stop:
                return TrailResult(opening, pd.Timestamp(timestamps[i]) + pd.Timedelta(minutes=1),
                                   True, initial_atr, initial_distance)
            if np.isfinite(upper) and upper >= stop:
                return TrailResult(stop, pd.Timestamp(timestamps[i]) + pd.Timedelta(minutes=1),
                                   True, initial_atr, initial_distance)
            if np.isfinite(lower):
                extreme = min(extreme, lower)
            if np.isfinite(atr[i]) and atr[i] > 0:
                stop = min(stop, extreme + multiple * float(atr[i]))
    return TrailResult(float(final_price), pd.Timestamp(final_time), False,
                       initial_atr, initial_distance)


def weekly_returns(values: pd.Series, *, require_complete: bool = True) -> pd.Series:
    """Compound daily returns into Friday weeks without silently filling gaps."""
    values = values.sort_index()
    labels = values.index.to_period("W-FRI")
    rows = {}
    for label, part in values.groupby(labels):
        if require_complete and part.isna().any():
            rows[label.end_time.normalize()] = np.nan
        elif part.notna().any():
            rows[label.end_time.normalize()] = float((1.0 + part.dropna()).prod() - 1.0)
        else:
            rows[label.end_time.normalize()] = np.nan
    return pd.Series(rows, dtype=float).sort_index()


def performance(values: pd.Series, *, annualization: int = TRADING_DAYS) -> dict[str, float]:
    """Compute observed-return statistics while retaining missing outcomes as gaps."""
    values = pd.Series(values, copy=False).astype(float)
    observed = values.dropna()
    active = observed.loc[observed.abs() > 1e-14]
    if observed.empty:
        return {key: np.nan for key in (
            "sharpe", "annual_vol", "mean", "median", "total_return", "max_drawdown",
            "win_rate", "median_gain", "median_loss", "worst", "best", "sharpe_ex_top5",
        )} | {"observations": 0, "active_observations": 0, "coverage": 0.0}
    standard_deviation = observed.std(ddof=1)
    equity = (1.0 + observed).cumprod()
    drawdown = equity / equity.cummax().clip(lower=1.0) - 1.0
    without = observed.copy()
    positive = without.loc[without > 0]
    if len(positive):
        without = without.drop(positive.nlargest(min(5, len(positive))).index)
    without_std = without.std(ddof=1)
    return {
        "observations": int(len(observed)),
        "active_observations": int(len(active)),
        "coverage": float(len(observed) / len(values)) if len(values) else np.nan,
        "sharpe": float(observed.mean() / standard_deviation * np.sqrt(annualization))
        if standard_deviation > 0 else np.nan,
        "annual_vol": float(standard_deviation * np.sqrt(annualization)),
        "mean": float(observed.mean()),
        "median": float(observed.median()),
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((active > 0).mean()) if len(active) else np.nan,
        "median_gain": float(active.loc[active > 0].median()) if (active > 0).any() else np.nan,
        "median_loss": float(active.loc[active < 0].median()) if (active < 0).any() else np.nan,
        "worst": float(observed.min()),
        "best": float(observed.max()),
        "sharpe_ex_top5": float(without.mean() / without_std * np.sqrt(annualization))
        if without_std > 0 else np.nan,
    }


def daily_and_weekly_metrics(values: pd.Series) -> dict[str, float]:
    daily = performance(values, annualization=TRADING_DAYS)
    weekly = performance(weekly_returns(values), annualization=TRADING_WEEKS)
    output = {f"daily_{key}": value for key, value in daily.items()}
    output.update({f"weekly_{key}": value for key, value in weekly.items()})
    return output


def rolling_sharpe(values: pd.Series, window: int, annualization: int = TRADING_DAYS) -> pd.Series:
    mean = values.rolling(window, min_periods=max(10, window // 2)).mean()
    std = values.rolling(window, min_periods=max(10, window // 2)).std(ddof=1)
    return mean.div(std.where(std > 0)).mul(np.sqrt(annualization))
