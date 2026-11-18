"""Pure execution-clock research: frozen signals, forward-only fills, causal timing fits."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from five_currents_front_intraday import ROOTS, MICROS, MICRO_MULTIPLIERS
from run_five_currents_masked_intraday import contract_month


CLOCKS = ("cme_open", "early_utc", "london_open", "london_close", "us_open", "settlement", "us_close")
HOLDS = (30, 60, 90, 120, 180, 240, 360, 720, 0)
OFFSETS = (-30, -15, 0, 5, 15, 30)
VARIANTS = ("net_magnitude", "change_magnitude", "reverse_net_magnitude", "reverse_change_magnitude",
            "net_sign", "change_sign", "reverse_net_sign", "reverse_change_sign")
PRIMARY_VARIANTS = VARIANTS[:4]
EXCHANGE_TICKS = {"CL": .01, "NG": .001, "GC": .1, "SI": .005, "HG": .0005}
SLIPPAGE_PRICE_PER_SIDE = {**EXCHANGE_TICKS,"SI":.01}
COMMISSION_RT = 2.16
TOLERANCE_MINUTES = 10
SETTLEMENT_NY = {"CL": (14, 30), "NG": (14, 30), "GC": (13, 30), "SI": (13, 25), "HG": (13, 0)}


def regional_time(session, clock, root="CL"):
    day = pd.Timestamp(session).date()
    if clock == "cme_open":
        day -= timedelta(days=1)
        zone, hour, minute = "America/Chicago", 17, 0
    elif clock == "early_utc":
        zone, hour, minute = "America/Chicago", 1, 0
    elif clock == "london_open":
        zone, hour, minute = "Europe/London", 8, 0
    elif clock == "london_close":
        zone, hour, minute = "Europe/London", 16, 30
    elif clock == "us_open":
        zone, hour, minute = "America/New_York", 9, 30
    elif clock == "us_close":
        zone, hour, minute = "America/New_York", 16, 0
    elif clock == "settlement":
        zone = "America/New_York"
        hour, minute = SETTLEMENT_NY[root]
    elif clock == "session_close":
        zone, hour, minute = "America/Chicago", 15, 55
    else:
        raise ValueError(clock)
    return pd.Timestamp(datetime.combine(day, time(hour, minute), tzinfo=ZoneInfo(zone))).tz_convert("UTC")


def within_session(stamp, session):
    opening = regional_time(session, "cme_open")
    end = regional_time(session, "session_close")+pd.Timedelta(minutes=5)
    return opening <= stamp < end


@dataclass(frozen=True)
class Fill:
    ts: pd.Timestamp
    price: float
    symbol: str
    table: str
    local_symbol: str
    month: str
    traded: bool


class PriceBook:
    def __init__(self, frame):
        frame = frame.copy()
        frame["ts"] = pd.to_datetime(frame.ts, utc=True)
        frame["open"] = pd.to_numeric(frame.open, errors="coerce")
        frame = frame.loc[np.isfinite(frame.open) & frame.open.gt(0) & frame.local_symbol.notna()]
        frame["month"] = [contract_month(code, stamp) for code, stamp in zip(frame.local_symbol, frame.ts)]
        frame = frame.dropna(subset=["month"])
        frame["traded"] = pd.to_numeric(frame.volume, errors="coerce").gt(0) | pd.to_numeric(frame.bar_count, errors="coerce").gt(0)
        frame["source_priority"] = frame.table.ne("bars_1m").astype(int)
        frame = frame.sort_values(["ts", "source_priority"], kind="stable")
        self.groups = {}
        for key, group in frame.groupby(["root", "month", "symbol"], sort=False):
            rows = [Fill(r.ts, float(r.open), r.symbol, r.table, str(r.local_symbol), r.month, bool(r.traded))
                    for r in group.itertuples()]
            self.groups[key] = (np.array([r.ts.value for r in rows], dtype=np.int64), rows)

    def find(self, root, month, stamp, *, symbol=None, require_trade=True):
        if not isinstance(month, str):
            return None
        wanted = pd.Timestamp(stamp).value
        limit = wanted+pd.Timedelta(minutes=TOLERANCE_MINUTES).value
        choices = []
        for candidate in ((symbol,) if symbol else (MICROS[root], root)):
            group = self.groups.get((root, month, candidate))
            if group is None:
                continue
            stamps, rows = group
            first, end = np.searchsorted(stamps, [wanted, limit], side="left")
            end = np.searchsorted(stamps, limit, side="right")
            for row in rows[first:end]:
                if require_trade and not row.traded:
                    continue
                choices.append((row.ts.value, candidate != MICROS[root], row.table != "bars_1m", row))
                break
        return min(choices, default=None, key=lambda r:r[:-1])[-1] if choices else None


def trade_pair(book, target, clock, hold, offset, *, require_trade=True):
    planned = regional_time(target.session, clock, target.root)+pd.Timedelta(minutes=offset)
    result = {"session":target.session,"root":target.root,"clock":clock,"hold":hold,"offset":offset,
              "month":target.month,"identity_basis":target.identity_basis,"rank":target.rank,
              "planned_entry":planned,"status":"missing entry"}
    if not within_session(planned, target.session):
        result["status"] = "entry outside trading session"
        return result
    entry = book.find(target.root, target.month, planned, require_trade=require_trade)
    if entry is None:
        return result
    result.update(entry_ts=entry.ts, entry_price=entry.price, symbol=entry.symbol,
                  entry_table=entry.table, local_symbol=entry.local_symbol,
                  entry_delay_minutes=(entry.ts-planned).total_seconds()/60,
                  entry_traded=entry.traded)
    end = regional_time(target.session,"session_close",target.root) if hold == 0 else entry.ts+pd.Timedelta(minutes=hold)
    result["planned_exit"] = end
    if end <= entry.ts or not within_session(end, target.session):
        result["status"] = "hold leaves trading session"
        return result
    last = book.find(target.root, target.month, end, symbol=entry.symbol, require_trade=require_trade)
    if last is None:
        result["status"] = "missing same-contract exit"
    elif not within_session(last.ts, target.session):
        result["status"] = "delayed exit leaves trading session"
    else:
        result.update(status="matched", exit_ts=last.ts, exit_price=last.price, exit_table=last.table,
                      exit_traded=last.traded, price_return=last.price/entry.price-1.,
                      exit_delay_minutes=(last.ts-end).total_seconds()/60)
    return result


def forecast_matrix(values, dates, sessions, *, lag=0, variant="net_magnitude", previous_values=None):
    """Map T's forecast to session T+1; lag=1 reproduces the older daily-held forecast."""
    if lag < 0:
        raise ValueError("Forecast lag cannot be negative")
    positions = dates.searchsorted(sessions, side="left")-1-lag
    valid = positions >= 1
    current = np.full((len(sessions),len(ROOTS)), np.nan)
    previous = current.copy()
    current[valid] = values[positions[valid]]
    prior = values if previous_values is None else previous_values
    previous[valid] = prior[positions[valid]-1]
    reverse = variant.startswith("reverse_")
    base = variant.removeprefix("reverse_")
    weight = current-previous if base.startswith("change") else current
    if base.endswith("sign"):
        weight = np.where(np.abs(weight)>1e-12,np.sign(weight)/len(ROOTS),0.)
        weight[~valid] = np.nan
    if reverse:
        weight = -weight
    # Every style has the same ex-ante unit gross ceiling, including two-sided changes.
    gross = np.abs(weight).sum(axis=1)
    weight /= np.maximum(gross,1.)[:,None]
    source_dates = pd.Series(pd.NaT,index=sessions,dtype="datetime64[ns]")
    source_dates.loc[valid] = dates[positions[valid]]
    if not (source_dates.dropna().to_numpy() < sessions[valid].to_numpy()).all():
        raise AssertionError("Forecast is not dated strictly before execution")
    return weight, source_dates


def asset_pnl(weight, price_return, entry_price, *, commission=COMMISSION_RT, slippage_ticks=1.):
    multipliers = np.array([MICRO_MULTIPLIERS[root] for root in ROOTS])
    ticks = np.array([SLIPPAGE_PRICE_PER_SIDE[root] for root in ROOTS])
    inactive = np.isfinite(weight) & (np.abs(weight)<=1e-12)
    gross = weight*price_return
    inactive = np.broadcast_to(inactive,gross.shape)
    with np.errstate(divide="ignore",invalid="ignore"):
        fee = np.abs(weight)*commission/(entry_price*multipliers)
        slip = np.abs(weight)*2*slippage_ticks*ticks/entry_price
    for array in (gross,fee,slip):
        array[inactive] = 0.
    return gross, fee, slip


def summarize_matrix(values):
    """One column per strategy; NaNs never become cash or zero returns."""
    x = np.asarray(values,float)
    if x.ndim == 1:
        x = x[:,None]
    if not len(x):
        return {name:np.full(x.shape[1],0 if name in ("days","active_days") else np.nan)
            for name in ("days","coverage","sharpe","annual_vol","mean_daily","observed_total_return",
                         "observed_max_drawdown","win_rate","active_days")}
    n = np.isfinite(x).sum(axis=0)
    total = np.nansum(x,axis=0)
    mean = np.divide(total,n,out=np.full(x.shape[1],np.nan),where=n>0)
    sumsq = np.nansum(x*x,axis=0)
    variance = np.divide(sumsq-n*mean*mean,n-1,out=np.full(x.shape[1],np.nan),where=n>1)
    std = np.sqrt(np.maximum(variance,0))
    sr = np.divide(mean,std,out=np.full_like(mean,np.nan),where=std>1e-14)*np.sqrt(252.)
    # Replacing missing with a multiplicative identity here only computes the sum of
    # observed compounded returns. This is explicitly not a complete equity history.
    factors = np.where(np.isfinite(x),1+x,1.)
    equity = np.cumprod(factors,axis=0)
    drawdown = equity/np.maximum(1.,np.maximum.accumulate(equity,axis=0))-1.
    positive = np.sum(x>0,axis=0)
    active = np.sum(np.isfinite(x)&(np.abs(x)>1e-15),axis=0)
    result = {"days":n,"coverage":n/len(x),"sharpe":sr,"annual_vol":std*np.sqrt(252.),
              "mean_daily":mean,"observed_total_return":equity[-1]-1.,"observed_max_drawdown":drawdown.min(axis=0),
              "win_rate":np.divide(positive,active,out=np.full(x.shape[1],np.nan),where=active>0),
              "active_days":active}
    for key in ("observed_total_return","observed_max_drawdown"):
        result[key] = np.where(n>0,result[key],np.nan)
    return result


def causal_choice(delayed_net, *, lookback=252, min_history=126, minimum_observations=60):
    """Use only the supplied trailing outcomes, including delayed-fill neighbors."""
    # time x candidate x three predeclared timing perturbations
    x = np.asarray(delayed_net,float)
    if len(x) < min_history:
        return -1, np.nan, 0
    x = x[-lookback:] if lookback else x
    common = np.isfinite(x).all(axis=2)
    x = np.where(common[:,:,None],x,np.nan)
    count = common.sum(axis=0)
    needed = max(minimum_observations,int(np.ceil(.55*len(x))))
    with np.errstate(divide="ignore",invalid="ignore"):
        mean = np.nansum(x,axis=0)/count[:,None]
        variance = (np.nansum(x*x,axis=0)-count[:,None]*mean*mean)/(count[:,None]-1)
        std = np.sqrt(np.maximum(variance,0))
        sr = np.where(std>1e-14,mean/std*np.sqrt(252.),np.nan)
    scores = np.median(sr,axis=1)
    half = len(x)//2
    with np.errstate(divide="ignore",invalid="ignore"):
        before = np.nansum(x[:half,:,0],axis=0)/np.isfinite(x[:half,:,0]).sum(axis=0)
        after = np.nansum(x[half:,:,0],axis=0)/np.isfinite(x[half:,:,0]).sum(axis=0)
    active = np.sum(np.isfinite(x[:,:,0])&(np.abs(x[:,:,0])>1e-15),axis=0)
    eligible = (count>=needed)&(active>=15)&np.isfinite(scores)&(scores>0)&(before>0)&(after>0)
    if not eligible.any():
        return -1,np.nan,int(eligible.sum())
    chosen = int(np.argmax(np.where(eligible,scores,-np.inf)))
    return chosen,float(scores[chosen]),int(eligible.sum())


def walk_forward_choices(stressed_neighbors, *, lookback=252, refit=21, min_history=126, embargo=1):
    n = len(stressed_neighbors)
    chosen = np.full(n,-1,dtype=int)
    records = []
    first = min_history+embargo
    for start in range(first,n,refit):
        stop = start-embargo
        begin = max(0,stop-lookback) if lookback else 0
        selected,score,eligible = causal_choice(stressed_neighbors[begin:stop],lookback=lookback,
                                               min_history=min_history)
        end = min(n,start+refit)
        chosen[start:end] = selected
        records.append({"start":start,"end":end,"train_start":begin,"train_end_exclusive":stop,
                        "candidate":selected,"train_median_stressed_sharpe":score,"eligible":eligible})
    return chosen,pd.DataFrame(records)


def choose_path(matrix, choices):
    result = np.zeros(len(choices))
    active = choices>=0
    result[active] = matrix[np.flatnonzero(active),choices[active]]
    return result


def block_interval(values, *, draws=500, block=10, seed=81723):
    values = np.asarray(values,float)
    rng = np.random.default_rng(seed)
    if np.isfinite(values).sum()<40:
        return {"sharpe_low":np.nan,"sharpe_high":np.nan,"mean_low":np.nan,"mean_high":np.nan}
    samples = []
    blocks = int(np.ceil(len(values)/block))
    for _ in range(draws):
        starts = rng.integers(0,len(values),size=blocks)
        indices = ((starts[:,None]+np.arange(block))%len(values)).ravel()[:len(values)]
        x = values[indices]
        x = x[np.isfinite(x)]
        samples.append((x.mean()/x.std(ddof=1)*np.sqrt(252.) if x.std(ddof=1)>0 else np.nan,x.mean()))
    bounds = np.nanquantile(samples,[.025,.975],axis=0)
    return dict(zip(("sharpe_low","sharpe_high","mean_low","mean_high"),bounds.T.ravel()))
