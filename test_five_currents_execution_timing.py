from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from five_currents_execution_timing import (
    PriceBook, regional_time, trade_pair, forecast_matrix, asset_pnl,
    summarize_matrix, causal_choice, walk_forward_choices, choose_path, block_interval,
)


def bar(stamp,price=100.,symbol="MCL",contract="MCLZ6",volume=1,table="bars_1m"):
    return {"root":"CL","symbol":symbol,"local_symbol":contract,"table":table,
            "ts":pd.Timestamp(stamp),"open":price,"volume":volume,"bar_count":volume}


def target():
    return SimpleNamespace(session=pd.Timestamp("2026-10-08"),root="CL",month="2026-12",
                           identity_basis="verified_month",rank=1.)


@pytest.mark.parametrize("day,clock,utc",[
    ("2025-03-17","london_open","2025-03-17T08:00Z"),
    ("2025-03-17","us_open","2025-03-17T13:30Z"),
    ("2025-03-31","london_open","2025-03-31T07:00Z"),
    ("2025-10-30","us_open","2025-10-30T13:30Z"),
    ("2025-11-03","us_open","2025-11-03T14:30Z"),
    ("2025-03-17","cme_open","2025-03-16T22:00Z"),
    ("2025-01-06","early_utc","2025-01-06T07:00Z"),
    ("2025-06-02","early_utc","2025-06-02T06:00Z"),
])
def test_independent_regional_dst(day,clock,utc):
    assert regional_time(day,clock)==pd.Timestamp(utc)


def test_product_specific_settlement_times():
    assert regional_time("2025-06-02","settlement","GC")==pd.Timestamp("2025-06-02T17:30Z")
    assert regional_time("2025-06-02","settlement","SI")==pd.Timestamp("2025-06-02T17:25Z")
    assert regional_time("2025-06-02","settlement","HG")==pd.Timestamp("2025-06-02T17:00Z")


def test_fill_never_uses_earlier_bar_and_hold_starts_at_actual_fill():
    book=PriceBook(pd.DataFrame([
        bar("2026-10-08T06:55Z",90),bar("2026-10-08T07:05Z",100),
        bar("2026-10-08T08:00Z",110),bar("2026-10-08T08:05Z",101)]))
    row=trade_pair(book,target(),"london_open",60,0)
    assert row["entry_ts"]==pd.Timestamp("2026-10-08T07:05Z")
    assert row["exit_ts"]==pd.Timestamp("2026-10-08T08:05Z")
    assert row["price_return"]==pytest.approx(.01)


def test_exit_cannot_change_month_or_price_instrument():
    book=PriceBook(pd.DataFrame([bar("2026-10-08T07:00Z"),
        bar("2026-10-08T08:00Z",110,contract="MCLF7"),
        bar("2026-10-08T08:00Z",110,symbol="CL",contract="CLZ6")]))
    assert trade_pair(book,target(),"london_open",60,0)["status"]=="missing same-contract exit"


def test_entry_route_does_not_use_future_exit_availability():
    book=PriceBook(pd.DataFrame([bar("2026-10-08T07:00Z",100),
        bar("2026-10-08T07:05Z",99,symbol="CL",contract="CLZ6"),
        bar("2026-10-08T08:05Z",105,symbol="CL",contract="CLZ6")]))
    row=trade_pair(book,target(),"london_open",60,0)
    assert row["symbol"]=="MCL"
    assert row["status"]=="missing same-contract exit"


def test_exit_can_change_cached_rank_table_but_not_contract():
    book=PriceBook(pd.DataFrame([bar("2026-10-08T07:00Z",100),
        bar("2026-10-08T08:00Z",101,table="bars_1m_2m")]))
    assert trade_pair(book,target(),"london_open",60,0)["status"]=="matched"


def test_nontrading_marks_are_not_primary_fills():
    book=PriceBook(pd.DataFrame([bar("2026-10-08T07:00Z",volume=0),
                                bar("2026-10-08T08:00Z",volume=0)]))
    assert trade_pair(book,target(),"london_open",60,0)["status"]=="missing entry"
    assert trade_pair(book,target(),"london_open",60,0,require_trade=False)["status"]=="matched"


def test_maintenance_hold_is_rejected():
    book=PriceBook(pd.DataFrame([bar("2026-10-08T20:00Z"),bar("2026-10-08T21:00Z",101)]))
    assert trade_pair(book,target(),"us_close",60,0)["status"]=="hold leaves trading session"


def test_preopen_offset_does_not_fill_after_maintenance():
    book=PriceBook(pd.DataFrame([bar("2026-10-07T22:00Z"),bar("2026-10-07T23:00Z",101)]))
    assert trade_pair(book,target(),"cme_open",60,-15)["status"]=="entry outside trading session"


def test_forecast_lag_and_sign_and_gross_cap():
    dates=pd.bdate_range("2026-10-01",periods=6)
    values=np.array([[.2]*5,[-.2]*5,[.2]*5,[-.2]*5,[.2]*5,[-.2]*5])
    sessions=dates[3:]
    weights,source=forecast_matrix(values,dates,sessions,variant="change_magnitude")
    assert np.max(np.abs(weights).sum(axis=1))<=1
    np.testing.assert_allclose(weights[0],np.full(5,.2))
    assert source.iloc[0]==dates[2]
    older,source=forecast_matrix(values,dates,sessions,lag=1)
    np.testing.assert_allclose(older[0],values[1])
    assert source.iloc[0]==dates[1]
    with pytest.raises(ValueError):
        forecast_matrix(values,dates,sessions,lag=-1)


def test_micro_fee_not_multiplied_by_ten_and_silver_tick():
    weight=np.ones((1,5))*.2
    price=np.array([[100.,3.,4000.,50.,6.]])
    move=np.ones((1,5))*.01
    gross,fee,slip=asset_pnl(weight,move,price)
    assert fee[0,0]==pytest.approx(.2*2.16/(100*100))
    assert slip[0,3]==pytest.approx(.2*2*.01/50)
    reverse,rfee,rslip=asset_pnl(-weight,move,price)
    np.testing.assert_allclose(reverse,-gross)
    np.testing.assert_allclose(rfee,fee)
    np.testing.assert_allclose(rslip,slip)


def test_silver_slippage_allowance_is_not_mislabelled_as_the_exchange_tick():
    from five_currents_execution_timing import EXCHANGE_TICKS,SLIPPAGE_PRICE_PER_SIDE
    assert EXCHANGE_TICKS["SI"]==.005
    assert SLIPPAGE_PRICE_PER_SIDE["SI"]==2*EXCHANGE_TICKS["SI"]


def test_inactive_missing_prices_are_zero_but_active_missing_is_unknown():
    weight=np.array([[0.,.2,0.,0.,0.]])
    gross,fee,slip=asset_pnl(weight,np.full((1,5),np.nan),np.full((1,5),np.nan))
    assert gross[0,0]==0 and fee[0,0]==0 and slip[0,0]==0
    assert np.isnan(gross[0,1]) and np.isnan(fee[0,1])


def test_pnl_vectorization_broadcasts_clock_cells_without_zeroing_missing_active_legs():
    weight=np.array([[[0.,.2,0.,0.,0.]]])
    gross,fee,slip=asset_pnl(weight,np.full((3,1,5),np.nan),np.full((3,1,5),np.nan))
    assert gross.shape==(3,1,5)
    assert (gross[:,:,0]==0).all()
    assert np.isnan(gross[:,:,1]).all()


def test_stats_preserve_missing_and_initial_loss_drawdown():
    out=summarize_matrix(np.array([[-.1,np.nan],[0,np.nan],[.02,np.nan]]))
    assert out["days"][0]==3 and out["days"][1]==0
    assert np.isnan(out["sharpe"][1])
    assert np.isnan(out["observed_total_return"][1])
    assert out["observed_max_drawdown"][0]==pytest.approx(-.1)


def test_no_common_observations_is_not_a_zero_return_statistic():
    out=summarize_matrix(np.array([]))
    assert out["days"][0]==0
    assert np.isnan(out["sharpe"][0])
    assert np.isnan(out["observed_total_return"][0])


def test_causal_fits_are_prefix_invariant_and_embargoed():
    rng=np.random.default_rng(887)
    matrix=rng.normal(.0002,.001,size=(350,4,3))
    choices,audit=walk_forward_choices(matrix,refit=21)
    changed=matrix.copy()
    changed[220:]=100
    other,_=walk_forward_choices(changed,refit=21)
    np.testing.assert_array_equal(choices[:220],other[:220])
    assert (audit.train_end_exclusive<audit.start).all()
    assert (choices[:127]==-1).all()


def test_no_cherry_pick_of_single_lucky_training_half():
    x=np.ones((150,2,3))*.001
    x[:75,0,:]=-.001
    x[75:,0,:]=.01
    x[:,1,:]+=np.sin(np.arange(150))[:,None]*.0002
    chosen,_,_=causal_choice(x)
    assert chosen==1


def test_cash_and_unknown_selected_return_are_distinct():
    paths=np.array([[np.nan,1],[np.nan,1],[.01,1]])
    np.testing.assert_allclose(choose_path(paths,np.array([-1,0,0])),[0,np.nan,.01],equal_nan=True)


def test_bootstrap_is_reproducible():
    values=np.random.default_rng(8).normal(.0001,.003,100)
    assert block_interval(values,draws=40)==block_interval(values,draws=40)


def test_event_window_follows_both_regional_clocks_during_dst_mismatch():
    from run_five_currents_regional_blocks import event_pair
    wanted=SimpleNamespace(session=pd.Timestamp("2025-03-17"),root="CL",month="2025-12",
                           identity_basis="verified_month",rank=1.)
    book=PriceBook(pd.DataFrame([bar("2025-03-17T08:00Z",100,contract="MCLZ5"),
                                bar("2025-03-17T13:30Z",101,contract="MCLZ5")]))
    row=event_pair(book,wanted,"london_open","us_open",0)
    assert row["status"]=="matched"
    assert row["exit_ts"]-row["entry_ts"]==pd.Timedelta(hours=5.5)


def test_held_branch_really_uses_the_older_settled_forecast():
    from run_five_currents_execution_timing_study import pnl_grid
    dates=pd.bdate_range("2026-10-01",periods=7)
    sessions=dates[3:]
    original=np.arange(35,dtype=float).reshape(7,5)/200
    source=SimpleNamespace(dates=dates,sessions=sessions,values={"portfolio_net":original/2},
                           settled={"portfolio_net":original})
    shape=(7,9,6,len(sessions),5)
    cube={"returns":np.ones(shape)*.01,"entry":np.ones(shape)*100}
    _,_,_,actual=pnl_grid(source,cube,"held__portfolio_net","net_magnitude")
    expected,_=forecast_matrix(original,dates,sessions,lag=1)
    np.testing.assert_allclose(actual,expected)
