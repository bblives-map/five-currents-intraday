"""Validate the saved Five Currents ATR study and rendered report."""
from __future__ import annotations

import base64
import re
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__import__("os").environ.get("FIVE_CURRENTS_DATA_ROOT", str(Path(__file__).resolve().parent))) / "five_currents_intraday_atr_trail_20261111"


def main() -> None:
    report = (OUT / "five_currents_joint_intraday_atr_trail_report.html").read_text(
        encoding="utf-8"
    )
    images = re.findall(r'data:image/png;base64,([^\"]+)', report)
    if len(images) != 5 or not all(base64.b64decode(value).startswith(b"\x89PNG")
                                   for value in images):
        raise AssertionError("Report does not contain five valid embedded PNG charts")
    metrics = pd.read_csv(OUT / "performance_metrics.csv")
    daily = pd.read_parquet(OUT / "daily_returns.parquet")
    expected = metrics.loc[
        metrics.model.eq("Fixed exit") & metrics.series.eq("Portfolio")
        & metrics["sample"].eq("Full"), "daily_sharpe"
    ].iloc[0]
    portfolio = daily["Fixed exit|Portfolio"].dropna()
    replayed = portfolio.mean() / portfolio.std(ddof=1) * np.sqrt(252.0)
    np.testing.assert_allclose(replayed, expected, atol=1e-12, rtol=0)
    trades = pd.read_parquet(OUT / "selected_trade_audit.parquet")
    if int(trades.atr_fallback.sum()) != 0:
        raise AssertionError("Selected trail contains ATR fallback trades")
    refits = pd.read_csv(OUT / "walkforward_refits.csv")
    if not (pd.to_datetime(refits.fit_through) < pd.to_datetime(refits.effective_start)).all():
        raise AssertionError("Walk-forward refit uses contemporaneous or future data")
    if daily.index.max().date().isoformat() != "2026-11-10":
        raise AssertionError("Daily replay does not end at the expected latest close")
    print(
        f"validated {len(images)} embedded PNGs, {len(portfolio)} portfolio days, "
        f"SR {replayed:.6f}, latest {daily.index.max().date()}"
    )


if __name__ == "__main__":
    main()
