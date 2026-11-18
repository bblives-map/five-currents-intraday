# Five Currents Intraday

The intraday research companion to [Five Currents Interday](https://github.com/bblives-map/five-currents-interday): daily-alpha origins, intraday signal transforms, regional clocks, front/second-nearby execution, complete backtest and report generators, ablations and robustness/sensitivity experiments.

This repository does not contain a second copy of the daily fitting engine. It consumes the same daily alpha definitions and private daily replay as the companion, avoiding inconsistent formulas or silently divergent refit conventions.

## Install and test

Clone both repositories next to one another, create an isolated Python environment, then:

```bash
python -m pip install -e "../five-currents-interday[test]"
python -m pip install -e ".[test]"
python -m pytest -q
python tools/check_inputs.py
```

Raw market data, private pickle caches, live holdings and dashboard access credentials are not distributed. The exported parameter/rule configurations and aggregate strategy-return reports are not substitutes for licensed one-minute bars. Set `FIVE_CURRENTS_DATA_ROOT` to a separate private workspace containing the layout in `private_input_manifest.json`. Only load locally generated, trusted pickle caches; never load downloaded untrusted pickles.

One named frozen-day integration check is retained but skipped when its private original audit JSON is absent; see `conftest.py`. All synthetic execution, cost, direction, lag, clock and allocation checks remain runnable. `results/` and `reports/archived_backtest.html` include aggregate return histories and per-asset curves from the independent archived replay.

## Alpha lineage

The original families are defined in the interday core and curve-signal modules. `five_currents_opening_study.py` rebuilds settled versus one-session-lagged volume/OI features, preserving exact-contract alignment. Curve-depth sensitivity is explicit: the current daily book uses CL/NG/GC 1-6 and SI/HG 1-4; the selected intraday audit has a separate GC 1-4 variant. This is documented rather than disguised as a universally optimized curve.

`five_currents_group_clock_core.py` defines gross family votes, their changes, net selected-book exposure, net exposure changes and second changes. Magnitude, sign and reversed treatments are separate candidates. Regional clocks are Asia open/close, London open/close and New York open/close, with local-zone daylight saving. `five_currents_front_intraday.py` retains the earlier 06:00/07:00 UTC opening tests; those are not interchangeable with New York open.

`calculate_five_currents_intraday_positions.py` reconstructs the frozen ten-leg schedule for CL/NG/GC/HG, combines overlapping intervals and exposes the same panel consumed by the interday dashboard. The frozen per-leg rules are in `config/frozen_intraday_rules.csv`. No signal is silently inverted to rescue a recent losing trade. Silver remains in cross-sectional signal construction but is excluded from the deployed four-asset intraday execution audit; exploratory five-market and no-silver comparisons remain available.

## Execution and accounting

The corrected `build_five_currents_frontmonth_prices.py` path uses stored front/second-nearby one-minute bars, prior-session liquidity, an expiry constraint, positive traded volume/count and the same actual contract for entry and exit. Sparse clocks may use the permitted nearby fill window, which is recorded in the trade ledger. Holdings are measured from actual entry. Bar-close versus nominal-clock offsets are retained in the independent deployed audit.

Commission is $2.16 per micro-lot round trip, plus the explicitly coded spread/slippage assumptions. Parent/micro multipliers are applied once. Costs, netting assumptions and unit-study weights are not a dollar-VaR allocation. Missing active trade outcomes are not zero-filled, and per-asset studies do not require unrelated assets or all holding periods to have matching bars.

## Full research and tests

| Question | Code |
| --- | --- |
| Signal versus held position, lag and reversal | `five_currents_intraday_replay.py`, `five_currents_front_intraday.py`, `five_currents_opening_study.py` |
| Regional clocks, holding periods and delayed entry | `five_currents_group_clock_core.py`, `run_five_currents_group_clock.py`, `audit_five_currents_market_clock_tolerance.py` |
| Corrected nearby-contract fills and coverage | `build_five_currents_frontmonth_prices.py`, `run_five_currents_frontmonth_alpha_audit.py`, `verify_five_currents_frontmonth_results.py` |
| Fixed versus rolling family/asset allocation | `run_five_currents_frontmonth_studies.py`, `run_five_currents_rolling_edge.py` |
| Leave-outs, costs, top-win removal and perturbations | `stress_five_currents_execution_timing.py`, `run_five_currents_consistency_study.py`, `validate_five_currents_intraday_atr_study.py` |
| Shuffled Sharpe admission | `five_currents_shuffled_admission.py`, `run_five_currents_shuffled_admission.py`, `verify_five_currents_shuffled_admission.py` |
| Independent deployed sign/time/price replay | `audit_five_currents_deployed_intraday.py`, `report_five_currents_deployed_intraday_audit.py` |
| Other report generators and regression checks | `build_five_currents_*report.py`, `report_five_currents_*.py`, `test_*.py`, `verify_five_currents_*.py` |

The shuffled-admission experiment gates the daily family selection feeding these tests. It is not presented as an independent intraday permutation test. The archived negative/unsuccessful robustness experiments remain part of the release. Earlier sampled-cache and exact-calendar-contract studies are retained as lineage; the corrected front-route replay is the reference for nearby intraday execution.

## Reproduce the corrected study

After supplying the private input manifests and the companion engine:

```bash
python build_five_currents_frontmonth_prices.py --help
python run_five_currents_frontmonth_studies.py
python audit_five_currents_deployed_intraday.py
python report_five_currents_deployed_intraday_audit.py
python calculate_five_currents_intraday_positions.py --help
```

Some scripts have fixed historical output directories and refuse to overwrite completed studies. Reproduce them in a fresh private workspace. `FIVE_CURRENTS_DATA_ROOT` controls data-rooted paths; the original relative-layout stages should be launched from that workspace. Commands that prepare bars or extend an audit can explicitly read your configured PostgreSQL database; unit tests and cached numerical helpers do not call a broker. No new live acquisition was performed for this release.

## Interpretation

The frozen per-asset family/treatment/clock/holding rules were selected with descriptive hindsight. Causal daily inputs, delayed fills and subsequent rolling allocation do not remove that roster/clock selection bias. The ten-rule sum is an unallocated research exposure, not a validated deployable $1m-VaR portfolio. Keep full-history discovery, trailing allocation, stress checks and prospective paper trading separate when assessing the results.

`source_manifest.json` preserves provenance, and the reports explicitly identify their archived cutoff. Reproducing a source-locked historical run with sanitized exports requires regenerating that code lock; no checksum guard is bypassed. Public visibility does not grant an open-source redistribution license; all rights reserved unless the owner provides a separate license.
