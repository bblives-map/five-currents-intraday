"""Vectorized reporting only; does not change signals or execution marks."""
import numpy as np
import pandas as pd

from five_currents_front_intraday import ROOTS
from five_currents_opening_study import FOUR, stats


def collect_metrics(paths, policy, signal, metrics, curves, *, cost_only=False):
    columns = ["gross_return", "commission_return", "slippage_return", "net_return", "slippage_net_return"]
    for (variant, hours), group in paths.groupby(["variant", "hours"]):
        pivot = group.pivot(index="session", columns="root", values=[*columns, "weight", "ready"])
        index = pivot.index
        for book, roots in (("four", FOUR), ("five", ROOTS), *((root, (root,)) for root in ROOTS)):
            ready = np.equal(pivot["ready"].reindex(columns=roots).to_numpy(), True).all(axis=1)
            exposure = np.abs(pivot["weight"].reindex(columns=roots).to_numpy(float)).sum(axis=1)
            values = {column: np.where(ready, pivot[column].reindex(columns=roots).to_numpy(float).sum(axis=1), np.nan)
                      for column in columns}
            metadata = {"policy": policy, "signal": signal, "variant": variant,
                        "hours": int(hours), "book": book}
            active = ready & (exposure > 1e-12)
            for basis in ("gross_return", "net_return", "slippage_net_return"):
                valid = np.flatnonzero(np.isfinite(values[basis]))
                metrics.append({**metadata, "cost_basis": basis, "active_days": int(active.sum()),
                                "start": index[valid[0]] if len(valid) else None,
                                "end": index[valid[-1]] if len(valid) else None,
                                "mean_gross_exposure": float(exposure[ready].mean()) if ready.any() else np.nan,
                                **stats(values[basis])})
            if book in ("four", "five") and not cost_only:
                totals = pd.DataFrame({**values, "gross_exposure": exposure, "ready": ready}, index=index)
                curves.append(totals.reset_index().assign(**metadata))
