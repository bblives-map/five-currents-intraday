"""Plot the audited Monday intraday model exposures in Singapore time."""
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from calculate_five_currents_intraday_positions import active_intervals


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
AUDIT = ROOT / "five_currents_frontmonth_20261025" / "monday_20261026_signal_audit.json"
OUTPUT = ROOT / "five_currents_frontmonth_20261025" / "monday_20261026_gantt.png"
SGT = ZoneInfo("Asia/Singapore")
ASSETS = ("CL", "NG", "GC", "HG")


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    session = pd.Timestamp(audit["session"])
    intervals = active_intervals(audit["positions"], session)
    start = session.tz_localize(SGT) + pd.Timedelta(hours=9)
    end = start + pd.Timedelta(hours=19)
    midnight = (session + pd.Timedelta(days=1)).tz_localize(SGT)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(15, 6.4), dpi=150)
    fig.patch.set_facecolor("#f7f4ed")
    ax.set_facecolor("#fcfaf5")
    ax.axvspan(midnight, end, color="#e9edf0", alpha=0.55, zorder=0)
    for i, root in enumerate(ASSETS):
        y = len(ASSETS) - 1 - i
        if i % 2:
            ax.axhspan(y - 0.45, y + 0.45, color="#e7ebe7", alpha=0.34, zorder=0)
        for begin, finish, weight in intervals.get(root, []):
            magnitude = abs(weight)
            weak = magnitude < 0.005
            color = ("#a9d7bc" if weak else "#167665") if weight > 0 else ("#edb8b3" if weak else "#b54843")
            label_color = "#173b31" if weak else "white"
            left, right = mdates.date2num(begin), mdates.date2num(finish)
            ax.add_patch(Rectangle((left, y - 0.30), right - left, 0.60,
                                   facecolor=color, edgecolor="none", zorder=3))
            label = f"{weight * 100:+.3f}%"
            ax.text((left + right) / 2, y, label, ha="center", va="center",
                    color=label_color, fontsize=11, fontweight="bold", zorder=4)
        ax.axhline(y - 0.5, color="#d9ded8", lw=0.8, zorder=1)

    ticks = pd.date_range(start, end, freq="2h")
    for tick in ticks:
        ax.axvline(tick, color="#d7ddd8", lw=0.8, zorder=1)
    ax.axvline(midnight, color="#7f8c90", lw=1.2, ls="--", zorder=2)
    ax.set_xlim(start, end)
    ax.set_ylim(-0.65, len(ASSETS) - 0.25)
    ax.set_yticks(range(len(ASSETS)), labels=list(reversed(ASSETS)), fontsize=13, fontweight="bold")
    ax.set_xticks(ticks)
    ax.set_xticklabels([tick.strftime("%H:%M") for tick in ticks])
    ax.tick_params(axis="both", length=0, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.07, 0.935, "FIVE CURRENTS  /  MONDAY 26 OCTOBER 2026", color="#355248",
             fontsize=10, fontweight="bold")
    fig.text(0.07, 0.875, "Planned intraday exposure by asset", color="#132c29",
             fontsize=22, fontweight="bold")
    fig.text(0.07, 0.825, "Singapore time (SGT)  |  Label = signed model weight, not contract lots",
             color="#5b6a63", fontsize=11)
    fig.text(0.70, 0.795, "MON 26 OCT", color="#566b64", fontsize=10, fontweight="bold")
    fig.text(0.865, 0.795, "TUE 27 OCT", color="#566b64", fontsize=10, fontweight="bold")
    fig.legend(handles=[Patch(facecolor="#167665", label="Long"),
                        Patch(facecolor="#b54843", label="Short"),
                        Patch(facecolor="#a9d7bc", label="Small long")],
               loc="lower left", bbox_to_anchor=(0.07, 0.093), ncol=3,
               frameon=False, fontsize=10)
    fig.text(0.07, 0.055,
             "CL 09:00 SGT entry has passed. Blocks show scheduled exposure; fills require an eligible traded bar within 10 minutes.",
             color="#65716c", fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.76, bottom=0.19)
    fig.savefig(OUTPUT, dpi=170, facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT.with_suffix(".svg"), facecolor=fig.get_facecolor())
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
