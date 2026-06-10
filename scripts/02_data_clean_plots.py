#!/usr/bin/env python3
"""
Explore the cleaned TfL journey data.

Figures produced:

  journey_by_time.png                  — total demand by hour, bus + rail stacked
  journey_by_weekday.png               — total demand by day, bus + rail stacked
  journey_station_imbalance_scatter.png — departures vs arrivals per station
  journey_station_imbalance_bars.png   — grouped bar chart, top 20 stations

Input : data/processed/bus_clean.csv
        data/processed/rail_clean.csv

Run from the project root:
    python scripts/02_data_exploration_clean.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from styles import BG, FG, FAINT, CYAN, AMBER, GREEN, apply_theme

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parents[1]
BUS_PATH    = ROOT / "data" / "processed" / "bus_clean.csv"
RAIL_PATH   = ROOT / "data" / "processed" / "rail_clean.csv"
FIGURES_DIR = ROOT / "figures" / "02_exploration_clean"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

bus  = pd.read_csv(BUS_PATH)
rail = pd.read_csv(RAIL_PATH)
df   = pd.concat([bus, rail], ignore_index=True)

print(f"Bus  : {len(bus):,} rows")
print(f"Rail : {len(rail):,} rows")
print(f"Total: {len(df):,} rows")

# Station imbalance charts are produced by 03_feature_stations.py

apply_theme()

CHART_COLORS = {
    "Bus":         AMBER,
    "Underground": CYAN,
    "Other rail":  FAINT,
}

df["mode_chart"] = df["mode"].apply(
    lambda m: m if m in ("Bus", "Underground") else "Other rail"
)

# ---------------------------------------------------------------------------
# Journeys by hour — stacked
# ---------------------------------------------------------------------------

hour_mode = (
    df.groupby(["journey_hour", "mode_chart"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=CHART_COLORS.keys(), fill_value=0)
)

fig, ax = plt.subplots(figsize=(12, 5))
bottom = np.zeros(24)
for mode in hour_mode.columns:
    vals = hour_mode.reindex(range(24), fill_value=0)[mode].values
    ax.bar(range(24), vals, bottom=bottom,
           color=CHART_COLORS[mode], edgecolor=BG, width=0.8, label=mode)
    bottom += vals

y_max = bottom.max()
for start, label in [(7, "AM peak"), (17, "PM peak")]:
    ax.axvspan(start, start + 2, alpha=0.15, color="white", zorder=0)
    ax.text(start + 1, y_max * 1.01, label,
            ha="center", va="bottom", fontsize=7, color=FAINT)
ax.set_title("Total network demand by hour — all modes")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Journeys")
ax.set_xticks(range(0, 24))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_by_time.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_by_time.png")

# ---------------------------------------------------------------------------
# Journeys by weekday — stacked
# ---------------------------------------------------------------------------

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

day_mode = (
    df.groupby(["daytype", "mode_chart"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=DAY_ORDER, columns=CHART_COLORS.keys(), fill_value=0)
)

fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(DAY_ORDER))
for mode in day_mode.columns:
    vals = day_mode[mode].values
    ax.bar(DAY_ORDER, vals, bottom=bottom,
           color=CHART_COLORS[mode], edgecolor=BG, width=0.7, label=mode)
    bottom += vals

ax.set_title("Total network demand by day — all modes")
ax.set_xlabel("")
ax.set_ylabel("Journeys")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=8, loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_by_weekday.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_by_weekday.png")

