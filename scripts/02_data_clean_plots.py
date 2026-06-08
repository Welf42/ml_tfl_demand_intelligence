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

ax.axvspan(7,  9,  alpha=0.08, color="white", label="AM peak (7–9h)")
ax.axvspan(17, 19, alpha=0.08, color="white", label="PM peak (17–19h)")
ax.set_title("Total network demand by hour — all modes")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Journeys")
ax.set_xticks(range(0, 24))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=8, loc="upper left", ncol=2)
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

# ---------------------------------------------------------------------------
# Station imbalance — departures vs arrivals (rail only)
# ---------------------------------------------------------------------------

top_origins      = rail["StartStn"].value_counts().head(20)
top_destinations = rail["EndStation"].value_counts().head(20)
top_stations     = top_origins.index.union(top_destinations.index)

station_df = pd.DataFrame({
    "departures": top_origins.reindex(top_stations, fill_value=0),
    "arrivals":   top_destinations.reindex(top_stations, fill_value=0),
})

# Scatter
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(station_df["departures"], station_df["arrivals"],
           color=CYAN, s=60, zorder=3)

lim = max(station_df["departures"].max(), station_df["arrivals"].max()) * 1.05
ax.plot([0, lim], [0, lim], color=FAINT, linewidth=1, linestyle="--", zorder=2)

for station, row in station_df.iterrows():
    ax.annotate(station, xy=(row["departures"], row["arrivals"]),
                xytext=(5, 3), textcoords="offset points", fontsize=7, color=FG)

ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_title("Departures vs arrivals — rail / tube (top stations)")
ax.set_xlabel("Departures")
ax.set_ylabel("Arrivals")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_station_imbalance_scatter.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_station_imbalance_scatter.png")

# Grouped bars
top20 = (
    station_df.assign(total=station_df["departures"] + station_df["arrivals"])
    .nlargest(20, "total")
    .sort_values("total")
)

y = np.arange(len(top20))
height = 0.35

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(y + height / 2, top20["departures"], height=height,
        color=CYAN,  edgecolor=BG, label="Departures")
ax.barh(y - height / 2, top20["arrivals"],   height=height,
        color=GREEN, edgecolor=BG, label="Arrivals")
ax.set_yticks(y)
ax.set_yticklabels(top20.index, fontsize=8)
ax.set_title("Departures vs arrivals — top 20 stations")
ax.set_xlabel("Journeys")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_station_imbalance_bars.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_station_imbalance_bars.png")
