#!/usr/bin/env python3
"""
Station-level feature engineering for TfL rail journeys.

Builds one row per station with demand, peak, mode, and imbalance features
for use in downstream clustering and classification.

Features:
  total_departures      — journeys starting at this station
  total_arrivals        — journeys ending at this station
  total_demand          — sum of the above
  am_peak_departures    — departures in AM peak (7–9h)
  am_peak_arrivals      — arrivals in AM peak
  pm_peak_departures    — departures in PM peak (17–19h)
  pm_peak_arrivals      — arrivals in PM peak
  peak_ratio            — (am_dep + pm_arr) / total_demand
  imbalance_ratio       — (departures − arrivals) / total_demand
  dominant_mode         — most frequent mode at station
  mode_diversity        — number of distinct modes (transfer proxy)
  mode_share_*          — fraction of departures per mode

Input : data/processed/rail_clean.csv
Output: data/processed/station_features.csv
        figures/03_stations/

Run from the project root:
    python scripts/03_feature_stations.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from styles import BG, FG, FAINT, GRID, CYAN, AMBER, GREEN, RED, apply_theme

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT         = Path(__file__).resolve().parents[1]
RAIL_PATH    = ROOT / "data" / "processed" / "rail_clean.csv"
FEAT_PATH    = ROOT / "data" / "processed" / "station_features.csv"
FIGURES_DIR  = ROOT / "figures" / "03_stations"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

apply_theme()

MODES = ["Underground", "National Rail", "Overground", "DLR",
         "Heathrow Express", "Tram", "Other"]

MODE_COLORS = {
    "Underground":      CYAN,
    "National Rail":    AMBER,
    "Overground":       GREEN,
    "DLR":              "#A78BFA",
    "Heathrow Express": "#FB923C",
    "Tram":             RED,
    "Other":            FAINT,
}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

rail = pd.read_csv(RAIL_PATH)
print(f"Loaded {len(rail):,} rail journeys")

# ---------------------------------------------------------------------------
# Base counts
# ---------------------------------------------------------------------------

dep = rail.groupby("StartStn").size().rename("total_departures")
arr = rail.groupby("EndStation").size().rename("total_arrivals")

stations = pd.concat([dep, arr], axis=1).fillna(0).astype(int)
stations.index.name = "station"
stations["total_demand"] = stations["total_departures"] + stations["total_arrivals"]

print(f"Stations with departures : {dep.shape[0]}")
print(f"Stations with arrivals   : {arr.shape[0]}")
print(f"Total unique stations    : {len(stations)}")

# ---------------------------------------------------------------------------
# Peak period counts
# ---------------------------------------------------------------------------

for period, col_dep, col_arr in [
    ("AM peak", "am_peak_departures", "am_peak_arrivals"),
    ("PM peak", "pm_peak_departures", "pm_peak_arrivals"),
]:
    p = rail[rail["period"] == period]
    stations[col_dep] = p.groupby("StartStn").size().reindex(stations.index, fill_value=0)
    stations[col_arr] = p.groupby("EndStation").size().reindex(stations.index, fill_value=0)

# ---------------------------------------------------------------------------
# Derived ratios
# ---------------------------------------------------------------------------

# peak_ratio: how much of total demand falls in the "core" peak windows
# (AM departures model commuter outflow; PM arrivals model commuter return)
stations["peak_ratio"] = (
    (stations["am_peak_departures"] + stations["pm_peak_arrivals"])
    / stations["total_demand"].replace(0, np.nan)
)

# imbalance_ratio: positive → net origin, negative → net destination
stations["imbalance_ratio"] = (
    (stations["total_departures"] - stations["total_arrivals"])
    / stations["total_demand"].replace(0, np.nan)
)

# ---------------------------------------------------------------------------
# Mode features (departure side)
# ---------------------------------------------------------------------------

mode_dep = (
    rail.groupby(["StartStn", "mode"])
    .size()
    .unstack(fill_value=0)
    .reindex(stations.index, fill_value=0)
)

# mode_share per mode
for mode in MODES:
    if mode in mode_dep.columns:
        col = f"mode_share_{mode.lower().replace(' ', '_')}"
        stations[col] = (
            mode_dep[mode] / mode_dep.sum(axis=1).replace(0, np.nan)
        ).fillna(0)

# mode_diversity = number of distinct modes with at least 1 departure
if not mode_dep.empty:
    stations["mode_diversity"] = (mode_dep > 0).sum(axis=1)
    stations["dominant_mode"]  = mode_dep.idxmax(axis=1)
else:
    stations["mode_diversity"] = 0
    stations["dominant_mode"]  = "Unknown"

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

stations.to_csv(FEAT_PATH)
print(f"\nSaved: {FEAT_PATH.relative_to(ROOT)}  ({len(stations)} rows)")

# ---------------------------------------------------------------------------
# 1. AM peak departures vs PM peak arrivals — commuter flow scatter
# ---------------------------------------------------------------------------

# Only stations with enough demand to be meaningful
active = stations[stations["total_demand"] >= 200].copy()

# Bubble size scaled to total demand
size_min, size_max = 20, 600
d = active["total_demand"]
active["dot_size"] = size_min + (d - d.min()) / (d.max() - d.min()) * (size_max - size_min)

lim = max(active["am_peak_departures"].max(), active["pm_peak_arrivals"].max()) * 1.1

fig, ax = plt.subplots(figsize=(9, 9))

ax.scatter(
    active["am_peak_departures"], active["pm_peak_arrivals"],
    c=CYAN, s=active["dot_size"],
    alpha=0.6, zorder=3, edgecolors=BG, linewidths=0.4,
)

# Diagonal reference line (y = x → symmetric AM/PM flow)
ax.plot([0, lim], [0, lim], color=FAINT, linewidth=1, linestyle="--", zorder=2)

# Labels for top 20 stations by total demand
for stn, row in active.nlargest(20, "total_demand").iterrows():
    ax.annotate(stn, xy=(row["am_peak_departures"], row["pm_peak_arrivals"]),
                xytext=(5, 3), textcoords="offset points", fontsize=7, color=FG)

ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_title("AM peak departures vs PM peak arrivals\n(bubble size = total demand)")
ax.set_xlabel("AM peak departures")
ax.set_ylabel("PM peak arrivals")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_station_am_pm_scatter.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_station_am_pm_scatter.png")

# ---------------------------------------------------------------------------
# 2. Peak ratio distribution
# ---------------------------------------------------------------------------

pr = stations["peak_ratio"].dropna()

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(pr, bins=40, color=CYAN, edgecolor=BG, alpha=0.9)
ax.axvline(pr.median(), color=AMBER, linewidth=1.5, linestyle="--",
           label=f"Median {pr.median():.2f}")
ax.set_title("Peak ratio distribution — all stations\n"
             "(AM departures + PM arrivals) / total demand")
ax.set_xlabel("Peak ratio")
ax.set_ylabel("Stations")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_station_peak_ratio.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_station_peak_ratio.png")

# ---------------------------------------------------------------------------
# 3. Station imbalance scatter — total departures vs total arrivals
# ---------------------------------------------------------------------------

top_origins      = rail["StartStn"].value_counts().head(20)
top_destinations = rail["EndStation"].value_counts().head(20)
top_stns         = top_origins.index.union(top_destinations.index)

stn_df = pd.DataFrame({
    "departures": top_origins.reindex(top_stns, fill_value=0),
    "arrivals":   top_destinations.reindex(top_stns, fill_value=0),
})

lim = max(stn_df["departures"].max(), stn_df["arrivals"].max()) * 1.05

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(stn_df["departures"], stn_df["arrivals"], color=CYAN, s=60, zorder=3)
ax.plot([0, lim], [0, lim], color=FAINT, linewidth=1, linestyle="--", zorder=2)
for stn, row in stn_df.iterrows():
    ax.annotate(stn, xy=(row["departures"], row["arrivals"]),
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

# ---------------------------------------------------------------------------
# 4. Station imbalance bars — grouped departures vs arrivals top 20
# ---------------------------------------------------------------------------

top20_stn = (
    stn_df.assign(total=stn_df["departures"] + stn_df["arrivals"])
    .nlargest(20, "total")
    .sort_values("total")
)

y      = np.arange(len(top20_stn))
height = 0.35

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(y + height / 2, top20_stn["departures"], height=height,
        color=CYAN,  edgecolor=BG, label="Departures")
ax.barh(y - height / 2, top20_stn["arrivals"],   height=height,
        color=GREEN, edgecolor=BG, label="Arrivals")
ax.set_yticks(y)
ax.set_yticklabels(top20_stn.index, fontsize=8)
ax.set_title("Departures vs arrivals — top 20 stations")
ax.set_xlabel("Journeys")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_station_imbalance_bars.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_station_imbalance_bars.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"""
--- Station features summary ---
Total stations          : {len(stations)}
Active (≥ 200 journeys) : {(stations['total_demand'] >= 200).sum()}
Median peak ratio       : {stations['peak_ratio'].median():.3f}
Max total demand        : {stations['total_demand'].max():,}  ({stations['total_demand'].idxmax()})
Most imbalanced origin  : {stations['imbalance_ratio'].idxmax()}  ({stations['imbalance_ratio'].max():.2f})
Most imbalanced dest    : {stations['imbalance_ratio'].idxmin()}  ({stations['imbalance_ratio'].min():.2f})
""")
