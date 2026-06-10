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
# 1. Top 20 stations by total demand
# ---------------------------------------------------------------------------

top20 = stations.nlargest(20, "total_demand").sort_values("total_demand")

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top20.index, top20["total_demand"], color=CYAN, height=0.7, edgecolor=BG)
for bar, val in zip(bars, top20["total_demand"]):
    ax.text(
        val + top20["total_demand"].max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:,}", va="center", ha="left", fontsize=8, color=FG,
    )
ax.set_title("Top 20 stations by total demand — rail / tube")
ax.set_xlabel("Total journeys (departures + arrivals)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xlim(0, top20["total_demand"].max() * 1.18)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_station_top20.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_station_top20.png")

# ---------------------------------------------------------------------------
# 2. AM peak departures vs PM peak arrivals — commuter flow scatter
# ---------------------------------------------------------------------------

# Only stations with enough demand to be meaningful
active = stations[stations["total_demand"] >= 200].copy()

# Label the four quadrants
q_dep = active["am_peak_departures"].median()
q_arr = active["pm_peak_arrivals"].median()

fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(
    active["am_peak_departures"], active["pm_peak_arrivals"],
    c=active["total_demand"], cmap="YlOrRd",
    s=30, alpha=0.8, zorder=3, edgecolors="none",
)

# label top 15 by total demand
top_labels = active.nlargest(15, "total_demand")
for stn, row in top_labels.iterrows():
    ax.annotate(stn, xy=(row["am_peak_departures"], row["pm_peak_arrivals"]),
                xytext=(5, 3), textcoords="offset points", fontsize=7, color=FG)

ax.axvline(q_dep, color=FAINT, linewidth=0.8, linestyle="--")
ax.axhline(q_arr, color=FAINT, linewidth=0.8, linestyle="--")

ax.set_title("AM peak departures vs PM peak arrivals\n(stations with ≥ 200 total journeys)")
ax.set_xlabel("AM peak departures")
ax.set_ylabel("PM peak arrivals")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_station_am_pm_scatter.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_station_am_pm_scatter.png")

# ---------------------------------------------------------------------------
# 3. Peak ratio distribution
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
# 4. Mode share — top 20 stations stacked bars
# ---------------------------------------------------------------------------

share_cols = [f"mode_share_{m.lower().replace(' ', '_')}" for m in MODES
              if f"mode_share_{m.lower().replace(' ', '_')}" in stations.columns]

top20_share = stations.nlargest(20, "total_demand").sort_values("total_demand")
share_data  = top20_share[share_cols].copy()
share_data.columns = [c.replace("mode_share_", "").replace("_", " ").title()
                      for c in share_data.columns]

fig, ax = plt.subplots(figsize=(10, 7))
bottom = np.zeros(len(share_data))

for col in share_data.columns:
    raw_key = col.lower().replace(" ", "_")
    # match back to original mode name for colour lookup
    mode_key = next((m for m in MODES if m.lower().replace(" ", "_") == raw_key), None)
    color = MODE_COLORS.get(mode_key, FAINT) if mode_key else FAINT
    vals = share_data[col].values
    ax.barh(share_data.index, vals, left=bottom,
            color=color, edgecolor=BG, height=0.7, label=col)
    bottom += vals

ax.set_title("Mode share — top 20 stations by total demand")
ax.set_xlabel("Share of departures")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_station_mode_share.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_station_mode_share.png")

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
