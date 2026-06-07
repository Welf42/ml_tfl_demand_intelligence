#!/usr/bin/env python3
"""
Data exploration for TfL Oyster card journey data (November 2009 week).

Questions answered:
- What does one row represent, and what columns exist?
- Which modes dominate the network?
- When is demand highest — by hour and by day of week?
- Which stations are the busiest origins and destinations?
- What are the top OD pairs for Underground journeys?
- How does demand split between peak and off-peak?
- What candidate cleaning rules should we apply?

Run from the project root:
    python scripts/01_data_exploration.py
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

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures" / "01_exploration"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df = pd.read_csv(RAW_DIR / "Nov09JnyExport.csv")

print(f"Shape: {df.shape}")
print(f"\nColumns and types:\n{df.dtypes.to_string()}")
print(f"\nMissing values:\n{df.isna().sum().to_string()}")
print(f"\nSample:\n{df.head(3).to_string()}")

# ---------------------------------------------------------------------------
# Mode classification
# ---------------------------------------------------------------------------
# SubSystem can be compound (e.g. 'LUL/NR'). Bus journeys are identified by
# StartStn == 'Bus' (confirmed: all LTB rows have StartStn == 'Bus').

df["mode"] = "Other"
df.loc[df["SubSystem"].str.contains("TRAM"), "mode"] = "Tram"
df.loc[df["SubSystem"].str.contains("HEX"),  "mode"] = "Heathrow Express"
df.loc[df["SubSystem"].str.contains("LRC"),  "mode"] = "Overground"
df.loc[df["SubSystem"].str.contains("NR"),   "mode"] = "National Rail"
df.loc[df["SubSystem"].str.contains("DLR"),  "mode"] = "DLR"
df.loc[df["SubSystem"].str.contains("LUL"),  "mode"] = "Underground"
df.loc[df["StartStn"] == "Bus",              "mode"] = "Bus"

mode_counts = df["mode"].value_counts()
print("\n--- Journeys by mode ---")
print(mode_counts.to_string())

apply_theme()

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(
    mode_counts.index[::-1], mode_counts.values[::-1],
    color=CYAN, height=0.6, edgecolor=BG,
)
for bar, val in zip(bars, mode_counts.values[::-1]):
    ax.text(
        val + mode_counts.max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:,}", va="center", ha="left", fontsize=9, color=FG,
    )
ax.set_title("Journeys by mode")
ax.set_xlabel("Journeys")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xlim(0, mode_counts.max() * 1.18)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "demand_by_mode.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: demand_by_mode.png")

# ---------------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------------
# Bus journeys have ExTime == 0 (no tap-out recorded). Temporal analysis
# is restricted to rail/tube journeys where ExTime > 0.
# Some overnight journeys have ExTime > 1440; wrap with modulo to get hour.

print(f"\nExTime == 0: {(df['ExTime'] == 0).sum():,} rows (bus journeys have no exit time)")

rail = df[(df["mode"] != "Bus") & (df["ExTime"] > 0)].copy()
rail["exit_hour"] = (rail["ExTime"] % 1440) // 60

print(f"Rail/tube rows with valid ExTime: {len(rail):,}")

# Demand by exit hour — all rail modes
hour_counts = rail.groupby("exit_hour").size().rename("journeys")

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(hour_counts.index, hour_counts.values, color=CYAN, edgecolor=BG, width=0.8)
ax.axvspan(7,  9,  alpha=0.15, color=AMBER, label="AM peak (7–9h)")
ax.axvspan(17, 19, alpha=0.15, color=GREEN, label="PM peak (17–19h)")
ax.set_title("Journeys by exit hour — rail / tube")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Journeys")
ax.set_xticks(range(0, 24))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "demand_by_hour.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: demand_by_hour.png")

# Underground only
tube_hour = rail[rail["mode"] == "Underground"].groupby("exit_hour").size()

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(tube_hour.index, tube_hour.values, color=GREEN, edgecolor=BG, width=0.8)
ax.axvspan(7,  9,  alpha=0.15, color=AMBER, label="AM peak (7–9h)")
ax.axvspan(17, 19, alpha=0.15, color=RED,   label="PM peak (17–19h)")
ax.set_title("Journeys by exit hour — Underground")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Journeys")
ax.set_xticks(range(0, 24))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "demand_by_hour_tube.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: demand_by_hour_tube.png")

# Demand by day of week
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
day_counts = df.groupby("daytype").size().reindex(DAY_ORDER).rename("journeys")

print("\n--- Journeys by day ---")
print(day_counts.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
colors = [AMBER if d in ("Sat", "Sun") else CYAN for d in DAY_ORDER]
ax.bar(day_counts.index, day_counts.values, color=colors, edgecolor=BG, width=0.7)
ax.set_title("Journeys by day of week")
ax.set_xlabel("")
ax.set_ylabel("Journeys")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(FIGURES_DIR / "demand_by_weekday.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: demand_by_weekday.png")

# ---------------------------------------------------------------------------
# Station analysis — exclude Bus and Unstarted
# ---------------------------------------------------------------------------

station_df = df[(df["StartStn"] != "Bus") & (df["StartStn"] != "Unstarted")].copy()
print(f"\nStation journeys (excluding Bus and Unstarted): {len(station_df):,}")

top_origins      = station_df["StartStn"].value_counts().head(20)
top_destinations = station_df["EndStation"].value_counts().head(20)

print("\n--- Top 10 origins ---")
print(top_origins.head(10).to_string())
print("\n--- Top 10 destinations ---")
print(top_destinations.head(10).to_string())

def station_chart(series, title, filename, color):
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(
        series.index[::-1], series.values[::-1],
        color=color, height=0.7, edgecolor=BG,
    )
    for bar, val in zip(bars, series.values[::-1]):
        ax.text(
            val + series.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", ha="left", fontsize=8, color=FG,
        )
    ax.set_title(title)
    ax.set_xlabel("Journeys")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(0, series.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {filename}")

station_chart(top_origins,      "Top 20 origin stations",      "top_origins.png",      CYAN)
station_chart(top_destinations, "Top 20 destination stations", "top_destinations.png", GREEN)

# ---------------------------------------------------------------------------
# Top OD pairs — Underground only
# ---------------------------------------------------------------------------

tube_od = station_df[station_df["mode"] == "Underground"].copy()
tube_od["od_pair"] = tube_od["StartStn"] + " → " + tube_od["EndStation"]
top_od = tube_od["od_pair"].value_counts().head(20)

print("\n--- Top 10 OD pairs (Underground) ---")
print(top_od.head(10).to_string())

station_chart(top_od, "Top 20 OD pairs — Underground", "top_od_pairs.png", AMBER)

# ---------------------------------------------------------------------------
# Peak / off-peak split
# ---------------------------------------------------------------------------

def peak_label(hour):
    if 7 <= hour < 10:
        return "AM peak (7–9h)"
    if 17 <= hour < 20:
        return "PM peak (17–19h)"
    if hour >= 23 or hour < 5:
        return "Night"
    return "Off-peak"

PEAK_ORDER  = ["AM peak (7–9h)", "Off-peak", "PM peak (17–19h)", "Night"]
PEAK_COLORS = [CYAN, FAINT, GREEN, "#444444"]

rail["period"] = rail["exit_hour"].apply(peak_label)
peak_totals = rail["period"].value_counts().reindex(PEAK_ORDER)

print("\n--- Rail journeys by period ---")
print(peak_totals.to_string())

fig, ax = plt.subplots(figsize=(8, 4))
colors = [PEAK_COLORS[PEAK_ORDER.index(p)] for p in PEAK_ORDER[::-1]]
bars = ax.barh(
    PEAK_ORDER[::-1], peak_totals.reindex(PEAK_ORDER[::-1]).values,
    color=colors, height=0.6, edgecolor=BG,
)
for bar, val in zip(bars, peak_totals.reindex(PEAK_ORDER[::-1]).values):
    ax.text(
        val + peak_totals.max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:,}", va="center", ha="left", fontsize=9, color=FG,
    )
ax.set_title("Rail journeys by time period")
ax.set_xlabel("Journeys")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xlim(0, peak_totals.max() * 1.15)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "peak_vs_offpeak.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: peak_vs_offpeak.png")

# ---------------------------------------------------------------------------
# Candidate cleaning rules
# ---------------------------------------------------------------------------

unstarted  = (df["StartStn"] == "Unstarted").sum()
unfinished = (df["EndStation"] == "Unfinished").sum()
no_exit    = (df["ExTime"] == 0) & (df["mode"] != "Bus")

# Overnight / negative duration for rail
rail_with_entry = rail[rail["EntTime"] > 0].copy()
rail_with_entry["duration_min"] = rail_with_entry["ExTime"] - rail_with_entry["EntTime"]
negative_dur = (rail_with_entry["duration_min"] < 0).sum()

quality = pd.DataFrame({
    "check": [
        "Unstarted (no tap-in) — no origin for OD matrix",
        "Unfinished (no tap-out) — no destination for OD matrix",
        "Non-bus rows with ExTime == 0 — unusable exit time",
        "Rail rows with negative duration — likely midnight crossings",
    ],
    "count": [unstarted, unfinished, int(no_exit.sum()), negative_dur],
})
quality["share"] = quality["count"] / len(df)

print("\n--- Quality checks ---")
print(quality.to_string(index=False))

print(f"""
--- Summary ---
Total rows          : {len(df):,}
Bus journeys        : {(df['mode'] == 'Bus').sum():,}
Underground journeys: {(df['mode'] == 'Underground').sum():,}
Unstarted entries   : {unstarted:,}
Rail with valid OD  : {len(station_df):,}
""")

# ---------------------------------------------------------------------------
# Initial findings
# ---------------------------------------------------------------------------
#
# - Bus dominates by volume (~1.77M of 2.62M rows); no entry/exit time recorded.
# - Underground is the primary mode for OD and temporal analysis.
# - Clear AM peak (7–9h) and PM peak (17–19h) in rail exit-hour distribution.
# - Weekday demand is roughly uniform Mon–Fri; Sat/Sun noticeably lower.
# - 45,994 Unstarted entries (no tap-in) — exclude from OD matrix.
# - 35,138 Unfinished entries (no tap-out) — exclude from OD matrix.
# - Negative duration rows are minimal (~12); midnight crossings are rare.
# - Next: build OD matrix, add journey duration, AM/PM directional flow analysis.
