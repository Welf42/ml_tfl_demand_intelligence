#!/usr/bin/env python3
"""
Data exploration for TfL Oyster card journey data (November 2009 week).

Goal: understand the raw dataset and surface the issues that justify cleaning.

Questions answered:
- What does one row represent and what columns exist?
- Which modes are in the data and how are they distributed?
- What week does the data cover?
- How are entry and exit times recorded — and where do they break down?
- Which station fields contain non-station values (Bus, Unstarted, Unfinished)?
- How many records are affected by each quality issue?

Run from the project root:
    python scripts/02_data_exploration.py
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
RAW_PATH = ROOT / "data" / "Nov09JnyExport.csv"
FIGURES_DIR = ROOT / "figures" / "02_exploration_raw"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df = pd.read_csv(RAW_PATH)

print(f"Shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes.to_string()}")
print(f"\nMissing values:\n{df.isna().sum().to_string()}")
print(f"\nSample rows:\n{df.head(3).to_string()}")

# EntTime and ExTime are minutes from midnight (e.g. 633 = 10:33).
# Values above 1440 indicate overnight services continuing past midnight.

apply_theme()

# ---------------------------------------------------------------------------
# 1. Mode breakdown
# ---------------------------------------------------------------------------
# SubSystem can be compound (e.g. 'LUL/NR'). Bus journeys are identified by
# StartStn == 'Bus' — confirmed that all LTB rows have StartStn == 'Bus'.

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

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(
    mode_counts.index[::-1], mode_counts.values[::-1],
    color=CYAN, height=0.6, edgecolor=BG,
)
for bar, val in zip(bars, mode_counts.values[::-1]):
    ax.text(
        val + mode_counts.max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:,}  ({val / len(df):.0%})",
        va="center", ha="left", fontsize=9, color=FG,
    )
ax.set_title("Raw journeys by mode")
ax.set_xlabel("Journeys")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xlim(0, mode_counts.max() * 1.25)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_by_mode_raw.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_by_mode_raw.png")

# ---------------------------------------------------------------------------
# 2. Temporal coverage
# ---------------------------------------------------------------------------

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
day_counts = df.groupby("daytype").size().reindex(DAY_ORDER).rename("journeys")

print("\n--- Journeys by day ---")
print(day_counts.to_string())

fig, ax = plt.subplots(figsize=(8, 4))
colors = [AMBER if d in ("Sat", "Sun") else CYAN for d in DAY_ORDER]
ax.bar(day_counts.index, day_counts.values, color=colors, edgecolor=BG, width=0.7)
ax.set_title("Journeys by day of week — all modes")
ax.set_xlabel("")
ax.set_ylabel("Journeys")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_by_weekday_raw.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_by_weekday_raw.png")

# ---------------------------------------------------------------------------
# 3. Time fields — EntTime and ExTime
# ---------------------------------------------------------------------------
# Bus:  EntTime is populated (boarding time), ExTime == 0 (no tap-out).
# Rail: both EntTime and ExTime are populated for completed journeys.
# Unstarted: EntTime == 0 (no tap-in), ExTime is populated.

bus_mask  = df["mode"] == "Bus"
rail_mask = df["mode"] != "Bus"

print(f"\nEntTime == 0 : {(df['EntTime'] == 0).sum():,} rows")
print(f"ExTime == 0  : {(df['ExTime'] == 0).sum():,} rows")
print(f"  of which bus     : {(bus_mask & (df['ExTime'] == 0)).sum():,}")
print(f"  of which non-bus : {(rail_mask & (df['ExTime'] == 0)).sum():,}")

bus_hours  = df.loc[bus_mask,  "EntTime"] // 60
rail_hours = df.loc[rail_mask & (df["ExTime"] > 0), "ExTime"] % 1440 // 60

bus_hour_counts  = bus_hours.value_counts().sort_index()
rail_hour_counts = rail_hours.value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(bus_hour_counts.index, bus_hour_counts.values,
            color=AMBER, edgecolor=BG, width=0.8)
axes[0].set_title("Bus — boarding hour (EntTime)")
axes[0].set_xlabel("Hour of day")
axes[0].set_ylabel("Journeys")
axes[0].set_xticks(range(0, 24))
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

axes[1].bar(rail_hour_counts.index, rail_hour_counts.values,
            color=CYAN, edgecolor=BG, width=0.8)
axes[1].set_title("Rail / tube — exit hour (ExTime)")
axes[1].set_xlabel("Hour of day")
axes[1].set_ylabel("Journeys")
axes[1].set_xticks(range(0, 24))
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

fig.suptitle("Time fields: how each mode records journey time", y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_by_time_raw.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_by_time_raw.png")

# ---------------------------------------------------------------------------
# 4. Station fields — raw (including Bus, Unstarted, Unfinished)
# ---------------------------------------------------------------------------

top_start = df["StartStn"].value_counts().head(20)
top_end   = df["EndStation"].value_counts().head(20)

print("\n--- Top 10 StartStn (raw) ---")
print(top_start.head(10).to_string())
print("\n--- Top 10 EndStation (raw) ---")
print(top_end.head(10).to_string())

PROBLEM_COLOR = RED

def raw_station_chart(series, title, filename, problem_values):
    colors = [PROBLEM_COLOR if v in problem_values else CYAN for v in series.index[::-1]]
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(
        series.index[::-1], series.values[::-1],
        color=colors, height=0.7, edgecolor=BG,
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

raw_station_chart(
    top_start, "Top 20 origin values — raw  (red = non-station)",
    "journey_stations_origins_raw.png", {"Bus", "Unstarted"},
)
raw_station_chart(
    top_end, "Top 20 destination values — raw  (red = non-station)",
    "journey_stations_destinations_raw.png", {"Unfinished"},
)

# ---------------------------------------------------------------------------
# 5. Data quality summary
# ---------------------------------------------------------------------------

issues = {
    "Bus (no station OD)":         (df["StartStn"] == "Bus").sum(),
    "Unstarted (no tap-in)":       (df["StartStn"] == "Unstarted").sum(),
    "Unfinished (no tap-out)":     (df["EndStation"] == "Unfinished").sum(),
    "Not Applicable (no destination)": (df["EndStation"] == "Not Applicable").sum(),
}

quality = pd.DataFrame({
    "issue":  list(issues.keys()),
    "count":  list(issues.values()),
})
quality["share"] = quality["count"] / len(df)

print("\n--- Data quality issues ---")
print(quality.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
colors = [AMBER, RED, RED]
bars = ax.barh(
    quality["issue"][::-1], quality["count"][::-1],
    color=colors, height=0.55, edgecolor=BG,
)
for bar, row in zip(bars, quality.iloc[::-1].itertuples()):
    ax.text(
        row.count + quality["count"].max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{row.count:,}  ({row.share:.1%})",
        va="center", ha="left", fontsize=9, color=FG,
    )
ax.set_title("Data quality issues — records affected")
ax.set_xlabel("Journeys")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xlim(0, quality["count"].max() * 1.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_data_quality_raw.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_data_quality_raw.png")

print(f"""
--- Raw dataset summary ---
Total rows        : {len(df):,}
Modes             : {df['mode'].nunique()}
Days covered      : {df['daytype'].nunique()} (Mon–Sun)
Quality issues    : {quality['count'].sum():,} rows ({quality['count'].sum() / len(df):.1%})
  → Bus kept, treated as demand signal without OD
  → Unstarted and Unfinished removed in 02_data_clean.py
""")
