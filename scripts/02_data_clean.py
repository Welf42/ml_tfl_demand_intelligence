#!/usr/bin/env python3
"""
Clean the raw TfL journey data and split into bus and rail datasets.

Cleaning rules (grounded in findings from 02_data_exploration_raw.py):

  1. Remove Unstarted      — StartStn == 'Unstarted'. No origin available.
  2. Remove Unfinished     — EndStation == 'Unfinished'. No destination available.
  3. Remove Not Applicable — EndStation == 'Not Applicable'. No destination available.

Bus journeys are kept but split into a separate file. They carry valid demand
information (route, boarding time, day) but have no station OD.

Derived columns added:

  mode         — primary mode (Bus, Underground, DLR, National Rail, …)
  journey_hour — bus: boarding hour (EntTime); rail: exit hour (ExTime)
  period       — AM peak / PM peak / Off-peak / Night

Input : data/Nov09JnyExport.csv
Output: data/processed/bus_clean.csv
        data/processed/rail_clean.csv

Run from the project root:
    python scripts/02_data_clean.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT      = Path(__file__).resolve().parents[1]
RAW_PATH  = ROOT / "data" / "Nov09JnyExport.csv"
BUS_PATH  = ROOT / "data" / "processed" / "bus_clean.csv"
RAIL_PATH = ROOT / "data" / "processed" / "rail_clean.csv"
BUS_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df = pd.read_csv(RAW_PATH)
n_raw = len(df)
print(f"Loaded {n_raw:,} rows")

# ---------------------------------------------------------------------------
# Mode classification (vectorised)
# ---------------------------------------------------------------------------

df["mode"] = "Other"
df.loc[df["SubSystem"].str.contains("TRAM"), "mode"] = "Tram"
df.loc[df["SubSystem"].str.contains("HEX"),  "mode"] = "Heathrow Express"
df.loc[df["SubSystem"].str.contains("LRC"),  "mode"] = "Overground"
df.loc[df["SubSystem"].str.contains("NR"),   "mode"] = "National Rail"
df.loc[df["SubSystem"].str.contains("DLR"),  "mode"] = "DLR"
df.loc[df["SubSystem"].str.contains("LUL"),  "mode"] = "Underground"
df.loc[df["StartStn"] == "Bus",              "mode"] = "Bus"

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

rules = {
    "unstarted":      df["StartStn"] == "Unstarted",
    "unfinished":     df["EndStation"] == "Unfinished",
    "not_applicable": df["EndStation"] == "Not Applicable",
}

print("\n--- Rows removed per rule ---")
for name, mask in rules.items():
    n = mask.sum()
    print(f"  {name:<16}  {n:>8,}  ({n / n_raw:.2%})")

invalid = pd.concat(rules.values(), axis=1).any(axis=1)
df = df[~invalid].copy()

n_removed = n_raw - len(df)
print(f"\nTotal removed  : {n_removed:,} ({n_removed / n_raw:.2%})")
print(f"Rows remaining : {len(df):,} ({len(df) / n_raw:.2%})")

# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

bus_mask = df["mode"] == "Bus"
df["journey_hour"] = np.where(
    bus_mask,
    df["EntTime"] // 60,
    (df["ExTime"] % 1440) // 60,
)

def peak_label(hour):
    if 7 <= hour < 10:
        return "AM peak"
    if 17 <= hour < 20:
        return "PM peak"
    if hour >= 23 or hour < 5:
        return "Night"
    return "Off-peak"

df["period"] = df["journey_hour"].apply(peak_label)

# ---------------------------------------------------------------------------
# Split and save
# ---------------------------------------------------------------------------

bus  = df[df["mode"] == "Bus"].copy()
rail = df[df["mode"] != "Bus"].copy()

print(f"\n--- Split ---")
print(f"Bus  : {len(bus):,} rows  → {BUS_PATH.relative_to(ROOT)}")
print(f"Rail : {len(rail):,} rows  → {RAIL_PATH.relative_to(ROOT)}")
print(f"\nRail modes:\n{rail['mode'].value_counts().to_string()}")

bus.to_csv(BUS_PATH,   index=False)
rail.to_csv(RAIL_PATH, index=False)
print("\nDone.")
