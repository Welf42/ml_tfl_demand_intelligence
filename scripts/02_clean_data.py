#!/usr/bin/env python3
"""
Clean the raw TfL journey data and produce a rail-ready dataset.

Cleaning rules (each grounded in findings from 01_data_exploration.py):

  1. Remove Bus journeys  — StartStn == 'Bus' or SubSystem contains 'LTB'.
                            No entry/exit time recorded; no meaningful OD info.

  2. Remove Unstarted     — StartStn == 'Unstarted'. Passenger tapped out only;
                            no origin station available for OD analysis.

  3. Remove Unfinished    — EndStation == 'Unfinished'. Passenger tapped in only;
                            no destination station available for OD analysis.

  4. Remove zero ExTime   — ExTime == 0 for non-bus rows. Exit time not recorded;
                            unusable for temporal analysis.

Derived columns added for downstream scripts:

  mode        — primary mode from SubSystem (Underground, DLR, National Rail, …)
  exit_hour   — hour of exit (0–23), wrapping overnight services past midnight
  period      — AM peak / PM peak / Off-peak / Night label

Input : data/Nov09JnyExport.csv
Output: data/processed/journeys_clean.csv

Run from the project root:
    python scripts/02_clean_data.py
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "Nov09JnyExport.csv"
OUT_PATH  = ROOT / "data" / "processed" / "journeys_clean.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

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
# Cleaning rules
# ---------------------------------------------------------------------------

rules = {
    "bus_journey":    df["mode"] == "Bus",
    "unstarted":      df["StartStn"] == "Unstarted",
    "unfinished":     df["EndStation"] == "Unfinished",
    "zero_exit_time": (df["ExTime"] == 0) & (df["mode"] != "Bus"),
}

print("\n--- Rows removed per rule ---")
for name, mask in rules.items():
    n = mask.sum()
    print(f"  {name:<20}  {n:>8,}  ({n / n_raw:.2%})")

invalid = pd.concat(rules.values(), axis=1).any(axis=1)
df = df[~invalid].copy()

n_removed = n_raw - len(df)
print(f"\nTotal removed  : {n_removed:,} ({n_removed / n_raw:.2%})")
print(f"Rows remaining : {len(df):,} ({len(df) / n_raw:.2%})")

# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

df["exit_hour"] = (df["ExTime"] % 1440) // 60

def peak_label(hour):
    if 7 <= hour < 10:
        return "AM peak"
    if 17 <= hour < 20:
        return "PM peak"
    if hour >= 23 or hour < 5:
        return "Night"
    return "Off-peak"

df["period"] = df["exit_hour"].apply(peak_label)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(df):,} rows → {OUT_PATH.relative_to(ROOT)}")
print(f"Columns: {list(df.columns)}")
