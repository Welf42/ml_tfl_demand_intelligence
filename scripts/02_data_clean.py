#!/usr/bin/env python3
"""
Clean the raw TfL journey data.

Cleaning rules (grounded in findings from 01_data_exploration.py):

  1. Remove Unstarted  — StartStn == 'Unstarted'. Passenger tapped out only;
                         no origin available. Useless for any analysis.

  2. Remove Unfinished — EndStation == 'Unfinished'. Passenger tapped in only;
                         no destination available. Useless for OD analysis.

Bus journeys are kept. They have no station OD but carry valid demand
information: route (RouteID), boarding time (EntTime), day, and ticket type.
Boarding ≈ departure at the network level, so bus demand patterns are valid.

Derived columns added for downstream scripts:

  mode         — primary mode (Bus, Underground, DLR, National Rail, …)
  journey_hour — hour of the journey event (0–23):
                   bus  → boarding hour from EntTime
                   rail → exit hour from ExTime (wraps overnight services)
  period       — AM peak / PM peak / Off-peak / Night

Input : data/Nov09JnyExport.csv
Output: data/processed/journeys_clean.csv

Run from the project root:
    python scripts/02_clean_data.py
"""

from pathlib import Path

import numpy as np
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
    "unstarted":  df["StartStn"] == "Unstarted",
    "unfinished": df["EndStation"] == "Unfinished",
}

print("\n--- Rows removed per rule ---")
for name, mask in rules.items():
    n = mask.sum()
    print(f"  {name:<12}  {n:>8,}  ({n / n_raw:.2%})")

invalid = pd.concat(rules.values(), axis=1).any(axis=1)
df = df[~invalid].copy()

n_removed = n_raw - len(df)
print(f"\nTotal removed  : {n_removed:,} ({n_removed / n_raw:.2%})")
print(f"Rows remaining : {len(df):,} ({len(df) / n_raw:.2%})")

# ---------------------------------------------------------------------------
# journey_hour
# ---------------------------------------------------------------------------
# Bus:  use EntTime (boarding = departure); ExTime is always 0 for bus.
# Rail: use ExTime (tap-out); wrap values > 1440 for overnight services.

bus_mask = df["mode"] == "Bus"
df["journey_hour"] = np.where(
    bus_mask,
    df["EntTime"] // 60,
    (df["ExTime"] % 1440) // 60,
)

# ---------------------------------------------------------------------------
# Period label
# ---------------------------------------------------------------------------

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
# Summary
# ---------------------------------------------------------------------------

print("\n--- Rows by mode ---")
print(df["mode"].value_counts().to_string())

print("\n--- Rows by period ---")
print(df["period"].value_counts().reindex(["AM peak", "Off-peak", "PM peak", "Night"]).to_string())

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(df):,} rows → {OUT_PATH.relative_to(ROOT)}")
print(f"Columns: {list(df.columns)}")
