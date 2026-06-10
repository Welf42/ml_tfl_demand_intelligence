#!/usr/bin/env python3
"""
OD matrix analysis for TfL rail journeys.

Builds an origin-destination matrix from cleaned rail data and analyses
demand patterns across periods, modes, and corridors.

Analyses:
  - top OD pairs overall
  - AM peak vs PM peak top pairs
  - directional imbalance (A→B vs B→A)
  - OD heatmap — top 20 origins × top 20 destinations

Input : data/processed/rail_clean.csv
Output: data/processed/od_matrix.csv
        figures/03_od_analysis/

Run from the project root:
    python scripts/03_feature_od_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from styles import BG, FG, FAINT, GRID, CYAN, AMBER, GREEN, RED, apply_theme

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parents[1]
RAIL_PATH   = ROOT / "data" / "processed" / "rail_clean.csv"
OD_PATH     = ROOT / "data" / "processed" / "od_matrix.csv"
FIGURES_DIR = ROOT / "figures" / "03_od_analysis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

apply_theme()

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

rail = pd.read_csv(RAIL_PATH)
print(f"Loaded {len(rail):,} rail journeys")

# ---------------------------------------------------------------------------
# Build OD matrix
# ---------------------------------------------------------------------------

od = (
    rail.groupby(["StartStn", "EndStation", "mode", "period"])
    .size()
    .reset_index(name="count")
)

print(f"\nOD matrix: {len(od):,} unique origin-destination-mode-period combinations")
print(f"Unique origins     : {od['StartStn'].nunique():,}")
print(f"Unique destinations: {od['EndStation'].nunique():,}")

od.to_csv(OD_PATH, index=False)
print(f"Saved: {OD_PATH.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# 1. Top OD pairs overall
# ---------------------------------------------------------------------------

top_pairs = (
    od.groupby(["StartStn", "EndStation"])["count"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)
top_pairs["od_pair"] = top_pairs["StartStn"] + " → " + top_pairs["EndStation"]

print("\n--- Top 10 OD pairs ---")
print(top_pairs.head(10)[["od_pair", "count"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(
    top_pairs["od_pair"][::-1], top_pairs["count"][::-1],
    color=CYAN, height=0.7, edgecolor=BG,
)
for bar, val in zip(bars, top_pairs["count"][::-1]):
    ax.text(
        val + top_pairs["count"].max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:,}", va="center", ha="left", fontsize=8, color=FG,
    )
ax.set_title("Top 20 OD pairs — rail / tube")
ax.set_xlabel("Journeys")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xlim(0, top_pairs["count"].max() * 1.18)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_od_top_pairs.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_od_top_pairs.png")

# ---------------------------------------------------------------------------
# 2. AM vs PM peak top pairs
# ---------------------------------------------------------------------------

def top_period_pairs(period, n=15):
    return (
        od[od["period"] == period]
        .groupby(["StartStn", "EndStation"])["count"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
        .assign(od_pair=lambda d: d["StartStn"] + " → " + d["EndStation"])
    )

am = top_period_pairs("AM peak")
pm = top_period_pairs("PM peak")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, df_p, title, color in [
    (axes[0], am, "Top 15 OD pairs — AM peak", CYAN),
    (axes[1], pm, "Top 15 OD pairs — PM peak", AMBER),
]:
    bars = ax.barh(df_p["od_pair"][::-1], df_p["count"][::-1],
                   color=color, height=0.7, edgecolor=BG)
    for bar, val in zip(bars, df_p["count"][::-1]):
        ax.text(
            val + df_p["count"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", ha="left", fontsize=8, color=FG,
        )
    ax.set_title(title)
    ax.set_xlabel("Journeys")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(0, df_p["count"].max() * 1.2)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_od_am_pm.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_od_am_pm.png")

# ---------------------------------------------------------------------------
# 3. Directional imbalance
# ---------------------------------------------------------------------------
# For each corridor A→B, compare with B→A.
# imbalance_ratio = max(A→B, B→A) / min(A→B, B→A)
# A ratio >> 1 means the corridor is heavily one-directional.

pair_totals = (
    od.groupby(["StartStn", "EndStation"])["count"]
    .sum()
    .reset_index()
)

pair_totals["reverse_key"] = pair_totals["EndStation"] + "|" + pair_totals["StartStn"]
pair_totals["forward_key"] = pair_totals["StartStn"]  + "|" + pair_totals["EndStation"]

merged = pair_totals.merge(
    pair_totals[["forward_key", "count"]].rename(
        columns={"forward_key": "reverse_key", "count": "reverse_count"}
    ),
    on="reverse_key", how="left"
).fillna({"reverse_count": 0})

merged["reverse_count"] = merged["reverse_count"].astype(int)
merged["total"]     = merged["count"] + merged["reverse_count"]
merged["imbalance"] = merged[["count", "reverse_count"]].max(axis=1) / \
                      merged[["count", "reverse_count"]].min(axis=1).replace(0, np.nan)

# Keep only one direction per corridor (the dominant one)
merged["corridor_key"] = merged.apply(
    lambda r: "|".join(sorted([r["StartStn"], r["EndStation"]])), axis=1
)
corridors = (
    merged.sort_values("total", ascending=False)
    .drop_duplicates("corridor_key")
    .dropna(subset=["imbalance"])
    .query("total >= 100")          # exclude low-volume corridors
    .nlargest(20, "imbalance")      # rank by imbalance, not volume
    .sort_values("imbalance")
)
corridors["label"] = corridors["StartStn"] + " ↔ " + corridors["EndStation"]

print("\n--- Top corridors by imbalance ratio ---")
print(corridors[["label", "count", "reverse_count", "imbalance"]]
      .rename(columns={"count": "dominant", "reverse_count": "reverse"})
      .to_string(index=False))

# Continuous color gradient CYAN → RED scaled to imbalance range
import matplotlib.colors as mcolors
cmap_dir = mcolors.LinearSegmentedColormap.from_list("dir", [CYAN, AMBER, RED])
norm = plt.Normalize(corridors["imbalance"].min(), corridors["imbalance"].max())
colors = [cmap_dir(norm(v)) for v in corridors["imbalance"]]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(corridors["label"], corridors["imbalance"],
               color=colors, height=0.7, edgecolor=BG)
for bar, row in zip(bars, corridors.itertuples()):
    ax.text(
        row.imbalance + 0.05, bar.get_y() + bar.get_height() / 2,
        f"{row.imbalance:.1f}×  ({row.count} / {row.reverse_count})",
        va="center", ha="left", fontsize=7.5, color=FG,
    )
ax.axvline(1, color=FAINT, linewidth=1, linestyle="--")
ax.set_title("Directional imbalance — top 20 corridors by ratio\n(min. 100 total journeys · dominant / reverse counts shown)")
ax.set_xlabel("Imbalance ratio  (dominant direction / reverse)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_od_directional.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_od_directional.png")

# ---------------------------------------------------------------------------
# 4. OD heatmap — top 20 origins × top 20 destinations
# ---------------------------------------------------------------------------

top20_origins = (
    od.groupby("StartStn")["count"].sum()
    .sort_values(ascending=False).head(20).index.tolist()
)
top20_dests = (
    od.groupby("EndStation")["count"].sum()
    .sort_values(ascending=False).head(20).index.tolist()
)

heatmap_data = (
    od[od["StartStn"].isin(top20_origins) & od["EndStation"].isin(top20_dests)]
    .groupby(["StartStn", "EndStation"])["count"]
    .sum()
    .unstack(fill_value=0)
    .reindex(index=top20_origins, columns=top20_dests, fill_value=0)
)

dark_cyan = mcolors.LinearSegmentedColormap.from_list("dark_cyan", [BG, CYAN])

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    heatmap_data,
    ax=ax,
    cmap=dark_cyan,
    linewidths=0.3,
    linecolor=BG,
    annot=False,
    fmt=",",
    cbar_kws={"label": "Journeys"},
)
ax.set_title("OD heatmap — top 20 origins × top 20 destinations")
ax.set_xlabel("Destination")
ax.set_ylabel("Origin")
ax.tick_params(axis="x", rotation=45, labelsize=7)
ax.tick_params(axis="y", rotation=0,  labelsize=7)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_od_heatmap.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_od_heatmap.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"""
--- OD analysis summary ---
Total OD combinations : {len(od):,}
Unique origins        : {od['StartStn'].nunique():,}
Unique destinations   : {od['EndStation'].nunique():,}
Top OD pair           : {top_pairs.iloc[0]['od_pair']}  ({top_pairs.iloc[0]['count']:,} journeys)
AM peak journeys      : {od[od['period'] == 'AM peak']['count'].sum():,}
PM peak journeys      : {od[od['period'] == 'PM peak']['count'].sum():,}
""")
