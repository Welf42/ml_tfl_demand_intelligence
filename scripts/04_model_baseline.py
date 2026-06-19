#!/usr/bin/env python3
"""
Baseline model for station demand prediction.

Predicts total station demand (total_demand) from structural station features.
Uses structural features only — ratios and mode mix — not raw counts that
directly compose total_demand, to avoid a trivially circular regression.

Features:
  peak_ratio         — fraction of demand in AM/PM peaks
  imbalance_ratio    — net origin vs destination character
  mode_diversity     — number of distinct modes (interchange proxy)
  mode_share_*       — departure share per mode

Baselines:
  1. Mean predictor  — always predict mean demand (no features)
  2. OLS regression  — linear fit using structural features

Metrics: RMSE, R²

Input : data/processed/station_features.csv
Output: data/processed/baseline_metrics.csv
        figures/04_baseline/

Run from the project root:
    python scripts/04_model_baseline.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from styles import BG, FG, FAINT, CYAN, AMBER, GREEN, RED, apply_theme

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parents[1]
FEAT_PATH   = ROOT / "data" / "processed" / "station_features.csv"
METR_PATH   = ROOT / "data" / "processed" / "baseline_metrics.csv"
FIGURES_DIR = ROOT / "figures" / "04_baseline"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

apply_theme()

# ---------------------------------------------------------------------------
# Load and prepare
# ---------------------------------------------------------------------------

stations = pd.read_csv(FEAT_PATH, index_col="station")
print(f"Loaded {len(stations)} stations")

FEATURE_COLS = [
    "peak_ratio",
    "imbalance_ratio",
    "mode_diversity",
    "mode_share_underground",
    "mode_share_national_rail",
    "mode_share_overground",
    "mode_share_dlr",
]
TARGET = "total_demand"

# Keep only rows with complete features
df = stations[FEATURE_COLS + [TARGET]].dropna()
print(f"Complete rows: {len(df)}")

X = df[FEATURE_COLS].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# Baseline 1 — mean predictor
# ---------------------------------------------------------------------------

y_mean = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
rmse_mean = np.sqrt(mean_squared_error(y_test, y_mean))
r2_mean   = r2_score(y_test, y_mean)

print(f"\n--- Mean predictor ---")
print(f"RMSE : {rmse_mean:,.0f}")
print(f"R²   : {r2_mean:.3f}")

# ---------------------------------------------------------------------------
# Baseline 2 — OLS regression
# ---------------------------------------------------------------------------

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

ols = LinearRegression()
ols.fit(X_train_s, y_train)
y_pred = ols.predict(X_test_s)

rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred))
r2_ols   = r2_score(y_test, y_pred)

print(f"\n--- OLS regression ---")
print(f"RMSE : {rmse_ols:,.0f}")
print(f"R²   : {r2_ols:.3f}")
print(f"\nCoefficients:")
for feat, coef in zip(FEATURE_COLS, ols.coef_):
    print(f"  {feat:<30} {coef:+,.1f}")

results = pd.DataFrame([
    {"baseline": "Mean predictor",  "rmse": rmse_mean, "r2": r2_mean},
    {"baseline": "OLS regression",  "rmse": rmse_ols,  "r2": r2_ols},
])
results.to_csv(METR_PATH, index=False)
print(f"\nSaved: {METR_PATH.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# 1. Actual vs predicted scatter
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred, color=CYAN, s=40, alpha=0.7,
           edgecolors=BG, linewidths=0.4, zorder=3)

lim = max(y_test.max(), y_pred.max()) * 1.05
ax.plot([0, lim], [0, lim], color=FAINT, linewidth=1,
        linestyle="--", zorder=2, label="Perfect prediction")

ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_title(f"OLS — actual vs predicted demand\nRMSE {rmse_ols:,.0f}  ·  R² {r2_ols:.2f}")
ax.set_xlabel("Actual total demand")
ax.set_ylabel("Predicted total demand")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_baseline_actual_vs_predicted.png",
            dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_baseline_actual_vs_predicted.png")

# ---------------------------------------------------------------------------
# 2. OLS coefficients
# ---------------------------------------------------------------------------

coef_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "coef":    ols.coef_,
}).sort_values("coef")

colors = [GREEN if c > 0 else RED for c in coef_df["coef"]]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(coef_df["feature"], coef_df["coef"],
               color=colors, edgecolor=BG, height=0.6)
for bar, val in zip(bars, coef_df["coef"]):
    x_pos = val + (coef_df["coef"].abs().max() * 0.02) * np.sign(val)
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+,.0f}", va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8, color=FG)
ax.axvline(0, color=FAINT, linewidth=1)
ax.set_title("OLS coefficients — effect on total demand\n(standardised features · green = increases demand · red = decreases)")
ax.set_xlabel("Coefficient (journeys per standard deviation)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_baseline_coefficients.png",
            dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_baseline_coefficients.png")

# ---------------------------------------------------------------------------
# 3. RMSE comparison
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(
    ["Mean predictor", "OLS regression"],
    [rmse_mean, rmse_ols],
    color=[FAINT, AMBER], edgecolor=BG, height=0.45,
)
for bar, val in zip(bars, [rmse_mean, rmse_ols]):
    ax.text(val + rmse_mean * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}", va="center", ha="left", fontsize=9, color=FG)
ax.set_xlabel("RMSE (journeys)")
ax.set_title("Baseline RMSE — step-05 models must beat amber")
ax.set_xlim(0, rmse_mean * 1.25)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(FIGURES_DIR / "journey_baseline_rmse.png",
            dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: journey_baseline_rmse.png")
