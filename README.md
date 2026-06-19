# Public Transit Demand Intelligence from TfL Journey Data

Raw journey counts tell you which stations are busy. They do not tell you what kind of station it is, why it is busy, or how it compares structurally to others in the network.

This project builds a station characterisation pipeline on TfL Oyster card data. It engineers structural features that describe each station's role — peak intensity, flow balance, mode mix, interchange character — and uses those features to segment stations into demand archetypes and explain what drives volume differences.

**Core question:** What type of station is this, and what does that tell a planner?

## Two complementary angles

**Descriptive — feature analysis:**
The features themselves are the first output. For any station you can immediately read: how peaked is demand, is it a net origin or destination, is it an interchange, which mode dominates. Together these dimensions characterise the station's role in the network without needing a model.

**Actionable — clustering:**
Combining all features simultaneously via clustering produces a single archetype label per station. Instead of reading six numbers, a planner sees "commuter origin" or "interchange hub". At 444 stations, that is operationally useful. Regression on top of the clusters explains which structural features drive demand differences between types.

## Use case

Potential users: public-transport agencies, service planners, transport consultants, city mobility teams.

Decisions supported: peak-load planning, service frequency adjustment, corridor prioritization, station demand monitoring, infrastructure investment.

## What ML adds

| Step | Method | Output |
|---|---|---|
| OD analysis | aggregation | demand matrix by corridor, period, mode |
| Feature engineering | pandas | structural feature matrix per station |
| Clustering | k-means / hierarchical | station archetypes |
| Demand regression | OLS → random forest | which features drive volume |

Exploration alone (steps 02–03) answers *who* is busy and *when*. Clustering (step 05) answers *what kind* of station it is. Regression answers *why* demand varies across types.

---

## 01 — Problem Framing

**Data:** one week of TfL Oyster card journeys (November 2009, 2.6M rows). Covers bus, Underground, DLR, National Rail, Overground, Tram, and Heathrow Express.

**Unit of analysis:** individual rail stations (444 stations with at least one tap-in or tap-out).

**Target outputs:**
- structural feature matrix — one row per station, describing peak pattern, flow balance, mode mix
- cluster labels — demand archetype per station (commuter origin / destination hub / interchange / local)
- demand regression — predicted total demand from structural features; OLS coefficients explain which features matter

**What the data cannot support:**
- bus station analysis — Oyster bus records have no destination tap-out, so OD and station-level analysis is rail-only
- forecasting future demand — the dataset is a single historical week, not a time series
- causal inference — correlations between features and demand reflect the network as it existed in 2009

## 02 — Data

**Raw exploration (`02_data_raw_plots.py`):**

- inspect journey records and column types,
- understand available modes (Bus, Underground, DLR, National Rail, Overground, Tram),
- inspect origin/destination fields and surface non-station values,
- inspect temporal fields — how EntTime and ExTime differ by mode,
- surface data quality issues that justify cleaning.

Visuals:

- journeys by mode (raw),
- journeys by time — bus boarding hour vs rail exit hour,
- journeys by weekday,
- top origin and destination values including Bus, Unstarted, Unfinished,
- data quality issues — records affected per rule.

**Cleaning (`02_data_clean.py`):**

- remove Unstarted — no tap-in, origin unknown,
- remove Unfinished — no tap-out, destination unknown,
- remove Not Applicable — system could not record destination,
- keep Bus — no station OD but valid demand signal (route, boarding time, day); boarding ≈ departure at network level,
- split into `bus_clean.csv` (1.77M rows) and `rail_clean.csv` (765k rows).

**Clean exploration (`02_data_clean_plots.py`):**

Visuals:

- journeys by time — bus + rail stacked,
- journeys by weekday — bus + rail stacked.

## 03 — Feature Engineering

**OD analysis:**

Origin-destination matrix:

| Origin | Destination | Trips | Mode | Period |
|---|---|---:|---|---|

Analyses:

- top OD pairs overall,
- AM vs PM peak OD flows,
- directional imbalance per corridor,
- demand concentration by station pair.

Visuals:

- top 20 OD pairs bar chart,
- AM vs PM peak top pairs side-by-side,
- directional imbalance ratio per corridor,
- OD heatmap — top 20 origins × top 20 destinations.

**Station features (`03_feature_stations.py`):**

| Feature | Meaning |
|---|---|
| total_departures | journeys starting at this station |
| total_arrivals | journeys ending at this station |
| total_demand | total_departures + total_arrivals |
| am_peak_departures | AM peak (7–9h) origin intensity |
| pm_peak_arrivals | PM peak (17–19h) destination intensity |
| peak_ratio | (am_dep + pm_arr) / total_demand |
| imbalance_ratio | (departures − arrivals) / total_demand |
| dominant_mode | most frequent mode at station |
| mode_diversity | distinct modes present (transfer proxy) |
| mode_share_* | departure fraction per mode |

Visuals:

- AM peak departures vs PM peak arrivals scatter,
- peak ratio distribution across all stations,
- station imbalance scatter — total departures vs arrivals,
- station imbalance bars — top 20 stations grouped.

## 04 — Baseline Model

**What we're predicting:** total station demand (continuous) from structural station features — peak intensity, flow imbalance, mode mix, and interchange character. Raw counts (departures, arrivals) are excluded as features because they directly compose the target.

**Why baselines?** Before training any ML model, simpler models establish a performance floor. If a model can't beat OLS, it's not adding value.

| Baseline | RMSE | R² |
|---|---:|---:|
| Mean predictor | 5,851 | 0.00 |
| OLS regression | 4,964 | 0.28 |

**Mean predictor** — always predicts the mean demand. No features used. RMSE and R² set the absolute floor.

**OLS regression** — ordinary least squares fit on structural features. R² = 0.28 means the linear model explains 28% of demand variance from mode mix and peak patterns alone.

**Key OLS findings:**
- Underground share is the strongest positive driver — Underground stations attract the highest demand
- Mode diversity is positive — interchange stations are busier
- DLR, Overground, and National Rail shares are negative — these tend to be lower-volume networks
- Peak ratio is negative — highly peaked stations tend to be narrower commuter hubs with lower total volume

**RMSE** measures average prediction error in journeys. Step-05 models must beat RMSE = 4,964 and R² = 0.28.

Visuals:

- actual vs predicted scatter — OLS test-set performance with diagonal reference,
- OLS coefficients — which features drive demand up or down,
- RMSE comparison — mean predictor vs OLS.

## 05 — Model

**Clustering — station archetypes:**

k-means and hierarchical clustering on the structural feature matrix. Expected archetypes:

| Archetype | Signal |
|---|---|
| commuter origin | high AM departures, high imbalance ratio, low PM arrivals |
| destination hub | high PM arrivals, negative imbalance ratio |
| interchange | high mode diversity, high all-day demand |
| local stop | low demand, low peak ratio, single mode |

Evaluation: silhouette score, inertia, interpretability of cluster centroids.

**Demand regression — what drives volume:**

Random forest regression predicting `total_demand` from structural features. Compared against OLS baseline from step 04. SHAP values explain feature contributions per station and per cluster.

## 06 — Evaluation

- cluster quality — silhouette score, inertia curve (elbow method)
- cluster interpretability — centroid feature profiles, representative stations per cluster
- regression performance — RMSE and R² vs OLS baseline
- SHAP — which features matter most and in which direction

## 07 — Decision Layer

1. **Station archetype map** — each station labelled with its cluster and key feature values
2. **Cluster profiles** — what makes each archetype distinctive; which stations are boundary cases
3. **Demand drivers** — SHAP summary showing which structural features explain volume differences
4. **OD corridor summary** — highest-demand and most directionally imbalanced corridors
5. **Planning notes** — stations where archetype and volume suggest capacity, frequency, or interchange investment could be studied

---

## Data

TfL Oyster card journey data — see [data/README.md](data/README.md).

Kaggle: <https://www.kaggle.com/datasets/astronasko/transport-for-london-journey-information>  
Source: <https://data.london.gov.uk/dataset/oyster-card-journey-information/>

## Stack

Python · pandas · scikit-learn · LightGBM · SHAP · Matplotlib

## Case study

> Raw journey counts tell you which stations are busy — feature engineering and clustering tell you what kind of station it is. I built a station characterisation pipeline on TfL Oyster data: engineered structural features describing peak intensity, flow balance, and mode mix; segmented 444 stations into demand archetypes using clustering; and used OLS and random forest regression with SHAP to explain what drives volume differences between types. The result is a planning-oriented view of the network — not just a ranked list of busy stations, but an understanding of their roles and the features that distinguish them.
