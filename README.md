# Public Transit Demand Intelligence from TfL Journey Data

This project analyzes public-transport journeys to identify demand patterns, origin-destination flows, peak periods, and high-demand stations or corridors.

**Mobility question:** How can origin-destination patterns and peak demand reveal capacity stress in a multimodal public-transport system?

## Use case

Potential users:

- public-transport agencies,
- service planners,
- transport consultants,
- city mobility teams.

Decisions supported:

- peak-load planning,
- service frequency adjustment,
- corridor prioritization,
- station demand monitoring,
- multimodal demand analysis.

## Why ML?

Good ML fit for:

- high-demand station prediction,
- OD pair demand classification,
- station clustering,
- demand segmentation.

Classical/non-ML parts:

- OD aggregation,
- mode counts,
- simple peak/off-peak summaries,
- pandas groupby analysis.

## ML / analytics tasks

| Task | Method |
|---|---|
| OD demand analysis | aggregation and matrix analysis |
| Station demand segmentation | clustering |
| High-demand prediction | classification |
| Demand count prediction | regression |
| Mode pattern analysis | segmentation / classification |

---

## 01 — Problem Framing

See mobility question and use case above.

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
- journeys by weekday — bus + rail stacked,
- station imbalance scatter — departures vs arrivals per station,
- station imbalance bars — top 20 stations grouped.

## 03 — Feature Engineering

**OD analysis:**

Origin-destination matrix:

| Origin | Destination | Trips | Mode | Period |
|---|---|---:|---|---|

Analyses:

- top OD pairs,
- AM peak OD flows,
- PM peak OD flows,
- directional imbalance,
- mode-specific OD flows,
- corridor concentration.

Visuals:

- OD heatmap,
- top 20 OD bar chart,
- Sankey diagram by mode.

**Station features:**

| Feature | Meaning |
|---|---|
| total_departures | origin demand |
| total_arrivals | destination demand |
| am_peak_departures | morning origin intensity |
| pm_peak_arrivals | evening destination intensity |
| mode_share_bus | bus share |
| mode_share_tube | Tube share |
| transfer_proxy | high mixed-mode demand |
| peak_ratio | peak demand / total demand |

## 04 — Baseline Model

- demand threshold — classify stations above the 80th percentile of total journeys as high-demand,
- majority class — predict the most frequent class for all stations,
- establishes minimum RMSE / F1 that any model must beat.

## 05 — Model

**Segmentation:**

Station clusters:

| Cluster | Interpretation |
|---|---|
| commuter origins | high AM departures |
| work destinations | high AM arrivals / PM departures |
| transfer hubs | high all-day demand |
| local stops | low/moderate demand |
| mixed-use areas | balanced demand |

Models: k-means, hierarchical clustering.

**Prediction:**

> Predict whether a station or OD pair is high-demand during rush hour.

| Target | Type |
|---|---|
| high_demand_station | classification |
| high_demand_od_pair | classification |
| journey_count | regression |

Features:

- hour, period, mode,
- origin frequency, destination frequency,
- historical OD count,
- station cluster,
- weekday/weekend.

Models: logistic regression, random forest, gradient boosting.

## 06 — Evaluation

- precision, recall, F1, confusion matrix,
- RMSE on journey count regression,
- cluster quality — silhouette score, inertia,
- segment-level error analysis (by mode, period, station type).

## 07 — Decision Layer

1. **Peak corridor map** — identifies strongest morning/evening OD flows
2. **Station archetype dashboard** — classifies stations by demand pattern
3. **High-demand warning** — flags stations or OD pairs likely to experience peak demand
4. **Mode comparison** — shows how bus, Tube, DLR, Overground patterns differ
5. **Planning notes** — identifies where frequency, capacity, or station management could be studied further

---

## Data

TfL Oyster card journey data — see [data/README.md](data/README.md).

Kaggle: <https://www.kaggle.com/datasets/astronasko/transport-for-london-journey-information>  
Source: <https://data.london.gov.uk/dataset/oyster-card-journey-information/>

## Stack

Python · pandas · scikit-learn · LightGBM · SHAP · Matplotlib

## Case study

> This project analyzes TfL journey data to understand public-transport demand patterns across modes, stations, and origin-destination flows. I built OD matrices, segmented stations into demand archetypes, and developed a high-demand classification workflow. The final dashboard translates raw journey records into planning-oriented insights about peak corridors, commuter flows, and station demand profiles.
