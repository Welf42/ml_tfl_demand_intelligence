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
- SQL/pandas groupby analysis.

## ML / analytics tasks

| Task | Method |
|---|---|
| OD demand analysis | aggregation and matrix analysis |
| Station demand segmentation | clustering |
| High-demand prediction | classification |
| Demand count prediction | regression |
| Mode pattern analysis | segmentation / classification |

## Phase 1 — Data understanding

Tasks:

- inspect journey records,
- understand available modes,
- inspect origin/destination fields,
- inspect temporal fields,
- check missing values,
- standardize station names,
- create peak/off-peak labels.

Questions:

- Which modes dominate?
- Which origins and destinations are busiest?
- How does demand differ by time period?
- Are there directional peak patterns?
- Are there clear commute corridors?

Visuals:

- demand by mode,
- demand by hour,
- top origins,
- top destinations,
- top OD pairs,
- peak vs off-peak comparison.

## Phase 2 — OD matrix

Create origin-destination matrix:

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
- Sankey diagram by mode,
- chord diagram if useful,
- map if coordinates are available.

## Phase 3 — Demand segmentation

Station-level features:

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

Station clusters:

| Cluster | Interpretation |
|---|---|
| commuter origins | high AM departures |
| work destinations | high AM arrivals / PM departures |
| transfer hubs | high all-day demand |
| local stops | low/moderate demand |
| mixed-use areas | balanced demand |

## Phase 4 — Predictive task

> Predict whether a station or OD pair is high-demand during rush hour.

Target options:

| Target | Type |
|---|---|
| high_demand_station | classification |
| high_demand_od_pair | classification |
| journey_count | regression |
| mode_choice | classification |

Features:

- hour,
- period,
- mode,
- origin frequency,
- destination frequency,
- historical OD count,
- peak/off-peak label,
- station cluster,
- weekday/weekend if available.

Models:

- logistic regression,
- decision tree,
- random forest,
- gradient boosting.

Evaluation:

- precision,
- recall,
- F1,
- confusion matrix,
- top-k demand capture.

## Phase 5 — Transport decision layer

1. **Peak corridor map** — identifies strongest morning/evening OD flows
2. **Station archetype dashboard** — classifies stations by demand pattern
3. **High-demand warning** — flags stations or OD pairs likely to experience peak demand
4. **Mode comparison** — shows how bus, Tube, DLR, Overground patterns differ
5. **Planning notes** — identifies where frequency, capacity, or station management could be studied further

## Data

TfL Oyster card journey data — see [data/README.md](data/README.md).

Kaggle: <https://www.kaggle.com/datasets/astronasko/transport-for-london-journey-information>  
Source: <https://data.london.gov.uk/dataset/oyster-card-journey-information/>

## Stack

Python · pandas · scikit-learn · LightGBM · SHAP · Matplotlib

## Case study

> This project analyzes TfL journey data to understand public-transport demand patterns across modes, stations, and origin-destination flows. I built OD matrices, segmented stations into demand archetypes, and developed a high-demand classification workflow. The final dashboard translates raw journey records into planning-oriented insights about peak corridors, commuter flows, and station demand profiles.
