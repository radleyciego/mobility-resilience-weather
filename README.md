# Mobility Resilience Under Extreme Weather

Analyzing how extreme weather affects rideshare demand and equity across New York City neighborhoods using NYC TLC trip data (Uber/Lyft) and NOAA weather observations.

## Findings

| Analysis | Result |
|----------|--------|
| **Causal Impact (DiD)** | +97 trips/zone/day on extreme precipitation (p < 0.0001, R² = 0.89) |
| **Equity** | No significant equity gap in recovery (Mann-Whitney p = 0.38) |
| **ML Benchmark** | XGBoost R² = 0.963, LightGBM R² = 0.964 (time-series CV) |
| **Top Predictors** | Lagged demand (7d rolling avg), calendar features, precipitation |

## Project Structure

```
src/
├── data/            # TLC download, NOAA weather, merge & labeling
├── features/        # Lag/rolling/calendar feature engineering
├── models/
│   ├── causal_did.py   # Difference-in-Differences with fixed effects
│   ├── equity.py       # Income quintile stratification + Mann-Whitney U
│   └── predict.py      # XGBoost/LightGBM + time-series CV + SHAP
└── visualization/   # Matplotlib/seaborn plots
```

## Quick Start

```bash
pip install -e .
# Set NOAA_CDO_TOKEN in .env (free at ncdc.noaa.gov/cdo-web/token)
make data
make features
make models
make reports
```

## Data Sources

- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — HVFHV (Uber/Lyft) trips, parquet
- [NOAA NCEI Climate Data Online](https://www.ncei.noaa.gov/cdo-web/) — Daily weather observations
- [NYC Taxi Zones](https://data.cityofnewyork.us/City-Government/NYC-Taxi-Zones/d3c5-ddwc)
- [ACS 5-Year Estimates](https://www.census.gov/programs-surveys/acs)
