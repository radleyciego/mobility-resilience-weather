# Mobility Resilience Under Extreme Weather

Analyzing how extreme weather affects rideshare demand, supply, and equity across New York City neighborhoods using NYC TLC trip data and NOAA weather observations.

## Modules

1. **Weather Shock Detection** — Merge HVFHV trip records with daily weather, label extreme events
2. **Causal Impact Analysis** — Difference-in-Differences estimating demand shocks from weather
3. **Equity Stratification** — Differential impacts by neighborhood income quintile
4. **Predictive Modeling** — XGBoost/LightGBM forecasting demand under weather scenarios
5. **Reproducible Pipeline** — Docker, Makefile, automated quality checks

## Quick Start

```bash
pip install -e .
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
