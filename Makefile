.PHONY: data features models reports test clean all

DATA_YEAR ?= 2024
SAMPLE ?=

data:
	python -c "from src.data.merge import save_processed; save_processed(year=$(DATA_YEAR), sample_frac=$(SAMPLE) if '$(SAMPLE)' else None)"

features:
	python -c "from src.features.build_features import build_feature_set; build_feature_set(year=$(DATA_YEAR))"

models:
	python -c "from src.models.causal_did import estimate_weather_impact; result = estimate_weather_impact(year=$(DATA_YEAR)); print(result['did_estimate'], result['p_value'])"
	python -c "from src.models.predict import cv_benchmark; print(cv_benchmark(year=$(DATA_YEAR)))"
	python -c "from src.models.equity import summarize_equity; print(summarize_equity(year=$(DATA_YEAR)))"

reports:
	python -c "from src.visualization.plot import *; from src.data.config import DATA_PROCESSED; import polars as pl; daily = pl.read_parquet(str(DATA_PROCESSED / f'trips_labeled_$(DATA_YEAR).parquet')).group_by('date').agg(pl.len().alias('trip_count')); plot_trip_volumes(daily)"

test:
	python -m pytest tests/ -v

clean:
	rm -rf data/processed/*.parquet reports/figures/*.png

all: data features models reports
