import polars as pl
import pytest
from pathlib import Path

SAMPLE_PATH = Path("data/processed/sample_test.parquet")


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "PULocationID": [1, 2, 3, 4, 5],
        "DOLocationID": [2, 3, 4, 5, 1],
        "trip_miles": [1.0, 2.5, 0.0, 3.2, 4.1],
        "trip_time": [300, 600, 0, 900, 1200],
        "base_passenger_fare": [5.0, 10.0, 0.0, 15.0, 20.0],
        "date": ["2024-01-01"] * 5,
        "extreme_precip": [False, True, False, False, True],
        "PRCP": [0.0, 50.0, 0.0, 0.0, 60.0],
    })


def test_valid_location_ids(sample_df):
    assert sample_df["PULocationID"].is_between(1, 263).all()
    assert sample_df["DOLocationID"].is_between(1, 263).all()


def test_no_negative_trip_miles(sample_df):
    assert (sample_df["trip_miles"] >= 0).all()


def test_no_negative_fare(sample_df):
    assert (sample_df["base_passenger_fare"] >= 0).all()


def test_precip_non_negative(sample_df):
    assert (sample_df["PRCP"] >= 0).all()


def test_extreme_precip_implies_precip(sample_df):
    extreme = sample_df.filter(pl.col("extreme_precip") == True)
    assert (extreme["PRCP"] > 0).all()


def test_date_column_type(sample_df):
    assert sample_df["date"].dtype == pl.Utf8 or sample_df["date"].dtype == pl.Date


def test_required_columns_present(sample_df):
    required = ["PULocationID", "DOLocationID", "trip_miles", "date"]
    for col in required:
        assert col in sample_df.columns
