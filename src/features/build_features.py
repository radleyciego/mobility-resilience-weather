import polars as pl
from src.data.config import DATA_PROCESSED


def add_lagged_features(df: pl.DataFrame, lag_days: list[int] | None = None) -> pl.DataFrame:
    lags = lag_days or [1, 2, 3, 7]
    df = df.sort(["PULocationID", "date"])
    for lag in lags:
        df = df.with_columns(
            pl.col("trip_count")
            .shift(lag)
            .over("PULocationID")
            .alias(f"trip_count_lag{lag}d")
        )
    return df


def add_rolling_features(df: pl.DataFrame, windows: list[int] | None = None) -> pl.DataFrame:
    windows = windows or [7, 30]
    df = df.sort(["PULocationID", "date"])
    for w in windows:
        df = df.with_columns(
            pl.col("trip_count")
            .rolling_mean(w)
            .over("PULocationID")
            .alias(f"trip_count_roll{w}d")
        )
    return df


def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.weekday().alias("weekday"),
        pl.col("date").dt.day().alias("day"),
        pl.col("date").dt.ordinal_day().alias("day_of_year"),
    )
    return df.with_columns(
        pl.col("weekday").is_in([5, 6]).alias("is_weekend"),
    )


def build_feature_set(year: int = 2024) -> pl.DataFrame:
    path = DATA_PROCESSED / f"trips_labeled_{year}.parquet"

    daily = (
        pl.scan_parquet(path)
        .group_by(["date", "PULocationID", "extreme_precip", "any_precip",
                    "heat_day", "freeze_day", "snow_day",
                    "PRCP", "TMAX", "TMIN", "AWND"])
        .agg(pl.len().alias("trip_count"))
        .collect()
    )

    daily = add_calendar_features(daily)
    daily = add_lagged_features(daily)
    daily = add_rolling_features(daily)

    out = DATA_PROCESSED / f"features_{year}.parquet"
    daily.write_parquet(out)
    return out
