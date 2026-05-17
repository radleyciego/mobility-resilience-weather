import polars as pl
from .config import DATA_PROCESSED
from .download_tlc import load_raw
from .download_weather import load_weather


WEATHER_COLS = [
    "date",
    "PRCP",
    "TMAX",
    "TMIN",
    "AWND",
    "SNOW",
    "SNWD",
]

ZONE_COLS = [
    "PULocationID",
    "DOLocationID",
    "pickup_datetime",
    "dropoff_datetime",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "driver_pay",
]


def _extract_date(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(pl.col("pickup_datetime").cast(pl.Date).alias("date"))


def _flag_weather_shocks(
    weather: pl.DataFrame,
    precip_pct: float = 0.95,
) -> pl.DataFrame:
    non_null_precip = weather.filter(pl.col("PRCP").is_not_null() & (pl.col("PRCP") > 0))
    threshold = (
        non_null_precip.select(pl.col("PRCP").quantile(precip_pct)).item()
        if non_null_precip.height > 0
        else 0.0
    )
    return weather.with_columns(
        pl.col("PRCP")
        .is_not_null()
        .and_(pl.col("PRCP") > 0)
        .alias("any_precip"),
        (pl.col("PRCP") > threshold).alias("extreme_precip"),
        (pl.col("TMAX") > 350).alias("heat_day"),
        (pl.col("TMIN") < 0).alias("freeze_day"),
        (pl.col("SNOW") > 25.4).alias("snow_day"),
    )


def build_labeled_trips(
    year: int = 2024,
    months: list[int] | None = None,
    sample_frac: float | None = None,
) -> pl.LazyFrame:
    tlc = load_raw()
    if months:
        tlc = tlc.with_columns(pl.col("pickup_datetime").dt.month().alias("_month"))
        tlc = tlc.filter(pl.col("_month").is_in(months)).drop("_month")
    tlc = tlc.select(ZONE_COLS).pipe(_extract_date)

    weather = load_weather(year)
    weather = _flag_weather_shocks(weather).select(WEATHER_COLS + [
        "any_precip", "extreme_precip", "heat_day", "freeze_day", "snow_day",
    ])

    merged = tlc.join(weather.lazy(), on="date", how="inner")

    if sample_frac:
        merged = merged.sample(fraction=sample_frac, seed=42)

    return merged


def save_processed(year: int = 2024, sample_frac: float | None = None):
    df = build_labeled_trips(year=year, sample_frac=sample_frac)
    out = DATA_PROCESSED / f"trips_labeled_{year}.parquet"
    df.sink_parquet(str(out))
    return out
