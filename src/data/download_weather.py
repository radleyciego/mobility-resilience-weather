import os
import requests
import polars as pl
from datetime import date
from dotenv import load_dotenv
from .config import DATA_RAW, NOAA_TOKEN_ENV

load_dotenv()

STATION_ID = "GHCND:USW00094728"  # Central Park, NY
DATASET_ID = "GHCND"

DAILY_DTYPES = [
    "PRCP",  # precipitation (mm)
    "TMAX",  # max temperature (tenths °C)
    "TMIN",  # min temperature (tenths °C)
    "AWND",  # avg wind speed (tenths m/s)
    "SNOW",  # snowfall (mm)
    "SNWD",  # snow depth (mm)
]


def _fetch_year(year: int) -> list[dict]:
    token = os.getenv(NOAA_TOKEN_ENV)
    url = "https://www.ncei.noaa.gov/access/services/data/v1"
    params = {
        "dataset": DATASET_ID,
        "stations": STATION_ID,
        "startDate": f"{year}-01-01",
        "endDate": f"{year}-12-31",
        "dataTypes": ",".join(DAILY_DTYPES),
        "format": "json",
        "units": "metric",
    }
    headers = {"token": token} if token else {}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_years(years: list[int]) -> pl.DataFrame:
    rows = []
    for year in years:
        rows.extend(_fetch_year(year))
    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    for col in DAILY_DTYPES:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
    return df


def load_weather(year: int = 2024) -> pl.DataFrame:
    path = DATA_RAW / f"weather_{year}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    df = download_years([year])
    df.write_parquet(path)
    return df
