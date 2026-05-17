import os
import time
import json
import subprocess
import polars as pl
from dotenv import load_dotenv
from .config import DATA_RAW, NOAA_TOKEN_ENV

load_dotenv()

STATION_ID = "GHCND:USW00094728"
DATASET_ID = "GHCND"
BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

DAILY_DTYPES = [
    "PRCP",
    "TMAX",
    "TMIN",
    "AWND",
    "SNOW",
    "SNWD",
]


def _curl_get(url: str, token: str, timeout: int = 30) -> dict:
    cmd = [
        "curl", "-s", "--max-time", str(timeout), url,
        "-H", f"token: {token}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    result.check_returncode()
    return json.loads(result.stdout)


def _fetch_datatype(year: int, datatype: str, token: str) -> list[dict]:
    all_records = []
    offset = 1
    limit = 1000

    while True:
        url = (
            f"{BASE_URL}?datasetid={DATASET_ID}"
            f"&stationid={STATION_ID}"
            f"&startdate={year}-01-01&enddate={year}-12-31"
            f"&datatypeid={datatype}"
            f"&limit={limit}&offset={offset}"
            f"&units=metric"
        )
        data = _curl_get(url, token)
        results = data.get("results", [])
        all_records.extend(results)

        count = data.get("metadata", {}).get("resultset", {}).get("count", 0)
        if offset + limit > count:
            break
        offset += limit
        time.sleep(0.2)

    return all_records


def _tall_to_wide(records: list[dict]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame()

    df = pl.DataFrame(records)
    df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%dT%H:%M:%S"))
    df = df.filter(pl.col("datatype").is_in(DAILY_DTYPES))
    df = df.select(["date", "datatype", "value"])
    df = df.with_columns(pl.col("value").cast(pl.Float64, strict=False))

    pivot = df.pivot(
        index="date",
        on="datatype",
        values="value",
        aggregate_function="first",
    )

    for dtype in DAILY_DTYPES:
        if dtype not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(None).alias(dtype))

    return pivot.sort("date")


def download_years(years: list[int]) -> pl.DataFrame:
    token = os.getenv(NOAA_TOKEN_ENV, "")
    all_pivots = []
    for year in years:
        all_records = []
        for dtype in DAILY_DTYPES:
            recs = _fetch_datatype(year, dtype, token)
            all_records.extend(recs)
        pivot = _tall_to_wide(all_records)
        all_pivots.append(pivot)
    return pl.concat(all_pivots) if all_pivots else pl.DataFrame()


def load_weather(year: int = 2024) -> pl.DataFrame:
    path = DATA_RAW / f"weather_{year}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    df = download_years([year])
    df.write_parquet(path)
    return df
