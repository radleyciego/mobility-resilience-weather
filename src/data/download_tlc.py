import polars as pl
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import TLC_BASE_URL, DATA_RAW

MONTHS = list(range(1, 13))
YEAR = 2024


def _monthly_url(year: int, month: int) -> str:
    return f"{TLC_BASE_URL}/fhvhv_tripdata_{year}-{month:02d}.parquet"


def _local_path(year: int, month: int) -> Path:
    return DATA_RAW / f"fhvhv_{year}-{month:02d}.parquet"


def download_month(year: int, month: int) -> Path:
    url = _monthly_url(year, month)
    path = _local_path(year, month)
    if path.exists():
        return path
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def download_range(years: list[int] | None = None, max_workers: int = 4) -> list[Path]:
    years = years or [YEAR]
    tasks = [(y, m) for y in years for m in MONTHS]
    paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_month, y, m): (y, m) for y, m in tasks}
        for future in as_completed(futures):
            paths.append(future.result())
    return sorted(paths)


def load_raw(pattern: str = "fhvhv_2024-*.parquet") -> pl.LazyFrame:
    return pl.scan_parquet(str(DATA_RAW / pattern))
