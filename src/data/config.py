from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(DATA_EXTERNAL, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
WEATHER_BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
NOAA_TOKEN_ENV = "NOAA_CDO_TOKEN"
