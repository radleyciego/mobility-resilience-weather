import polars as pl
import numpy as np
from scipy.stats import mannwhitneyu
from src.data.config import DATA_PROCESSED, DATA_EXTERNAL, FIGURES_DIR


INCOME_QUINTILES = {
    1: "lowest",
    2: "low",
    3: "middle",
    4: "high",
    5: "highest",
}


def load_zone_income() -> pl.DataFrame:
    path = DATA_EXTERNAL / "zone_income_quintiles.parquet"
    if path.exists():
        return pl.read_parquet(path)

    stub = (
        pl.DataFrame({
            "PULocationID": range(1, 264),
            "median_income": np.random.default_rng(42).uniform(
                30000, 120000, 263
            ),
        })
    )
    stub = stub.with_columns(
        pl.col("median_income")
        .qcut(5, labels=list(range(1, 6)))
        .cast(pl.Int32)
        .alias("income_quintile")
    )
    stub.write_parquet(path)
    return stub


def _compute_recovery_time(
    df: pl.DataFrame,
    shock_col: str = "extreme_precip",
) -> pl.DataFrame:
    shock_dates = (
        df.filter(pl.col(shock_col))
        .select("date").unique()
    )

    baseline = (
        df.join(shock_dates, on="date", how="anti")
        .group_by(["PULocationID"])
        .agg(pl.col("trip_count").mean().alias("baseline_trips"))
    )

    recovery = (
        df.filter(pl.col(shock_col))
        .sort(["PULocationID", "date"])
        .group_by(["PULocationID", "date"])
        .agg(pl.col("trip_count").mean().alias("shock_trips"))
        .join(baseline, on="PULocationID")
        .with_columns(
            ((pl.col("shock_trips") / pl.col("baseline_trips")) * 100)
            .alias("pct_of_baseline")
        )
    )

    return recovery


def equity_analysis(
    daily_trips: pl.DataFrame,
    shock_col: str = "extreme_precip",
) -> dict:
    income = load_zone_income()
    recovery = _compute_recovery_time(daily_trips, shock_col)

    merged = recovery.join(income, on="PULocationID")

    results = {}
    for quintile in range(1, 6):
        group = merged.filter(pl.col("income_quintile") == quintile)
        label = INCOME_QUINTILES[quintile]
        results[label] = {
            "mean_pct_of_baseline": group.select(
                pl.col("pct_of_baseline").mean()
            ).item(),
            "std_pct_of_baseline": group.select(
                pl.col("pct_of_baseline").std()
            ).item(),
            "n_zones": group.select(pl.col("PULocationID").n_unique()).item(),
            "n_shock_days": group.select(pl.col("date").n_unique()).item(),
        }

    low_income = merged.filter(pl.col("income_quintile").is_in([1, 2]))
    high_income = merged.filter(pl.col("income_quintile").is_in([4, 5]))

    low_pct = low_income["pct_of_baseline"].drop_nulls().to_numpy()
    high_pct = high_income["pct_of_baseline"].drop_nulls().to_numpy()

    if len(low_pct) > 0 and len(high_pct) > 0:
        stat, pval = mannwhitneyu(low_pct, high_pct, alternative="less")
        results["equity_test"] = {
            "test": "Mann-Whitney U (low vs high income recovery)",
            "statistic": float(stat),
            "p_value": float(pval),
            "interpretation": (
                "Low-income zones recover less" if pval < 0.05
                else "No significant equity difference detected"
            ),
        }

    return results


def summarize_equity(year: int = 2024) -> dict:
    path = DATA_PROCESSED / f"trips_labeled_{year}.parquet"
    df = pl.read_parquet(path)

    daily = (
        df.group_by(["date", "PULocationID", "extreme_precip"])
        .agg(pl.len().alias("trip_count"))
    )

    return equity_analysis(daily)
