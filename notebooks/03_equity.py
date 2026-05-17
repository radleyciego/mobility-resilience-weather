# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03 — Equity Analysis
# Stratify weather impact by neighborhood income quintile.

# %%
import polars as pl
from src.data.config import DATA_PROCESSED
from src.models.equity import load_zone_income, equity_analysis, _compute_recovery_time
from src.visualization.plot import plot_equity_comparison

# %%
df = pl.read_parquet(str(DATA_PROCESSED / "trips_labeled_2024.parquet"))

daily = (
    df.group_by(["date", "PULocationID", "extreme_precip"])
    .agg(pl.len().alias("trip_count"))
)

# %%
income = load_zone_income()
income.head()

# %%
results = equity_analysis(daily)
results

# %%
plot_equity_comparison(results)
print("Equity plot saved to reports/figures/")
