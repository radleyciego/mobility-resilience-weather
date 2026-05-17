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
# # 02 — Causal Impact Analysis
# Difference-in-Differences estimation of weather shocks on trip demand.

# %%
import polars as pl
from src.data.config import DATA_PROCESSED
from src.models.causal_did import aggregate_daily, run_did

# %%
df = pl.read_parquet(str(DATA_PROCESSED / "trips_labeled_2024.parquet"))
daily = aggregate_daily(df)

# %%
result = run_did(
    daily,
    treatment_col="extreme_precip",
    outcome="trip_count",
    fixed_effects=["PULocationID", "month", "weekday"],
)

print(f"DiD Estimate: {result['did_estimate']:.2f}")
print(f"95% CI: {result['ci_95']}")
print(f"p-value: {result['p_value']:.4f}")
print(f"R²: {result['r_squared']:.3f}")

# %%
result["model"].summary()
