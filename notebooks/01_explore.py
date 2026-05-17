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
# # 01 — Data Exploration
# Explore TLC trip records and weather data.

# %%
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.config import DATA_PROCESSED

# %%
df = pl.read_parquet(str(DATA_PROCESSED / "trips_labeled_2024.parquet"))
print(f"Rows: {df.height:,}, Cols: {df.width}")
df.head()

# %%
df.describe()

# %%
daily = df.group_by("date").agg(pl.len().alias("trip_count")).sort("date")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(daily["date"], daily["trip_count"], linewidth=0.7)
ax.set_title("Daily Trip Volume")
plt.show()

# %%
weather_summary = df.group_by("date").agg(
    pl.col("PRCP").first(),
    pl.col("TMAX").first(),
    pl.col("TMIN").first(),
    pl.col("extreme_precip").first(),
).sort("date")

weather_summary.filter(pl.col("extreme_precip") == True)
