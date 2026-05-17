import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import contextily as ctx
from src.data.config import FIGURES_DIR


def set_style():
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10


def plot_trip_volumes(daily: pl.DataFrame, filename: str = "trip_volumes.png"):
    set_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    agg = daily.group_by("date").agg(pl.col("trip_count").sum()).sort("date")
    ax.plot(agg["date"], agg["trip_count"], color="steelblue", linewidth=0.8)
    ax.set_title("Daily NYC Rideshare Trip Volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Trips")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def plot_weather_shocks(
    daily: pl.DataFrame,
    weather: pl.DataFrame,
    filename: str = "weather_shocks.png",
):
    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    agg = daily.group_by("date").agg(pl.col("trip_count").sum()).sort("date")
    ax1.plot(agg["date"], agg["trip_count"], color="steelblue", linewidth=0.8)
    ax1.set_ylabel("Trips")

    w = weather.sort("date")
    ax2.bar(w["date"], w["PRCP"], color="navy", width=0.8, alpha=0.7)
    ax2.set_ylabel("Precip (mm)")
    ax2.set_xlabel("Date")

    fig.suptitle("Trip Volume and Precipitation Over Time")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def plot_equity_comparison(
    equity_results: dict,
    filename: str = "equity_comparison.png",
):
    set_style()
    quintiles = [q for q in ["lowest", "low", "middle", "high", "highest"]
                 if q in equity_results]
    means = [equity_results[q]["mean_pct_of_baseline"] for q in quintiles]
    stds = [equity_results[q]["std_pct_of_baseline"] for q in quintiles]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]
    ax.bar(quintiles, means, yerr=stds, color=colors[:len(quintiles)],
           capsize=5, alpha=0.8)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.7)
    ax.set_title("Trip Recovery by Income Quintile (% of Baseline)")
    ax.set_ylabel("% of Baseline Trip Volume")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def plot_feature_importance(
    feature_names: list[str],
    importance_values: list[float],
    filename: str = "feature_importance.png",
):
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feature_names[:10][::-1], importance_values[:10][::-1],
            color="steelblue", alpha=0.8)
    ax.set_title("Top 10 Features by Mean |SHAP|")
    ax.set_xlabel("Mean |SHAP Value|")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)
