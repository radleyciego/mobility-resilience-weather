import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from src.data.config import DATA_PROCESSED, FIGURES_DIR


def aggregate_daily(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["date", "PULocationID", "extreme_precip", "any_precip",
                      "heat_day", "freeze_day", "snow_day"])
        .agg(
            pl.len().alias("trip_count"),
            pl.col("trip_miles").mean().alias("avg_trip_miles"),
            pl.col("base_passenger_fare").mean().alias("avg_fare"),
            pl.col("driver_pay").mean().alias("avg_driver_pay"),
        )
        .with_columns(
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.weekday().alias("weekday"),
        )
    )


def _make_treatment_controls(
    df: pl.DataFrame,
    treatment_col: str = "extreme_precip",
    zone_col: str = "PULocationID",
) -> tuple:
    pandas_df = df.to_pandas()

    treated = pandas_df[pandas_df[treatment_col] == True].copy()
    control = pandas_df[pandas_df[treatment_col] == False].copy()

    treated["post"] = 1
    control["post"] = 0

    did_data = treated.copy()
    # sample control days to match treated set size for balance
    n_treated = len(treated)
    if len(control) > n_treated:
        control = control.sample(n=n_treated, random_state=42)

    did_data = pl.concat([
        pl.from_pandas(treated),
        pl.from_pandas(control),
    ])

    return did_data


def run_did(
    df: pl.DataFrame,
    treatment_col: str = "extreme_precip",
    outcome: str = "trip_count",
    fixed_effects: list[str] | None = None,
) -> dict:
    did_data = _make_treatment_controls(df, treatment_col)
    pd_df = did_data.to_pandas()

    pd_df["treated"] = (pd_df[treatment_col] == True).astype(int)

    fe_terms = " + ".join(f"C({fe})" for fe in (fixed_effects or []))
    fe_str = f" + {fe_terms}" if fe_terms else ""

    formula = f"{outcome} ~ treated * post{fe_str}"
    model = smf.ols(formula, data=pd_df)
    result = model.fit(cov_type="HC3")

    interaction_term = "treated:post"
    coef = None
    ci = None
    pval = None
    for name, param in result.params.items():
        if "treated:post" in name or "treated:Post" in name:
            coef = param
            ci = result.conf_int().loc[name].tolist()
            pval = result.pvalues[name]
            break

    if coef is None:
        coef = result.params.get("treated:post", result.params.filter(
            like="treated").filter(like="post").iloc[0] if any("treated" in n and "post" in n for n in result.params.index) else None)
        idx = result.params.filter(like="treated").filter(like="post").index[0] if any(
            "treated" in n and "post" in n for n in result.params.index) else None
        if idx:
            ci = result.conf_int().loc[idx].tolist()
            pval = result.pvalues[idx]

    return {
        "model": result,
        "did_estimate": coef,
        "ci_95": ci,
        "p_value": pval,
        "formula": formula,
        "n_obs": int(result.nobs),
        "r_squared": result.rsquared,
    }


def estimate_weather_impact(
    year: int = 2024,
    treatment: str = "extreme_precip",
) -> dict:
    path = DATA_PROCESSED / f"trips_labeled_{year}.parquet"
    df = pl.read_parquet(path)
    daily = aggregate_daily(df)
    result = run_did(daily, treatment_col=treatment)
    return result
