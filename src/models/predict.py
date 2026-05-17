import polars as pl
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
from src.data.config import DATA_PROCESSED, FIGURES_DIR


MODEL_PARAMS = {
    "xgb": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "lgb": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
}

FEATURE_COLS = [
    "month", "weekday", "day_of_year", "is_weekend",
    "PRCP", "TMAX", "TMIN", "AWND",
    "any_precip", "extreme_precip", "heat_day", "freeze_day", "snow_day",
    "trip_count_lag1d", "trip_count_lag2d", "trip_count_lag3d", "trip_count_lag7d",
    "trip_count_roll7d", "trip_count_roll30d",
]

TARGET = "trip_count"


def _load_features(year: int) -> pl.DataFrame:
    path = DATA_PROCESSED / f"features_{year}.parquet"
    return pl.read_parquet(path)


def _prepare_xy(
    df: pl.DataFrame,
    feature_cols: list[str] | None = None,
    target: str = TARGET,
) -> tuple:
    cols = feature_cols or FEATURE_COLS
    available = [c for c in cols if c in df.columns]
    pandas_df = df.to_pandas().dropna(subset=available + [target])
    X = pandas_df[available]
    y = pandas_df[target]
    return X, y


def _timeseries_cv(X, y, n_splits: int = 3):
    return TimeSeriesSplit(n_splits=n_splits).split(X, y)


def train_xgb(X, y) -> tuple:
    params = MODEL_PARAMS["xgb"].copy()
    native_params = {k: v for k, v in params.items() if k != "random_state"}
    native_params["random_state"] = params["random_state"]
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def train_lgb(X, y) -> tuple:
    params = MODEL_PARAMS["lgb"].copy()
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def evaluate_model(model, X, y, name: str) -> dict:
    preds = model.predict(X)
    return {
        "model": name,
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }


def cv_benchmark(
    year: int = 2024,
    n_splits: int = 3,
) -> dict:
    df = _load_features(year)
    X, y = _prepare_xy(df)

    tscv = _timeseries_cv(X, y, n_splits)

    xgb_scores = []
    lgb_scores = []

    for train_idx, val_idx in tscv:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        xgb_model = train_xgb(X_train, y_train)
        lgb_model = train_lgb(X_train, y_train)

        xgb_scores.append(evaluate_model(xgb_model, X_val, y_val, "xgboost"))
        lgb_scores.append(evaluate_model(lgb_model, X_val, y_val, "lightgbm"))

    def _avg(scores: list[dict]) -> dict:
        result = {"model": scores[0]["model"]}
        for metric in ["rmse", "mae", "r2"]:
            result[metric] = float(np.mean([s[metric] for s in scores]))
            result[f"{metric}_std"] = float(np.std([s[metric] for s in scores]))
        return result

    return {"xgboost_cv": _avg(xgb_scores), "lightgbm_cv": _avg(lgb_scores)}


def shap_explanation(year: int = 2024) -> dict:
    df = _load_features(year)
    X, y = _prepare_xy(df)

    model = train_xgb(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    feature_importance = list(zip(X.columns, np.abs(shap_values).mean(axis=0)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return {
        "top_features": [f[0] for f in feature_importance[:10]],
        "mean_abs_shap": [float(f[1]) for f in feature_importance[:10]],
        "model_r2": float(r2_score(y, model.predict(X))),
    }
