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
# # 04 — Predictive Modeling
# XGBoost and LightGBM with time-series cross-validation and SHAP.

# %%
from src.features.build_features import build_feature_set
from src.models.predict import (
    cv_benchmark,
    shap_explanation,
    _load_features,
    _prepare_xy,
    train_xgb,
)
from src.visualization.plot import plot_feature_importance

# %%
features_path = build_feature_set(year=2024)
print(f"Features saved: {features_path}")

# %%
results = cv_benchmark(year=2024, n_splits=3)
results

# %%
shap_results = shap_explanation(year=2024)
shap_results

# %%
plot_feature_importance(
    shap_results["top_features"],
    shap_results["mean_abs_shap"],
)
print("Feature importance plot saved to reports/figures/")
