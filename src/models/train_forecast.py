"""
Trains and evaluates forecast models for each horizon (24h/48h/72h):
  1. Naive persistence baseline (tomorrow = today) — the bar any real model must clear
  2. LightGBM regressor per horizon, with SHAP explainability

Evaluation design (chronological, no shuffling):
  fit   : earliest part of the data, trains the models
  calib : middle slice, used for early stopping AND for conformal calibration
          of the prediction intervals
  test  : latest slice, touched ONLY for the reported metrics
A purge gap of 72h separates the slices. Row t's target is the AQI at t+h, so
without the gap a fit row near a boundary would carry labels from the NEXT
slice (leakage). Early stopping used to run on the test set itself, which made
the reported test metrics optimistic; that is fixed here.

Run: python src/models/train_forecast.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import json
import os
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", message=".*eval_set.*")  # noisy LightGBM deprecation

FEATURES_PATH = "data/processed/features.csv"
MODEL_DIR = "src/models/artifacts"
HORIZONS = [24, 48, 72]

# Calendar columns that act as "which period is this?" labels when the data
# spans only ~4 months: a tree can use month=7,8,9 to memorise the training
# period, and month=10/11 at test time was never seen. Weekday effects can't
# be learned from ~15 weeks either. Excluded from the features (not from the CSV).
DROP_FEATURES = ["month", "day_of_week", "is_weekend"]

NON_FEATURE_COLS = ["city", "timestamp"] + [f"target_{h}h" for h in HORIZONS] + DROP_FEATURES


def time_aware_split(df: pd.DataFrame, test_frac: float = 0.2,
                     calib_frac: float = 0.15, purge_hours: int = max(HORIZONS)):
    """Chronological fit / calib / test with a purge gap before each boundary."""
    df = df.sort_values("timestamp")
    t_test = df["timestamp"].quantile(1 - test_frac)
    t_calib = df["timestamp"].quantile(1 - test_frac - calib_frac)
    gap = pd.Timedelta(hours=purge_hours)
    fit = df[df["timestamp"] < t_calib - gap]
    calib = df[(df["timestamp"] >= t_calib) & (df["timestamp"] < t_test - gap)]
    test = df[df["timestamp"] >= t_test]
    return fit, calib, test


def _xy(df: pd.DataFrame, feature_cols: list, target_col: str):
    d = df.dropna(subset=[target_col])
    return d[feature_cols], d[target_col]


def naive_persistence_baseline(test: pd.DataFrame, horizon: int) -> dict:
    valid = test.dropna(subset=[f"target_{horizon}h", "aqi_proxy"])
    y_true = valid[f"target_{horizon}h"]
    y_pred = valid["aqi_proxy"]
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def train_lightgbm(fit: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame, horizon: int):
    target_col = f"target_{horizon}h"
    feature_cols = [c for c in fit.columns if c not in NON_FEATURE_COLS]

    X_fit, y_fit = _xy(fit, feature_cols, target_col)
    X_cal, y_cal = _xy(calib, feature_cols, target_col)
    X_test, y_test = _xy(test, feature_cols, target_col)

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    # early stopping watches the CALIB slice, never the test slice
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_cal, y_cal)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    preds = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
    }
    return model, metrics, feature_cols, X_test


def train_quantile_models(fit: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame,
                          horizon: int, quantiles=(0.1, 0.9)) -> dict:
    """
    p10/p90 quantile regressors, then conformalized quantile regression (CQR):
    the raw band is widened (or narrowed) by `offset`, the finite-sample
    quantile of how far calibration actuals fell outside the band. That makes
    coverage honest instead of whatever the pinball-loss fit happened to give.
    Assumes calib and test look alike; a season shift can still under-cover,
    which is why the raw and calibrated test coverage are both reported.
    """
    target_col = f"target_{horizon}h"
    feature_cols = [c for c in fit.columns if c not in NON_FEATURE_COLS]

    X_fit, y_fit = _xy(fit, feature_cols, target_col)
    X_cal, y_cal = _xy(calib, feature_cols, target_col)
    X_test, y_test = _xy(test, feature_cols, target_col)

    models = {}
    for q in quantiles:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1,
        )
        m.fit(X_fit, y_fit, eval_set=[(X_cal, y_cal)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        models[q] = m

    def band(X):
        a, b = models[quantiles[0]].predict(X), models[quantiles[1]].predict(X)
        return np.minimum(a, b), np.maximum(a, b)

    nominal = quantiles[1] - quantiles[0]
    lo_c, hi_c = band(X_cal)
    y_c = y_cal.to_numpy()
    scores = np.maximum(lo_c - y_c, y_c - hi_c)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * nominal) / n)
    offset = float(np.quantile(scores, level, method="higher"))

    lo_t, hi_t = band(X_test)
    y_t = y_test.to_numpy()
    raw_cov = float(((y_t >= lo_t) & (y_t <= hi_t)).mean())
    cal_cov = float(((y_t >= lo_t - offset) & (y_t <= hi_t + offset)).mean())

    return {"models": models, "coverage": cal_cov, "coverage_raw": raw_cov,
            "offset": offset, "nominal": nominal}


def compute_shap_summary(model, X_test: pd.DataFrame, top_n: int = 10) -> list:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.sample(min(500, len(X_test)), random_state=42))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(X_test.columns, mean_abs_shap), key=lambda x: -x[1])
    return [{"feature": f, "mean_abs_shap": round(float(v), 3)} for f, v in ranked[:top_n]]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    fit, calib, test = time_aware_split(df)

    for name, part in (("Fit  ", fit), ("Calib", calib), ("Test ", test)):
        print(f"{name}: {len(part)} rows ({part['timestamp'].min()} to {part['timestamp'].max()})")
    print()

    results = {}
    for horizon in HORIZONS:
        print(f"--- {horizon}h horizon ---")

        baseline_metrics = naive_persistence_baseline(test, horizon)
        print(f"  Naive baseline   MAE={baseline_metrics['mae']:.2f}  RMSE={baseline_metrics['rmse']:.2f}")

        model, model_metrics, feature_cols, X_test = train_lightgbm(fit, calib, test, horizon)
        print(f"  LightGBM         MAE={model_metrics['mae']:.2f}  RMSE={model_metrics['rmse']:.2f}")

        improvement = (baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"] * 100
        print(f"  Improvement over baseline: {improvement:.1f}%\n")

        quantile_result = train_quantile_models(fit, calib, test, horizon)
        print(f"  Interval coverage: raw {quantile_result['coverage_raw']:.1%} -> "
              f"calibrated {quantile_result['coverage']:.1%} "
              f"(nominal {quantile_result['nominal']:.0%}, offset {quantile_result['offset']:+.1f})")

        top_features = compute_shap_summary(model, X_test)

        model.booster_.save_model(f"{MODEL_DIR}/lgbm_{horizon}h.txt")
        for q, qm in quantile_result["models"].items():
            qm.booster_.save_model(f"{MODEL_DIR}/lgbm_{horizon}h_q{int(q * 100)}.txt")

        # feature order must be saved — LightGBM boosters take a raw array at
        # predict time, so the dashboard has to rebuild columns in this order
        results[f"{horizon}h"] = {
            "baseline": baseline_metrics,
            "model": model_metrics,
            "improvement_pct": improvement,
            "interval_coverage": quantile_result["coverage"],
            "interval_coverage_raw": quantile_result["coverage_raw"],
            "interval_offset": quantile_result["offset"],
            "interval_nominal": quantile_result["nominal"],
            "top_features": top_features,
            "feature_cols": feature_cols,
        }

    with open(f"{MODEL_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved models and metrics to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
