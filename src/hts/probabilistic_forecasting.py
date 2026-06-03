"""Probabilistic daily cafeteria sales forecasting.

This module forecasts total dish sales for regular cafeteria service days.  It
is intentionally small and explicit: chronological splits, leakage-safe
similar-day features, quantile forecasts, and an empirical baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.50, 0.90, 0.95)
QUANTILE_COLUMNS: tuple[str, ...] = ("p05", "p10", "p50", "p90", "p95")
IMPLEMENTATION_VERSION = "probabilistic_forecasting_2026_06_03_2"

BASE_NUMERIC_FEATURES: tuple[str, ...] = (
    "weekday_number",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
    "year_sin",
    "year_cos",
    "sunshine_duration",
    "apparent_temperature_mean",
)
OPTIONAL_NUMERIC_FEATURES: tuple[str, ...] = (
    "is_first_lecture_week",
    "is_orientation_week",
)
BASE_CATEGORICAL_FEATURES: tuple[str, ...] = ("academic_bucket",)
OPTIONAL_CATEGORICAL_FEATURES: tuple[str, ...] = ("semester_season",)

SIMILAR_CONTEXT_FEATURES: tuple[str, ...] = (
    "similar_sales_weighted_mean",
    "similar_sales_median",
    "similar_sales_p05",
    "similar_sales_p10",
    "similar_sales_p90",
    "similar_sales_p95",
    "similar_day_count",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "is_regular_service_day",
    "sales",
    *BASE_NUMERIC_FEATURES,
    *BASE_CATEGORICAL_FEATURES,
)


def load_feature_frame(path: str | Path) -> pd.DataFrame:
    """Load a feature frame and normalize the date column."""

    df = pd.read_csv(path, parse_dates=["date"])
    return _normalize_frame(df)


def make_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return regular service days with observed sales, sorted by date."""

    normalized = _normalize_frame(df)
    _require_columns(normalized, REQUIRED_COLUMNS)

    training = normalized[
        normalized["is_regular_service_day"] & normalized["sales"].notna()
    ].copy()
    training["sales"] = pd.to_numeric(training["sales"], errors="coerce")
    training = training[training["sales"].notna()].copy()
    training["sales"] = training["sales"].astype(float)

    if training.empty:
        raise ValueError("No regular service rows with observed sales were found.")

    return training.sort_values("date").reset_index(drop=True)


def select_similar_days(
    train_df: pd.DataFrame,
    target_row: pd.Series | dict[str, Any],
    k: int = 5,
) -> pd.DataFrame:
    """Select the k most similar historical regular service days.

    Only rows before the target date are considered when the target has a date.
    Similarity is based on standardized calendar/weather distance with clear
    categorical penalties for different academic states.
    """

    history = make_training_frame(train_df)
    target = pd.Series(target_row)
    if "date" in target and pd.notna(target["date"]):
        target_date = pd.to_datetime(target["date"])
        history = history[history["date"] < target_date].copy()

    if history.empty:
        return _empty_similar_days()

    numeric_columns = [col for col in BASE_NUMERIC_FEATURES if col in history.columns]
    numeric_columns.extend(col for col in OPTIONAL_NUMERIC_FEATURES if col in history.columns)

    history_numeric = history[numeric_columns].apply(pd.to_numeric, errors="coerce")
    means = history_numeric.mean()
    stds = history_numeric.std(ddof=0).replace(0, 1).fillna(1)
    target_numeric = pd.to_numeric(
        pd.Series({col: target.get(col, np.nan) for col in numeric_columns}),
        errors="coerce",
    ).fillna(means)

    standardized = (history_numeric.fillna(means) - means) / stds
    target_standardized = (target_numeric - means) / stds
    numeric_distance = np.sqrt(((standardized - target_standardized) ** 2).sum(axis=1))

    penalties = pd.Series(0.0, index=history.index)
    if "academic_bucket" in history.columns and "academic_bucket" in target:
        penalties += np.where(history["academic_bucket"] == target["academic_bucket"], 0.0, 1.5)
    if "semester_season" in history.columns and "semester_season" in target:
        penalties += np.where(history["semester_season"] == target["semester_season"], 0.0, 0.5)

    ranked = history.assign(distance=numeric_distance + penalties)
    ranked = ranked.sort_values(["distance", "date"]).head(k).copy()
    ranked.insert(0, "similar_rank", np.arange(1, len(ranked) + 1))

    output_columns = [
        "similar_rank",
        "date",
        "sales",
        "distance",
        "weekday_number",
        "academic_bucket",
        "sunshine_duration",
        "apparent_temperature_mean",
    ]
    if "semester_season" in ranked.columns:
        output_columns.append("semester_season")

    return ranked[[col for col in output_columns if col in ranked.columns]].reset_index(drop=True)


def fit_probabilistic_forecaster(train_df: pd.DataFrame) -> "ProbabilisticForecaster":
    """Fit quantile models for p05, p10, p50, p90, and p95."""

    training = make_training_frame(train_df)
    model_frame = _add_leakage_safe_similar_context(training)
    model_frame = model_frame[model_frame["similar_day_count"] > 0].copy()

    if len(model_frame) < 30:
        raise ValueError(
            "At least 30 regular service rows with prior similar-day context are required."
        )

    numeric_features = _numeric_feature_columns(model_frame)
    categorical_features = _categorical_feature_columns(model_frame)
    feature_columns = [*numeric_features, *categorical_features]

    models: dict[str, Pipeline] = {}
    x_train = model_frame[feature_columns]
    y_train = model_frame["sales"]

    for quantile, column in zip(QUANTILES, QUANTILE_COLUMNS, strict=True):
        regressor = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            n_estimators=150,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=8,
            random_state=42,
        )
        model = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(numeric_features, categorical_features)),
                ("regressor", regressor),
            ]
        )
        model.fit(x_train, y_train)
        models[column] = model

    return ProbabilisticForecaster(
        models=models,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        training_history=training,
    )


def forecast_next_service_days(
    history_df: pd.DataFrame,
    future_covariates_df: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """Forecast the next horizon regular service days from future covariates."""

    forecaster = fit_probabilistic_forecaster(history_df)
    return forecaster.predict(future_covariates_df, horizon=horizon)


def backtest_probabilistic_forecaster(
    df: pd.DataFrame,
    initial_train_fraction: float = 0.60,
    horizon: int = 5,
    step_size: int = 5,
    final_holdout_service_days: int = 40,
) -> dict[str, Any]:
    """Run an expanding-window chronological backtest."""

    data = make_training_frame(df)
    holdout_start = max(
        int(np.floor(len(data) * initial_train_fraction)),
        len(data) - final_holdout_service_days,
    )
    initial_train_size = max(30, int(np.floor(len(data) * initial_train_fraction)))
    first_split = min(initial_train_size, holdout_start)

    forecast_parts: list[pd.DataFrame] = []
    for train_end in range(first_split, len(data), step_size):
        train = data.iloc[:train_end].copy()
        test = data.iloc[train_end : train_end + horizon].copy()
        if len(test) == 0 or len(train) < 30:
            continue

        forecaster = fit_probabilistic_forecaster(train)
        forecast = forecaster.predict(test, horizon=len(test))
        actuals = test[["date", "sales"]].copy()
        forecast = forecast.merge(actuals, on="date", how="left")
        forecast["split_train_end_date"] = train["date"].max()
        forecast_parts.append(forecast)

    if not forecast_parts:
        raise ValueError("Backtest could not produce any forecast windows.")

    forecasts = pd.concat(forecast_parts, ignore_index=True)
    metrics = _evaluate_forecasts(forecasts)

    final_holdout = data.iloc[-final_holdout_service_days:].copy()
    metrics["final_holdout_start"] = final_holdout["date"].min()
    metrics["final_holdout_end"] = final_holdout["date"].max()

    return {
        "metrics": metrics,
        "forecasts": forecasts,
    }


@dataclass(frozen=True)
class ProbabilisticForecaster:
    """Fitted probabilistic sales forecaster."""

    models: dict[str, Pipeline]
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    training_history: pd.DataFrame

    def predict(self, future_covariates_df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """Predict quantiles for the next regular service days."""

        future = _normalize_frame(future_covariates_df)
        _require_columns(
            future,
            ("date", "is_regular_service_day", *BASE_NUMERIC_FEATURES, *BASE_CATEGORICAL_FEATURES),
        )

        future = future[future["is_regular_service_day"]].sort_values("date").head(horizon)
        if future.empty:
            raise ValueError("No future regular service days were found.")

        rows: list[dict[str, Any]] = []
        model_frame = future.copy()
        context_rows = []
        similar_days_by_date: dict[pd.Timestamp, pd.DataFrame] = {}

        for _, row in model_frame.iterrows():
            similar_days = select_similar_days(self.training_history, row, k=5)
            similar_days_by_date[pd.to_datetime(row["date"])] = similar_days
            context_rows.append(_similar_context_from_days(similar_days))

        context_df = pd.DataFrame(context_rows, index=model_frame.index)
        model_frame = pd.concat([model_frame, context_df], axis=1)

        raw_predictions = pd.DataFrame(index=model_frame.index)
        for column in QUANTILE_COLUMNS:
            raw_predictions[column] = self.models[column].predict(
                model_frame[self.feature_columns]
            )

        quantiles = _postprocess_quantiles(raw_predictions)

        for idx, (_, row) in enumerate(model_frame.iterrows()):
            baseline = _baseline_quantiles(self.training_history, row)
            date = pd.to_datetime(row["date"])
            similar_days = similar_days_by_date[date]
            forecast_row: dict[str, Any] = {
                "date": date,
                **quantiles.iloc[idx].to_dict(),
                "interval_80": _format_interval(quantiles.iloc[idx]["p10"], quantiles.iloc[idx]["p90"]),
                "interval_90": _format_interval(quantiles.iloc[idx]["p05"], quantiles.iloc[idx]["p95"]),
                "baseline_p05": baseline["p05"],
                "baseline_p10": baseline["p10"],
                "baseline_p50": baseline["p50"],
                "baseline_p90": baseline["p90"],
                "baseline_p95": baseline["p95"],
                "baseline_interval_80": _format_interval(baseline["p10"], baseline["p90"]),
                "baseline_interval_90": _format_interval(baseline["p05"], baseline["p95"]),
                "similar_days": similar_days.to_dict(orient="records"),
            }
            rows.append(forecast_row)

        return pd.DataFrame(rows)


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"])
    if "is_regular_service_day" in normalized.columns:
        normalized["is_regular_service_day"] = normalized["is_regular_service_day"].map(
            _to_bool
        )
    for column in OPTIONAL_NUMERIC_FEATURES:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(_to_bool).astype(float)
    return normalized.sort_values("date").reset_index(drop=True) if "date" in normalized else normalized


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = list(BASE_NUMERIC_FEATURES)
    columns.extend(column for column in OPTIONAL_NUMERIC_FEATURES if column in df.columns)
    columns.extend(column for column in SIMILAR_CONTEXT_FEATURES if column in df.columns)
    return columns


def _categorical_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = list(BASE_CATEGORICAL_FEATURES)
    columns.extend(column for column in OPTIONAL_CATEGORICAL_FEATURES if column in df.columns)
    return columns


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def _add_leakage_safe_similar_context(training: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in training.iterrows():
        history = training.iloc[:index]
        similar_days = select_similar_days(history, row, k=5) if not history.empty else _empty_similar_days()
        rows.append(_similar_context_from_days(similar_days))
    context = pd.DataFrame(rows, index=training.index)
    return pd.concat([training, context], axis=1)


def _similar_context_from_days(similar_days: pd.DataFrame) -> dict[str, float]:
    if similar_days.empty:
        return {
            "similar_sales_weighted_mean": np.nan,
            "similar_sales_median": np.nan,
            "similar_sales_p05": np.nan,
            "similar_sales_p10": np.nan,
            "similar_sales_p90": np.nan,
            "similar_sales_p95": np.nan,
            "similar_day_count": 0.0,
        }

    sales = similar_days["sales"].astype(float)
    weights = 1.0 / (similar_days["distance"].astype(float) + 1e-6)
    return {
        "similar_sales_weighted_mean": float(np.average(sales, weights=weights)),
        "similar_sales_median": float(sales.median()),
        "similar_sales_p05": float(sales.quantile(0.05)),
        "similar_sales_p10": float(sales.quantile(0.10)),
        "similar_sales_p90": float(sales.quantile(0.90)),
        "similar_sales_p95": float(sales.quantile(0.95)),
        "similar_day_count": float(len(sales)),
    }


def _baseline_quantiles(history: pd.DataFrame, target_row: pd.Series) -> dict[str, int]:
    candidates = history[
        (history["weekday_number"] == target_row["weekday_number"])
        & (history["academic_bucket"] == target_row["academic_bucket"])
    ]
    if len(candidates) < 5:
        candidates = history[history["weekday_number"] == target_row["weekday_number"]]
    if len(candidates) < 5:
        candidates = history

    quantiles = candidates["sales"].astype(float).quantile(QUANTILES)
    return {
        column: int(round(max(0.0, float(quantiles.loc[quantile]))))
        for quantile, column in zip(QUANTILES, QUANTILE_COLUMNS, strict=True)
    }


def _postprocess_quantiles(predictions: pd.DataFrame) -> pd.DataFrame:
    clipped = predictions.reindex(columns=list(QUANTILE_COLUMNS))
    if clipped.isna().all(axis=None):
        raise ValueError(
            "No quantile prediction columns were produced. Expected columns: "
            f"{list(QUANTILE_COLUMNS)}. Got columns: {list(predictions.columns)}."
        )
    clipped = clipped.clip(lower=0)
    monotonic = np.maximum.accumulate(clipped.to_numpy(dtype=float), axis=1)
    processed = pd.DataFrame(monotonic, columns=QUANTILE_COLUMNS, index=predictions.index)
    return processed.round().astype(int)


def _format_interval(lower: float, upper: float) -> str:
    return f"{int(round(lower))}-{int(round(upper))}"


def _empty_similar_days() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "similar_rank",
            "date",
            "sales",
            "distance",
            "weekday_number",
            "academic_bucket",
            "sunshine_duration",
            "apparent_temperature_mean",
        ]
    )


def _evaluate_forecasts(forecasts: pd.DataFrame) -> dict[str, float]:
    y_true = forecasts["sales"].astype(float)
    y_pred = forecasts["p50"].astype(float)
    baseline_pred = forecasts["baseline_p50"].astype(float)

    metrics: dict[str, float] = {
        "ml_mae": float(mean_absolute_error(y_true, y_pred)),
        "baseline_mae": float(mean_absolute_error(y_true, baseline_pred)),
        "ml_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_true, baseline_pred))),
        "coverage_80": float(((y_true >= forecasts["p10"]) & (y_true <= forecasts["p90"])).mean()),
        "coverage_90": float(((y_true >= forecasts["p05"]) & (y_true <= forecasts["p95"])).mean()),
        "mean_interval_80_width": float((forecasts["p90"] - forecasts["p10"]).mean()),
        "mean_interval_90_width": float((forecasts["p95"] - forecasts["p05"]).mean()),
        "bias": float((y_pred - y_true).mean()),
    }

    for quantile, column in zip(QUANTILES, QUANTILE_COLUMNS, strict=True):
        metrics[f"pinball_loss_{column}"] = _pinball_loss(
            y_true.to_numpy(),
            forecasts[column].astype(float).to_numpy(),
            quantile,
        )

    metrics["recommended_operational_model"] = (
        "ml" if metrics["ml_mae"] <= metrics["baseline_mae"] else "baseline"
    )
    return metrics


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    errors = y_true - y_pred
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1.0) * errors)))
