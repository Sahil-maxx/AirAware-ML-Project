from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .feature_engineering import category


CATEGORICAL_FEATURES = {"City", "Month", "DayOfWeek", "Season"}
CATEGORY_LABELS = [
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe",
]


def _make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [
        column for column in features if column in CATEGORICAL_FEATURES
    ]
    numeric = [column for column in features if column not in categorical]

    numeric_pipeline = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ]
    )


def _make_pipeline(
    features: list[str],
    regressor,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _make_preprocessor(features)),
            ("model", regressor),
        ]
    )


def _regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
    }


def _chronological_split(frame: pd.DataFrame):
    dates = np.array(sorted(frame.Target_Date.unique()))
    if len(dates) < 10:
        raise ValueError(
            "Not enough distinct target dates for chronological "
            "train/validation/test splits."
        )

    train_end = dates[int(0.70 * len(dates))]
    valid_end = dates[int(0.85 * len(dates))]
    train = frame[frame.Target_Date <= train_end]
    valid = frame[
        (frame.Target_Date > train_end)
        & (frame.Target_Date <= valid_end)
    ]
    test = frame[frame.Target_Date > valid_end]

    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition.")

    print("\nChronological split (by target date; no shuffling):")
    for label, partition in (
        ("Train", train),
        ("Validation", valid),
        ("Test", test),
    ):
        first_date = partition.Target_Date.min().date()
        last_date = partition.Target_Date.max().date()
        print(
            f"  {label:10} {len(partition):6,} rows | "
            f"{first_date} to {last_date}"
        )

    return train, valid, test


def _validation_results(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    candidates: dict,
) -> tuple[list[dict], str]:
    validation_end = pd.Timestamp(valid.Target_Date.max()).date()
    evaluation_period = f"validation through {validation_end}"
    persistence_prediction = valid.AQI_Today.to_numpy()
    results = [
        {
            "Model_Name": "Persistence",
            **_regression_metrics(valid.AQI_Tomorrow, persistence_prediction),
            "Evaluation_Period": evaluation_period,
        }
    ]

    for name, regressor in candidates.items():
        print(f"\nFitting {name} on training rows...")
        pipeline = _make_pipeline(features, regressor)
        pipeline.fit(train[features], train.AQI_Tomorrow)
        prediction = pipeline.predict(valid[features])
        results.append(
            {
                "Model_Name": name,
                **_regression_metrics(valid.AQI_Tomorrow, prediction),
                "Evaluation_Period": evaluation_period,
            }
        )

    best_model = min(
        (result for result in results if result["Model_Name"] != "Persistence"),
        key=lambda result: result["MAE"],
    )["Model_Name"]
    print(
        f"\nValidation MAE selects {best_model} among the ML models. "
        "Persistence remains the baseline."
    )
    return results, best_model


def _test_predictions(
    train_valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    candidates: dict,
    selected_model: str,
) -> tuple[dict[str, np.ndarray], Pipeline]:
    fitted_models = {}
    selected_pipeline = None

    for name, regressor in candidates.items():
        print(f"Fitting {name} on train + validation rows for the test comparison...")
        pipeline = _make_pipeline(features, regressor)
        pipeline.fit(train_valid[features], train_valid.AQI_Tomorrow)
        fitted_models[name] = pipeline
        if name == selected_model:
            selected_pipeline = pipeline

    predictions = {
        "Persistence": test.AQI_Today.to_numpy(),
        **{
            name: pipeline.predict(test[features])
            for name, pipeline in fitted_models.items()
        },
    }
    if selected_pipeline is None:
        raise RuntimeError(f"Selected model was not fitted: {selected_model}")

    return predictions, selected_pipeline


def fit_and_evaluate(
    frame: pd.DataFrame,
    features: list[str],
    model_dir: Path,
    reports_dir: Path,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, list[str]]:
    """Train candidates chronologically, select on validation, and score test."""
    frame = frame.sort_values(["Target_Date", "City"]).reset_index(drop=True)
    train, valid, test = _chronological_split(frame)

    missing_by_feature = train[features].isna().sum().sort_values(
        ascending=False
    )
    missing_by_feature = missing_by_feature[missing_by_feature > 0]
    total_missing = int(train[features].isna().sum().sum())
    print(
        f"\nTraining predictors with missing values: {total_missing:,} "
        f"missing cells across {len(missing_by_feature)} features."
    )
    if not missing_by_feature.empty:
        print(missing_by_feature.head(8).to_string())
    print(
        "  Numeric inputs use training medians; categorical inputs use "
        "training most-frequent values."
    )
    print("  Imputers are fitted inside each pipeline using training rows only.")

    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }
    validation_metrics, selected_model = _validation_results(
        train,
        valid,
        features,
        candidates,
    )

    train_valid = pd.concat([train, valid])
    test_predictions, final_model = _test_predictions(
        train_valid,
        test,
        features,
        candidates,
        selected_model,
    )

    predictions = test_predictions[selected_model]
    results = test[
        ["City", "Date", "Target_Date", "AQI_Tomorrow", "AQI_Today"]
    ].copy()
    results = results.rename(
        columns={
            "Date": "Feature_Date",
            "Target_Date": "Date",
            "AQI_Tomorrow": "Actual_AQI_Tomorrow",
        }
    )
    results["Predicted_AQI_Tomorrow"] = predictions
    results["Absolute_Error"] = (
        results.Actual_AQI_Tomorrow - results.Predicted_AQI_Tomorrow
    ).abs()
    results["Actual_Category"] = results.Actual_AQI_Tomorrow.map(category)
    results["Predicted_Category"] = results.Predicted_AQI_Tomorrow.map(category)
    results["Risk_Category"] = results.Predicted_Category

    test_period = (
        f"test {results.Date.min().date()} to {results.Date.max().date()}"
    )
    test_metrics = []
    for name, prediction in test_predictions.items():
        test_metrics.append(
            {
                "Model_Name": name,
                **_regression_metrics(test.AQI_Tomorrow, prediction),
                "Evaluation_Period": test_period,
            }
        )

    print("\nHeld-out test results (not used for model selection):")
    print(pd.DataFrame(test_metrics).to_string(index=False))

    reports_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = pd.DataFrame([*validation_metrics, *test_metrics])
    all_metrics.to_csv(reports_dir / "model_comparison.csv", index=False)
    results.to_csv(reports_dir / "test_predictions.csv", index=False)

    confusion = confusion_matrix(
        results.Actual_Category,
        results.Predicted_Category,
        labels=CATEGORY_LABELS,
    )
    pd.DataFrame(
        confusion,
        index=CATEGORY_LABELS,
        columns=CATEGORY_LABELS,
    ).to_csv(reports_dir / "category_confusion_matrix.csv")

    report = classification_report(
        results.Actual_Category,
        results.Predicted_Category,
        labels=CATEGORY_LABELS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(
        reports_dir / "category_classification_report.csv"
    )

    transformed_features = (
        final_model.named_steps["preprocess"].get_feature_names_out()
    )
    importances = final_model.named_steps["model"].feature_importances_
    feature_importance = pd.DataFrame(
        {"Feature": transformed_features, "Importance": importances}
    ).sort_values("Importance", ascending=False)
    feature_importance.to_csv(
        reports_dir / "feature_importance.csv",
        index=False,
    )

    model_path = model_dir / "aqi_forecaster.joblib"
    dump(final_model, model_path)
    print(f"\nSaved model and preprocessing pipeline to {model_path}")

    return final_model, results, all_metrics, features
