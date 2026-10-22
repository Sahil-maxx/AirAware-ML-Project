"""Print each stage of the AQI forecasting and model-training workflow."""

from pathlib import Path
import shutil

from src.data_preprocessing import load_and_validate
from src.feature_engineering import make_forecasting_frame
from src.train_model import fit_and_evaluate


ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "raw" / "city_day.csv"


def main() -> None:
    source_copy = ROOT / "city_day.csv"
    if not RAW_DATA.exists() and source_copy.exists():
        RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_copy, RAW_DATA)

    print("=" * 72)
    print("AIR AWARE AI | MODEL TRAINING WALKTHROUGH")
    print("=" * 72)

    print("\n[1/4] Load and validate the source data")
    data = load_and_validate(RAW_DATA)
    print(f"Dataset: {RAW_DATA.relative_to(ROOT)}")
    print(
        f"Shape after date parsing and duplicate removal: "
        f"{data.shape[0]:,} rows x {data.shape[1]} columns"
    )
    print(
        f"Cities: {data.City.nunique()} | Date range: "
        f"{data.Date.min().date()} to {data.Date.max().date()}"
    )

    print("\nColumn types:")
    print(data.dtypes.to_string())

    missing = data.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    print("\nSource columns with missing values:")
    print(missing.to_string() if not missing.empty else "  None")
    print(f"Total missing cells: {int(data.isna().sum().sum()):,}")

    print("\n[2/4] Build the target and past-only features")
    ordered = data.sort_values(["City", "Date"])
    next_aqi = ordered.groupby("City", sort=False)["AQI"].shift(-1)
    no_current_aqi = int(ordered.AQI.isna().sum())
    no_next_aqi = int(next_aqi.isna().sum())
    usable_targets = int((ordered.AQI.notna() & next_aqi.notna()).sum())

    print(f"Rows in validated source: {len(data):,}")
    print(f"Rows without current AQI: {no_current_aqi:,}")
    print(f"Rows without a next-observation AQI: {no_next_aqi:,}")
    print(f"Rows with both target values available: {usable_targets:,}")
    print(
        "Rows missing a target value are excluded. Missing predictor values "
        "are kept and imputed during model fitting."
    )

    frame, features = make_forecasting_frame(data)
    print(f"\nTraining frame: {len(frame):,} rows, {len(features)} predictors")
    print("Target: AQI at the city's next available observation")
    print("Predictors:", ", ".join(features))

    feature_missing = frame[features].isna().sum().sort_values(ascending=False)
    feature_missing = feature_missing[feature_missing > 0]
    print("\nMissing predictor values before imputation:")
    print(feature_missing.to_string() if not feature_missing.empty else "  None")

    print("\n[3/4] Split data, fit models, and evaluate")
    print("Training code reports date boundaries and test metrics below.")
    model, predictions, metrics, _ = fit_and_evaluate(
        frame,
        features,
        ROOT / "models",
        ROOT / "outputs" / "reports",
    )

    print("\n[4/4] Output files")
    outputs = {
        "Model and preprocessing": ROOT / "models" / "aqi_forecaster.joblib",
        "Model scores": ROOT / "outputs" / "reports" / "model_comparison.csv",
        "Test predictions": ROOT / "outputs" / "reports" / "test_predictions.csv",
        "Feature importance": ROOT / "outputs" / "reports" / "feature_importance.csv",
    }
    for label, path in outputs.items():
        print(f"  {label:24} {path}")

    selected_model = model.named_steps["model"].__class__.__name__
    print(f"\nTest prediction rows: {len(predictions):,}")
    print(f"Selected model: {selected_model}")
    print("\nModel scores:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
