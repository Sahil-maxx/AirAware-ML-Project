from pathlib import Path
import shutil

from src.data_preprocessing import load_and_validate, write_data_report
from src.eda import make_eda
from src.export_tableau_data import export_all
from src.feature_engineering import make_forecasting_frame
from src.model_visualization import plot_model_dashboard
from src.train_model import fit_and_evaluate


ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "raw" / "city_day.csv"


def main() -> None:
    # Keep the provided root-level CSV as a convenient first-run option.
    source_copy = ROOT / "city_day.csv"
    if not RAW_DATA.exists() and source_copy.exists():
        RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_copy, RAW_DATA)

    print("[1/6] Loading and validating the dataset...")
    data = load_and_validate(RAW_DATA)
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed_dir / "city_day_clean.csv", index=False)
    write_data_report(data, ROOT / "outputs" / "reports")
    print(
        f"  {len(data):,} records, {data.City.nunique()} cities, "
        f"{data.Date.min().date()} to {data.Date.max().date()}"
    )

    print("[2/6] Creating exploratory charts...")
    make_eda(data, ROOT / "outputs" / "figures")

    print("[3/6] Building features and training forecasting models...")
    forecasting_frame, features = make_forecasting_frame(data)
    _, predictions, metrics, _ = fit_and_evaluate(
        forecasting_frame,
        features,
        ROOT / "models",
        ROOT / "outputs" / "reports",
    )
    print(metrics.to_string(index=False))

    print("[4/6] Creating the model-results figure...")
    figure_path = plot_model_dashboard(
        ROOT / "outputs" / "reports",
        ROOT / "outputs" / "figures" / "model_results_dashboard.png",
    )
    print(f"  Saved: {figure_path.relative_to(ROOT)}")

    print("[5/6] Exporting Tableau data...")
    export_all(data, predictions, metrics, ROOT / "data" / "tableau")

    print("[6/6] Complete. Review outputs/reports/ and outputs/figures/.")


if __name__ == "__main__":
    main()
