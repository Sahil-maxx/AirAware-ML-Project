from pathlib import Path

import pandas as pd


POLLUTANTS = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene",
]
REQUIRED_COLUMNS = ["City", "Date", "AQI"]
DATASET_URL = "https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india"


def load_and_validate(path: Path) -> pd.DataFrame:
    """Read the source CSV and standardize its key fields."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Download city_day.csv from {DATASET_URL} and save it to "
            "data/raw/city_day.csv"
        )

    data = pd.read_csv(path)
    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["City", "Date"]).copy()
    data = data.sort_values(["City", "Date"])
    data = data.drop_duplicates(["City", "Date"], keep="last")

    numeric_columns = [
        "AQI",
        *[column for column in POLLUTANTS if column in data.columns],
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if "AQI_Bucket" not in data.columns:
        data["AQI_Bucket"] = pd.NA

    return data.reset_index(drop=True)


def write_data_report(data: pd.DataFrame, output_dir: Path) -> None:
    """Write dataset missingness and summary reports as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = data.isna().sum().rename("Missing_Count").to_frame()
    missing["Missing_Pct"] = (
        missing["Missing_Count"] / max(len(data), 1) * 100
    ).round(2)
    missing.to_csv(output_dir / "missing_values.csv")

    summary = {
        "rows": len(data),
        "columns": len(data.columns),
        "cities": int(data.City.nunique()),
        "date_min": data.Date.min().date().isoformat(),
        "date_max": data.Date.max().date().isoformat(),
        "duplicate_city_date_rows": int(
            data.duplicated(["City", "Date"]).sum()
        ),
        "aqi_nonmissing": int(data.AQI.notna().sum()),
    }
    pd.Series(summary, name="value").to_csv(
        output_dir / "data_summary.csv",
        header=True,
    )
