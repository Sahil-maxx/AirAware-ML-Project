from pathlib import Path

import pandas as pd

from .data_preprocessing import POLLUTANTS
from .feature_engineering import SEASONS


def export_all(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write the historical, forecast, city-summary and model-metric CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    historical = data.copy()
    historical["Month"] = historical.Date.dt.month
    historical["Year"] = historical.Date.dt.year
    historical["Season"] = historical.Date.dt.month.map(SEASONS)

    historical_columns = [
        "City",
        "Date",
        "AQI",
        "AQI_Bucket",
        *[column for column in POLLUTANTS if column in historical.columns],
        "Month",
        "Year",
        "Season",
    ]
    historical[historical_columns].to_csv(
        output_dir / "tableau_historical_aqi.csv",
        index=False,
    )
    predictions.to_csv(output_dir / "tableau_predictions.csv", index=False)

    valid_aqi = data.dropna(subset=["AQI"])
    city_summary = (
        valid_aqi.groupby("City")
        .agg(
            **{
                "Average AQI": ("AQI", "mean"),
                "Maximum AQI": ("AQI", "max"),
                "Available observation count": ("AQI", "count"),
            }
        )
        .reset_index()
    )
    poor_days = (
        valid_aqi.assign(PoorOrWorse=valid_aqi.AQI > 200)
        .groupby("City")["PoorOrWorse"]
        .sum()
        .rename("Poor-or-worse days")
        .reset_index()
    )
    city_summary = city_summary.merge(poor_days, on="City")
    city_summary.to_csv(output_dir / "tableau_city_summary.csv", index=False)
    metrics.to_csv(output_dir / "tableau_model_metrics.csv", index=False)
