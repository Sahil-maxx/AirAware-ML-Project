import numpy as np
import pandas as pd

from .data_preprocessing import POLLUTANTS


SEASONS = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Summer",
    4: "Summer",
    5: "Summer",
    6: "Monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Post-Monsoon",
    11: "Post-Monsoon",
}


def make_forecasting_frame(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build past-only features to predict each city's next observed AQI."""
    data = data.sort_values(["City", "Date"]).copy()
    grouped = data.groupby("City", sort=False)

    frame = pd.DataFrame(index=data.index)
    frame["City"] = data["City"].astype(str)
    frame["Date"] = data["Date"]
    frame["AQI_Today"] = data["AQI"]

    for lag in (1, 3, 7):
        frame[f"AQI_Lag_{lag}"] = grouped["AQI"].shift(lag)

    for pollutant in POLLUTANTS:
        if pollutant in data.columns:
            frame[f"{pollutant}_Lag_1"] = grouped[pollutant].shift(1)

    previous_aqi = grouped["AQI"].shift(1)
    previous_by_city = previous_aqi.groupby(data["City"])
    frame["AQI_Rolling_Mean_3"] = previous_by_city.transform(
        lambda values: values.rolling(3, min_periods=1).mean()
    )
    frame["AQI_Rolling_Mean_7"] = previous_by_city.transform(
        lambda values: values.rolling(7, min_periods=1).mean()
    )

    frame["Month"] = data["Date"].dt.month.astype(str)
    frame["DayOfWeek"] = data["Date"].dt.dayofweek.astype(str)
    frame["Season"] = data["Date"].dt.month.map(SEASONS)
    frame["Target_Date"] = grouped["Date"].shift(-1)
    frame["AQI_Tomorrow"] = grouped["AQI"].shift(-1)

    target_columns = {"Date", "Target_Date", "AQI_Tomorrow"}
    features = [column for column in frame.columns if column not in target_columns]

    # Rows without current or next-observation AQI cannot train or score a model.
    frame = frame.dropna(subset=["AQI_Today", "AQI_Tomorrow"]).copy()
    return frame, features


def category(aqi: float | int | None) -> str | None:
    """Convert a predicted AQI value to the project's CPCB category bands."""
    if pd.isna(aqi):
        return None

    # Round continuous predictions to the nearest integer before categorizing.
    rounded_aqi = int(np.floor(float(aqi) + 0.5))
    if rounded_aqi <= 50:
        return "Good"
    if rounded_aqi <= 100:
        return "Satisfactory"
    if rounded_aqi <= 200:
        return "Moderate"
    if rounded_aqi <= 300:
        return "Poor"
    if rounded_aqi <= 400:
        return "Very Poor"
    return "Severe"
