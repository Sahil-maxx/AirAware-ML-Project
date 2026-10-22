from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_preprocessing import POLLUTANTS


def make_eda(data: pd.DataFrame, output_dir: Path) -> None:
    """Save a small set of exploratory charts for the source dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    valid_aqi = data.dropna(subset=["AQI"])
    if not valid_aqi.empty:
        _save_aqi_distribution(valid_aqi, output_dir)
        _save_monthly_aqi(valid_aqi, output_dir)
        _save_city_aqi(valid_aqi, output_dir)

    pollutant_columns = [
        column for column in POLLUTANTS if column in data.columns
    ]
    if pollutant_columns:
        _save_pollutant_missingness(data, pollutant_columns, output_dir)


def _save_aqi_distribution(data: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 5))
    sns.histplot(data.AQI, bins=50, ax=axis)
    axis.set(title="Historical AQI distribution", xlabel="AQI")
    figure.tight_layout()
    figure.savefig(output_dir / "aqi_distribution.png", dpi=150)
    plt.close(figure)


def _save_monthly_aqi(data: pd.DataFrame, output_dir: Path) -> None:
    monthly_aqi = data.assign(Month=data.Date.dt.month).groupby("Month").AQI.mean()
    figure, axis = plt.subplots(figsize=(9, 4))
    monthly_aqi.plot(marker="o", ax=axis)
    axis.set(
        title="Mean historical AQI by month",
        xlabel="Month",
        ylabel="AQI",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "monthly_aqi.png", dpi=150)
    plt.close(figure)


def _save_city_aqi(data: pd.DataFrame, output_dir: Path) -> None:
    city_aqi = data.groupby("City").AQI.mean().sort_values()
    figure, axis = plt.subplots(figsize=(10, 8))
    city_aqi.plot.barh(ax=axis)
    axis.set(title="Mean historical AQI by city", xlabel="AQI")
    figure.tight_layout()
    figure.savefig(output_dir / "city_mean_aqi.png", dpi=150)
    plt.close(figure)


def _save_pollutant_missingness(
    data: pd.DataFrame,
    pollutant_columns: list[str],
    output_dir: Path,
) -> None:
    missing_share = data[pollutant_columns].isna().mean().sort_values(
        ascending=False
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    missing_share.plot.bar(ax=axis)
    axis.set(title="Pollutant missing-value share", ylabel="Share missing")
    figure.tight_layout()
    figure.savefig(output_dir / "pollutant_missingness.png", dpi=150)
    plt.close(figure)
