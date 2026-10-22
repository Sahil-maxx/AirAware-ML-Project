from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def plot_model_dashboard(
    reports_dir: Path,
    output_path: Path,
    show: bool = False,
) -> Path:
    """Create a one-page visual summary of the model evaluation reports."""
    report_files = {
        "predictions": reports_dir / "test_predictions.csv",
        "metrics": reports_dir / "model_comparison.csv",
        "confusion": reports_dir / "category_confusion_matrix.csv",
        "importance": reports_dir / "feature_importance.csv",
    }
    missing_files = [
        str(path) for path in report_files.values() if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Cannot create the model-results figure; missing reports: "
            + ", ".join(missing_files)
        )

    predictions = pd.read_csv(
        report_files["predictions"],
        parse_dates=["Date"],
    )
    metrics = pd.read_csv(report_files["metrics"])
    confusion = pd.read_csv(report_files["confusion"], index_col=0)
    importance = (
        pd.read_csv(report_files["importance"])
        .head(10)
        .sort_values("Importance")
    )

    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(21, 12),
        constrained_layout=True,
    )
    figure.suptitle(
        "AirAware AI | Model Results (Chronological Held-out Test)",
        fontsize=20,
        fontweight="bold",
    )

    _plot_actual_vs_predicted(predictions, axes[0, 0])
    _plot_prediction_parity(predictions, axes[0, 1])
    _plot_error_distribution(predictions, axes[0, 2])
    _plot_confusion_matrix(confusion, axes[1, 0])
    _plot_feature_importance(importance, axes[1, 1])
    _plot_metric_table(metrics, axes[1, 2])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output_path


def _plot_actual_vs_predicted(
    predictions: pd.DataFrame,
    axis: Axes,
) -> None:
    # Daily city averages keep the time-series panel readable.
    daily = predictions.groupby("Date", as_index=False)[
        ["Actual_AQI_Tomorrow", "Predicted_AQI_Tomorrow"]
    ].mean()
    axis.plot(
        daily.Date,
        daily.Actual_AQI_Tomorrow,
        label="Actual",
        linewidth=1.8,
        color="#245b8f",
    )
    axis.plot(
        daily.Date,
        daily.Predicted_AQI_Tomorrow,
        label="Predicted",
        linewidth=1.8,
        color="#e1812c",
    )
    axis.set_title("Actual vs predicted AQI by date")
    axis.set_xlabel("Target date")
    axis.set_ylabel("AQI (city average)")
    axis.legend(frameon=False)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=7)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _plot_prediction_parity(
    predictions: pd.DataFrame,
    axis: Axes,
) -> None:
    actual = predictions.Actual_AQI_Tomorrow
    predicted = predictions.Predicted_AQI_Tomorrow
    axis.scatter(
        actual,
        predicted,
        alpha=0.22,
        s=13,
        color="#3973a5",
        edgecolors="none",
    )

    lower_bound = min(actual.min(), predicted.min())
    upper_bound = max(actual.max(), predicted.max())
    axis.plot(
        [lower_bound, upper_bound],
        [lower_bound, upper_bound],
        linestyle="--",
        color="#c44e52",
        linewidth=1.5,
        label="Perfect prediction",
    )
    axis.set_title("Prediction parity")
    axis.set_xlabel("Actual AQI")
    axis.set_ylabel("Predicted AQI")
    axis.legend(frameon=False)


def _plot_error_distribution(
    predictions: pd.DataFrame,
    axis: Axes,
) -> None:
    errors = (
        predictions.Predicted_AQI_Tomorrow
        - predictions.Actual_AQI_Tomorrow
    )
    sns.histplot(errors, bins=45, kde=True, ax=axis, color="#5b9b85")
    axis.axvline(0, color="#c44e52", linestyle="--", linewidth=1.3)
    axis.set_title("Prediction error distribution")
    axis.set_xlabel("Predicted - actual AQI")
    axis.set_ylabel("Test records")


def _plot_confusion_matrix(
    confusion: pd.DataFrame,
    axis: Axes,
) -> None:
    sns.heatmap(
        confusion,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        linewidths=0.4,
        ax=axis,
        annot_kws={"size": 8},
    )
    axis.set_title("AQI category confusion matrix")
    axis.set_xlabel("Predicted category")
    axis.set_ylabel("Actual category")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)


def _plot_feature_importance(
    importance: pd.DataFrame,
    axis: Axes,
) -> None:
    axis.barh(
        importance.Feature,
        importance.Importance,
        color="#547f9f",
    )
    axis.set_title("Top 10 feature importances")
    axis.set_xlabel("Importance")
    axis.set_ylabel("")
    axis.tick_params(axis="y", labelsize=8)


def _plot_metric_table(metrics: pd.DataFrame, axis: Axes) -> None:
    test_metrics = metrics[
        metrics.Evaluation_Period.str.startswith("test")
    ].copy()
    shown = test_metrics[["Model_Name", "MAE", "RMSE", "R2"]].rename(
        columns={"Model_Name": "Model", "R2": "R-squared"}
    )
    shown = shown.round({"MAE": 2, "RMSE": 2, "R-squared": 3})

    axis.axis("off")
    period = test_metrics.Evaluation_Period.iloc[0].removeprefix("test ")
    table = axis.table(
        cellText=shown.values,
        colLabels=shown.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.40, 0.18, 0.20, 0.20],
        bbox=[0.01, 0.19, 0.98, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="#ffffff")
            cell.set_facecolor("#365f7d")
        else:
            cell.set_facecolor("#f2f5f7" if row % 2 == 0 else "#ffffff")
    axis.set_title(
        f"Held-out test metrics\n{period}",
        pad=14,
    )
