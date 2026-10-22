"""Build the model-results figure and open it in a plotting window."""

from pathlib import Path

from src.model_visualization import plot_model_dashboard


ROOT = Path(__file__).resolve().parent


def main() -> None:
    figure_path = plot_model_dashboard(
        reports_dir=ROOT / "outputs" / "reports",
        output_path=ROOT / "outputs" / "figures" / "model_results_dashboard.png",
        show=True,
    )
    print(f"Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
