# AirAware AI

MCA final-year project for historical air-quality analysis, next-observation AQI forecasting, pollution risk summaries, and Tableau dashboards.

## Dataset and current inspection

The supplied `city_day.csv` is the Kaggle Air Quality Data in India file. Inspection found 29,531 rows, 16 columns, 26 cities, dates 2015-01-01 through 2020-07-01, and no duplicate City-Date records. AQI is missing in 4,681 rows; AQI_Bucket is also missing in those rows. The project does not manufacture or fill target values. `city_day.csv` at the project root is copied to `data/raw/city_day.csv` on first run. If absent, download the primary file from [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india).

Records are not available for every city on every calendar day. Therefore, “tomorrow” means the next available dated observation for that city. This is recorded in prediction outputs and is a limitation of this dataset for strict calendar-day forecasting.

## Windows setup

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

To inspect and run only the model-training workflow, with detailed console output for missing values, target-row exclusions, chronological splits, imputation, fitting and evaluation, run:

```powershell
python model_training_walkthrough.py
```
See [`docs/model_training_walkthrough.md`](docs/model_training_walkthrough.md) for the code map and explanation of each stage.

Windows CMD:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Put the Kaggle `city_day.csv` at `data/raw/city_day.csv` or leave the supplied copy in the project root. The pipeline validates columns, writes cleaned data, EDA figures and missingness reports, builds city-grouped historical features, compares persistence context (today's AQI), Random Forest and Gradient Boosting, selects by validation MAE, evaluates once on a later chronological test period, saves the model, and exports Tableau data. No random row split is used. Pollutant and AQI feature imputers are fit only on training rows.

## Outputs

- `data/processed/city_day_clean.csv`: parsed, deduplicated observations.
- `outputs/reports/missing_values.csv`, `data_summary.csv`: inspection outputs.
- `outputs/figures/`: AQI, monthly, city and missingness charts.
- `outputs/figures/model_results_dashboard.png`: one figure with held-out actual/predicted AQI, prediction parity, error distribution, category confusion matrix, feature importance, and validation/test metric comparisons.
- Run `python show_model_results.py` in VS Code to regenerate that figure from the existing reports and open it with `plt.show()`.
- `outputs/reports/model_comparison.csv`: validation model results plus final test result.
- `outputs/reports/test_predictions.csv`: held-out test predictions.
- `outputs/reports/category_confusion_matrix.csv` and `category_classification_report.csv`.
- `models/aqi_forecaster.joblib`: fitted preprocessing and model pipeline.
- `data/tableau/`: historical, prediction, city summary and model metrics CSVs.
- `tableau/dashboard_guide.md`: worksheet and dashboard instructions.
- `docs/project_report.md`, `docs/viva_questions.md`: submission drafts.

## Forecast interpretation

AQI category uses CPCB project ranges: Good 0–50, Satisfactory 51–100, Moderate 101–200, Poor 201–300, Very Poor 301–400, Severe >400. Model classification is derived from regression predictions, not an independently fitted classifier. Feature importance indicates predictive association/usefulness and does not establish causation or health outcomes. There is no weather forecast input. Model metrics in the reports are generated from this dataset; refer to those files instead of assuming a target accuracy.

## Tableau

Connect Tableau to the four CSVs under `data/tableau/` as separate text-file data sources. Follow `tableau/dashboard_guide.md`. This version uses exported CSVs; it does not connect to a live prediction API.
