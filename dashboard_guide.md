# Tableau dashboard guide

## Connect the exports

In Tableau Desktop or Tableau Public, choose **Connect → To a File → Text file** and open each CSV under `data/tableau/`: `tableau_historical_aqi.csv`, `tableau_predictions.csv`, `tableau_city_summary.csv`, and `tableau_model_metrics.csv`. Keep them as separate data sources initially; their row grains differ. For dates, confirm `Date` is a Date field. Historical and forecast records are distinct sources and should be titled accordingly. Refresh the CSV extract after rerunning `python main.py`.

## Dashboard 1 — National Pollution Overview

Use historical source. Create KPI sheets with `COUNT(AQI)` (valid AQI observations), `COUNTD(City)`, `AVG(AQI)`, and `MAX(AQI)`. Create category distribution with `AQI_Bucket` on Rows and `COUNT(AQI)` on Columns. City ranking: City on Rows, `AVG(AQI)` on Columns, descending sort. Add Date and City filters and apply to all worksheets using this source. Label the observation count as nonmissing AQI records.

For a calculated field **Poor-or-worse observation** use `IF ROUND([AQI]) > 200 THEN 1 ELSE 0 END`; aggregate with `SUM`. For historical AQI category, prefer the supplied AQI_Bucket and retain nulls as unknown.

## Dashboard 2 — Pollution Trends

Use historical source. Daily trend: exact Date on Columns and `AVG(AQI)` on Rows. Monthly trend: use MONTH(Date), optionally YEAR(Date) as color or filter. Seasonal view: Season on Columns and `AVG(AQI)` on Rows. Pollutant comparison: pivot pollutants in Tableau if desired; otherwise create separate measures and use Measure Names/Measure Values, noting units and scales differ. City comparison: City filter plus AQI line. A map is not included because city names alone do not provide reliable coordinates; add a map only after validating geocoding or adding a documented location lookup.

## Dashboard 3 — AQI Prediction & Risk

Use `tableau_predictions.csv`. Actual vs predicted: Date on Columns and `AVG(Actual_AQI_Tomorrow)` / `AVG(Predicted_AQI_Tomorrow)` on Rows; use distinct colors and label the view **held-out test forecasts**. Prediction error: Date or City against `AVG(Absolute_Error)`. Predicted-category distribution: `Predicted_Category` and record count. High-risk cities: City rows, `AVG(Predicted_AQI_Tomorrow)` columns, filter Predicted_Category to Poor, Very Poor, Severe. Metric tiles from model metrics source: display Model_Name, MAE, RMSE, R2 and Evaluation_Period; select the test-period row for final performance.

Category rules if recreating predicted classification (round to integer first): `IF ROUND([Predicted_AQI_Tomorrow]) <= 50 THEN "Good" ELSEIF ROUND([Predicted_AQI_Tomorrow]) <= 100 THEN "Satisfactory" ELSEIF ROUND([Predicted_AQI_Tomorrow]) <= 200 THEN "Moderate" ELSEIF ROUND([Predicted_AQI_Tomorrow]) <= 300 THEN "Poor" ELSEIF ROUND([Predicted_AQI_Tomorrow]) <= 400 THEN "Very Poor" ELSE "Severe" END`.

## Layout and publishing

Use a consistent sequential palette for AQI severity (green, yellow, orange, red, purple, dark maroon), and a separate blue/orange pair for actual vs predicted lines. Use readable titles, units and tooltips. Arrange KPI cards across the top, trend/ranking views below, and filters in a side panel. Add a subtitle that distinguishes **Historical observations** from **Held-out model forecasts**. In Tableau Public, sign in, choose **Server → Tableau Public → Save to Tableau Public**, review the workbook for unintended data exposure, and publish. Refresh/re-publish after replacing the CSV exports.
