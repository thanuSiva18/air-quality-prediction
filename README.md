# 💨 Air Quality Prediction Using Time Series (Prophet)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Model: Prophet](https://img.shields.io/badge/model-Prophet-orange.svg)](#modeling--evaluation)
[![Dataset rows](https://img.shields.io/badge/rows-9357-lightgrey.svg)](#dataset)
[![Status](https://img.shields.io/badge/status-Completed-green.svg)](#summary)

One-line description
A time-series forecasting project using Meta's Prophet to predict CO (Carbon Monoxide) concentration from historical air-quality measurements. The pipeline includes data preparation for Prophet, chronological train/test split, model training with explicit seasonal components, and an evaluation plan using MAE, RMSE and R².

Table of Contents
- [Summary](#summary)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preparation](#data-preparation)
- [Train / Test Split](#train--test-split)
- [Modeling & Evaluation](#modeling--evaluation)
- [Example Code (Prophet)](#example-code-prophet)
- [Reproduce / Quick Start](#reproduce--quick-start)
- [Project Structure (suggested)](#project-structure-suggested)
- [Results](#results)
- [Limitations & Next Steps](#limitations--next-steps)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

Summary
-------
This project forecasts CO concentration (target: "CO(GT)") using Prophet. The dataset contains 9,357 records. A custom DataFrame with columns `ds` (datetime) and `y` (target) was created for Prophet. A chronological 80/20 split was used (7,485 training / 1,872 testing). Prophet was configured with daily, weekly, and yearly seasonality and multiplicative seasonality mode. Model predictions for the test period are produced and evaluated with MAE, RMSE, and R².

Dataset
-------
- Rows: 9,357
- Target: `CO(GT)` (Carbon Monoxide)
- Key preprocessing step: create a DataFrame with columns:
  - `ds` — datetime (parsed to pandas.Timestamp)
  - `y` — numeric target (CO concentrations)
- Missing values: verify and handle as required (imputation or removal).

Exploratory Data Analysis (EDA)
-------------------------------
- Plot time series to inspect trends, seasonality, and possible gaps.
- Visualize autocorrelation (ACF/PACF) to inspect lag structure.
- Check distributions and outliers for `CO(GT)` and other sensors.
- Plot rolling statistics (mean, std) to identify non-stationarity.

Data Preparation
----------------
- Create Prophet-compatible DataFrame:
  - df_prophet = df[['date_col', 'CO(GT)']].rename(columns={'date_col': 'ds', 'CO(GT)': 'y'})
  - Ensure `ds` is timezone-aware or naive consistently and sorted chronologically.
- Missing timestamps:
  - If timestamps are missing, consider reindexing to a regular frequency and imputing.
- Optional regressors:
  - Add external regressors (temperature, humidity, NOx, traffic) as extra columns and register them with Prophet using model.add_regressor(...).
- Logging / transformations:
  - Apply log-transform to `y` if distribution is heavily skewed; remember to inverse-transform predictions.

Train / Test Split
------------------
- Chronological split (no shuffling). For example:
  - Train: first 80% (7,485 samples)
  - Test: last 20% (1,872 samples)
- Use the test period timestamps as the prediction horizon (i.e., predict exactly on test `ds` values rather than forecasting fixed frequency steps when irregular timestamps are present).

Modeling & Evaluation
---------------------
- Model: Prophet (from prophet import Prophet)
- Configuration used:
  - yearly_seasonality=True
  - weekly_seasonality=True
  - daily_seasonality=True
  - seasonality_mode='multiplicative'
  - Add custom seasonalities if domain knowledge indicates (e.g., rush-hour).
- Training: fit on training DataFrame (ds, y, and regressors if any).
- Prediction: predict on test timestamps (or use make_future_dataframe if timestamps are regular).
- Evaluation metrics (compute after inverse-transforming if you transformed y):
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²) — useful but interpret carefully for time series
- Visuals:
  - Plot actual vs predicted (time series plot)
  - Residual plot and histogram
  - Forecast components (trend + seasonality) from Prophet's plot_components

Example Code (Prophet)
----------------------
```python
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# load
df = pd.read_csv("data/air_quality.csv", parse_dates=['date_col'])  # replace date_col with real name

# prepare for Prophet
df_prophet = df[['date_col', 'CO(GT)']].rename(columns={'date_col': 'ds', 'CO(GT)': 'y'})
df_prophet = df_prophet.sort_values('ds').dropna(subset=['y', 'ds'])

# chronological split
n = len(df_prophet)
train_end = int(0.8 * n)
train_df = df_prophet.iloc[:train_end].copy()
test_df = df_prophet.iloc[train_end:].copy()

# initialize and fit Prophet
model = Prophet(yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                seasonality_mode='multiplicative')

# if you have external regressors:
# model.add_regressor('temperature')
# model.fit(train_df[['ds','y','temperature']])

model.fit(train_df)

# predict on test timestamps
future = test_df[['ds']].copy()   # preserve exact test timestamps
forecast = model.predict(future)

# align predictions
y_true = test_df['y'].values
y_pred = forecast['yhat'].values

# metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
```

Reproduce / Quick Start
-----------------------
1. Clone the repo:
   git clone https://github.com/your-org/your-repo.git
2. Create virtual environment and install dependencies:
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   - Example requirements: prophet, pandas, scikit-learn, matplotlib, seaborn, jupyter
3. Prepare data:
   - Place the CSV file as data/air_quality.csv or update the path in scripts/notebook.
4. Run training:
   python src/train_prophet.py
   or open notebooks/prophet_forecast.ipynb and run cells.
5. Evaluate:
   python src/evaluate_prophet.py

Project Structure (suggested)
-----------------------------
- data/
  - air_quality.csv
- notebooks/
  - eda_prophet.ipynb
  - prophet_forecast.ipynb
- src/
  - prepare_data.py
  - train_prophet.py
  - evaluate_prophet.py
  - predict_api.py
- requirements.txt
- README.md
- LICENSE

Results
-------
- Dataset: 9,357 records
- Train / Test: 7,485 / 1,872 (chronological split)
- Model: Prophet with daily/weekly/yearly multiplicative seasonality
- Model predictions were produced for the test period. Compute and record:
  - MAE: (compute and record here)
  - RMSE: (compute and record here)
  - R²: (compute and record here)
- Visualization: plot actual vs predicted and forecast components to interpret seasonality and trend.

Limitations & Next Steps
------------------------
- Limitations:
  - Prophet models additive/multiplicative seasonality and trend extrapolation; abrupt external changes (events, sensor faults) are not captured unless provided as regressors or changepoints.
  - Irregular or sparse timestamps need careful handling (reindexing and imputation).
  - R² can be misleading for non-stationary time series — prefer MAE/RMSE and cross-validated error.
- Next steps / improvements:
  - Add exogenous regressors (temperature, humidity, traffic) to improve predictive power.
  - Use Prophet's built-in cross-validation (prophet.diagnostics.cross_validation) to get time-series CV metrics and horizon-specific evaluation.
  - Tune changepoint_prior_scale and seasonality_prior_scale for better trend/seasonality flexibility.
  - Compare against other models (ARIMA/SARIMAX, Random Forest/GBM on lag features, LSTM).
  - Create lag / rolling statistical features and test ML models using those features.
  - Build a small API (FastAPI/Flask) to serve live predictions.
  - Containerize and schedule daily predictions with a cron/CI pipeline.

Contributing
------------
- Fork the repository and create a feature branch:
  git checkout -b feat/your-feature
- Run tests and linters locally and include reproducible steps in PR description.
- Provide notebooks with reproducible EDA and experiment logs.

License
-------
This project is provided under the MIT License — see the LICENSE file for details.

Contact
-------
Maintainer: Your Name — thanusivanallaperumal@gmail.com  

Acknowledgements
----------------
Thanks to the dataset authors and the open-source libraries used in this project (Prophet, pandas, scikit-learn, matplotlib, seaborn).
