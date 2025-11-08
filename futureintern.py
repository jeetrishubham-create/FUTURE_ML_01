 # sales_forecast.py
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os
from math import sqrt

# ---------- Config ----------
INPUT_CSV = "sales.csv"          # expects 'date' and 'sales' columns
DATE_COL = "date"
TARGET_COL = "sales"
FREQ = "D"                       # 'D' daily, 'W' weekly, 'M' monthly (set appropriately)
HORIZON = 90                     # forecast horizon in days/weeks/months depending on FREQ
OUTPUT_DIR = "forecast_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Helpers ----------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # avoid division by zero
    mask = y_true != 0
    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100) if mask.any() else np.nan

# ---------- Load & prepare ----------
df = pd.read_csv(INPUT_CSV, parse_dates=[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)
df = df[[DATE_COL, TARGET_COL]].dropna()
df = df.rename(columns={DATE_COL: "ds", TARGET_COL: "y"})  # Prophet naming

# If data is not regular, resample to a regular freq filling missing with interpolation or forward fill
df = df.set_index("ds").asfreq(FREQ).reset_index()
df['y'] = df['y'].interpolate()  # linear interpolation for missing values

# ---------- Train/Test split ----------
train = df.iloc[:-HORIZON].copy()
test = df.iloc[-HORIZON:].copy()

# ---------- 1) Prophet model ----------
m = Prophet(
    daily_seasonality = True if FREQ == "D" else False,
    weekly_seasonality = True if FREQ in ["D","W"] else False,
    yearly_seasonality = True
)

# Optionally: add country holidays, e.g. m.add_country_holidays(country_name='US')
m.fit(train)

future = m.make_future_dataframe(periods=HORIZON, freq=FREQ)
forecast = m.predict(future)

# merge prophet forecast with actual
prophet_pred = forecast[['ds','yhat','yhat_lower','yhat_upper']].set_index('ds')
actual = df.set_index('ds')[['y']]
prophet_all = prophet_pred.join(actual, how='left').reset_index()

# Evaluate on test
prophet_test = prophet_all.set_index('ds').loc[test['ds']]
mae_p = mean_absolute_error(test['y'], prophet_test['yhat'])
rmse_p = sqrt(mean_squared_error(test['y'], prophet_test['yhat']))
mape_p = mape(test['y'], prophet_test['yhat'])

# save prophet model and forecast
joblib.dump(m, os.path.join(OUTPUT_DIR, "prophet_model.joblib"))
prophet_all.to_csv(os.path.join(OUTPUT_DIR, "prophet_forecast.csv"), index=False)

# ---------- 2) Gradient Boosting regression with time features ----------
# Feature engineering
full = df.copy().set_index('ds')
full['day'] = full.index.day
full['weekday'] = full.index.weekday
full['month'] = full.index.month
full['quarter'] = full.index.quarter
full['year'] = full.index.year
# Cyclical encoding for month/day/weekday
full['sin_weekday'] = np.sin(2 * np.pi * full['weekday'] / 7)
full['cos_weekday'] = np.cos(2 * np.pi * full['weekday'] / 7)
full['sin_month'] = np.sin(2 * np.pi * (full['month']-1) / 12)
full['cos_month'] = np.cos(2 * np.pi * (full['month']-1) / 12)

# lag features
for lag in [1,7,14,28]:
    full[f'lag_{lag}'] = full['y'].shift(lag)

# rolling features
full['rolling_7'] = full['y'].rolling(window=7, min_periods=1).mean()
full['rolling_30'] = full['y'].rolling(window=30, min_periods=1).mean()

full = full.dropna().copy()

# split indices
train_idx = full.index < test['ds'].min()
X = full.drop(columns=['y'])
y = full['y']

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[~train_idx], y[~train_idx]

gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
gbr.fit(X_train, y_train)

# Predict iterative forecasting for horizon (since many features are lag-based)
X_full = full.copy()
preds = []
last_known = full.loc[train.index[-1]]['y']  # not used directly, but we'll do iterative

# We'll produce predictions by iteratively appending new rows for future
X_future = X_test.copy()  # contains engineered features for future dates already
gbr_pred = gbr.predict(X_test)

mae_g = mean_absolute_error(y_test, gbr_pred)
rmse_g = sqrt(mean_squared_error(y_test, gbr_pred))
mape_g = mape(y_test, gbr_pred)

# Save model and predictions
joblib.dump(gbr, os.path.join(OUTPUT_DIR, "gbr_model.joblib"))
gbr_out = X_test.copy()
gbr_out['y_pred'] = gbr_pred
gbr_out['y_true'] = y_test
gbr_out.reset_index().to_csv(os.path.join(OUTPUT_DIR, "gbr_forecast.csv"), index=False)

# ---------- Summarize & save metrics ----------
metrics = pd.DataFrame({
    'model': ['prophet','gbr'],
    'mae': [mae_p, mae_g],
    'rmse': [rmse_p, rmse_g],
    'mape': [mape_p, mape_g]
})
metrics.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

# ---------- Plots ----------
plt.figure(figsize=(12,6))
plt.plot(df.set_index('ds')['y'], label='actual')
plt.plot(prophet_all.set_index('ds')['yhat'], label='prophet_forecast')
plt.plot(gbr_out.set_index(gbr_out.index)['y_pred'], label='gbr_forecast')
plt.axvline(df['ds'].iloc[-HORIZON], color='k', linestyle='--', alpha=0.4, label='forecast_start')
plt.legend()
plt.title('Actual vs Forecast')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_forecast.png"), dpi=150)

# Prophet components plot (saved)
fig = m.plot_components(forecast)
fig.savefig(os.path.join(OUTPUT_DIR, "prophet_components.png"))

print("Done. Outputs written to:", OUTPUT_DIR)
print(metrics)
