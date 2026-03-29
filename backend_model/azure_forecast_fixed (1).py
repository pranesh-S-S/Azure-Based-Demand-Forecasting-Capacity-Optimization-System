import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import joblib
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ===============================
# FIX 1 — SPLIT DATE DEFINED FIRST
# All statistics (medians, IQR bounds) must be derived from
# train rows only. Split date must exist before any .median() call.
# ===============================

split_date = '2024-07-01'

df = pd.read_csv("data/azure_dataset_missing_values.csv")
print(df)

print(df[['usage_units', 'cost_usd']].corr())

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')

# ---- TIME FEATURES (MANDATORY FOR FORECASTING) ----
df['year']        = df['timestamp'].dt.year
df['month']       = df['timestamp'].dt.month
df['day']         = df['timestamp'].dt.day
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['quarter']     = df['timestamp'].dt.quarter

print(df)

df['region'] = df['region'].str.lower().str.replace(" ", "-")
print(df)

numeric_cols = [
    'usage_units', 'provisioned_capacity', 'cost_usd',
    'availability_pct', 'economic_index', 'market_demand_index'
]

# Boxplot (shows outliers clearly)
plt.figure(figsize=(14, 6))
df[numeric_cols].boxplot()
plt.title("Before Preprocessing - Boxplot")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histograms (distribution & skewness)
df[numeric_cols].hist(bins=30, figsize=(14, 10))
plt.suptitle("Before Preprocessing - Distribution", fontsize=16)
plt.tight_layout()
plt.show()

df.info()

print(df.duplicated())
print(df.duplicated().sum())

df = df.drop_duplicates()

print(df.columns)
print(df["usage_units"].skew())

columns = [
    'usage_units', 'provisioned_capacity', 'cost_usd',
    'availability_pct', 'economic_index', 'market_demand_index'
]

print(df.isnull().sum())

# ===============================
# FIX 2 — IMPUTATION USING TRAIN MEDIANS ONLY
# Original code called .median() on the full df which includes
# test rows (Jul–Dec 2024). Medians must come from train rows only.
# ===============================

train_mask = df['timestamp'] < split_date

# Compute medians from TRAIN rows only
medians = df.loc[train_mask, numeric_cols].median()

# Fill nulls using train-derived medians
df[numeric_cols] = df[numeric_cols].fillna(medians)

# Calculate pricing rate from TRAIN valid rows only
valid_rows = df[train_mask & df['cost_usd'].notnull() & df['usage_units'].notnull()]
rate = (valid_rows['cost_usd'] / valid_rows['usage_units']).median()
print("Estimated pricing rate:", rate)

# Fill remaining cost_usd nulls using derived rate
df['cost_usd'] = df['cost_usd'].fillna(df['usage_units'] * rate)

print(df.isnull().sum())

# ===============================
# FIX 3 — IQR CLIPPING USING TRAIN BOUNDS ONLY
# Original code computed Q1/Q3 on the full df.
# Bounds must be derived from train rows and saved for production use.
# capacity_utilization must be computed AFTER clipping, not before.
# ===============================

clip_bounds = {}

for col in columns:
    Q1 = df.loc[train_mask, col].quantile(0.25)   # train only
    Q3 = df.loc[train_mask, col].quantile(0.75)   # train only
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] >= upper_bound) | (df[col] <= lower_bound)]
    print(len(outliers))
    clip_bounds[col] = (lower_bound, upper_bound)  # save for production

for col in columns:
    lower_bound, upper_bound = clip_bounds[col]
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

for col in columns:
    lower_bound, upper_bound = clip_bounds[col]
    outliers = df[(df[col] > upper_bound) | (df[col] < lower_bound)]
    print(len(outliers))

# FIX — capacity_utilization computed AFTER clipping (was before in original)
df['capacity_utilization'] = df['usage_units'] / df['provisioned_capacity']

print("Rows before lag:", len(df))

# ---- LAG FEATURES (CRITICAL FOR TIME SERIES) ----

df = df.sort_values(['region', 'service_type', 'timestamp'])

df['lag_1']  = df.groupby(['region', 'service_type'])['usage_units'].shift(1)
df['lag_2']  = df.groupby(['region', 'service_type'])['usage_units'].shift(2)
df['lag_3']  = df.groupby(['region', 'service_type'])['usage_units'].shift(3)
df['lag_7']  = df.groupby(['region', 'service_type'])['usage_units'].shift(7)
df['lag_14'] = df.groupby(['region', 'service_type'])['usage_units'].shift(14)
df['lag_30'] = df.groupby(['region', 'service_type'])['usage_units'].shift(30)

# Weekly momentum (short-term trend signal)
df['usage_trend_7'] = df['lag_1'] - df['lag_7']

# ---- ROLLING FEATURES ----

df['rolling_mean_7'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
)

df['rolling_std_7'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())
)

df['rolling_max_7'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).rolling(7, min_periods=1).max())
)

df['rolling_min_7'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).rolling(7, min_periods=1).min())
)

df['rolling_mean_14'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
)

df['rolling_std_14'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).rolling(14, min_periods=1).std())
)

df['rolling_mean_30'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
)

df['rolling_std_30'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).rolling(30, min_periods=1).std())
)

# ---- MOMENTUM RATIOS ----
epsilon = 1e-6
df['momentum_3_7'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
) / (df['rolling_mean_7'] + epsilon)

df['momentum_7_30'] = df['rolling_mean_7'] / (df['rolling_mean_30'] + epsilon)

# ---- FOURIER SEASONALITY TERMS ----
df['fourier_weekly_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['fourier_weekly_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['fourier_monthly_sin'] = np.sin(2 * np.pi * df['day'] / 30.44)
df['fourier_monthly_cos'] = np.cos(2 * np.pi * df['day'] / 30.44)

# Drop rows created by lag warm-up period
df = df.dropna(subset=['lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_std_7'])

print("Rows after lag:", len(df))

# BUSINESS LOGIC VALIDATION

print("\n--- BUSINESS RULE VALIDATION ---")

print("Negative usage values:",      (df['usage_units'] < 0).sum())
print("Availability above 100%:",    (df['availability_pct'] > 100).sum())
print("Usage > Provisioned Capacity:", (df['usage_units'] > df['provisioned_capacity']).sum())

print("\nSample Over-Capacity Rows:")
print(df[df['usage_units'] > df['provisioned_capacity']][
    ['timestamp', 'region', 'service_type', 'usage_units', 'provisioned_capacity']
].head())

df['over_capacity_flag'] = (df['usage_units'] > df['provisioned_capacity']).astype(int)

print("Economic index outside realistic range:",
      ((df['economic_index'] < 80) | (df['economic_index'] > 120)).sum())

# ---------------------------------
# Feature Engineering (Production-Level)
# ---------------------------------

df['usage_spike_flag'] = (
    df['usage_units'] > df['rolling_mean_7'] + 2 * df['rolling_std_7']
).astype(int)

df = df.dropna(subset=['rolling_std_7'])
df = df.reset_index(drop=True)

# Seasonality flags
df['is_weekend']    = (df['day_of_week'] >= 5).astype(int)
df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
df['is_month_end']   = df['timestamp'].dt.is_month_end.astype(int)

# ---- EWMA (Exponentially Weighted Moving Average) ----
# Captures recent trends with exponential decay — critical for production accuracy
df['ewma_7'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).ewm(span=7, min_periods=1).mean())
)
df['ewma_14'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).ewm(span=14, min_periods=1).mean())
)
df['ewma_30'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).ewm(span=30, min_periods=1).mean())
)

# ---- INTERACTION FEATURES ----
# Cross-domain signals the model can't learn from individual features alone
df['usage_x_economic']   = df['usage_units'] * df['economic_index']
df['capacity_x_demand']  = df['provisioned_capacity'] * df['market_demand_index']
df['util_x_availability'] = df['capacity_utilization'] * df['availability_pct']

# ---- GROUP-LEVEL TARGET ENCODING (train-only) ----
# Historical average anchoring per (region, service_type) — prevents cold-start drift
train_mask_fe = df['timestamp'] < split_date
group_means = df.loc[train_mask_fe].groupby(['region', 'service_type'])['usage_units'].mean()
group_stats = group_means.to_dict()
df['group_historical_avg'] = df.apply(
    lambda row: group_stats.get((row['region'], row['service_type']), df.loc[train_mask_fe, 'usage_units'].mean()),
    axis=1
)

# Deviation from group historical average
df['deviation_from_group'] = df['lag_1'] - df['group_historical_avg']

# ---- DAILY GROWTH RATE ----
df['daily_growth'] = df.groupby(['region', 'service_type'])['usage_units'].transform(lambda x: x.pct_change())
df['daily_growth'] = df['daily_growth'].fillna(0).clip(-1, 1)  # clip extreme growth rates

# Time granularity validation
print("Date range:", df['timestamp'].min(), "to", df['timestamp'].max())
date_range = pd.date_range(df['timestamp'].min(), df['timestamp'].max(), freq='D')
print("Expected number of days:", len(date_range))
print("Actual unique timestamps:", df['timestamp'].nunique())
print("Missing dates:", len(date_range) - df['timestamp'].nunique())

# ---- ENCODING CATEGORICAL VARIABLES ----

df = pd.get_dummies(df, columns=['region', 'service_type'], drop_first=True)

# ---- Core Continuous Variables ----
core_vars = [
    'usage_units', 'provisioned_capacity', 'cost_usd',
    'availability_pct', 'economic_index', 'market_demand_index',
    'capacity_utilization', 'lag_1', 'lag_7', 'lag_14',
    'rolling_mean_7', 'rolling_std_7'
]

plt.figure(figsize=(14, 6))
df[core_vars].boxplot()
plt.title("After Preprocessing - Core Variables Boxplot")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df[core_vars].hist(bins=30, figsize=(16, 12))
plt.suptitle("After Preprocessing - Core Variables Distribution", fontsize=16)
plt.tight_layout()
plt.show()

# ---- TIME FEATURE DISTRIBUTION ----

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.countplot(x='month', data=df)

plt.subplot(2, 2, 2)
sns.countplot(x='day_of_week', data=df)

plt.subplot(2, 2, 3)
sns.countplot(x='quarter', data=df)

plt.subplot(2, 2, 4)
sns.countplot(x='year', data=df)

plt.tight_layout()
plt.show()

# ----------------- FINAL SHAPE OF THE DATASET -----------------
print("\nFinal dataset shape:", df.shape)

df.to_csv("final_cleaned_azure_data.csv", index=False)

# ----------------- PREPARE FOR MODELING -----------------

target   = 'usage_units'
features = df.drop(columns=['usage_units', 'timestamp'])

X = features
y = df[target]

# ---- SPLIT INTO TRAIN/TEST BASED ON TIME ----
# split_date already defined at top of script

train = df[df['timestamp'] < split_date]
test  = df[df['timestamp'] >= split_date]

X_train = train.drop(['usage_units', 'timestamp'], axis=1)
y_train = train['usage_units']

X_test = test.drop(['usage_units', 'timestamp'], axis=1)
y_test = test['usage_units']

print("Training set shape:", X_train.shape)

# ===============================
# BACKTESTING USING TIMESERIES SPLIT
# ===============================

# FIX 4 — backtesting must run on X_train/y_train only.
# Original used full X which included test rows, letting future
# data leak into CV folds.

tscv = TimeSeriesSplit(n_splits=5)

print("\nBacktesting RMSE values:")

for train_index, test_index in tscv.split(X_train):   # X_train only

    X_train_bt, X_test_bt = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_bt, y_test_bt = y_train.iloc[train_index], y_train.iloc[test_index]

    model_bt = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6
    )

    model_bt.fit(X_train_bt, y_train_bt)

    pred_bt   = model_bt.predict(X_test_bt)
    rmse_bt   = np.sqrt(mean_squared_error(y_test_bt, pred_bt))

    print("Backtest RMSE:", round(rmse_bt, 4))

# ===============================
# XGBOOST MODEL (Production Baseline — Two-Stage Training)
# ===============================

# Stage 1: Find optimal tree count via early stopping on 85/15 split
val_split = int(len(X_train) * 0.85)
X_tr_early, X_val_early = X_train.iloc[:val_split], X_train.iloc[val_split:]
y_tr_early, y_val_early = y_train.iloc[:val_split], y_train.iloc[val_split:]

xgb_probe = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=5,
    early_stopping_rounds=50,
    random_state=42
)

xgb_probe.fit(
    X_tr_early, y_tr_early,
    eval_set=[(X_val_early, y_val_early)],
    verbose=False
)

optimal_trees_baseline = xgb_probe.best_iteration
print(f"Baseline: Early stopping found optimal trees = {optimal_trees_baseline}")

# Stage 2: Retrain on FULL training data with optimal tree count (eliminates bias)
xgb_model = XGBRegressor(
    n_estimators=optimal_trees_baseline,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=5,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

# Clip predictions to physical valid range (non-negative, within observed bounds)
xgb_pred = np.clip(xgb_pred, 0, y_train.max() * 1.5)

xgb_mae  = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

# FIX 5 — safe MAPE: original divides by y_test directly which
# causes inf/nan if any value is zero.
xgb_mape = np.mean(np.abs((y_test - xgb_pred) / np.where(y_test == 0, np.nan, y_test))) * 100

# NRMSE — normalized by range of actual values
y_test_range = y_test.max() - y_test.min()
xgb_nrmse = xgb_rmse / y_test_range if y_test_range > 0 else 0.0

# --- ACCURACY METRICS (Baseline XGBoost) ---
xgb_r2        = r2_score(y_test, xgb_pred)
xgb_accuracy  = max(0.0, 100.0 - xgb_mape)  # 100 - MAPE

# Directional Accuracy (Did the model predict the correct move up/down?)
actual_diff = np.diff(y_test)
pred_diff   = np.diff(xgb_pred)
if len(actual_diff) > 0:
    xgb_dir_acc = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100
else:
    xgb_dir_acc = 0.0

print("\n===== XGBOOST RESULTS =====")
print("MAE  :", xgb_mae)
print("RMSE :", xgb_rmse)
print(f"NRMSE: {xgb_nrmse:.4f}")
print("MAPE :", xgb_mape)
print(f"R² Score     : {xgb_r2:.4f}")
print(f"Accuracy (%) : {xgb_accuracy:.2f}%")
print(f"Directional Acc (%): {xgb_dir_acc:.2f}%")

xgb_bias_baseline = np.mean(xgb_pred - y_test)
print("Forecast Bias (XGBoost):", xgb_bias_baseline)

# ===============================
# XGBOOST HYPERPARAMETER TUNING (Production-Level RandomizedSearchCV)
# ===============================

param_distributions = {
    'n_estimators':      [500, 700, 1000, 1500],
    'max_depth':         [4, 5, 6, 7, 8],
    'learning_rate':     [0.005, 0.01, 0.03, 0.05],
    'subsample':         [0.7, 0.8, 0.85, 0.9],
    'colsample_bytree':  [0.6, 0.7, 0.8, 0.9],
    'min_child_weight':  [1, 3, 5, 7],
    'reg_alpha':         [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda':        [1, 3, 5, 10, 15],
    'gamma':             [0, 0.1, 0.3, 0.5],
}

# FIX 6 — TimeSeriesSplit ensures validation is always
# chronologically after training, matching real deployment.
tscv_gs = TimeSeriesSplit(n_splits=5)

random_search = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,            # 50 random combos for fast-yet-thorough search
    scoring='neg_mean_squared_error',
    cv=tscv_gs,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

best_xgb_model = random_search.best_estimator_

print("\nBest XGBoost Parameters:", random_search.best_params_)

# Stage 1: Find optimal tree count via early stopping
best_params_probe = random_search.best_params_.copy()
best_params_probe['n_estimators'] = 2000
best_params_probe['random_state'] = 42
best_params_probe['early_stopping_rounds'] = 50

probe_model = XGBRegressor(**best_params_probe)
probe_model.fit(
    X_tr_early, y_tr_early,
    eval_set=[(X_val_early, y_val_early)],
    verbose=False
)

optimal_trees_tuned = probe_model.best_iteration
print(f"Tuned: Early stopping found optimal trees = {optimal_trees_tuned}")

# Stage 2: Retrain on FULL training data with optimal tree count (eliminates bias)
best_params_final = random_search.best_params_.copy()
best_params_final['n_estimators'] = optimal_trees_tuned
best_params_final['random_state'] = 42

best_xgb_model = XGBRegressor(**best_params_final)
best_xgb_model.fit(X_train, y_train)

xgb_tuned_pred = best_xgb_model.predict(X_test)

# Clip predictions to physical valid range
xgb_tuned_pred = np.clip(xgb_tuned_pred, 0, y_train.max() * 1.5)

xgb_tuned_mae  = mean_absolute_error(y_test, xgb_tuned_pred)
xgb_tuned_rmse = np.sqrt(mean_squared_error(y_test, xgb_tuned_pred))
xgb_tuned_mape = np.mean(np.abs((y_test - xgb_tuned_pred) / np.where(y_test == 0, np.nan, y_test))) * 100

# NRMSE for Tuned XGBoost
xgb_tuned_nrmse = xgb_tuned_rmse / y_test_range if y_test_range > 0 else 0.0

# --- ACCURACY METRICS (Tuned XGBoost) ---
xgb_tuned_r2       = r2_score(y_test, xgb_tuned_pred)
xgb_tuned_accuracy = max(0.0, 100.0 - xgb_tuned_mape)

# Directional Accuracy for Tuned XGBoost
actual_diff_tuned = np.diff(y_test)
pred_diff_tuned   = np.diff(xgb_tuned_pred)
if len(actual_diff_tuned) > 0:
    xgb_tuned_dir_acc = np.mean(np.sign(actual_diff_tuned) == np.sign(pred_diff_tuned)) * 100
else:
    xgb_tuned_dir_acc = 0.0

xgb_bias_tuned = np.mean(xgb_tuned_pred - y_test)
print("\n===== TUNED XGBOOST RESULTS =====")
print("Forecast Bias (Tuned XGBoost):", xgb_bias_tuned)
print(f"RMSE (Tuned)  : {xgb_tuned_rmse:.4f}")
print(f"NRMSE (Tuned) : {xgb_tuned_nrmse:.4f}")
print(f"R² Score (Tuned)     : {xgb_tuned_r2:.4f}")
print(f"Accuracy (Tuned) (%) : {xgb_tuned_accuracy:.2f}%")
print(f"Directional Acc (%): {xgb_tuned_dir_acc:.2f}%")

# ===============================
# ARIMA MODEL (AGGREGATED SERIES)
# ===============================

ts_data = df.groupby('timestamp')['usage_units'].sum()

print("Total time series points:", len(ts_data))

train_ts = ts_data[:'2024-06-30']
test_ts  = ts_data['2024-07-01':]

print("ARIMA training points:", len(train_ts))
print("ARIMA testing points:",  len(test_ts))

# FIX 7 — added ADF stationarity test so differencing order d
# is validated by data rather than assumed.
adf_stat, p_value, *_ = adfuller(train_ts.dropna())
print(f"\nADF statistic: {adf_stat:.4f}  |  p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Series is stationary — d=0 or 1 is appropriate")
else:
    print("Series is non-stationary — use d >= 1")

arima_model = ARIMA(train_ts, order=(5, 1, 2))
arima_fit   = arima_model.fit()

print(arima_fit.summary())

arima_pred = arima_fit.forecast(steps=len(test_ts))
arima_pred = pd.Series(arima_pred, index=test_ts.index)

arima_mae  = mean_absolute_error(test_ts, arima_pred)
arima_rmse = np.sqrt(mean_squared_error(test_ts, arima_pred))
arima_mape = np.mean(np.abs((test_ts - arima_pred) / np.where(test_ts == 0, np.nan, test_ts))) * 100

# NRMSE for Baseline ARIMA
arima_ts_range = test_ts.max() - test_ts.min()
arima_nrmse = arima_rmse / arima_ts_range if arima_ts_range > 0 else 0.0

# --- ACCURACY METRICS (Baseline ARIMA) ---
arima_r2       = r2_score(test_ts, arima_pred)
arima_accuracy = max(0.0, 100.0 - arima_mape)

# Directional Accuracy for ARIMA (aggregated usage)
actual_diff_arima = np.diff(test_ts.values)
pred_diff_arima   = np.diff(arima_pred.values)
if len(actual_diff_arima) > 0:
    arima_dir_acc = np.mean(np.sign(actual_diff_arima) == np.sign(pred_diff_arima)) * 100
else:
    arima_dir_acc = 0.0

print("\n===== ARIMA RESULTS =====")
print("MAE  :", arima_mae)
print("RMSE :", arima_rmse)
print(f"NRMSE: {arima_nrmse:.4f}")
print("MAPE :", arima_mape)
print(f"R² Score     : {arima_r2:.4f}")
print(f"Accuracy (%) : {arima_accuracy:.2f}%")
print(f"Directional Acc (%): {arima_dir_acc:.2f}%")

arima_bias = np.mean(arima_pred - test_ts)
print("Forecast Bias (ARIMA):", arima_bias)

arima_results = pd.DataFrame({
    "timestamp":        test_ts.index,
    "Actual_Usage":     test_ts.values,
    "ARIMA_Prediction": arima_pred.values
})

print("\nARIMA Prediction Table")
print(arima_results.head(20))

# ===============================
# ARIMA HYPERPARAMETER TUNING
# ===============================

best_aic   = float("inf")
best_order = None

for p in range(0, 4):
    for d in range(0, 2):
        for q in range(0, 4):
            try:
                model   = ARIMA(train_ts, order=(p, d, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic   = results.aic
                    best_order = (p, d, q)
            except Exception:
                continue

print("Best ARIMA order:", best_order)

arima_tuned     = ARIMA(train_ts, order=best_order)
arima_tuned_fit = arima_tuned.fit()

arima_tuned_pred = arima_tuned_fit.forecast(steps=len(test_ts))
arima_tuned_pred = pd.Series(arima_tuned_pred, index=test_ts.index)

arima_tuned_mae  = mean_absolute_error(test_ts, arima_tuned_pred)
arima_tuned_rmse = np.sqrt(mean_squared_error(test_ts, arima_tuned_pred))
arima_tuned_mape = np.mean(np.abs((test_ts - arima_tuned_pred) / np.where(test_ts == 0, np.nan, test_ts))) * 100

# NRMSE for Tuned ARIMA
arima_tuned_nrmse = arima_tuned_rmse / arima_ts_range if arima_ts_range > 0 else 0.0

# --- ACCURACY METRICS (Tuned ARIMA) ---
arima_tuned_r2       = r2_score(test_ts, arima_tuned_pred)
arima_tuned_accuracy = max(0.0, 100.0 - arima_tuned_mape)

actual_diff_arima_t = np.diff(test_ts.values)
pred_diff_arima_t   = np.diff(arima_tuned_pred.values)
if len(actual_diff_arima_t) > 0:
    arima_tuned_dir_acc = np.mean(np.sign(actual_diff_arima_t) == np.sign(pred_diff_arima_t)) * 100
else:
    arima_tuned_dir_acc = 0.0

arima_bias_tuned = np.mean(arima_tuned_pred - test_ts)

# ===============================
# Create results dataframe
# ===============================

results_df = test[['timestamp', 'usage_units']].copy()
results_df['XGBoost_Prediction'] = xgb_pred

arima_df = pd.DataFrame({
    'timestamp':         test_ts.index,
    'Actual_ARIMA_Usage': test_ts.values,
    'ARIMA_Prediction':  arima_pred.values
})

results_df = results_df.merge(arima_df, on='timestamp', how='left')

results_df = results_df[[
    'timestamp', 'usage_units',
    'XGBoost_Prediction', 'Actual_ARIMA_Usage', 'ARIMA_Prediction'
]]

print(results_df.head(20))

# ===============================
# MODEL VALIDATION RESULTS
# ===============================

validation_df = pd.DataFrame({
    "Model": [
        "Baseline XGBoost",
        "Tuned XGBoost",
        "Baseline ARIMA",
        "Tuned ARIMA"
    ],
    "MAE": [
        xgb_mae,
        xgb_tuned_mae,
        arima_mae,
        arima_tuned_mae
    ],
    "RMSE": [
        xgb_rmse,
        xgb_tuned_rmse,
        arima_rmse,
        arima_tuned_rmse
    ],
    "NRMSE": [
        xgb_nrmse,
        xgb_tuned_nrmse,
        arima_nrmse,
        arima_tuned_nrmse
    ],
    "MAPE (%)": [
        xgb_mape,
        xgb_tuned_mape,
        arima_mape,
        arima_tuned_mape
    ],
    "Bias": [
        xgb_bias_baseline,
        xgb_bias_tuned,
        arima_bias,
        arima_bias_tuned
    ],
    "R² Score": [
        xgb_r2,
        xgb_tuned_r2,
        arima_r2,
        arima_tuned_r2
    ],
    "Accuracy (%)": [
        xgb_accuracy,
        xgb_tuned_accuracy,
        arima_accuracy,
        arima_tuned_accuracy
    ],
    "Directional Acc (%)": [
        xgb_dir_acc,
        xgb_tuned_dir_acc,
        arima_dir_acc,
        arima_tuned_dir_acc
    ]
})

validation_df = validation_df.set_index("Model")
validation_df = validation_df.round(4)

print("\nMODEL VALIDATION RESULTS")
print(validation_df)

# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_distributions(df):
    core_vars = [
        'usage_units', 'provisioned_capacity', 'cost_usd',
        'availability_pct', 'economic_index', 'market_demand_index',
        'capacity_utilization', 'lag_1', 'lag_7', 'lag_14',
        'rolling_mean_7', 'rolling_std_7'
    ]
    df[core_vars].hist(bins=30, figsize=(14, 10))
    plt.suptitle("Feature Distributions", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_xgb_forecast(test, y_test, xgb_pred, xgb_tuned_pred):
    plt.figure(figsize=(14, 5))
    plt.plot(test["timestamp"], y_test,        label="Actual Usage",      linewidth=2)
    plt.plot(test["timestamp"], xgb_pred,      label="Baseline XGBoost",  linestyle="--")
    plt.plot(test["timestamp"], xgb_tuned_pred, label="Tuned XGBoost",    linestyle="--")
    plt.title("XGBoost Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Usage Units")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_arima_forecast(test_ts, arima_pred, arima_tuned_pred):
    plt.figure(figsize=(14, 5))
    plt.plot(test_ts.index, test_ts.values,       label="Actual Usage",    linewidth=2)
    plt.plot(test_ts.index, arima_pred,            label="Baseline ARIMA", linestyle="--")
    plt.plot(test_ts.index, arima_tuned_pred,      label="Tuned ARIMA",    linestyle="--")
    plt.title("ARIMA Forecast vs Actual (Aggregated)")
    plt.xlabel("Date")
    plt.ylabel("Total Usage Units")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.nlargest(top_n).sort_values()
    plt.figure(figsize=(10, 6))
    importance.plot(kind="barh")
    plt.title("Top Feature Importances (XGBoost)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()


# ===============================
# CALL VISUALIZATION FUNCTIONS
# ===============================

plot_distributions(df)
plot_xgb_forecast(test, y_test, xgb_pred, xgb_tuned_pred)
plot_arima_forecast(test_ts, arima_pred, arima_tuned_pred)
plot_feature_importance(best_xgb_model, X_train.columns)

# ===============================
# BEST MODEL SELECTION
# ===============================

best_model_name = validation_df["RMSE"].idxmin()
best_rmse       = validation_df.loc[best_model_name, "RMSE"]

print(f"\nBest Model: {best_model_name} (RMSE = {best_rmse:.4f})")

# ===============================
# SAVE BEST MODEL + ARTIFACTS
# FIX 8 — also save clip_bounds and medians so the exact same
# preprocessing can be reproduced at inference time.
# ===============================

import os
os.makedirs("models", exist_ok=True)
joblib.dump(best_xgb_model,          "models/best_xgboost_model.pkl")
joblib.dump(X_train.columns.tolist(), "models/model_features.pkl")
joblib.dump(medians.to_dict(),        "models/imputation_medians.pkl")
joblib.dump(clip_bounds,              "models/clip_bounds.pkl")
joblib.dump(group_stats,              "models/group_stats.pkl")

print("Best model saved as best_xgboost_model.pkl")
print("Imputation medians saved as imputation_medians.pkl")
print("Clip bounds saved as clip_bounds.pkl")
print("Group stats saved as group_stats.pkl")
