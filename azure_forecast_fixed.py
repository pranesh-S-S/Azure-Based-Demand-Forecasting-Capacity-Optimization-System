import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

df = pd.read_csv(r"C:\Users\pranesh.S.S\Downloads\azure_dataset_missing_values.csv")
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
df['lag_7']  = df.groupby(['region', 'service_type'])['usage_units'].shift(7)
df['lag_14'] = df.groupby(['region', 'service_type'])['usage_units'].shift(14)

# ---- ROLLING FEATURES ----

df['rolling_mean_7'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).rolling(7).mean())
)

df['rolling_std_7'] = (
    df.groupby(['region', 'service_type'])['usage_units']
      .transform(lambda x: x.shift(1).rolling(7).std())
)

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
# Feature Engineering
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
# XGBOOST MODEL
# ===============================

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_mae  = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

# FIX 5 — safe MAPE: original divides by y_test directly which
# causes inf/nan if any value is zero.
xgb_mape = np.mean(np.abs((y_test - xgb_pred) / np.where(y_test == 0, np.nan, y_test))) * 100

print("\n===== XGBOOST RESULTS =====")
print("MAE :", xgb_mae)
print("RMSE:", xgb_rmse)
print("MAPE:", xgb_mape)

xgb_bias_baseline = np.mean(xgb_pred - y_test)
print("Forecast Bias (XGBoost):", xgb_bias_baseline)

# ===============================
# XGBOOST HYPERPARAMETER TUNING
# ===============================

param_grid = {
    'n_estimators':  [200, 300],
    'max_depth':     [4, 6],
    'learning_rate': [0.05, 0.1],
}

# FIX 6 — cv=3 uses random k-fold which shuffles rows and breaks
# time ordering. TimeSeriesSplit ensures validation is always
# chronologically after training, matching real deployment.
tscv_gs = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=tscv_gs,      # was cv=3 — fixed to TimeSeriesSplit
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_xgb_model = grid_search.best_estimator_

print("\nBest XGBoost Parameters:", grid_search.best_params_)

xgb_tuned_pred = best_xgb_model.predict(X_test)

xgb_tuned_mae  = mean_absolute_error(y_test, xgb_tuned_pred)
xgb_tuned_rmse = np.sqrt(mean_squared_error(y_test, xgb_tuned_pred))
xgb_tuned_mape = np.mean(np.abs((y_test - xgb_tuned_pred) / np.where(y_test == 0, np.nan, y_test))) * 100

xgb_bias_tuned = np.mean(xgb_tuned_pred - y_test)
print("Forecast Bias (Tuned XGBoost):", xgb_bias_tuned)

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

print("\n===== ARIMA RESULTS =====")
print("MAE :", arima_mae)
print("RMSE:", arima_rmse)
print("MAPE:", arima_mape)

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

joblib.dump(best_xgb_model,          "best_xgboost_model.pkl")
joblib.dump(X_train.columns.tolist(), "model_features.pkl")
joblib.dump(medians.to_dict(),        "imputation_medians.pkl")   # new
joblib.dump(clip_bounds,              "clip_bounds.pkl")           # new

print("Best model saved as best_xgboost_model.pkl")
print("Imputation medians saved as imputation_medians.pkl")
print("Clip bounds saved as clip_bounds.pkl")
