import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(r"C:\Users\pranesh.S.S\Downloads\azure_dataset_missing_values.csv"
                 )
print(df)


print(df[['usage_units', 'cost_usd']].corr())

df['timestamp'] = pd.to_datetime(df['timestamp']) #converts the string data into datetime objects
df = df.sort_values(by='timestamp') # sorting is mandatory for the time-series

# ---- TIME FEATURES (MANDATORY FOR FORECASTING) ----
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['quarter'] = df['timestamp'].dt.quarter


print(df)

df['region'] = df['region'].str.lower().str.replace(" ", "-")# str.lower- removes the case diff
#.replaces- standaridizes the formatting
print(df)

numeric_cols = [
    'usage_units','provisioned_capacity','cost_usd',
    'availability_pct','economic_index','market_demand_index'
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

df.info() #gives information about the dataset

print(df.duplicated())# shows dupliccated values by checking each rows

print(df.duplicated().sum())# shows the total no of duplicated values

df = df.drop_duplicates()   # drops the duplicate values

print(df.columns)# shows the column name in a list 

print(df["usage_units"].skew())# shows if the specific column in the dataset is skewed or not " if 0 normal if not skewed distribution

columns = ['usage_units','provisioned_capacity', 'cost_usd', 'availability_pct',
           'economic_index', 'market_demand_index']
# store only the colums with numerical values to check if it is skewed or not


print(df.isnull().sum())# shows how many null values in each column

# important note : " if a specific column's null values is greater than 50% we should drop the column because it will cause errors .
# we cannot fill it without an machine learning model "

# Fills usage first 
df["usage_units"] = df["usage_units"].fillna(df["usage_units"].median())

# Calculate rate only using valid rows 
valid_rows = df[df['cost_usd'].notnull() & df['usage_units'].notnull()]
rate = (valid_rows['cost_usd'] / valid_rows['usage_units']).median()
print("Estimated pricing rate:", rate)

# Now fill cost using calculated rate
df['cost_usd'] = df['cost_usd'].fillna(df['usage_units'] * rate)


df["provisioned_capacity"] = df["provisioned_capacity"].fillna(df["provisioned_capacity"].median())


df["availability_pct"] = df["availability_pct"].fillna(df["availability_pct"].median())

df["economic_index"] = df["economic_index"].fillna(df["economic_index"].median())

df["market_demand_index"] = df["market_demand_index"].fillna(df["market_demand_index"].median())       

print(df.isnull().sum())

# ===============================
# Feature Engineering: Capacity Utilization
# ===============================

df['capacity_utilization'] = df['usage_units'] / df['provisioned_capacity']

columns = [
    'usage_units','provisioned_capacity','cost_usd',
    'availability_pct','economic_index','market_demand_index'
]

for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] >= upper_bound) | (df[col]<= lower_bound)]
    print(len(outliers))


for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] >= upper_bound) | (df[col]<= lower_bound)]
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)


for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] > upper_bound) | (df[col] < lower_bound)]
    print(len(outliers))


#  ---------  prints the rows before lag  ------------
print("Rows before lag:", len(df))



# ---- LAG FEATURES (CRITICAL FOR TIME SERIES) ----


df = df.sort_values(['region','service_type','timestamp'])

df['lag_1'] = df.groupby(['region','service_type'])['usage_units'].shift(1)

df['lag_7'] = df.groupby(['region','service_type'])['usage_units'].shift(7)

df['lag_14'] = df.groupby(['region','service_type'])['usage_units'].shift(14)


# ---- ROLLING FEATURES ----

df['rolling_mean_7'] = (
    df.groupby(['region','service_type'])['usage_units']
      .transform(lambda x: x.shift(1).rolling(7).mean())
)

df['rolling_std_7'] = (
    df.groupby(['region','service_type'])['usage_units']
      .transform(lambda x: x.shift(1).rolling(7).std())
)
# Drop rows created by lag

df = df.dropna(subset=['lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_std_7'])



# ----------- prints the rows after lag ---------

print("Rows after lag:", len(df))

# BUSINESS LOGIC VALIDATION


print("\n--- BUSINESS RULE VALIDATION ---")

# 1. Negative usage check
print("Negative usage values:", (df['usage_units'] < 0).sum())

# 2. Availability > 100%
print("Availability above 100%:", (df['availability_pct'] > 100).sum())

# 3. Usage exceeding capacity
print("Usage > Provisioned Capacity:",
      (df['usage_units'] > df['provisioned_capacity']).sum())

# Inspect a few rows
print("\nSample Over-Capacity Rows:")
print(df[df['usage_units'] > df['provisioned_capacity']][
    ['timestamp','region','service_type','usage_units','provisioned_capacity']
].head())

# Create over-capacity flag feature
df['over_capacity_flag'] = (df['usage_units'] > df['provisioned_capacity']).astype(int)

# 4. Economic index sanity check
print("Economic index outside realistic range:",
      ((df['economic_index'] < 80) | (df['economic_index'] > 120)).sum())


# ---------------------------------
# Feature Engineering
# ---------------------------------

# Usage spike flag

df['usage_spike_flag'] = (
    df['usage_units'] >
    df['rolling_mean_7'] + 2 * df['rolling_std_7']
).astype(int)

df = df.dropna(subset=['rolling_std_7'])
df = df.reset_index(drop=True)

# Seasonality flags
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)

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
    'usage_units',
    'provisioned_capacity',
    'cost_usd',
    'availability_pct',
    'economic_index',
    'market_demand_index',
    'capacity_utilization',
    'lag_1',
    'lag_7',
    'lag_14',
    'rolling_mean_7',
    'rolling_std_7'
]

plt.figure(figsize=(14,6))
df[core_vars].boxplot()
plt.title("After Preprocessing - Core Variables Boxplot")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histogram
df[core_vars].hist(bins=30, figsize=(16,12))
plt.suptitle("After Preprocessing - Core Variables Distribution", fontsize=16)
plt.tight_layout()
plt.show()

# ---- TIME FEATURE DISTRIBUTION ----


plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
sns.countplot(x='month', data=df)

plt.subplot(2,2,2)
sns.countplot(x='day_of_week', data=df)

plt.subplot(2,2,3)
sns.countplot(x='quarter', data=df)

plt.subplot(2,2,4)
sns.countplot(x='year', data=df)

plt.tight_layout()
plt.show()  

# ----------------- FINAL SHAPE OF THE DATASET -----------------
print("\nFinal dataset shape:", df.shape)

# Save the cleaned dataset
df.to_csv("final_cleaned_azure_data.csv", index=False)


# ----------------- PREPARE FOR MODELING -----------------

target = 'usage_units'

features = df.drop(columns=['usage_units','timestamp'])

X = features
y = df[target]



# ---- SPLIT INTO TRAIN/TEST BASED ON TIME ----

split_date = '2024-07-01'

train = df[df['timestamp'] < split_date]
test = df[df['timestamp'] >= split_date]

X_train = train.drop(['usage_units','timestamp'], axis=1)
y_train = train['usage_units']

X_test = test.drop(['usage_units','timestamp'], axis=1)
y_test = test['usage_units']

print("Training set shape:", X_train.shape)

# ===============================
# BACKTESTING USING TIMESERIES SPLIT
# ===============================

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

tscv = TimeSeriesSplit(n_splits=5)

print("\nBacktesting RMSE values:")

for train_index, test_index in tscv.split(X):

    X_train_bt, X_test_bt = X.iloc[train_index], X.iloc[test_index]
    y_train_bt, y_test_bt = y.iloc[train_index], y.iloc[test_index]

    model_bt = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6
    )

    model_bt.fit(X_train_bt, y_train_bt)

    pred_bt = model_bt.predict(X_test_bt)

    rmse_bt = np.sqrt(mean_squared_error(y_test_bt, pred_bt))

    print("Backtest RMSE:", rmse_bt)

# ===============================
# XGBOOST MODEL
# ===============================

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Train model
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Prediction
xgb_pred = xgb_model.predict(X_test)

# Evaluation Metrics
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mape = np.mean(np.abs((y_test - xgb_pred) / y_test)) * 100

print("\n===== XGBOOST RESULTS =====")
print("MAE :", xgb_mae)
print("RMSE:", xgb_rmse)
print("MAPE:", xgb_mape)

# ===============================
# Forecast Bias (XGBoost)
# ===============================

xgb_bias_baseline = np.mean(xgb_pred - y_test)

print("Forecast Bias (XGBoost):", xgb_bias_baseline)

# ===============================
# XGBOOST HYPERPARAMETER TUNING
# ===============================

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200,300],
    'max_depth': [4,6],
    'learning_rate': [0.05,0.1],
}

grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_xgb_model = grid_search.best_estimator_

print("\nBest XGBoost Parameters:", grid_search.best_params_)


# Prediction using tuned model
xgb_tuned_pred = best_xgb_model.predict(X_test)

# Evaluation of tuned model
xgb_tuned_mae = mean_absolute_error(y_test, xgb_tuned_pred)
xgb_tuned_rmse = np.sqrt(mean_squared_error(y_test, xgb_tuned_pred))
xgb_tuned_mape = np.mean(np.abs((y_test - xgb_tuned_pred)/y_test))*100

# Forecast bias using tuned model
xgb_bias_tuned = np.mean(xgb_tuned_pred - y_test)

print("Forecast Bias (Tuned XGBoost):", xgb_bias_tuned)


# plot XGboost predictions vs actuals



# ===============================
# ARIMA MODEL (AGGREGATED SERIES)
# ===============================

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------
# Step 1: Aggregate usage per day
# ---------------------------------

ts_data = df.groupby('timestamp')['usage_units'].sum()

print("Total time series points:", len(ts_data))


# ---------------------------------
# Step 2: Train-Test Split
# ---------------------------------

train_ts = ts_data[:'2024-06-30']
test_ts = ts_data['2024-07-01':]

print("ARIMA training points:", len(train_ts))
print("ARIMA testing points:", len(test_ts))


# ---------------------------------
# Step 3: Train ARIMA Model
# ---------------------------------

arima_model = ARIMA(train_ts, order=(5,1,2))

arima_fit = arima_model.fit()

print(arima_fit.summary())


# ---------------------------------
# Step 4: Forecast
# ---------------------------------

arima_pred = arima_fit.forecast(steps=len(test_ts))

# Convert predictions to pandas series
arima_pred = pd.Series(arima_pred, index=test_ts.index)


# ---------------------------------
# Step 5: Evaluation Metrics
# ---------------------------------

arima_mae = mean_absolute_error(test_ts, arima_pred)

arima_rmse = np.sqrt(mean_squared_error(test_ts, arima_pred))

arima_mape = np.mean(np.abs((test_ts - arima_pred) / test_ts)) * 100


print("\n===== ARIMA RESULTS =====")

print("MAE :", arima_mae)

print("RMSE:", arima_rmse)

print("MAPE:", arima_mape)

# ===============================
# Forecast Bias (ARIMA)
# ===============================

arima_bias = np.mean(arima_pred - test_ts)

print("Forecast Bias (ARIMA):", arima_bias)


# ---------------------------------
# Step 6: Plot Forecast vs Actual
# ---------------------------------



# ---------------------------------
# Step 7: Prediction Table
# ---------------------------------

arima_results = pd.DataFrame({
    "timestamp": test_ts.index,
    "Actual_Usage": test_ts.values,
    "ARIMA_Prediction": arima_pred.values
})

print("\nARIMA Prediction Table")

print(arima_results.head(20))


# ===============================
# ARIMA HYPERPARAMETER TUNING
# ===============================

import warnings
warnings.filterwarnings("ignore")

best_aic = float("inf")
best_order = None

for p in range(0,4):
    for d in range(0,2):
        for q in range(0,4):

            try:
                model = ARIMA(train_ts, order=(p,d,q))
                results = model.fit()

                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p,d,q)

            except:
                continue

print("Best ARIMA order:", best_order)


# Train ARIMA with best order
arima_tuned = ARIMA(train_ts, order=best_order)
arima_tuned_fit = arima_tuned.fit()

# Forecast with tuned ARIMA
arima_tuned_pred = arima_tuned_fit.forecast(steps=len(test_ts))

# Align prediction index with timestamps
arima_tuned_pred = pd.Series(arima_tuned_pred, index=test_ts.index)


# Evaluation of tuned ARIMA
arima_tuned_mae = mean_absolute_error(test_ts, arima_tuned_pred)
arima_tuned_rmse = np.sqrt(mean_squared_error(test_ts, arima_tuned_pred))
arima_tuned_mape = np.mean(np.abs((test_ts-arima_tuned_pred)/test_ts))*100

# Forecast bias for tuned ARIMA
arima_bias_tuned = np.mean(arima_tuned_pred - test_ts)

# ===============================
# Create results dataframe
# ===============================

results_df = test[['timestamp','usage_units']].copy()

# Add XGBoost predictions
results_df['XGBoost_Prediction'] = xgb_pred


# ===============================
# Create ARIMA dataframe
# ===============================

arima_df = pd.DataFrame({
    'timestamp': test_ts.index,
    'Actual_ARIMA_Usage': test_ts.values,
    'ARIMA_Prediction': arima_pred.values
})


# ===============================
# Merge ARIMA data with results
# ===============================

results_df = results_df.merge(arima_df, on='timestamp', how='left')


# Reorder columns for better comparison
results_df = results_df[[
    'timestamp',
    'usage_units',
    'XGBoost_Prediction',
    'Actual_ARIMA_Usage',
    'ARIMA_Prediction'
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

# Set model as index
validation_df = validation_df.set_index("Model")

# Round values for cleaner output
validation_df = validation_df.round(4)

print("\nMODEL VALIDATION RESULTS")
print(validation_df)

# =============================================================================
# VISUALISATIONS
# =============================================================================

# ------------------------------
# Feature Distributions
# ------------------------------
def plot_distributions(df):
    
    core_vars = [
        'usage_units',
        'provisioned_capacity',
        'cost_usd',
        'availability_pct',
        'economic_index',
        'market_demand_index',
        'capacity_utilization',
        'lag_1',
        'lag_7',
        'lag_14',
        'rolling_mean_7',
        'rolling_std_7'
    ]

    df[core_vars].hist(bins=30, figsize=(14,10))
    plt.suptitle("Feature Distributions", fontsize=14)
    plt.tight_layout()
    plt.show()


# ------------------------------
# XGBoost Forecast Plot
# ------------------------------
def plot_xgb_forecast(test, y_test, xgb_pred, xgb_tuned_pred):

    plt.figure(figsize=(14,5))

    plt.plot(test["timestamp"], y_test,
             label="Actual Usage",
             linewidth=2)

    plt.plot(test["timestamp"], xgb_pred,
             label="Baseline XGBoost",
             linestyle="--")

    plt.plot(test["timestamp"], xgb_tuned_pred,
             label="Tuned XGBoost",
             linestyle="--")

    plt.title("XGBoost Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Usage Units")

    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------
# ARIMA Forecast Plot
# ------------------------------
def plot_arima_forecast(test_ts, arima_pred, arima_tuned_pred):

    plt.figure(figsize=(14,5))

    plt.plot(test_ts.index, test_ts.values,
             label="Actual Usage",
             linewidth=2)

    plt.plot(test_ts.index, arima_pred,
             label="Baseline ARIMA",
             linestyle="--")

    plt.plot(test_ts.index, arima_tuned_pred,
             label="Tuned ARIMA",
             linestyle="--")

    plt.title("ARIMA Forecast vs Actual (Aggregated)")
    plt.xlabel("Date")
    plt.ylabel("Total Usage Units")

    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------
# Feature Importance Plot
# ------------------------------
def plot_feature_importance(model, feature_names, top_n=20):

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    )

    importance = importance.nlargest(top_n).sort_values()

    plt.figure(figsize=(10,6))

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

best_model = validation_df["RMSE"].idxmin()
best_rmse = validation_df.loc[best_model, "RMSE"]

print(f"\nBest Model: {best_model} (RMSE = {best_rmse:.4f})")