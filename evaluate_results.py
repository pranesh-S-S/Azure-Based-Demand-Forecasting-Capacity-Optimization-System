import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("final_cleaned_azure_data.csv")
split_date = '2024-07-01'
test = df[df['timestamp'] >= split_date]

# Load model and features
model = joblib.load('models/best_xgboost_model.pkl')
features_list = joblib.load('models/model_features.pkl')

# Define leak columns (same as in the main script)
leak_cols = ['usage_units', 'timestamp', 'capacity_utilization', 'over_capacity_flag', 'usage_spike_flag', 'cost_usd']

# Prepare X_test, y_test
# Note: In the training script, we drop leak_cols from X_test.
X_test = test.drop(columns=leak_cols, errors='ignore')
y_test = test['usage_units']

# Predict (Log-based)
log_preds = model.predict(X_test)
xgb_tuned_pred = np.expm1(log_preds)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, xgb_tuned_pred))
print(f"Final RMSE (with log transformation and no leakage): {rmse:.4f}")
