import pandas as pd
import numpy as np
import joblib

df = pd.read_csv('final_cleaned_azure_data.csv')
model = joblib.load('best_xgboost_model.pkl')
features = joblib.load('model_features.pkl')
clip_bounds = joblib.load('clip_bounds.pkl')

split_date = '2024-07-01'
df['timestamp'] = pd.to_datetime(df['timestamp'])

train = df[df['timestamp'] < split_date]
test  = df[df['timestamp'] >= split_date]

X_test = test[features]
y_test = test['usage_units'].values
preds  = model.predict(X_test)

residuals = preds - y_test
bias = np.mean(residuals)

results = {}
results['bias'] = round(bias, 6)
results['median_residual'] = round(np.median(residuals), 6)
results['std_residuals'] = round(np.std(residuals), 6)
results['bias_std_ratio'] = round(abs(bias) / np.std(residuals), 6)
results['train_mean'] = round(train['usage_units'].mean(), 4)
results['test_mean'] = round(y_test.mean(), 4)
results['distribution_shift'] = round(y_test.mean() - train['usage_units'].mean(), 4)
results['bias_pct_of_mean'] = round((bias / y_test.mean()) * 100, 4)

if 'usage_units' in clip_bounds:
    lb, ub = clip_bounds['usage_units']
    results['iqr_lower_bound'] = round(lb, 2)
    results['iqr_upper_bound'] = round(ub, 2)
    results['test_at_upper_bound'] = int((np.abs(y_test - ub) < 0.5).sum())
    results['test_near_upper_bound'] = int((y_test > ub * 0.95).sum())

train_max_val = train['usage_units'].max()
clip_ceil = train_max_val * 1.5
raw_preds = model.predict(X_test)
results['clip_ceiling'] = round(clip_ceil, 2)
results['preds_clipped_above'] = int((raw_preds > clip_ceil).sum())
results['bias_without_clipping'] = round(np.mean(raw_preds - y_test), 6)

params = model.get_params()
results['n_estimators'] = params.get('n_estimators')
results['max_depth'] = params.get('max_depth')
results['learning_rate'] = params.get('learning_rate')
results['reg_alpha'] = params.get('reg_alpha')
results['reg_lambda'] = params.get('reg_lambda')
results['subsample'] = params.get('subsample')
results['colsample_bytree'] = params.get('colsample_bytree')

# Monthly bias
test_a = test[['timestamp', 'usage_units']].copy()
test_a['pred'] = preds
test_a['residual'] = residuals
test_a['month'] = test_a['timestamp'].dt.month
mb = test_a.groupby('month').agg(
    mean_actual=('usage_units', 'mean'),
    mean_pred=('pred', 'mean'),
    bias=('residual', 'mean'),
    n=('residual', 'count')
).round(4)

# Residual percentiles
for p in [5, 25, 50, 75, 95]:
    results['residual_P' + str(p)] = round(np.percentile(residuals, p), 4)

# Service type bias
service_cols = [c for c in test.columns if c.startswith('service_type_')]
svc_bias = {}
for col in service_cols:
    mask = test[col].values == 1
    if mask.sum() > 0:
        svc_bias[col] = {'bias': round(np.mean(residuals[mask]), 4), 'n': int(mask.sum())}

# Region bias (top 5)
region_cols = [c for c in test.columns if c.startswith('region_')]
reg_bias = {}
for col in region_cols:
    mask = test[col].values == 1
    if mask.sum() > 0:
        reg_bias[col] = {'bias': round(np.mean(residuals[mask]), 4), 'n': int(mask.sum())}

# Write everything as JSON for easy reading
import json
report = {
    'overall_metrics': results,
    'monthly_bias': mb.to_dict('index'),
    'service_type_bias': svc_bias,
    'region_bias': reg_bias,
}

with open('bias_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, default=str)

print('Report saved to bias_report.json')
