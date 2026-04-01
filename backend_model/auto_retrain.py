import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
import os
import json

class AutoRetrainer:
    """Automatic Model Retraining Engine"""
    
    def __init__(self, model_path, features_path, version_dir='models/versions'):
        self.model_path = model_path
        self.features_path = features_path
        self.version_dir = version_dir
        self.current_model = joblib.load(model_path)
        self.current_features = joblib.load(features_path)
        self.version = 1
        self.history = []
        
        os.makedirs(version_dir, exist_ok=True)
    
    def needs_retraining(self, new_data, current_threshold=0.75):
        """Check if model needs retraining"""
        try:
            # Test model on new data
            X_test = self._prepare_features(new_data)
            y_test = new_data['usage_units'].values
            
            predictions = self.current_model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            # Retrain if R² drops below threshold
            trigger_retrain = r2 < current_threshold
            
            return {
                'should_retrain': trigger_retrain,
                'reason': f'Low R² score ({r2:.3f})',
                'current_r2': r2,
                'mae': mae,
                'threshold': current_threshold
            }
        except Exception as e:
            return {'should_retrain': False, 'error': str(e)}
    
    def _prepare_features(self, df):
        """Prepare features from dataframe"""
        df_enc = pd.get_dummies(df, columns=['region','service_type'], drop_first=True)
        for col in self.current_features:
            if col not in df_enc.columns:
                df_enc[col] = 0
        return df_enc[self.current_features]
    
    def retrain(self, training_data, test_data=None):
        """Retrain model on new data"""
        try:
            X_train = self._prepare_features(training_data)
            y_train = training_data['usage_units'].values
            
            # Train new model
            new_model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            new_model.fit(X_train, y_train, verbose=False)
            
            # Validate on test data if provided
            if test_data is not None:
                X_test = self._prepare_features(test_data)
                y_test = test_data['usage_units'].values
                
                old_preds = self.current_model.predict(X_test)
                new_preds = new_model.predict(X_test)
                
                old_r2 = r2_score(y_test, old_preds)
                new_r2 = r2_score(y_test, new_preds)
                old_mae = mean_absolute_error(y_test, old_preds)
                new_mae = mean_absolute_error(y_test, new_preds)
                
                improvement = ((new_r2 - old_r2) / abs(old_r2)) * 100 if old_r2 != 0 else 0
                
                validation_result = {
                    'old_r2': old_r2,
                    'new_r2': new_r2,
                    'old_mae': old_mae,
                    'new_mae': new_mae,
                    'improvement_pct': improvement,
                    'deploy': new_r2 > old_r2  # Only deploy if better
                }
            else:
                validation_result = {
                    'improvement_pct': 0,
                    'deploy': True
                }
            
            # Save model version
            self.version += 1
            version_path = os.path.join(self.version_dir, f'model_v{self.version}.pkl')
            joblib.dump(new_model, version_path)
            
            # Log history
            retrain_log = {
                'timestamp': datetime.now().isoformat(),
                'version': self.version,
                'training_rows': len(training_data),
                'validation': validation_result,
                'path': version_path
            }
            self.history.append(retrain_log)
            
            # Deploy if validation passed
            if validation_result['deploy']:
                joblib.dump(new_model, self.model_path)
                self.current_model = new_model
                return {
                    'success': True,
                    'version': self.version,
                    'message': f"✅ Model v{self.version} deployed! Improvement: {validation_result['improvement_pct']:.1f}%",
                    'validation': validation_result
                }
            else:
                return {
                    'success': False,
                    'version': self.version,
                    'message': f"❌ New model v{self.version} is worse. Keeping current model.",
                    'validation': validation_result
                }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def save_history(self, path='models/retrain_history.json'):
        """Save retraining history"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_history(self, path='models/retrain_history.json'):
        """Load retraining history"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.history = json.load(f)
    
    def get_history_df(self):
        """Get history as dataframe"""
        if not self.history:
            return pd.DataFrame()
        
        data = []
        for entry in self.history:
            data.append({
                'Date': entry['timestamp'][:10],
                'Version': entry['version'],
                'Rows': entry['training_rows'],
                'R² Improvement': f"{entry['validation'].get('improvement_pct', 0):.1f}%",
                'Deployed': '✅' if entry['validation'].get('deploy', False) else '❌'
            })
        
        return pd.DataFrame(data)
