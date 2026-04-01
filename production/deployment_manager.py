"""
Production Deployment Manager
Handles model deployment to production environments with versioning, rollback, and health checks
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import joblib
import pandas as pd
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionDeploymentManager:
    """Manages model deployment, versioning, and rollback in production"""
    
    def __init__(self, prod_models_dir='production/models', staging_dir='production/staging',
                 deployments_log='production/logs/deployments.json'):
        self.prod_models_dir = prod_models_dir
        self.staging_dir = staging_dir
        self.deployments_log = deployments_log
        
        # Create directories
        Path(prod_models_dir).mkdir(parents=True, exist_ok=True)
        Path(staging_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(deployments_log)).mkdir(parents=True, exist_ok=True)
        
        self.deployments = self._load_deployments()
        self.active_version = self._get_active_version()
        
        logger.info(f"Deployment Manager initialized. Active version: {self.active_version}")
    
    def _load_deployments(self) -> List[Dict]:
        """Load deployment history"""
        if os.path.exists(self.deployments_log):
            try:
                with open(self.deployments_log, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_deployments(self):
        """Save deployment history"""
        with open(self.deployments_log, 'w') as f:
            json.dump(self.deployments, f, indent=2)
    
    def _get_active_version(self) -> Optional[str]:
        """Get currently active model version"""
        for dep in reversed(self.deployments):
            if dep.get('status') == 'active':
                return dep.get('version')
        return None
    
    def _calculate_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of model file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def stage_model(self, model_path: str, features_path: str, version: str,
                    metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Stage model for production deployment with validation"""
        try:
            # Load and validate model
            model = joblib.load(model_path)
            features = joblib.load(features_path)
            
            # Calculate hash for integrity checking
            model_hash = self._calculate_hash(model_path)
            features_hash = self._calculate_hash(features_path)
            
            # Create staging directory for this version
            version_staging = os.path.join(self.staging_dir, f'v{version}')
            os.makedirs(version_staging, exist_ok=True)
            
            # Copy files to staging
            staged_model_path = os.path.join(version_staging, 'model.pkl')
            staged_features_path = os.path.join(version_staging, 'features.pkl')
            shutil.copy(model_path, staged_model_path)
            shutil.copy(features_path, staged_features_path)
            
            # Record staging info
            staging_record = {
                'version': version,
                'staged_at': datetime.now().isoformat(),
                'model_hash': model_hash,
                'features_hash': features_hash,
                'model_path': staged_model_path,
                'features_path': staged_features_path,
                'metrics': metrics,
                'status': 'staged'
            }
            
            logger.info(f"Model v{version} staged successfully")
            return {
                'success': True,
                'version': version,
                'staging_record': staging_record
            }
        
        except Exception as e:
            logger.error(f"Staging failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def deploy_to_production(self, version: str, approval_notes: str = "",
                            pre_checks: Optional[Dict] = None) -> Dict[str, Any]:
        """Deploy staged model to production with pre-deployment checks"""
        try:
            staged_version_path = os.path.join(self.staging_dir, f'v{version}')
            
            if not os.path.exists(staged_version_path):
                return {'success': False, 'error': f'Staged version {version} not found'}
            
            # Run pre-deployment checks
            check_results = pre_checks or self._run_pre_deployment_checks(version)
            if not all(check_results.values()):
                return {
                    'success': False,
                    'error': 'Pre-deployment checks failed',
                    'checks': check_results
                }
            
            # If there's an active version, back it up
            if self.active_version:
                backup_path = os.path.join(
                    self.prod_models_dir,
                    f'model_v{self.active_version}_backup'
                )
                active_path = os.path.join(self.prod_models_dir, 'model_active')
                if os.path.exists(active_path):
                    shutil.move(active_path, backup_path)
                    logger.info(f"Backed up previous version to {backup_path}")
            
            # Copy staged model to production
            prod_version_path = os.path.join(self.prod_models_dir, f'model_v{version}')
            shutil.copytree(staged_version_path, prod_version_path, dirs_exist_ok=True)
            
            # Symlink to active model for easy access
            active_link = os.path.join(self.prod_models_dir, 'model_active')
            if os.path.exists(active_link) or os.path.islink(active_link):
                os.remove(active_link)
            os.symlink(prod_version_path, active_link)
            
            # Record deployment
            deployment_record = {
                'version': version,
                'deployed_at': datetime.now().isoformat(),
                'status': 'active',
                'approval_notes': approval_notes,
                'pre_checks': check_results,
                'previous_version': self.active_version,
                'environment': 'production'
            }
            self.deployments.append(deployment_record)
            self._save_deployments()
            
            # Update active version
            self.active_version = version
            
            logger.info(f"✅ Model v{version} deployed to production successfully")
            return {
                'success': True,
                'version': version,
                'deployment_record': deployment_record,
                'message': f'Model v{version} is now active in production'
            }
        
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _run_pre_deployment_checks(self, version: str) -> Dict[str, bool]:
        """Run pre-deployment validation checks"""
        try:
            staged_version_path = os.path.join(self.staging_dir, f'v{version}')
            model_path = os.path.join(staged_version_path, 'model.pkl')
            features_path = os.path.join(staged_version_path, 'features.pkl')
            
            checks = {
                'files_exist': os.path.exists(model_path) and os.path.exists(features_path),
                'model_loadable': self._verify_model_loadable(model_path),
                'features_loadable': self._verify_features_loadable(features_path),
            }
            
            return checks
        except Exception as e:
            logger.error(f"Pre-deployment checks error: {str(e)}")
            return {'error': False}
    
    def _verify_model_loadable(self, model_path: str) -> bool:
        """Verify model can be loaded"""
        try:
            model = joblib.load(model_path)
            return hasattr(model, 'predict')
        except:
            return False
    
    def _verify_features_loadable(self, features_path: str) -> bool:
        """Verify features file can be loaded"""
        try:
            features = joblib.load(features_path)
            return isinstance(features, list)
        except:
            return False
    
    def rollback(self, reason: str = "") -> Dict[str, Any]:
        """Rollback to previous active version"""
        try:
            # Find previous deployment
            previous_deployments = [d for d in reversed(self.deployments)
                                   if d.get('status') in ['active', 'rollback_to']]
            
            if len(previous_deployments) < 2:
                return {'success': False, 'error': 'No previous version to rollback to'}
            
            previous_version = previous_deployments[1].get('version')
            
            # Verify previous version exists
            prev_model_path = os.path.join(self.prod_models_dir, f'model_v{previous_version}')
            if not os.path.exists(prev_model_path):
                return {'success': False, 'error': f'Previous version {previous_version} not found'}
            
            # Update active symlink
            active_link = os.path.join(self.prod_models_dir, 'model_active')
            if os.path.exists(active_link) or os.path.islink(active_link):
                os.remove(active_link)
            os.symlink(prev_model_path, active_link)
            
            # Record rollback
            rollback_record = {
                'version': previous_version,
                'rolled_back_at': datetime.now().isoformat(),
                'status': 'active',
                'from_version': self.active_version,
                'reason': reason,
                'environment': 'production'
            }
            self.deployments.append(rollback_record)
            self._save_deployments()
            
            self.active_version = previous_version
            
            logger.warning(f"🔄 Rolled back to v{previous_version}. Reason: {reason}")
            return {
                'success': True,
                'version': previous_version,
                'message': f'Rolled back to version {previous_version}'
            }
        
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_deployment_history(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get deployment history as dataframe"""
        if not self.deployments:
            return pd.DataFrame()
        
        # Convert to dataframe
        df = pd.DataFrame(self.deployments)
        
        # Select and rename columns
        columns = ['deployed_at' if 'deployed_at' in df.columns else 'rolled_back_at',
                   'version', 'status', 'environment']
        df = df[[col for col in columns if col in df.columns]]
        
        if limit:
            df = df.tail(limit)
        
        return df.sort_values('deployed_at' if 'deployed_at' in df.columns else 'rolled_back_at',
                             ascending=False)
    
    def get_active_model_info(self) -> Dict[str, Any]:
        """Get information about currently active production model"""
        if not self.active_version:
            return {'status': 'No active model'}
        
        try:
            active_model_path = os.path.join(self.prod_models_dir, 'model_active/model.pkl')
            features_path = os.path.join(self.prod_models_dir, 'model_active/features.pkl')
            
            model = joblib.load(active_model_path)
            features = joblib.load(features_path)
            
            # Find deployment record
            deployment = next(d for d in reversed(self.deployments)
                            if d.get('version') == self.active_version and d.get('status') == 'active')
            
            return {
                'active_version': self.active_version,
                'deployed_at': deployment.get('deployed_at'),
                'features_count': len(features),
                'model_type': type(model).__name__,
                'model_size_mb': os.path.getsize(active_model_path) / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting active model info: {str(e)}")
            return {'error': str(e)}
