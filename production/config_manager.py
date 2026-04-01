"""
Production Configuration Management
Centralized configuration for all production components
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ProductionConfig:
    """Production configuration manager"""
    
    def __init__(self, config_file: str = 'production/config/production.json'):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config file: {str(e)}, using defaults")
        
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default production configuration"""
        return {
            'environment': 'production',
            'deployment': {
                'models_dir': 'production/models',
                'staging_dir': 'production/staging',
                'logs_dir': 'production/logs',
                'max_versions_to_keep': 5,
                'auto_cleanup': True
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'timeout_seconds': 30,
                'rate_limit': {
                    'enabled': True,
                    'requests_per_minute': 1000
                }
            },
            'monitoring': {
                'enabled': True,
                'metrics_window_size': 100,
                'performance_alert_threshold': 0.1,
                'health_check_interval_seconds': 60,
                'anomaly_sensitivity': 2.0
            },
            'alarting': {
                'enabled': True,
                'notification_channels': ['log'],  # log, email, slack, teams
                'critical_alert_escalation': True
            },
            'scheduling': {
                'enabled': True,
                'retraining_frequency': 'daily',
                'retraining_time': '02:00',
                'report_frequency': 'daily',
                'report_time': '06:00',
                'health_check_frequency': 'hourly'
            },
            'azure': {
                'enabled': True,
                'subscription_id': os.getenv('AZURE_SUBSCRIPTION_ID', ''),
                'resource_group': os.getenv('AZURE_RESOURCE_GROUP', ''),
                'autoscale_enabled': True,
                'cost_optimization_enabled': True
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'production/logs/production.log',
                'max_file_size_mb': 100,
                'backup_count': 5
            },
            'security': {
                'api_key_required': True,
                'ssl_enabled': True,
                'cors_enabled': True,
                'cors_origins': ['http://localhost:3000', 'https://localhost:3000']
            },
            'performance': {
                'cache_enabled': True,
                'cache_ttl_seconds': 300,
                'async_enabled': True,
                'batch_prediction_enabled': True
            },
            'data': {
                'retraining_data_window_days': 30,
                'min_samples_for_retraining': 100,
                'drift_detection_enabled': True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.info(f"Configuration updated: {key} = {value}")
    
    def save(self):
        """Save configuration to file"""
        try:
            Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self.config
    
    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = [
            'environment',
            'deployment.models_dir',
            'api.host',
            'api.port',
            'monitoring.enabled'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                logger.error(f"Missing required configuration: {key}")
                return False
        
        logger.info("Configuration validation passed")
        return True


class EnvironmentConfigLoader:
    """Load configuration from environment variables"""
    
    @staticmethod
    def load_from_env() -> Dict[str, str]:
        """Load all AZURE_* and FORECAST_* environment variables"""
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(('AZURE_', 'FORECAST_')):
                env_config[key] = value
        
        return env_config
    
    @staticmethod
    def get_database_connection(config: ProductionConfig) -> str:
        """Get database connection string"""
        db_host = os.getenv('FORECAST_DB_HOST', 'localhost')
        db_port = os.getenv('FORECAST_DB_PORT', '5432')
        db_name = os.getenv('FORECAST_DB_NAME', 'forecast_db')
        db_user = os.getenv('FORECAST_DB_USER', 'forecast_user')
        db_password = os.getenv('FORECAST_DB_PASSWORD', '')
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    @staticmethod
    def get_storage_connection(config: ProductionConfig) -> str:
        """Get storage connection string (Azure Storage)"""
        return os.getenv('AZURE_STORAGE_CONNECTION_STRING', '')


class ConfigurationValidator:
    """Validate production configuration"""
    
    @staticmethod
    def validate_paths(config: ProductionConfig) -> bool:
        """Validate all required directories exist or can be created"""
        paths = [
            config.get('deployment.models_dir'),
            config.get('deployment.staging_dir'),
            config.get('deployment.logs_dir'),
            config.get('logging.file')
        ]
        
        for path in paths:
            if path:
                try:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(f"Cannot access path {path}: {str(e)}")
                    return False
        
        return True
    
    @staticmethod
    def validate_azure_config(config: ProductionConfig) -> bool:
        """Validate Azure configuration"""
        if not config.get('azure.enabled'):
            return True
        
        required_azure_settings = [
            'azure.subscription_id',
            'azure.resource_group'
        ]
        
        for setting in required_azure_settings:
            if not config.get(setting):
                logger.warning(f"Azure setting not configured: {setting}")
        
        return True
    
    @staticmethod
    def validate_api_config(config: ProductionConfig) -> bool:
        """Validate API configuration"""
        port = config.get('api.port')
        workers = config.get('api.workers', 1)
        
        if not isinstance(port, int) or port < 1 or port > 65535:
            logger.error(f"Invalid API port: {port}")
            return False
        
        if workers < 1:
            logger.error(f"Invalid worker count: {workers}")
            return False
        
        return True
    
    @staticmethod
    def validate_all(config: ProductionConfig) -> bool:
        """Run all validation checks"""
        checks = [
            ('paths', ConfigurationValidator.validate_paths),
            ('azure', ConfigurationValidator.validate_azure_config),
            ('api', ConfigurationValidator.validate_api_config)
        ]
        
        all_valid = config.validate()
        
        for check_name, check_fn in checks:
            if not check_fn(config):
                logger.warning(f"Configuration validation failed: {check_name}")
                all_valid = False
        
        return all_valid


# Global configuration instance
_global_config = None


def get_config() -> ProductionConfig:
    """Get global configuration instance"""
    global _global_config
    
    if _global_config is None:
        _global_config = ProductionConfig()
    
    return _global_config


def initialize_config(config_file: Optional[str] = None) -> ProductionConfig:
    """Initialize configuration"""
    global _global_config
    
    _global_config = ProductionConfig(config_file or 'production/config/production.json')
    
    # Validate configuration
    if not ConfigurationValidator.validate_all(_global_config):
        logger.warning("Configuration has warnings but continuing...")
    
    return _global_config
