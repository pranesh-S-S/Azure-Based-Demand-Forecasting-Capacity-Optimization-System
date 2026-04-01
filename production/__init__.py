"""
Production Package
Contains all production deployment, monitoring, and integration components
"""

__version__ = '1.0.0'
__author__ = 'Azure Forecast Team'

from .deployment_manager import ProductionDeploymentManager
from .monitoring import ModelPerformanceMonitor, SystemHealthMonitor, AnomalyDetector, AlertManager
from .azure_integration import (
    AzureMonitorIntegration,
    AzureAutoscaleIntegration,
    AzureCostOptimization,
    CapacityPlanningIntegration
)
from .reporting import ReportGenerator
from .orchestration import OrchestrationScheduler, WorkflowOrchestrator
from .config_manager import ProductionConfig, initialize_config, get_config
from .initialization import ProductionEnvironment, create_production_environment

__all__ = [
    'ProductionDeploymentManager',
    'ModelPerformanceMonitor',
    'SystemHealthMonitor',
    'AnomalyDetector',
    'AlertManager',
    'AzureMonitorIntegration',
    'AzureAutoscaleIntegration',
    'AzureCostOptimization',
    'CapacityPlanningIntegration',
    'ReportGenerator',
    'OrchestrationScheduler',
    'WorkflowOrchestrator',
    'ProductionConfig',
    'initialize_config',
    'get_config',
    'ProductionEnvironment',
    'create_production_environment'
]
