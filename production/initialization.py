"""
Production Environment Initialization & Orchestration
Bootstrap and manage all production components
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import os

# Import production components
from production.config_manager import initialize_config, get_config, ConfigurationValidator
from production.deployment_manager import ProductionDeploymentManager
from production.monitoring import ModelPerformanceMonitor, SystemHealthMonitor, AnomalyDetector, AlertManager
from production.azure_integration import (
    AzureMonitorIntegration, AzureAutoscaleIntegration, AzureCostOptimization, CapacityPlanningIntegration
)
from production.reporting import ReportGenerator
from production.orchestration import OrchestrationScheduler, WorkflowOrchestrator, AutomatedPipelineFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionEnvironment:
    """Manages the complete production environment"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize production environment
        
        Args:
            config_file: Path to configuration file
        """
        logger.info("=" * 60)
        logger.info("Initializing Azure Forecast Production Environment")
        logger.info("=" * 60)
        
        # Initialize configuration
        self.config = initialize_config(config_file)
        
        # Initialize core components
        self.deployment_manager = ProductionDeploymentManager(
            prod_models_dir=self.config.get('deployment.models_dir'),
            staging_dir=self.config.get('deployment.staging_dir'),
            deployments_log=str(Path(self.config.get('deployment.logs_dir')) / 'deployments.json')
        )
        
        if self.config.get('monitoring.enabled'):
            self.performance_monitor = ModelPerformanceMonitor(
                window_size=self.config.get('monitoring.metrics_window_size', 100)
            )
            self.health_monitor = SystemHealthMonitor()
            self.anomaly_detector = AnomalyDetector(
                sensitivity=self.config.get('monitoring.anomaly_sensitivity', 2.0)
            )
            self.alert_manager = AlertManager()
        
        # Initialize Azure integrations
        if self.config.get('azure.enabled'):
            self.azure_monitor = AzureMonitorIntegration(
                workspace_id=os.getenv('AZURE_WORKSPACE_ID', 'default'),
                workspace_key=os.getenv('AZURE_WORKSPACE_KEY', 'default')
            )
            self.azure_autoscale = AzureAutoscaleIntegration(
                subscription_id=self.config.get('azure.subscription_id', ''),
                resource_group=self.config.get('azure.resource_group', '')
            )
            self.cost_optimization = AzureCostOptimization()
            self.capacity_planning = CapacityPlanningIntegration()
        
        # Initialize reporting
        self.report_generator = ReportGenerator(
            reports_dir=str(Path(self.config.get('deployment.logs_dir')) / 'reports')
        )
        
        # Initialize orchestration
        if self.config.get('scheduling.enabled'):
            self.scheduler = OrchestrationScheduler()
            self.workflow_orchestrator = WorkflowOrchestrator()
        
        self.initialized = True
        logger.info("✅ Production environment initialized successfully")
    
    def setup_monitoring(self):
        """Setup monitoring and alerting rules"""
        if not self.config.get('monitoring.enabled'):
            logger.info("Monitoring is disabled in configuration")
            return
        
        logger.info("Setting up monitoring and alerting...")
        
        # Register alert rules
        self.alert_manager.register_notification_channel(
            'log',
            lambda alert: logger.warning(f"ALERT: {alert['rule']} - {alert['context']}")
        )
        
        # Define common alert rules
        def high_error_rate_check(context):
            metrics = context.get('metrics', {})
            return metrics.get('avg_error', 0) > self.config.get('monitoring.performance_alert_threshold')
        
        def high_inference_time_check(context):
            metrics = context.get('metrics', {})
            return metrics.get('avg_inference_time_ms', 0) > 100
        
        def low_availability_check(context):
            health = context.get('health', {})
            return health.get('availability_pct', 100) < 95
        
        self.alert_manager.add_alert_rule(
            'high_error_rate',
            high_error_rate_check,
            ['log'],
            'HIGH'
        )
        
        self.alert_manager.add_alert_rule(
            'high_inference_time',
            high_inference_time_check,
            ['log'],
            'MEDIUM'
        )
        
        self.alert_manager.add_alert_rule(
            'low_availability',
            low_availability_check,
            ['log'],
            'CRITICAL'
        )
        
        logger.info("✅ Monitoring and alerts configured")
    
    def setup_scheduling(self):
        """Setup automated scheduling"""
        if not self.config.get('scheduling.enabled'):
            logger.info("Scheduling is disabled in configuration")
            return
        
        logger.info("Setting up automated scheduling...")
        
        # Register scheduled tasks
        retraining_config = {
            'frequency': self.config.get('scheduling.retraining_frequency', 'daily'),
            'time': self.config.get('scheduling.retraining_time', '02:00')
        }
        
        reporting_config = {
            'frequency': self.config.get('scheduling.report_frequency', 'daily'),
            'time': self.config.get('scheduling.report_time', '06:00')
        }
        
        health_check_config = {
            'frequency': 'hourly',
            'interval': 1
        }
        
        # Define task functions
        def retraining_task():
            logger.info("Running scheduled retraining task...")
            return {'status': 'completed'}
        
        def reporting_task():
            logger.info("Running scheduled reporting task...")
            metrics = self.performance_monitor.get_current_metrics() if hasattr(self, 'performance_monitor') else {}
            return {'status': 'completed', 'metrics': metrics}
        
        def health_check_task():
            logger.info("Running scheduled health check...")
            if hasattr(self, 'health_monitor'):
                return self.health_monitor.run_health_check()
            return {'status': 'healthy'}
        
        # Register tasks
        self.scheduler.register_task(
            'scheduled_retraining',
            'Model Retraining',
            retraining_task,
            retraining_config
        )
        
        self.scheduler.register_task(
            'scheduled_reporting',
            'Report Generation',
            reporting_task,
            reporting_config
        )
        
        self.scheduler.register_task(
            'scheduled_health_check',
            'System Health Check',
            health_check_task,
            health_check_config
        )
        
        logger.info("✅ Scheduling configured")
    
    def start(self):
        """Start the production environment"""
        logger.info("Starting production environment...")
        
        if self.config.get('scheduling.enabled') and hasattr(self, 'scheduler'):
            self.scheduler.start()
            logger.info("✅ Scheduler started")
        
        logger.info("=" * 60)
        logger.info("Production environment is OPERATIONAL")
        logger.info("=" * 60)
        
        self._print_status()
    
    def stop(self):
        """Stop the production environment"""
        logger.info("Stopping production environment...")
        
        if hasattr(self, 'scheduler') and self.scheduler.running:
            self.scheduler.stop()
            logger.info("Scheduler stopped")
        
        logger.info("Production environment stopped")
    
    def _print_status(self):
        """Print current status"""
        logger.info("\n" + "=" * 60)
        logger.info("PRODUCTION ENVIRONMENT STATUS")
        logger.info("=" * 60)
        
        logger.info(f"Configuration File: {self.config.config_file}")
        logger.info(f"Models Directory: {self.config.get('deployment.models_dir')}")
        logger.info(f"Active Model Version: {self.deployment_manager.active_version}")
        
        if hasattr(self, 'performance_monitor'):
            metrics = self.performance_monitor.get_current_metrics()
            logger.info(f"Monitoring Enabled: Yes")
            logger.info(f"  - Predictions Tracked: {metrics.get('predictions_count', 0)}")
        
        if self.config.get('azure.enabled'):
            logger.info(f"Azure Integration: Enabled")
            logger.info(f"  - Subscription: {self.config.get('azure.subscription_id', 'Not configured')}")
            logger.info(f"  - Resource Group: {self.config.get('azure.resource_group', 'Not configured')}")
        
        if self.config.get('scheduling.enabled') and hasattr(self, 'scheduler'):
            logger.info(f"Scheduling: Enabled")
            tasks = self.scheduler.get_all_tasks_status()
            logger.info(f"  - Scheduled Tasks: {len(tasks)}")
        
        logger.info("=" * 60 + "\n")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'environment': 'production',
            'configuration': {
                'config_file': self.config.config_file,
                'models_dir': self.config.get('deployment.models_dir'),
                'azure_enabled': self.config.get('azure.enabled'),
                'monitoring_enabled': self.config.get('monitoring.enabled'),
                'scheduling_enabled': self.config.get('scheduling.enabled')
            },
            'deployment': {
                'active_model_version': self.deployment_manager.active_version,
                'active_model_info': self.deployment_manager.get_active_model_info()
            },
            'monitoring': {
                'status': 'healthy' if hasattr(self, 'health_monitor') else 'disabled'
            },
            'azure': {
                'integration_enabled': self.config.get('azure.enabled'),
                'autoscale_enabled': self.config.get('azure.autoscale_enabled'),
                'cost_optimization_enabled': self.config.get('azure.cost_optimization_enabled')
            }
        }


class ProductionDeploymentGuide:
    """Provides deployment guidance and checklists"""
    
    @staticmethod
    def print_pre_deployment_checklist():
        """Print pre-deployment checklist"""
        logger.info("\n" + "=" * 60)
        logger.info("PRE-DEPLOYMENT CHECKLIST")
        logger.info("=" * 60)
        
        checklist = [
            "☐ Model performance validated in staging",
            "☐ Model files (model.pkl, features.pkl) ready",
            "☐ Configuration file created and validated",
            "☐ Azure credentials configured (subscription_id, resource_group)",
            "☐ Database connection tested",
            "☐ Monitoring alerts configured",
            "☐ Backup strategy in place",
            "☐ Rollback procedure documented",
            "☐ Team notified of deployment",
            "☐ Deployment window scheduled"
        ]
        
        for item in checklist:
            logger.info(f"  {item}")
        
        logger.info("=" * 60 + "\n")
    
    @staticmethod
    def print_post_deployment_checklist():
        """Print post-deployment checklist"""
        logger.info("\n" + "=" * 60)
        logger.info("POST-DEPLOYMENT CHECKLIST")
        logger.info("=" * 60)
        
        checklist = [
            "☐ Health checks passing",
            "☐ Model predictions generating correctly",
            "☐ Monitoring metrics being collected",
            "☐ Alerts functioning properly",
            "☐ Logs being written to correct location",
            "☐ Azure integrations working",
            "☐ Scheduled tasks running",
            "☐ Performance metrics within baseline",
            "☐ Cost monitoring enabled",
            "☐ Team notified of successful deployment"
        ]
        
        for item in checklist:
            logger.info(f"  {item}")
        
        logger.info("=" * 60 + "\n")


def create_production_environment(config_file: Optional[str] = None) -> ProductionEnvironment:
    """Factory function to create production environment
    
    Args:
        config_file: Optional path to configuration file
    
    Returns:
        Initialized ProductionEnvironment
    """
    env = ProductionEnvironment(config_file)
    env.setup_monitoring()
    env.setup_scheduling()
    return env


# CLI interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Production Environment Manager')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--command', choices=['start', 'stop', 'status', 'checklist'], default='status')
    
    args = parser.parse_args()
    
    if args.command == 'checklist':
        ProductionDeploymentGuide.print_pre_deployment_checklist()
        sys.exit(0)
    
    # Create environment
    env = create_production_environment(args.config)
    
    if args.command == 'start':
        env.start()
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            env.stop()
    
    elif args.command == 'stop':
        env.stop()
    
    elif args.command == 'status':
        info = env.get_environment_info()
        import json
        logger.info("\n" + json.dumps(info, indent=2))
