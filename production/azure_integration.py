"""
Azure Integration Module
Integrates forecast predictions with Azure provisioning ecosystem for capacity planning
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AzureIntegrationBase(ABC):
    """Abstract base for Azure service integrations"""
    
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def provision_capacity(self, resource_id: str, capacity: float) -> Dict[str, Any]:
        pass


class AzureMonitorIntegration:
    """Integration with Azure Monitor for metrics and alerts"""
    
    def __init__(self, workspace_id: str, workspace_key: str):
        """Initialize Azure Monitor integration
        
        Args:
            workspace_id: Log Analytics workspace ID
            workspace_key: Workspace shared key
        """
        self.workspace_id = workspace_id
        self.workspace_key = workspace_key
        self.custom_log_name = "ForecastMetrics_CL"
        
        logger.info("Azure Monitor integration initialized")
    
    def send_forecast_metrics(self, forecast_data: Dict[str, Any],
                              prediction: float, region: str,
                              service_type: str) -> bool:
        """Send forecast metrics to Azure Monitor
        
        Args:
            forecast_data: Input forecast data
            prediction: Predicted usage value
            region: Azure region
            service_type: Service type (compute, storage, etc)
        
        Returns:
            Success status
        """
        try:
            # Format metrics for Azure Monitor
            timestamp = datetime.utcnow().isoformat()
            
            metrics_payload = {
                'TimeGenerated': timestamp,
                'Region': region,
                'ServiceType': service_type,
                'PredictedUsage': prediction,
                'ProvisionnedCapacity': forecast_data.get('provisioned_capacity'),
                'Availability_pct': forecast_data.get('availability_pct'),
                'EconomicIndex': forecast_data.get('economic_index'),
                'MarketDemandIndex': forecast_data.get('market_demand_index'),
                'UtilizationPercentage': (prediction / forecast_data.get('provisioned_capacity', 1)) * 100,
                'ComputerName': 'ForecastEngine'
            }
            
            # In production, this would use Azure SDK to send to Log Analytics
            # For now, log the metrics locally
            logger.info(f"📊 Forecast metrics: {json.dumps(metrics_payload)}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to send metrics to Azure Monitor: {str(e)}")
            return False
    
    def create_alert_rule(self, metric_name: str, threshold: float,
                         comparison: str = 'GreaterThan',
                         evaluation_periods: int = 1) -> Dict[str, Any]:
        """Create alert rule in Azure Monitor
        
        Args:
            metric_name: Metric to monitor
            threshold: Alert threshold
            comparison: Comparison operator (GreaterThan, LessThan, etc)
            evaluation_periods: Number of periods to evaluate
        
        Returns:
            Alert rule configuration
        """
        alert_rule = {
            'name': f'{metric_name}_Alert',
            'description': f'Alert when {metric_name} {comparison.lower()} {threshold}',
            'type': 'Microsoft.Insights/metricAlerts',
            'properties': {
                'description': f'Auto-scaling trigger for {metric_name}',
                'scopes': [f'/subscriptions/{{subscriptionId}}/resourceGroups/{{resourceGroup}}'],
                'enabled': True,
                'evaluationFrequency': 'PT5M',
                'windowSize': 'PT15M',
                'criteria': {
                    'odata.type': 'Microsoft.Azure.Monitor.MultipleResourceMultipleMetricCriteria',
                    'allOf': [
                        {
                            'name': metric_name,
                            'metricName': metric_name,
                            'operator': comparison,
                            'threshold': threshold,
                            'timeAggregation': 'Average'
                        }
                    ]
                },
                'actions': [
                    {
                        'actionGroupId': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}'
                                        '/providers/microsoft.insights/actionGroups/ForecastActionGroup'
                    }
                ]
            }
        }
        
        logger.info(f"Created alert rule: {alert_rule['name']}")
        return alert_rule


class AzureAutoscaleIntegration:
    """Integration with Azure Autoscale for automated resource scaling"""
    
    def __init__(self, subscription_id: str, resource_group: str):
        """Initialize autoscale integration
        
        Args:
            subscription_id: Azure subscription ID
            resource_group: Resource group name
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.autoscale_settings = {}
        
        logger.info("Azure Autoscale integration initialized")
    
    def create_autoscale_rule(self, resource_name: str, resource_type: str,
                             metric_name: str, scale_threshold: float,
                             min_capacity: int = 1, max_capacity: int = 100) -> Dict[str, Any]:
        """Create autoscale rule based on forecast
        
        Args:
            resource_name: Name of resource to autoscale
            resource_type: Type of resource (VirtualMachineScaleSets, AppServicePlan, etc)
            metric_name: Metric to trigger scaling
            scale_threshold: Threshold to trigger scaling
            min_capacity: Minimum instance count
            max_capacity: Maximum instance count
        
        Returns:
            Autoscale rule configuration
        """
        autoscale_rule = {
            'name': f'{resource_name}_AutoscaleRule',
            'resource_id': f'/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}'
                          f'/providers/{resource_type}/{resource_name}',
            'enabled': True,
            'profiles': [
                {
                    'name': 'ForecastDrivenScaling',
                    'capacity': {
                        'minimum': str(min_capacity),
                        'maximum': str(max_capacity),
                        'default': str(min_capacity)
                    },
                    'rules': [
                        {
                            'metricTrigger': {
                                'metricName': metric_name,
                                'metricResourceId': f'/subscriptions/{self.subscription_id}'
                                                   f'/resourceGroups/{self.resource_group}'
                                                   f'/providers/{resource_type}/{resource_name}',
                                'timeGrain': 'PT1M',
                                'statistic': 'Average',
                                'timeWindow': 'PT5M',
                                'timeAggregation': 'Average',
                                'operator': 'GreaterThan',
                                'threshold': scale_threshold
                            },
                            'scaleAction': {
                                'direction': 'Increase',
                                'type': 'PercentChangeCount',
                                'value': '25',
                                'cooldown': 'PT5M'
                            }
                        }
                    ]
                }
            ]
        }
        
        self.autoscale_settings[resource_name] = autoscale_rule
        logger.info(f"Created autoscale rule for {resource_name}")
        return autoscale_rule
    
    def get_scaling_recommendations(self, current_utilization: float,
                                   predicted_utilization: float,
                                   current_capacity: int) -> Dict[str, Any]:
        """Get resource scaling recommendations based on forecast
        
        Args:
            current_utilization: Current utilization percentage
            predicted_utilization: Predicted utilization percentage
            current_capacity: Current resource capacity
        
        Returns:
            Scaling recommendations
        """
        recommendations = {
            'current_utilization': current_utilization,
            'predicted_utilization': predicted_utilization,
            'current_capacity': current_capacity,
            'timestamp': datetime.utcnow().isoformat(),
            'actions': []
        }
        
        # Scaling logic based on predictions
        if predicted_utilization > 85:
            scale_up_to = int(current_capacity * 1.5)
            recommendations['actions'].append({
                'action': 'SCALE_UP',
                'reason': 'High predicted utilization',
                'recommended_capacity': scale_up_to,
                'priority': 'HIGH',
                'urgency': 'IMMEDIATE' if predicted_utilization > 95 else 'SOON'
            })
        elif predicted_utilization > 75:
            recommendations['actions'].append({
                'action': 'PREPARE_SCALE_UP',
                'reason': 'Moderate high predicted utilization',
                'recommended_capacity': int(current_capacity * 1.25),
                'priority': 'MEDIUM',
                'urgency': 'PLANNED'
            })
        elif predicted_utilization < 40 and current_utilization < 50:
            scale_down_to = max(1, int(current_capacity * 0.75))
            recommendations['actions'].append({
                'action': 'SCALE_DOWN',
                'reason': 'Low predicted utilization - cost optimization',
                'recommended_capacity': scale_down_to,
                'priority': 'LOW',
                'urgency': 'PLANNED'
            })
        else:
            recommendations['actions'].append({
                'action': 'NO_CHANGE',
                'reason': 'Utilization within optimal range',
                'priority': 'INFO'
            })
        
        return recommendations


class AzureCostOptimization:
    """Integration for cost optimization recommendations"""
    
    def __init__(self):
        logger.info("Azure Cost Optimization integration initialized")
    
    def calculate_cost_impact(self, current_capacity: int, recommended_capacity: int,
                             hourly_rate: float = 0.1) -> Dict[str, Any]:
        """Calculate cost impact of scaling decisions
        
        Args:
            current_capacity: Current capacity units
            recommended_capacity: Recommended capacity units
            hourly_rate: Cost per capacity unit per hour
        
        Returns:
            Cost impact analysis
        """
        daily_cost_current = current_capacity * hourly_rate * 24
        daily_cost_recommended = recommended_capacity * hourly_rate * 24
        monthly_cost_current = daily_cost_current * 30
        monthly_cost_recommended = daily_cost_recommended * 30
        
        cost_change = monthly_cost_recommended - monthly_cost_current
        cost_change_pct = (cost_change / monthly_cost_current * 100) if monthly_cost_current > 0 else 0
        
        return {
            'current_daily_cost': round(daily_cost_current, 2),
            'recommended_daily_cost': round(daily_cost_recommended, 2),
            'current_monthly_cost': round(monthly_cost_current, 2),
            'recommended_monthly_cost': round(monthly_cost_recommended, 2),
            'monthly_cost_change': round(cost_change, 2),
            'monthly_cost_change_pct': round(cost_change_pct, 2),
            'annual_cost_change': round(cost_change * 12, 2),
            'recommendation': 'COST_SAVING' if cost_change < 0 else 'COST_INCREASE'
        }
    
    def get_rightsizing_opportunities(self, utilization_history: List[float],
                                     current_capacity: int) -> Dict[str, Any]:
        """Identify rightsizing opportunities
        
        Args:
            utilization_history: Historical utilization percentages
            current_capacity: Current capacity
        
        Returns:
            Rightsizing opportunities
        """
        avg_utilization = sum(utilization_history) / len(utilization_history)
        peak_utilization = max(utilization_history)
        min_utilization = min(utilization_history)
        
        # Recommend capacity based on peak + buffer
        recommended_capacity = int(current_capacity * (peak_utilization + 15) / 100)
        
        opportunities = {
            'average_utilization': round(avg_utilization, 2),
            'peak_utilization': round(peak_utilization, 2),
            'minimum_utilization': round(min_utilization, 2),
            'current_capacity': current_capacity,
            'recommended_capacity': recommended_capacity,
            'efficiency_improvement': round(((current_capacity - recommended_capacity) / current_capacity) * 100, 2)
        }
        
        if opportunities['efficiency_improvement'] > 10:
            opportunities['opportunity'] = 'SIGNIFICANT_RIGHTSIZING'
        elif opportunities['efficiency_improvement'] > 5:
            opportunities['opportunity'] = 'MODERATE_RIGHTSIZING'
        else:
            opportunities['opportunity'] = 'MINIMAL_IMPROVEMENT'
        
        return opportunities


class CapacityPlanningIntegration:
    """Integration with capacity planning systems"""
    
    def __init__(self):
        self.capacity_plans = {}
        logger.info("Capacity Planning integration initialized")
    
    def generate_capacity_plan(self, region: str, service_type: str,
                              forecast_data: List[Dict], forecast_horizon_days: int = 30) -> Dict[str, Any]:
        """Generate capacity plan based on forecasts
        
        Args:
            region: Azure region
            service_type: Service type
            forecast_data: List of forecast data points
            forecast_horizon_days: Planning horizon in days
        
        Returns:
            Capacity plan
        """
        # Calculate capacity requirements
        required_capacity_data = []
        for data_point in forecast_data:
            required = data_point.get('predicted_usage', 0) * 1.2  # Add 20% buffer
            required_capacity_data.append(required)
        
        capacity_plan = {
            'region': region,
            'service_type': service_type,
            'created_at': datetime.utcnow().isoformat(),
            'forecast_horizon_days': forecast_horizon_days,
            'current_capacity': forecast_data[0].get('provisioned_capacity', 0) if forecast_data else 0,
            'required_capacity_avg': sum(required_capacity_data) / len(required_capacity_data) if required_capacity_data else 0,
            'required_capacity_peak': max(required_capacity_data) if required_capacity_data else 0,
            'required_capacity_min': min(required_capacity_data) if required_capacity_data else 0,
            'milestones': self._generate_milestones(required_capacity_data, forecast_horizon_days),
            'risk_factors': self._identify_risk_factors(forecast_data)
        }
        
        self.capacity_plans[f'{region}_{service_type}'] = capacity_plan
        return capacity_plan
    
    def _generate_milestones(self, capacity_data: List[float], horizon_days: int) -> List[Dict]:
        """Generate capacity milestones"""
        milestones = []
        step = max(1, len(capacity_data) // 4)  # Quarterly milestones
        
        for i in range(0, len(capacity_data), step):
            milestone_day = int((i / len(capacity_data)) * horizon_days) if capacity_data else 0
            milestones.append({
                'day': milestone_day,
                'required_capacity': round(capacity_data[i], 2) if i < len(capacity_data) else 0,
                'date': (datetime.utcnow() + timedelta(days=milestone_day)).isoformat()
            })
        
        return milestones
    
    def _identify_risk_factors(self, forecast_data: List[Dict]) -> List[Dict]:
        """Identify capacity planning risk factors"""
        risks = []
        
        # Check for high utilization
        high_util = [d for d in forecast_data if (d.get('predicted_usage', 0) / d.get('provisioned_capacity', 1)) > 0.9]
        if high_util:
            risks.append({
                'factor': 'HIGH_UTILIZATION',
                'severity': 'HIGH',
                'count': len(high_util),
                'recommendation': 'Urgent scaling needed'
            })
        
        # Check for volatility
        predictions = [d.get('predicted_usage', 0) for d in forecast_data]
        if predictions:
            std_dev = (sum((x - sum(predictions)/len(predictions))**2 for x in predictions) / len(predictions)) ** 0.5
            mean = sum(predictions) / len(predictions)
            if mean > 0 and std_dev / mean > 0.3:
                risks.append({
                    'factor': 'HIGH_VOLATILITY',
                    'severity': 'MEDIUM',
                    'coefficient_of_variation': round(std_dev / mean, 2),
                    'recommendation': 'Consider flexible scaling policies'
                })
        
        return risks
    
    def get_capacity_plan(self, region: str, service_type: str) -> Optional[Dict]:
        """Retrieve capacity plan"""
        return self.capacity_plans.get(f'{region}_{service_type}')
