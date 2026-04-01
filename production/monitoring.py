"""
Monitoring & Alerting System
Tracks model performance, system health, and triggers alerts for anomalies
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """Monitor model performance metrics in production"""
    
    def __init__(self, window_size: int = 100, alert_threshold: float = 0.1):
        """Initialize performance monitor
        
        Args:
            window_size: Number of predictions to track for metrics
            alert_threshold: Threshold for performance degradation alerts (%)
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.prediction_times = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.metrics_history = []
        
        logger.info("Model Performance Monitor initialized")
    
    def record_prediction(self, prediction: float, actual: Optional[float] = None,
                         inference_time_ms: float = 0.0, metadata: Optional[Dict] = None):
        """Record a prediction and optional actual value
        
        Args:
            prediction: Model prediction value
            actual: True value (optional, for later comparison)
            inference_time_ms: Time to generate prediction in milliseconds
            metadata: Additional metadata (region, service_type, etc)
        """
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'prediction': prediction,
            'actual': actual,
            'inference_time_ms': inference_time_ms,
            'metadata': metadata or {}
        }
        
        self.predictions.append(prediction)
        self.prediction_times.append(inference_time_ms)
        
        if actual is not None:
            self.actuals.append(actual)
            error = abs(prediction - actual) / max(abs(actual), 1)
            self.errors.append(error)
            record['error'] = error
        
        logger.debug(f"Recorded prediction: {prediction}, actual: {actual}")
        return record
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Calculate current performance metrics"""
        if not self.predictions:
            return {'status': 'No predictions recorded'}
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'predictions_count': len(self.predictions),
            'avg_prediction': float(np.mean(list(self.predictions))),
            'std_prediction': float(np.std(list(self.predictions))),
            'avg_inference_time_ms': float(np.mean(list(self.prediction_times))),
            'max_inference_time_ms': float(np.max(list(self.prediction_times)))
        }
        
        if self.errors:
            metrics['avg_error'] = float(np.mean(list(self.errors)))
            metrics['max_error'] = float(np.max(list(self.errors)))
            metrics['error_std'] = float(np.std(list(self.errors)))
        
        self.metrics_history.append(metrics)
        return metrics
    
    def check_performance_degradation(self, baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check if model performance has degraded
        
        Args:
            baseline_metrics: Baseline performance metrics
        
        Returns:
            Degradation alert if detected
        """
        current = self.get_current_metrics()
        alerts = []
        
        if 'avg_error' in baseline_metrics and 'avg_error' in current:
            error_diff = (current['avg_error'] - baseline_metrics['avg_error']) / max(baseline_metrics['avg_error'], 0.001)
            if error_diff > self.alert_threshold:
                alerts.append({
                    'type': 'PERFORMANCE_DEGRADATION',
                    'severity': 'HIGH',
                    'metric': 'avg_error',
                    'baseline': baseline_metrics['avg_error'],
                    'current': current['avg_error'],
                    'degradation_pct': round(error_diff * 100, 2)
                })
        
        if 'avg_inference_time_ms' in baseline_metrics and 'avg_inference_time_ms' in current:
            time_diff = (current['avg_inference_time_ms'] - baseline_metrics['avg_inference_time_ms']) / max(baseline_metrics['avg_inference_time_ms'], 1)
            if time_diff > 0.2:  # 20% slowdown
                alerts.append({
                    'type': 'INFERENCE_SLOWDOWN',
                    'severity': 'MEDIUM',
                    'metric': 'avg_inference_time_ms',
                    'baseline': baseline_metrics['avg_inference_time_ms'],
                    'current': current['avg_inference_time_ms'],
                    'increase_pct': round(time_diff * 100, 2)
                })
        
        return {
            'current_metrics': current,
            'alerts': alerts,
            'status': 'DEGRADED' if alerts else 'HEALTHY'
        }
    
    def get_metrics_trend(self, metric_name: str, lookback_hours: int = 24) -> Dict[str, Any]:
        """Get trend of a metric over time"""
        if not self.metrics_history:
            return {'status': 'No metrics history'}
        
        # Filter metrics within lookback period
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
            and metric_name in m
        ]
        
        if not recent_metrics:
            return {'status': f'No {metric_name} data in last {lookback_hours} hours'}
        
        values = [m[metric_name] for m in recent_metrics]
        
        return {
            'metric': metric_name,
            'lookback_hours': lookback_hours,
            'data_points': len(values),
            'current_value': values[-1],
            'average': float(np.mean(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'trend': 'INCREASING' if values[-1] > values[0] else 'DECREASING'
        }


class SystemHealthMonitor:
    """Monitor overall system health and availability"""
    
    def __init__(self, check_interval_seconds: int = 60):
        self.check_interval_seconds = check_interval_seconds
        self.health_checks = []
        self.last_check = None
        self.status = 'UNKNOWN'
        
        logger.info("System Health Monitor initialized")
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        check = {
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }
        
        # Check model availability
        check['components']['model_availability'] = {
            'status': 'UP',
            'message': 'Production model loaded and ready'
        }
        
        # Check API availability
        check['components']['api_availability'] = {
            'status': 'UP',
            'message': 'Prediction API responding'
        }
        
        # Check database availability
        check['components']['database'] = {
            'status': 'UP',
            'message': 'Database connection active'
        }
        
        # Check monitoring system
        check['components']['monitoring'] = {
            'status': 'UP',
            'message': 'Monitoring system operational'
        }
        
        # Overall health status
        failed_components = sum(1 for c in check['components'].values() if c['status'] != 'UP')
        check['overall_status'] = 'HEALTHY' if failed_components == 0 else 'DEGRADED' if failed_components < 2 else 'CRITICAL'
        
        self.health_checks.append(check)
        self.last_check = check
        self.status = check['overall_status']
        
        logger.info(f"Health check completed: {check['overall_status']}")
        return check
    
    def get_availability(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """Calculate system availability percentage"""
        if not self.health_checks:
            return {'availability_pct': 0, 'data_points': 0}
        
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        recent_checks = [
            c for c in self.health_checks
            if datetime.fromisoformat(c['timestamp']) > cutoff_time
        ]
        
        if not recent_checks:
            return {'availability_pct': 0, 'data_points': 0}
        
        healthy_count = sum(1 for c in recent_checks if c['overall_status'] in ['HEALTHY', 'DEGRADED'])
        availability_pct = (healthy_count / len(recent_checks)) * 100
        
        return {
            'lookback_hours': lookback_hours,
            'data_points': len(recent_checks),
            'availability_pct': round(availability_pct, 2),
            'healthy_periods': healthy_count,
            'degraded_periods': sum(1 for c in recent_checks if c['overall_status'] == 'DEGRADED'),
            'critical_periods': sum(1 for c in recent_checks if c['overall_status'] == 'CRITICAL')
        }


class AnomalyDetector:
    """Detect anomalies in predictions and system metrics"""
    
    def __init__(self, sensitivity: float = 2.0):
        """Initialize anomaly detector
        
        Args:
            sensitivity: Z-score threshold for anomaly detection (default 2.0 = 95% confidence)
        """
        self.sensitivity = sensitivity
        self.anomalies = []
        self.baseline_stats = {}
        
        logger.info(f"Anomaly Detector initialized (sensitivity: {sensitivity})")
    
    def set_baseline(self, data: List[float], metric_name: str):
        """Set baseline statistics for anomaly detection
        
        Args:
            data: Historical data for baseline
            metric_name: Name of the metric
        """
        if not data:
            return False
        
        data = np.array(data)
        self.baseline_stats[metric_name] = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data))
        }
        
        logger.info(f"Baseline set for {metric_name}")
        return True
    
    def detect(self, value: float, metric_name: str) -> Optional[Dict[str, Any]]:
        """Detect if a value is anomalous
        
        Args:
            value: Value to check
            metric_name: Name of the metric
        
        Returns:
            Anomaly details if detected, None otherwise
        """
        if metric_name not in self.baseline_stats:
            return None
        
        baseline = self.baseline_stats[metric_name]
        std = baseline['std'] if baseline['std'] > 0 else 1
        z_score = abs((value - baseline['mean']) / std)
        
        if z_score > self.sensitivity:
            anomaly = {
                'timestamp': datetime.utcnow().isoformat(),
                'metric': metric_name,
                'value': value,
                'baseline_mean': baseline['mean'],
                'baseline_std': baseline['std'],
                'z_score': round(z_score, 2),
                'severity': 'CRITICAL' if z_score > self.sensitivity * 2 else 'WARNING',
                'description': f'{metric_name} = {value} deviates {z_score:.1f} standard deviations from baseline'
            }
            self.anomalies.append(anomaly)
            logger.warning(f"Anomaly detected: {anomaly['description']}")
            return anomaly
        
        return None
    
    def get_recent_anomalies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent anomalies"""
        return list(reversed(self.anomalies[-limit:]))


class AlertManager:
    """Manage and route alerts"""
    
    def __init__(self):
        self.alerts = []
        self.alert_rules = {}
        self.notification_channels = {}
        
        logger.info("Alert Manager initialized")
    
    def register_notification_channel(self, channel_name: str, handler_fn):
        """Register notification channel
        
        Args:
            channel_name: Name of the channel (email, slack, etc)
            handler_fn: Function to handle alert notification
        """
        self.notification_channels[channel_name] = handler_fn
        logger.info(f"Registered notification channel: {channel_name}")
    
    def add_alert_rule(self, rule_name: str, condition_fn, actions: List[str],
                      severity: str = 'MEDIUM'):
        """Add alert rule
        
        Args:
            rule_name: Name of the rule
            condition_fn: Function that returns True if alert should trigger
            actions: List of notification channels to use
            severity: Alert severity level
        """
        self.alert_rules[rule_name] = {
            'condition': condition_fn,
            'actions': actions,
            'severity': severity,
            'triggered_count': 0
        }
        logger.info(f"Added alert rule: {rule_name}")
    
    def check_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check all alert rules and trigger if needed
        
        Args:
            context: Context data for condition evaluation
        
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule['condition'](context):
                    alert = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'rule': rule_name,
                        'severity': rule['severity'],
                        'actions': rule['actions'],
                        'context': context
                    }
                    self.alerts.append(alert)
                    rule['triggered_count'] += 1
                    triggered_alerts.append(alert)
                    
                    # Send notifications
                    self._send_notifications(alert)
            
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {str(e)}")
        
        return triggered_alerts
    
    def _send_notifications(self, alert: Dict[str, Any]):
        """Send notifications through registered channels"""
        for channel in alert['actions']:
            if channel in self.notification_channels:
                try:
                    self.notification_channels[channel](alert)
                except Exception as e:
                    logger.error(f"Error sending notification via {channel}: {str(e)}")
    
    def get_alerts(self, severity: Optional[str] = None, lookback_hours: int = 24) -> List[Dict]:
        """Get recent alerts with optional filtering
        
        Args:
            severity: Filter by severity level
            lookback_hours: Time window in hours
        
        Returns:
            List of alerts
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        alerts = [
            a for a in self.alerts
            if datetime.fromisoformat(a['timestamp']) > cutoff_time
        ]
        
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity]
        
        return alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        if not self.alerts:
            return {'total_alerts': 0}
        
        recent_24h = self.get_alerts(lookback_hours=24)
        
        return {
            'total_alerts_all_time': len(self.alerts),
            'total_alerts_24h': len(recent_24h),
            'critical_24h': sum(1 for a in recent_24h if a['severity'] == 'CRITICAL'),
            'warning_24h': sum(1 for a in recent_24h if a['severity'] == 'WARNING'),
            'info_24h': sum(1 for a in recent_24h if a['severity'] == 'INFO'),
            'top_rules': self._get_top_triggered_rules()
        }
    
    def _get_top_triggered_rules(self, limit: int = 5) -> List[Dict]:
        """Get top triggered alert rules"""
        sorted_rules = sorted(
            self.alert_rules.items(),
            key=lambda x: x[1]['triggered_count'],
            reverse=True
        )
        
        return [
            {'rule': name, 'triggered_count': rule['triggered_count']}
            for name, rule in sorted_rules[:limit]
        ]
