"""
Automated Reporting Engine
Generate comprehensive reports on model performance, forecasts, and infrastructure actions
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive automated reports"""
    
    def __init__(self, reports_dir: str = 'production/reports'):
        self.reports_dir = reports_dir
        Path(reports_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Report Generator initialized")
    
    def generate_daily_report(self, metrics: Dict[str, Any], alerts: List[Dict],
                             predictions: List[Dict], recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate daily executive summary report
        
        Args:
            metrics: Performance metrics
            alerts: Alert events from the day
            predictions: Predictions made
            recommendations: Infrastructure recommendations
        
        Returns:
            Daily report
        """
        report_date = datetime.utcnow().date()
        
        report = {
            'report_type': 'DAILY_EXECUTIVE_SUMMARY',
            'date': report_date.isoformat(),
            'generated_at': datetime.utcnow().isoformat(),
            'executive_summary': {
                'status': 'OPERATIONAL',
                'predictions_processed': len(predictions),
                'alerts_triggered': len(alerts),
                'critical_alerts': sum(1 for a in alerts if a.get('severity') == 'CRITICAL'),
                'recommendations_pending': len([r for r in recommendations.values() if r.get('action') != 'NO_CHANGE'])
            },
            'performance_metrics': self._summarize_metrics(metrics),
            'alerts_summary': self._summarize_alerts(alerts),
            'forecast_summary': self._summarize_predictions(predictions),
            'infrastructure_recommendations': recommendations,
            'key_actions': self._generate_key_actions(alerts, recommendations),
            'next_steps': self._generate_next_steps(alerts, recommendations)
        }
        
        self._save_report(report, f'daily_report_{report_date.isoformat()}.json')
        logger.info(f"Daily report generated for {report_date}")
        
        return report
    
    def generate_weekly_report(self, daily_reports: List[Dict], trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate weekly trend and performance report
        
        Args:
            daily_reports: List of daily reports from the week
            trend_data: Trend analysis data
        
        Returns:
            Weekly report
        """
        report_week = datetime.utcnow().isocalendar()[1]
        report_year = datetime.utcnow().year
        
        report = {
            'report_type': 'WEEKLY_TREND_ANALYSIS',
            'week': report_week,
            'year': report_year,
            'generated_at': datetime.utcnow().isoformat(),
            'week_summary': {
                'total_predictions': sum(d.get('executive_summary', {}).get('predictions_processed', 0) for d in daily_reports),
                'total_alerts': sum(d.get('executive_summary', {}).get('alerts_triggered', 0) for d in daily_reports),
                'total_critical_alerts': sum(d.get('executive_summary', {}).get('critical_alerts', 0) for d in daily_reports),
                'days_analyzed': len(daily_reports)
            },
            'performance_trends': {
                'accuracy_trend': trend_data.get('accuracy_trend'),
                'inference_time_trend': trend_data.get('inference_time_trend'),
                'utilization_trend': trend_data.get('utilization_trend')
            },
            'top_issues': self._identify_top_issues(daily_reports),
            'recommendations': self._generate_weekly_recommendations(daily_reports),
            'capacity_trends': trend_data.get('capacity_trends'),
            'cost_analysis': trend_data.get('cost_analysis')
        }
        
        self._save_report(report, f'weekly_report_week{report_week}_{report_year}.json')
        logger.info(f"Weekly report generated for week {report_week}")
        
        return report
    
    def generate_model_performance_report(self, performance_data: Dict[str, Any],
                                         baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed model performance report
        
        Args:
            performance_data: Current performance metrics
            baseline_metrics: Baseline metrics for comparison
        
        Returns:
            Performance report
        """
        report = {
            'report_type': 'MODEL_PERFORMANCE_REPORT',
            'generated_at': datetime.utcnow().isoformat(),
            'performance_summary': {
                'current_metrics': performance_data,
                'baseline_metrics': baseline_metrics,
                'comparison': self._compare_metrics(performance_data, baseline_metrics)
            },
            'inference_statistics': {
                'avg_inference_time_ms': performance_data.get('avg_inference_time_ms'),
                'max_inference_time_ms': performance_data.get('max_inference_time_ms'),
                'throughput_predictions_per_second': 1000 / max(performance_data.get('avg_inference_time_ms', 1), 1)
            },
            'accuracy_metrics': {
                'mean_absolute_error': performance_data.get('avg_error'),
                'max_error': performance_data.get('max_error'),
                'error_std_deviation': performance_data.get('error_std')
            },
            'health_status': self._assess_health(performance_data, baseline_metrics),
            'recommendations': self._generate_performance_recommendations(performance_data, baseline_metrics)
        }
        
        self._save_report(report, f'model_performance_{datetime.utcnow().isoformat()[:10]}.json')
        logger.info("Model performance report generated")
        
        return report
    
    def generate_infrastructure_action_report(self, scaling_decisions: List[Dict],
                                             cost_impact: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report on infrastructure actions and results
        
        Args:
            scaling_decisions: List of scaling decisions made
            cost_impact: Cost impact analysis
        
        Returns:
            Infrastructure action report
        """
        report = {
            'report_type': 'INFRASTRUCTURE_ACTION_REPORT',
            'generated_at': datetime.utcnow().isoformat(),
            'summary': {
                'total_actions': len(scaling_decisions),
                'scale_up_actions': sum(1 for a in scaling_decisions if a.get('action') == 'SCALE_UP'),
                'scale_down_actions': sum(1 for a in scaling_decisions if a.get('action') == 'SCALE_DOWN'),
                'no_change_actions': sum(1 for a in scaling_decisions if a.get('action') == 'NO_CHANGE')
            },
            'scaling_decisions': scaling_decisions,
            'cost_impact': cost_impact,
            'effectiveness_metrics': self._calculate_effectiveness_metrics(scaling_decisions),
            'pending_approvals': [a for a in scaling_decisions if not a.get('approved')],
            'audit_trail': self._generate_audit_trail(scaling_decisions)
        }
        
        self._save_report(report, f'infrastructure_actions_{datetime.utcnow().isoformat()[:10]}.json')
        logger.info("Infrastructure action report generated")
        
        return report
    
    def generate_forecast_accuracy_report(self, forecast_history: List[Dict]) -> Dict[str, Any]:
        """Generate forecast accuracy analysis report
        
        Args:
            forecast_history: Historical forecast data with actuals
        
        Returns:
            Forecast accuracy report
        """
        # Calculate accuracy metrics
        errors = []
        by_region = {}
        by_service = {}
        
        for entry in forecast_history:
            if 'error' in entry:
                errors.append(entry['error'])
                
                region = entry.get('metadata', {}).get('region', 'unknown')
                service = entry.get('metadata', {}).get('service_type', 'unknown')
                
                if region not in by_region:
                    by_region[region] = []
                if service not in by_service:
                    by_service[service] = []
                
                by_region[region].append(entry['error'])
                by_service[service].append(entry['error'])
        
        # Calculate statistics
        report = {
            'report_type': 'FORECAST_ACCURACY_REPORT',
            'generated_at': datetime.utcnow().isoformat(),
            'summary': {
                'total_forecasts': len(forecast_history),
                'forecasts_with_actuals': len(errors),
                'accuracy_period_days': 30
            },
            'overall_accuracy': self._calculate_accuracy_stats(errors) if errors else {},
            'accuracy_by_region': {
                region: self._calculate_accuracy_stats(region_errors)
                for region, region_errors in by_region.items()
            },
            'accuracy_by_service': {
                service: self._calculate_accuracy_stats(service_errors)
                for service, service_errors in by_service.items()
            },
            'insights': self._generate_accuracy_insights(by_region, by_service, errors)
        }
        
        self._save_report(report, f'forecast_accuracy_{datetime.utcnow().isoformat()[:10]}.json')
        logger.info("Forecast accuracy report generated")
        
        return report
    
    def _summarize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize performance metrics"""
        return {
            'predictions_processed': metrics.get('predictions_count', 0),
            'average_inference_time_ms': round(metrics.get('avg_inference_time_ms', 0), 2),
            'average_error': round(metrics.get('avg_error', 0), 4),
            'model_status': 'HEALTHY' if metrics.get('avg_error', 0) < 0.1 else 'NEEDS_ATTENTION'
        }
    
    def _summarize_alerts(self, alerts: List[Dict]) -> Dict[str, Any]:
        """Summarize alerts"""
        return {
            'total_alerts': len(alerts),
            'critical': sum(1 for a in alerts if a.get('severity') == 'CRITICAL'),
            'warning': sum(1 for a in alerts if a.get('severity') == 'WARNING'),
            'info': sum(1 for a in alerts if a.get('severity') == 'INFO'),
            'most_common_type': self._get_most_common_alert_type(alerts)
        }
    
    def _summarize_predictions(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Summarize predictions"""
        if not predictions:
            return {'total': 0}
        
        pred_values = [p.get('prediction', 0) for p in predictions]
        
        return {
            'total': len(predictions),
            'average_prediction': round(sum(pred_values) / len(pred_values), 2),
            'min_prediction': min(pred_values),
            'max_prediction': max(pred_values),
            'by_region': self._group_predictions_by_field(predictions, 'region'),
            'by_service': self._group_predictions_by_field(predictions, 'service_type')
        }
    
    def _generate_key_actions(self, alerts: List[Dict], recommendations: Dict[str, Any]) -> List[str]:
        """Generate list of key actions to take"""
        actions = []
        
        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.get('severity') == 'CRITICAL']
        if critical_alerts:
            actions.append(f"⚠️  Address {len(critical_alerts)} critical alerts immediately")
        
        # Check for scale-up recommendations
        scale_ups = [r for r in recommendations.values() if r.get('action') == 'SCALE_UP']
        if scale_ups:
            actions.append(f"📈 Scale up {len(scale_ups)} resource(s)")
        
        # Check for optimization opportunities
        scale_downs = [r for r in recommendations.values() if r.get('action') == 'SCALE_DOWN']
        if scale_downs:
            actions.append(f"💰 Optimize costs: Scale down {len(scale_downs)} resource(s)")
        
        return actions if actions else ["✅ No immediate actions required"]
    
    def _generate_next_steps(self, alerts: List[Dict], recommendations: Dict[str, Any]) -> List[str]:
        """Generate next steps"""
        steps = [
            "Review and approve pending infrastructure actions",
            "Monitor forecast accuracy metrics",
            "Update capacity planning based on latest trends",
            "Investigate failed predictions"
        ]
        return steps
    
    def _identify_top_issues(self, daily_reports: List[Dict]) -> List[Dict]:
        """Identify top issues from daily reports"""
        issues = {}
        
        for report in daily_reports:
            alerts = report.get('alerts_summary', {})
            for severity in ['critical', 'warning']:
                key = f'top_{severity}_issues'
                if severity == 'critical':
                    count = alerts.get('critical', 0)
                    if count > 0:
                        issues['critical_alerts'] = issues.get('critical_alerts', 0) + count
                elif severity == 'warning':
                    count = alerts.get('warning', 0)
                    if count > 0:
                        issues['warning_alerts'] = issues.get('warning_alerts', 0) + count
        
        return [
            {'issue': k, 'count': v, 'priority': 'HIGH' if k == 'critical_alerts' else 'MEDIUM'}
            for k, v in sorted(issues.items(), key=lambda x: x[1], reverse=True)
        ]
    
    def _generate_weekly_recommendations(self, daily_reports: List[Dict]) -> List[str]:
        """Generate weekly recommendations"""
        recommendations = [
            "Continue monitoring forecast accuracy trends",
            "Review infrastructure scaling decisions effectiveness",
            "Analyze cost optimization opportunities",
            "Plan for capacity scaling if utilization trends continue rising"
        ]
        return recommendations
    
    def _compare_metrics(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current metrics to baseline"""
        comparison = {}
        
        for metric in ['avg_error', 'avg_inference_time_ms']:
            if metric in current and metric in baseline:
                diff = current[metric] - baseline[metric]
                diff_pct = (diff / baseline[metric] * 100) if baseline[metric] != 0 else 0
                comparison[metric] = {
                    'current': current[metric],
                    'baseline': baseline[metric],
                    'difference': diff,
                    'difference_pct': round(diff_pct, 2),
                    'status': 'IMPROVED' if diff < 0 else 'DEGRADED' if diff > 0 else 'SAME'
                }
        
        return comparison
    
    def _assess_health(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> str:
        """Assess overall model health"""
        if current.get('avg_error', 0) > baseline.get('avg_error', 1) * 1.2:
            return 'DEGRADED'
        elif current.get('avg_inference_time_ms', 0) > baseline.get('avg_inference_time_ms', 0) * 1.5:
            return 'SLOW'
        else:
            return 'HEALTHY'
    
    def _generate_performance_recommendations(self, current: Dict[str, Any],
                                            baseline: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if current.get('avg_error', 0) > baseline.get('avg_error', 0) * 1.1:
            recommendations.append("Consider retraining model - accuracy has degraded")
        
        if current.get('avg_inference_time_ms', 0) > baseline.get('avg_inference_time_ms', 0) * 1.3:
            recommendations.append("Optimize model inference - consider model compression")
        
        if not recommendations:
            recommendations.append("Model performance is stable - no action required")
        
        return recommendations
    
    def _calculate_effectiveness_metrics(self, scaling_decisions: List[Dict]) -> Dict[str, Any]:
        """Calculate effectiveness of scaling decisions"""
        if not scaling_decisions:
            return {'status': 'No decisions yet'}
        
        executed = sum(1 for d in scaling_decisions if d.get('executed'))
        successful = sum(1 for d in scaling_decisions if d.get('successful'))
        
        return {
            'total_decisions': len(scaling_decisions),
            'executed': executed,
            'successful': successful,
            'success_rate': round((successful / executed * 100) if executed > 0 else 0, 2)
        }
    
    def _generate_audit_trail(self, scaling_decisions: List[Dict]) -> List[Dict]:
        """Generate audit trail for scaling decisions"""
        return [
            {
                'decision_id': d.get('id'),
                'timestamp': d.get('timestamp'),
                'action': d.get('action'),
                'resource': d.get('resource'),
                'approved_by': d.get('approved_by'),
                'executed': d.get('executed')
            }
            for d in scaling_decisions[-10:]  # Last 10 decisions
        ]
    
    def _calculate_accuracy_stats(self, errors: List[float]) -> Dict[str, Any]:
        """Calculate accuracy statistics"""
        if not errors:
            return {}
        
        import numpy as np
        errors_array = np.array(errors)
        
        return {
            'mean_absolute_error': round(float(np.mean(errors_array)), 4),
            'median_error': round(float(np.median(errors_array)), 4),
            'std_deviation': round(float(np.std(errors_array)), 4),
            'min_error': round(float(np.min(errors_array)), 4),
            'max_error': round(float(np.max(errors_array)), 4),
            'percentile_95': round(float(np.percentile(errors_array, 95)), 4)
        }
    
    def _generate_accuracy_insights(self, by_region: Dict, by_service: Dict, overall_errors: List) -> List[str]:
        """Generate insights from accuracy data"""
        insights = []
        
        if by_region:
            best_region = min(by_region.items(), key=lambda x: sum(x[1]) / len(x[1]) if x[1] else float('inf'))
            insights.append(f"Best performing region: {best_region[0]}")
        
        if by_service:
            worst_service = max(by_service.items(), key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0)
            insights.append(f"Service needing improvement: {worst_service[0]}")
        
        return insights
    
    def _get_most_common_alert_type(self, alerts: List[Dict]) -> str:
        """Get most common alert type"""
        if not alerts:
            return 'NONE'
        
        types = {}
        for alert in alerts:
            alert_type = alert.get('rule', 'UNKNOWN')
            types[alert_type] = types.get(alert_type, 0) + 1
        
        return max(types.items(), key=lambda x: x[1])[0] if types else 'UNKNOWN'
    
    def _group_predictions_by_field(self, predictions: List[Dict], field: str) -> Dict:
        """Group predictions by a field"""
        grouped = {}
        
        for pred in predictions:
            key = pred.get('metadata', {}).get(field, 'unknown')
            if key not in grouped:
                grouped[key] = 0
            grouped[key] += 1
        
        return grouped
    
    def _save_report(self, report: Dict[str, Any], filename: str):
        """Save report to file"""
        try:
            filepath = Path(self.reports_dir) / filename
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
