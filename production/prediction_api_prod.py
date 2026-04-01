"""
Production Prediction API - Enhanced
Integrates model serving, monitoring, Azure integration, and deployment management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import os
from pathlib import Path

from production.deployment_manager import ProductionDeploymentManager
from production.monitoring import ModelPerformanceMonitor, AnomalyDetector, AlertManager
from production.azure_integration import AzureMonitorIntegration, AzureAutoscaleIntegration, CapacityPlanningIntegration
from production.reporting import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Azure Demand Forecast API",
    description="Production-ready forecasting API with monitoring and Azure integration",
    version="1.0.0"
)

# Initialize components
deployment_manager = ProductionDeploymentManager()
performance_monitor = ModelPerformanceMonitor()
anomaly_detector = AnomalyDetector()
alert_manager = AlertManager()
report_generator = ReportGenerator()

# Azure integrations
azure_monitor = AzureMonitorIntegration(
    workspace_id=os.getenv('AZURE_WORKSPACE_ID', 'default'),
    workspace_key=os.getenv('AZURE_WORKSPACE_KEY', 'default')
)
azure_autoscale = AzureAutoscaleIntegration(
    subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID', 'default'),
    resource_group=os.getenv('AZURE_RESOURCE_GROUP', 'default')
)
capacity_planning = CapacityPlanningIntegration()

# Load production model
try:
    model_path = os.path.join(deployment_manager.prod_models_dir, 'model_active/model.pkl')
    features_path = os.path.join(deployment_manager.prod_models_dir, 'model_active/features.pkl')
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        logger.info("✅ Production model loaded successfully")
    else:
        model = None
        features = None
        logger.warning("⚠️ Production model not found, API will operate in demo mode")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None
    features = None


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    status = {
        'status': 'HEALTHY' if model is not None else 'DEGRADED',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': model is not None,
        'active_version': deployment_manager.active_version,
        'components': {
            'api': 'UP',
            'model': 'UP' if model is not None else 'DOWN',
            'monitoring': 'UP'
        }
    }
    return status


@app.get("/model/info")
async def get_model_info():
    """Get information about active production model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return deployment_manager.get_active_model_info()


@app.post("/predict")
async def predict(
    data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    track_metrics: bool = True
):
    """
    Generate forecast prediction with monitoring
    
    Args:
        data: Input features (provisioned_capacity, availability_pct, economic_index, market_demand_index)
        track_metrics: Whether to track prediction metrics
    
    Returns:
        Prediction with metadata and recommendations
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        start_time = datetime.utcnow()
        
        # Prepare input data
        df = pd.DataFrame([data])
        for col in features:
            if col not in df.columns:
                df[col] = 0
        df = df[features]
        
        # Generate prediction
        prediction = model.predict(df)[0]
        inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Calculate derived metrics
        capacity = data.get('provisioned_capacity', 100)
        utilization = (prediction / capacity) * 100
        region = data.get('region', 'unknown')
        service_type = data.get('service_type', 'compute')
        
        # Track metrics asynchronously
        if track_metrics:
            background_tasks.add_task(
                _track_prediction_metrics,
                prediction, inference_time, data, region, service_type, utilization
            )
        
        # Generate recommendations
        recommendations = _generate_recommendations(utilization, capacity, prediction)
        
        # Send to Azure Monitor
        background_tasks.add_task(
            azure_monitor.send_forecast_metrics,
            data, prediction, region, service_type
        )
        
        response = {
            'prediction': {
                'predicted_usage': round(prediction, 2),
                'confidence': 0.92,  # Placeholder - would come from model
                'timestamp': start_time.isoformat()
            },
            'utilization': {
                'predicted_percentage': round(utilization, 2),
                'capacity': capacity,
                'status': 'HIGH_UTILIZATION' if utilization > 85 else 'NORMAL'
            },
            'recommendations': recommendations,
            'metadata': {
                'region': region,
                'service_type': service_type,
                'inference_time_ms': round(inference_time, 2),
                'model_version': deployment_manager.active_version
            }
        }
        
        logger.info(f"✅ Prediction generated successfully (inference: {inference_time:.1f}ms)")
        return response
    
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(
    predictions: list,
    background_tasks: BackgroundTasks
):
    """Generate batch predictions
    
    Args:
        predictions: List of prediction requests
    
    Returns:
        List of predictions
    """
    results = []
    
    for pred_data in predictions:
        try:
            df = pd.DataFrame([pred_data])
            for col in features:
                if col not in df.columns:
                    df[col] = 0
            df = df[features]
            
            prediction = model.predict(df)[0]
            results.append({
                'input': pred_data,
                'prediction': round(prediction, 2),
                'status': 'SUCCESS'
            })
        except Exception as e:
            results.append({
                'input': pred_data,
                'error': str(e),
                'status': 'FAILED'
            })
    
    logger.info(f"Batch prediction completed: {len(results)} predictions")
    return {'predictions': results}


@app.get("/monitoring/metrics")
async def get_metrics():
    """Get current performance metrics"""
    metrics = performance_monitor.get_current_metrics()
    return metrics


@app.get("/monitoring/health")
async def get_system_health():
    """Get system health status"""
    return {
        'system_health': {
            'status': 'HEALTHY',
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_hours': 72,  # Placeholder
            'metrics': performance_monitor.get_current_metrics()
        }
    }


@app.get("/alerts/summary")
async def get_alerts_summary():
    """Get alert summary"""
    return alert_manager.get_alert_summary()


@app.get("/deployment/history")
async def get_deployment_history(limit: int = 10):
    """Get deployment history"""
    history_df = deployment_manager.get_deployment_history(limit=limit)
    return {
        'deployments': history_df.to_dict(orient='records') if not history_df.empty else []
    }


@app.post("/deployment/stage")
async def stage_model(
    model_file_path: str,
    features_file_path: str,
    version: str,
    metrics: Dict[str, Any]
):
    """Stage model for production deployment"""
    try:
        result = deployment_manager.stage_model(
            model_file_path, features_file_path, version, metrics
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/deployment/approve")
async def approve_deployment(version: str, approval_notes: str = ""):
    """Approve and deploy staged model to production"""
    try:
        result = deployment_manager.deploy_to_production(version, approval_notes)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/deployment/rollback")
async def rollback_deployment(reason: str = ""):
    """Rollback to previous model version"""
    try:
        result = deployment_manager.rollback(reason)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/capacity-planning/recommendations")
async def get_capacity_recommendations(
    region: str,
    service_type: str,
    current_utilization: float,
    predicted_utilization: float,
    current_capacity: int
):
    """Get capacity scaling recommendations"""
    recommendations = azure_autoscale.get_scaling_recommendations(
        current_utilization, predicted_utilization, current_capacity
    )
    
    cost_impact = _calculate_cost_impact(
        recommendations, current_capacity
    )
    
    return {
        'recommendations': recommendations,
        'cost_impact': cost_impact
    }


@app.get("/reports/latest")
async def get_latest_report():
    """Get latest report"""
    reports_dir = Path(report_generator.reports_dir)
    
    if not reports_dir.exists():
        raise HTTPException(status_code=404, detail="No reports available")
    
    # Get most recent report
    report_files = list(reports_dir.glob('*.json'))
    if not report_files:
        raise HTTPException(status_code=404, detail="No reports available")
    
    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_report, 'r') as f:
        import json
        return json.load(f)


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        'api': 'Azure Demand Forecast Production API',
        'version': '1.0.0',
        'status': 'API_OPERATIONAL' if model is not None else 'API_DEGRADED',
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
            'batch_predict': '/predict/batch',
            'metrics': '/monitoring/metrics',
            'alerts': '/alerts/summary',
            'deployments': '/deployment/history',
            'recommendations': '/capacity-planning/recommendations',
            'reports': '/reports/latest'
        }
    }


def _track_prediction_metrics(prediction: float, inference_time: float,
                              data: Dict[str, Any], region: str,
                              service_type: str, utilization: float):
    """Track prediction metrics (background task)"""
    try:
        performance_monitor.record_prediction(
            prediction, inference_time_ms=inference_time, metadata={
                'region': region,
                'service_type': service_type
            }
        )
        
        # Check for anomalies
        anomaly = anomaly_detector.detect(utilization, 'utilization_pct')
        if anomaly:
            logger.warning(f"Anomaly detected: {anomaly}")
    
    except Exception as e:
        logger.error(f"Error tracking metrics: {str(e)}")


def _generate_recommendations(utilization: float, capacity: int, prediction: float) -> Dict[str, Any]:
    """Generate infrastructure recommendations based on prediction"""
    recommendations = {
        'scaling': None,
        'cost_optimization': None,
        'monitoring': None
    }
    
    if utilization > 85:
        recommendations['scaling'] = {
            'action': 'SCALE_UP',
            'priority': 'HIGH',
            'recommended_capacity': int(capacity * 1.5),
            'reason': 'High predicted utilization exceeds safe threshold'
        }
    elif utilization > 75:
        recommendations['scaling'] = {
            'action': 'PREPARE_SCALE_UP',
            'priority': 'MEDIUM',
            'recommended_capacity': int(capacity * 1.25),
            'reason': 'Moderate utilization trend detected'
        }
    elif utilization < 40:
        recommendations['cost_optimization'] = {
            'action': 'SCALE_DOWN',
            'priority': 'LOW',
            'recommended_capacity': int(capacity * 0.75),
            'reason': 'Low utilization - cost optimization opportunity'
        }
    
    recommendations['monitoring'] = {
        'action': 'CONTINUE_MONITORING',
        'check_interval_minutes': 15,
        'alert_threshold': 0.9
    }
    
    return recommendations


def _calculate_cost_impact(recommendations: Dict, current_capacity: int) -> Dict[str, Any]:
    """Calculate cost impact of recommendations"""
    recommended_capacity = recommendations.get('recommendations', [{}])[0].get('recommended_capacity', current_capacity)
    
    return {
        'current_monthly_cost': current_capacity * 0.1 * 24 * 30,  # Placeholder calculation
        'recommended_monthly_cost': recommended_capacity * 0.1 * 24 * 30,
        'estimated_monthly_savings': (current_capacity - recommended_capacity) * 0.1 * 24 * 30
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
