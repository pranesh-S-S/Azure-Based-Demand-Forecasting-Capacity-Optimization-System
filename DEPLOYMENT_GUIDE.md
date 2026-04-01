# Production Deployment Guide

## Overview

This guide covers the complete operationalization of the Azure Demand Forecasting model for production deployment with:

- **Model Deployment Management** - Versioning, staging, deployment, and rollback
- **Real-time Prediction API** - FastAPI-based service for forecast generation
- **Monitoring & Alerting** - Performance tracking and anomaly detection
- **Azure Integration** - Capacity planning, autoscaling, and cost optimization
- **Automated Reporting** - Daily/weekly reports and metrics dashboards
- **Orchestration & Scheduling** - Automated workflows and retraining

---

## Quick Start

### 1. Prerequisites

```bash
# Install production dependencies
pip install -r requirements.txt
pip install fastapi uvicorn schedule python-dotenv azure-identity azure-monitor-opentelemetry
```

### 2. Configuration

Create `.env` file:

```bash
# Azure Configuration
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_WORKSPACE_ID=your-workspace-id
AZURE_WORKSPACE_KEY=your-workspace-key

# Database Configuration
FORECAST_DB_HOST=localhost
FORECAST_DB_PORT=5432
FORECAST_DB_NAME=forecast_db
FORECAST_DB_USER=forecast_user
FORECAST_DB_PASSWORD=your-password

# Storage
AZURE_STORAGE_CONNECTION_STRING=your-connection-string
```

### 3. Initialize Production Environment

```bash
# Check pre-deployment checklist
python -m production.initialization --command checklist

# Start production environment
python -m production.initialization --command start

# Check status
python -m production.initialization --command status
```

### 4. Start Prediction API

```bash
# Development
uvicorn production.prediction_api_prod:app --reload

# Production
gunicorn -w 4 -b 0.0.0.0:8000 production.prediction_api_prod:app
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              PRODUCTION ENVIRONMENT                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │  PREDICTION API & MODEL SERVING                  │  │
│  │  - Real-time predictions (/predict)              │  │
│  │  - Batch predictions (/predict/batch)            │  │
│  │  - Health checks (/health)                       │  │
│  └──────────────────────────────────────────────────┘  │
│                           ▲                              │
│                           │                              │
│  ┌─────────────────┬──────┴────────┬──────────────────┐ │
│  │                 │               │                  │ │
│  ▼                 ▼               ▼                  ▼ │
│ ┌──────────┐  ┌─────────────┐  ┌─────────┐  ┌────────┐ │
│ │Deployment│  │ Monitoring &│  │ Azure   │  │Reporting│
│ │Manager   │  │ Alerting    │  │Integr.  │  │ Engine  │
│ │          │  │             │  │         │  │         │
│ │- Stage   │  │- Perf Track │  │- Autoscale│- Daily  │
│ │- Deploy  │  │- Anomalies  │  │- Cost Opt│- Weekly │
│ │- Rollback│  │- Alerts     │  │- Capacity│- Reports│
│ └──────────┘  └─────────────┘  └─────────┘  └────────┘
│        │             │              │           │
│        └─────────────┴──────────────┴───────────┘
│                     │
│                     ▼
│        ┌─────────────────────────────┐
│        │ Orchestration & Scheduling   │
│        │ - Retraining workflows       │
│        │ - Report generation          │
│        │ - Health monitoring          │
│        │ - Scaling decisions          │
│        └─────────────────────────────┘
│
└─────────────────────────────────────────────────────────┘
```

---

## Deployment Process

### Step 1: Model Staging

```python
from production.deployment_manager import ProductionDeploymentManager

manager = ProductionDeploymentManager()

# Stage new model version
result = manager.stage_model(
    model_path='models/best_xgboost_model.pkl',
    features_path='models/model_features.pkl',
    version='2.0.1',
    metrics={
        'avg_error': 0.0845,
        'avg_inference_time_ms': 12.3,
        'r2_score': 0.92
    }
)
```

### Step 2: Pre-Deployment Validation

```python
# Automatic pre-deployment checks run:
# ✓ Model files exist and are loadable
# ✓ Features file is valid
# ✓ Model can generate predictions
# ✓ File integrity verified
```

### Step 3: Deploy to Production

```python
# Deploy staged model
result = manager.deploy_to_production(
    version='2.0.1',
    approval_notes='Approved by ML team - 5% performance improvement'
)

# Check active model
info = manager.get_active_model_info()
print(f"Active version: {info['active_version']}")
print(f"Deployed at: {info['deployed_at']}")
```

### Step 4: Monitoring

Once deployed, monitoring automatically:

```python
from production.monitoring import ModelPerformanceMonitor

monitor = ModelPerformanceMonitor()

# Predictions are automatically tracked
metrics = monitor.get_current_metrics()
print(f"Avg inference time: {metrics['avg_inference_time_ms']}ms")
print(f"Avg error: {metrics['avg_error']}")

# Anomalies automatically detected
anomaly = anomaly_detector.detect(utilization_value, 'utilization_pct')
```

---

## API Endpoints

### Predictions

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "provisioned_capacity": 500,
    "availability_pct": 99.5,
    "economic_index": 100,
    "market_demand_index": 105,
    "region": "east-us",
    "service_type": "compute"
  }'

# Response
{
  "prediction": {
    "predicted_usage": 425.67,
    "confidence": 0.92,
    "timestamp": "2025-03-31T12:34:56.789Z"
  },
  "utilization": {
    "predicted_percentage": 85.13,
    "capacity": 500,
    "status": "HIGH_UTILIZATION"
  },
  "recommendations": {
    "scaling": {
      "action": "SCALE_UP",
      "priority": "HIGH",
      "recommended_capacity": 750
    }
  }
}
```

### Monitoring

```bash
# Get current metrics
curl http://localhost:8000/monitoring/metrics

# Get system health
curl http://localhost:8000/monitoring/health

# Get alerts summary
curl http://localhost:8000/alerts/summary

# Get deployment history
curl http://localhost:8000/deployment/history?limit=10
```

### Deployment Management

```bash
# Stage new model
curl -X POST http://localhost:8000/deployment/stage \
  -d 'model_file_path=...' \
  -d 'features_file_path=...' \
  -d 'version=2.0.1'

# Approve deployment
curl -X POST http://localhost:8000/deployment/approve \
  -d 'version=2.0.1' \
  -d 'approval_notes=...'

# Rollback if needed
curl -X POST http://localhost:8000/deployment/rollback \
  -d 'reason=Performance degradation'
```

---

## Monitoring & Alerts

### Automated Alerts

The system automatically monitors:

1. **Performance Metrics**
   - Prediction accuracy (MAE, R²)
   - Inference time
   - Throughput

2. **System Health**
   - API availability
   - Model loading status
   - Database connectivity

3. **Business Metrics**
   - Utilization thresholds
   - Capacity gaps
   - Cost trends

4. **Data Quality**
   - Input data drift
   - Missing values
   - Outliers

### Alert Configuration

```python
from production.monitoring import AlertManager

alerts = AlertManager()

# Register notification channels
alerts.register_notification_channel('log', log_handler)
alerts.register_notification_channel('email', email_handler)
alerts.register_notification_channel('slack', slack_handler)

# Add alert rules
alerts.add_alert_rule(
    'high_prediction_error',
    lambda ctx: ctx['metrics']['avg_error'] > 0.15,
    ['log', 'email'],
    'HIGH'
)
```

---

## Azure Integration

### Autoscaling

```python
from production.azure_integration import AzureAutoscaleIntegration

autoscale = AzureAutoscaleIntegration(subscription_id, resource_group)

# Get scaling recommendations
recommendations = autoscale.get_scaling_recommendations(
    current_utilization=75,
    predicted_utilization=88,
    current_capacity=500
)

# Apply autoscale rule
rule = autoscale.create_autoscale_rule(
    resource_name='vmss-compute-prod',
    resource_type='Microsoft.Compute/virtualMachineScaleSets',
    metric_name='Percentage CPU',
    scale_threshold=80,
    min_capacity=2,
    max_capacity=20
)
```

### Cost Optimization

```python
from production.azure_integration import AzureCostOptimization

cost_opt = AzureCostOptimization()

# Calculate cost impact
impact = cost_opt.calculate_cost_impact(
    current_capacity=500,
    recommended_capacity=375,
    hourly_rate=0.1
)

# Identify rightsizing opportunities
opportunities = cost_opt.get_rightsizing_opportunities(
    utilization_history=[45, 52, 48, 51, 49],
    current_capacity=500
)
```

---

## Automated Workflows

### Daily Retraining

```python
scheduler.register_task(
    'daily_retrain',
    'Daily Model Retraining',
    task_fn=retraining_workflow,
    schedule_config={
        'frequency': 'daily',
        'time': '02:00'
    }
)
```

### Report Generation

```python
scheduler.register_task(
    'daily_reports',
    'Daily Report Generation',
    task_fn=reporting_workflow,
    schedule_config={
        'frequency': 'daily',
        'time': '06:00'
    }
)
```

### Health Monitoring

```bash
# Continuous health checks every hour
scheduler.register_task(
    'health_check',
    'System Health Check',
    task_fn=health_check_workflow,
    schedule_config={
        'frequency': 'hourly',
        'interval': 1
    }
)
```

---

## Reporting

Reports are automatically generated in `production/logs/reports/`:

### Daily Reports

- Executive summary
- Performance metrics
- Alerts triggered
- Infrastructure recommendations
- Key actions required

### Weekly Reports

- Trend analysis
- Performance trends
- Top issues
- Capacity planning updates
- Cost analysis

### Model Performance Reports

- Accuracy metrics
- Inference statistics
- Health assessment
- Recommendations

### Infrastructure Action Reports

- Scaling decisions made
- Cost impact analysis
- Effectiveness metrics
- Audit trail

---

## Troubleshooting

### Model Not Loaded

```bash
# Check deployment status
python -m production.initialization --command status

# Verify model files exist
ls -la production/models/model_active/

# Check logs
tail -f production/logs/production.log
```

### High Error Rates

```python
# Get performance analysis
metrics = monitor.get_metrics_trend('avg_error', lookback_hours=24)

# Check if retraining is needed
retraining_check = retrainer.needs_retraining(new_data)
```

### Slow Inference

```python
# Monitor inference times
inference_times = monitor.get_metrics_trend('avg_inference_time_ms', lookback_hours=24)

# Check for resource contention
```

---

## Rollback Procedure

```python
# If issues detected, rollback is simple
result = manager.rollback(reason='Performance degradation detected')

# Verify rollback
info = manager.get_active_model_info()
print(f"Rolled back to version: {info['active_version']}")
```

---

## Performance Baseline

Expected production performance:

| Metric | Target | Acceptable Range |
|--------|--------|-----------------|
| Inference Time | <20ms | <50ms |
| Avg Error | <0.10 | <0.15 |
| Throughput | >50 pred/sec | >40 pred/sec |
| Availability | 99.9% | >99% |
| Cost per 1k predictions | $0.05 | <$0.10 |

---

## Support & Escalation

### Critical Issues

- Page on-call engineer
- Initiate rollback if needed
- Notify stakeholders

### High Priority

- Alert team within 1 hour
- Create incident ticket
- Begin investigation

### Medium Priority

- Log in tracking system
- Schedule for next review
- Monitor trends

---

## Security

### API Security

- API key authentication required
- SSL/TLS for all connections
- CORS restrictions configured
- Rate limiting enabled

### Data Security

- Model files encrypted at rest
- Predictions encrypted in transit
- Logs anonymized where needed
- Access controls enforced

---

## Maintenance

### Daily

- Monitor system health
- Review alerts
- Check inference metrics

### Weekly

- Review performance trends
- Analyze cost impacts
- Audit scaling decisions

### Monthly

- Performance analysis
- Capacity planning review
- Security audit
- Cost optimization review

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Azure Autoscale](https://docs.microsoft.com/en-us/azure/azure-monitor/autoscale/)
- [Production ML Systems](https://mlops.community/)
- [Model Monitoring](https://arize.com/ml-monitoring/)

---

**Last Updated:** March 31, 2025  
**Version:** 1.0.0
