# Production Module - Azure Demand Forecasting System

Complete production operationalization system for the Azure Demand Forecasting & Capacity Optimization Model.

## Components

### 1. **Deployment Manager** (`deployment_manager.py`)
Handles model versioning, staging, deployment to production, and rollback capabilities.

**Features:**
- Model file hashing for integrity verification
- Pre-deployment validation checks
- Automatic backup of previous versions
- Rollback to previous versions
- Deployment history tracking

**Usage:**
```python
from production.deployment_manager import ProductionDeploymentManager

manager = ProductionDeploymentManager()
result = manager.stage_model(model_path, features_path, version='2.0.1', metrics={...})
result = manager.deploy_to_production(version='2.0.1', approval_notes='...')
```

### 2. **Monitoring System** (`monitoring.py`)
Real-time monitoring of model performance, system health, and anomaly detection.

**Components:**
- `ModelPerformanceMonitor` - Tracks predictions, errors, and inference times
- `SystemHealthMonitor` - Monitors overall system availability
- `AnomalyDetector` - Detects anomalies in metrics using Z-score analysis
- `AlertManager` - Routes alerts through configurable channels

**Usage:**
```python
from production.monitoring import ModelPerformanceMonitor, AlertManager

monitor = ModelPerformanceMonitor()
monitor.record_prediction(prediction=425.67, inference_time_ms=12.3)
metrics = monitor.get_current_metrics()

alerts = AlertManager()
alerts.add_alert_rule('high_error', lambda ctx: ctx['metrics']['avg_error'] > 0.15, ['log'], 'HIGH')
```

### 3. **Azure Integration** (`azure_integration.py`)
Integrates with Azure services for infrastructure management and optimization.

**Components:**
- `AzureMonitorIntegration` - Sends metrics to Azure Monitor
- `AzureAutoscaleIntegration` - Manages autoscaling rules
- `AzureCostOptimization` - Analyzes cost impact and opportunities
- `CapacityPlanningIntegration` - Generates capacity plans

**Usage:**
```python
from production.azure_integration import AzureAutoscaleIntegration

autoscale = AzureAutoscaleIntegration(subscription_id, resource_group)
recommendations = autoscale.get_scaling_recommendations(
    current_utilization=75, 
    predicted_utilization=88,
    current_capacity=500
)
```

### 4. **Reporting Engine** (`reporting.py`)
Generates automated reports on model performance, infrastructure actions, and forecasts.

**Reports Generated:**
- Daily Executive Summary
- Weekly Trend Analysis
- Model Performance Report
- Infrastructure Action Report
- Forecast Accuracy Report

**Usage:**
```python
from production.reporting import ReportGenerator

reporter = ReportGenerator()
report = reporter.generate_daily_report(metrics={...}, alerts=[...], predictions=[...], recommendations={...})
```

### 5. **Orchestration & Scheduling** (`orchestration.py`)
Automates workflows and schedules recurring tasks.

**Features:**
- Flexible task scheduling (hourly, daily, weekly, monthly)
- Multi-step workflow orchestration
- Task execution history tracking
- Error handling and retries

**Usage:**
```python
from production.orchestration import OrchestrationScheduler

scheduler = OrchestrationScheduler()
scheduler.register_task('daily_retrain', 'Daily Retraining', retraining_task, {
    'frequency': 'daily',
    'time': '02:00'
})
scheduler.start()
```

### 6. **Configuration Management** (`config_manager.py`)
Centralized configuration with validation and environment variable support.

**Features:**
- YAML-based configuration
- Environment variable support
- Configuration validation
- Path management

**Usage:**
```python
from production.config_manager import initialize_config, get_config

config = initialize_config('production/config/production.json')
models_dir = config.get('deployment.models_dir')
```

### 7. **Production API** (`prediction_api_prod.py`)
FastAPI-based production-ready prediction service with integrated monitoring.

**Endpoints:**
- `POST /predict` - Single prediction with recommendations
- `POST /predict/batch` - Batch predictions
- `GET /health` - Health check
- `GET /monitoring/metrics` - Current metrics
- `GET /monitoring/health` - System health
- `GET /alerts/summary` - Alert summary
- `GET /deployment/history` - Deployment history
- `POST /deployment/stage` - Stage model
- `POST /deployment/approve` - Approve deployment
- `POST /deployment/rollback` - Rollback deployment
- `GET /capacity-planning/recommendations` - Capacity recommendations

### 8. **Environment Initialization** (`initialization.py`)
Bootstrap and configure the production environment.

**Usage:**
```bash
# Check pre-deployment checklist
python -m production.initialization --command checklist

# Start environment
python -m production.initialization --command start

# Check status
python -m production.initialization --command status
```

---

## Directory Structure

```
production/
├── __init__.py                          # Package initialization
├── deployment_manager.py                # Model deployment and versioning
├── monitoring.py                        # Performance monitoring and alerting
├── azure_integration.py                 # Azure service integrations
├── reporting.py                         # Automated reporting engine
├── orchestration.py                     # Workflow orchestration
├── config_manager.py                    # Configuration management
├── prediction_api_prod.py               # Production prediction API
├── initialization.py                    # Environment initialization
├── requirements_prod.txt                # Production dependencies
│
├── config/
│   └── production.json                  # Default configuration
│
├── models/                              # Production models directory
│   └── model_active/                    # Symlink to active model
│       ├── model.pkl
│       └── features.pkl
│
├── staging/                             # Model staging area
│   └── v2.0.1/                          # Version directories
│
├── logs/
│   ├── production.log                   # Main production log
│   ├── deployments.json                 # Deployment history
│   └── reports/                         # Generated reports
│       ├── daily_report_2025-03-31.json
│       ├── weekly_report_week13_2025.json
│       ├── model_performance_2025-03-31.json
│       └── infrastructure_actions_2025-03-31.json
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r production/requirements_prod.txt
```

### 2. Configure Environment

Create `.env`:
```bash
AZURE_SUBSCRIPTION_ID=your-id
AZURE_RESOURCE_GROUP=your-group
AZURE_WORKSPACE_ID=your-workspace
AZURE_WORKSPACE_KEY=your-key
```

### 3. Initialize Production Environment

```bash
python -m production.initialization --command checklist
python -m production.initialization --command start
```

### 4. Start Prediction API

```bash
# Development
uvicorn production.prediction_api_prod:app --reload

# Production
gunicorn -w 4 -b 0.0.0.0:8000 production.prediction_api_prod:app
```

### 5. Make Predictions

```bash
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
```

---

## Configuration

Key configuration options in `production/config/production.json`:

```json
{
  "deployment": {
    "models_dir": "production/models",      # Where production models are stored
    "max_versions_to_keep": 5              # Keep last 5 versions
  },
  "api": {
    "port": 8000,                          # API port
    "workers": 4                           # Number of workers
  },
  "monitoring": {
    "enabled": true,
    "performance_alert_threshold": 0.1     # 10% error increase triggers alert
  },
  "scheduling": {
    "retraining_frequency": "daily",       # Retrain once daily
    "retraining_time": "02:00",            # At 2 AM
    "report_frequency": "daily",           # Daily reports
    "report_time": "06:00"                 # At 6 AM
  }
}
```

---

## Monitoring & Alerts

### Automatic Monitoring

The system monitors:
- Prediction accuracy (MAE, R²)
- Inference time
- System availability
- Resource utilization
- Cost metrics

### Alert Configuration

```python
from production.monitoring import AlertManager

alerts = AlertManager()

# Register notification channels
alerts.register_notification_channel('log', log_handler)
alerts.register_notification_channel('email', email_handler)

# Add alert rules
alerts.add_alert_rule(
    'high_error',
    lambda ctx: ctx['metrics']['avg_error'] > 0.15,
    ['log', 'email'],
    'HIGH'
)
```

---

## Deployment Workflow

### Deploy New Model

```python
from production.deployment_manager import ProductionDeploymentManager

manager = ProductionDeploymentManager()

# Stage model
manager.stage_model(
    model_path='models/best_model.pkl',
    features_path='models/features.pkl',
    version='2.1.0',
    metrics={'avg_error': 0.0845}
)

# Deploy to production
manager.deploy_to_production(
    version='2.1.0',
    approval_notes='5% accuracy improvement'
)
```

### Rollback on Issues

```python
# If issues detected
result = manager.rollback(reason='Performance degradation')
```

---

## Reports & Insights

### Generated Reports

**Daily Reports** include:
- Executive summary
- Performance metrics
- Alerts triggered
- Infrastructure recommendations

**Weekly Reports** include:
- Trend analysis
- Top issues
- Capacity trends
- Cost analysis

**Model Performance Reports** include:
- Accuracy metrics
- Inference statistics
- Health assessment
- Recommendations

All reports are saved to `production/logs/reports/` as JSON files.

---

## API Security

### Enabled by Default

- SSL/TLS encryption
- CORS restrictions
- Rate limiting (1000 requests/minute)
- API key authentication

### Configure in production.json

```json
{
  "security": {
    "api_key_required": true,
    "ssl_enabled": true,
    "cors_enabled": true,
    "cors_origins": ["https://yourdomain.com"]
  }
}
```

---

## Performance Targets

| Metric | Target | Acceptable |
|--------|--------|-----------|
| Inference Time | <20ms | <50ms |
| Avg Prediction Error | <0.10 | <0.15 |
| Throughput | >50 pred/sec | >40 pred/sec |
| System Availability | 99.9% | >99% |
| Cost per 1k predictions | $0.05 | <$0.10 |

---

## Troubleshooting

### Model Not Loading

```bash
python -m production.initialization --command status
ls -la production/models/model_active/
tail -f production/logs/production.log
```

### High Error Rates

```python
metrics = monitor.get_metrics_trend('avg_error', lookback_hours=24)
retraining_needed = retrainer.needs_retraining(new_data)
```

### API Connection Issues

- Check API is running: `curl http://localhost:8000/health`
- Check firewall rules
- Verify configuration in production.json

---

## Support

For issues or questions:

1. Check [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)
2. Review logs in `production/logs/`
3. Check deployment history: `/deployment/history`
4. Consult monitoring dashboard: `/monitoring/metrics`

---

## License

MIT License - See LICENSE file

---

**Version:** 1.0.0  
**Last Updated:** March 31, 2025
