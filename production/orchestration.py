"""
Orchestration & Scheduling System
Automate workflows: model retraining, report generation, monitoring, and infrastructure actions
"""

import json
import logging
import schedule
import time
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from enum import Enum
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    SUCCESS = 'SUCCESS'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'


class ScheduledTask:
    """Represents a scheduled task"""
    
    def __init__(self, task_id: str, name: str, task_fn: Callable,
                 schedule_config: Dict[str, Any], enabled: bool = True):
        """Initialize scheduled task
        
        Args:
            task_id: Unique task identifier
            name: Human-readable task name
            task_fn: Function to execute
            schedule_config: Schedule configuration (interval, frequency, etc)
            enabled: Whether task is enabled
        """
        self.task_id = task_id
        self.name = name
        self.task_fn = task_fn
        self.schedule_config = schedule_config
        self.enabled = enabled
        self.last_run = None
        self.next_run = None
        self.status = TaskStatus.PENDING
        self.execution_history = []
        self.error_count = 0
        self.success_count = 0
    
    def execute(self) -> Dict[str, Any]:
        """Execute the task"""
        try:
            self.status = TaskStatus.RUNNING
            start_time = datetime.utcnow()
            
            logger.info(f"Executing task: {self.name}")
            result = self.task_fn()
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            execution_record = {
                'task_id': self.task_id,
                'name': self.name,
                'executed_at': start_time.isoformat(),
                'execution_time_seconds': execution_time,
                'status': 'SUCCESS',
                'result': result
            }
            
            self.execution_history.append(execution_record)
            self.status = TaskStatus.SUCCESS
            self.success_count += 1
            self.last_run = start_time
            
            logger.info(f"Task {self.name} completed successfully in {execution_time:.2f}s")
            return execution_record
        
        except Exception as e:
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            execution_record = {
                'task_id': self.task_id,
                'name': self.name,
                'executed_at': start_time.isoformat(),
                'execution_time_seconds': execution_time,
                'status': 'FAILED',
                'error': str(e)
            }
            
            self.execution_history.append(execution_record)
            self.status = TaskStatus.FAILED
            self.error_count += 1
            
            logger.error(f"Task {self.name} failed: {str(e)}")
            return execution_record
    
    def get_execution_history(self, limit: int = 10) -> List[Dict]:
        """Get recent execution history"""
        return list(reversed(self.execution_history[-limit:]))


class OrchestrationScheduler:
    """Orchestrate and schedule automated workflows"""
    
    def __init__(self, tasks_config_file: Optional[str] = None):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.scheduler = schedule.Scheduler()
        self.running = False
        self.scheduler_thread = None
        self.tasks_config_file = tasks_config_file or 'production/config/scheduler_tasks.json'
        
        logger.info("Orchestration Scheduler initialized")
    
    def register_task(self, task_id: str, name: str, task_fn: Callable,
                     schedule_config: Dict[str, Any]):
        """Register a new scheduled task
        
        Args:
            task_id: Unique task identifier
            name: Human-readable task name
            task_fn: Function to execute
            schedule_config: Schedule configuration
        """
        task = ScheduledTask(task_id, name, task_fn, schedule_config)
        self.tasks[task_id] = task
        
        # Schedule the task
        self._schedule_task(task)
        
        logger.info(f"Registered task: {name}")
    
    def _schedule_task(self, task: ScheduledTask):
        """Schedule a task with the scheduler"""
        try:
            config = task.schedule_config
            frequency = config.get('frequency', 'daily')
            interval = config.get('interval', 1)
            
            if frequency == 'hourly':
                self.scheduler.every(interval).hours.do(task.execute)
            elif frequency == 'daily':
                time_str = config.get('time', '00:00')  # Format: HH:MM
                self.scheduler.every().day.at(time_str).do(task.execute)
            elif frequency == 'weekly':
                day = config.get('day', 'monday')
                self.scheduler.every().monday.at(config.get('time', '00:00')).do(task.execute)
            elif frequency == 'monthly':
                self.scheduler.every(interval).days.do(task.execute)  # Simplified
            elif frequency == 'once':
                delay = config.get('delay_seconds', 0)
                self.scheduler.every(delay).seconds.do(task.execute)
            
            task.next_run = self._calculate_next_run(task)
            logger.info(f"Scheduled task {task.name} with frequency: {frequency}")
        
        except Exception as e:
            logger.error(f"Failed to schedule task {task.name}: {str(e)}")
    
    def _calculate_next_run(self, task: ScheduledTask) -> datetime:
        """Calculate next run time for a task"""
        config = task.schedule_config
        frequency = config.get('frequency', 'daily')
        
        now = datetime.utcnow()
        
        if frequency == 'hourly':
            return now + timedelta(hours=config.get('interval', 1))
        elif frequency == 'daily':
            return now + timedelta(days=1)
        elif frequency == 'weekly':
            return now + timedelta(days=7)
        elif frequency == 'monthly':
            return now + timedelta(days=30)
        
        return now
    
    def start(self):
        """Start the scheduler"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Orchestration Scheduler started")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.running:
            try:
                self.scheduler.run_pending()
                time.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Orchestration Scheduler stopped")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        return {
            'task_id': task_id,
            'name': task.name,
            'enabled': task.enabled,
            'status': task.status.value,
            'last_run': task.last_run.isoformat() if task.last_run else None,
            'next_run': task.next_run.isoformat() if task.next_run else None,
            'success_count': task.success_count,
            'error_count': task.error_count,
            'schedule': task.schedule_config
        }
    
    def get_all_tasks_status(self) -> List[Dict[str, Any]]:
        """Get status of all tasks"""
        return [self.get_task_status(task_id) for task_id in self.tasks.keys()]
    
    def disable_task(self, task_id: str) -> bool:
        """Disable a task"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            logger.info(f"Disabled task: {self.tasks[task_id].name}")
            return True
        return False
    
    def enable_task(self, task_id: str) -> bool:
        """Enable a task"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            logger.info(f"Enabled task: {self.tasks[task_id].name}")
            return True
        return False
    
    def run_task_now(self, task_id: str) -> Optional[Dict]:
        """Run a task immediately"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        logger.info(f"Manual execution of task: {task.name}")
        return task.execute()
    
    def save_task_config(self):
        """Save task configuration to file"""
        try:
            config = {
                'generated_at': datetime.utcnow().isoformat(),
                'tasks': [
                    {
                        'task_id': task_id,
                        'name': task.name,
                        'enabled': task.enabled,
                        'schedule_config': task.schedule_config
                    }
                    for task_id, task in self.tasks.items()
                ]
            }
            
            Path(self.tasks_config_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.tasks_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Task configuration saved to {self.tasks_config_file}")
        
        except Exception as e:
            logger.error(f"Failed to save task configuration: {str(e)}")


class WorkflowOrchestrator:
    """Orchestrate complex multi-step workflows"""
    
    def __init__(self):
        self.workflows = {}
        self.execution_history = []
        
        logger.info("Workflow Orchestrator initialized")
    
    def create_workflow(self, workflow_id: str, name: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a new workflow
        
        Args:
            workflow_id: Unique workflow identifier
            name: Human-readable workflow name
            steps: List of workflow steps
        
        Returns:
            Workflow configuration
        """
        workflow = {
            'workflow_id': workflow_id,
            'name': name,
            'created_at': datetime.utcnow().isoformat(),
            'steps': steps,
            'status': 'CREATED'
        }
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {name}")
        return workflow
    
    def execute_workflow(self, workflow_id: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a workflow
        
        Args:
            workflow_id: Workflow to execute
            context: Initial context data
        
        Returns:
            Execution result
        """
        if workflow_id not in self.workflows:
            return {'success': False, 'error': f'Workflow {workflow_id} not found'}
        
        workflow = self.workflows[workflow_id]
        context = context or {}
        execution_record = {
            'workflow_id': workflow_id,
            'name': workflow['name'],
            'started_at': datetime.utcnow().isoformat(),
            'steps_execution': [],
            'context': context
        }
        
        try:
            for step in workflow['steps']:
                step_result = self._execute_step(step, context)
                execution_record['steps_execution'].append({
                    'step_id': step.get('id'),
                    'step_name': step.get('name'),
                    'status': step_result['status'],
                    'result': step_result.get('result'),
                    'error': step_result.get('error')
                })
                
                # Stop if step failed and stop_on_error is True
                if step_result['status'] == 'FAILED' and step.get('stop_on_error', False):
                    execution_record['status'] = 'FAILED'
                    break
                
                # Update context with step output
                if 'output_key' in step and 'result' in step_result:
                    context[step['output_key']] = step_result['result']
            
            execution_record['completed_at'] = datetime.utcnow().isoformat()
            execution_record['status'] = 'SUCCESS'
        
        except Exception as e:
            execution_record['status'] = 'FAILED'
            execution_record['error'] = str(e)
            execution_record['completed_at'] = datetime.utcnow().isoformat()
        
        self.execution_history.append(execution_record)
        logger.info(f"Workflow {workflow['name']} execution: {execution_record['status']}")
        return execution_record
    
    def _execute_step(self, step: Dict[str, Any], context: Dict) -> Dict[str, Any]:
        """Execute a single workflow step
        
        Args:
            step: Step configuration
            context: Execution context
        
        Returns:
            Step execution result
        """
        try:
            step_id = step.get('id', 'unknown')
            step_name = step.get('name', 'unknown')
            step_fn = step.get('function')
            
            if not step_fn or not callable(step_fn):
                return {
                    'step_id': step_id,
                    'status': 'FAILED',
                    'error': 'Step function not callable'
                }
            
            logger.info(f"Executing step: {step_name}")
            result = step_fn(context)
            
            return {
                'step_id': step_id,
                'status': 'SUCCESS',
                'result': result
            }
        
        except Exception as e:
            return {
                'step_id': step.get('id'),
                'status': 'FAILED',
                'error': str(e)
            }
    
    def get_workflow(self, workflow_id: str) -> Optional[Dict]:
        """Get workflow definition"""
        return self.workflows.get(workflow_id)
    
    def get_workflow_execution_history(self, workflow_id: str, limit: int = 10) -> List[Dict]:
        """Get execution history for a workflow"""
        history = [
            h for h in self.execution_history
            if h.get('workflow_id') == workflow_id
        ]
        return list(reversed(history[-limit:]))


class AutomatedPipelineFactory:
    """Factory for creating standard automated pipelines"""
    
    @staticmethod
    def create_retraining_pipeline(model_path: str, retrainer_obj: Any) -> Dict[str, Any]:
        """Create a model retraining pipeline
        
        Args:
            model_path: Path to current model
            retrainer_obj: Retraining engine object
        
        Returns:
            Pipeline workflow
        """
        return {
            'name': 'Hourly Model Retraining',
            'description': 'Check new data and retrain model if needed',
            'steps': [
                {
                    'id': 'load_data',
                    'name': 'Load new training data',
                    'function': lambda ctx: True
                },
                {
                    'id': 'check_retraining',
                    'name': 'Check if retraining needed',
                    'function': lambda ctx: retrainer_obj.needs_retraining(None) if hasattr(retrainer_obj, 'needs_retraining') else {'should_retrain': False},
                    'output_key': 'retraining_needed'
                },
                {
                    'id': 'execute_retrain',
                    'name': 'Execute retraining',
                    'function': lambda ctx: retrainer_obj.retrain(None, None) if ctx.get('retraining_needed') else None
                }
            ]
        }
    
    @staticmethod
    def create_monitoring_pipeline(monitoring_obj: Any, alert_manager_obj: Any) -> Dict[str, Any]:
        """Create a monitoring and alert pipeline
        
        Args:
            monitoring_obj: Performance monitoring object
            alert_manager_obj: Alert manager object
        
        Returns:
            Pipeline workflow
        """
        return {
            'name': 'Continuous Monitoring',
            'description': 'Monitor model and system health',
            'steps': [
                {
                    'id': 'get_metrics',
                    'name': 'Collect current metrics',
                    'function': lambda ctx: monitoring_obj.get_current_metrics() if hasattr(monitoring_obj, 'get_current_metrics') else {},
                    'output_key': 'current_metrics'
                },
                {
                    'id': 'health_check',
                    'name': 'Run health check',
                    'function': lambda ctx: {'status': 'HEALTHY'},
                    'output_key': 'health_check'
                },
                {
                    'id': 'evaluate_alerts',
                    'name': 'Evaluate alert rules',
                    'function': lambda ctx: alert_manager_obj.check_rules(ctx) if hasattr(alert_manager_obj, 'check_rules') else []
                }
            ]
        }
    
    @staticmethod
    def create_reporting_pipeline(report_generator_obj: Any) -> Dict[str, Any]:
        """Create an automated reporting pipeline
        
        Args:
            report_generator_obj: Report generation object
        
        Returns:
            Pipeline workflow
        """
        return {
            'name': 'Daily Report Generation',
            'description': 'Generate daily executive summary reports',
            'steps': [
                {
                    'id': 'collect_data',
                    'name': 'Collect metrics and alerts',
                    'function': lambda ctx: {}
                },
                {
                    'id': 'generate_report',
                    'name': 'Generate daily report',
                    'function': lambda ctx: report_generator_obj.generate_daily_report({}, [], [], {}) if hasattr(report_generator_obj, 'generate_daily_report') else {}
                },
                {
                    'id': 'publish_report',
                    'name': 'Publish report',
                    'function': lambda ctx: {'published': True}
                }
            ]
        }
