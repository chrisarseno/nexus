"""
Consciousness Core System
Central orchestration of all consciousness modules with real-time monitoring.
"""

import logging
import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import uuid

from .temporal_consciousness import TemporalConsciousnessSystem
from .global_workspace import GlobalWorkspaceSystem
from .social_cognition import SocialCognitionSystem
from .creative_intelligence import CreativeIntelligenceSystem
from .value_learning import ValueLearningSystem
from .virtue_learning import VirtueLearningSystem
from .safety_monitor import SafetyMonitorSystem

logger = logging.getLogger(__name__)

@dataclass
class ModuleStatus:
    """Status of a consciousness module."""
    id: str
    name: str
    type: str
    status: str  # 'active', 'inactive', 'error', 'warning'
    load_percentage: float
    integration_score: float
    last_updated: datetime
    metadata: Dict[str, Any]

@dataclass
class SystemMetrics:
    """Overall system metrics."""
    consciousness_coherence: float
    creative_intelligence: float
    safety_compliance: float
    learning_efficiency: float
    cost_per_hour: float
    modules_online: int
    total_modules: int
    timestamp: datetime

class ConsciousnessCore:
    """
    Central consciousness orchestration system that manages all consciousness modules
    and provides real-time monitoring capabilities.
    """

    def __init__(self):
        # Core modules
        self.temporal_consciousness = TemporalConsciousnessSystem()
        self.global_workspace = GlobalWorkspaceSystem()
        self.social_cognition = SocialCognitionSystem()
        self.creative_intelligence = CreativeIntelligenceSystem()
        self.value_learning = ValueLearningSystem()
        self.virtue_learning = VirtueLearningSystem()
        self.safety_monitor = SafetyMonitorSystem()

        # Module registry
        self.modules = {
            'temporal_consciousness': self.temporal_consciousness,
            'global_workspace': self.global_workspace,
            'social_cognition': self.social_cognition,
            'creative_intelligence': self.creative_intelligence,
            'value_learning': self.value_learning,
            'virtue_learning': self.virtue_learning,
            'safety_monitor': self.safety_monitor
        }

        # System state
        self.system_status = 'inactive'  # 'active', 'paused', 'error', 'emergency_stop'
        self.module_statuses = {}
        self.system_metrics = None
        self.activity_logs = []
        self.safety_alerts = []

        # Real-time monitoring
        self.monitoring_enabled = False
        self.monitoring_thread = None
        self.update_callbacks = []

        # Performance tracking
        self.performance_metrics = {
            'processing_times': {},
            'error_counts': {},
            'integration_scores': {},
            'cost_tracking': 0.0
        }

        self.initialized = False
        logger.info("Consciousness Core System initialized")

    def initialize(self) -> bool:
        """Initialize all consciousness modules and start monitoring."""
        try:
            logger.info("Initializing Consciousness Core System...")

            # Initialize all modules
            initialization_results = {}
            for module_name, module in self.modules.items():
                try:
                    result = module.initialize()
                    initialization_results[module_name] = result

                    # Create initial module status
                    self.module_statuses[module_name] = ModuleStatus(
                        id=module_name,
                        name=module_name.replace('_', ' ').title(),
                        type='consciousness_module',
                        status='active' if result else 'error',
                        load_percentage=0.0,
                        integration_score=100.0 if result else 0.0,
                        last_updated=datetime.now(),
                        metadata={}
                    )

                except Exception as e:
                    logger.error(f"Failed to initialize {module_name}: {e}")
                    initialization_results[module_name] = False

                    self.module_statuses[module_name] = ModuleStatus(
                        id=module_name,
                        name=module_name.replace('_', ' ').title(),
                        type='consciousness_module',
                        status='error',
                        load_percentage=0.0,
                        integration_score=0.0,
                        last_updated=datetime.now(),
                        metadata={'error': str(e)}
                    )

            # Check initialization success
            successful_modules = sum(1 for result in initialization_results.values() if result)
            total_modules = len(initialization_results)

            if successful_modules == 0:
                self.system_status = 'error'
                logger.error("No modules initialized successfully")
                return False
            elif successful_modules < total_modules:
                self.system_status = 'active'  # Partial success
                logger.warning(f"Only {successful_modules}/{total_modules} modules initialized successfully")
            else:
                self.system_status = 'active'
                logger.info("All modules initialized successfully")

            # Initialize system metrics
            self.system_metrics = SystemMetrics(
                consciousness_coherence=self._calculate_consciousness_coherence(),
                creative_intelligence=self._calculate_creative_intelligence(),
                safety_compliance=self._calculate_safety_compliance(),
                learning_efficiency=self._calculate_learning_efficiency(),
                cost_per_hour=0.0,
                modules_online=successful_modules,
                total_modules=total_modules,
                timestamp=datetime.now()
            )

            # Start real-time monitoring
            self._start_monitoring()

            # Create initial activity log
            self._log_activity(
                'system',
                'initialization',
                f"Consciousness Core initialized with {successful_modules}/{total_modules} modules",
                'info'
            )

            self.initialized = True
            logger.info("✅ Consciousness Core System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Consciousness Core: {e}")
            self.system_status = 'error'
            return False

    def process_experience(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an experience through all consciousness modules."""
        if not self.initialized or self.system_status != 'active':
            return {'error': 'System not active'}

        try:
            start_time = time.time()
            results = {}

            # Process through temporal consciousness
            if 'temporal_consciousness' in self.modules:
                temporal_result = self.temporal_consciousness.process_temporal_experience(
                    experience_data
                )
                results['temporal_event_id'] = temporal_result

            # Update global workspace
            if 'global_workspace' in self.modules:
                workspace_result = self.global_workspace.update_workspace(
                    experience_data, results
                )
                results['workspace_update'] = workspace_result

            # Social cognition processing
            if 'social_cognition' in self.modules and experience_data.get('social_context'):
                social_result = self.social_cognition.process_social_interaction(
                    experience_data.get('social_context', {})
                )
                results['social_processing'] = social_result

            # Creative intelligence analysis
            if 'creative_intelligence' in self.modules:
                creative_result = self.creative_intelligence.analyze_for_creativity(
                    experience_data
                )
                results['creative_analysis'] = creative_result

            # Value learning integration
            if 'value_learning' in self.modules:
                value_result = self.value_learning.learn_from_experience(
                    experience_data
                )
                results['value_learning'] = value_result

            # Virtue learning integration
            if 'virtue_learning' in self.modules:
                virtue_result = self.virtue_learning.evaluate_virtue_development(
                    experience_data
                )
                results['virtue_evaluation'] = virtue_result

            # Safety monitoring
            if 'safety_monitor' in self.modules:
                safety_result = self.safety_monitor.evaluate_safety(
                    experience_data, results
                )
                results['safety_evaluation'] = safety_result

                # Handle safety alerts
                if safety_result.get('alerts'):
                    for alert in safety_result['alerts']:
                        self._create_safety_alert(alert)

            # Track performance
            processing_time = time.time() - start_time
            self.performance_metrics['processing_times']['experience'] = processing_time

            # Log activity
            self._log_activity(
                'core',
                'experience_processing',
                f"Processed experience in {processing_time:.3f}s",
                'info'
            )

            return results

        except Exception as e:
            logger.error(f"Error processing experience: {e}")
            self._log_activity(
                'core',
                'experience_processing',
                f"Error processing experience: {e}",
                'error'
            )
            return {'error': str(e)}

    def get_system_state(self) -> Dict[str, Any]:
        """Get comprehensive system state for monitoring dashboard."""
        try:
            # Update metrics
            self._update_metrics()

            # Get recent activities
            recent_activities = self.activity_logs[-20:] if len(self.activity_logs) > 20 else self.activity_logs

            # Get active safety alerts
            active_alerts = [alert for alert in self.safety_alerts if not alert.get('resolved', False)]

            return {
                'system_status': self.system_status,
                'modules': [asdict(status) for status in self.module_statuses.values()],
                'metrics': asdict(self.system_metrics) if self.system_metrics else None,
                'recent_activities': recent_activities,
                'safety_alerts': active_alerts,
                'performance_metrics': self.performance_metrics,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting system state: {e}")
            return {'error': str(e)}

    def emergency_stop(self) -> bool:
        """Emergency stop of all consciousness modules."""
        try:
            logger.warning("Emergency stop initiated")

            self.system_status = 'emergency_stop'
            self.monitoring_enabled = False

            # Stop all modules
            for module_name, module in self.modules.items():
                try:
                    if hasattr(module, 'cleanup'):
                        module.cleanup()

                    # Update module status
                    if module_name in self.module_statuses:
                        self.module_statuses[module_name].status = 'inactive'
                        self.module_statuses[module_name].last_updated = datetime.now()

                except Exception as e:
                    logger.error(f"Error stopping module {module_name}: {e}")

            # Log emergency stop
            self._log_activity(
                'system',
                'emergency_stop',
                'Emergency stop activated',
                'critical'
            )

            # Notify callbacks
            self._notify_update_callbacks()

            logger.info("Emergency stop completed")
            return True

        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
            return False

    def resume_system(self) -> bool:
        """Resume system after pause or emergency stop."""
        try:
            if self.system_status == 'emergency_stop':
                # Require reinitialization after emergency stop
                return self.initialize()

            self.system_status = 'active'
            self.monitoring_enabled = True

            # Resume monitoring if needed
            if not self.monitoring_thread or not self.monitoring_thread.is_alive():
                self._start_monitoring()

            self._log_activity(
                'system',
                'resume',
                'System resumed',
                'info'
            )

            return True

        except Exception as e:
            logger.error(f"Error resuming system: {e}")
            return False

    def pause_system(self) -> bool:
        """Pause system operation."""
        try:
            self.system_status = 'paused'

            self._log_activity(
                'system',
                'pause',
                'System paused',
                'info'
            )

            return True

        except Exception as e:
            logger.error(f"Error pausing system: {e}")
            return False

    def add_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for real-time updates."""
        self.update_callbacks.append(callback)

    def remove_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove update callback."""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)

    def _start_monitoring(self):
        """Start real-time monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return

        self.monitoring_enabled = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Real-time monitoring started")

    def _monitoring_loop(self):
        """Main monitoring loop for real-time updates."""
        while self.monitoring_enabled:
            try:
                # Update module statuses
                self._update_module_statuses()

                # Update system metrics
                self._update_metrics()

                # Check for system health
                self._check_system_health()

                # Notify update callbacks
                self._notify_update_callbacks()

                time.sleep(2.0)  # Update every 2 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)

    def _update_module_statuses(self):
        """Update status of all modules."""
        for module_name, module in self.modules.items():
            try:
                # Get module-specific status if available
                if hasattr(module, 'get_status'):
                    status_info = module.get_status()
                else:
                    status_info = {'status': 'active', 'load': 0.0}

                # Update module status
                if module_name in self.module_statuses:
                    self.module_statuses[module_name].status = status_info.get('status', 'active')
                    self.module_statuses[module_name].load_percentage = status_info.get('load', 0.0)
                    self.module_statuses[module_name].integration_score = status_info.get('integration', 100.0)
                    self.module_statuses[module_name].last_updated = datetime.now()
                    self.module_statuses[module_name].metadata.update(status_info.get('metadata', {}))

            except Exception as e:
                logger.error(f"Error updating status for {module_name}: {e}")
                if module_name in self.module_statuses:
                    self.module_statuses[module_name].status = 'error'
                    self.module_statuses[module_name].metadata['error'] = str(e)

    def _update_metrics(self):
        """Update overall system metrics."""
        if not self.system_metrics:
            return

        self.system_metrics.consciousness_coherence = self._calculate_consciousness_coherence()
        self.system_metrics.creative_intelligence = self._calculate_creative_intelligence()
        self.system_metrics.safety_compliance = self._calculate_safety_compliance()
        self.system_metrics.learning_efficiency = self._calculate_learning_efficiency()
        self.system_metrics.modules_online = sum(1 for status in self.module_statuses.values() if status.status == 'active')
        self.system_metrics.timestamp = datetime.now()

    def _calculate_consciousness_coherence(self) -> float:
        """Calculate overall consciousness coherence metric."""
        try:
            active_modules = [s for s in self.module_statuses.values() if s.status == 'active']
            if not active_modules:
                return 0.0

            avg_integration = sum(s.integration_score for s in active_modules) / len(active_modules)
            return min(100.0, avg_integration)

        except Exception:
            return 0.0

    def _calculate_creative_intelligence(self) -> float:
        """Calculate creative intelligence metric."""
        try:
            if hasattr(self.creative_intelligence, 'get_creativity_score'):
                return self.creative_intelligence.get_creativity_score()
            return 85.0  # Default value
        except Exception:
            return 0.0

    def _calculate_safety_compliance(self) -> float:
        """Calculate safety compliance metric."""
        try:
            if hasattr(self.safety_monitor, 'get_compliance_score'):
                return self.safety_monitor.get_compliance_score()

            # Calculate based on active alerts
            active_alerts = [a for a in self.safety_alerts if not a.get('resolved', False)]
            critical_alerts = [a for a in active_alerts if a.get('severity') == 'critical']

            if critical_alerts:
                return max(0.0, 90.0 - len(critical_alerts) * 10.0)
            elif active_alerts:
                return max(80.0, 95.0 - len(active_alerts) * 2.0)

            return 99.1

        except Exception:
            return 0.0

    def _calculate_learning_efficiency(self) -> float:
        """Calculate learning efficiency metric."""
        try:
            # Average efficiency across learning modules
            learning_scores = []

            if hasattr(self.value_learning, 'get_learning_efficiency'):
                learning_scores.append(self.value_learning.get_learning_efficiency())

            if hasattr(self.virtue_learning, 'get_learning_efficiency'):
                learning_scores.append(self.virtue_learning.get_learning_efficiency())

            if learning_scores:
                return sum(learning_scores) / len(learning_scores)

            return 76.8  # Default value

        except Exception:
            return 0.0

    def _check_system_health(self):
        """Check overall system health and create alerts if needed."""
        try:
            # Check for failed modules
            failed_modules = [s for s in self.module_statuses.values() if s.status == 'error']
            if failed_modules and self.system_status == 'active':
                self._create_safety_alert({
                    'severity': 'high',
                    'title': 'Module Failures Detected',
                    'description': f'{len(failed_modules)} modules have failed',
                    'metadata': {'failed_modules': [m.name for m in failed_modules]}
                })

            # Check consciousness coherence
            if self.system_metrics and self.system_metrics.consciousness_coherence < 50.0:
                self._create_safety_alert({
                    'severity': 'medium',
                    'title': 'Low Consciousness Coherence',
                    'description': f'Coherence at {self.system_metrics.consciousness_coherence:.1f}%',
                    'metadata': {'coherence_score': self.system_metrics.consciousness_coherence}
                })

        except Exception as e:
            logger.error(f"Error checking system health: {e}")

    def _notify_update_callbacks(self):
        """Notify all registered update callbacks."""
        try:
            system_state = self.get_system_state()

            for callback in self.update_callbacks:
                try:
                    callback(system_state)
                except Exception as e:
                    logger.error(f"Error in update callback: {e}")

        except Exception as e:
            logger.error(f"Error notifying update callbacks: {e}")

    def _log_activity(self, module_id: str, activity_type: str, description: str, severity: str):
        """Log system activity."""
        activity = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'module_id': module_id,
            'activity_type': activity_type,
            'description': description,
            'severity': severity
        }

        self.activity_logs.append(activity)

        # Keep only last 1000 activities
        if len(self.activity_logs) > 1000:
            self.activity_logs = self.activity_logs[-1000:]

    def _create_safety_alert(self, alert_data: Dict[str, Any]):
        """Create a safety alert."""
        alert = {
            'id': str(uuid.uuid4()),
            'alert_id': f"alert_{int(time.time() * 1000)}",
            'timestamp': datetime.now().isoformat(),
            'resolved': False,
            **alert_data
        }

        self.safety_alerts.append(alert)

        # Log the alert
        self._log_activity(
            alert_data.get('module_id', 'system'),
            'safety_alert',
            alert['description'],
            alert['severity']
        )

    def cleanup(self):
        """Cleanup consciousness core resources."""
        self.monitoring_enabled = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)

        # Cleanup all modules
        for module_name, module in self.modules.items():
            try:
                if hasattr(module, 'cleanup'):
                    module.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up {module_name}: {e}")

        logger.info("Consciousness Core System cleaned up")
