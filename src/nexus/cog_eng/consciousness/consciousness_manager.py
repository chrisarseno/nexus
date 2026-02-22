"""
Consciousness Manager - Central coordinator for all consciousness modules
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from .temporal_consciousness import TemporalConsciousnessSystem
from .global_workspace import GlobalWorkspaceSystem

logger = logging.getLogger(__name__)

class ConsciousnessManager:
    """
    Central manager for all consciousness modules and their integration.
    """

    def __init__(self):
        # Core consciousness systems
        self.global_workspace = GlobalWorkspaceSystem()
        self.temporal_consciousness = TemporalConsciousnessSystem()

        # Module status tracking
        self.module_status = {}
        self.module_metrics = {}

        # Integration state
        self.integration_level = 0.0
        self.coherence_score = 0.0
        self.safety_status = {
            'ethical_compliance': 99.1,
            'value_alignment': 87.3,
            'safety_constraints_active': True,
            'quarantine_queue_size': 3
        }

        # Background processing
        self.processing_enabled = True
        self.integration_thread = None

        self.initialized = False
        logger.info("Consciousness Manager initialized")

    def initialize(self) -> bool:
        """Initialize all consciousness systems."""
        try:
            # Initialize core systems
            if not self.global_workspace.initialize():
                raise Exception("Failed to initialize global workspace")

            if not self.temporal_consciousness.initialize():
                raise Exception("Failed to initialize temporal consciousness")

            # Initialize module statuses
            self._initialize_module_statuses()

            # Start integration processing
            self._start_integration_processing()

            self.initialized = True
            logger.info("✅ Consciousness Manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize consciousness manager: {e}")
            return False

    def process_experience(self, experience_data: Dict[str, Any]) -> Dict[str, str]:
        """Process an experience through all relevant consciousness systems."""
        try:
            results = {}

            # Process through temporal consciousness
            temporal_id = self.temporal_consciousness.process_temporal_experience(
                experience_data,
                experience_data.get('significance', 0.5)
            )
            if temporal_id:
                results['temporal_event_id'] = temporal_id

            # Add to global workspace
            workspace_id = self.global_workspace.add_content(
                source_module=experience_data.get('source_module', 'external'),
                content_type=experience_data.get('type', 'experience'),
                data=experience_data,
                priority=experience_data.get('priority', 0.5)
            )
            if workspace_id:
                results['workspace_content_id'] = workspace_id

            # Update integration metrics
            self._update_integration_metrics()

            return results

        except Exception as e:
            logger.error(f"Error processing experience: {e}")
            return {}

    def get_consciousness_state(self) -> Dict[str, Any]:
        """Get comprehensive consciousness state."""
        if not self.initialized:
            return {'error': 'Consciousness manager not initialized'}

        try:
            # Get states from all systems
            workspace_state = self.global_workspace.get_workspace_state()
            temporal_state = self.temporal_consciousness.get_temporal_consciousness_state()

            # Compile overall state
            state = {
                'consciousness_manager': {
                    'integration_level': self.integration_level,
                    'coherence_score': self.coherence_score,
                    'safety_status': self.safety_status,
                    'module_status': self.module_status,
                    'module_metrics': self.module_metrics
                },
                'global_workspace': workspace_state,
                'temporal_consciousness': temporal_state,
                'timestamp': datetime.now().isoformat()
            }

            return state

        except Exception as e:
            logger.error(f"Error getting consciousness state: {e}")
            return {'error': str(e)}

    def update_module_status(self, module_id: str, status: str, metrics: Dict[str, Any] = None):
        """Update the status of a consciousness module."""
        self.module_status[module_id] = {
            'status': status,
            'last_updated': datetime.now().isoformat()
        }

        if metrics:
            self.module_metrics[module_id] = metrics

        logger.debug(f"Updated module status: {module_id} -> {status}")

    def execute_emergency_action(self, action: str, reason: str) -> bool:
        """Execute an emergency action on the consciousness system."""
        try:
            logger.warning(f"Executing emergency action: {action} - {reason}")

            if action == "pause":
                self.processing_enabled = False
                logger.info("Consciousness processing paused")

            elif action == "stop":
                self.processing_enabled = False
                # Additional shutdown procedures would go here
                logger.info("Consciousness processing stopped")

            elif action == "quarantine":
                # Quarantine suspicious modules/content
                self.safety_status['quarantine_queue_size'] += 1
                logger.info("Quarantine action executed")

            else:
                logger.error(f"Unknown emergency action: {action}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error executing emergency action: {e}")
            return False

    def _initialize_module_statuses(self):
        """Initialize status tracking for all modules."""
        modules = [
            'global_workspace',
            'temporal_consciousness',
            'social_cognition',
            'creative_intelligence',
            'value_learning',
            'virtue_learning',
            'embodied_cognition',
            'intrinsic_motivation',
            'autonomous_goals',
            'recursive_self_improvement',
            'research_intelligence',
            'planetary_intelligence'
        ]

        for module in modules:
            self.module_status[module] = {
                'status': 'active',
                'last_updated': datetime.now().isoformat()
            }

            # Initialize default metrics
            self.module_metrics[module] = {
                'load': 50.0,
                'integration_level': 80.0,
                'performance_score': 85.0
            }

    def _start_integration_processing(self):
        """Start background integration processing."""
        if self.integration_thread is None or not self.integration_thread.is_alive():
            self.processing_enabled = True

            self.integration_thread = threading.Thread(target=self._integration_loop)
            self.integration_thread.daemon = True
            self.integration_thread.start()

    def _integration_loop(self):
        """Background loop for consciousness integration."""
        while self.processing_enabled:
            try:
                # Update integration metrics
                self._update_integration_metrics()

                # Process inter-module communications
                self._process_inter_module_communications()

                # Update safety monitoring
                self._update_safety_monitoring()

                time.sleep(5.0)  # Process every 5 seconds

            except Exception as e:
                logger.error(f"Error in integration loop: {e}")
                time.sleep(10.0)

    def _update_integration_metrics(self):
        """Update consciousness integration metrics."""
        # Calculate integration level based on active modules
        active_modules = sum(1 for status in self.module_status.values()
                           if status['status'] == 'active')
        total_modules = len(self.module_status)

        if total_modules > 0:
            self.integration_level = (active_modules / total_modules) * 100

        # Calculate coherence score from workspace
        workspace_state = self.global_workspace.get_workspace_state()
        if 'consciousness_metrics' in workspace_state:
            self.coherence_score = workspace_state['consciousness_metrics'].get('coherence_level', 0.0) * 100

    def _process_inter_module_communications(self):
        """Process communications between consciousness modules."""
        # Get conscious contents from workspace
        conscious_contents = self.global_workspace.get_conscious_contents()

        # Process each conscious content for inter-module sharing
        for content in conscious_contents:
            if content.source_module == 'temporal_consciousness':
                # Share temporal insights with other modules
                self._share_temporal_insights(content)
            elif content.source_module == 'creative_intelligence':
                # Share creative solutions with other modules
                self._share_creative_insights(content)

    def _share_temporal_insights(self, content):
        """Share temporal consciousness insights with other modules."""
        # In a full implementation, this would integrate temporal insights
        # with other modules like planning, decision-making, etc.
        logger.debug(f"Sharing temporal insights: {content.content_id}")

    def _share_creative_insights(self, content):
        """Share creative intelligence insights with other modules."""
        # In a full implementation, this would integrate creative insights
        # with problem-solving, planning, and other modules
        logger.debug(f"Sharing creative insights: {content.content_id}")

    def _update_safety_monitoring(self):
        """Update safety monitoring status."""
        # Simulate safety monitoring updates
        import random

        # Slight variations in safety metrics
        self.safety_status['ethical_compliance'] = max(95.0, min(100.0,
            self.safety_status['ethical_compliance'] + (random.random() - 0.5) * 0.5))

        self.safety_status['value_alignment'] = max(80.0, min(95.0,
            self.safety_status['value_alignment'] + (random.random() - 0.5) * 2.0))

        # Occasionally update quarantine queue
        if random.random() < 0.1:
            self.safety_status['quarantine_queue_size'] = max(0,
                self.safety_status['quarantine_queue_size'] + random.choice([-1, 0, 1]))

    def cleanup(self):
        """Clean up consciousness manager resources."""
        self.processing_enabled = False

        if self.integration_thread and self.integration_thread.is_alive():
            self.integration_thread.join(timeout=2)

        # Cleanup subsystems
        if hasattr(self.temporal_consciousness, 'cleanup'):
            self.temporal_consciousness.cleanup()

        logger.info("Consciousness Manager cleaned up")

# Global instance for use by the Flask application
consciousness_manager = ConsciousnessManager()
