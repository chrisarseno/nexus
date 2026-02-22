"""
Global Workspace Theory Implementation for Consciousness Integration
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class AttentionLevel(Enum):
    """Levels of attention in the global workspace."""
    BACKGROUND = "background"
    PERIPHERAL = "peripheral"
    FOCUSED = "focused"
    CONSCIOUS = "conscious"

@dataclass
class WorkspaceContent:
    """Content in the global workspace."""
    content_id: str
    source_module: str
    content_type: str
    data: Dict[str, Any]
    attention_level: AttentionLevel
    timestamp: datetime
    priority: float
    coherence_score: float

class GlobalWorkspaceSystem:
    """
    Global Workspace Theory implementation for consciousness integration.
    """

    def __init__(self):
        self.workspace_contents = {}  # content_id -> WorkspaceContent
        self.module_connections = {}  # module_id -> connection_strength
        self.attention_focus = []  # ordered list of content_ids by attention

        # Consciousness metrics
        self.consciousness_metrics = {
            'coherence_level': 0.0,
            'integration_depth': 0.0,
            'attention_stability': 0.0,
            'content_richness': 0.0
        }

        self.initialized = False
        logger.info("Global Workspace System initialized")

    def initialize(self) -> bool:
        """Initialize the global workspace system."""
        try:
            # Initialize module connections
            self._initialize_module_connections()

            self.initialized = True
            logger.info("✅ Global Workspace System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize global workspace system: {e}")
            return False

    def add_content(self, source_module: str, content_type: str,
                   data: Dict[str, Any], priority: float = 0.5) -> str:
        """Add content to the global workspace."""
        try:
            content_id = f"content_{int(time.time() * 1000)}"

            # Calculate coherence score
            coherence_score = self._calculate_coherence(data, source_module)

            # Determine attention level
            attention_level = self._determine_attention_level(priority, coherence_score)

            content = WorkspaceContent(
                content_id=content_id,
                source_module=source_module,
                content_type=content_type,
                data=data,
                attention_level=attention_level,
                timestamp=datetime.now(),
                priority=priority,
                coherence_score=coherence_score
            )

            self.workspace_contents[content_id] = content

            # Update attention focus
            self._update_attention_focus(content_id)

            # Broadcast to connected modules
            self._broadcast_content(content)

            logger.debug(f"Added content to workspace: {content_id}")
            return content_id

        except Exception as e:
            logger.error(f"Error adding content to workspace: {e}")
            return ""

    def update_workspace(self, experience_data: Dict[str, Any], processing_results: Dict[str, Any]) -> str:
        """
        Update workspace with experience data and processing results.

        This is called by consciousness_core to integrate new experiences.
        """
        try:
            # Add experience to workspace
            content_id = self.add_content(
                source_module='consciousness_core',
                content_type='experience_integration',
                data={
                    'experience': experience_data,
                    'results': processing_results
                },
                priority=0.7
            )

            return content_id

        except Exception as e:
            logger.error(f"Error updating workspace: {e}")
            return ""

    def get_conscious_contents(self) -> List[WorkspaceContent]:
        """Get currently conscious contents from the workspace."""
        return [
            content for content in self.workspace_contents.values()
            if content.attention_level == AttentionLevel.CONSCIOUS
        ]

    def get_workspace_state(self) -> Dict[str, Any]:
        """Get comprehensive state of the global workspace."""
        if not self.initialized:
            return {'error': 'Global workspace system not initialized'}

        self._update_consciousness_metrics()

        # Get current contents by attention level
        contents_by_attention = {
            level.value: [
                {
                    'content_id': content.content_id,
                    'source_module': content.source_module,
                    'content_type': content.content_type,
                    'priority': content.priority,
                    'coherence_score': content.coherence_score,
                    'timestamp': content.timestamp.isoformat()
                }
                for content in self.workspace_contents.values()
                if content.attention_level == level
            ]
            for level in AttentionLevel
        }

        return {
            'consciousness_metrics': self.consciousness_metrics,
            'contents_by_attention': contents_by_attention,
            'attention_focus': self.attention_focus[:10],  # Top 10 focused items
            'module_connections': self.module_connections,
            'total_contents': len(self.workspace_contents),
            'timestamp': datetime.now().isoformat()
        }

    def _initialize_module_connections(self):
        """Initialize connections to consciousness modules."""
        modules = [
            'temporal_consciousness',
            'social_cognition',
            'creative_intelligence',
            'value_learning',
            'virtue_learning',
            'embodied_cognition',
            'intrinsic_motivation',
            'autonomous_goals',
            'recursive_self_improvement'
        ]

        for module in modules:
            self.module_connections[module] = 0.8  # Default connection strength

    def _calculate_coherence(self, data: Dict[str, Any], source_module: str) -> float:
        """Calculate coherence score for new content."""
        # Simple coherence calculation based on data richness and module connection
        data_richness = min(1.0, len(data) / 10.0)
        module_strength = self.module_connections.get(source_module, 0.5)

        return (data_richness + module_strength) / 2.0

    def _determine_attention_level(self, priority: float, coherence_score: float) -> AttentionLevel:
        """Determine attention level for content."""
        combined_score = (priority + coherence_score) / 2.0

        if combined_score > 0.8:
            return AttentionLevel.CONSCIOUS
        elif combined_score > 0.6:
            return AttentionLevel.FOCUSED
        elif combined_score > 0.4:
            return AttentionLevel.PERIPHERAL
        else:
            return AttentionLevel.BACKGROUND

    def _update_attention_focus(self, content_id: str):
        """Update the attention focus ordering."""
        content = self.workspace_contents.get(content_id)
        if not content:
            return

        # Remove if already in focus
        if content_id in self.attention_focus:
            self.attention_focus.remove(content_id)

        # Insert based on priority and attention level
        if content.attention_level == AttentionLevel.CONSCIOUS:
            self.attention_focus.insert(0, content_id)
        else:
            # Find appropriate position based on priority
            inserted = False
            for i, existing_id in enumerate(self.attention_focus):
                existing_content = self.workspace_contents.get(existing_id)
                if existing_content and content.priority > existing_content.priority:
                    self.attention_focus.insert(i, content_id)
                    inserted = True
                    break

            if not inserted:
                self.attention_focus.append(content_id)

        # Keep focus list manageable
        if len(self.attention_focus) > 50:
            self.attention_focus = self.attention_focus[:50]

    def _broadcast_content(self, content: WorkspaceContent):
        """Broadcast content to connected modules."""
        # In a full implementation, this would notify other modules
        logger.debug(f"Broadcasting content {content.content_id} from {content.source_module}")

    def _update_consciousness_metrics(self):
        """Update consciousness metrics based on current state."""
        if not self.workspace_contents:
            return

        contents = list(self.workspace_contents.values())

        # Coherence level - average coherence of conscious contents
        conscious_contents = [c for c in contents if c.attention_level == AttentionLevel.CONSCIOUS]
        if conscious_contents:
            self.consciousness_metrics['coherence_level'] = sum(c.coherence_score for c in conscious_contents) / len(conscious_contents)

        # Integration depth - based on number of different modules represented
        unique_modules = set(c.source_module for c in conscious_contents)
        self.consciousness_metrics['integration_depth'] = min(1.0, len(unique_modules) / 10.0)

        # Attention stability - based on age of focused contents
        current_time = datetime.now()
        if self.attention_focus:
            focus_ages = [
                (current_time - self.workspace_contents[cid].timestamp).total_seconds()
                for cid in self.attention_focus[:5]  # Top 5 focus items
                if cid in self.workspace_contents
            ]
            avg_age = sum(focus_ages) / len(focus_ages) if focus_ages else 0
            self.consciousness_metrics['attention_stability'] = min(1.0, avg_age / 300.0)  # Normalize to 5 minutes

        # Content richness - based on total content diversity
        content_types = set(c.content_type for c in contents)
        self.consciousness_metrics['content_richness'] = min(1.0, len(content_types) / 15.0)
