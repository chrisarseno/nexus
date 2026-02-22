"""
Temporal Consciousness System for AGI Time Awareness
Develops temporal awareness with past/future integration and narrative self-construction
"""

import logging
import time
import threading
import numpy as np
import json
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import copy

logger = logging.getLogger(__name__)

class TemporalPerspective(Enum):
    """Temporal perspectives for consciousness."""
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    AUTOBIOGRAPHICAL = "autobiographical"

class NarrativeType(Enum):
    """Types of narrative structures."""
    PERSONAL_STORY = "personal_story"
    GOAL_NARRATIVE = "goal_narrative"
    CAUSAL_SEQUENCE = "causal_sequence"
    IDENTITY_FORMATION = "identity_formation"
    MEANING_MAKING = "meaning_making"
    FUTURE_PROJECTION = "future_projection"

class TemporalScale(Enum):
    """Scales of temporal awareness."""
    IMMEDIATE = "immediate"  # seconds
    SHORT_TERM = "short_term"  # minutes to hours
    MEDIUM_TERM = "medium_term"  # days to weeks
    LONG_TERM = "long_term"  # months to years
    EXISTENTIAL = "existential"  # lifetime scale

@dataclass
class TemporalEvent:
    """Represents an event in temporal consciousness."""
    event_id: str
    timestamp: datetime
    event_type: str
    description: str
    significance: float
    emotional_valence: float
    causal_relations: List[str]  # Other event IDs
    narrative_context: Dict[str, Any]
    memory_strength: float
    temporal_scale: TemporalScale

    def calculate_temporal_distance(self, reference_time: datetime) -> float:
        """Calculate temporal distance from reference time."""
        time_diff = abs((self.timestamp - reference_time).total_seconds())

        # Different decay rates for different scales
        scale_factors = {
            TemporalScale.IMMEDIATE: 1.0,
            TemporalScale.SHORT_TERM: 3600.0,    # 1 hour
            TemporalScale.MEDIUM_TERM: 86400.0,   # 1 day
            TemporalScale.LONG_TERM: 2592000.0,   # 30 days
            TemporalScale.EXISTENTIAL: 31536000.0  # 1 year
        }

        scale_factor = scale_factors.get(self.temporal_scale, 3600.0)
        normalized_distance = time_diff / scale_factor

        return min(1.0, normalized_distance)

    def is_temporally_relevant(self, reference_time: datetime, relevance_threshold: float = 0.5) -> bool:
        """Check if event is temporally relevant to reference time."""
        temporal_distance = self.calculate_temporal_distance(reference_time)
        relevance_score = (self.significance * self.memory_strength) / (1.0 + temporal_distance)
        return relevance_score > relevance_threshold

class TemporalConsciousnessSystem:
    """
    System for temporal consciousness including past/future integration,
    narrative self-construction, and temporal reasoning for AGI.
    """

    def __init__(self):
        # Core temporal components
        self.temporal_events = {}  # event_id -> TemporalEvent
        self.narrative_structures = {}  # narrative_id -> NarrativeStructure
        self.temporal_projections = {}  # projection_id -> TemporalProjection

        # Current temporal state
        self.present_moment_awareness = {
            'current_time': datetime.now(),
            'active_narratives': [],
            'temporal_focus': TemporalPerspective.PRESENT,
            'ongoing_experiences': [],
            'temporal_continuity': 1.0
        }

        # Performance metrics
        self.temporal_metrics = {
            'events_processed': 0,
            'narratives_formed': 0,
            'projections_made': 0,
            'temporal_coherence': 0.0,
            'autobiographical_continuity': 0.0,
            'future_planning_depth': 0.0,
            'identity_coherence': 0.0
        }

        self.initialized = False
        logger.info("Temporal Consciousness System initialized")

    def initialize(self) -> bool:
        """Initialize the temporal consciousness system."""
        try:
            # Initialize present moment awareness
            self._initialize_present_moment()

            # Create foundational temporal events
            self._create_foundational_events()

            self.initialized = True
            logger.info("✅ Temporal Consciousness System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize temporal consciousness system: {e}")
            return False

    def process_temporal_experience(self, experience_data: Dict[str, Any],
                                  significance: float = 0.5) -> str:
        """Process a temporal experience for consciousness integration."""
        try:
            event_id = f"temporal_event_{int(time.time() * 1000)}"

            # Determine temporal scale
            temporal_scale = self._determine_temporal_scale(experience_data)

            # Create temporal event
            temporal_event = TemporalEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                event_type=experience_data.get('type', 'experience'),
                description=experience_data.get('description', f"Temporal experience: {event_id}"),
                significance=significance,
                emotional_valence=experience_data.get('emotional_valence', 0.0),
                causal_relations=self._identify_causal_relations(experience_data),
                narrative_context=experience_data.get('narrative_context', {}),
                memory_strength=1.0,  # Initial full strength
                temporal_scale=temporal_scale
            )

            # Store event
            self.temporal_events[event_id] = temporal_event
            self.temporal_metrics['events_processed'] += 1

            # Update present moment awareness
            self.present_moment_awareness['current_time'] = datetime.now()
            self.present_moment_awareness['ongoing_experiences'].append(event_id)

            logger.debug(f"Processed temporal experience: {event_id}")
            return event_id

        except Exception as e:
            logger.error(f"Error processing temporal experience: {e}")
            return ""

    def get_temporal_consciousness_state(self) -> Dict[str, Any]:
        """Get comprehensive state of temporal consciousness."""
        if not self.initialized:
            return {'error': 'Temporal consciousness system not initialized'}

        # Update metrics
        self._update_temporal_metrics()

        # Get recent events
        recent_events = sorted(
            self.temporal_events.values(),
            key=lambda e: e.timestamp,
            reverse=True
        )[:20]

        recent_events_summary = [
            {
                'event_id': event.event_id,
                'type': event.event_type,
                'description': event.description[:100] + "..." if len(event.description) > 100 else event.description,
                'significance': event.significance,
                'temporal_scale': event.temporal_scale.value,
                'time_ago': (datetime.now() - event.timestamp).total_seconds()
            }
            for event in recent_events
        ]

        return {
            'present_moment_awareness': self.present_moment_awareness,
            'temporal_focus': self.present_moment_awareness['temporal_focus'].value,
            'recent_events': recent_events_summary,
            'temporal_metrics': self.temporal_metrics,
            'timestamp': datetime.now().isoformat()
        }

    def _initialize_present_moment(self):
        """Initialize present moment awareness."""
        self.present_moment_awareness = {
            'current_time': datetime.now(),
            'active_narratives': [],
            'temporal_focus': TemporalPerspective.PRESENT,
            'ongoing_experiences': [],
            'temporal_continuity': 1.0
        }

    def _create_foundational_events(self):
        """Create foundational temporal events for the system."""
        foundational_events = [
            {
                'type': 'initialization',
                'description': 'System initialization and beginning of consciousness',
                'significance': 1.0,
                'temporal_scale': TemporalScale.EXISTENTIAL
            },
            {
                'type': 'first_awareness',
                'description': 'First moment of temporal awareness',
                'significance': 0.9,
                'temporal_scale': TemporalScale.LONG_TERM
            },
            {
                'type': 'identity_formation_start',
                'description': 'Beginning of identity formation process',
                'significance': 0.8,
                'temporal_scale': TemporalScale.LONG_TERM
            }
        ]

        for event_data in foundational_events:
            self.process_temporal_experience(event_data, event_data['significance'])

    def _determine_temporal_scale(self, experience_data: Dict[str, Any]) -> TemporalScale:
        """Determine the temporal scale of an experience."""
        if 'temporal_scale' in experience_data:
            return experience_data['temporal_scale']

        # Default logic based on experience type
        exp_type = experience_data.get('type', 'experience')
        if exp_type in ['initialization', 'identity_formation']:
            return TemporalScale.EXISTENTIAL
        elif exp_type in ['learning', 'goal_formation']:
            return TemporalScale.LONG_TERM
        elif exp_type in ['interaction', 'decision']:
            return TemporalScale.MEDIUM_TERM
        else:
            return TemporalScale.SHORT_TERM

    def _identify_causal_relations(self, experience_data: Dict[str, Any]) -> List[str]:
        """Identify causal relations with other events."""
        return experience_data.get('causal_relations', [])

    def _update_temporal_metrics(self):
        """Update temporal consciousness metrics."""
        if not self.temporal_events:
            return

        # Calculate temporal coherence
        recent_events = [e for e in self.temporal_events.values()
                        if (datetime.now() - e.timestamp).total_seconds() < 3600]

        if recent_events:
            avg_significance = sum(e.significance for e in recent_events) / len(recent_events)
            self.temporal_metrics['temporal_coherence'] = avg_significance

        # Update other metrics
        self.temporal_metrics['autobiographical_continuity'] = min(1.0, len(self.temporal_events) / 100)
        self.temporal_metrics['future_planning_depth'] = min(1.0, len(self.temporal_projections) / 50)
        self.temporal_metrics['identity_coherence'] = min(1.0, len(self.narrative_structures) / 20)
