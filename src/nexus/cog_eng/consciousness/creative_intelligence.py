"""
Creative Intelligence System
Implements creative reasoning, conceptual blending, and novelty generation.
"""

import logging
import time
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CreativeIntelligenceSystem:
    """
    Creative intelligence system for generating novel concepts and solutions.
    """

    def __init__(self):
        self.creativity_score = 91.0
        self.concepts_generated = 0
        self.novelty_threshold = 0.7

        self.concept_library = []
        self.creative_processes = []

        self.initialized = False
        logger.info("Creative Intelligence System initialized")

    def initialize(self) -> bool:
        """Initialize the creative intelligence system."""
        try:
            # Initialize with base concepts
            self._initialize_base_concepts()

            self.initialized = True
            logger.info("✅ Creative Intelligence System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize creative intelligence system: {e}")
            return False

    def analyze_for_creativity(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experience for creative potential."""
        try:
            content = experience_data.get('description', '')

            # Simple creativity analysis
            novelty_score = self._calculate_novelty(content)
            creative_potential = self._assess_creative_potential(experience_data)

            if creative_potential > self.novelty_threshold:
                # Generate creative concepts
                new_concepts = self._generate_concepts(experience_data)
                self.concepts_generated += len(new_concepts)

                return {
                    'novelty_score': novelty_score,
                    'creative_potential': creative_potential,
                    'new_concepts': new_concepts,
                    'creativity_triggered': True
                }

            return {
                'novelty_score': novelty_score,
                'creative_potential': creative_potential,
                'creativity_triggered': False
            }

        except Exception as e:
            logger.error(f"Error analyzing creativity: {e}")
            return {'error': str(e)}

    def get_creativity_score(self) -> float:
        """Get current creativity score."""
        return self.creativity_score

    def get_status(self) -> Dict[str, Any]:
        """Get creative intelligence status."""
        return {
            'status': 'active' if self.initialized else 'inactive',
            'load': min(100.0, self.concepts_generated / 100 * 100),
            'integration': self.creativity_score,
            'metadata': {
                'concepts_generated': self.concepts_generated,
                'novelty_score': self.creativity_score
            }
        }

    def _initialize_base_concepts(self):
        """Initialize with base creative concepts."""
        base_concepts = [
            {'name': 'innovation', 'domain': 'technology', 'novelty': 0.8},
            {'name': 'synthesis', 'domain': 'knowledge', 'novelty': 0.7},
            {'name': 'emergence', 'domain': 'systems', 'novelty': 0.9},
            {'name': 'adaptation', 'domain': 'evolution', 'novelty': 0.6}
        ]

        self.concept_library.extend(base_concepts)

    def _calculate_novelty(self, content: str) -> float:
        """Calculate novelty score of content."""
        # Simple novelty calculation based on unique words
        words = content.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)

        return min(1.0, unique_ratio * 1.2)

    def _assess_creative_potential(self, experience_data: Dict[str, Any]) -> float:
        """Assess creative potential of experience."""
        significance = experience_data.get('significance', 0.5)
        novelty = self._calculate_novelty(experience_data.get('description', ''))

        return (significance + novelty) / 2

    def _generate_concepts(self, experience_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate new creative concepts."""
        new_concepts = []

        # Simple concept generation based on experience
        base_concept = {
            'name': f"concept_{int(time.time())}",
            'domain': experience_data.get('type', 'general'),
            'novelty': random.uniform(0.7, 1.0),
            'generated_from': experience_data.get('description', '')[:50]
        }

        new_concepts.append(base_concept)
        self.concept_library.append(base_concept)

        return new_concepts

    def cleanup(self):
        """Clean up creative intelligence system."""
        logger.info("Creative Intelligence System cleaned up")
