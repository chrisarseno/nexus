"""
Analytics cognitive module - provides data analysis and insights with memory integration.
"""

import logging
import statistics
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AnalyticsModule:
    """
    Analytics module that provides statistical analysis and insights.
    Integrates with memory systems for learning and pattern recognition.
    """

    def __init__(self, memory_manager=None, skill_memory=None, factual_memory=None):
        self.name = "analytics"
        self.version = "2.0.0"
        self.processed_count = 0
        self.memory_manager = memory_manager
        self.skill_memory = skill_memory
        self.factual_memory = factual_memory
        self.analysis_patterns = {}
        self.confidence_threshold = 0.8

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input and return analytical insights with memory integration.

        Args:
            input_data: The input to analyze

        Returns:
            Dictionary containing analytical results, metadata, and memory insights
        """
        logger.info(f"Analytics module processing: {input_data}")
        self.processed_count += 1

        # Perform core analysis
        analysis = self._analyze_input(input_data)
        
        # Memory-enhanced analysis
        memory_insights = self._get_memory_insights(input_data, analysis)
        learned_patterns = self._check_learned_patterns(analysis)
        
        # Store new patterns and facts
        self._update_memory_systems(input_data, analysis)

        result = {
            'module': self.name,
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'processed_count': self.processed_count,
            'input_analysis': analysis,
            'memory_insights': memory_insights,
            'learned_patterns': learned_patterns,
            'confidence_score': self._calculate_confidence(analysis, memory_insights)
        }

        return result

    def _analyze_input(self, data: Any) -> Dict[str, Any]:
        """Analyze the input data and provide insights."""
        analysis = {
            'type': type(data).__name__,
            'size': len(str(data)),
        }

        # Text analysis
        if isinstance(data, str):
            analysis.update({
                'character_count': len(data),
                'word_count': len(data.split()),
                'unique_words': len(set(data.lower().split())),
                'avg_word_length': statistics.mean([len(word) for word in data.split()]) if data.split() else 0,
                'has_numbers': any(char.isdigit() for char in data),
                'has_special_chars': any(not char.isalnum() and not char.isspace() for char in data)
            })

        # Numeric analysis
        elif isinstance(data, (int, float)):
            analysis.update({
                'value': data,
                'is_positive': data > 0,
                'is_even': data % 2 == 0 if isinstance(data, int) else None,
                'magnitude': abs(data)
            })

        # List/array analysis
        elif isinstance(data, (list, tuple)):
            if data and all(isinstance(x, (int, float)) for x in data):
                try:
                    analysis.update({
                        'length': len(data),
                        'mean': statistics.mean(data),
                        'median': statistics.median(data),
                        'min': min(data),
                        'max': max(data),
                        'range': max(data) - min(data),
                        'std_dev': statistics.stdev(data) if len(data) > 1 else 0
                    })
                except statistics.StatisticsError:
                    analysis['length'] = len(data)
            else:
                analysis['length'] = len(data)

        return analysis

    def _get_memory_insights(self, input_data: Any, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get insights from memory systems based on current analysis."""
        insights = {
            'similar_analyses': [],
            'relevant_facts': [],
            'applicable_skills': []
        }
        
        if not self.memory_manager:
            return insights

        try:
            # Search for similar analysis patterns in skill memory
            if self.skill_memory:
                analysis_context = f"analytics_{analysis.get('type', 'unknown')}"
                similar_skills = self.skill_memory.get_contextual_skills(analysis_context)
                insights['applicable_skills'] = similar_skills[:3]  # Top 3

            # Search for relevant facts in factual memory
            if self.factual_memory:
                if isinstance(input_data, str):
                    # Look for facts related to text analysis
                    text_facts = self.factual_memory.get_facts_by_category('text_analysis', min_confidence=0.6)
                    insights['relevant_facts'] = text_facts[:3]
                elif isinstance(input_data, (list, tuple)) and all(isinstance(x, (int, float)) for x in input_data):
                    # Look for facts related to numeric analysis
                    numeric_facts = self.factual_memory.get_facts_by_category('numeric_analysis', min_confidence=0.6)
                    insights['relevant_facts'] = numeric_facts[:3]

        except Exception as e:
            logger.warning(f"Error retrieving memory insights: {e}")

        return insights

    def _check_learned_patterns(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for previously learned patterns that match current analysis."""
        patterns = []
        
        # Check for statistical patterns
        if 'mean' in analysis and 'std_dev' in analysis:
            pattern_key = f"stats_{analysis['type']}"
            if pattern_key in self.analysis_patterns:
                stored_pattern = self.analysis_patterns[pattern_key]
                similarity = self._calculate_pattern_similarity(analysis, stored_pattern)
                if similarity > 0.7:
                    patterns.append({
                        'type': 'statistical_pattern',
                        'similarity': similarity,
                        'description': f"Similar to previous {analysis['type']} analysis",
                        'confidence': stored_pattern.get('confidence', 0.5)
                    })

        # Check for text patterns
        if 'character_count' in analysis and 'word_count' in analysis:
            if analysis['word_count'] > 0:
                avg_word_length = analysis['character_count'] / analysis['word_count']
                if avg_word_length > 7:
                    patterns.append({
                        'type': 'text_complexity',
                        'description': 'Text shows high complexity (long average word length)',
                        'confidence': 0.8
                    })

        return patterns

    def _update_memory_systems(self, input_data: Any, analysis: Dict[str, Any]):
        """Update memory systems with new analysis results."""
        if not self.memory_manager:
            return

        try:
            # Store analytical skills
            if self.skill_memory:
                skill_id = f"analysis_{analysis.get('type', 'unknown')}_{self.processed_count}"
                skill_data = {
                    'analysis_type': analysis.get('type'),
                    'input_pattern': self._extract_input_pattern(input_data),
                    'analysis_results': analysis,
                    'processing_method': 'statistical_analysis'
                }
                context = f"analytics_{analysis.get('type', 'general')}"
                self.skill_memory.learn_skill(skill_id, skill_data, context, {
                    'source': 'analytics_module',
                    'confidence': self._calculate_analysis_confidence(analysis)
                })

            # Store factual knowledge about data patterns
            if self.factual_memory and isinstance(input_data, (list, tuple)):
                if all(isinstance(x, (int, float)) for x in input_data) and len(input_data) > 2:
                    fact_id = f"numeric_pattern_{hash(str(sorted(input_data)))}"
                    fact_content = {
                        'data_summary': {
                            'mean': analysis.get('mean'),
                            'median': analysis.get('median'),
                            'std_dev': analysis.get('std_dev'),
                            'range': analysis.get('range')
                        },
                        'pattern_type': 'numeric_sequence'
                    }
                    self.factual_memory.store_fact(
                        fact_id, fact_content, 'numeric_analysis', 
                        source='analytics_module',
                        confidence=0.9
                    )

        except Exception as e:
            logger.warning(f"Error updating memory systems: {e}")

    def _extract_input_pattern(self, input_data: Any) -> Dict[str, Any]:
        """Extract pattern information from input data."""
        pattern = {'type': type(input_data).__name__}
        
        if isinstance(input_data, str):
            pattern.update({
                'length_category': 'short' if len(input_data) < 50 else 'medium' if len(input_data) < 200 else 'long',
                'has_numbers': any(char.isdigit() for char in input_data),
                'has_special_chars': any(not char.isalnum() and not char.isspace() for char in input_data)
            })
        elif isinstance(input_data, (list, tuple)):
            pattern.update({
                'length': len(input_data),
                'element_types': list(set(type(x).__name__ for x in input_data))
            })
            
        return pattern

    def _calculate_pattern_similarity(self, analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> float:
        """Calculate similarity between two analysis patterns."""
        if analysis1.get('type') != analysis2.get('type'):
            return 0.0
            
        common_keys = set(analysis1.keys()) & set(analysis2.keys())
        if not common_keys:
            return 0.0
            
        similarities = []
        for key in common_keys:
            if isinstance(analysis1[key], (int, float)) and isinstance(analysis2[key], (int, float)):
                # Numeric similarity
                if analysis1[key] == 0 and analysis2[key] == 0:
                    similarities.append(1.0)
                elif analysis1[key] == 0 or analysis2[key] == 0:
                    similarities.append(0.0)
                else:
                    ratio = min(analysis1[key], analysis2[key]) / max(analysis1[key], analysis2[key])
                    similarities.append(ratio)
                    
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_analysis_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for analysis results."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on data completeness
        if 'mean' in analysis and 'std_dev' in analysis:
            confidence += 0.2
        if 'median' in analysis:
            confidence += 0.1
        if analysis.get('length', 0) > 10:
            confidence += 0.1
            
        return min(confidence, 1.0)

    def _calculate_confidence(self, analysis: Dict[str, Any], memory_insights: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the analysis."""
        base_confidence = self._calculate_analysis_confidence(analysis)
        
        # Boost confidence if we have relevant memory insights
        if memory_insights.get('applicable_skills'):
            base_confidence += 0.1
        if memory_insights.get('relevant_facts'):
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)

    def get_capabilities(self) -> Dict[str, Any]:
        """Return module capabilities."""
        return {
            'name': self.name,
            'version': self.version,
            'description': 'Provides statistical analysis and data insights with memory integration',
            'input_types': ['string', 'number', 'list', 'any'],
            'output_type': 'dict',
            'features': [
                'text_analysis',
                'numeric_analysis', 
                'statistical_computations',
                'data_profiling',
                'pattern_recognition',
                'memory_integration',
                'learning_from_experience',
                'confidence_scoring'
            ],
            'memory_enabled': True,
            'learning_capable': True
        }