
"""
Pattern Reasoning Module - Advanced cognitive reasoning capabilities.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class PatternReasoner:
    """
    Advanced reasoning module that identifies patterns, makes inferences,
    and provides sophisticated analytical capabilities.
    """

    def __init__(self, memory_manager=None, skill_memory=None, factual_memory=None):
        self.name = "pattern_reasoner"
        self.version = "1.0.0"
        self.processed_count = 0
        self.memory_manager = memory_manager
        self.skill_memory = skill_memory
        self.factual_memory = factual_memory
        self.pattern_cache = {}
        self.reasoning_history = []

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Advanced reasoning process that identifies patterns and makes inferences.
        
        Args:
            input_data: The input to analyze and reason about
            
        Returns:
            Dictionary containing reasoning results, patterns, and inferences
        """
        logger.info(f"Pattern reasoner processing: {type(input_data)}")
        self.processed_count += 1

        reasoning_result = {
            'module': self.name,
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'processed_count': self.processed_count,
            'reasoning_chain': [],
            'patterns_identified': [],
            'inferences': [],
            'confidence_scores': {},
            'memory_integration': {}
        }

        # Multi-step reasoning process
        reasoning_result['reasoning_chain'] = self._build_reasoning_chain(input_data)
        reasoning_result['patterns_identified'] = self._identify_patterns(input_data)
        reasoning_result['inferences'] = self._make_inferences(input_data, reasoning_result['patterns_identified'])
        reasoning_result['confidence_scores'] = self._calculate_reasoning_confidence(reasoning_result)
        reasoning_result['memory_integration'] = self._integrate_memory_reasoning(input_data, reasoning_result)

        # Store reasoning session for future reference
        self._store_reasoning_session(input_data, reasoning_result)

        return reasoning_result

    def _build_reasoning_chain(self, input_data: Any) -> List[Dict[str, Any]]:
        """Build a step-by-step reasoning chain."""
        chain = []

        # Step 1: Data classification
        data_type = type(input_data).__name__
        chain.append({
            'step': 1,
            'operation': 'data_classification',
            'description': f'Classified input as {data_type}',
            'result': data_type,
            'confidence': 1.0
        })

        # Step 2: Structure analysis
        if isinstance(input_data, str):
            structure_analysis = self._analyze_text_structure(input_data)
        elif isinstance(input_data, (list, tuple)):
            structure_analysis = self._analyze_sequence_structure(input_data)
        elif isinstance(input_data, dict):
            structure_analysis = self._analyze_dict_structure(input_data)
        else:
            structure_analysis = {'type': 'primitive', 'complexity': 'low'}

        chain.append({
            'step': 2,
            'operation': 'structure_analysis',
            'description': 'Analyzed data structure and complexity',
            'result': structure_analysis,
            'confidence': 0.9
        })

        # Step 3: Context inference
        context_inference = self._infer_context(input_data, structure_analysis)
        chain.append({
            'step': 3,
            'operation': 'context_inference',
            'description': 'Inferred contextual meaning and purpose',
            'result': context_inference,
            'confidence': 0.8
        })

        return chain

    def _identify_patterns(self, input_data: Any) -> List[Dict[str, Any]]:
        """Identify various patterns in the input data."""
        patterns = []

        # Numerical patterns
        if isinstance(input_data, (list, tuple)) and all(isinstance(x, (int, float)) for x in input_data):
            patterns.extend(self._identify_numerical_patterns(input_data))

        # Text patterns
        elif isinstance(input_data, str):
            patterns.extend(self._identify_text_patterns(input_data))

        # Structural patterns
        patterns.extend(self._identify_structural_patterns(input_data))

        return patterns

    def _identify_numerical_patterns(self, data: List) -> List[Dict[str, Any]]:
        """Identify patterns in numerical sequences."""
        patterns = []

        if len(data) < 2:
            return patterns

        # Arithmetic sequence detection
        differences = [data[i+1] - data[i] for i in range(len(data)-1)]
        if all(abs(d - differences[0]) < 0.001 for d in differences):
            patterns.append({
                'type': 'arithmetic_sequence',
                'description': f'Arithmetic sequence with common difference {differences[0]}',
                'confidence': 0.95,
                'parameters': {'common_difference': differences[0]}
            })

        # Geometric sequence detection
        if all(x != 0 for x in data[:-1]):
            ratios = [data[i+1] / data[i] for i in range(len(data)-1)]
            if all(abs(r - ratios[0]) < 0.001 for r in ratios):
                patterns.append({
                    'type': 'geometric_sequence',
                    'description': f'Geometric sequence with common ratio {ratios[0]:.3f}',
                    'confidence': 0.95,
                    'parameters': {'common_ratio': ratios[0]}
                })

        # Fibonacci-like sequence detection
        if len(data) >= 3:
            is_fibonacci = all(abs(data[i] - (data[i-1] + data[i-2])) < 0.001 for i in range(2, len(data)))
            if is_fibonacci:
                patterns.append({
                    'type': 'fibonacci_sequence',
                    'description': 'Fibonacci-like sequence where each term is sum of previous two',
                    'confidence': 0.9,
                    'parameters': {}
                })

        # Trend analysis
        if len(data) > 2:
            slope = self._calculate_trend_slope(data)
            if abs(slope) > 0.1:
                trend_type = 'increasing' if slope > 0 else 'decreasing'
                patterns.append({
                    'type': 'trend',
                    'description': f'{trend_type.capitalize()} trend with slope {slope:.3f}',
                    'confidence': 0.8,
                    'parameters': {'slope': slope, 'direction': trend_type}
                })

        return patterns

    def _identify_text_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Identify patterns in text data."""
        patterns = []

        # Repetition patterns
        words = text.lower().split()
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1

        repeated_words = [(word, freq) for word, freq in word_freq.items() if freq > 1]
        if repeated_words:
            patterns.append({
                'type': 'word_repetition',
                'description': f'Found {len(repeated_words)} repeated words',
                'confidence': 0.9,
                'parameters': {'repeated_words': repeated_words[:5]}  # Top 5
            })

        # Sentence structure patterns
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if avg_sentence_length > 20:
            patterns.append({
                'type': 'complex_sentences',
                'description': f'Complex sentence structure (avg {avg_sentence_length:.1f} words per sentence)',
                'confidence': 0.8,
                'parameters': {'avg_sentence_length': avg_sentence_length}
            })

        # Character patterns
        if any(char.isdigit() for char in text):
            digit_ratio = sum(1 for char in text if char.isdigit()) / len(text)
            patterns.append({
                'type': 'mixed_content',
                'description': f'Mixed text and numbers ({digit_ratio:.1%} digits)',
                'confidence': 0.9,
                'parameters': {'digit_ratio': digit_ratio}
            })

        return patterns

    def _identify_structural_patterns(self, data: Any) -> List[Dict[str, Any]]:
        """Identify structural patterns in data."""
        patterns = []

        # Nesting depth for collections
        if isinstance(data, (list, tuple, dict)):
            depth = self._calculate_nesting_depth(data)
            if depth > 2:
                patterns.append({
                    'type': 'deep_nesting',
                    'description': f'Deep nested structure with depth {depth}',
                    'confidence': 0.9,
                    'parameters': {'nesting_depth': depth}
                })

        # Size patterns
        if hasattr(data, '__len__'):
            size = len(data)
            if size > 100:
                patterns.append({
                    'type': 'large_dataset',
                    'description': f'Large dataset with {size} elements',
                    'confidence': 0.95,
                    'parameters': {'size': size}
                })

        return patterns

    def _make_inferences(self, input_data: Any, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make logical inferences based on identified patterns."""
        inferences = []

        # Inferences from numerical patterns
        arithmetic_patterns = [p for p in patterns if p['type'] == 'arithmetic_sequence']
        if arithmetic_patterns:
            pattern = arithmetic_patterns[0]
            next_value = input_data[-1] + pattern['parameters']['common_difference']
            inferences.append({
                'type': 'prediction',
                'description': f'Next value in arithmetic sequence likely to be {next_value}',
                'prediction': next_value,
                'confidence': 0.9
            })

        # Inferences from trends
        trend_patterns = [p for p in patterns if p['type'] == 'trend']
        if trend_patterns and isinstance(input_data, (list, tuple)):
            trend = trend_patterns[0]
            if trend['parameters']['direction'] == 'increasing':
                inferences.append({
                    'type': 'forecast',
                    'description': 'Data shows increasing trend, expect continued growth',
                    'implication': 'upward_trajectory',
                    'confidence': 0.8
                })

        # Inferences from text complexity
        complex_sentence_patterns = [p for p in patterns if p['type'] == 'complex_sentences']
        if complex_sentence_patterns:
            inferences.append({
                'type': 'content_analysis',
                'description': 'Complex sentence structure suggests formal or technical content',
                'implication': 'formal_register',
                'confidence': 0.7
            })

        return inferences

    def _calculate_reasoning_confidence(self, reasoning_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for different aspects of reasoning."""
        scores = {}

        # Pattern identification confidence
        patterns = reasoning_result.get('patterns_identified', [])
        if patterns:
            avg_pattern_confidence = sum(p.get('confidence', 0) for p in patterns) / len(patterns)
            scores['pattern_identification'] = avg_pattern_confidence
        else:
            scores['pattern_identification'] = 0.5

        # Inference confidence
        inferences = reasoning_result.get('inferences', [])
        if inferences:
            avg_inference_confidence = sum(i.get('confidence', 0) for i in inferences) / len(inferences)
            scores['inference_quality'] = avg_inference_confidence
        else:
            scores['inference_quality'] = 0.5

        # Overall reasoning confidence
        reasoning_chain = reasoning_result.get('reasoning_chain', [])
        if reasoning_chain:
            avg_chain_confidence = sum(step.get('confidence', 0) for step in reasoning_chain) / len(reasoning_chain)
            scores['reasoning_chain'] = avg_chain_confidence
        else:
            scores['reasoning_chain'] = 0.5

        scores['overall'] = (scores['pattern_identification'] + scores['inference_quality'] + scores['reasoning_chain']) / 3

        return scores

    def _integrate_memory_reasoning(self, input_data: Any, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate memory systems into reasoning process."""
        memory_integration = {
            'memory_patterns_found': [],
            'historical_context': [],
            'learned_associations': []
        }

        if not self.memory_manager:
            return memory_integration

        try:
            # Look for similar reasoning patterns in skill memory
            if self.skill_memory:
                reasoning_skills = self.skill_memory.get_contextual_skills('pattern_reasoning')
                memory_integration['memory_patterns_found'] = reasoning_skills[:3]

            # Check factual memory for relevant context
            if self.factual_memory:
                patterns = reasoning_result.get('patterns_identified', [])
                for pattern in patterns:
                    pattern_type = pattern['type']
                    related_facts = self.factual_memory.get_facts_by_category(f'pattern_{pattern_type}', min_confidence=0.6)
                    if related_facts:
                        memory_integration['historical_context'].extend(related_facts[:2])

        except Exception as e:
            logger.warning(f"Error in memory integration: {e}")

        return memory_integration

    def _store_reasoning_session(self, input_data: Any, reasoning_result: Dict[str, Any]):
        """Store reasoning session for future learning."""
        if not self.skill_memory:
            return

        try:
            session_id = f"reasoning_session_{self.processed_count}"
            skill_data = {
                'input_type': type(input_data).__name__,
                'patterns_found': len(reasoning_result.get('patterns_identified', [])),
                'inferences_made': len(reasoning_result.get('inferences', [])),
                'reasoning_steps': len(reasoning_result.get('reasoning_chain', [])),
                'confidence_scores': reasoning_result.get('confidence_scores', {})
            }

            self.skill_memory.learn_skill(session_id, skill_data, 'pattern_reasoning', {
                'session_timestamp': reasoning_result['timestamp'],
                'reasoning_quality': reasoning_result['confidence_scores'].get('overall', 0.5)
            })

        except Exception as e:
            logger.warning(f"Error storing reasoning session: {e}")

    # Helper methods
    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structure of text data."""
        return {
            'word_count': len(text.split()),
            'sentence_count': len(text.split('.')),
            'complexity': 'high' if len(text.split()) > 50 else 'medium' if len(text.split()) > 20 else 'low',
            'has_punctuation': any(char in '.,!?;:' for char in text)
        }

    def _analyze_sequence_structure(self, sequence: List) -> Dict[str, Any]:
        """Analyze the structure of sequence data."""
        return {
            'length': len(sequence),
            'element_types': list(set(type(x).__name__ for x in sequence)),
            'homogeneous': len(set(type(x).__name__ for x in sequence)) == 1,
            'complexity': 'high' if len(sequence) > 100 else 'medium' if len(sequence) > 10 else 'low'
        }

    def _analyze_dict_structure(self, data: Dict) -> Dict[str, Any]:
        """Analyze the structure of dictionary data."""
        return {
            'key_count': len(data.keys()),
            'key_types': list(set(type(k).__name__ for k in data.keys())),
            'value_types': list(set(type(v).__name__ for v in data.values())),
            'complexity': 'high' if len(data) > 20 else 'medium' if len(data) > 5 else 'low'
        }

    def _infer_context(self, input_data: Any, structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Infer context and purpose from data and structure."""
        context = {'domain': 'unknown', 'purpose': 'unknown', 'formality': 'unknown'}

        if isinstance(input_data, str):
            if any(word in input_data.lower() for word in ['algorithm', 'function', 'variable', 'data']):
                context['domain'] = 'technical'
            elif any(word in input_data.lower() for word in ['analysis', 'result', 'conclusion', 'hypothesis']):
                context['domain'] = 'analytical'
            
            if structure_analysis.get('complexity') == 'high':
                context['formality'] = 'formal'

        elif isinstance(input_data, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in input_data):
                context['domain'] = 'numerical'
                context['purpose'] = 'analysis'

        return context

    def _calculate_trend_slope(self, data: List) -> float:
        """Calculate the slope of a trend in numerical data."""
        n = len(data)
        x = list(range(n))
        y = data
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x_squared = sum(x_i ** 2 for x_i in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
        return slope

    def _calculate_nesting_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate the maximum nesting depth of a data structure."""
        if isinstance(data, dict):
            return max(self._calculate_nesting_depth(v, current_depth + 1) for v in data.values()) if data else current_depth + 1
        elif isinstance(data, (list, tuple)):
            return max(self._calculate_nesting_depth(item, current_depth + 1) for item in data) if data else current_depth + 1
        else:
            return current_depth

    def get_capabilities(self) -> Dict[str, Any]:
        """Return module capabilities."""
        return {
            'name': self.name,
            'version': self.version,
            'description': 'Advanced pattern recognition and reasoning capabilities',
            'input_types': ['any'],
            'output_type': 'dict',
            'features': [
                'pattern_identification',
                'logical_inference',
                'trend_analysis',
                'context_inference',
                'reasoning_chain_construction',
                'memory_integration',
                'confidence_scoring',
                'prediction_capabilities'
            ],
            'reasoning_types': [
                'arithmetic_sequence_detection',
                'geometric_sequence_detection',
                'trend_analysis',
                'text_pattern_recognition',
                'structural_analysis',
                'contextual_inference'
            ]
        }
