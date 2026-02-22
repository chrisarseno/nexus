
"""
Dynamic learning and adaptation system for continuous improvement.
"""

import logging
import time
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

@dataclass
class LearningEvent:
    """Record of a learning event."""
    event_id: str
    event_type: str
    input_data: Any
    expected_output: Any
    actual_output: Any
    performance_score: float
    timestamp: float
    context: Dict[str, Any]
    feedback: Optional[str] = None

@dataclass
class AdaptationRule:
    """Rule for system adaptation."""
    rule_id: str
    condition: str
    action: str
    priority: int
    success_rate: float
    usage_count: int
    last_applied: float

class DynamicLearner:
    """
    System for continuous learning and adaptation based on
    performance feedback and pattern recognition.
    """
    
    def __init__(self, memory_manager=None, skill_memory=None, factual_memory=None):
        self.memory_manager = memory_manager
        self.skill_memory = skill_memory
        self.factual_memory = factual_memory
        
        # Learning history
        self.learning_events = deque(maxlen=10000)
        self.performance_history = defaultdict(list)
        
        # Adaptation rules
        self.adaptation_rules = {}
        self.rule_performance = defaultdict(list)
        
        # Learning metrics
        self.learning_metrics = {
            "total_events": 0,
            "successful_adaptations": 0,
            "failed_adaptations": 0,
            "average_performance": 0.0,
            "improvement_trend": 0.0
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "acceptable": 0.5,
            "poor": 0.3
        }
        
        # Initialize base adaptation rules
        self._initialize_base_rules()
    
    def record_learning_event(self, event_type: str, input_data: Any, 
                            expected_output: Any, actual_output: Any,
                            context: Dict[str, Any] = None) -> str:
        """Record a learning event for analysis and adaptation."""
        event_id = f"event_{int(time.time())}_{len(self.learning_events)}"
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(
            expected_output, actual_output, event_type
        )
        
        event = LearningEvent(
            event_id=event_id,
            event_type=event_type,
            input_data=input_data,
            expected_output=expected_output,
            actual_output=actual_output,
            performance_score=performance_score,
            timestamp=time.time(),
            context=context or {}
        )
        
        self.learning_events.append(event)
        self.performance_history[event_type].append(performance_score)
        self.learning_metrics["total_events"] += 1
        
        # Update average performance
        self._update_performance_metrics()
        
        # Trigger adaptation if needed
        self._check_adaptation_triggers(event)
        
        logger.info(f"Recorded learning event {event_id} with performance {performance_score:.3f}")
        
        return event_id
    
    def add_feedback(self, event_id: str, feedback: str, corrected_output: Any = None):
        """Add human feedback to a learning event."""
        for event in self.learning_events:
            if event.event_id == event_id:
                event.feedback = feedback
                
                if corrected_output is not None:
                    # Recalculate performance with corrected output
                    new_score = self._calculate_performance_score(
                        event.expected_output, corrected_output, event.event_type
                    )
                    
                    # Store correction as new learning opportunity
                    self._learn_from_correction(event, corrected_output, new_score)
                
                logger.info(f"Added feedback to event {event_id}")
                return True
        
        logger.warning(f"Event {event_id} not found for feedback")
        return False
    
    def adapt_system(self, trigger_event: LearningEvent) -> List[str]:
        """Adapt the system based on learning events and patterns."""
        adaptations_made = []
        
        # Analyze recent performance patterns
        patterns = self._analyze_performance_patterns(trigger_event.event_type)
        
        # Apply relevant adaptation rules
        for pattern in patterns:
            applicable_rules = self._find_applicable_rules(pattern, trigger_event)
            
            for rule in applicable_rules:
                if self._should_apply_rule(rule, trigger_event):
                    success = self._apply_adaptation_rule(rule, trigger_event)
                    
                    if success:
                        adaptations_made.append(rule.rule_id)
                        self.learning_metrics["successful_adaptations"] += 1
                        rule.usage_count += 1
                        rule.last_applied = time.time()
                    else:
                        self.learning_metrics["failed_adaptations"] += 1
        
        # Generate new rules if needed
        if not adaptations_made and trigger_event.performance_score < self.performance_thresholds["acceptable"]:
            new_rule = self._generate_new_rule(trigger_event, patterns)
            if new_rule:
                self.adaptation_rules[new_rule.rule_id] = new_rule
                adaptations_made.append(f"new_rule_{new_rule.rule_id}")
        
        # Update skill memory with new adaptations
        if adaptations_made and self.skill_memory:
            self._store_adaptation_skills(adaptations_made, trigger_event)
        
        return adaptations_made
    
    def get_learning_insights(self, event_type: str = None) -> Dict[str, Any]:
        """Get insights from learning history."""
        if event_type:
            events = [e for e in self.learning_events if e.event_type == event_type]
            performance_data = self.performance_history[event_type]
        else:
            events = list(self.learning_events)
            performance_data = []
            for scores in self.performance_history.values():
                performance_data.extend(scores)
        
        if not events:
            return {"error": "No events found"}
        
        insights = {
            "total_events": len(events),
            "average_performance": statistics.mean(performance_data) if performance_data else 0,
            "performance_trend": self._calculate_trend(performance_data),
            "best_performance": max(performance_data) if performance_data else 0,
            "worst_performance": min(performance_data) if performance_data else 0,
            "recent_performance": statistics.mean(performance_data[-10:]) if len(performance_data) >= 10 else 0,
            "improvement_areas": self._identify_improvement_areas(events),
            "successful_patterns": self._identify_successful_patterns(events),
            "adaptation_effectiveness": self._measure_adaptation_effectiveness()
        }
        
        return insights
    
    def _calculate_performance_score(self, expected: Any, actual: Any, event_type: str) -> float:
        """Calculate performance score based on expected vs actual output."""
        if expected is None or actual is None:
            return 0.0
        
        # Handle different types of comparison
        if isinstance(expected, str) and isinstance(actual, str):
            # Text similarity
            return self._calculate_text_similarity(expected, actual)
        
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            # Numerical accuracy
            if expected == 0:
                return 1.0 if actual == 0 else 0.0
            error_rate = abs(expected - actual) / abs(expected)
            return max(0.0, 1.0 - error_rate)
        
        elif isinstance(expected, dict) and isinstance(actual, dict):
            # Dictionary comparison
            return self._calculate_dict_similarity(expected, actual)
        
        elif isinstance(expected, list) and isinstance(actual, list):
            # List comparison
            return self._calculate_list_similarity(expected, actual)
        
        else:
            # Default exact match
            return 1.0 if expected == actual else 0.0
    
    def _calculate_text_similarity(self, expected: str, actual: str) -> float:
        """Calculate similarity between text strings."""
        if expected == actual:
            return 1.0
        
        # Simple word overlap metric
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(expected_words.intersection(actual_words))
        return overlap / len(expected_words)
    
    def _calculate_dict_similarity(self, expected: dict, actual: dict) -> float:
        """Calculate similarity between dictionaries."""
        if not expected:
            return 1.0 if not actual else 0.0
        
        total_keys = set(expected.keys()).union(set(actual.keys()))
        matching_keys = 0
        
        for key in total_keys:
            if key in expected and key in actual:
                if expected[key] == actual[key]:
                    matching_keys += 1
                elif isinstance(expected[key], (int, float)) and isinstance(actual[key], (int, float)):
                    # Partial credit for numerical values
                    similarity = self._calculate_performance_score(expected[key], actual[key], "numerical")
                    matching_keys += similarity
        
        return matching_keys / len(total_keys) if total_keys else 0.0
    
    def _calculate_list_similarity(self, expected: list, actual: list) -> float:
        """Calculate similarity between lists."""
        if not expected:
            return 1.0 if not actual else 0.0
        
        # Simple overlap ratio
        expected_set = set(str(item) for item in expected)
        actual_set = set(str(item) for item in actual)
        
        overlap = len(expected_set.intersection(actual_set))
        return overlap / len(expected_set)
    
    def _update_performance_metrics(self):
        """Update overall performance metrics."""
        if not self.learning_events:
            return
        
        recent_events = list(self.learning_events)[-100:]  # Last 100 events
        recent_scores = [event.performance_score for event in recent_events]
        
        self.learning_metrics["average_performance"] = statistics.mean(recent_scores)
        
        if len(recent_scores) >= 10:
            # Calculate improvement trend
            first_half = recent_scores[:len(recent_scores)//2]
            second_half = recent_scores[len(recent_scores)//2:]
            
            trend = statistics.mean(second_half) - statistics.mean(first_half)
            self.learning_metrics["improvement_trend"] = trend
    
    def _check_adaptation_triggers(self, event: LearningEvent):
        """Check if adaptation should be triggered based on event."""
        # Trigger on poor performance
        if event.performance_score < self.performance_thresholds["poor"]:
            self.adapt_system(event)
        
        # Trigger on consistent degradation
        recent_scores = self.performance_history[event.event_type][-5:]
        if len(recent_scores) >= 5:
            if all(score < self.performance_thresholds["acceptable"] for score in recent_scores):
                self.adapt_system(event)
    
    def _analyze_performance_patterns(self, event_type: str) -> List[Dict[str, Any]]:
        """Analyze patterns in performance data."""
        patterns = []
        
        if event_type not in self.performance_history:
            return patterns
        
        scores = self.performance_history[event_type]
        if len(scores) < 5:
            return patterns
        
        # Declining performance pattern
        recent_scores = scores[-5:]
        if self._calculate_trend(recent_scores) < -0.1:
            patterns.append({
                "type": "declining_performance",
                "severity": abs(self._calculate_trend(recent_scores)),
                "event_type": event_type
            })
        
        # Inconsistent performance pattern
        if len(scores) >= 10:
            variance = statistics.variance(scores[-10:])
            if variance > 0.1:
                patterns.append({
                    "type": "inconsistent_performance",
                    "variance": variance,
                    "event_type": event_type
                })
        
        # Low performance ceiling pattern
        max_recent = max(scores[-10:]) if len(scores) >= 10 else max(scores)
        if max_recent < self.performance_thresholds["good"]:
            patterns.append({
                "type": "low_ceiling",
                "max_performance": max_recent,
                "event_type": event_type
            })
        
        return patterns
    
    def _calculate_trend(self, scores: List[float]) -> float:
        """Calculate trend in scores (positive = improving, negative = declining)."""
        if len(scores) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = list(range(len(scores)))
        y = scores
        
        n = len(scores)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_xx = sum(x[i] * x[i] for i in range(n))
        
        if n * sum_xx - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        return slope
    
    def _initialize_base_rules(self):
        """Initialize base adaptation rules."""
        base_rules = [
            AdaptationRule(
                rule_id="increase_confidence_threshold",
                condition="declining_performance",
                action="increase_confidence_threshold",
                priority=1,
                success_rate=0.0,
                usage_count=0,
                last_applied=0.0
            ),
            AdaptationRule(
                rule_id="adjust_reasoning_depth",
                condition="inconsistent_performance",
                action="adjust_reasoning_depth",
                priority=2,
                success_rate=0.0,
                usage_count=0,
                last_applied=0.0
            ),
            AdaptationRule(
                rule_id="enhance_memory_integration",
                condition="low_ceiling",
                action="enhance_memory_integration",
                priority=3,
                success_rate=0.0,
                usage_count=0,
                last_applied=0.0
            )
        ]
        
        for rule in base_rules:
            self.adaptation_rules[rule.rule_id] = rule
    
    def _find_applicable_rules(self, pattern: Dict[str, Any], event: LearningEvent) -> List[AdaptationRule]:
        """Find adaptation rules applicable to the given pattern."""
        applicable_rules = []
        
        for rule in self.adaptation_rules.values():
            if rule.condition == pattern["type"]:
                applicable_rules.append(rule)
        
        # Sort by priority and success rate
        applicable_rules.sort(key=lambda r: (r.priority, -r.success_rate))
        
        return applicable_rules
    
    def _should_apply_rule(self, rule: AdaptationRule, event: LearningEvent) -> bool:
        """Determine if a rule should be applied."""
        # Don't apply the same rule too frequently
        if time.time() - rule.last_applied < 300:  # 5 minutes cooldown
            return False
        
        # Consider success rate
        if rule.usage_count > 5 and rule.success_rate < 0.3:
            return False
        
        return True
    
    def _apply_adaptation_rule(self, rule: AdaptationRule, event: LearningEvent) -> bool:
        """Apply an adaptation rule."""
        try:
            if rule.action == "increase_confidence_threshold":
                return self._adjust_confidence_threshold(0.1)
            elif rule.action == "adjust_reasoning_depth":
                return self._adjust_reasoning_depth(event)
            elif rule.action == "enhance_memory_integration":
                return self._enhance_memory_integration(event)
            else:
                logger.warning(f"Unknown adaptation action: {rule.action}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to apply adaptation rule {rule.rule_id}: {e}")
            return False
    
    def _adjust_confidence_threshold(self, adjustment: float) -> bool:
        """Adjust confidence thresholds."""
        # This would interact with other system components
        # For now, just log the adjustment
        logger.info(f"Adjusting confidence threshold by {adjustment}")
        return True
    
    def _adjust_reasoning_depth(self, event: LearningEvent) -> bool:
        """Adjust reasoning depth for better consistency."""
        logger.info(f"Adjusting reasoning depth for event type {event.event_type}")
        return True
    
    def _enhance_memory_integration(self, event: LearningEvent) -> bool:
        """Enhance memory integration to improve performance ceiling."""
        logger.info(f"Enhancing memory integration for {event.event_type}")
        return True
    
    def _generate_new_rule(self, event: LearningEvent, patterns: List[Dict]) -> Optional[AdaptationRule]:
        """Generate a new adaptation rule based on patterns."""
        if not patterns:
            return None
        
        pattern = patterns[0]  # Use the first pattern
        rule_id = f"auto_rule_{int(time.time())}"
        
        new_rule = AdaptationRule(
            rule_id=rule_id,
            condition=pattern["type"],
            action=f"auto_adjust_{pattern['type']}",
            priority=10,  # Low priority for auto-generated rules
            success_rate=0.0,
            usage_count=0,
            last_applied=0.0
        )
        
        logger.info(f"Generated new adaptation rule: {rule_id}")
        return new_rule
    
    def _learn_from_correction(self, original_event: LearningEvent, 
                              corrected_output: Any, new_score: float):
        """Learn from human corrections."""
        correction_event = LearningEvent(
            event_id=f"correction_{original_event.event_id}",
            event_type=f"{original_event.event_type}_corrected",
            input_data=original_event.input_data,
            expected_output=original_event.expected_output,
            actual_output=corrected_output,
            performance_score=new_score,
            timestamp=time.time(),
            context={**original_event.context, "is_correction": True}
        )
        
        self.learning_events.append(correction_event)
        
        # Store correction pattern in skill memory
        if self.skill_memory:
            correction_pattern = {
                "original_output": original_event.actual_output,
                "corrected_output": corrected_output,
                "improvement": new_score - original_event.performance_score,
                "context": original_event.context
            }
            
            self.skill_memory.learn_skill(
                f"correction_{original_event.event_type}_{int(time.time())}",
                correction_pattern,
                "error_correction",
                {"event_type": original_event.event_type}
            )
    
    def _store_adaptation_skills(self, adaptations: List[str], trigger_event: LearningEvent):
        """Store successful adaptations as skills."""
        adaptation_skill = {
            "adaptations": adaptations,
            "trigger_performance": trigger_event.performance_score,
            "trigger_context": trigger_event.context,
            "timestamp": time.time()
        }
        
        try:
            self.skill_memory.learn_skill(
                f"adaptation_{trigger_event.event_type}_{int(time.time())}",
                adaptation_skill,
                "system_adaptation",
                {"event_type": trigger_event.event_type, "adaptation_count": len(adaptations)}
            )
        except Exception as e:
            logger.error(f"Failed to store adaptation skills: {e}")
    
    def _identify_improvement_areas(self, events: List[LearningEvent]) -> List[str]:
        """Identify areas that need improvement."""
        areas = []
        
        # Group by event type and find low performers
        performance_by_type = defaultdict(list)
        for event in events:
            performance_by_type[event.event_type].append(event.performance_score)
        
        for event_type, scores in performance_by_type.items():
            avg_score = statistics.mean(scores)
            if avg_score < self.performance_thresholds["good"]:
                areas.append(event_type)
        
        return areas
    
    def _identify_successful_patterns(self, events: List[LearningEvent]) -> List[Dict]:
        """Identify successful patterns in the events."""
        patterns = []
        
        # Find high-performing events and extract patterns
        high_performers = [e for e in events if e.performance_score > self.performance_thresholds["excellent"]]
        
        if high_performers:
            # Group by context patterns
            context_patterns = defaultdict(list)
            for event in high_performers:
                # Simple pattern based on context keys
                pattern_key = tuple(sorted(event.context.keys())) if event.context else "no_context"
                context_patterns[pattern_key].append(event)
            
            for pattern_key, pattern_events in context_patterns.items():
                if len(pattern_events) >= 3:  # Need at least 3 instances
                    patterns.append({
                        "pattern": pattern_key,
                        "count": len(pattern_events),
                        "avg_performance": statistics.mean(e.performance_score for e in pattern_events),
                        "event_types": list(set(e.event_type for e in pattern_events))
                    })
        
        return patterns
    
    def _measure_adaptation_effectiveness(self) -> Dict[str, float]:
        """Measure how effective adaptations have been."""
        if not self.adaptation_rules:
            return {"no_adaptations": True}
        
        total_rules = len(self.adaptation_rules)
        used_rules = sum(1 for rule in self.adaptation_rules.values() if rule.usage_count > 0)
        
        if used_rules == 0:
            return {"rules_applied": 0, "effectiveness": 0.0}
        
        avg_success_rate = statistics.mean(
            rule.success_rate for rule in self.adaptation_rules.values() 
            if rule.usage_count > 0
        )
        
        return {
            "total_rules": total_rules,
            "rules_applied": used_rules,
            "average_success_rate": avg_success_rate,
            "successful_adaptations": self.learning_metrics["successful_adaptations"],
            "failed_adaptations": self.learning_metrics["failed_adaptations"]
        }
