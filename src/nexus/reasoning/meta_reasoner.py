
"""
Meta-reasoning system for cross-validation and reasoning quality assessment.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ReasoningQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good" 
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"

@dataclass
class ReasoningAssessment:
    """Assessment of reasoning quality."""
    quality: ReasoningQuality
    confidence: float
    logical_consistency: float
    evidence_support: float
    completeness: float
    issues: List[str]
    recommendations: List[str]

class MetaReasoner:
    """
    Meta-reasoning system that evaluates and improves reasoning quality.
    """
    
    def __init__(self):
        self.assessment_history = []
        self.quality_thresholds = {
            ReasoningQuality.EXCELLENT: 0.9,
            ReasoningQuality.GOOD: 0.8,
            ReasoningQuality.ACCEPTABLE: 0.6,
            ReasoningQuality.POOR: 0.4
        }
        
    def assess_reasoning_chain(self, reasoning_result: Dict[str, Any]) -> ReasoningAssessment:
        """Assess the quality of a reasoning chain."""
        
        # Extract reasoning components
        chain = reasoning_result.get('reasoning_chain', {})
        steps = chain.get('steps', [])
        final_answer = reasoning_result.get('final_answer', '')
        confidence = reasoning_result.get('overall_confidence', 0.0)
        
        # Assess different dimensions
        logical_consistency = self._assess_logical_consistency(steps)
        evidence_support = self._assess_evidence_support(steps)
        completeness = self._assess_completeness(steps, final_answer)
        
        # Calculate overall quality score
        overall_score = (logical_consistency * 0.4 + 
                        evidence_support * 0.3 + 
                        completeness * 0.3)
        
        # Determine quality level
        quality = self._determine_quality_level(overall_score)
        
        # Identify issues and recommendations
        issues = self._identify_issues(steps, logical_consistency, evidence_support, completeness)
        recommendations = self._generate_recommendations(quality, issues)
        
        assessment = ReasoningAssessment(
            quality=quality,
            confidence=min(confidence, overall_score),  # Cap confidence by quality
            logical_consistency=logical_consistency,
            evidence_support=evidence_support,
            completeness=completeness,
            issues=issues,
            recommendations=recommendations
        )
        
        self.assessment_history.append(assessment)
        return assessment
    
    def _assess_logical_consistency(self, steps: List[Dict]) -> float:
        """Assess logical consistency of reasoning steps."""
        if not steps:
            return 0.0
        
        consistency_score = 0.0
        valid_steps = 0
        
        for i, step in enumerate(steps):
            step_score = 0.8  # Base score
            
            # Check for contradictions with previous steps
            if i > 0:
                if self._check_contradiction(step, steps[:i]):
                    step_score -= 0.3
            
            # Check internal consistency
            if step.get('confidence', 0) > 0.9 and 'uncertain' in step.get('reasoning', '').lower():
                step_score -= 0.2
            
            consistency_score += max(0, step_score)
            valid_steps += 1
        
        return consistency_score / max(1, valid_steps)
    
    def _assess_evidence_support(self, steps: List[Dict]) -> float:
        """Assess how well evidence supports conclusions."""
        if not steps:
            return 0.0
        
        evidence_score = 0.0
        
        for step in steps:
            reasoning = step.get('reasoning', '')
            confidence = step.get('confidence', 0.5)
            
            # Higher confidence should be backed by stronger reasoning
            reasoning_strength = len(reasoning.split()) / 20.0  # Rough measure
            evidence_ratio = min(1.0, reasoning_strength / max(0.1, confidence))
            evidence_score += evidence_ratio
        
        return min(1.0, evidence_score / len(steps))
    
    def _assess_completeness(self, steps: List[Dict], final_answer: str) -> float:
        """Assess completeness of reasoning."""
        completeness = 0.0
        
        # Check if key reasoning types are present
        step_types = [step.get('type', '') for step in steps]
        required_types = ['analysis', 'synthesis', 'evaluation']
        
        for req_type in required_types:
            if any(req_type in stype.lower() for stype in step_types):
                completeness += 0.3
        
        # Check if final answer is substantive
        if len(final_answer) > 20:
            completeness += 0.1
        
        return min(1.0, completeness)
    
    def _determine_quality_level(self, score: float) -> ReasoningQuality:
        """Determine quality level from score."""
        for quality, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return quality
        return ReasoningQuality.INVALID
    
    def _check_contradiction(self, step: Dict, previous_steps: List[Dict]) -> bool:
        """Check if step contradicts previous steps."""
        # Simple contradiction detection - could be enhanced with NLP
        step_reasoning = step.get('reasoning', '').lower()
        
        for prev_step in previous_steps:
            prev_reasoning = prev_step.get('reasoning', '').lower()
            
            # Look for explicit contradictions
            if ('not' in step_reasoning and any(word in prev_reasoning for word in step_reasoning.split()) or
                'however' in step_reasoning or 'but' in step_reasoning):
                return True
        
        return False
    
    def _identify_issues(self, steps: List[Dict], logical_consistency: float, 
                        evidence_support: float, completeness: float) -> List[str]:
        """Identify specific issues in reasoning."""
        issues = []
        
        if logical_consistency < 0.6:
            issues.append("Logical inconsistencies detected between reasoning steps")
        
        if evidence_support < 0.5:
            issues.append("Insufficient evidence to support conclusions")
        
        if completeness < 0.6:
            issues.append("Reasoning appears incomplete or missing key analysis")
        
        if len(steps) < 3:
            issues.append("Reasoning chain too brief for complex problems")
        
        return issues
    
    def _generate_recommendations(self, quality: ReasoningQuality, issues: List[str]) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        if quality in [ReasoningQuality.POOR, ReasoningQuality.INVALID]:
            recommendations.append("Consider breaking down the problem into smaller sub-problems")
            recommendations.append("Gather additional evidence before reaching conclusions")
        
        if "inconsistencies" in ' '.join(issues):
            recommendations.append("Review reasoning steps for logical consistency")
        
        if "incomplete" in ' '.join(issues):
            recommendations.append("Add verification and evaluation steps")
        
        return recommendations
    
    def get_reasoning_insights(self) -> Dict[str, Any]:
        """Get insights from reasoning assessments."""
        if not self.assessment_history:
            return {"status": "no_data"}
        
        recent_assessments = self.assessment_history[-10:]
        
        avg_quality_score = sum(
            list(self.quality_thresholds.values())[list(self.quality_thresholds.keys()).index(a.quality)]
            for a in recent_assessments
        ) / len(recent_assessments)
        
        common_issues = {}
        for assessment in recent_assessments:
            for issue in assessment.issues:
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        return {
            "avg_quality_score": avg_quality_score,
            "total_assessments": len(self.assessment_history),
            "recent_quality_trend": [a.quality.value for a in recent_assessments],
            "common_issues": common_issues,
            "improvement_areas": [
                issue for issue, count in common_issues.items() 
                if count >= len(recent_assessments) * 0.3
            ]
        }
