"""
Consensus Engine - Multi-expert voting and conflict resolution
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import statistics

from .base import ExpertOpinion, Task, TaskType


class ConsensusStrategy(Enum):
    """Strategies for reaching consensus."""
    WEIGHTED_VOTE = "weighted_vote"      # Weight by expert confidence and role fit
    MAJORITY = "majority"                 # Simple majority wins
    UNANIMOUS = "unanimous"               # All must agree
    HIGHEST_CONFIDENCE = "highest_conf"   # Trust most confident expert
    SYNTHESIZED = "synthesized"           # Combine all perspectives


@dataclass
class ConsensusResult:
    """Result of consensus process."""
    decision: str
    confidence: float
    strategy_used: ConsensusStrategy
    participating_experts: List[str]
    agreement_level: float  # 0-1, how much experts agreed
    synthesis: str  # Combined reasoning
    dissenting_views: List[str] = field(default_factory=list)
    voting_breakdown: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsensusEngine:
    """
    Aggregates expert opinions and reaches consensus through voting.
    
    Supports multiple strategies:
    - Weighted voting based on confidence and task fit
    - Majority voting
    - Unanimous agreement requirement  
    - Highest confidence selection
    - Synthesized combination of all views
    """
    
    def __init__(self, default_strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_VOTE):
        self.default_strategy = default_strategy
        self._history: List[ConsensusResult] = []
    
    def reach_consensus(
        self,
        opinions: List[ExpertOpinion],
        task: Task,
        strategy: Optional[ConsensusStrategy] = None
    ) -> ConsensusResult:
        """
        Reach consensus from multiple expert opinions.
        
        Args:
            opinions: List of expert opinions to consider
            task: The original task being decided on
            strategy: Consensus strategy to use (defaults to engine default)
            
        Returns:
            ConsensusResult with decision and metadata
        """
        if not opinions:
            return ConsensusResult(
                decision="No opinions provided",
                confidence=0.0,
                strategy_used=strategy or self.default_strategy,
                participating_experts=[],
                agreement_level=0.0,
                synthesis=""
            )
        
        strategy = strategy or self.default_strategy
        
        if strategy == ConsensusStrategy.WEIGHTED_VOTE:
            result = self._weighted_vote(opinions, task)
        elif strategy == ConsensusStrategy.MAJORITY:
            result = self._majority_vote(opinions)
        elif strategy == ConsensusStrategy.UNANIMOUS:
            result = self._unanimous_vote(opinions)
        elif strategy == ConsensusStrategy.HIGHEST_CONFIDENCE:
            result = self._highest_confidence(opinions)
        elif strategy == ConsensusStrategy.SYNTHESIZED:
            result = self._synthesize(opinions, task)
        else:
            result = self._weighted_vote(opinions, task)
        
        result.strategy_used = strategy
        self._history.append(result)
        return result

    def _weighted_vote(self, opinions: List[ExpertOpinion], task: Task) -> ConsensusResult:
        """Weighted voting based on confidence and task fit."""
        weighted_scores = {}
        
        for opinion in opinions:
            # Base weight from confidence
            weight = opinion.confidence
            
            # Bonus for task-type match (would need persona info)
            # For now, use confidence as primary weight
            weighted_scores[opinion.expert_name] = {
                "weight": weight,
                "recommendation": opinion.recommendation,
                "analysis": opinion.analysis
            }
        
        # Aggregate
        total_weight = sum(s["weight"] for s in weighted_scores.values())
        avg_confidence = total_weight / len(opinions) if opinions else 0
        
        # Find highest weighted recommendation
        best = max(weighted_scores.items(), key=lambda x: x[1]["weight"])
        
        # Calculate agreement
        confidences = [o.confidence for o in opinions]
        agreement = 1.0 - (statistics.stdev(confidences) if len(confidences) > 1 else 0)
        
        return ConsensusResult(
            decision=best[1]["recommendation"],
            confidence=avg_confidence,
            strategy_used=ConsensusStrategy.WEIGHTED_VOTE,
            participating_experts=[o.expert_name for o in opinions],
            agreement_level=agreement,
            synthesis=self._create_synthesis(opinions),
            voting_breakdown={k: v["weight"] for k, v in weighted_scores.items()}
        )
    
    def _majority_vote(self, opinions: List[ExpertOpinion]) -> ConsensusResult:
        """Simple majority voting."""
        # Group by recommendation
        votes = {}
        for opinion in opinions:
            rec = opinion.recommendation
            if rec not in votes:
                votes[rec] = []
            votes[rec].append(opinion)
        
        # Find majority
        majority_rec = max(votes.keys(), key=lambda k: len(votes[k]))
        majority_count = len(votes[majority_rec])
        
        agreement = majority_count / len(opinions) if opinions else 0
        avg_conf = statistics.mean([o.confidence for o in votes[majority_rec]])
        
        # Dissenting views
        dissenting = []
        for rec, ops in votes.items():
            if rec != majority_rec:
                dissenting.extend([f"{o.expert_name}: {o.recommendation}" for o in ops])
        
        return ConsensusResult(
            decision=majority_rec,
            confidence=avg_conf,
            strategy_used=ConsensusStrategy.MAJORITY,
            participating_experts=[o.expert_name for o in opinions],
            agreement_level=agreement,
            synthesis=self._create_synthesis(opinions),
            dissenting_views=dissenting
        )
    
    def _unanimous_vote(self, opinions: List[ExpertOpinion]) -> ConsensusResult:
        """Require unanimous agreement."""
        recommendations = set(o.recommendation for o in opinions)
        
        if len(recommendations) == 1:
            # Unanimous
            return ConsensusResult(
                decision=list(recommendations)[0],
                confidence=statistics.mean([o.confidence for o in opinions]),
                strategy_used=ConsensusStrategy.UNANIMOUS,
                participating_experts=[o.expert_name for o in opinions],
                agreement_level=1.0,
                synthesis=self._create_synthesis(opinions)
            )
        else:
            # No consensus
            return ConsensusResult(
                decision="NO CONSENSUS - Experts disagree",
                confidence=0.0,
                strategy_used=ConsensusStrategy.UNANIMOUS,
                participating_experts=[o.expert_name for o in opinions],
                agreement_level=1.0 / len(recommendations),
                synthesis=self._create_synthesis(opinions),
                dissenting_views=[f"{o.expert_name}: {o.recommendation}" for o in opinions]
            )

    def _highest_confidence(self, opinions: List[ExpertOpinion]) -> ConsensusResult:
        """Trust the most confident expert."""
        best = max(opinions, key=lambda o: o.confidence)
        
        # Others as dissenting
        others = [o for o in opinions if o.expert_name != best.expert_name]
        
        return ConsensusResult(
            decision=best.recommendation,
            confidence=best.confidence,
            strategy_used=ConsensusStrategy.HIGHEST_CONFIDENCE,
            participating_experts=[o.expert_name for o in opinions],
            agreement_level=best.confidence,  # Use confidence as proxy
            synthesis=f"Deferred to {best.expert_name} (highest confidence: {best.confidence:.0%})",
            dissenting_views=[f"{o.expert_name}: {o.recommendation}" for o in others]
        )
    
    def _synthesize(self, opinions: List[ExpertOpinion], task: Task) -> ConsensusResult:
        """Synthesize all perspectives into combined view."""
        synthesis = self._create_synthesis(opinions)
        
        # Average confidence
        avg_conf = statistics.mean([o.confidence for o in opinions])
        
        # Agreement based on confidence variance
        conf_std = statistics.stdev([o.confidence for o in opinions]) if len(opinions) > 1 else 0
        agreement = 1.0 - min(conf_std, 1.0)
        
        # Combine recommendations
        combined = "Synthesized approach: " + " | ".join(
            f"{o.expert_name}: {o.recommendation}" for o in opinions
        )
        
        return ConsensusResult(
            decision=combined,
            confidence=avg_conf,
            strategy_used=ConsensusStrategy.SYNTHESIZED,
            participating_experts=[o.expert_name for o in opinions],
            agreement_level=agreement,
            synthesis=synthesis,
            voting_breakdown={o.expert_name: o.confidence for o in opinions}
        )
    
    def _create_synthesis(self, opinions: List[ExpertOpinion]) -> str:
        """Create a synthesis of all expert analyses."""
        parts = []
        for opinion in opinions:
            summary = opinion.analysis[:200] + "..." if len(opinion.analysis) > 200 else opinion.analysis
            parts.append(f"**{opinion.expert_name}** ({opinion.confidence:.0%}): {summary}")
        return "\n\n".join(parts)
    
    def detect_conflicts(self, opinions: List[ExpertOpinion]) -> List[Dict[str, Any]]:
        """Detect conflicts between expert opinions."""
        conflicts = []
        
        for i, op1 in enumerate(opinions):
            for op2 in opinions[i+1:]:
                # Check for significant confidence gap with different recommendations
                conf_diff = abs(op1.confidence - op2.confidence)
                rec_differ = op1.recommendation != op2.recommendation
                
                if rec_differ and conf_diff > 0.3:
                    conflicts.append({
                        "experts": [op1.expert_name, op2.expert_name],
                        "type": "recommendation_conflict",
                        "severity": conf_diff,
                        "details": f"{op1.expert_name} recommends '{op1.recommendation}' vs {op2.expert_name} recommends '{op2.recommendation}'"
                    })
                
                # Check for opposing concerns
                op1_concerns = set(op1.concerns)
                op2_concerns = set(op2.concerns)
                if op1_concerns and op2_concerns and not op1_concerns.intersection(op2_concerns):
                    conflicts.append({
                        "experts": [op1.expert_name, op2.expert_name],
                        "type": "concern_mismatch",
                        "severity": 0.5,
                        "details": "Experts have non-overlapping concerns"
                    })
        
        return conflicts
    
    def get_history(self, limit: int = 10) -> List[ConsensusResult]:
        """Get recent consensus history."""
        return self._history[-limit:]
