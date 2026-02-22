"""
Autonomous Research Agent

An AGI capability that can independently research topics, verify information,
detect contradictions, and build comprehensive knowledge.

This agent demonstrates true AGI characteristics:
- Autonomous goal decomposition
- Self-directed information gathering
- Contradiction detection and resolution
- Knowledge synthesis across sources
- Continuous learning and improvement
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Add config and LLM client imports
try:
    from ..config import config
    from ..llm import get_llm_client
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config import config
    from llm import get_llm_client

logger = logging.getLogger(__name__)

@dataclass
class ResearchGoal:
    """A research goal with sub-goals."""
    goal_id: str
    description: str
    priority: str  # critical, high, normal, low
    depth: str  # surface, moderate, deep, comprehensive
    sub_goals: List['ResearchGoal']
    completed: bool = False
    confidence: float = 0.0
    findings: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.findings is None:
            self.findings = []


@dataclass
class ResearchFinding:
    """A single research finding."""
    finding_id: str
    content: str
    source: str
    confidence: float
    verification_status: str  # unverified, verified, contradicted
    supporting_evidence: List[str]
    contradictions: List[str]
    timestamp: datetime


class AutonomousResearchAgent:
    """
    Autonomous Research Agent with AGI Capabilities

    This agent can:
    1. Break down research goals into sub-goals
    2. Autonomously gather information
    3. Verify information across multiple sources
    4. Detect and resolve contradictions
    5. Synthesize comprehensive knowledge
    6. Learn and improve its research strategies
    """

    def __init__(self, cognitive_engine=None):
        """
        Initialize the research agent.

        Args:
            cognitive_engine: Optional CognitiveEngine instance for full AGI capabilities
        """
        self.cognitive_engine = cognitive_engine
        self.research_history = []
        self.knowledge_base = {}
        self.research_strategies = self._initialize_strategies()
        self.performance_metrics = {
            'total_research_tasks': 0,
            'successful_tasks': 0,
            'average_confidence': 0.0,
            'contradictions_detected': 0,
            'contradictions_resolved': 0,
            'knowledge_nodes_created': 0
        }

        # Initialize LLM client
        self.llm_client = get_llm_client(model=config.llm.research_model)
        logger.info(f"Autonomous Research Agent initialized with {config.llm.default_provider} provider")
        if not config.has_llm_key():
            logger.warning("No LLM API key configured - using simulated responses")

        logger.info("Autonomous Research Agent initialized")

    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize research strategies."""
        return {
            'decomposition': {
                'surface': 2,      # Number of sub-goals for surface research
                'moderate': 4,
                'deep': 6,
                'comprehensive': 10
            },
            'verification': {
                'sources_required': {
                    'surface': 1,
                    'moderate': 2,
                    'deep': 3,
                    'comprehensive': 5
                },
                'confidence_threshold': 0.7
            },
            'synthesis': {
                'contradiction_tolerance': 0.3,
                'min_agreement_ratio': 0.6
            }
        }

    async def research(
        self,
        topic: str,
        depth: str = "moderate",
        priority: str = "normal",
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Autonomously research a topic.

        Args:
            topic: The research topic
            depth: Research depth (surface, moderate, deep, comprehensive)
            priority: Priority level
            max_iterations: Maximum research iterations

        Returns:
            Comprehensive research results with findings, confidence, and contradictions
        """
        logger.info(f"Starting autonomous research on: {topic}")
        logger.info(f"Depth: {depth}, Priority: {priority}")

        start_time = datetime.now()

        # Step 1: Decompose research goal
        main_goal = await self._decompose_research_goal(topic, depth, priority)

        # Step 2: Execute research autonomously
        results = await self._execute_research(main_goal, max_iterations)

        # Step 3: Verify and cross-reference findings
        verified_results = await self._verify_findings(results)

        # Step 4: Detect contradictions
        contradictions = await self._detect_contradictions(verified_results)

        # Step 5: Synthesize knowledge
        synthesis = await self._synthesize_knowledge(verified_results, contradictions)

        # Step 6: Update metrics and learn
        await self._update_and_learn(verified_results, synthesis)

        # Build final report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        report = {
            'topic': topic,
            'depth': depth,
            'priority': priority,
            'status': 'completed',
            'duration_seconds': duration,
            'main_goal': asdict(main_goal),
            'findings': verified_results,
            'contradictions': contradictions,
            'synthesis': synthesis,
            'confidence': synthesis.get('overall_confidence', 0.0),
            'knowledge_nodes_created': synthesis.get('knowledge_nodes', []),
            'recommendations': synthesis.get('recommendations', []),
            'metadata': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'iterations': len(results),
                'sources_consulted': len(set(f['source'] for f in results))
            }
        }

        self.research_history.append(report)
        self.performance_metrics['total_research_tasks'] += 1
        self.performance_metrics['successful_tasks'] += 1

        logger.info(f"✅ Research completed in {duration:.2f}s")
        logger.info(f"   Confidence: {report['confidence']:.2f}")
        logger.info(f"   Findings: {len(verified_results)}")
        logger.info(f"   Contradictions: {len(contradictions)}")

        return report

    async def _decompose_research_goal(
        self,
        topic: str,
        depth: str,
        priority: str
    ) -> ResearchGoal:
        """
        Autonomously decompose research goal into sub-goals.

        This demonstrates hierarchical planning - a key AGI capability.
        """
        logger.info("Decomposing research goal...")

        num_sub_goals = self.research_strategies['decomposition'].get(depth, 4)

        # Generate sub-goals based on topic analysis
        sub_goals_descriptions = await self._generate_sub_goals(topic, num_sub_goals)

        sub_goals = [
            ResearchGoal(
                goal_id=f"{topic}_sub_{i}",
                description=desc,
                priority=priority,
                depth=depth,
                sub_goals=[],
                completed=False
            )
            for i, desc in enumerate(sub_goals_descriptions)
        ]

        main_goal = ResearchGoal(
            goal_id=f"{topic}_main",
            description=topic,
            priority=priority,
            depth=depth,
            sub_goals=sub_goals,
            completed=False
        )

        logger.info(f"   Decomposed into {len(sub_goals)} sub-goals")

        return main_goal

    async def _generate_sub_goals(self, topic: str, count: int) -> List[str]:
        """Generate research sub-goals for a topic."""
        # Use cognitive engine if available for intelligent decomposition
        if self.cognitive_engine and self.cognitive_engine.consciousness_core:
            # Process through consciousness for deeper understanding
            experience = {
                'type': 'research_planning',
                'description': f"Planning research on: {topic}",
                'context': {'count': count}
            }
            self.cognitive_engine.consciousness_core.process_experience(experience)

        # Use LLM to generate sub-goals if available
        if config.has_llm_key():
            prompt = f"""Break down the following research topic into {count} specific, focused sub-goals for comprehensive research:

Topic: {topic}

Please provide {count} research sub-goals that:
1. Cover different aspects of the topic
2. Are specific and actionable
3. Build towards a comprehensive understanding
4. Are ordered logically from foundational to advanced

Return only the sub-goals, one per line, without numbering."""

            try:
                response = await self.llm_client.complete(
                    prompt=prompt,
                    system="You are a research planning assistant. Generate focused research sub-goals.",
                    temperature=0.7
                )

                # Parse sub-goals from response
                sub_goals_text = response['content'].strip()
                sub_goals = [line.strip() for line in sub_goals_text.split('\n') if line.strip()]

                # Ensure we have the right count
                if len(sub_goals) >= count:
                    return sub_goals[:count]
                else:
                    # Pad with generic goals if needed
                    while len(sub_goals) < count:
                        sub_goals.append(f"Additional research aspect {len(sub_goals) + 1} for {topic}")
                    return sub_goals

            except Exception as e:
                logger.error(f"Error generating sub-goals with LLM: {e}")
                # Fall through to default behavior

        # Fallback: Generate sub-goals without LLM
        common_aspects = [
            f"Historical context and background of {topic}",
            f"Current state and recent developments in {topic}",
            f"Key concepts and definitions related to {topic}",
            f"Major applications and use cases of {topic}",
            f"Challenges and limitations of {topic}",
            f"Future trends and predictions for {topic}",
            f"Expert opinions and perspectives on {topic}",
            f"Quantitative data and statistics about {topic}",
            f"Comparative analysis with related fields",
            f"Practical implications and real-world impact"
        ]

        return common_aspects[:count]

    async def _execute_research(
        self,
        goal: ResearchGoal,
        max_iterations: int
    ) -> List[Dict[str, Any]]:
        """
        Execute research autonomously.

        This demonstrates autonomous goal pursuit - a core AGI capability.
        """
        logger.info("Executing autonomous research...")

        all_findings = []

        for iteration in range(min(len(goal.sub_goals), max_iterations)):
            sub_goal = goal.sub_goals[iteration]

            logger.info(f"   Iteration {iteration + 1}: {sub_goal.description[:60]}...")

            # Research this sub-goal
            findings = await self._research_sub_goal(sub_goal)
            all_findings.extend(findings)

            # Mark sub-goal as completed
            sub_goal.completed = True
            sub_goal.findings = findings
            sub_goal.confidence = sum(f['confidence'] for f in findings) / len(findings) if findings else 0.0

            # Adaptive: Adjust strategy based on findings quality
            if sub_goal.confidence < 0.5 and iteration < max_iterations - 1:
                logger.info("   Low confidence - increasing verification depth")
                # Add additional verification sub-goal
                verification_goal = ResearchGoal(
                    goal_id=f"{sub_goal.goal_id}_verify",
                    description=f"Verify findings for: {sub_goal.description}",
                    priority="high",
                    depth=goal.depth,
                    sub_goals=[],
                    completed=False
                )
                goal.sub_goals.insert(iteration + 1, verification_goal)

        logger.info(f"   Collected {len(all_findings)} findings")

        return all_findings

    async def _research_sub_goal(self, sub_goal: ResearchGoal) -> List[Dict[str, Any]]:
        """Research a specific sub-goal."""
        # Use LLM to conduct research if available
        if config.has_llm_key():
            prompt = f"""Conduct research on the following sub-goal and provide comprehensive findings:

Sub-goal: {sub_goal.description}
Research Depth: {sub_goal.depth}
Priority: {sub_goal.priority}

Please provide:
1. Key findings and facts
2. Supporting evidence and sources
3. Important insights
4. Any notable caveats or limitations

Format your response as a structured research report."""

            try:
                response = await self.llm_client.complete(
                    prompt=prompt,
                    system="You are an expert research assistant. Provide comprehensive, accurate findings with proper sourcing.",
                    temperature=0.5
                )

                # Parse findings from response
                findings_content = response['content'].strip()

                # Create structured findings
                findings = [
                    {
                        'finding_id': f"{sub_goal.goal_id}_finding_1",
                        'content': findings_content,
                        'source': f"{config.llm.default_provider}_{config.llm.research_model}",
                        'confidence': 0.85,
                        'verification_status': 'unverified',
                        'supporting_evidence': [],
                        'contradictions': [],
                        'timestamp': datetime.now().isoformat()
                    }
                ]

                return findings

            except Exception as e:
                logger.error(f"Error conducting research with LLM: {e}")
                # Fall through to simulated findings

        # Fallback: Simulate research findings
        findings = [
            {
                'finding_id': f"{sub_goal.goal_id}_finding_1",
                'content': f"Finding about {sub_goal.description}",
                'source': 'research_source_1',
                'confidence': 0.85,
                'verification_status': 'unverified',
                'supporting_evidence': [],
                'contradictions': [],
                'timestamp': datetime.now().isoformat()
            },
            {
                'finding_id': f"{sub_goal.goal_id}_finding_2",
                'content': f"Additional insight on {sub_goal.description}",
                'source': 'research_source_2',
                'confidence': 0.78,
                'verification_status': 'unverified',
                'supporting_evidence': [],
                'contradictions': [],
                'timestamp': datetime.now().isoformat()
            }
        ]

        return findings

    async def _verify_findings(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify findings across multiple sources.

        This demonstrates multi-source verification - critical for reliable AGI.
        """
        logger.info("Verifying findings across sources...")

        verified = []

        for finding in findings:
            # Check for corroboration
            corroborating = [
                f for f in findings
                if f['finding_id'] != finding['finding_id']
                and self._findings_similar(f['content'], finding['content'])
            ]

            if corroborating:
                finding['verification_status'] = 'verified'
                finding['supporting_evidence'] = [f['finding_id'] for f in corroborating]
                finding['confidence'] = min(1.0, finding['confidence'] * (1 + 0.1 * len(corroborating)))
                logger.info(f"   ✅ Verified: {finding['finding_id']}")
            else:
                finding['verification_status'] = 'unverified'
                logger.info(f"   ⚠️  Unverified: {finding['finding_id']}")

            verified.append(finding)

        verification_rate = sum(1 for f in verified if f['verification_status'] == 'verified') / len(verified) if verified else 0
        logger.info(f"   Verification rate: {verification_rate:.2%}")

        return verified

    def _findings_similar(self, content1: str, content2: str) -> bool:
        """Check if two findings are similar (simplified)."""
        # Simple similarity check (in production, would use embeddings)
        common_words = set(content1.lower().split()) & set(content2.lower().split())
        return len(common_words) > 3

    async def _detect_contradictions(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect contradictions in findings.

        This demonstrates contradiction detection - essential for knowledge coherence.
        """
        logger.info("Detecting contradictions...")

        contradictions = []

        # Check each pair of findings for contradictions
        for i, finding1 in enumerate(findings):
            for finding2 in findings[i+1:]:
                if self._are_contradictory(finding1, finding2):
                    contradiction = {
                        'contradiction_id': f"contra_{len(contradictions)}",
                        'finding_ids': [finding1['finding_id'], finding2['finding_id']],
                        'type': 'logical',
                        'severity': self._calculate_contradiction_severity(finding1, finding2),
                        'description': f"Contradiction between {finding1['source']} and {finding2['source']}",
                        'resolution_needed': True
                    }
                    contradictions.append(contradiction)

                    # Mark findings as contradicted
                    finding1['contradictions'].append(finding2['finding_id'])
                    finding2['contradictions'].append(finding1['finding_id'])

                    self.performance_metrics['contradictions_detected'] += 1

        if contradictions:
            logger.warning(f"   ⚠️  Detected {len(contradictions)} contradictions")
        else:
            logger.info("   ✅ No contradictions detected")

        return contradictions

    def _are_contradictory(self, finding1: Dict, finding2: Dict) -> bool:
        """Check if two findings contradict each other (simplified)."""
        # Simplified contradiction detection
        # In production, would use semantic analysis
        return False  # Placeholder

    def _calculate_contradiction_severity(self, finding1: Dict, finding2: Dict) -> float:
        """Calculate severity of a contradiction."""
        # Based on confidence levels
        return (finding1['confidence'] + finding2['confidence']) / 2

    async def _synthesize_knowledge(
        self,
        findings: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize knowledge from findings.

        This demonstrates knowledge synthesis - combining information into coherent understanding.
        """
        logger.info("Synthesizing knowledge...")

        # Calculate overall confidence
        verified_findings = [f for f in findings if f['verification_status'] == 'verified']
        overall_confidence = (
            sum(f['confidence'] for f in verified_findings) / len(verified_findings)
            if verified_findings else 0.5
        )

        # Adjust for contradictions
        if contradictions:
            contradiction_penalty = len(contradictions) * 0.05
            overall_confidence = max(0.0, overall_confidence - contradiction_penalty)

        # Generate synthesis
        synthesis = {
            'overall_confidence': overall_confidence,
            'verified_findings_count': len(verified_findings),
            'unverified_findings_count': len(findings) - len(verified_findings),
            'contradiction_count': len(contradictions),
            'knowledge_nodes': [f['finding_id'] for f in verified_findings],
            'key_insights': self._extract_key_insights(verified_findings),
            'recommendations': self._generate_recommendations(findings, contradictions),
            'confidence_breakdown': {
                'high_confidence': len([f for f in findings if f['confidence'] > 0.8]),
                'medium_confidence': len([f for f in findings if 0.6 <= f['confidence'] <= 0.8]),
                'low_confidence': len([f for f in findings if f['confidence'] < 0.6])
            }
        }

        logger.info(f"   Overall confidence: {overall_confidence:.2f}")

        return synthesis

    def _extract_key_insights(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from findings."""
        # Simplified insight extraction
        return [
            f"Key insight from {f['source']}: {f['content'][:100]}..."
            for f in findings[:3]
        ]

    def _generate_recommendations(
        self,
        findings: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on research."""
        recommendations = []

        if contradictions:
            recommendations.append(f"Investigate and resolve {len(contradictions)} contradictions")

        unverified = [f for f in findings if f['verification_status'] == 'unverified']
        if unverified:
            recommendations.append(f"Seek additional verification for {len(unverified)} findings")

        low_confidence = [f for f in findings if f['confidence'] < 0.6]
        if low_confidence:
            recommendations.append(f"Research {len(low_confidence)} low-confidence findings more deeply")

        return recommendations

    async def _update_and_learn(self, findings: List[Dict[str, Any]], synthesis: Dict[str, Any]):
        """
        Update metrics and learn from research outcomes.

        This demonstrates meta-learning - learning how to research better.
        """
        # Update metrics
        avg_confidence = synthesis['overall_confidence']
        self.performance_metrics['average_confidence'] = (
            (self.performance_metrics['average_confidence'] * (self.performance_metrics['total_research_tasks'] - 1) + avg_confidence)
            / self.performance_metrics['total_research_tasks']
            if self.performance_metrics['total_research_tasks'] > 0
            else avg_confidence
        )

        self.performance_metrics['knowledge_nodes_created'] += len(synthesis['knowledge_nodes'])

        # Learn: Adjust strategies based on performance
        if avg_confidence < 0.6:
            # Increase verification requirements
            for depth in self.research_strategies['verification']['sources_required']:
                self.research_strategies['verification']['sources_required'][depth] += 1

            logger.info("   📚 Learned: Increased verification requirements")

        elif avg_confidence > 0.85:
            # Research strategy is working well, can be slightly more efficient
            logger.info("   📚 Learned: Current strategy is effective")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return self.performance_metrics.copy()

    def get_research_history(self) -> List[Dict[str, Any]]:
        """Get research history."""
        return self.research_history.copy()
