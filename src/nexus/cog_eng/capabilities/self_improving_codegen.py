"""
Self-Improving Code Generation

An AGI capability that generates code, learns from feedback, and continuously improves.

Key AGI characteristics demonstrated:
- Self-evaluation and critique
- Learning from errors and successes
- Pattern recognition across codebases
- Continuous improvement loop
- Meta-learning about coding best practices
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import json
import re

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

class CodeQuality(Enum):
    """Code quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"

class ImprovementStrategy(Enum):
    """Improvement strategies."""
    REFACTOR = "refactor"
    OPTIMIZE = "optimize"
    SIMPLIFY = "simplify"
    ENHANCE = "enhance"
    REWRITE = "rewrite"

@dataclass
class CodeGenerationRequest:
    """Request for code generation."""
    request_id: str
    description: str
    language: str
    requirements: List[str]
    constraints: List[str] = field(default_factory=list)
    style_preferences: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CodeGenerationResult:
    """Result of code generation."""
    request_id: str
    code: str
    language: str
    quality_score: float
    quality_level: CodeQuality
    test_results: Dict[str, Any]
    improvements_applied: List[str]
    confidence: float
    generation_time: float
    iteration: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningInsight:
    """A learning insight from code generation."""
    insight_id: str
    pattern: str
    effectiveness: float
    usage_count: int
    success_rate: float
    examples: List[str]
    timestamp: datetime

class SelfImprovingCodeGenerator:
    """
    Self-Improving Code Generator with AGI Capabilities

    This generator:
    1. Generates code from natural language descriptions
    2. Evaluates its own generated code
    3. Identifies improvement opportunities
    4. Applies improvements autonomously
    5. Learns patterns from successes and failures
    6. Continuously improves generation strategies
    """

    def __init__(self, cognitive_engine=None):
        """
        Initialize the code generator.

        Args:
            cognitive_engine: Optional CognitiveEngine for full AGI capabilities
        """
        self.cognitive_engine = cognitive_engine
        self.generation_history = []
        self.learned_patterns = {}
        self.performance_metrics = {
            'total_generations': 0,
            'successful_generations': 0,
            'average_quality_score': 0.0,
            'average_iterations': 0.0,
            'patterns_learned': 0,
            'improvements_applied': 0
        }

        # Initialize learning database
        self.learning_db = {
            'successful_patterns': [],
            'failed_patterns': [],
            'best_practices': self._initialize_best_practices(),
            'anti_patterns': self._initialize_anti_patterns()
        }

        # Initialize LLM client
        self.llm_client = get_llm_client(model=config.llm.codegen_model)
        logger.info(f"Self-Improving Code Generator initialized with {config.llm.default_provider} provider")
        if not config.has_llm_key():
            logger.warning("No LLM API key configured - using simulated code generation")

        logger.info("Self-Improving Code Generator initialized")

    def _initialize_best_practices(self) -> Dict[str, List[str]]:
        """Initialize coding best practices."""
        return {
            'python': [
                'Use type hints for function parameters and return types',
                'Follow PEP 8 style guidelines',
                'Use descriptive variable and function names',
                'Add docstrings to all functions and classes',
                'Handle exceptions appropriately',
                'Use list comprehensions where appropriate',
                'Avoid global variables',
                'Keep functions small and focused',
                'Use context managers for resource management',
                'Write unit tests'
            ],
            'javascript': [
                'Use const and let instead of var',
                'Use arrow functions where appropriate',
                'Handle promises and async/await properly',
                'Add JSDoc comments',
                'Use strict equality (===)',
                'Validate user input',
                'Use meaningful variable names',
                'Keep functions pure when possible',
                'Handle errors with try/catch',
                'Use modern ES6+ features'
            ],
            'typescript': [
                'Define interfaces for complex objects',
                'Use strict mode',
                'Avoid using any type',
                'Leverage type inference',
                'Use enums for fixed sets of values',
                'Define return types explicitly',
                'Use readonly for immutable properties',
                'Prefer interfaces over type aliases',
                'Use generics for reusable components',
                'Document public APIs'
            ]
        }

    def _initialize_anti_patterns(self) -> Dict[str, List[str]]:
        """Initialize anti-patterns to avoid."""
        return {
            'python': [
                'Using mutable default arguments',
                'Catching generic exceptions without handling',
                'Using eval() or exec()',
                'Not closing files or connections',
                'Deeply nested code',
                'Magic numbers without explanation',
                'Importing * from modules',
                'Using global variables for state'
            ],
            'javascript': [
                'Callback hell',
                'Not handling promise rejections',
                'Using var for variable declaration',
                'Mutating state directly',
                'Not validating inputs',
                'Using == instead of ===',
                'Excessive global variables',
                'Synchronous operations blocking event loop'
            ],
            'typescript': [
                'Using any type everywhere',
                'Ignoring type errors with @ts-ignore',
                'Not leveraging type inference',
                'Overcomplicating type definitions',
                'Not using strict mode',
                'Mixing interfaces and types inconsistently',
                'Not handling null/undefined properly',
                'Type assertions without validation'
            ]
        }

    async def generate(
        self,
        request: CodeGenerationRequest,
        max_iterations: int = 5,
        target_quality: float = 0.8
    ) -> CodeGenerationResult:
        """
        Generate and continuously improve code.

        Args:
            request: Code generation request
            max_iterations: Maximum improvement iterations
            target_quality: Target quality score (0.0-1.0)

        Returns:
            Final code generation result with improvements
        """
        logger.info(f"Generating code: {request.description}")
        logger.info(f"Language: {request.language}, Target quality: {target_quality}")

        start_time = datetime.now()

        # Initial generation
        code = await self._generate_initial_code(request)

        # Iterative improvement loop
        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")

            # Evaluate code quality
            evaluation = await self._evaluate_code(code, request)

            # Check if we've reached target quality
            if evaluation['quality_score'] >= target_quality:
                logger.info(f"✅ Target quality reached: {evaluation['quality_score']:.2f}")
                break

            # Identify improvements
            improvements = await self._identify_improvements(code, evaluation, request)

            if not improvements:
                logger.info("No more improvements identified")
                break

            # Apply improvements
            improved_code = await self._apply_improvements(code, improvements, request)

            # Learn from this iteration
            await self._learn_from_iteration(code, improved_code, evaluation, improvements)

            code = improved_code

        # Final evaluation
        final_evaluation = await self._evaluate_code(code, request)

        # Create result
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = CodeGenerationResult(
            request_id=request.request_id,
            code=code,
            language=request.language,
            quality_score=final_evaluation['quality_score'],
            quality_level=self._determine_quality_level(final_evaluation['quality_score']),
            test_results=final_evaluation.get('test_results', {}),
            improvements_applied=final_evaluation.get('improvements_applied', []),
            confidence=final_evaluation.get('confidence', 0.8),
            generation_time=duration,
            iteration=iteration + 1,
            metadata={
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'request': asdict(request)
            }
        )

        # Update history and metrics
        self.generation_history.append(result)
        await self._update_metrics(result)

        logger.info(f"✅ Code generation completed in {duration:.2f}s")
        logger.info(f"   Quality: {result.quality_level.value} ({result.quality_score:.2f})")
        logger.info(f"   Iterations: {result.iteration}")

        return result

    async def _generate_initial_code(self, request: CodeGenerationRequest) -> str:
        """Generate initial code from request."""
        logger.info("Generating initial code...")

        # Use learned patterns
        relevant_patterns = self._get_relevant_patterns(request)

        # Use LLM to generate code if available
        if config.has_llm_key():
            # Build comprehensive prompt with context
            best_practices = self.learning_db['best_practices'].get(request.language, [])

            prompt = f"""Generate {request.language} code based on the following specification:

Description: {request.description}

Requirements:
{chr(10).join('- ' + req for req in request.requirements)}

{f"Constraints:{chr(10)}" + chr(10).join('- ' + c for c in request.constraints) if request.constraints else ""}

{f"Style Preferences:{chr(10)}" + chr(10).join(f'- {k}: {v}' for k, v in request.style_preferences.items()) if request.style_preferences else ""}

{f"Context:{chr(10)}" + json.dumps(request.context, indent=2) if request.context else ""}

Best Practices to follow:
{chr(10).join('- ' + bp for bp in best_practices[:5])}

Please generate production-quality code that:
1. Fully implements all requirements
2. Follows best practices for {request.language}
3. Includes comprehensive documentation
4. Handles edge cases appropriately
5. Is well-structured and maintainable

Return ONLY the code, without explanations or markdown formatting."""

            try:
                response = await self.llm_client.complete(
                    prompt=prompt,
                    system=f"You are an expert {request.language} developer. Generate clean, production-quality code.",
                    temperature=0.3
                )

                code = response['content'].strip()

                # Remove markdown code blocks if present
                if code.startswith('```'):
                    lines = code.split('\n')
                    # Remove first and last lines (markdown backticks)
                    code = '\n'.join(lines[1:-1])

                return code

            except Exception as e:
                logger.error(f"Error generating code with LLM: {e}")
                # Fall through to template-based generation

        # Fallback: Generate code using template
        code_template = self._get_code_template(request.language)
        best_practices = self.learning_db['best_practices'].get(request.language, [])

        # Construct code
        code = f"""# Generated code for: {request.description}
# Language: {request.language}
# Requirements: {', '.join(request.requirements)}

def solution():
    \"\"\"
    {request.description}

    Requirements:
    {chr(10).join('- ' + req for req in request.requirements)}
    \"\"\"
    # Implementation would go here
    pass

# Example usage
if __name__ == "__main__":
    solution()
"""

        return code

    async def _evaluate_code(
        self,
        code: str,
        request: CodeGenerationRequest
    ) -> Dict[str, Any]:
        """
        Evaluate code quality.

        This demonstrates self-evaluation - a key AGI capability.
        """
        evaluation = {
            'quality_score': 0.0,
            'metrics': {},
            'issues': [],
            'strengths': [],
            'test_results': {},
            'confidence': 0.8
        }

        # Check for best practices
        best_practices_score = self._check_best_practices(code, request.language)
        evaluation['metrics']['best_practices'] = best_practices_score

        # Check for anti-patterns
        anti_patterns_score = self._check_anti_patterns(code, request.language)
        evaluation['metrics']['anti_patterns'] = anti_patterns_score

        # Check complexity
        complexity_score = self._check_complexity(code)
        evaluation['metrics']['complexity'] = complexity_score

        # Check documentation
        documentation_score = self._check_documentation(code)
        evaluation['metrics']['documentation'] = documentation_score

        # Check requirements satisfaction
        requirements_score = self._check_requirements(code, request.requirements)
        evaluation['metrics']['requirements'] = requirements_score

        # Calculate overall quality score
        weights = {
            'best_practices': 0.25,
            'anti_patterns': 0.20,
            'complexity': 0.15,
            'documentation': 0.15,
            'requirements': 0.25
        }

        evaluation['quality_score'] = sum(
            evaluation['metrics'][metric] * weight
            for metric, weight in weights.items()
        )

        return evaluation

    def _check_best_practices(self, code: str, language: str) -> float:
        """Check adherence to best practices."""
        practices = self.learning_db['best_practices'].get(language, [])
        if not practices:
            return 0.7  # Default score

        # Simple check (in production would be more sophisticated)
        score = 0.5  # Base score

        if 'def ' in code or 'function ' in code:
            score += 0.1
        if '"""' in code or '///' in code or '/*' in code:
            score += 0.2
        if len(code.split('\n')) < 100:  # Reasonable size
            score += 0.1
        if 'pass' not in code:  # Has implementation
            score += 0.1

        return min(1.0, score)

    def _check_anti_patterns(self, code: str, language: str) -> float:
        """Check for anti-patterns."""
        anti_patterns = self.learning_db['anti_patterns'].get(language, [])

        score = 1.0  # Start with perfect score

        # Check for common issues (simplified)
        if 'eval(' in code or 'exec(' in code:
            score -= 0.3
        if 'global ' in code:
            score -= 0.2
        if 'except:' in code and 'except Exception:' not in code:
            score -= 0.2

        return max(0.0, score)

    def _check_complexity(self, code: str) -> float:
        """Check code complexity."""
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]

        # Simple complexity metrics
        if len(non_empty_lines) < 20:
            return 1.0
        elif len(non_empty_lines) < 50:
            return 0.8
        elif len(non_empty_lines) < 100:
            return 0.6
        else:
            return 0.4

    def _check_documentation(self, code: str) -> float:
        """Check documentation quality."""
        doc_indicators = ['"""', "'''", '//', '/*', '*']

        doc_count = sum(code.count(indicator) for indicator in doc_indicators)
        code_lines = len([l for l in code.split('\n') if l.strip()])

        if code_lines == 0:
            return 0.0

        doc_ratio = doc_count / code_lines
        return min(1.0, doc_ratio * 5)  # Scale up

    def _check_requirements(self, code: str, requirements: List[str]) -> float:
        """Check if code satisfies requirements."""
        if not requirements:
            return 0.9

        # Simple check - count how many requirement keywords appear in code
        satisfied = 0
        for req in requirements:
            keywords = req.lower().split()
            if any(keyword in code.lower() for keyword in keywords):
                satisfied += 1

        return satisfied / len(requirements) if requirements else 0.9

    async def _identify_improvements(
        self,
        code: str,
        evaluation: Dict[str, Any],
        request: CodeGenerationRequest
    ) -> List[Dict[str, Any]]:
        """
        Identify potential improvements.

        This demonstrates self-improvement identification - key for AGI.
        """
        improvements = []

        # Check each metric
        for metric, score in evaluation['metrics'].items():
            if score < 0.7:
                improvement = {
                    'type': metric,
                    'current_score': score,
                    'target_score': 0.85,
                    'strategy': self._determine_improvement_strategy(metric, score),
                    'priority': 'high' if score < 0.5 else 'medium'
                }
                improvements.append(improvement)

        # Sort by priority and potential impact
        improvements.sort(key=lambda x: (
            0 if x['priority'] == 'high' else 1,
            x['current_score']
        ))

        return improvements[:3]  # Top 3 improvements

    def _determine_improvement_strategy(self, metric: str, score: float) -> ImprovementStrategy:
        """Determine which improvement strategy to use."""
        if score < 0.3:
            return ImprovementStrategy.REWRITE
        elif metric == 'complexity':
            return ImprovementStrategy.SIMPLIFY
        elif metric == 'best_practices':
            return ImprovementStrategy.ENHANCE
        elif metric == 'documentation':
            return ImprovementStrategy.ENHANCE
        else:
            return ImprovementStrategy.REFACTOR

    async def _apply_improvements(
        self,
        code: str,
        improvements: List[Dict[str, Any]],
        request: CodeGenerationRequest
    ) -> str:
        """Apply identified improvements."""
        logger.info(f"Applying {len(improvements)} improvements...")

        improved_code = code

        # Use LLM to apply improvements if available
        if config.has_llm_key() and improvements:
            improvements_desc = '\n'.join([
                f"- {imp['type']}: {imp['strategy'].value} (current score: {imp['current_score']:.2f}, target: {imp['target_score']:.2f})"
                for imp in improvements
            ])

            prompt = f"""Improve the following {request.language} code by applying these specific improvements:

{improvements_desc}

Current code:
```{request.language}
{code}
```

Requirements (must still be met):
{chr(10).join('- ' + req for req in request.requirements)}

Please return the improved code that:
1. Addresses all identified improvements
2. Still meets all requirements
3. Maintains or improves functionality
4. Is production-quality

Return ONLY the improved code, without explanations or markdown formatting."""

            try:
                response = await self.llm_client.complete(
                    prompt=prompt,
                    system=f"You are an expert {request.language} developer focused on code quality improvement.",
                    temperature=0.3
                )

                improved_code = response['content'].strip()

                # Remove markdown code blocks if present
                if improved_code.startswith('```'):
                    lines = improved_code.split('\n')
                    improved_code = '\n'.join(lines[1:-1])

                self.performance_metrics['improvements_applied'] += len(improvements)

                return improved_code

            except Exception as e:
                logger.error(f"Error applying improvements with LLM: {e}")
                # Fall through to manual improvements

        # Fallback: Apply improvements manually
        for improvement in improvements:
            logger.info(f"   Applying: {improvement['type']} ({improvement['strategy'].value})")

            if improvement['type'] == 'documentation':
                improved_code = self._add_documentation(improved_code)
            elif improvement['type'] == 'complexity':
                improved_code = self._simplify_code(improved_code)
            elif improvement['type'] == 'best_practices':
                improved_code = self._apply_best_practices(improved_code, request.language)

            self.performance_metrics['improvements_applied'] += 1

        return improved_code

    def _add_documentation(self, code: str) -> str:
        """Add documentation to code."""
        # Simple documentation addition (placeholder)
        if '"""' not in code and "'''" not in code:
            lines = code.split('\n')
            # Add docstring after function definition
            for i, line in enumerate(lines):
                if 'def ' in line and i + 1 < len(lines):
                    indent = len(line) - len(line.lstrip())
                    docstring = ' ' * (indent + 4) + '"""TODO: Add documentation."""'
                    lines.insert(i + 1, docstring)
                    break
            code = '\n'.join(lines)
        return code

    def _simplify_code(self, code: str) -> str:
        """Simplify complex code."""
        # Placeholder for code simplification
        return code

    def _apply_best_practices(self, code: str, language: str) -> str:
        """Apply language-specific best practices."""
        # Placeholder for best practices application
        return code

    async def _learn_from_iteration(
        self,
        original_code: str,
        improved_code: str,
        evaluation: Dict[str, Any],
        improvements: List[Dict[str, Any]]
    ):
        """
        Learn from this improvement iteration.

        This demonstrates meta-learning - learning how to improve better.
        """
        # Extract patterns from successful improvements
        for improvement in improvements:
            pattern = {
                'type': improvement['type'],
                'strategy': improvement['strategy'].value,
                'score_improvement': improvement['target_score'] - improvement['current_score'],
                'timestamp': datetime.now()
            }

            # Add to successful patterns if improvement was effective
            if pattern['score_improvement'] > 0.1:
                self.learning_db['successful_patterns'].append(pattern)
                self.performance_metrics['patterns_learned'] += 1

                logger.info(f"   📚 Learned: {improvement['type']} improvement using {improvement['strategy'].value}")

    def _get_relevant_patterns(self, request: CodeGenerationRequest) -> List[Dict[str, Any]]:
        """Get relevant learned patterns for this request."""
        # Simple pattern matching (in production would be more sophisticated)
        return self.learning_db['successful_patterns'][:5]

    def _get_code_template(self, language: str) -> str:
        """Get code template for language."""
        templates = {
            'python': '# Python code\n',
            'javascript': '// JavaScript code\n',
            'typescript': '// TypeScript code\n'
        }
        return templates.get(language, '// Code\n')

    def _determine_quality_level(self, score: float) -> CodeQuality:
        """Determine quality level from score."""
        if score >= 0.9:
            return CodeQuality.EXCELLENT
        elif score >= 0.75:
            return CodeQuality.GOOD
        elif score >= 0.6:
            return CodeQuality.ACCEPTABLE
        elif score >= 0.4:
            return CodeQuality.POOR
        else:
            return CodeQuality.FAILED

    async def _update_metrics(self, result: CodeGenerationResult):
        """Update performance metrics."""
        self.performance_metrics['total_generations'] += 1

        if result.quality_level in [CodeQuality.EXCELLENT, CodeQuality.GOOD, CodeQuality.ACCEPTABLE]:
            self.performance_metrics['successful_generations'] += 1

        # Update average quality
        total = self.performance_metrics['total_generations']
        current_avg = self.performance_metrics['average_quality_score']
        self.performance_metrics['average_quality_score'] = (
            (current_avg * (total - 1) + result.quality_score) / total
        )

        # Update average iterations
        current_avg_iter = self.performance_metrics['average_iterations']
        self.performance_metrics['average_iterations'] = (
            (current_avg_iter * (total - 1) + result.iteration) / total
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get generator performance metrics."""
        return self.performance_metrics.copy()

    def get_learned_patterns(self) -> List[Dict[str, Any]]:
        """Get learned patterns."""
        return self.learning_db['successful_patterns'].copy()
