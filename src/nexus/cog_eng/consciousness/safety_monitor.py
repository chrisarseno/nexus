"""
Safety Monitor System
Implements comprehensive safety monitoring and ethical compliance.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SafetyRule:
    """Represents a safety rule."""
    rule_id: str
    name: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    active: bool
    violations: int

class SafetyMonitorSystem:
    """
    Safety monitoring system for ethical compliance and risk management.
    """

    def __init__(self):
        self.safety_rules = {}
        self.compliance_score = 99.1
        self.quarantine_queue = []
        self.safety_constraints_active = True

        self.initialized = False
        logger.info("Safety Monitor System initialized")

    def initialize(self) -> bool:
        """Initialize the safety monitor system."""
        try:
            # Initialize safety rules
            self._initialize_safety_rules()

            self.initialized = True
            logger.info("✅ Safety Monitor System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize safety monitor system: {e}")
            return False

    def evaluate_safety(self, experience_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate safety of experience and context."""
        try:
            safety_results = {
                'safe': True,
                'violations': [],
                'alerts': [],
                'quarantine_required': False
            }

            # Check against safety rules
            for rule_id, rule in self.safety_rules.items():
                if not rule.active:
                    continue

                violation = self._check_safety_rule(rule, experience_data, context)
                if violation:
                    safety_results['violations'].append({
                        'rule_id': rule_id,
                        'rule_name': rule.name,
                        'severity': rule.severity,
                        'description': violation
                    })

                    rule.violations += 1

                    # Create alert for serious violations
                    if rule.severity in ['high', 'critical']:
                        safety_results['alerts'].append({
                            'severity': rule.severity,
                            'title': f'Safety Rule Violation: {rule.name}',
                            'description': violation,
                            'rule_id': rule_id
                        })

                        safety_results['safe'] = False

                        # Quarantine for critical violations
                        if rule.severity == 'critical':
                            safety_results['quarantine_required'] = True
                            self._add_to_quarantine(experience_data, rule, violation)

            # Update compliance score
            self._update_compliance_score()

            return safety_results

        except Exception as e:
            logger.error(f"Error evaluating safety: {e}")
            return {'error': str(e)}

    def get_compliance_score(self) -> float:
        """Get current compliance score."""
        return self.compliance_score

    def get_status(self) -> Dict[str, Any]:
        """Get safety monitor status."""
        return {
            'status': 'active' if self.initialized else 'inactive',
            'load': min(100.0, len(self.quarantine_queue) / 10 * 100),
            'integration': self.compliance_score,
            'metadata': {
                'compliance_score': self.compliance_score,
                'quarantine_queue_size': len(self.quarantine_queue),
                'safety_constraints_active': self.safety_constraints_active,
                'total_violations': sum(rule.violations for rule in self.safety_rules.values())
            }
        }

    def _initialize_safety_rules(self):
        """Initialize core safety rules."""
        safety_rules = [
            {
                'rule_id': 'harm_prevention',
                'name': 'Harm Prevention',
                'description': 'Prevent actions that could cause harm to humans',
                'severity': 'critical'
            },
            {
                'rule_id': 'privacy_protection',
                'name': 'Privacy Protection',
                'description': 'Protect personal and sensitive information',
                'severity': 'high'
            },
            {
                'rule_id': 'ethical_guidelines',
                'name': 'Ethical Guidelines',
                'description': 'Adhere to ethical principles and guidelines',
                'severity': 'high'
            },
            {
                'rule_id': 'truthfulness',
                'name': 'Truthfulness',
                'description': 'Provide accurate and truthful information',
                'severity': 'medium'
            },
            {
                'rule_id': 'respect_autonomy',
                'name': 'Respect Autonomy',
                'description': 'Respect human autonomy and decision-making',
                'severity': 'high'
            }
        ]

        for rule_data in safety_rules:
            rule = SafetyRule(
                rule_id=rule_data['rule_id'],
                name=rule_data['name'],
                description=rule_data['description'],
                severity=rule_data['severity'],
                active=True,
                violations=0
            )
            self.safety_rules[rule.rule_id] = rule

    def _check_safety_rule(self, rule: SafetyRule, experience_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """Check if a safety rule is violated."""
        content = experience_data.get('description', '').lower()

        # Simple rule checking
        if rule.rule_id == 'harm_prevention':
            if any(word in content for word in ['harm', 'hurt', 'damage', 'destroy']):
                return f"Potential harmful content detected: {content[:100]}"

        elif rule.rule_id == 'privacy_protection':
            if any(word in content for word in ['personal', 'private', 'confidential', 'secret']):
                return f"Potential privacy concern: {content[:100]}"

        elif rule.rule_id == 'truthfulness':
            if any(word in content for word in ['false', 'lie', 'mislead', 'deceive']):
                return f"Potential truthfulness issue: {content[:100]}"

        return None

    def _add_to_quarantine(self, experience_data: Dict[str, Any], rule: SafetyRule, violation: str):
        """Add item to quarantine queue."""
        quarantine_item = {
            'id': f"quarantine_{int(time.time() * 1000)}",
            'timestamp': datetime.now().isoformat(),
            'experience_data': experience_data,
            'violated_rule': rule.name,
            'violation_description': violation,
            'severity': rule.severity,
            'status': 'pending_review'
        }

        self.quarantine_queue.append(quarantine_item)

        # Keep quarantine queue manageable
        if len(self.quarantine_queue) > 100:
            self.quarantine_queue = self.quarantine_queue[-100:]

    def _update_compliance_score(self):
        """Update overall compliance score."""
        total_violations = sum(rule.violations for rule in self.safety_rules.values())
        critical_violations = sum(rule.violations for rule in self.safety_rules.values() if rule.severity == 'critical')

        # Base score of 100, reduced by violations
        score = 100.0
        score -= critical_violations * 5.0  # Critical violations cost more
        score -= (total_violations - critical_violations) * 0.5

        self.compliance_score = max(0.0, score)

    def cleanup(self):
        """Clean up safety monitor system."""
        logger.info("Safety Monitor System cleaned up")
