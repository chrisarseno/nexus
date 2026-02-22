"""
Chief Operating Officer (COO) Agent.

The COO is the neural center of the C-suite, responsible for:
- Goal management and operationalization
- Organizational health monitoring
- Strategic prioritization
- Executive reporting to CEO
- Cross-agent coordination and task routing

The COO does NOT execute domain-specific work - it delegates to:
- CIO (Sentinel) - Infrastructure
- CTO (Forge) - Technology/Product
- CSO - Content Strategy
- CKO - Knowledge/Memory
- CRO - Research
- CFO - Finance
"""

from nexus.csuite.coo.agent import COOAgent
from nexus.csuite.coo.managers import (
    GoalManager,
    RoutingManager,
    HealthManager,
    ReportingManager,
)

__all__ = [
    "COOAgent",
    "GoalManager",
    "RoutingManager",
    "HealthManager",
    "ReportingManager",
]
