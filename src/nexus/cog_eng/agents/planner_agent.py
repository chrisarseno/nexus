from typing import Any, Dict, List
from .base_agent import BaseAgent
from .utils.common import write_text

class PlannerAgent(BaseAgent):
    pod_name = "planner"
    def run(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        plan_md = f"""# Plan for {ticket['id']}

**Req IDs:** {', '.join(ticket.get('req_ids', []))}
**Definition of Done:**\n{ticket.get('definition_of_done','')}

## Steps
- Analyze requirements
- Break into sub-tasks
- Map to pods and add fixtures
- Define invariants and budgets
- Create acceptance tests
"""
        plan_path = f"{self.outdir}/plan_{ticket['id']}.md"
        write_text(plan_path, plan_md)
        return self.make_report(ticket, [plan_path], notes="Planner produced initial plan.md")
