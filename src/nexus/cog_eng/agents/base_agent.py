from typing import Any, Dict, List
from .utils.common import (
    ensure_dir, write_text, now_iso, sign_report,
    default_metrics, default_tests, default_security, default_policy, default_determinism
)

class BaseAgent:
    pod_name = "base"
    def __init__(self, outdir: str):
        self.outdir = outdir
        ensure_dir(outdir)

    def run(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def make_report(self, ticket: Dict[str, Any], artefacts: List[str], notes: str = "") -> Dict[str, Any]:
        report = {
            "ticket_id": ticket["id"],
            "req_ids": ticket.get("req_ids", []),
            "artefacts": artefacts,
            "tests": default_tests(),
            "metrics": default_metrics(),
            "security": default_security(),
            "policy": default_policy(),
            "determinism": default_determinism(),
            "sbom_delta": [],
            "notes": notes,
        }
        report["signature"] = sign_report(report)
        return report
