from typing import Any, Dict, List
from .base_agent import BaseAgent
from .utils.common import write_text

import csv, pathlib, json

class VerificationAgent(BaseAgent):
    pod_name = "verification"
    def run(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        job_root = pathlib.Path(self.outdir).parent
        findings = []
        tests = {"passed": 0, "failed": 0, "skipped": 0}

        # R1: look for dataops/export.csv and ensure at least 1 data row
        export_csv = job_root/'dataops'/'export.csv'
        if export_csv.exists():
            with open(export_csv, newline='', encoding='utf-8') as f:
                rows = list(csv.reader(f))
            data_rows = max(0, len(rows)-1)
            if data_rows > 0:
                findings.append(f"R1 check: export.csv has {data_rows} rows")
                tests["passed"] += 1
            else:
                findings.append("R1 check: export.csv has no data rows")
                tests["failed"] += 1

        # R2: look for extractor_tasks/tasks.csv and ensure at least 1 row
        tasks_csv = job_root/'extractor_tasks'/'tasks.csv'
        if tasks_csv.exists():
            with open(tasks_csv, newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            if len(rows) > 0:
                findings.append(f"R2 check: tasks.csv has {len(rows)} tasks")
                tests["passed"] += 1
            else:
                findings.append("R2 check: tasks.csv has no tasks")
                tests["failed"] += 1

        # R6: dataops_pivot/pivot.csv has at least 1 data row
        pivot_csv = job_root/'dataops_pivot'/'pivot.csv'
        if pivot_csv.exists():
            with open(pivot_csv, newline='', encoding='utf-8') as f:
                rows = list(csv.reader(f))
            data_rows = max(0, len(rows)-1)
            if data_rows > 0:
                findings.append(f"R6 check: pivot.csv has {data_rows} rows")
                tests["passed"] += 1
            else:
                findings.append("R6 check: pivot.csv has no data rows")
                tests["failed"] += 1

        # R7: diff_engine/changes.json has >= 1 change
        ch_json = job_root/'diff_engine'/'changes.json'
        if ch_json.exists():
            try:
                import json as _json
                changes = _json.load(open(ch_json,'r',encoding='utf-8')).get('changes', [])
                if len(changes) > 0:
                    findings.append(f"R7 check: {len(changes)} changed file(s)")
                    tests["passed"] += 1
                else:
                    findings.append("R7 check: no changes detected")
                    tests["failed"] += 1
            except Exception:
                findings.append("R7 check: invalid changes.json")
                tests["failed"] += 1

        # Write a tiny HTML report
        html = "<html><body><h1>QA Report</h1><ul>" + "".join(f"<li>{x}</li>" for x in findings) + "</ul></body></html>"
        path = f"{self.outdir}/qa_{ticket['id']}.html"
        write_text(path, html)

        # Build report
        rep = self.make_report(ticket, [path], notes="Verification executed basic invariant checks")
        rep["tests"] = tests
        # Success heuristic: if any failed, tweak metrics success_rate a bit
        if tests["failed"] > 0:
            rep["metrics"]["success_rate"] = max(0.0, rep["metrics"]["success_rate"] - 0.1)
        else:
            rep["metrics"]["success_rate"] = max(rep["metrics"]["success_rate"], 0.97)
        return rep
