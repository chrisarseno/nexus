#!/usr/bin/env python3
"""Agent Referee — CI/Local validator for agent work reports.

Usage:
  python scripts/agent_referee.py --report path/to/work_report.json --policy config/policy.yaml [--baseline path/to/baseline.json] [--check-artifacts]

Exits 0 on pass, 1 on failure, prints a compact verdict with reasons.
Requires: jsonschema, pyyaml
"""
import argparse, json, sys, os, hashlib
from typing import Any, Dict, Optional

try:
    import jsonschema
except ImportError:
    print("[REFEREE] Missing dependency: jsonschema. Please: pip install jsonschema pyyaml", file=sys.stderr)
    sys.exit(2)

try:
    import yaml
except ImportError:
    print("[REFEREE] Missing dependency: pyyaml. Please: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'schemas', 'work_report.schema.json')

def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def validate_schema(payload: Dict[str, Any], schema_path: str) -> None:
    schema = load_json(schema_path)
    jsonschema.validate(instance=payload, schema=schema)

def check_thresholds(report: Dict[str, Any], policy: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> list:
    failures = []
    thr = policy.get('thresholds', {})
    metrics = report.get('metrics', {})
    security = report.get('security', {})
    policy_state = report.get('policy', {})
    determinism = report.get('determinism', {})
    tests = report.get('tests', {})

    # Required thresholds
    succ_min = thr.get('success_min', 0.95)
    p95_max = thr.get('p95_ms_max', 240000)
    cost_max_delta = thr.get('cost_job_max_delta', 0.20)
    d_tau = thr.get('disagreement_tau', 0.35)

    # Tests
    if tests.get('failed', 0) > 0:
        failures.append(f"tests.failed > 0: {tests.get('failed')}")

    # Metrics
    if metrics.get('success_rate', 1.0) < succ_min:
        failures.append(f"success_rate {metrics.get('success_rate')} < {succ_min}")
    if metrics.get('p95_ms', 0) > p95_max:
        failures.append(f"p95_ms {metrics.get('p95_ms')} > {p95_max}")
    # Optional baseline delta
    if baseline and 'metrics' in baseline and 'cost_job' in baseline['metrics'] and 'cost_job' in metrics:
        base = baseline['metrics']['cost_job']
        if base > 0:
            delta = (metrics['cost_job'] - base) / base
            if delta > cost_max_delta:
                failures.append(f"cost_job delta {delta:.2f} > {cost_max_delta}")

    # Disagreement (if present)
    if 'disagreement' in metrics and metrics['disagreement'] > d_tau:
        failures.append(f"disagreement {metrics['disagreement']} > {d_tau}")

    # Security
    if security.get('vulns', 0) > 0:
        failures.append(f"security.vulns {security.get('vulns')} > 0")
    if security.get('pii_found', 0) > 0:
        failures.append(f"security.pii_found {security.get('pii_found')} > 0")

    # Policy & determinism
    if not policy_state.get('allowlist_only', True):
        failures.append("policy.allowlist_only is False")
    if not determinism.get('hash_match', False):
        failures.append("determinism.hash_match is False")

    return failures

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--baseline', required=False)
    ap.add_argument('--check-artifacts', action='store_true')
    args = ap.parse_args()

    report = load_json(args.report)
    policy = load_yaml(args.policy)
    baseline = load_json(args.baseline) if args.baseline else None

    # Schema validation
    try:
        validate_schema(report, SCHEMA_PATH)
    except Exception as e:
        print("[REFEREE] SCHEMA_FAIL:", e, file=sys.stderr)
        sys.exit(1)

    # Optional artifacts check
    if args.check_artifacts:
        missing = [p for p in report.get('artefacts', []) if not os.path.exists(p)]
        if missing:
            print(f"[REFEREE] ARTEFACTS_MISSING: {missing}", file=sys.stderr)
            sys.exit(1)

    failures = check_thresholds(report, policy, baseline)
    if failures:
        print("[REFEREE] FAIL:")
        for f in failures:
            print(" -", f)
        sys.exit(1)

    print("[REFEREE] PASS: all checks satisfied")
    sys.exit(0)

if __name__ == '__main__':
    main()
