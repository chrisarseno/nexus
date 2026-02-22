import json, os, time, hashlib, hmac, pathlib
from typing import Dict, Any, List

# HMAC-SHA256 signature using shared secret in env AGENT_SIGNING_KEY
def sign_report(report: Dict[str, Any]) -> str:
    key = os.environ.get("AGENT_SIGNING_KEY", "changeme-secret").encode("utf-8")
    msg = json.dumps(report, sort_keys=True, separators=(',', ':')).encode("utf-8")
    sig = hmac.new(key, msg, hashlib.sha256).hexdigest()
    return f"hmac-sha256:{sig}"

def ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')

def write_text(path: str, content: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def baseline_metrics() -> Dict[str, Any]:
    return {"success_rate": 0.96, "p95_ms": 150000, "cost_job": 0.10, "disagreement": 0.22}

def default_metrics() -> Dict[str, Any]:
    return {"success_rate": 0.97, "p95_ms": 142000, "cost_job": 0.12, "disagreement": 0.22}

def default_tests() -> Dict[str, int]:
    return {"passed": 42, "failed": 0, "skipped": 2}

def default_security() -> Dict[str, int]:
    return {"vulns": 0, "pii_found": 0}

def default_policy() -> Dict[str, bool]:
    return {"allowlist_only": True}

def default_determinism() -> Dict[str, Any]:
    return {"reruns": 3, "hash_match": True}
