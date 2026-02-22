import json, os, time, hashlib
from pathlib import Path
from typing import Dict, Any

TRACES_DIR = Path(__file__).resolve().parent / "traces"
ROUTES_PATH = TRACES_DIR / "routes.jsonl"
EVENTS_PATH = TRACES_DIR / "events.jsonl"
STATS_PATH = TRACES_DIR / "route_stats.json"

def _ensure():
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

def record_event(event: Dict[str, Any]) -> None:
    _ensure()
    event = dict(event)
    event["ts"] = int(time.time())
    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

def record_route(job_id: str, pod: str, report: Dict[str, Any]) -> None:
    _ensure()
    row = {
        "job": job_id,
        "pod": pod,
        "success_rate": report.get("metrics",{}).get("success_rate"),
        "p95_ms": report.get("metrics",{}).get("p95_ms"),
        "cost_job": report.get("metrics",{}).get("cost_job"),
    }
    with open(ROUTES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

def recompute_stats() -> Dict[str, Any]:
    _ensure()
    import itertools
    rows = []
    if ROUTES_PATH.exists():
        with open(ROUTES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    stats = {}
    for r in rows:
        key = r["pod"]
        d = stats.setdefault(key, {"n":0,"succ":0.0,"p95_ms":[],"cost":[]})
        d["n"] += 1
        if r.get("success_rate") is not None:
            d["succ"] += float(r["success_rate"])
        if r.get("p95_ms") is not None:
            d["p95_ms"].append(float(r["p95_ms"]))
        if r.get("cost_job") is not None:
            d["cost"].append(float(r["cost_job"]))
    for k,v in stats.items():
        n = max(1, v["n"])
        v["avg_success_rate"] = v["succ"]/n
        v["avg_p95_ms"] = sum(v["p95_ms"])/len(v["p95_ms"]) if v["p95_ms"] else None
        v["avg_cost_job"] = sum(v["cost"])/len(v["cost"]) if v["cost"] else None
        v.pop("succ",None); v.pop("p95_ms",None); v.pop("cost",None)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats
