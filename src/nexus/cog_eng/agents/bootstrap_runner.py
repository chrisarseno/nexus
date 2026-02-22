#!/usr/bin/env python3
import os, json, subprocess, sys, time, yaml
from pathlib import Path
import importlib.util, json as _json

ROOT = Path(__file__).resolve().parent.parent
AGENT_ROOT = ROOT / "agent_skeletons"
REFEREE = ROOT / "scripts" / "agent_referee.py"
POLICY = ROOT / "config" / "policy.yaml"

def run_cmd(cmd:list, cwd=None)->int:
    print("➜", " ".join(map(str, cmd)))
    return subprocess.call(cmd, cwd=cwd)

def ensure_dir(p:Path):
    p.mkdir(parents=True, exist_ok=True)

def make_ticket(seed:dict)->dict:
    return {
        "id": seed["id"],
        "title": seed["title"],
        "req_ids": seed["req_ids"],
        "definition_of_done": f"Run pods for {seed['title']} and produce artifacts that satisfy invariants.",
        "fixtures": [str((ROOT / fx).resolve()) for fx in seed.get("fixtures", [])],
        "invariants": seed.get("invariants", []),
        "budget": {"cpu_s": 30, "gpu_s": 0, "net_mb": 5, "proxy": 0},
        "assignee": ",".join(seed["pods"]),
        "labels": ["bootstrap"]
    }

def _load_trace_logger():
    import importlib.util, sys
    p = Path(__file__).resolve().parent / 'trace_logger.py'
    spec = importlib.util.spec_from_file_location('trace_logger', str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    seeds_path = ROOT / "bootstrap" / "seeds.yaml"
    out_root = ROOT / "bootstrap" / "out"
    ensure_dir(out_root)
    seeds = yaml.safe_load(open(seeds_path, "r", encoding="utf-8"))
    results = []
    tlog = _load_trace_logger()

    for seed in seeds["seeds"]:
        ticket = make_ticket(seed)
        ticket_path = out_root / f"TaskTicket_{seed['id']}.json"
        json.dump(ticket, open(ticket_path, "w", encoding="utf-8"), indent=2)

        # For each pod in order, run the agent
        job_dir = out_root / seed["id"]
        ensure_dir(job_dir)
        for pod in seed["pods"]:
            code = run_cmd([sys.executable, str(AGENT_ROOT/"run_agent.py"),
                            "--agent", pod,
                            "--ticket", str(ticket_path),
                            "--outdir", str(job_dir/f"{pod}")])
            # Log event
            try:
                tlog.record_event({"job": seed['id'], "pod": pod, "event": "agent_run"})
            except Exception:
                pass
            if code != 0:
                print(f"[ERROR] Agent {pod} failed for {seed['id']} (code={code})")
                results.append({"seed": seed["id"], "pod": pod, "status": "agent_fail"})
                break

            # Validate work_report.json
            report = job_dir/f"{pod}"/"work_report.json"
            # Referee validation will follow
            if not report.exists():
                print(f"[ERROR] Missing work_report.json from {pod}")
                results.append({"seed": seed["id"], "pod": pod, "status": "missing_report"})
                break

            code = run_cmd([sys.executable, str(REFEREE),
                            "--report", str(report),
                            "--policy", str(POLICY)])
            # Log event
            try:
                tlog.record_event({"job": seed['id'], "pod": pod, "event": "agent_run"})
            except Exception:
                pass
            # Record route metrics
            try:
                rep = _json.load(open(report,'r',encoding='utf-8'))
                tlog.record_route(seed['id'], pod, rep)
            except Exception:
                pass
            if code != 0:
                print(f"[BLOCK] Referee blocked {pod} for {seed['id']}")
                results.append({"seed": seed["id"], "pod": pod, "status": "referee_block"})
                break
        else:
            # All pods passed
            results.append({"seed": seed["id"], "status": "pass"})

    # Write summary
    summary_path = out_root / "bootstrap_summary.json"
    json.dump(results, open(summary_path, "w", encoding="utf-8"), indent=2)
    try:
        stats = tlog.recompute_stats()
        print("\n=== Route Stats ===\n", json.dumps(stats, indent=2))
    except Exception:
        pass
    print("\n=== Bootstrap Summary ===")
    print(json.dumps(results, indent=2))
    print(f"Saved: {summary_path}")

if __name__ == "__main__":
    main()
