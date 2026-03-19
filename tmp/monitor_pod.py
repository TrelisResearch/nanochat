"""
Monitor a RunPod training pod.
Polls status, SSHes in for logs once available, terminates on completion,
and writes a final report to tmp/monitor_report.md.

Usage:
  uv run tmp/monitor_pod.py <pod_id>
  uv run tmp/monitor_pod.py <pod_id> --no-terminate
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
    sys.exit("Usage: uv run tmp/monitor_pod.py <pod_id> [--no-terminate]")
POD_ID = sys.argv[1]
NO_TERMINATE = "--no-terminate" in sys.argv
REPORT_PATH = Path(__file__).parent / f"monitor_report_{POD_ID}.md"
POLL_INTERVAL = 120   # seconds between status polls
SSH_POLL_INTERVAL = 300  # seconds between log grabs via SSH

repo_root = Path(__file__).parent.parent
env_path = repo_root / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ["RUNPOD_API_KEY"]


def api_get(path):
    req = urllib.request.Request(
        f"https://rest.runpod.io/v1{path}",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def api_delete(path):
    req = urllib.request.Request(
        f"https://rest.runpod.io/v1{path}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        method="DELETE",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def get_pod():
    return api_get(f"/pods/{POD_ID}")


SSH_KEYS = [k for k in [
    os.path.expanduser("~/.ssh/id_runpod"),
    os.path.expanduser("~/.ssh/id_ed25519"),
] if os.path.exists(k)]


def ssh_cmd(host, port, cmd, timeout=30):
    """Run cmd on pod via direct SSH. Tries each key in SSH_KEYS."""
    for key in SSH_KEYS:
        result = subprocess.run(
            [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",
                "-i", key,
                "-p", str(port),
                host,
                cmd,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout, result.stderr, result.returncode
    return result.stdout, result.stderr, result.returncode


def get_ssh_target(pod):
    """Return (host_str, port) for SSH using publicIp + portMappings["22"]."""
    port_mappings = pod.get("portMappings") or {}
    mapped_port = port_mappings.get("22")
    public_ip = pod.get("publicIp", "")
    if public_ip and mapped_port:
        return f"root@{public_ip}", int(mapped_port)
    return None, None


def get_logs(host, port):
    """Grab last 60 lines of pod logs."""
    for logfile in ["/tmp/train.log", "/workspace/nanochat/train.log", "/tmp/pod.log"]:
        stdout, stderr, rc = ssh_cmd(
            host, port,
            f"tail -n 60 {logfile} 2>/dev/null || echo 'no log at {logfile}'",
            timeout=15,
        )
        if "no log at" not in stdout and stdout.strip():
            return stdout
    stdout, _, _ = ssh_cmd(host, port, "ps aux | grep -E 'torchrun|python|pip|apt|git' | grep -v grep | head -20", timeout=15)
    return stdout or "(no logs found)"


def write_report(lines):
    REPORT_PATH.write_text("\n".join(lines))
    print(f"[monitor] Report written to {REPORT_PATH}")


def main():
    log = []
    log.append(f"# Monitor Report — Pod {POD_ID}")
    log.append(f"Started: {now()}\n")
    print(f"[monitor] Starting. Polling pod {POD_ID} every {POLL_INTERVAL}s")

    ssh_host = None
    ssh_port = None
    last_log_grab = 0
    log_snapshots = []
    start_time = time.time()

    while True:
        try:
            pod = get_pod()
        except Exception as e:
            print(f"[monitor] API error: {e} — retrying in 60s")
            time.sleep(60)
            continue

        desired = pod.get("desiredStatus", "?")
        cost = pod.get("costPerHr", "?")
        elapsed_hr = (time.time() - start_time) / 3600

        # Get SSH target from portMappings (available early, no runtime needed)
        if not ssh_host:
            ssh_host, ssh_port = get_ssh_target(pod)

        status_line = (
            f"[{now()}] status={desired}  ssh={ssh_host}:{ssh_port or '?'}  "
            f"cost=${cost}/hr  elapsed={elapsed_hr:.1f}hr"
        )
        print(f"[monitor] {status_line}")

        # Grab logs periodically via SSH
        if ssh_host and ssh_port and (time.time() - last_log_grab > SSH_POLL_INTERVAL):
            print(f"[monitor] Grabbing logs via SSH {ssh_host}:{ssh_port}")
            try:
                logs = get_logs(ssh_host, ssh_port)
                snapshot = f"### Log snapshot @ {now()}\n```\n{logs}\n```"
                log_snapshots.append(snapshot)
                print(f"[monitor] Log tail:\n{logs[-500:]}")
                last_log_grab = time.time()
            except Exception as e:
                print(f"[monitor] SSH log grab failed: {e}")

        # Training done when pod exits
        if desired in ("EXITED", "TERMINATED"):
            print(f"[monitor] Pod reached status={desired}. Training complete (or crashed).")
            log.append(f"## Final status: {desired}")
            log.append(f"Elapsed: {elapsed_hr:.2f} hr  (~${cost * elapsed_hr:.2f} total)\n")

            # One last log grab
            if ssh_host and ssh_port:
                try:
                    logs = get_logs(ssh_host, ssh_port)
                    log_snapshots.append(f"### Final log @ {now()}\n```\n{logs}\n```")
                except Exception as e:
                    log_snapshots.append(f"### Final log: SSH failed ({e})")

            # Check HF push
            print("[monitor] Checking HF for pushed model...")
            hf_ok = False
            try:
                hf_req = urllib.request.Request(
                    "https://huggingface.co/api/models/Trelis/nanochat-gated-recursive",
                    headers={"Authorization": f"Bearer {os.environ.get('HUGGING_FACE_HUB_TOKEN', '')}"},
                )
                with urllib.request.urlopen(hf_req, timeout=20) as r:
                    hf_data = json.loads(r.read())
                    hf_ok = True
                    last_modified = hf_data.get("lastModified", "unknown")
                    log.append(f"## HF Push: SUCCESS")
                    log.append(f"Repo: Trelis/nanochat-gated-recursive  lastModified: {last_modified}\n")
            except Exception as e:
                log.append(f"## HF Push: UNKNOWN (check manually) — {e}\n")

            # Terminate pod
            if desired != "TERMINATED":
                print(f"[monitor] Terminating pod {POD_ID}...")
                try:
                    result = api_delete(f"/pods/{POD_ID}")
                    log.append(f"## Pod terminated: {result}\n")
                    print(f"[monitor] Terminated: {result}")
                except Exception as e:
                    log.append(f"## Pod termination failed: {e}\n")
                    print(f"[monitor] Termination error: {e}")

            break

        time.sleep(POLL_INTERVAL)

    # Assemble report
    log.append("## Log Snapshots\n")
    log.extend(log_snapshots)
    log.append(f"\nMonitor finished: {now()}")
    write_report(log)
    print(f"[monitor] Done. Report at {REPORT_PATH}")


if __name__ == "__main__":
    main()
