"""
Test SSH connectivity to a RunPod single A40 pod.

Proves the robust SSH pattern:
  1. Primary:  GraphQL machine.podHostId  → <podHostId>@ssh.runpod.io  (no public IP needed)
  2. Fallback: REST publicIp + portMappings["22"]  → root@<ip> -p <port>

Launches a cheap single A40, waits for it, SSHes in to verify, then terminates.

Usage:
  uv run tmp/test_ssh_a40.py
  uv run tmp/test_ssh_a40.py --no-terminate   # keep pod alive after test
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
if not API_KEY:
    sys.exit("RUNPOD_API_KEY not set. Add to .env or export.")

# SSH keys to try, in order
SSH_KEYS = [
    os.path.expanduser("~/.ssh/id_runpod"),
    os.path.expanduser("~/.ssh/id_ed25519"),
    os.path.expanduser("~/.ssh/id_rsa"),
]
SSH_KEYS = [k for k in SSH_KEYS if os.path.exists(k)]

NO_TERMINATE = "--no-terminate" in sys.argv


# ── REST API helpers ───────────────────────────────────────────────────────────

def rest(method, path, body=None):
    url = f"https://rest.runpod.io/v1{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def graphql(query, variables=None):
    url = "https://api.runpod.io/graphql"
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


# ── Pod lifecycle ──────────────────────────────────────────────────────────────

def create_pod() -> str:
    config = {
        "name": "nanochat-ssh-test",
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "gpuTypeIds": ["NVIDIA A40"],
        "gpuCount": 1,
        "cloudType": "SECURE",
        "containerDiskInGb": 20,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "supportPublicIp": True,   # request a public IP for the direct fallback
    }
    result = rest("POST", "/pods", config)
    return result["id"]


def delete_pod(pod_id: str):
    req = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pod_id}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        method="DELETE",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def get_pod_rest(pod_id: str) -> dict:
    return rest("GET", f"/pods/{pod_id}")


def get_pod_host_id(pod_id: str) -> str | None:
    """Query GraphQL for machine.podHostId — the stable SSH gateway hostname."""
    query = """
    query pod($podId: String!) {
      pod(input: { podId: $podId }) {
        machine { podHostId }
      }
    }
    """
    try:
        resp = graphql(query, {"podId": pod_id})
        return resp["data"]["pod"]["machine"]["podHostId"]
    except Exception as e:
        print(f"  [graphql] podHostId lookup failed: {e}")
        return None


# ── SSH helpers ────────────────────────────────────────────────────────────────

def ssh_run(user_at_host: str, port: int, key: str, cmd: str, timeout=30):
    result = subprocess.run(
        [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-i", key,
            "-p", str(port),
            user_at_host,
            cmd,
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


def try_ssh(user_at_host: str, port: int, cmd: str) -> tuple[str, str, int] | None:
    """Try all available keys; return first success, or None."""
    for key in SSH_KEYS:
        print(f"  Trying key {key} → ssh {user_at_host} -p {port}")
        try:
            stdout, stderr, rc = ssh_run(user_at_host, port, key, cmd)
            if rc == 0:
                print(f"  ✅ Key {key} works")
                return stdout, stderr, rc
            print(f"  ✗ rc={rc}  stderr={stderr.strip()[:100]}")
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timed out")
    return None


# ── Wait loop ─────────────────────────────────────────────────────────────────

def wait_for_pod(pod_id: str, timeout=600) -> dict:
    """Poll until RUNNING with publicIp and portMappings available."""
    start = time.time()
    while time.time() - start < timeout:
        pod = get_pod_rest(pod_id)
        status = pod.get("desiredStatus", "?")
        ip = pod.get("publicIp", "")
        ports = pod.get("portMappings") or {}
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:>3}s] status={status}  publicIp={ip or '—'}  ssh_port={ports.get('22', '—')}")
        if status == "RUNNING" and ip and ports.get("22"):
            return pod
        time.sleep(10)
    raise TimeoutError(f"Pod {pod_id} didn't become ready in {timeout}s")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("nanochat RunPod SSH connectivity test — single A40")
    print("=" * 60)

    # 1. Create pod
    print("\n[1] Creating pod...")
    pod_id = create_pod()
    print(f"    Pod ID : {pod_id}")
    print(f"    Console: https://console.runpod.io/pods/{pod_id}")

    try:
        # 2. Wait for RUNNING
        print("\n[2] Waiting for pod to reach RUNNING status (up to 10 min)...")
        pod = wait_for_pod(pod_id)

        # 3. Resolve SSH target — GraphQL gateway first, direct IP fallback
        print("\n[3] Resolving SSH target...")

        # Primary: GraphQL podHostId → gateway
        pod_host_id = get_pod_host_id(pod_id)
        gateway_target = None
        if pod_host_id:
            gateway_target = (f"{pod_host_id}@ssh.runpod.io", 22)
            print(f"    [GraphQL] podHostId={pod_host_id}")
            print(f"    Gateway:  ssh {pod_host_id}@ssh.runpod.io -p 22")

        # Fallback: direct public IP + mapped port
        public_ip = pod.get("publicIp")
        mapped_port = (pod.get("portMappings") or {}).get("22")
        direct_target = None
        if public_ip and mapped_port:
            direct_target = (f"root@{public_ip}", int(mapped_port))
            print(f"    Direct:   ssh root@{public_ip} -p {mapped_port}")

        if not gateway_target and not direct_target:
            print("    ❌ No SSH target available!")
            return

        # 4. Wait for SSH daemon (pod just started, give it a moment)
        print("\n[4] Waiting 20s for SSH daemon to start...")
        time.sleep(20)

        # 5. Test SSH — gateway first
        print("\n[5] Testing SSH connections...")
        test_cmd = "echo 'SSH_OK'; hostname; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi'"
        success = False

        if gateway_target:
            user_host, port = gateway_target
            print(f"\n  --- Gateway attempt: {user_host} -p {port} ---")
            result = try_ssh(user_host, port, test_cmd)
            if result:
                stdout, _, _ = result
                print(f"\n  Gateway output:\n{stdout}")
                success = True

        if not success and direct_target:
            user_host, port = direct_target
            print(f"\n  --- Direct IP fallback: {user_host} -p {port} ---")
            result = try_ssh(user_host, port, test_cmd)
            if result:
                stdout, _, _ = result
                print(f"\n  Direct output:\n{stdout}")
                success = True

        # 6. Grab a log snapshot too
        if success:
            print("\n[6] Grabbing process list and log tail...")
            log_cmd = "ps aux | grep -E 'torchrun|python|pip|apt|git' | grep -v grep | head -10; echo '---'; tail -n 20 /tmp/pod.log 2>/dev/null || echo '(no /tmp/pod.log yet)'"
            target = gateway_target if gateway_target else direct_target
            result = try_ssh(target[0], target[1], log_cmd)
            if result:
                stdout, _, _ = result
                print(stdout)

        print("\n" + "=" * 60)
        if success:
            print("✅  SSH test PASSED")
        else:
            print("❌  SSH test FAILED — check keys and pod status")
        print("=" * 60)

    finally:
        if NO_TERMINATE:
            print(f"\nPod kept alive (--no-terminate).  Pod ID: {pod_id}")
            print(f"Terminate manually: uv run runpod/stop_pod.py {pod_id} --terminate")
        else:
            print(f"\n[7] Terminating pod {pod_id}...")
            try:
                result = delete_pod(pod_id)
                print(f"    Terminated: {result or 'ok'}")
            except Exception as e:
                print(f"    Termination error: {e}")
                print(f"    Terminate manually: uv run runpod/stop_pod.py {pod_id} --terminate")


if __name__ == "__main__":
    main()
