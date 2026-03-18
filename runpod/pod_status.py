"""
Check status of one or all running RunPod pods.

Usage:
  uv run runpod/pod_status.py              # list all pods
  uv run runpod/pod_status.py <pod_id>     # status of one pod
"""

import json
import os
import sys
import urllib.request
from pathlib import Path


def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def get_pod(api_key: str, pod_id: str) -> dict:
    url = f"https://rest.runpod.io/v1/pods/{pod_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def list_pods(api_key: str) -> list:
    url = "https://rest.runpod.io/v1/pods"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def fmt_pod(pod: dict) -> str:
    pid    = pod.get("id", "?")
    name   = pod.get("name", "?")
    status = pod.get("desiredStatus", "?")
    cost   = pod.get("costPerHr", "?")
    gpu    = pod.get("machine", {}).get("gpuDisplayName", "?") if pod.get("machine") else "?"
    ip     = pod.get("publicIp", "—")
    ports  = pod.get("portMappings", {})
    ssh_port = ports.get("22", "?") if ports else "?"
    return (
        f"  id={pid}  name={name}  status={status}  "
        f"gpu={gpu}  cost=${cost}/hr  ip={ip}  ssh_port={ssh_port}"
    )


def main():
    load_env()
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        sys.exit("RUNPOD_API_KEY not set.")

    if len(sys.argv) > 1:
        pod_id = sys.argv[1]
        pod = get_pod(api_key, pod_id)
        print(fmt_pod(pod))
        print(json.dumps(pod, indent=2))
    else:
        pods = list_pods(api_key)
        if not pods:
            print("No pods found.")
            return
        print(f"Found {len(pods)} pod(s):")
        for pod in pods:
            print(fmt_pod(pod))


if __name__ == "__main__":
    main()
