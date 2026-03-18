"""
Stop (pause) or terminate (destroy) a RunPod pod.

Usage:
  uv run runpod/stop_pod.py <pod_id>              # stop (pause, preserves pod)
  uv run runpod/stop_pod.py <pod_id> --terminate  # permanently destroy
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


def stop_pod(api_key: str, pod_id: str):
    url = f"https://rest.runpod.io/v1/pods/{pod_id}/stop"
    req = urllib.request.Request(
        url,
        data=b"",
        headers={"Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode()


def terminate_pod(api_key: str, pod_id: str):
    url = f"https://rest.runpod.io/v1/pods/{pod_id}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="DELETE",
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode()


def main():
    load_env()
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        sys.exit("RUNPOD_API_KEY not set.")
    if len(sys.argv) < 2:
        sys.exit("Usage: uv run runpod/stop_pod.py <pod_id> [--terminate]")

    pod_id = sys.argv[1]
    terminate = "--terminate" in sys.argv

    if terminate:
        print(f"Terminating (destroying) pod {pod_id} — this is permanent.")
        confirm = input("Type pod id to confirm: ").strip()
        if confirm != pod_id:
            sys.exit("Aborted.")
        result = terminate_pod(api_key, pod_id)
        print("Terminated:", result)
    else:
        result = stop_pod(api_key, pod_id)
        print(f"Stopped pod {pod_id}:", result)


if __name__ == "__main__":
    main()
