"""
Launch a nanochat training pod on RunPod.

Usage:
  uv run runpod/launch_pod.py                          # defaults from pod_config.py
  uv run runpod/launch_pod.py --branch gated-recursive --gpus 8 --name my-run
  uv run runpod/launch_pod.py --dry-run                # print config, don't create

Reads RUNPOD_API_KEY from environment (or .env file in repo root).
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_env():
    """Load .env file from repo root into os.environ if present."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def build_env_vars(branch: str, extra: dict) -> dict:
    """Build the env dict to pass to the pod."""
    keys_to_forward = [
        "WANDB_API_KEY",
        "HUGGING_FACE_HUB_TOKEN",
        "GITHUB_PAT",
        "GIT_USER_NAME",
        "GIT_USER_EMAIL",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "NANOCHAT_BASE_DIR",
    ]
    env = {}
    for k in keys_to_forward:
        v = os.environ.get(k, "")
        if v:
            env[k] = v
    env["NANOCHAT_BRANCH"] = branch
    env.update(extra)
    return env


def create_pod(api_key: str, config: dict, dry_run: bool = False) -> dict:
    import urllib.request

    url = "https://rest.runpod.io/v1/pods"
    body = json.dumps(config).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    if dry_run:
        print("DRY RUN — would POST to:", url)
        print(json.dumps(config, indent=2))
        return {}
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def list_pods(api_key: str) -> list:
    import urllib.request

    url = "https://rest.runpod.io/v1/pods"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main():
    load_env()
    sys.path.insert(0, str(Path(__file__).parent))
    import pod_config as cfg

    parser = argparse.ArgumentParser(description="Launch a RunPod training pod")
    parser.add_argument("--name",    default="nanochat-gated", help="Pod name")
    parser.add_argument("--branch",  default=cfg.DEFAULT_BRANCH, help="Git branch to check out")
    parser.add_argument("--image",   default=cfg.IMAGE, help="Docker image")
    parser.add_argument("--gpus",    type=int, default=cfg.GPU_COUNT, help="Number of GPUs")
    parser.add_argument("--gpu-type", nargs="+", default=cfg.GPU_TYPE_IDS, help="GPU type IDs")
    parser.add_argument("--disk",    type=int, default=cfg.CONTAINER_DISK_GB, help="Container disk GB")
    parser.add_argument("--volume",  type=int, default=cfg.VOLUME_GB, help="Volume GB")
    parser.add_argument("--env",     nargs="*", default=[], metavar="KEY=VALUE",
                        help="Extra env vars, e.g. --env MYVAR=foo")
    parser.add_argument("--dry-run", action="store_true", help="Print config, don't create pod")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("RUNPOD_API_KEY not set. Add it to .env or export it.")

    extra_env = {}
    for item in (args.env or []):
        k, _, v = item.partition("=")
        extra_env[k.strip()] = v.strip()

    env = build_env_vars(args.branch, extra_env)

    # Read onstart script
    onstart_path = Path(__file__).parent.parent / "runpod_onstart.sh"
    if not onstart_path.exists():
        sys.exit(f"runpod_onstart.sh not found at {onstart_path}")

    pod_config = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": args.gpu_type,
        "gpuCount": args.gpus,
        "cloudType": cfg.CLOUD_TYPE,
        "containerDiskInGb": args.disk,
        "volumeInGb": args.volume,
        "volumeMountPath": cfg.VOLUME_MOUNT_PATH,
        "ports": cfg.PORTS,
        "env": env,
    }

    print(f"Launching pod '{args.name}' on {args.gpus}×{args.gpu_type[0]}, branch={args.branch}")
    result = create_pod(api_key, pod_config, dry_run=args.dry_run)
    if result:
        pod_id = result.get("id", "?")
        cost = result.get("costPerHr", "?")
        print(f"Pod created: id={pod_id}  cost=${cost}/hr")
        print(f"Check status: uv run runpod/pod_status.py {pod_id}")


if __name__ == "__main__":
    main()
