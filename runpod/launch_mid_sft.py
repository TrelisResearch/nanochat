"""
Launch gated-recursive mid-training + SFT on RunPod.

This is the recommended first run — it does NOT require downloading the large
pre-training dataset. It starts from the nanochat-recursive base checkpoint,
continues with gated mid-training, then SFT.

Workflow on the pod (executed via dockerStartCmd):
  1. Pull base checkpoint from HF  (Trelis/nanochat-recursive, base stage)
  2. Run mid_train.py  with gate_cost in loss
  3. Run chat_sft.py   with gate_cost in loss
  4. Push gated model  to HF  (Trelis/nanochat-gated-recursive)

Usage:
  uv run runpod/launch_mid_sft.py --dry-run        # preview config
  uv run runpod/launch_mid_sft.py --name my-run    # launch
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path


def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


# ── Training command that runs on the pod ────────────────────────────────────
TRAIN_CMD = textwrap.dedent("""\
    set -euo pipefail
    export PIP_ROOT_USER_ACTION=ignore

    # ── System packages ──────────────────────────────────────────────────────
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y git ninja-build build-essential python3-dev curl ca-certificates -qq

    # ── Git identity ─────────────────────────────────────────────────────────
    [ -n "${GIT_USER_NAME:-}"  ] && git config --global user.name  "$GIT_USER_NAME"
    [ -n "${GIT_USER_EMAIL:-}" ] && git config --global user.email "$GIT_USER_EMAIL"

    # ── Clone or update repo ─────────────────────────────────────────────────
    cd /workspace
    NANOCHAT_BRANCH="${NANOCHAT_BRANCH:-gated-recursive}"
    if [ -d nanochat/.git ]; then
      git -C nanochat fetch --all
      git -C nanochat checkout "$NANOCHAT_BRANCH"
      git -C nanochat pull --ff-only
    else
      if [ -n "${GITHUB_PAT:-}" ]; then
        git clone "https://${GITHUB_PAT}@github.com/TrelisResearch/nanochat.git"
      else
        git clone https://github.com/TrelisResearch/nanochat.git
      fi
      git -C nanochat checkout "$NANOCHAT_BRANCH"
    fi
    cd nanochat

    # Install dependencies
    pip install -e "." --quiet

    # Pull base checkpoint + tokenizer from nanochat-recursive HF repo
    python -m scripts.pull_from_hf \\
      --repo-id Trelis/nanochat-recursive \\
      --repo-path base/d20 \\
      --stage base \\
      --target-tag d20

    python -m scripts.pull_from_hf \\
      --repo-id Trelis/nanochat-recursive \\
      --repo-path tokenizer/latest \\
      --dest-dir "${NANOCHAT_BASE_DIR:-/root/.cache/nanochat}/tokenizer"

    # Mid-training with gated loss
    # device_batch_size=16: gated model stores extra tensors for backward through gated update
    # (s_old + u-s per step x4 recurrences ≈ +1.7 GB vs recursive at same batch).
    # Halving from 32→16 frees several GB; grad_accum=2 keeps total_batch_size at 524288 tokens.
    torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \\
      --run=gated-recursive-mid \\
      --lambda_gate=1e-3 \\
      --gate_warmup_ratio=0.2 \\
      --device_batch_size=16

    # SFT with gated loss
    torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \\
      --run=gated-recursive-sft \\
      --source=mid \\
      --lambda_gate=1e-3 \\
      --gate_warmup_ratio=0.2

    # Push to HF
    python -m scripts.push_to_hf \\
      --stage sft \\
      --repo-id Trelis/nanochat-gated-recursive \\
      --path-in-repo sft/d20
""")


def create_pod(api_key: str, config: dict, dry_run: bool = False):
    import urllib.request
    url = "https://rest.runpod.io/v1/pods"
    body = json.dumps(config).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    if dry_run:
        print("DRY RUN — would POST to:", url)
        print(json.dumps(config, indent=2))
        return {}
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main():
    load_env()
    sys.path.insert(0, str(Path(__file__).parent))
    import pod_config as cfg

    parser = argparse.ArgumentParser(description="Launch gated-recursive mid+SFT on RunPod")
    parser.add_argument("--name",    default="nanochat-gated-mid-sft")
    parser.add_argument("--branch",  default="gated-recursive")
    parser.add_argument("--gpus",    type=int, default=cfg.GPU_COUNT)
    parser.add_argument("--image",   default=cfg.IMAGE)
    parser.add_argument("--disk",    type=int, default=cfg.CONTAINER_DISK_GB)
    parser.add_argument("--volume",  type=int, default=cfg.VOLUME_GB)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("RUNPOD_API_KEY not set.")

    def fwd(key):
        v = os.environ.get(key, "")
        return {key: v} if v else {}

    env = {"NANOCHAT_BRANCH": args.branch}
    for k in ["WANDB_API_KEY", "HUGGING_FACE_HUB_TOKEN", "GITHUB_PAT",
              "GIT_USER_NAME", "GIT_USER_EMAIL", "HF_HUB_ENABLE_HF_TRANSFER"]:
        env.update(fwd(k))

    pod_config = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": cfg.GPU_TYPE_IDS,
        "gpuCount": args.gpus,
        "cloudType": cfg.CLOUD_TYPE,
        "containerDiskInGb": args.disk,
        "volumeInGb": args.volume,
        "volumeMountPath": cfg.VOLUME_MOUNT_PATH,
        "ports": cfg.PORTS,
        "env": env,
        "dockerStartCmd": ["bash", "-c", TRAIN_CMD],
    }

    print(f"Launching '{args.name}': mid+SFT on {args.gpus}×H100, branch={args.branch}")
    result = create_pod(api_key, pod_config, dry_run=args.dry_run)
    if result:
        print(f"Pod created: id={result.get('id')}  cost=${result.get('costPerHr')}/hr")
        print(f"Check status: uv run runpod/pod_status.py {result.get('id')}")


if __name__ == "__main__":
    main()
