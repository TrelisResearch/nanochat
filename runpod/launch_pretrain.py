"""
Launch gated-recursive continued pre-training on RunPod.

Use this AFTER mid+SFT is working. Pre-training downloads ~24GB of data
(240 shards × ~100MB each) before training starts.

Workflow on the pod:
  1. Pull nanochat-recursive base checkpoint from HF
  2. Download training data (~24GB)
  3. Run base_train.py with load_pretrained pointing at recursive checkpoint
     and gated loss (lambda schedule: 0→lambda_gate over training)
  4. Then kick off mid_train + chat_sft
  5. Push all stages to HF

Target budget: ~20-30% of original nanochat training tokens (~$20-30 on 8×H100)
This is set via --target_param_data_ratio=5 (vs default Chinchilla=20).

Usage:
  uv run runpod/launch_pretrain.py --dry-run
  uv run runpod/launch_pretrain.py --name gated-pretrain
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


TRAIN_CMD = textwrap.dedent("""\
    set -euo pipefail
    /usr/sbin/sshd  # start SSH daemon so we can monitor remotely
    exec > >(tee /tmp/train.log) 2>&1
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

    pip install -e "." --quiet

    # Download identity conversations
    IDENTITY_DIR="${NANOCHAT_BASE_DIR:-/root/.cache/nanochat}"
    mkdir -p "$IDENTITY_DIR"
    if [ ! -f "$IDENTITY_DIR/identity_conversations.jsonl" ]; then
      curl -L -o "$IDENTITY_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    fi

    # Pull tokenizer from nanochat-recursive HF repo
    python -m scripts.pull_from_hf \\
      --repo-id Trelis/nanochat-recursive \\
      --repo-path tokenizer/latest \\
      --dest-dir "${NANOCHAT_BASE_DIR:-/root/.cache/nanochat}/tokenizer"

    # Download pre-training data shards (240 shards × ~100MB ≈ 24GB)
    python -m nanochat.dataset -n 8
    python -m nanochat.dataset -n 240 &
    DATASET_DOWNLOAD_PID=$!
    echo "Waiting for dataset download to complete..."
    wait $DATASET_DOWNLOAD_PID

    # Pre-training from scratch with gated loss.
    # Gates co-adapt with representations from the start (no pre-training mismatch).
    # target_param_data_ratio=5 gives ~20% of Chinchilla budget.
    torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \\
      --run=gated-recursive-pretrain \\
      --lambda_gate=1e-3 \\
      --gate_warmup_ratio=0.2 \\
      --target_param_data_ratio=5 \\
      --warmdown_ratio=0.3

    # Push pre-trained gated model
    python -m scripts.push_to_hf \\
      --stage base \\
      --repo-id Trelis/nanochat-gated-recursive \\
      --path-in-repo base/d20

    # Mid-training (device_batch_size=32: gradient checkpointing eliminates the v6 OOM)
    torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \\
      --run=gated-recursive-mid \\
      --lambda_gate=1e-3 \\
      --gate_warmup_ratio=0.2 \\
      --device_batch_size=32

    # SFT
    torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \\
      --run=gated-recursive-sft \\
      --source=mid \\
      --lambda_gate=1e-3 \\
      --gate_warmup_ratio=0.2

    # Push all
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

    parser = argparse.ArgumentParser(description="Launch gated-recursive continued pre-training")
    parser.add_argument("--name",    default="nanochat-gated-pretrain")
    parser.add_argument("--branch",  default="gated-recursive")
    parser.add_argument("--gpus",    type=int, default=cfg.GPU_COUNT)
    parser.add_argument("--image",   default=cfg.IMAGE)
    parser.add_argument("--disk",    type=int, default=cfg.CONTAINER_DISK_GB)
    parser.add_argument("--volume",  type=int, default=1000)   # larger for pre-training data
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
              "GIT_USER_NAME", "GIT_USER_EMAIL", "HF_HUB_ENABLE_HF_TRANSFER",
              "NANOCHAT_BASE_DIR"]:
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
        "supportPublicIp": True,
        "env": env,
        "dockerStartCmd": ["bash", "-c", TRAIN_CMD],
    }

    print(f"Launching '{args.name}': continued pre-train+mid+SFT on {args.gpus}×H100")
    print(f"NOTE: Data download will take ~1-2hrs before training starts.")
    result = create_pod(api_key, pod_config, dry_run=args.dry_run)
    if result:
        print(f"Pod created: id={result.get('id')}  cost=${result.get('costPerHr')}/hr")
        print(f"Check status: uv run runpod/pod_status.py {result.get('id')}")


if __name__ == "__main__":
    main()
