"""
Shared pod configuration for nanochat training runs.
Edit this file to change defaults; override per-script as needed.
"""

# ── Image ────────────────────────────────────────────────────────────────────
# RunPod official PyTorch image with CUDA 12.1
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# ── GPU ──────────────────────────────────────────────────────────────────────
GPU_TYPE_IDS   = ["NVIDIA H100 80GB HBM3"]   # SXM H100 preferred
GPU_TYPE_IDS_FALLBACK = [                      # fall back if SXM unavailable
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
    "NVIDIA A100 80GB PCIe",
]
GPU_COUNT      = 8
CLOUD_TYPE     = "SECURE"                      # datacenter-grade GPUs

# ── Storage ──────────────────────────────────────────────────────────────────
CONTAINER_DISK_GB = 100    # root disk (code, pip packages)
VOLUME_GB         = 500    # persistent /workspace volume (data, checkpoints)
VOLUME_MOUNT_PATH = "/workspace"

# ── Networking ───────────────────────────────────────────────────────────────
PORTS = "22/tcp"           # SSH only; add "8888/http" for Jupyter

# ── Repo / branch ────────────────────────────────────────────────────────────
REPO = "TrelisResearch/nanochat"
DEFAULT_BRANCH = "gated-recursive"
