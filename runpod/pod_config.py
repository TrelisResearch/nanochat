"""
Shared pod configuration for nanochat training runs.
Edit this file to change defaults; override per-script as needed.
"""

# ── Image ────────────────────────────────────────────────────────────────────
# RunPod official PyTorch image with CUDA 12.1
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# ── GPU ──────────────────────────────────────────────────────────────────────
# Ordered by preference; launch scripts try each in turn until one is available.
# H100 SXM preferred; A100 SXM4 80GB is the reliable fallback (same BF16/CUDA support).
GPU_TYPE_IDS_PRIORITY = [
    "NVIDIA H100 80GB HBM3",      # H100 SXM 80GB   ~$28/hr for 8×
    "NVIDIA A100-SXM4-80GB",      # A100 SXM4 80GB  ~$12/hr for 8× (tested available)
    "NVIDIA H100 PCIe",           # H100 PCIe 80GB  fallback
]
GPU_TYPE_IDS   = GPU_TYPE_IDS_PRIORITY   # default: try all in order
GPU_COUNT      = 8
CLOUD_TYPE     = "SECURE"                # datacenter-grade GPUs

# ── Storage ──────────────────────────────────────────────────────────────────
CONTAINER_DISK_GB = 100    # root disk (code, pip packages)
VOLUME_GB         = 500    # persistent /workspace volume (data, checkpoints)
VOLUME_MOUNT_PATH = "/workspace"

# ── Networking ───────────────────────────────────────────────────────────────
PORTS = ["22/tcp"]          # SSH only; add "8888/http" for Jupyter

# ── Repo / branch ────────────────────────────────────────────────────────────
REPO = "TrelisResearch/nanochat"
DEFAULT_BRANCH = "gated-recursive"
