"""
Modal runner for scripts/chat_sft_depth.py (batched single-forward with depth_mask).

Same volume / HF source as modal_train_split_sft.py; just invokes the faster
training script.

Usage:
  uv run --with modal modal run --env=dev-ronan modal_train_depth_sft.py --num-iterations 1000
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent

app = modal.App("nanochat-depth-sft")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install(
        "torch==2.8.0",
        "datasets>=4.0.0",
        "huggingface_hub>=0.24",
        "hf-transfer>=0.1.9",
        "tiktoken>=0.11.0",
        "tokenizers>=0.22.0",
        "regex>=2025.9.1",
        "requests>=2.32.5",
        "psutil>=7.1.0",
        "filelock",
        "wandb>=0.21",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/opt/nanochat",
        ignore=[
            "**/__pycache__", "**/*.pyc", ".git/**", ".venv/**",
            "tmp/**", "bench_results/**", "dev/**", "**/.pytest_cache/**",
        ],
    )
)

vol = modal.Volume.from_name("nanochat-models", create_if_missing=True)


def _ensure_source_checkpoint(nanochat_dir, repo_id, sft_subpath, tokenizer_subpath):
    from huggingface_hub import snapshot_download
    sft_tag = sft_subpath.split("/")[-1]
    sft_dst = Path(nanochat_dir) / "chatsft_checkpoints" / sft_tag
    tok_dst = Path(nanochat_dir) / "tokenizer"
    if sft_dst.exists() and any(sft_dst.glob("model_*.pt")) and (tok_dst / "tokenizer.pkl").exists():
        return
    tmp = Path(nanochat_dir) / "_hf_tmp_depth"
    tmp.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{sft_subpath}/**", f"{tokenizer_subpath}/**"],
        local_dir=str(tmp),
    )
    import shutil
    if not (sft_dst.exists() and any(sft_dst.glob("model_*.pt"))):
        sft_src = tmp / sft_subpath
        sft_dst.parent.mkdir(parents=True, exist_ok=True)
        if sft_dst.exists():
            shutil.rmtree(sft_dst)
        sft_src.rename(sft_dst)
    if not (tok_dst / "tokenizer.pkl").exists():
        tok_src = tmp / tokenizer_subpath
        tok_dst.mkdir(parents=True, exist_ok=True)
        for p in tok_src.iterdir():
            target = tok_dst / p.name
            if target.exists():
                target.unlink()
            p.rename(target)
    shutil.rmtree(tmp, ignore_errors=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 8,
    volumes={"/cache": vol},
)
def run_train(
    num_iterations: int = 1000,
    device_batch_size: int = 16,
    grad_accum_steps: int = 2,
    num_recur_prefill: int = 4,
    num_recur_decode: int = 1,
    max_seq_len: int = 1024,
    eval_every: int = 100,
    eval_batches: int = 8,
    init_lr_frac: float = 0.02,
    split_prob: float = 1.0,
    out_tag: str = "d20_depth",
    repo_id: str = "Trelis/nanochat-recursive",
    sft_subpath: str = "sft/d20",
    tokenizer_subpath: str = "tokenizer/latest",
    source_tag: str = "d20",
):
    import subprocess
    import sys

    nanochat_base = "/cache/nanochat"
    os.environ["NANOCHAT_BASE_DIR"] = nanochat_base
    Path(nanochat_base).mkdir(parents=True, exist_ok=True)
    _ensure_source_checkpoint(nanochat_base, repo_id, sft_subpath, tokenizer_subpath)
    vol.commit()

    cmd = [
        sys.executable, "-m", "scripts.chat_sft_depth",
        "-i", "sft",
        "-g", source_tag,
        "--num-iterations", str(num_iterations),
        "--device-batch-size", str(device_batch_size),
        "--grad-accum-steps", str(grad_accum_steps),
        "--num-recur-prefill", str(num_recur_prefill),
        "--num-recur-decode", str(num_recur_decode),
        "--max-seq-len", str(max_seq_len),
        "--eval-every", str(eval_every),
        "--eval-batches", str(eval_batches),
        "--init-lr-frac", str(init_lr_frac),
        "--split-prob", str(split_prob),
        "--out-tag", out_tag,
    ]
    print("[train] " + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/nanochat"
    proc = subprocess.run(cmd, cwd="/opt/nanochat", env=env)
    vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"chat_sft_depth exited with {proc.returncode}")

    ckpt_dir = f"{nanochat_base}/chatsft_checkpoints/{out_tag}"
    files = [f.name for f in Path(ckpt_dir).iterdir() if f.is_file()]
    return {"ckpt_dir_remote": ckpt_dir, "files": files}


@app.local_entrypoint()
def main(
    num_iterations: int = 1000,
    device_batch_size: int = 16,
    grad_accum_steps: int = 2,
    num_recur_prefill: int = 4,
    num_recur_decode: int = 1,
    max_seq_len: int = 1024,
    eval_every: int = 100,
    eval_batches: int = 8,
    init_lr_frac: float = 0.02,
    split_prob: float = 1.0,
    out_tag: str = "d20_depth",
):
    print(f"[local] launching depth-mask SFT on Modal dev-ronan "
          f"(iters={num_iterations}, bsz={device_batch_size}*{grad_accum_steps})")
    result = run_train.remote(
        num_iterations=num_iterations,
        device_batch_size=device_batch_size,
        grad_accum_steps=grad_accum_steps,
        num_recur_prefill=num_recur_prefill,
        num_recur_decode=num_recur_decode,
        max_seq_len=max_seq_len,
        eval_every=eval_every,
        eval_batches=eval_batches,
        init_lr_frac=init_lr_frac,
        split_prob=split_prob,
        out_tag=out_tag,
    )
    print("[local] done:", result)
