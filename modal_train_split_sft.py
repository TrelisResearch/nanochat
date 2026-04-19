"""
Modal runner for scripts/chat_sft_split.py on a single H100.

Trains the recursive SFT model with mismatched prefill/decode recurrence depths
(num_recur_prefill=4, num_recur_decode=1) so that iter-1 decode queries learn to
attend to iter-4 prefill K/V. Starts from Trelis/nanochat-recursive sft/d20 and
saves the resulting checkpoint onto the shared Modal Volume for later eval.

Usage:
  # Smoke:
  uv run --with modal modal run --env=dev-ronan modal_train_split_sft.py --num-iterations 20
  # Full:
  uv run --with modal modal run --env=dev-ronan modal_train_split_sft.py --num-iterations 300
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent

app = modal.App("nanochat-split-sft")

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
            "**/__pycache__",
            "**/*.pyc",
            ".git/**",
            ".venv/**",
            "tmp/**",
            "bench_results/**",
            "dev/**",
            "**/.pytest_cache/**",
        ],
    )
)

vol = modal.Volume.from_name("nanochat-models", create_if_missing=True)


def _ensure_source_checkpoint(nanochat_dir: str, repo_id: str, sft_subpath: str,
                               tokenizer_subpath: str) -> None:
    """Download base SFT model + tokenizer if not already on the volume."""
    from huggingface_hub import snapshot_download

    sft_tag = sft_subpath.split("/")[-1]
    sft_dst = Path(nanochat_dir) / "chatsft_checkpoints" / sft_tag
    tok_dst = Path(nanochat_dir) / "tokenizer"
    sft_done = sft_dst.exists() and any(sft_dst.glob("model_*.pt"))
    tok_done = (tok_dst / "tokenizer.pkl").exists()
    if sft_done and tok_done:
        print(f"[setup] model+tokenizer already cached at {nanochat_dir}")
        return
    print(f"[setup] downloading {repo_id} -> {nanochat_dir}")
    tmp = Path(nanochat_dir) / "_hf_tmp_train"
    tmp.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{sft_subpath}/**", f"{tokenizer_subpath}/**"],
        local_dir=str(tmp),
    )
    if not sft_done:
        sft_src = tmp / sft_subpath
        sft_dst.parent.mkdir(parents=True, exist_ok=True)
        if sft_dst.exists():
            import shutil; shutil.rmtree(sft_dst)
        sft_src.rename(sft_dst)
    if not tok_done:
        tok_src = tmp / tokenizer_subpath
        tok_dst.mkdir(parents=True, exist_ok=True)
        for p in tok_src.iterdir():
            target = tok_dst / p.name
            if target.exists():
                target.unlink()
            p.rename(target)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 8,
    volumes={"/cache": vol},
)
def run_train(
    num_iterations: int = 300,
    target_examples_per_step: int = 16,
    num_recur_prefill: int = 4,
    num_recur_decode: int = 1,
    eval_every: int = 50,
    eval_steps: int = 32,
    init_lr_frac: float = 0.02,
    out_tag: str = "d20_split",
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
        sys.executable, "-m", "scripts.chat_sft_split",
        "-i", "sft",
        "-g", source_tag,
        "--num-iterations", str(num_iterations),
        "--target-examples-per-step", str(target_examples_per_step),
        "--num-recur-prefill", str(num_recur_prefill),
        "--num-recur-decode", str(num_recur_decode),
        "--eval-every", str(eval_every),
        "--eval-steps", str(eval_steps),
        "--init-lr-frac", str(init_lr_frac),
        "--out-tag", out_tag,
    ]
    print("[train] " + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/nanochat"
    proc = subprocess.run(cmd, cwd="/opt/nanochat", env=env)
    vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"chat_sft_split exited with {proc.returncode}")

    ckpt_dir = f"{nanochat_base}/chatsft_checkpoints/{out_tag}"
    files = list(Path(ckpt_dir).glob("*.pt")) + list(Path(ckpt_dir).glob("*.json"))
    print(f"[train] saved: {[f.name for f in files]}")
    return {"ckpt_dir_remote": ckpt_dir, "files": [f.name for f in files]}


@app.local_entrypoint()
def main(
    num_iterations: int = 300,
    target_examples_per_step: int = 16,
    num_recur_prefill: int = 4,
    num_recur_decode: int = 1,
    eval_every: int = 50,
    eval_steps: int = 32,
    init_lr_frac: float = 0.02,
    out_tag: str = "d20_split",
):
    print(f"[local] launching split SFT training on Modal dev-ronan "
          f"(iters={num_iterations}, prefill={num_recur_prefill}, decode={num_recur_decode})")
    result = run_train.remote(
        num_iterations=num_iterations,
        target_examples_per_step=target_examples_per_step,
        num_recur_prefill=num_recur_prefill,
        num_recur_decode=num_recur_decode,
        eval_every=eval_every,
        eval_steps=eval_steps,
        init_lr_frac=init_lr_frac,
        out_tag=out_tag,
    )
    print("[local] done:", result)
