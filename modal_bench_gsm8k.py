"""
Modal runner for scripts/bench_gsm8k.py on a single H100.

Runs the recurrence benchmark on Trelis/nanochat-recursive (sft/d20) and writes
the resulting JSON + summary back to local ./bench_results/.

Usage:
  uv run --with modal modal run --env=dev-ronan modal_bench_gsm8k.py
  uv run --with modal modal run --env=dev-ronan modal_bench_gsm8k.py --max-problems 32

Tuning knobs are below the `@app.function` decorator — edit args there or pass via CLI.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent

app = modal.App("nanochat-bench-gsm8k")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install(
        # Match pyproject.toml runtime deps (subset needed for eval).
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
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Mount the repo into /opt/nanochat (ignore caches & large junk).
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

# Persistent volume for model weights between runs.
vol = modal.Volume.from_name("nanochat-models", create_if_missing=True)


def _ensure_model_and_tokenizer(nanochat_dir: str, repo_id: str, sft_subpath: str,
                                 tokenizer_subpath: str) -> None:
    """Download (if missing) the sft checkpoint + tokenizer into nanochat_dir."""
    from huggingface_hub import snapshot_download

    sft_tag = sft_subpath.split("/")[-1]  # e.g. "d20"
    sft_dst = Path(nanochat_dir) / "chatsft_checkpoints" / sft_tag
    tok_dst = Path(nanochat_dir) / "tokenizer"

    sft_done = sft_dst.exists() and any(sft_dst.glob("model_*.pt"))
    tok_done = (tok_dst / "tokenizer.pkl").exists()

    if sft_done and tok_done:
        print(f"[setup] model+tokenizer already cached at {nanochat_dir}")
        return

    print(f"[setup] downloading {repo_id} artifacts -> {nanochat_dir}")
    tmp = Path(nanochat_dir) / "_hf_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{sft_subpath}/**", f"{tokenizer_subpath}/**"],
        local_dir=str(tmp),
    )
    # Move sft
    if not sft_done:
        sft_src = tmp / sft_subpath
        sft_dst.parent.mkdir(parents=True, exist_ok=True)
        if sft_dst.exists():
            import shutil; shutil.rmtree(sft_dst)
        sft_src.rename(sft_dst)
        print(f"[setup]  sft -> {sft_dst}")
    # Move tokenizer (tokenizer_subpath is e.g. "tokenizer/latest"; we want its contents
    # flat under <base>/tokenizer/)
    if not tok_done:
        tok_src = tmp / tokenizer_subpath
        tok_dst.mkdir(parents=True, exist_ok=True)
        for p in tok_src.iterdir():
            target = tok_dst / p.name
            if target.exists():
                target.unlink()
            p.rename(target)
        print(f"[setup]  tokenizer -> {tok_dst}")
    # Cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 6,  # 6 hours
    volumes={"/cache": vol},
)
def run_bench(
    max_problems: int | None = None,
    full_rs: str = "1,2,4",
    split_prefill_rs: str = "2,4",
    split_decode_rs: str = "1",
    no_full: bool = False,
    no_split: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 50,
    repo_id: str = "Trelis/nanochat-recursive",
    sft_subpath: str = "sft/d20",
    tokenizer_subpath: str = "tokenizer/latest",
    model_tag: str | None = None,
    no_warm_start: bool = False,
):
    import subprocess
    import sys

    nanochat_base = "/cache/nanochat"
    os.environ["NANOCHAT_BASE_DIR"] = nanochat_base
    Path(nanochat_base).mkdir(parents=True, exist_ok=True)

    _ensure_model_and_tokenizer(nanochat_base, repo_id, sft_subpath, tokenizer_subpath)
    vol.commit()

    # Build CLI args. If model_tag not given, fall back to the HF subpath's last segment
    # (i.e. the upstream checkpoint we pulled). Pass our own tag to bench a locally trained
    # checkpoint sitting on the same Volume.
    effective_tag = model_tag or sft_subpath.split("/")[-1]
    cmd = [
        sys.executable, "-m", "scripts.bench_gsm8k",
        "-i", "sft",
        "-g", effective_tag,
        "-m", str(max_new_tokens),
        "-t", str(temperature),
        "-k", str(top_k),
        "--full-rs", full_rs,
        "--split-prefill-rs", split_prefill_rs,
        "--split-decode-rs", split_decode_rs,
    ]
    if max_problems is not None:
        cmd += ["-x", str(max_problems)]
    if no_full:
        cmd.append("--no-full")
    if no_split:
        cmd.append("--no-split")
    if no_warm_start:
        cmd.append("--no-warm-start")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = f"{nanochat_base}/bench_gsm8k_{ts}.json"
    cmd += ["--out", out_json]

    print("[run] " + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/nanochat"
    # Stream stdout/stderr live.
    proc = subprocess.run(cmd, cwd="/opt/nanochat", env=env)
    vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"bench_gsm8k exited with {proc.returncode}")

    # Read results back and return as python dict so caller can save locally.
    import json
    with open(out_json) as f:
        payload = json.load(f)
    return {"json_path_remote": out_json, "payload": payload}


@app.local_entrypoint()
def main(
    max_problems: int = 0,
    full_rs: str = "1,2,4",
    split_prefill_rs: str = "2,4",
    split_decode_rs: str = "1",
    no_full: bool = False,
    no_split: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 50,
    out_dir: str = "bench_results",
    model_tag: str = "",
    no_warm_start: bool = False,
):
    mp = None if max_problems <= 0 else max_problems
    tag = model_tag.strip() or None
    print(f"[local] launching on Modal (dev-ronan) with max_problems={mp} model_tag={tag}")
    result = run_bench.remote(
        max_problems=mp,
        full_rs=full_rs,
        split_prefill_rs=split_prefill_rs,
        split_decode_rs=split_decode_rs,
        no_full=no_full,
        no_split=no_split,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        model_tag=tag,
        no_warm_start=no_warm_start,
    )
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    local_path = out_dir_p / f"bench_gsm8k_{ts}.json"
    import json
    with open(local_path, "w") as f:
        json.dump(result["payload"], f, indent=2, default=str)
    print(f"[local] wrote {local_path}")
    # Print summary
    print("\n" + "=" * 78)
    print(f"{'config':<24} {'acc':>8} {'N':>6} {'time(s)':>10} {'s/prob':>8}")
    print("-" * 78)
    for r in result["payload"]["results"]:
        c = r["config"]
        print(f"{c['name']:<24} {100*r['accuracy']:>7.2f}% {r['num_problems']:>6} "
              f"{r['elapsed_sec']:>10.1f} {r['sec_per_problem']:>8.2f}")
    print("=" * 78)
