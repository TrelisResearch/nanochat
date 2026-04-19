"""
Single-GPU GSM8K benchmark for the recursive transformer, sweeping recurrence
counts independently for prefill and decode.

Matrix:
  - Full-recur: num_recur_prefill == num_recur_decode == r   for r in {1,2,4}
  - Split: num_recur_prefill in {2,4}, num_recur_decode == 1, kv_keep in {"last","first"}

For each config we record: accuracy (% solved), total wall-clock (sec), sec/problem.

Example:
  uv run python -m scripts.bench_gsm8k -i sft --max-problems 32
"""

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from nanochat.common import autodetect_device_type, get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

from tasks.gsm8k import GSM8K


def build_config_matrix(include_full, include_split, full_rs, split_prefill_rs):
    configs = []
    if include_full:
        for r in full_rs:
            configs.append({
                "name": f"full_r{r}",
                "prefill": r,
                "decode": r,
                "kv_keep": "last",  # irrelevant when prefill==decode
            })
    if include_split:
        for r in split_prefill_rs:
            if r <= 1:
                continue  # split only interesting for prefill > decode
            for keep in ("last", "first"):
                configs.append({
                    "name": f"split_p{r}_d1_{keep}",
                    "prefill": r,
                    "decode": 1,
                    "kv_keep": keep,
                })
    return configs


def run_one_config(task, tokenizer, engine, *, prefill, decode, kv_keep, max_problems,
                   max_new_tokens, temperature, top_k, use_warm_start=True):
    num_problems = min(len(task), max_problems) if max_problems is not None else len(task)
    num_passed = 0
    t_start = time.perf_counter()
    for i in range(num_problems):
        conversation = task[i]
        prompt_ids = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(
            prompt_ids,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            num_recur_prefill=prefill,
            num_recur_decode=decode,
            prefill_kv_keep=kv_keep,
            use_warm_start=use_warm_start,
        )
        prefix_len = len(prompt_ids)
        completion = tokenizer.decode(results[0][prefix_len:])
        if task.evaluate(conversation, completion):
            num_passed += 1
        if (i + 1) % 8 == 0 or (i + 1) == num_problems:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"  [{i+1}/{num_problems}] pass={num_passed} ({100*num_passed/(i+1):.1f}%) "
                  f"| {elapsed:.1f}s | {rate:.2f} prob/s", flush=True)
    elapsed = time.perf_counter() - t_start
    acc = num_passed / num_problems if num_problems > 0 else 0.0
    return {
        "num_passed": num_passed,
        "num_problems": num_problems,
        "accuracy": acc,
        "elapsed_sec": elapsed,
        "sec_per_problem": elapsed / max(1, num_problems),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--source", default="sft", choices=["base", "mid", "sft", "rl"])
    ap.add_argument("-g", "--model-tag", default=None)
    ap.add_argument("-s", "--step", type=int, default=None)
    ap.add_argument("-x", "--max-problems", type=int, default=None,
                    help="Cap on problems per config (default: full test set)")
    ap.add_argument("-m", "--max-new-tokens", type=int, default=512)
    ap.add_argument("-t", "--temperature", type=float, default=0.0)
    ap.add_argument("-k", "--top-k", type=int, default=50)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--full-rs", default="1,2,4",
                    help="Comma-separated recurrence counts for the full (prefill=decode) sweep")
    ap.add_argument("--split-prefill-rs", default="2,4",
                    help="Comma-separated prefill recurrence counts for the split (decode=1) sweep")
    ap.add_argument("--no-full", action="store_true", help="Skip full-recur configs")
    ap.add_argument("--no-split", action="store_true", help="Skip split prefill/decode configs")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="Disable warm_start_state at decode — matches depth_mask SFT regime")
    ap.add_argument("--out", default=None, help="Output JSON path (default: <base>/bench_gsm8k_<ts>.json)")
    args = ap.parse_args()

    device_type = autodetect_device_type()
    device = torch.device(device_type if device_type != "mps" else "mps")
    if device_type == "cuda":
        device = torch.device("cuda")
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    print(f"Loading model: source={args.source} tag={args.model_tag} step={args.step}")
    model, tokenizer, meta = load_model(args.source, device, phase="eval",
                                        model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params/1e6:.1f}M params, config={model.config}")

    task = GSM8K(subset="main", split="test")
    print(f"GSM8K test set: {len(task)} problems; running up to "
          f"{args.max_problems if args.max_problems is not None else len(task)} per config")

    full_rs = [int(x) for x in args.full_rs.split(",") if x.strip()]
    split_prefill_rs = [int(x) for x in args.split_prefill_rs.split(",") if x.strip()]
    configs = build_config_matrix(
        include_full=not args.no_full,
        include_split=not args.no_split,
        full_rs=full_rs,
        split_prefill_rs=split_prefill_rs,
    )
    print(f"Running {len(configs)} configs:")
    for c in configs:
        print(f"  {c['name']}: prefill={c['prefill']} decode={c['decode']} kv_keep={c['kv_keep']}")

    results = []
    for c in configs:
        print(f"\n=== {c['name']} ===", flush=True)
        with autocast_ctx:
            r = run_one_config(
                task, tokenizer, engine,
                prefill=c["prefill"], decode=c["decode"], kv_keep=c["kv_keep"],
                max_problems=args.max_problems,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                use_warm_start=not args.no_warm_start,
            )
        r["config"] = c
        results.append(r)
        print(f"  -> acc={100*r['accuracy']:.2f}% ({r['num_passed']}/{r['num_problems']}) "
              f"in {r['elapsed_sec']:.1f}s ({r['sec_per_problem']:.2f} s/prob)")

    out_path = args.out
    if out_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = str(Path(get_base_dir()) / f"bench_gsm8k_{ts}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "model": {
            "source": args.source,
            "tag": args.model_tag,
            "step": args.step,
            "n_params": n_params,
            "config": {k: v for k, v in vars(model.config).items() if not k.startswith("_")},
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote results to: {out_path}")

    # Pretty summary table
    print("\n" + "=" * 78)
    print(f"{'config':<24} {'acc':>8} {'N':>6} {'time(s)':>10} {'s/prob':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r['config']['name']:<24} {100*r['accuracy']:>7.2f}% {r['num_problems']:>6} "
              f"{r['elapsed_sec']:>10.1f} {r['sec_per_problem']:>8.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
