"""
Single-GPU SFT with split prefill/decode recurrence depths.

For each training example, the prompt portion is processed with `num_recur_prefill`
(default 4) and the final recur K/V is left in the cache ("last"). The response
portion is then forwarded at `num_recur_decode` (default 1), reading the deep-prefill
K/V during attention. Loss is computed on response tokens only (as in standard SFT
via the -1 ignore_index), which naturally aligns with the prompt/response split.

The hope: teach the model to produce K/V at iter-4 (prefill) that iter-1 decode
queries can actually use — fixing the mismatch that made the `split_pK_d1_last`
bench configs collapse to r=1 baseline.

Run:
  uv run python -m scripts.chat_sft_split -i sft -g d20 --num-iterations 200
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type, get_base_dir, print0
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.engine import KVCache

from tasks.common import TaskMixture
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee


def build_train_mixture():
    """Mirror chat_sft.py's training mixture."""
    identity_path = os.path.join(get_base_dir(), "identity_conversations.jsonl")
    datasets = [
        ARC(subset="ARC-Easy", split="train"),
        ARC(subset="ARC-Challenge", split="train"),
        GSM8K(subset="main", split="train"),
        SmolTalk(split="train", stop=10_000),
    ]
    # Identity conversations file may not exist in fresh setups — skip silently.
    if os.path.exists(identity_path):
        datasets.append(CustomJSON(filepath=identity_path))
    datasets += [
        SimpleSpelling(size=300, split="train"),
        SpellingBee(size=300, split="train"),
    ]
    return TaskMixture(datasets)


def row_iter(dataset, tokenizer, device):
    """Yield (inputs_row, targets_row, n_valid) one sequence at a time."""
    pad_id = tokenizer.encode_special("<|assistant_end|>")
    while True:
        for i in range(len(dataset)):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            n = len(ids)
            if n < 2:
                continue
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            mask_t = torch.tensor(mask, dtype=torch.long, device=device)
            inputs_row = ids_t[:-1]
            targets_row = ids_t[1:].clone()
            # mask is aligned with ids; mask[1:] corresponds to targets (what we predict).
            targets_row[mask_t[1:] == 0] = -1
            n_valid = (targets_row != -1).sum().item()
            if n_valid == 0:
                continue
            yield inputs_row, targets_row, n_valid


def compute_row_loss(model, inputs_row, targets_row, num_recur_prefill, num_recur_decode):
    """Two-stage forward on a single row. Returns (loss, n_valid) or (None, 0)."""
    T = inputs_row.size(0)
    valid_mask = targets_row != -1
    first_valid_idx = int(torch.nonzero(valid_mask, as_tuple=False)[0].item())
    P = first_valid_idx  # positions [0, P) = prompt, [P, T) = response
    if P == 0:
        # Edge case: no prompt tokens (shouldn't happen in chat data) — just do one-stage.
        prefix_ids = inputs_row[:0].unsqueeze(0)
        suffix_ids = inputs_row.unsqueeze(0)
        suffix_targets = targets_row.unsqueeze(0)
    elif P >= T:
        # Whole row is prompt — no supervised positions. Skip.
        return None, 0
    else:
        prefix_ids = inputs_row[:P].unsqueeze(0)
        suffix_ids = inputs_row[P:].unsqueeze(0)
        suffix_targets = targets_row[P:].unsqueeze(0)
    S = suffix_ids.size(1)
    device = inputs_row.device

    m = model.config
    num_layers = m.n_prelude + m.n_recur_block + m.n_coda
    cache = KVCache(
        batch_size=1, num_heads=m.n_kv_head, seq_len=T + 4,
        head_dim=m.n_embd // m.n_head, num_layers=num_layers,
    )

    # Stage 1: prefill
    if P > 0:
        _, warm = model.forward(
            prefix_ids, kv_cache=cache,
            num_recur=num_recur_prefill, prefill_kv_keep="last",
        )
        warm_last = warm[:, -1:, :]
    else:
        warm_last = None

    # Stage 2: decode (response)
    logits, _ = model.forward(
        suffix_ids, kv_cache=cache,
        num_recur=num_recur_decode, warm_start_state=warm_last,
    )
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        suffix_targets.view(-1),
        ignore_index=-1,
        reduction="sum",
    )
    n_valid = int((suffix_targets != -1).sum().item())
    return loss, n_valid


def run_val_loss(model, val_iter_fn, tokenizer, device, eval_steps, num_recur_prefill, num_recur_decode):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    it = val_iter_fn()
    with torch.no_grad():
        for _ in range(eval_steps):
            try:
                inputs_row, targets_row, _ = next(it)
            except StopIteration:
                break
            res = compute_row_loss(model, inputs_row, targets_row, num_recur_prefill, num_recur_decode)
            if res[0] is None:
                continue
            total_loss += res[0].item()
            total_tokens += res[1]
    model.train()
    return total_loss / max(1, total_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--source", default="sft", choices=["base", "mid", "sft", "rl"])
    ap.add_argument("-g", "--model-tag", default=None)
    ap.add_argument("-s", "--step", type=int, default=None)
    ap.add_argument("--num-iterations", type=int, default=300,
                    help="Training steps (each = target_examples_per_step examples)")
    ap.add_argument("--target-examples-per-step", type=int, default=16)
    ap.add_argument("--num-recur-prefill", type=int, default=4)
    ap.add_argument("--num-recur-decode", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=32)
    ap.add_argument("--unembedding-lr", type=float, default=0.004)
    ap.add_argument("--embedding-lr", type=float, default=0.2)
    ap.add_argument("--matrix-lr", type=float, default=0.02)
    ap.add_argument("--init-lr-frac", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out-tag", default=None,
                    help="Tag for the saved chatsft_checkpoints/<tag> dir; default = current tag + '_split'")
    ap.add_argument("--save-step", type=int, default=None,
                    help="Step number to save under; default = num_iterations")
    args = ap.parse_args()

    device_type = autodetect_device_type()
    if device_type == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    print0(f"Loading {args.source} model, tag={args.model_tag}, step={args.step}")
    model, tokenizer, meta = load_model(args.source, device, phase="train",
                                         model_tag=args.model_tag, step=args.step)
    model.train()

    # Datasets
    train_ds = build_train_mixture()
    val_ds = SmolTalk(split="test")
    train_iter = row_iter(train_ds, tokenizer, device)
    make_val_iter = lambda: row_iter(val_ds, tokenizer, device)

    # Optimizer
    optimizers = model.setup_optimizers(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = g["lr"] * args.init_lr_frac
            g["initial_lr"] = g["lr"]

    def lr_mult(it):
        return 1.0 - it / args.num_iterations

    # Training loop
    t_start = time.perf_counter()
    for step in range(args.num_iterations):
        # Eval
        if step % args.eval_every == 0:
            val_loss = run_val_loss(model, make_val_iter, tokenizer, device,
                                    args.eval_steps, args.num_recur_prefill, args.num_recur_decode)
            print0(f"step {step:05d} | val_loss (per-token) = {val_loss:.4f}")

        # Accumulate grads across target_examples_per_step rows
        total_loss_val = 0.0
        total_tokens = 0
        for r in range(args.target_examples_per_step):
            inputs_row, targets_row, _ = next(train_iter)
            with autocast_ctx:
                loss, n_valid = compute_row_loss(
                    model, inputs_row, targets_row,
                    args.num_recur_prefill, args.num_recur_decode,
                )
            if loss is None:
                continue
            # Scale by 1/target_examples so the per-step grad is roughly equivalent
            # to batching over target_examples with mean reduction (per-row sum / total N).
            # Using n_valid weighting keeps long-response rows from dominating? We use
            # simple uniform row-averaging to mirror typical SFT (loss = mean over rows of
            # per-row mean loss); in practice n_valid is similar enough.
            (loss / (n_valid * args.target_examples_per_step)).backward()
            total_loss_val += loss.item()
            total_tokens += n_valid

        # LR
        lrm = lr_mult(step)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * lrm

        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        if step % 10 == 0 or step == args.num_iterations - 1:
            elapsed = time.perf_counter() - t_start
            ppt = total_loss_val / max(1, total_tokens)
            print0(f"step {step:05d}/{args.num_iterations} | train_loss/tok={ppt:.4f} | lrm={lrm:.4f} | "
                   f"tokens={total_tokens} | {elapsed:.0f}s")

    # Final eval + save
    val_loss = run_val_loss(model, make_val_iter, tokenizer, device,
                             args.eval_steps, args.num_recur_prefill, args.num_recur_decode)
    print0(f"Final val_loss={val_loss:.4f}")

    base_dir = get_base_dir()
    src_tag = args.model_tag or f"d{model.config.n_layer}"
    out_tag = args.out_tag or (src_tag + "_split")
    ckpt_dir = os.path.join(base_dir, "chatsft_checkpoints", out_tag)
    save_step = args.save_step if args.save_step is not None else args.num_iterations
    save_checkpoint(
        ckpt_dir, save_step, model.state_dict(), None,
        {
            "step": save_step,
            "val_loss": val_loss,
            "num_recur_prefill": args.num_recur_prefill,
            "num_recur_decode": args.num_recur_decode,
            "source_tag": src_tag,
            "model_config": {k: v for k, v in model.config.__dict__.items() if not k.startswith("_")},
        },
    )
    print0(f"Saved checkpoint to {ckpt_dir}/model_{save_step:06d}.pt")


if __name__ == "__main__":
    main()
