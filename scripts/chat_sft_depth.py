"""
Batched single-GPU SFT with per-position depth schedule (prefill=4, decode=1).

Unlike chat_sft_split.py (which does one row at a time with a KVCache between
stage-1 prefill and stage-2 decode), this script batches rows and uses a single
forward per micro-step via GPT.forward's `depth_mask` kwarg. Each position
carries its own recurrence depth — prompt positions iterate num_recur_prefill
times, response/padding positions iterate num_recur_decode times. All positions
share the same *final* iteration so depth-1 response queries see depth-K prompt
K/V during attention.

This gives full batch parallelism on the GPU (5–10× faster per step than the
per-row approach), making real SFT scale feasible.

Run:
  uv run python -m scripts.chat_sft_depth -i sft -g d20 --num-iterations 1000
"""

import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type, get_base_dir, print0
from nanochat.checkpoint_manager import load_model, save_checkpoint

from tasks.common import TaskMixture
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee


def build_train_mixture():
    identity_path = os.path.join(get_base_dir(), "identity_conversations.jsonl")
    datasets = [
        ARC(subset="ARC-Easy", split="train"),
        ARC(subset="ARC-Challenge", split="train"),
        GSM8K(subset="main", split="train"),
        SmolTalk(split="train", stop=10_000),
    ]
    if os.path.exists(identity_path):
        datasets.append(CustomJSON(filepath=identity_path))
    datasets += [
        SimpleSpelling(size=300, split="train"),
        SpellingBee(size=300, split="train"),
    ]
    return TaskMixture(datasets)


def row_stream(dataset, tokenizer):
    """Yield (ids, targets, n_valid) as CPU tensors, one row at a time."""
    while True:
        for i in range(len(dataset)):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            n = len(ids)
            if n < 2:
                continue
            ids_t = torch.tensor(ids, dtype=torch.long)
            mask_t = torch.tensor(mask, dtype=torch.long)
            inputs_row = ids_t[:-1]
            targets_row = ids_t[1:].clone()
            targets_row[mask_t[1:] == 0] = -1
            n_valid = int((targets_row != -1).sum().item())
            if n_valid == 0:
                continue
            yield inputs_row, targets_row, n_valid


def collate_batch(rows, pad_id, num_recur_prefill, num_recur_decode, max_seq_len):
    """Build (inputs, targets, depth_mask) with per-position depth.
    Prompt positions (targets==-1 and before the first valid target) get
    num_recur_prefill. Everything else (response + right-padding) gets
    num_recur_decode. That matches the 2-stage semantics per-row.
    """
    B = len(rows)
    T_max = min(max(r[0].numel() for r in rows), max_seq_len)
    inputs = torch.full((B, T_max), pad_id, dtype=torch.long)
    targets = torch.full((B, T_max), -1, dtype=torch.long)
    depth_mask = torch.full((B, T_max), num_recur_decode, dtype=torch.long)
    for i, (ids, tgts, _) in enumerate(rows):
        T = min(ids.numel(), T_max)
        inputs[i, :T] = ids[:T]
        targets[i, :T] = tgts[:T]
        # find first valid (response) position within the truncated row
        valid_idxs = torch.nonzero(tgts[:T] != -1, as_tuple=False)
        first_valid = int(valid_idxs[0].item()) if valid_idxs.numel() > 0 else T
        depth_mask[i, :first_valid] = num_recur_prefill
    return inputs, targets, depth_mask


@torch.no_grad()
def run_val_loss(model, val_iter, pad_id, num_recur_prefill, num_recur_decode, device,
                 batch_size, eval_batches, max_seq_len, autocast_ctx):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    with ctx:
        for _ in range(eval_batches):
            rows = []
            for _ in range(batch_size):
                try:
                    rows.append(next(val_iter))
                except StopIteration:
                    break
            if not rows:
                break
            inputs, targets, depth_mask = collate_batch(
                rows, pad_id, num_recur_prefill, num_recur_decode, max_seq_len,
            )
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            depth_mask = depth_mask.to(device, non_blocking=True)
            logits, _ = model.forward(inputs, depth_mask=depth_mask)
            per_tok = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction="sum",
            )
            n = (targets != -1).sum().item()
            if torch.isfinite(per_tok) and n > 0:
                total_loss += per_tok.item()
                total_tokens += n
    model.train()
    return total_loss / max(1, total_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--source", default="sft", choices=["base", "mid", "sft", "rl"])
    ap.add_argument("-g", "--model-tag", default=None)
    ap.add_argument("-s", "--step", type=int, default=None)
    ap.add_argument("--num-iterations", type=int, default=1000)
    ap.add_argument("--device-batch-size", type=int, default=16,
                    help="Rows forwarded in a single call. Scale down if OOM, up if GPU idle.")
    ap.add_argument("--grad-accum-steps", type=int, default=2,
                    help="Effective examples per optimizer step = device_batch_size × grad_accum_steps.")
    ap.add_argument("--num-recur-prefill", type=int, default=4)
    ap.add_argument("--num-recur-decode", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=1024,
                    help="Truncate rows longer than this to cap T_max.")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--unembedding-lr", type=float, default=0.004)
    ap.add_argument("--embedding-lr", type=float, default=0.2)
    ap.add_argument("--matrix-lr", type=float, default=0.02)
    ap.add_argument("--init-lr-frac", type=float, default=0.02,
                    help="Matches chat_sft.py default. Lower if training is unstable.")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--save-step", type=int, default=None)
    args = ap.parse_args()

    device_type = autodetect_device_type()
    device = torch.device("cuda" if device_type == "cuda" else device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    print0(f"Loading {args.source} model, tag={args.model_tag}, step={args.step}")
    model, tokenizer, meta = load_model(
        args.source, device, phase="train", model_tag=args.model_tag, step=args.step,
    )
    model.train()
    pad_id = tokenizer.encode_special("<|assistant_end|>")

    train_ds = build_train_mixture()
    val_ds = SmolTalk(split="test")
    train_stream = row_stream(train_ds, tokenizer)
    make_val_stream = lambda: row_stream(val_ds, tokenizer)

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

    examples_per_step = args.device_batch_size * args.grad_accum_steps
    print0(f"device_batch_size={args.device_batch_size}, grad_accum={args.grad_accum_steps}, "
           f"examples/step={examples_per_step}, num_iterations={args.num_iterations}")
    print0(f"num_recur_prefill={args.num_recur_prefill}, num_recur_decode={args.num_recur_decode}")

    t_start = time.perf_counter()
    for step in range(args.num_iterations):
        if step % args.eval_every == 0:
            val_loss = run_val_loss(
                model, make_val_stream(), pad_id,
                args.num_recur_prefill, args.num_recur_decode, device,
                args.device_batch_size, args.eval_batches, args.max_seq_len, autocast_ctx,
            )
            print0(f"step {step:05d} | val_loss/tok = {val_loss:.4f}")

        step_loss_sum = 0.0
        step_tokens = 0
        for _ in range(args.grad_accum_steps):
            rows = [next(train_stream) for _ in range(args.device_batch_size)]
            inputs, targets, depth_mask = collate_batch(
                rows, pad_id, args.num_recur_prefill, args.num_recur_decode, args.max_seq_len,
            )
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            depth_mask = depth_mask.to(device, non_blocking=True)
            with autocast_ctx:
                logits, _ = model.forward(inputs, depth_mask=depth_mask)
                loss_sum = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1),
                    ignore_index=-1, reduction="sum",
                )
            n_valid = (targets != -1).sum().item()
            if n_valid == 0 or not torch.isfinite(loss_sum):
                continue
            # mean CE across active tokens, scaled by 1/grad_accum so the accumulated grad
            # matches the expectation of a single-shot forward at batch = examples_per_step.
            (loss_sum / n_valid / args.grad_accum_steps).backward()
            step_loss_sum += loss_sum.item()
            step_tokens += n_valid

        lrm = lr_mult(step)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * lrm

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            grad_norm = None
        any_nan = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if any_nan:
            print0(f"step {step:05d} | WARNING: non-finite gradient; skipping opt step")
        else:
            for opt in optimizers:
                opt.step()
        model.zero_grad(set_to_none=True)

        if step % 10 == 0 or step == args.num_iterations - 1:
            elapsed = time.perf_counter() - t_start
            ppt = step_loss_sum / max(1, step_tokens)
            gn = f" gnorm={grad_norm.item():.2f}" if grad_norm is not None else ""
            print0(f"step {step:05d}/{args.num_iterations} | "
                   f"train_loss/tok={ppt:.4f} | lrm={lrm:.4f} | tokens={step_tokens}{gn} | {elapsed:.0f}s")

    val_loss = run_val_loss(
        model, make_val_stream(), pad_id,
        args.num_recur_prefill, args.num_recur_decode, device,
        args.device_batch_size, args.eval_batches, args.max_seq_len, autocast_ctx,
    )
    print0(f"Final val_loss/tok = {val_loss:.4f}")

    base_dir = get_base_dir()
    src_tag = args.model_tag or f"d{model.config.n_layer}"
    out_tag = args.out_tag or (src_tag + "_depth")
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
            "regime": "depth_mask_single_forward",
            "model_config": {k: v for k, v in model.config.__dict__.items() if not k.startswith("_")},
        },
    )
    print0(f"Saved checkpoint to {ckpt_dir}/model_{save_step:06d}.pt")


if __name__ == "__main__":
    main()
