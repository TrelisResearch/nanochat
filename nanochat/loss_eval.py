"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanochat.diffusion import sample_diffusion_mask, apply_diffusion_mask

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes, mask_token_id):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    device = model.get_device()
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0.0, dtype=torch.float32, device=device)
    batch_iter = iter(batches)
    token_bytes = token_bytes.to(device)
    for _ in range(steps):
        x, _ = next(batch_iter)
        eligible_mask = torch.ones_like(x, dtype=torch.bool)
        loss_mask, seq_lens = sample_diffusion_mask(eligible_mask)
        mask_counts = loss_mask.sum(dim=1).clamp(min=1)
        corrupted = apply_diffusion_mask(x, loss_mask, mask_token_id)
        logits = model(corrupted)
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            x.view(-1),
            reduction='none'
        ).view_as(x)
        per_token_bytes = token_bytes[x]
        masked_loss = ce * loss_mask
        masked_bytes = per_token_bytes * loss_mask
        scale = seq_lens / mask_counts.to(dtype=torch.float32)
        total_nats += (masked_loss.sum(dim=1) * scale).sum()
        scaled_bytes = masked_bytes.sum(dim=1).to(torch.float32) * scale
        total_bytes += scaled_bytes.sum()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
