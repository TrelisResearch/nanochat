import torch


def sample_diffusion_mask(eligible_mask):
    """
    Sample a random mask for masked diffusion training.
    eligible_mask: bool tensor (B, T) indicating which positions can be masked.
    Returns:
        mask: bool tensor (B, T)
        seq_lens: float tensor (B,) giving the number of eligible tokens per row (used for scaling)
    """
    assert eligible_mask.dtype == torch.bool
    B, T = eligible_mask.shape
    device = eligible_mask.device
    mask = torch.zeros_like(eligible_mask)
    seq_lens = eligible_mask.sum(dim=1).clamp(min=1).to(dtype=torch.float32)
    for b in range(B):
        valid_idx = torch.nonzero(eligible_mask[b], as_tuple=False).squeeze(-1)
        n_valid = valid_idx.numel()
        if n_valid == 0:
            continue
        num_mask = torch.randint(1, n_valid + 1, (1,), device=device).item()
        perm = torch.randperm(n_valid, device=device)[:num_mask]
        chosen = valid_idx[perm]
        mask[b, chosen] = True
    return mask, seq_lens


def apply_diffusion_mask(tokens, mask, mask_token_id):
    """
    Replace masked positions in `tokens` with the mask token.
    """
    assert tokens.shape == mask.shape
    corrupted = tokens.clone()
    corrupted[mask] = mask_token_id
    return corrupted
