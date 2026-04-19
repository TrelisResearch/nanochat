"""
Tests for the `depth_mask` single-forward training path in GPT.forward.

Key invariant: for a single row with prefix depth=K and suffix depth=1, the
depth_mask single-forward should produce response logits equivalent to the
two-stage KVCache approach (prefill depth=K keep=last, then decode depth=1).
"""
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache


def _make_model(n_recur_block=2, n_prelude=1, n_coda=1, n_embd=64, n_layer=4):
    cfg = GPTConfig(
        sequence_len=128, vocab_size=256, n_layer=n_layer, n_head=4, n_kv_head=2,
        n_embd=n_embd, n_prelude=n_prelude, n_recur_block=n_recur_block, n_coda=n_coda,
        fixed_k=4, bptt_k=None,
    )
    torch.manual_seed(0)
    model = GPT(cfg)
    model.init_weights()
    # Make the recur loop non-trivial (init_weights zeros c_proj and the tail of inject).
    with torch.no_grad():
        for block in list(model.transformer.prelude) + list(model.transformer.recur) + list(model.transformer.coda):
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.1)
            torch.nn.init.normal_(block.attn.c_proj.weight, std=0.1)
        torch.nn.init.normal_(model.inject.weight, std=0.05)
        torch.nn.init.normal_(model.lm_head.weight, std=0.05)
    model.eval()  # deterministic behavior
    return model, cfg


def _two_stage_logits(model, cfg, prefix_ids, suffix_ids, num_recur_prefill, num_recur_decode):
    """Reference: run prefill at depth=K via KVCache, decode at depth=1 through it.
    No warm_start_state (matches the training recipe in chat_sft_split.py)."""
    B = prefix_ids.size(0)
    P = prefix_ids.size(1)
    S = suffix_ids.size(1)
    num_layers = cfg.n_prelude + cfg.n_recur_block + cfg.n_coda
    cache = KVCache(
        batch_size=B, num_heads=cfg.n_kv_head, seq_len=P + S + 4,
        head_dim=cfg.n_embd // cfg.n_head, num_layers=num_layers,
    )
    with torch.no_grad():
        if P > 0:
            model.forward(prefix_ids, kv_cache=cache, num_recur=num_recur_prefill, prefill_kv_keep="last")
        logits, _ = model.forward(
            suffix_ids, kv_cache=cache, num_recur=num_recur_decode, warm_start_state=None,
        )
    return logits


def _depth_mask_logits(model, prefix_ids, suffix_ids, num_recur_prefill, num_recur_decode):
    """Single-forward with per-position depth_mask. Returns the response portion of logits."""
    full_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
    B, T = full_ids.shape
    P = prefix_ids.size(1)
    depth_mask = torch.full((B, T), num_recur_decode, dtype=torch.long, device=full_ids.device)
    depth_mask[:, :P] = num_recur_prefill
    with torch.no_grad():
        logits, _ = model.forward(full_ids, depth_mask=depth_mask)
    return logits[:, P:, :]  # response-only


def test_depth_mask_matches_two_stage():
    model, cfg = _make_model()
    torch.manual_seed(42)
    B, P, S = 1, 6, 4
    prefix_ids = torch.randint(0, cfg.vocab_size, (B, P))
    suffix_ids = torch.randint(0, cfg.vocab_size, (B, S))

    ref = _two_stage_logits(model, cfg, prefix_ids, suffix_ids, num_recur_prefill=4, num_recur_decode=1)
    got = _depth_mask_logits(model, prefix_ids, suffix_ids, num_recur_prefill=4, num_recur_decode=1)
    # Numerical tolerance: tests on bf16-capable model params cast to fp32 here (no autocast in test).
    # Paths should be identical at fp32.
    assert torch.allclose(ref, got, atol=1e-5), (
        f"depth_mask and two-stage diverge: max diff={(ref - got).abs().max().item():.2e}"
    )


def test_depth_mask_grads_flow_to_all_recur_params():
    """Training usage: backward through a depth_mask forward must reach every recur param."""
    model, cfg = _make_model()
    model.train()
    torch.manual_seed(7)
    B, P, S = 1, 6, 4
    full_ids = torch.randint(0, cfg.vocab_size, (B, P + S))
    targets = torch.full((B, P + S), -1, dtype=torch.long)
    response_targets = torch.randint(0, cfg.vocab_size, (B, S))
    targets[:, P:] = response_targets

    depth_mask = torch.full((B, P + S), 1, dtype=torch.long)
    depth_mask[:, :P] = 4

    logits, _ = model.forward(full_ids, depth_mask=depth_mask)
    loss = F.cross_entropy(
        logits.view(-1, cfg.vocab_size), targets.view(-1), ignore_index=-1, reduction="mean",
    )
    loss.backward()

    for i, block in enumerate(model.transformer.recur):
        for name, p in block.named_parameters():
            assert p.grad is not None and p.grad.abs().sum().item() > 0, (
                f"recur[{i}].{name} has no/zero grad"
            )
    assert model.inject.weight.grad.abs().sum().item() > 0
    assert model.lm_head.weight.grad.abs().sum().item() > 0


def test_depth_mask_uniform_equals_num_recur():
    """A uniform depth_mask should produce the same output as passing num_recur directly."""
    model, cfg = _make_model()
    torch.manual_seed(123)
    B, T = 1, 10
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    depth_mask = torch.full((B, T), 3, dtype=torch.long)

    with torch.no_grad():
        logits_a, _ = model.forward(ids, num_recur=3)
        logits_b, _ = model.forward(ids, depth_mask=depth_mask)

    assert torch.allclose(logits_a, logits_b, atol=1e-5), (
        f"uniform depth_mask != num_recur path: max diff={(logits_a - logits_b).abs().max().item():.2e}"
    )
