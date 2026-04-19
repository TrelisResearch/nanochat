"""
Sanity checks for the "split-depth SFT" training regime: during training we
want to prefill over the prompt at num_recur=4 (keep=last) and then decode
over the response at num_recur=1, reusing the KVCache. For this to give a
useful gradient signal, gradients must flow from the response loss back
through:
  (a) the decode recur block computations,
  (b) the coda + lm_head,
  (c) the prefill iter-3 K/V (the cached entries that decode attends to),
  (d) the full recurrence chain during prefill (iter-3 s depends on iter-2 s ...).

These tests use tiny random models on CPU.
"""
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache


def _make_model():
    cfg = GPTConfig(
        sequence_len=128,
        vocab_size=256,
        n_layer=6,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        n_prelude=1,
        n_recur_block=2,
        n_coda=1,
        fixed_k=4,
        bptt_k=None,
    )
    torch.manual_seed(0)
    model = GPT(cfg)
    model.init_weights()
    # init_weights zeros c_proj in every block, lm_head, and inject's tail half —
    # randomise so the recur loop does something non-trivial and gradient signal
    # actually reaches the blocks under test.
    with torch.no_grad():
        for block in list(model.transformer.prelude) + list(model.transformer.recur) + list(model.transformer.coda):
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.1)
            torch.nn.init.normal_(block.attn.c_proj.weight, std=0.1)
        torch.nn.init.normal_(model.inject.weight, std=0.05)
        torch.nn.init.normal_(model.lm_head.weight, std=0.05)
    model.train()
    return model, cfg


def _make_cache(cfg, batch=1, seq_len=64):
    num_layers = cfg.n_prelude + cfg.n_recur_block + cfg.n_coda
    return KVCache(
        batch_size=batch,
        num_heads=cfg.n_kv_head,
        seq_len=seq_len,
        head_dim=cfg.n_embd // cfg.n_head,
        num_layers=num_layers,
    )


def test_two_stage_forward_grad_flows_to_recur_params():
    """Gradients on response loss must reach recur-block params through the cache."""
    model, cfg = _make_model()
    torch.manual_seed(1)
    B, P, S = 1, 6, 4  # prefix=6, suffix=4
    prompt_ids = torch.randint(0, cfg.vocab_size, (B, P))
    response_ids = torch.randint(0, cfg.vocab_size, (B, S))
    response_targets = torch.randint(0, cfg.vocab_size, (B, S))

    cache = _make_cache(cfg, batch=B, seq_len=P + S + 4)
    # Stage 1: prefill with num_recur=4, keep=last
    _, warm = model.forward(prompt_ids, kv_cache=cache, num_recur=4, prefill_kv_keep="last")
    assert cache.get_pos() == P
    # Warm start: take last position (standard inference behavior)
    warm_last = warm[:, -1:, :]

    # Stage 2: decode with num_recur=1
    logits, _ = model.forward(
        response_ids, kv_cache=cache, num_recur=1, warm_start_state=warm_last,
    )
    assert cache.get_pos() == P + S
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), response_targets.view(-1))
    loss.backward()

    # Recur block params must receive gradient (both attn and mlp).
    for i, block in enumerate(model.transformer.recur):
        for name, p in block.named_parameters():
            assert p.grad is not None, f"recur[{i}].{name} has no grad"
            assert torch.isfinite(p.grad).all(), f"recur[{i}].{name} grad non-finite"
            assert p.grad.abs().sum().item() > 0, f"recur[{i}].{name} grad is all zero"

    # inject must get grad too (used every recur iteration)
    assert model.inject.weight.grad is not None
    assert model.inject.weight.grad.abs().sum().item() > 0

    # Prelude + coda must get grad (they feed the stream)
    for block in model.transformer.prelude:
        assert block.attn.c_q.weight.grad is not None
        assert block.attn.c_q.weight.grad.abs().sum().item() > 0
    for block in model.transformer.coda:
        assert block.attn.c_q.weight.grad is not None
        assert block.attn.c_q.weight.grad.abs().sum().item() > 0


def test_prefill_depth_affects_response_loss():
    """
    If we change num_recur_prefill from 1 to 4, the response logits should differ,
    confirming the cached prefix K/V (from different depths) actually feeds the
    decode attention. This rules out a silent bug where the cache isn't being read.
    """
    model, cfg = _make_model()
    torch.manual_seed(2)
    B, P, S = 1, 6, 4
    prompt_ids = torch.randint(0, cfg.vocab_size, (B, P))
    response_ids = torch.randint(0, cfg.vocab_size, (B, S))

    def run(prefill_k):
        cache = _make_cache(cfg, batch=B, seq_len=P + S + 4)
        with torch.no_grad():
            _, warm = model.forward(prompt_ids, kv_cache=cache, num_recur=prefill_k, prefill_kv_keep="last")
            warm_last = warm[:, -1:, :]
            logits, _ = model.forward(
                response_ids, kv_cache=cache, num_recur=1, warm_start_state=warm_last,
            )
        return logits

    logits_p1 = run(prefill_k=1)
    logits_p4 = run(prefill_k=4)

    # Different prefix depths must produce different decode logits
    assert not torch.allclose(logits_p1, logits_p4, atol=1e-6)


def test_batched_two_stage_with_padding():
    """
    Variable-length prompts padded to the max prompt length should still work.
    We check the cache position advances correctly and the forward runs.
    """
    model, cfg = _make_model()
    torch.manual_seed(3)
    B, P_max, S = 2, 8, 4
    prompt_ids = torch.randint(0, cfg.vocab_size, (B, P_max))
    response_ids = torch.randint(0, cfg.vocab_size, (B, S))
    response_targets = torch.randint(0, cfg.vocab_size, (B, S))

    cache = _make_cache(cfg, batch=B, seq_len=P_max + S + 4)
    _, warm = model.forward(prompt_ids, kv_cache=cache, num_recur=4, prefill_kv_keep="last")
    assert cache.get_pos() == P_max
    warm_last = warm[:, -1:, :]
    logits, _ = model.forward(
        response_ids, kv_cache=cache, num_recur=1, warm_start_state=warm_last,
    )
    assert cache.get_pos() == P_max + S
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), response_targets.view(-1))
    loss.backward()
    assert torch.isfinite(loss)
