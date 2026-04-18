"""
Tests for the prefill_kv_keep="first" / "last" path in GPT.forward.

The key invariant: when prefill_num_recur > 1 and prefill_kv_keep="first",
the recur-layer K/V slots left in the KV cache after prefill must exactly
match the K/V that would have been produced by running a single recurrence
(num_recur=1) over the same inputs.
"""
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache


def _make_model(n_recur_block=2, n_prelude=1, n_coda=1, n_embd=64, n_layer=4):
    cfg = GPTConfig(
        sequence_len=128,
        vocab_size=256,
        n_layer=n_layer,
        n_head=4,
        n_kv_head=2,
        n_embd=n_embd,
        n_prelude=n_prelude,
        n_recur_block=n_recur_block,
        n_coda=n_coda,
        fixed_k=4,
        bptt_k=None,
    )
    torch.manual_seed(0)
    model = GPT(cfg)
    model.init_weights()
    # init_weights zeros c_proj in every block and sets inject to [I|0] (ignores s),
    # which makes the recur loop a no-op at init. Randomise those so the loop actually
    # mixes u and s and successive iterations produce different K/V.
    with torch.no_grad():
        for block in model.transformer.recur:
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.1)
            torch.nn.init.normal_(block.attn.c_proj.weight, std=0.1)
        torch.nn.init.normal_(model.inject.weight, std=0.05)
    model.eval()
    return model, cfg


def _make_cache(cfg, batch=1, seq_len=32):
    num_layers = cfg.n_prelude + cfg.n_recur_block + cfg.n_coda
    return KVCache(
        batch_size=batch,
        num_heads=cfg.n_kv_head,
        seq_len=seq_len,
        head_dim=cfg.n_embd // cfg.n_head,
        num_layers=num_layers,
    )


def test_keep_first_matches_single_recurrence_kv():
    model, cfg = _make_model()
    B, T = 1, 8
    torch.manual_seed(1)
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    # Run with num_recur=1 — whatever K/V land in cache is the reference.
    cache_ref = _make_cache(cfg, batch=B)
    with torch.no_grad():
        model(idx, kv_cache=cache_ref, num_recur=1)
    recur_start = cfg.n_prelude
    recur_end = cfg.n_prelude + cfg.n_recur_block
    ref_kv = cache_ref.kv_cache[recur_start:recur_end, :, :, :, :T, :].clone()

    # Run with num_recur=4 and prefill_kv_keep="first". Post-loop restore should
    # write iter-0 K/V into the same slots — matching the single-recurrence run.
    cache_first = _make_cache(cfg, batch=B)
    with torch.no_grad():
        model(idx, kv_cache=cache_first, num_recur=4, prefill_kv_keep="first")
    first_kv = cache_first.kv_cache[recur_start:recur_end, :, :, :, :T, :].clone()

    assert torch.allclose(first_kv, ref_kv, atol=1e-5), "keep=first K/V should equal num_recur=1 K/V"


def test_keep_last_differs_from_first_when_num_recur_gt_1():
    model, cfg = _make_model()
    B, T = 1, 8
    torch.manual_seed(2)
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    recur_start = cfg.n_prelude
    recur_end = cfg.n_prelude + cfg.n_recur_block

    cache_last = _make_cache(cfg, batch=B)
    with torch.no_grad():
        model(idx, kv_cache=cache_last, num_recur=4, prefill_kv_keep="last")
    last_kv = cache_last.kv_cache[recur_start:recur_end, :, :, :, :T, :].clone()

    cache_first = _make_cache(cfg, batch=B)
    with torch.no_grad():
        model(idx, kv_cache=cache_first, num_recur=4, prefill_kv_keep="first")
    first_kv = cache_first.kv_cache[recur_start:recur_end, :, :, :, :T, :].clone()

    # With init weights, inject is identity-like on (e, s=e) so iter 0 may be near-trivial;
    # still, later iters should perturb K/V at least a little.
    assert not torch.allclose(last_kv, first_kv, atol=1e-6), "keep=last and keep=first should differ"


def test_keep_first_no_op_for_num_recur_1():
    model, cfg = _make_model()
    B, T = 1, 8
    torch.manual_seed(3)
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    cache_a = _make_cache(cfg, batch=B)
    cache_b = _make_cache(cfg, batch=B)
    with torch.no_grad():
        logits_a, _ = model(idx, kv_cache=cache_a, num_recur=1, prefill_kv_keep="last")
        logits_b, _ = model(idx, kv_cache=cache_b, num_recur=1, prefill_kv_keep="first")

    assert torch.allclose(logits_a, logits_b)
    assert torch.allclose(cache_a.kv_cache[:, :, :, :, :T, :], cache_b.kv_cache[:, :, :, :, :T, :])
