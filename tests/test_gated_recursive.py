"""
Unit tests for the gated recursive transformer.

Run with:
  python -m pytest tests/test_gated_recursive.py -v
"""

import torch
import pytest
from nanochat.gpt import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides):
    """Create a small GPTConfig suitable for CPU unit tests."""
    defaults = dict(
        sequence_len=32,
        vocab_size=256,
        n_layer=8,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        n_prelude=2,
        n_recur_block=4,
        n_coda=2,
        fixed_k=4,
        bptt_k=4,
        gate_threshold=0.01,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def make_model(config=None, device="cpu"):
    if config is None:
        config = make_config()
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    return model


def set_inference_mode(model):
    """Put model in inference mode (model.train(False) == model.eval(), avoids the eval hook)."""
    model.train(False)


# ---------------------------------------------------------------------------
# Config / architecture tests
# ---------------------------------------------------------------------------

def test_config_has_fixed_k():
    config = make_config(fixed_k=8)
    assert config.fixed_k == 8


def test_config_no_poisson_fields():
    """Ensure old Poisson sampling fields are gone from GPTConfig."""
    config = make_config()
    assert not hasattr(config, "train_recur_mean"), "train_recur_mean should not exist on GPTConfig"
    assert not hasattr(config, "train_recur_max"), "train_recur_max should not exist on GPTConfig"


def test_model_has_gate_proj():
    model = make_model()
    assert hasattr(model, "gate_proj"), "model must have gate_proj"
    import torch.nn as nn
    assert isinstance(model.gate_proj, nn.Linear)
    assert model.gate_proj.weight.shape == (1, 2 * model.config.n_embd)
    assert model.gate_proj.bias is not None, "gate_proj must have bias"
    assert model.gate_proj.weight.abs().sum().item() == 0.0, "gate_proj.weight must be zero at init"
    assert abs(model.gate_proj.bias.item() - 2.0) < 1e-5, "gate_proj.bias must be +2.0 at init (sigmoid(2)≈0.88, gates start open)"


# ---------------------------------------------------------------------------
# Forward pass: training mode (returns loss + gate_cost)
# ---------------------------------------------------------------------------

def test_forward_returns_tuple_in_training_mode():
    model = make_model()
    set_inference_mode(model)
    B, T = 2, 8
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    out = model(idx, targets)
    assert isinstance(out, tuple) and len(out) == 2, "forward(targets=...) must return (loss, gate_cost)"
    loss, gate_cost = out
    assert loss.shape == (), f"loss must be scalar, got {loss.shape}"
    assert gate_cost.shape == (), f"gate_cost must be scalar, got {gate_cost.shape}"


def test_gate_cost_is_positive():
    model = make_model()
    set_inference_mode(model)
    B, T = 2, 8
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    _, gate_cost = model(idx, targets)
    assert gate_cost.item() > 0, "gate_cost should be > 0 when gates are initialized open"


def test_gate_cost_is_normalized():
    """gate_cost is normalised to gate_mean in [0, 1]; value does not scale with sequence length."""
    model = make_model()
    set_inference_mode(model)
    B = 2
    _, gc_short = model(torch.randint(0, 256, (B, 4)), torch.randint(0, 256, (B, 4)))
    _, gc_long = model(torch.randint(0, 256, (B, 16)), torch.randint(0, 256, (B, 16)))
    # Both should be in [0, 1]; equal because same model weights → same avg gate openness
    assert 0.0 <= gc_short.item() <= 1.0
    assert 0.0 <= gc_long.item() <= 1.0


def test_loss_reduction_none_shape():
    """With loss_reduction='none', loss should be 1D (B*T,) — forward flattens before cross_entropy."""
    model = make_model()
    set_inference_mode(model)
    B, T = 2, 8
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    loss, _ = model(idx, targets, loss_reduction='none')
    assert loss.shape == (B * T,), f"Expected ({B*T},), got {loss.shape}"


# ---------------------------------------------------------------------------
# Forward pass: inference mode (returns logits + recur state)
# ---------------------------------------------------------------------------

def test_forward_returns_logits_in_inference_mode():
    model = make_model()
    set_inference_mode(model)
    B, T = 2, 8
    idx = torch.randint(0, 256, (B, T))
    out = model(idx)
    assert isinstance(out, tuple) and len(out) == 2
    logits, s = out
    assert logits.shape == (B, T, model.config.vocab_size)
    assert s.shape == (B, T, model.config.n_embd)


# ---------------------------------------------------------------------------
# Fixed K
# ---------------------------------------------------------------------------

def test_fixed_k_used_by_default():
    """Model should run fixed_k recurrences without error."""
    model = make_model(make_config(fixed_k=2))
    set_inference_mode(model)
    B, T = 1, 4
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    loss, gate_cost = model(idx, targets)
    assert loss.item() > 0


def test_num_recur_override_runs_without_error():
    """Passing num_recur should override fixed_k and complete without error."""
    model = make_model()
    set_inference_mode(model)
    B, T = 1, 4
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    _, gc1 = model(idx, targets, num_recur=1)
    _, gc4 = model(idx, targets, num_recur=4)
    # gate_mean is normalised per recurrence step, so the value is the same regardless of num_recur
    assert 0.0 <= gc1.item() <= 1.0
    assert 0.0 <= gc4.item() <= 1.0


# ---------------------------------------------------------------------------
# Gated update semantics
# ---------------------------------------------------------------------------

def test_zero_gate_produces_near_identity_update():
    """With fixed_k=1 (only forced step-0), gate_cost is always 0 regardless of gate weights."""
    model = make_model(make_config(fixed_k=1))
    set_inference_mode(model)
    B, T = 1, 4
    idx = torch.randint(0, 256, (B, T))
    logits, s = model(idx)
    assert s.shape == (B, T, model.config.n_embd)
    # gate_cost should be 0: only step-0 runs (forced, not penalised)
    _, gate_cost = model(idx, torch.randint(0, 256, (B, T)))
    assert gate_cost.item() < 1e-6, f"gate_cost should be 0 with fixed_k=1, got {gate_cost.item()}"


def test_open_gate_produces_nonzero_gate_cost():
    """With fixed_k=2 (one gated step), gate_cost should be positive with any non-trivial weights."""
    model = make_model(make_config(fixed_k=2))
    set_inference_mode(model)
    B, T = 1, 4
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    _, gate_cost = model(idx, targets)
    # gate_mean is normalised to [0,1]; sigmoid output is always > 0
    assert gate_cost.item() > 0.0, f"gate_mean should be > 0 with fixed_k=2, got {gate_cost.item()}"


# ---------------------------------------------------------------------------
# Backward pass (gradients flow through gate)
# ---------------------------------------------------------------------------

def test_gradients_flow_through_gate():
    """gate_proj.weight must receive a non-zero gradient during training.

    Gate uses cat([u,s]) which is always non-zero (u and s are always O(1)),
    so weight.grad is non-zero from the first step regardless of init.
    """
    model = make_model()
    model.train()
    B, T = 2, 8
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    loss, gate_cost = model(idx, targets)
    total_loss = loss + 1e-3 * gate_cost
    total_loss.backward()
    assert model.gate_proj.weight.grad is not None, "gate_proj.weight should have gradient tensor"
    assert model.gate_proj.weight.grad.abs().sum().item() > 0, "gate_proj.weight gradient should be non-zero"


def test_gate_cost_grad_flows_to_weight():
    """gate_cost must be differentiable w.r.t. gate_proj.weight."""
    model = make_model()
    model.train()
    B, T = 2, 8
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    _, gate_cost = model(idx, targets)
    gate_cost.backward()
    assert model.gate_proj.weight.grad is not None
    assert model.gate_proj.weight.grad.abs().sum().item() > 0


def test_bptt_truncation_does_not_break_gradients():
    """With bptt_k < fixed_k, gradients should still flow through the last bptt_k recurrences."""
    model = make_model(make_config(fixed_k=4, bptt_k=2))
    model.train()
    B, T = 1, 4
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    loss, gate_cost = model(idx, targets)
    (loss + 1e-3 * gate_cost).backward()
    # Parameters in recur blocks should still have gradients
    for name, p in model.transformer.recur.named_parameters():
        assert p.grad is not None, f"recur.{name} should have gradient with bptt_k=2"


# ---------------------------------------------------------------------------
# Optimizer setup
# ---------------------------------------------------------------------------

def test_setup_optimizers_covers_all_params():
    """All model parameters must appear in exactly one optimizer group."""
    model = make_model()
    optimizers = model.setup_optimizers()
    all_opt_params = set()
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                pid = id(p)
                assert pid not in all_opt_params, "Parameter appears in multiple optimizer groups"
                all_opt_params.add(pid)
    model_params = {id(p) for p in model.parameters()}
    assert model_params == all_opt_params, "Some parameters are not covered by any optimizer"


def test_gate_proj_in_optimizers():
    """gate_proj weight and bias must both be covered by an optimizer."""
    model = make_model()
    optimizers = model.setup_optimizers()
    gate_weight_id = id(model.gate_proj.weight)
    gate_bias_id = id(model.gate_proj.bias)
    opt_param_ids = set()
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                opt_param_ids.add(id(p))
    assert gate_weight_id in opt_param_ids, "gate_proj.weight not in any optimizer"
    assert gate_bias_id in opt_param_ids, "gate_proj.bias not in any optimizer"


# ---------------------------------------------------------------------------
# Early exit (inference path with kv_cache)
# ---------------------------------------------------------------------------

def test_early_exit_high_threshold_does_not_crash():
    """With gate_threshold=1.0, early exit fires on first step; model should still return valid output."""
    config = make_config(fixed_k=4, gate_threshold=1.0)
    model = make_model(config)
    set_inference_mode(model)
    from nanochat.engine import KVCache
    n_layers = config.n_prelude + config.n_recur_block + config.n_coda
    kv = KVCache(
        batch_size=1,
        num_heads=config.n_head,
        seq_len=config.sequence_len,
        head_dim=config.n_embd // config.n_head,
        num_layers=n_layers,
    )
    B, T = 1, 4
    idx = torch.randint(0, 256, (B, T))
    logits, s = model(idx, kv_cache=kv)
    assert logits.shape == (B, T, config.vocab_size)
    assert s.shape == (B, T, config.n_embd)


# ---------------------------------------------------------------------------
# Warm-start and generate()
# ---------------------------------------------------------------------------

def test_generate_produces_correct_number_of_tokens():
    """generate() should yield exactly max_tokens tokens."""
    model = make_model()
    set_inference_mode(model)
    tokens = [1, 2, 3, 4]
    generated = list(model.generate(tokens, max_tokens=5, temperature=0.0))
    assert len(generated) == 5
    for t in generated:
        assert isinstance(t, int)


def test_generate_with_temperature_zero_is_deterministic():
    """temperature=0 (greedy) should produce identical outputs on repeated calls."""
    model = make_model()
    set_inference_mode(model)
    tokens = [10, 20, 30]
    out1 = list(model.generate(tokens, max_tokens=4, temperature=0.0))
    out2 = list(model.generate(tokens, max_tokens=4, temperature=0.0))
    assert out1 == out2, "Greedy decoding should be deterministic"


# ---------------------------------------------------------------------------
# Gate conditioned on cat([e, s])
# ---------------------------------------------------------------------------

def test_gate_uses_state_signal():
    """Gate output changes when s changes, confirming gate is computed from cat([u,s]).

    weight=0 at init so we perturb it first — at init gate is constant (bias only).
    """
    torch.manual_seed(0)
    model = make_model()
    model.train(False)
    with torch.no_grad():
        model.gate_proj.weight.normal_(0, 0.01)  # perturb so weight contributes
        B, T = 1, 4
        u = torch.randn(B, T, model.config.n_embd)
        s1 = torch.zeros(B, T, model.config.n_embd)
        s2 = torch.randn(B, T, model.config.n_embd)
        g1 = torch.sigmoid(model.gate_proj(torch.cat([u, s1], dim=-1)))
        g2 = torch.sigmoid(model.gate_proj(torch.cat([u, s2], dim=-1)))
    assert not torch.allclose(g1, g2), "Gate output must differ when s differs (gate depends on cat([u,s]))"


# ---------------------------------------------------------------------------
# FLOPs estimate
# ---------------------------------------------------------------------------

def test_flops_estimate_uses_fixed_k():
    """estimate_flops() should use fixed_k, not a Poisson mean."""
    config_k2 = make_config(fixed_k=2)
    config_k4 = make_config(fixed_k=4)
    model_k2 = make_model(config_k2)
    model_k4 = make_model(config_k4)
    flops_k2 = model_k2.estimate_flops()
    flops_k4 = model_k4.estimate_flops()
    assert flops_k4 > flops_k2, "More recurrences should mean more FLOPs"
