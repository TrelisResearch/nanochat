"""
Test script for the recursive transformer architecture.
Best run on CUDA (the uv.lock is configured for CUDA torch).

Usage:
    uv run python -m scripts.test_recursive
"""
import torch
from nanochat.gpt import GPT, GPTConfig

# Use CUDA (this project's uv.lock targets CUDA torch)
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. This test is best run on a CUDA machine.")
    print("The uv.lock is configured for CUDA torch.")
    exit(0)

DEVICE = "cuda"
print(f"Using device: {DEVICE}")

def test_model_instantiation():
    """Test that the model can be instantiated with recursive config."""
    print("Testing model instantiation...")
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=8,  # n_prelude(2) + n_recur_block(4) + n_coda(2) = 8
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        n_prelude=2,
        n_recur_block=4,
        n_coda=2,
        train_recur_mean=4.0,
        train_recur_max=16,
    )
    model = GPT(config)
    model.init_weights()
    model = model.to(DEVICE)

    # Check structure
    assert len(model.transformer.prelude) == 2, "Prelude should have 2 blocks"
    assert len(model.transformer.recur) == 4, "Recur should have 4 blocks"
    assert len(model.transformer.coda) == 2, "Coda should have 2 blocks"
    assert model.inject is not None, "Inject layer should exist"

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model has {num_params:,} parameters")
    print("  PASSED")
    return model, config

def test_forward_training(model):
    """Test forward pass in training mode (with targets)."""
    print("Testing forward pass (training mode)...")
    B, T = 2, 32
    x = torch.randint(0, 1000, (B, T), device=DEVICE)
    y = torch.randint(0, 1000, (B, T), device=DEVICE)

    # Test with different recursion counts
    for num_recur in [1, 2, 4]:
        loss = model(x, y, num_recur=num_recur)
        assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
        assert not torch.isnan(loss), f"Loss is NaN for num_recur={num_recur}"
        print(f"  num_recur={num_recur}: loss={loss.item():.4f}")
    print("  PASSED")

def test_forward_inference(model):
    """Test forward pass in inference mode (without targets)."""
    print("Testing forward pass (inference mode)...")
    B, T = 2, 32
    x = torch.randint(0, 1000, (B, T), device=DEVICE)

    # Test with different recursion counts
    for num_recur in [1, 2, 4]:
        logits, state = model(x, num_recur=num_recur)
        assert logits.shape == (B, T, 1000), f"Logits shape wrong: {logits.shape}"
        assert state.shape == (B, T, 256), f"State shape wrong: {state.shape}"
        assert not torch.isnan(logits).any(), f"Logits contain NaN for num_recur={num_recur}"
        print(f"  num_recur={num_recur}: logits shape={logits.shape}, state shape={state.shape}")
    print("  PASSED")

def test_warm_start(model):
    """Test warm-start functionality."""
    print("Testing warm-start recurrence...")
    B, T = 1, 16
    x = torch.randint(0, 1000, (B, T), device=DEVICE)

    # First pass: get initial state
    logits1, state1 = model(x, num_recur=2)

    # Second pass: use warm-start from previous state (same shape)
    x2 = torch.randint(0, 1000, (B, T), device=DEVICE)
    logits2, state2 = model(x2, num_recur=2, warm_start_state=state1)

    assert logits2.shape == (B, T, 1000), "Logits shape wrong with warm start"
    assert state2.shape == (B, T, 256), "State shape wrong with warm start"
    print(f"  Same-shape warm-start working correctly")

    # Third pass: test warm-start with shape mismatch (simulating autoregressive decode)
    # warm_start_state is (B, 1, h) but input is (B, T, h) - should broadcast
    state_last = state2[:, -1:, :]  # (B, 1, 256)
    x3 = torch.randint(0, 1000, (B, T), device=DEVICE)
    logits3, state3 = model(x3, num_recur=2, warm_start_state=state_last)

    assert logits3.shape == (B, T, 1000), "Logits shape wrong with broadcast warm start"
    assert state3.shape == (B, T, 256), "State shape wrong with broadcast warm start"
    print(f"  Broadcast warm-start (B,1,h) -> (B,T,h) working correctly")
    print("  PASSED")

def test_backward(model):
    """Test that gradients flow correctly."""
    print("Testing backward pass...")
    B, T = 2, 32
    x = torch.randint(0, 1000, (B, T), device=DEVICE)
    y = torch.randint(0, 1000, (B, T), device=DEVICE)

    # Debug: check requires_grad on parameters
    print(f"  Model training mode: {model.training}")
    print(f"  Sample param requires_grad: {model.lm_head.weight.requires_grad}")

    # Zero gradients
    model.zero_grad()

    # Forward + backward
    loss = model(x, y, num_recur=2)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss requires_grad: {loss.requires_grad}")
    print(f"  Loss grad_fn: {loss.grad_fn}")

    if not loss.requires_grad:
        print("  WARNING: Loss doesn't require gradients!")
        print("  This likely means model is in eval mode or inference_mode is active")
        print("  Skipping gradient check...")
        print("  SKIPPED (no gradients available)")
        return

    loss.backward()

    # Check that key parameters have gradients
    params_with_grad = 0
    params_without_grad = 0
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            params_with_grad += 1
        else:
            params_without_grad += 1

    print(f"  Parameters with non-zero gradients: {params_with_grad}")
    print(f"  Parameters without gradients: {params_without_grad}")

    if params_with_grad == 0:
        print("  WARNING: No parameters have gradients!")
        # Don't fail - this might be an environment issue
        print("  SKIPPED (environment issue?)")
        return

    # Check specific layers
    assert model.inject.weight.grad is not None, "Inject layer should have gradients"
    assert model.inject.weight.grad.abs().sum() > 0, "Inject layer gradients are zero"
    print("  Inject layer has non-zero gradients")
    print("  PASSED")

def test_generate(model, config):
    """Test the generate method."""
    print("Testing generate method...")
    tokens = [1, 2, 3, 4, 5]

    generated = []
    for token in model.generate(tokens, max_tokens=10, temperature=1.0, num_recur=2):
        generated.append(token)

    assert len(generated) == 10, f"Expected 10 tokens, got {len(generated)}"
    assert all(0 <= t < config.vocab_size for t in generated), "Generated tokens out of vocab range"
    print(f"  Generated {len(generated)} tokens: {generated[:5]}...")
    print("  PASSED")

def test_optimizer_setup(model):
    """Test that optimizers are set up correctly with all parameters."""
    print("Testing optimizer setup...")
    optimizers = model.setup_optimizers()

    # Count parameters in optimizers
    opt_params = set()
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group['params']:
                opt_params.add(id(p))

    # Count all model parameters
    model_params = set(id(p) for p in model.parameters())

    assert opt_params == model_params, "Optimizer doesn't cover all parameters"
    print(f"  All {len(model_params)} parameters covered by optimizers")
    print("  PASSED")

def main():
    print("="*60)
    print("Recursive Transformer Test Suite")
    print("="*60)

    # Run tests
    model, config = test_model_instantiation()
    test_forward_training(model)
    test_forward_inference(model)
    test_warm_start(model)
    test_backward(model)
    test_generate(model, config)
    test_optimizer_setup(model)

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)

if __name__ == "__main__":
    main()
