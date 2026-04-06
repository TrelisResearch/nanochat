import pytest
import torch
import torch.nn.functional as F

from nanochat.gpt import GPTConfig, GPT, EBT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_GPU = torch.cuda.is_available()
RTOL = 1e-4
ATOL = 1e-4

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="CUDA not available")


def precompute_rotary_embeddings(seq_len, head_dim, base=10000, device=None):
    if device is None:
        device = DEVICE
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.bfloat16(), sin.bfloat16()
    cos, sin = (
        cos[None, :, None, :],
        sin[None, :, None, :],
    )  # add batch and head dims for later broadcasting
    return cos, sin


def energy_optim(model, input, cos_sin, n_steps, opt_step_size):
    x = input.clone().requires_grad_(True)
    _, T, _ = x.shape
    S = T // 2

    with torch.set_grad_enabled(True):
        for _ in range(n_steps):
            energy = model(x, cos_sin)[:, S:, :].sum()
            grad = torch.autograd.grad(
                energy,
                x,
                create_graph=True,
            )[0]
            grad[:, :S, :] = 0
            x = x - opt_step_size * grad

    return x


def _make_ebt_config(use_flash_attn=False):
    return GPTConfig(
        sequence_len=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=256,
        use_flash_attn=use_flash_attn,
    )


def _make_gpt_config(use_flash_attn=False):
    return GPTConfig(
        sequence_len=32,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        use_flash_attn=use_flash_attn,
        n_steps=2,
        opt_step_size=0.5,
    )


# ---------------------------------------------------------------------------
# EBT unit tests (no flash)
# ---------------------------------------------------------------------------


def test_ebt_no_flash():
    config = _make_ebt_config(use_flash_attn=False)
    B, T, D = 4, config.sequence_len, config.n_embd
    input = torch.randn(B, T, D, device=DEVICE)
    model = EBT(config).to(DEVICE)
    head_dim = config.n_embd // config.n_head
    cos, sin = precompute_rotary_embeddings(T // 2, head_dim, device=DEVICE)

    pred = energy_optim(model, input, (cos, sin), 2, 0.8)
    assert pred.shape == (B, T, D)


# ---------------------------------------------------------------------------
# EBT unit tests (flash attention — GPU only)
# ---------------------------------------------------------------------------


@requires_gpu
def test_ebt_with_flash():
    config = _make_ebt_config(use_flash_attn=True)
    B, T, D = 4, config.sequence_len, config.n_embd
    input = torch.randn(B, T, D, device=DEVICE)
    model = EBT(config).to(DEVICE)
    head_dim = config.n_embd // config.n_head
    cos, sin = precompute_rotary_embeddings(T // 2, head_dim, device=DEVICE)

    pred = energy_optim(model, input, (cos, sin), 2, 0.8)
    assert pred.shape == (B, T, D)


@requires_gpu
def test_flash_closeness():
    """Verify flash and non-flash attention produce numerically close results."""
    config = _make_ebt_config(use_flash_attn=False)
    B, T, D = 4, config.sequence_len, config.n_embd
    input = torch.randn(B, T, D, device=DEVICE)
    target = torch.randn(B, T // 2, D, device=DEVICE)
    head_dim = config.n_embd // config.n_head
    cos, sin = precompute_rotary_embeddings(T // 2, head_dim, device=DEVICE)
    cos_sin = (cos, sin)

    # Non-flash model
    model = EBT(config).to(DEVICE)
    model.zero_grad()
    orig_pred = energy_optim(model, input, cos_sin, 2, 0.8)
    loss_1 = F.mse_loss(orig_pred[:, T // 2 :, :], target)
    loss_1.backward()

    # Flash model (same weights)
    flash_config = _make_ebt_config(use_flash_attn=True)
    flash_model = EBT(flash_config)
    flash_model.load_state_dict(model.state_dict())
    flash_model = flash_model.to(DEVICE)

    flash_model.zero_grad()
    flash_pred = energy_optim(flash_model, input, cos_sin, 2, 0.8)
    loss_2 = F.mse_loss(flash_pred[:, T // 2 :, :], target)
    loss_2.backward()

    assert torch.allclose(orig_pred, flash_pred, rtol=RTOL, atol=ATOL), (
        "Final optimization outputs are different"
    )
    param1 = next(model.parameters())
    param2 = next(flash_model.parameters())
    assert torch.allclose(param1.grad, param2.grad, rtol=RTOL, atol=ATOL), (
        "Gradients are different"
    )


# ---------------------------------------------------------------------------
# Training step tests (GPT wrapper)
# ---------------------------------------------------------------------------


def _run_training_step(config):
    """Helper: one forward + backward + optimizer step. Returns True on success."""
    model = GPT(config).to(DEVICE)
    model.train()

    B, T = 4, config.sequence_len
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=DEVICE)
    targets = torch.randint(0, config.vocab_size, (B, T), device=DEVICE)

    optimizers = model.setup_optimizers(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
    )
    for opt in optimizers:
        opt.zero_grad()

    loss = model(input_ids, targets=targets)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    loss.backward()

    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0, "No gradients computed"

    for opt in optimizers:
        opt.step()

    return model


def test_training_step():
    config = _make_gpt_config(use_flash_attn=False)
    _run_training_step(config)


@requires_gpu
@pytest.mark.xfail(
    reason="jvp_flash_attention triton kernel incompatible with torch.func.grad tensors",
    raises=RuntimeError,
)
def test_training_step_flash():
    config = _make_gpt_config(use_flash_attn=True)
    _run_training_step(config)


# ---------------------------------------------------------------------------
# torch.compile tests
# ---------------------------------------------------------------------------


def _run_compile_ebt(use_flash_attn):
    """Helper: compile EBT and verify outputs match eager mode."""
    config = _make_ebt_config(use_flash_attn=use_flash_attn)
    B, T, D = 4, config.sequence_len, config.n_embd
    model = EBT(config).to(DEVICE)
    head_dim = config.n_embd // config.n_head
    cos, sin = precompute_rotary_embeddings(T // 2, head_dim, device=DEVICE)
    cos_sin = (cos, sin)
    input_tensor = torch.randn(B, T, D, device=DEVICE)

    compiled_model = torch.compile(model)

    with torch.no_grad():
        eager_out = model(input_tensor, cos_sin)
        compiled_out = compiled_model(input_tensor, cos_sin)

    assert torch.allclose(eager_out, compiled_out, rtol=RTOL, atol=ATOL), (
        f"Compiled output differs from eager (max diff: "
        f"{(eager_out - compiled_out).abs().max().item():.8f})"
    )


def test_compile_no_flash():
    _run_compile_ebt(use_flash_attn=False)


@requires_gpu
def test_compile_flash():
    _run_compile_ebt(use_flash_attn=True)


# ---------------------------------------------------------------------------
# torch.compile training test (full GPT with compiled model)
# ---------------------------------------------------------------------------


def test_compile_training_step():
    config = _make_gpt_config(use_flash_attn=False)
    model = GPT(config).to(DEVICE)
    model.train()
    compiled_model = torch.compile(model)

    B, T = 4, config.sequence_len
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=DEVICE)
    targets = torch.randint(0, config.vocab_size, (B, T), device=DEVICE)

    optimizers = model.setup_optimizers(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
    )
    for opt in optimizers:
        opt.zero_grad()

    loss = compiled_model(input_ids, targets=targets)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    loss.backward()

    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0, "No gradients computed"

    for opt in optimizers:
        opt.step()


@requires_gpu
@pytest.mark.xfail(
    reason="jvp_flash_attention triton kernel incompatible with torch.func.grad tensors",
    raises=RuntimeError,
)
def test_compile_training_step_flash():
    config = _make_gpt_config(use_flash_attn=True)
    model = GPT(config).to(DEVICE)
    model.train()
    compiled_model = torch.compile(model)

    B, T = 4, config.sequence_len
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=DEVICE)
    targets = torch.randint(0, config.vocab_size, (B, T), device=DEVICE)

    optimizers = model.setup_optimizers(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
    )
    for opt in optimizers:
        opt.zero_grad()

    loss = compiled_model(input_ids, targets=targets)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    loss.backward()

    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0, "No gradients computed"

    for opt in optimizers:
        opt.step()
