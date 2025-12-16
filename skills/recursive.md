# Converting your GPT (12-layer, pre-norm RMSNorm, RoPE, GQA) into a Recursive Transformer
Goal: replace `n_layer` stacked blocks with **Prelude + Recurrent Block + Coda**, run the recurrent block `r` times.
Inference default: `r_default = train_recur_mean` (mean effective depth ≈ original depth).

---

## 0) Add config fields
In `GPTConfig`, add:

- `n_prelude: int = 2`
- `n_recur_block: int = 4`
- `n_coda: int = 2`
- `train_recur_mean: float = 4.0`   # also used as default r at inference
- `train_recur_max: int = 16`       # cap sampled r during training
- `bptt_k: int = 4`                 # truncate backward through last k recurrences
- `recur_warm_start: bool = True`
- `inject_mode: str = "concat_linear"`  # we'll use learned adapter
- `kv_cache_recur_budget: int = 1`      # 1 => store only final step (simplest); >1 => ring buffer

Sanity: enforce `n_prelude + n_coda + n_recur_block <= n_layer` for surgery-from-pretrained.

---

## 1) Restructure the model modules
Replace `self.blocks = ModuleList([Block]*n_layer)` with:

- `self.prelude = ModuleList([Block]*n_prelude)`
- `self.recur = ModuleList([Block]*n_recur_block)`
- `self.coda = ModuleList([Block]*n_coda)`

### Weight surgery (if converting an existing checkpoint)
Copy weights like:
- prelude: layers `[0 : n_prelude]`
- recur: pick a contiguous chunk from later-mid / late layers (e.g. `[mid : mid+n_recur_block]`)
- coda: last `[n_coda]` layers

Delete the dropped middle layers.

---

## 2) Add learned input-injection adapter (concat + linear)
We need to inject prelude output `e` into each recurrence step while evolving recurrent state `s`.

Add module:
- `self.inject = Linear(2*h -> h, bias=False)`  (h = n_embd)
- **Initialization**: identity-like `[I | 0]` so `inject(concat(e,s)) = e` initially. This ensures gradients flow (unlike zero-init which kills gradients since inject has no residual bypass).

Forward step:
- compute once: `e = prelude(embed(x))` (shape [B,T,h])
- recurrent state init `s0`:
  - training: `s = e` (simplest) or small noise around e
  - inference: warm-start option (see below)
- each recurrence i:
  - `u = self.inject(concat(e, s))`
  - run recurrent block layers on u: `s = recur_block(u)`
- final: `y = coda(s)` then final RMSNorm + LM head as usual

**Norm guidance**: do NOT add "norm per recurrence". Keep your existing pre-norm RMSNorm in each Block and the final RMSNorm once at the end.

---

## 3) Training recurrence schedule
Choose (P,R,C) so mean effective depth matches baseline:
- effective layers ≈ `P + r_mean*R + C`
- e.g. for depth=20: (2,4,2) with r_mean=4 ⇒ 2 + 4*4 + 2 = 20 effective layers

During training:
- sample r per batch from **Poisson log-normal distribution** (per paper Section 3.3):
  - τ ~ N(log(r̄) - ½σ², σ) where σ=0.5
  - r ~ Poisson(e^τ) + 1
  - clamp to [1, train_recur_max]
- run exactly r recurrences in forward
- backward:
  - truncate to last `bptt_k` recurrences via `s.detach()` (limits gradient depth)
  - with bptt_k=4 and n_recur_block=4, gradient flows through 16 recur layers max

---

## 4) Warm-start recurrence at inference (continuous latent CoT)
When decoding token-by-token, reuse the final recurrent state from the previous token as the next token's initial state (paper Section 6.3).

Implementation:
- After each forward, keep only the **last position's** final state: `warm_start_state = s[:, -1:, :]` (shape [B,1,h])
- At each decode step:
  - compute `e` for the current token (shape [B,1,h] with KV cache)
  - set `s0`:
    - if `recur_warm_start` and `warm_start_state` exists: `s = warm_start_state` (broadcast if needed)
    - else: `s = e`
  - run r_default recurrences
  - save `warm_start_state = s[:, -1:, :]` for the next step

This reduces the average number of steps required to converge by 1-2 (per paper Figure 10).

---

## 5) KV-cache changes for inference
You already have KV caching for standard GPT decoding. With recurrence, you must decide what to cache across:
- token steps (time)
- recurrence steps (depth-iterations)

### 5.1 Strategy: All recurrences read/write, only final persists
Per paper Section 6.2 (Remark 6.1), because all recurrent KV cache entries use the **same K,V projection matrices** (weight sharing), they "match" and the model can attend to entries from different recurrence depths.

Our approach:
- **All recurrences read from and write to the KV cache**
- Since cache position only advances after the last layer (coda), recur layers overwrite the same slot each iteration
- Only the final recurrence's write persists for subsequent tokens

This works because:
1. Weight sharing makes KV values from different depths compatible
2. Previous tokens' final (deepest) states are the most refined representation
3. Memory is O(seq_len), same as standard transformer

### 5.2 What to change in your engine
In your model forward:
1. Pass `kv_cache` to ALL recurrences (not just final)
2. The cache position only advances after coda's last layer processes
3. Recur layers keep overwriting the same position slot
4. Final recurrence's KV values persist for next token to attend to

```python
for i in range(num_recur):
    u = self.inject(torch.cat([e, s], dim=-1))
    for block in self.transformer.recur:
        u = block(u, cos_sin, kv_cache)  # all recurrences use cache
    s = u
```

---

## 6) Minimal forward pseudocode (high level)
```python
x = tok_embed(idx)
x = rmsnorm(x)

# Prelude (once)
for blk in prelude:
    x = blk(x, cos_sin, kv_cache)
e = x

# Initialize recurrent state
if warm_start_state is not None:
    s = warm_start_state.expand(-1, T, -1) if warm_start_state.size(1) != T else warm_start_state
else:
    s = e

# Recurrence (r times)
for i in range(num_recur):
    u = inject(concat(e, s))
    for blk in recur:
        u = blk(u, cos_sin, kv_cache)  # all recurrences read/write cache
    s = u
    # Truncated BPTT
    if bptt_k is not None and i < num_recur - bptt_k:
        s = s.detach()

# Coda (once)
x = s
for blk in coda:
    x = blk(x, cos_sin, kv_cache)
x = final_rmsnorm(x)
logits = lm_head(x)

return logits, s  # return state for warm-start
```

---

## 7) Implemented defaults (nanochat)
- (P, R, C) = (2, 4, 2) → 8 unique layer weights
- train_recur_mean = 4.0 → effective depth 20 (matches original depth=20)
- train_recur_max = 16
- bptt_k = 4 → gradient flows through max 16 recur layers
- inject_mode = "concat_linear" (learned adapter, identity-initialized: passes through e initially)
- recur_warm_start = True
- kv_cache_recur_budget = 1 (cache only final recurrence)
- Sampling: Poisson log-normal distribution (σ=0.5)
