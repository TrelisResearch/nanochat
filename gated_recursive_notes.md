# Gated Recursive Training — v8 Run Notes
Date: 2026-03-18

## What we ran
Mid-training + SFT starting from `Trelis/nanochat-recursive` base checkpoint (d20),
gated loss with `λ_max=1e-3`, `gate_warmup_ratio=0.2`.
Branch: `gated-recursive`. Pod: `h7yow9j2mfz462`, 8×H100, $21.52/hr.

---

## Bugs fixed en route (v1–v8)

| Pod | Error | Fix |
|-----|-------|-----|
| v1 | `cd /workspace/nanochat: No such file` | `dockerStartCmd` bypasses `runpod_onstart.sh`; inlined full git clone into `TRAIN_CMD` |
| v2 | `pip install .[train]` extra missing + torchrun `--run` ambiguous | Removed extra, added `--` separator |
| v3 | `GPTConfig unexpected kwarg train_recur_mean` | Compat shim in `checkpoint_manager.py`: strips old fields, defaults `fixed_k=4`, `strict=False` |
| v4 | `tokenizer.pkl not found` | Added explicit tokenizer pull from HF (base checkpoint doesn't bundle tokenizer) |
| v5 | Stuck provisioning on new machine | Killed |
| v6 | CUDA OOM at `relu²` | `device_batch_size` reduced 32→16 (gated model stores `s_old` + `u-s` per recurrence step ≈ +1.7 GB vs recursive) |
| v7 | `ValueError: input tensor must be the same size as output size times world_size` in `DistAdamW` | `gate_proj.bias` shape `(1,)` not divisible by world_size=8; fixed with `CompositeOptimizer` |
| v8 | ✅ Completed successfully | — |

### v7 fix detail: CompositeOptimizer
`DistAdamW.step()` calls `reduce_scatter_tensor` which requires `param.shape[0] % world_size == 0`.
`gate_proj.weight` is `(1, 1280)` and `gate_proj.bias` is `(1,)` — both have `shape[0]=1`, not
divisible by 8.

Fix: route `gate_proj_params` through plain `torch.optim.AdamW` in DDP mode. A new
`CompositeOptimizer` wraps `[DistAdamW, gate_adamw]` behind a unified interface
(`.step()`, `.state_dict()`, `.load_state_dict()`, `.param_groups` by reference so LR
scheduling propagates). Training scripts required zero changes.

---

## Training results

### Mid-training (812 steps, ~10 min)
| Checkpoint | Val bpb |
|------------|---------|
| Step 300 (37%) | 0.4584 |
| Step 750 (92%) | 0.4210 |

Throughput: ~810ms/step, ~647k tok/sec, ~29% MFU. Solid throughout.

### SFT (701 steps)
| Metric | Value |
|--------|-------|
| Val loss @ step 600 | 1.072 |
| MMLU (step 600) | 342/1024 = **33.4%** |
| ARC-Easy (step 600) | 461/1024 = **45.0%** |

MMLU above random (25%). Modest but expected for this model size.
Model pushed to `Trelis/nanochat-gated-recursive`, path `sft/d20`.

---

## Gate analysis: gates never closed

Throughout all of mid-training and SFT, the logged `gate` metric was **constant at 131072.00**.

`gate_cost = g.sum()` accumulated over 4 recurrences × 2 grad_accum steps.
When `g = 1.0` for all tokens: `16 × 1024 × 4 × 2 = 131072` — the theoretical maximum.
The initial gate bias is +2.0 → sigmoid(2.0) ≈ 0.88, which would give ~115k, not 131072.
So during training the CE loss gradient actively drove gates **more open** (bias grew from +2
toward a large positive value), not less. λ never won.

### Why λ-insensitivity didn't apply here

The design intent was that even a small λ should close gates on "easy" tokens for free
(no CE cost → gate penalty wins). This is correct in principle, but requires the model to
have representations where some tokens *can* early-exit.

The recursive model was pre-trained with **all 4 recurrences always firing**. Every token's
representation was shaped by the assumption of full recursion. There are no "free" tokens to
gate off — closing any gate hurts CE loss, so the CE gradient always wins regardless of λ size.

**This is a pre-training mismatch problem, not a λ-tuning problem.** The λ-insensitivity
argument holds only when the model has the flexibility to choose easy tokens. That flexibility
has to be built in from pre-training, not retrofitted via fine-tuning.

### Implication

To get meaningful gate sparsity, gates need to be present during pre-training so the model
co-adapts its recurrent representations and the gating mechanism together. Mid+SFT fine-tuning
from a fully-recursive checkpoint is unlikely to produce real sparsity regardless of λ.

---

## Recommendations for next run

1. **Pre-train with gates from scratch** (or use `launch_pretrain.py` for continued pre-training
   at ~20% of original token budget). This gives the model enough token budget to co-adapt
   representations with gating.

2. **λ schedule**: with pre-training, the "free gating" property should emerge naturally —
   but still worth monitoring `g.mean()` (not just `g.sum()`) to detect subtle gate movement
   before full closure.

3. **Log `g.mean()` instead of `g.sum()`** in future runs — the sum is proportional to batch size
   and harder to interpret; mean gives a direct read on average gate openness across tokens.

---

## SSH / RunPod operational notes

### SSH gateway format
```bash
# Always use the gateway, never direct IP:port
echo "command" | ssh -tt -i ~/.ssh/id_runpod -o StrictHostKeyChecking=no \
  <pod_id>-<machine_suffix>@ssh.runpod.io
```

- `-tt` forces PTY allocation (required by RunPod gateway)
- Pipe command via stdin
- Machine suffix is **not** exposed by the REST API (`machine` field returns `{}`)
- Get it from the RunPod web console (click pod → SSH) or ask the user

### Known machine → suffix mappings
| Machine ID | Suffix | Key |
|------------|--------|-----|
| `s85w9odbixr0` | `64411fbf` | `~/.ssh/id_runpod` |
| `kef0latr3q1u` | `64411fa8` | `~/.ssh/id_ed25519` (v8 pod) |
| `eim0q9sow7cw` | unknown | — |

Note: `id_runpod.pub` matches the first key in the pod's `PUBLIC_KEY` env var.
`id_ed25519` is registered separately in the RunPod account (not visible in `PUBLIC_KEY`).

### v8 SSH outcome
Was not able to SSH in during the run (permission denied with both keys piped via `-tt`).
The user connected successfully via interactive SSH (`ssh <target> -i ~/.ssh/id_ed25519`).
The `-tt` + piped stdin method may require the key to be in the ssh-agent or there may be
a gateway quirk with non-interactive sessions on this machine. Worth investigating further.

---

## Cost summary
- v8 pod: ~11 min training time × $21.52/hr ≈ **~$4** for mid+SFT
- All v1–v7 debugging costs were dominated by provisioning/crash overhead, not training time

---

# Gated Recursive Training — v9b Run Notes
Date: 2026-03-19

## What we ran
Full pipeline from scratch: base_train → mid_train → chat_sft.
Branch: `gated-recursive`. Pod: `doirbu3jdewmdv`, 8×A100 SXM4, $11.92/hr.
`target_param_data_ratio=5` (~20% Chinchilla), `warmdown_ratio=0.3`, `lambda_gate=1e-3`, `gate_warmup_ratio=0.2`.

## Changes vs v8
- Train from scratch (no `--load_pretrained`) — fixes pre-training mismatch
- gate_cost normalised to gate_mean (÷ B×T×K) in model forward — λ=1e-3 now interpretable as nats/unit gate openness
- Gradient checkpointing removed — N/4 layers × 4 recur steps ≈ same activation memory as master
- Variable-K workarounds removed from mid_train (cache_size_limit=64, commented compile)
- SSH fix: `/start.sh &` + `sleep 15` at top of TRAIN_CMD

## Training results
- base_train completed: ~57 min, final loss ~3.15 nats
- mid_train + chat_sft completed; pushed to `Trelis/nanochat-gated-recursive` (base/d20, sft/d20)
- Total cost: ~2.5hr × $11.92/hr ≈ **~$30**

## Gate analysis
- gate_mean settled at ~0.09–0.10 during warmup and stayed there throughout
- Did not collapse to zero (unlike v8 where gates pegged at max)
- Did not become selective — no differentiation between easy/hard tokens
- λ ramps the entire last 80% of the 3131-step run, never plateauing at full strength
- Root cause: `target_param_data_ratio=5` too short — recurrence never had enough training to become CE-useful

## Bugs fixed
| Issue | Fix |
|-------|-----|
| λ=1e-3 collapsed gates instantly (gate penalty ~14× CE) | gate_cost was raw sum; normalised to gate_mean in model forward |
| Gradient checkpointing overhead (~33%) unnecessary | Removed — activation memory comparable to master without it |
| Container restarts after script completes | Added self-terminate: `curl -X DELETE .../pods/${RUNPOD_POD_ID}` at end of TRAIN_CMD |
| mid/sft unnecessarily ramp λ from 0 | Set `gate_warmup_ratio=0.0` for mid+sft (gates already trained) |

## Recommendations for v10
1. `target_param_data_ratio=20` — full Chinchilla, 12,525 steps (4× more than v9b)
2. `warmdown_ratio=0.2` — match master
3. `gate_warmup_ratio=0.0` for mid+sft
4. Watch gate_mean trajectory during warmup — if it rises from 0.09 as recurrence becomes useful, that's the signal we want
5. If gate_mean stays stuck at 0.09 all through warmup even in v10: try freezing gate_proj during warmup so recur blocks learn against open gates before λ kicks in (increasing bias init won't help — equilibrium is set by CE gradient)

---

# Gated Recursive Training — v10 Run Notes
Date: 2026-03-19 (active)

## What we ran
Full Chinchilla pipeline from scratch: base_train → mid_train → chat_sft.
Branch: `gated-recursive`. Pod: `qw9z4xqf03fcwx`, 8×A100, $11.92/hr.
`target_param_data_ratio=20`, `warmdown_ratio=0.2`, `lambda_gate=1e-3`, `gate_warmup_ratio=0.2` (base), `gate_warmup_ratio=0.0` (mid+sft).
12,525 steps base_train, ~8–9hr total. ETA ~$95–110.

## Status
- Pod started ~16:20 UTC 2026-03-19, initialising optimizer/compile
- Results TBD

---

# Gated Recursive Training — v11 Run Notes
Date: 2026-03-19 (active)

## What we ran
Full Chinchilla pipeline from scratch: base_train → mid_train → chat_sft.
Branch: `gated-recursive`. Pod: `ng5pqtr3vrlhcq`, 8×A100, $11.92/hr.
Same hyperparams as v10 (`target_param_data_ratio=20`, `lambda_gate=1e-3`, etc.) but with architectural fixes to the gate design.

## Changes vs v10

| Fix | Detail |
|-----|--------|
| `norm(u-s)` gate input | Raw `u-s` caused hard saturation (sigmoid→1.6e-15) after first Muon step on inject. inject=[I|0] init + zero c_proj means u-s=0 at init; after one Muon update inject changes by O(1), making u-s a large unscaled vector. `norm(u-s)` keeps gate input bounded (RMS≈1) throughout training. norm(0)=0 preserves sigmoid(bias)=0.88 at init. |
| Forced step-0 uses scalar multipliers | `gate_scale=0.0 if i==0 else 1.0` instead of `torch.ones_like` conditional. Functionally equivalent but uniform tensor ops across loop iterations. |
| Early exit from step 0 | Removed `i>0` guard from inference exit check. Model can now exit after a single recurrence at inference (minimum 1, not 2). gate_cost penalty unaffected (still excluded at step 0 via gate_scale=0). |

## Previous v11 pods (killed)
- `5ex219xmt2pu56`: gate_mean collapsed to ~1.6e-15 at step 2. Caused by unsaturated gate_proj input after first Muon step — fixed by norm(u-s).

## Status
- Pod killed early — gate_mean collapsed to ~1e-24 within first few hundred steps
- Root cause: CE gradient consistently drives gate_proj.bias negative before recur has learned anything useful (recur hasn't adapted yet → random u-s updates hurt CE → model prefers gates closed). norm(u-s) prevented instant saturation but not gradual drift.

---

# Gated Recursive Training — v12 Run Notes
Date: 2026-03-19 (active)

## What we ran
Full Chinchilla pipeline from scratch: base_train → mid_train → chat_sft.
Branch: `gated-recursive`. Pod: TBD, 8×A100, $11.92/hr.
Same hyperparams as v11 but with gate_proj frozen during warmup.

## Changes vs v11

| Fix | Detail |
|-----|--------|
| Freeze gate_proj during warmup | After backward and before optimizer.step(), gate_proj gradients are zeroed while lambda_t==0. This prevents gate_proj.bias from drifting negative before recur has learned useful representations. Once lambda kicks in (after warmup), gate_proj unfreezes automatically. |

## Status / Results
Pod: `2ojwof97su030d`, 8×A100, $11.92/hr. Running in parallel with v10.

**Warmup phase (steps 0–25 on wandb chart ≈ training steps 0–2505):** gate_mean stable at ~0.85–0.88. Freeze working correctly. Small drift during warmup is inject/recur changing what `norm(u-s)` looks like through frozen but non-zero gate_proj weights.

**At unfreeze (step ~25):** gate_mean jumps UP toward 1.0. Key positive signal — recur learned something useful during warmup (forced step-0 gradients did their job). CE strongly prefers recurrence, overcoming λ=1e-3.

**After full λ ramp (step ~60, training step ~6000):** gate_mean pinned at 1.0. λ=1e-3 too weak to close any gates — penalty is ~1e-3 nats vs CE gradient strongly preferring full recurrence. Unclear if gate_mean will move in second half of training. Running to completion to check per-task eval gate_mean differentiation.

**CE vs λ dynamics:** The soft gating design (`s = s + g*(u-s)`) means CE always benefits slightly from every recurrence step even at g≈0. No token has a clean "recurrence gives zero benefit" signal, so CE never strongly prefers g=0. λ needs to outweigh CE gradient at the token level, which requires larger λ than 1e-3 for this model.

---

# Gated Recursive Training — v13 Run Notes
Date: 2026-03-19 (active)

## What we ran
Same as v12 but with `lambda_gate=1e-2` (10× larger). Testing whether stronger λ can push gates off 1.0 and create token-level selectivity.
Pod: `4a25jrpxvljhjz`, 8×H100, $21.52/hr.

## Results
Pod killed early after confirming gate dynamics. gate_mean dropped to steady **~0.667** after λ ramp — exactly 2/3, consistent with one of three gated steps closing uniformly across all tokens.

**0.667 is suspiciously round:** with fixed_k=4 (3 gated steps), 2/3 open = 0.667. But the value is uniform across all tokens/positions, indicating the **bias is dominating** — the model found a global equilibrium where sigmoid(bias) = 0.667 rather than learning per-token selectivity. Weight contribution (which would create token-specific variation) is negligible.

This is λ and bias finding a lazy global equilibrium, not dynamic compute allocation.

**λ calibration:** one recurrence estimated worth ~0.01 nats val loss. λ=1e-2 is right at the boundary — consistent with one step closing. λ is correctly sized but the bias scalar is preventing per-token differentiation.

---

## Notes on bias and global equilibrium

With a trainable bias, `gate = sigmoid(weight @ norm(u-s) + bias)`. When weight is small, bias dominates and gate is nearly identical for all tokens. λ pushes bias to a global equilibrium (0.667 here) rather than forcing per-token decisions.

**Fix for v14: remove gate_proj bias.** Without bias, `gate = sigmoid(weight @ norm(u-s))` — must be token-specific from the start, no global scalar to lean on. Initialises at sigmoid(0)=0.5 but since gate_proj is frozen during warmup this doesn't matter. λ=1e-2 stays correct (same nats threshold argument applies).

---

# Gated Recursive Training — v14
Date: 2026-03-20

## Changes vs v13
- Remove `gate_proj` bias (`bias=False` in `nn.Linear`)
- Keep λ=1e-2, all other settings unchanged

## Results
gate_mean crashed to ~0 immediately after unfreeze. Same failure mode as v11: CE gradient drives gate_proj.weight negative before recur has learned anything useful, so all gates close.

**Root cause:** Without a bias floor, λ can drive `gate_proj.weight` to produce arbitrarily negative outputs → sigmoid→0 → gate collapses. With bias=True (v13), the bias stabilised at a global equilibrium (0.667); without it, there is no floor and λ wins completely. λ=1e-2 is strong enough to collapse the gate rather than find a partial equilibrium.

**Pattern across runs:**
- v11 (bias=True, no freeze): gate collapses to ~0 immediately
- v12 (bias=True, freeze): stable at 0.88 during freeze, jumps to 1.0 on unfreeze (CE dominates λ=1e-3)
- v13 (bias=True, freeze, λ=1e-2): stable at 0.667 — global bias equilibrium, no per-token selectivity
- v14 (bias=False, freeze, λ=1e-2): crashes to 0 after unfreeze — λ collapses weight with no bias floor

---

# Gated Recursive Training — v15
Date: 2026-03-20

## Changes vs v14
- Gate input changed from `norm(u-s)` to `cat([e, s])` — same input as inject
- Remove explicit gate freeze (no longer needed — see below)
- Keep λ ramp

## Motivation
`cat([e, s])` is semantically richer: gate asks "given this input (e) and my current state (s), should I recurse further?" rather than "is the proposed update large?". After step 0, s is a meaningful updated state, so the gate has real information. No degeneracy at init (cat([e,s]) always non-zero → gradient flows from step 1).

**Why drop the freeze:** With `cat([e,s])`, CE gradient flows to gate_proj from the start. During the warmup phase (λ=0), CE wants gates open, which is fine — recur is learning. As λ ramps up it pushes back. No sudden unfreeze transition, so no collapse risk. The freeze was a workaround for the u-s degeneracy; cat([e,s]) eliminates the need for it.

## Results
Gate crashed to 0 by step ~3. Root cause: gate_delay_ratio=0 means λ starts ramping from step 1, pushing toward closure before recur has learned anything. Even tiny λ at step 3 is enough to collapse the gate with a noisy recur.

Also renamed `gate_warmup_ratio` → `gate_delay_ratio` (warmup implies a ramp; this is a flat zero-λ period before the ramp).

---

# Gated Recursive Training — v16
Date: 2026-03-20

## Changes vs v15
- Restore `gate_delay_ratio=0.2`: λ=0 for first 20% of training before ramp begins
- Everything else unchanged: cat([e,s]) gate, bias=+2, weight=0, no explicit freeze

## Results
Gate still pushed to zero during the delay period (λ=0). CE alone is enough to collapse the gate — bias=+2 doesn't provide enough resistance against the CE gradient over 20% of training.

---

# Gated Recursive Training — v17
Date: 2026-03-20

## Changes vs v16
- Add freeze back: zero gate_proj grads when λ=0 (delay period)
- Everything else unchanged: cat([e,s]) gate, bias=+2, weight=0, gate_delay_ratio=0.2

## Motivation
CE alone collapses the gate during the delay period — bias=+2 is not enough resistance. Freeze locks gate_proj at init (0.88) while recur learns. When λ kicks in and gate unfreezes, CE and λ act together to find a real equilibrium. Now with cat([e,s]), the weight has a rich input signal to learn token-specific patterns, unlike v12/v13 where norm(u-s)≈0 caused global equilibrium.

## Results
Gate slammed to 0 immediately on unfreeze. λ=1e-2 dominates at the moment of unfreeze, collapses the gate before CE can establish per-token patterns.

---

# Gated Recursive Training — v18
Date: 2026-03-20

## Changes vs v17
- Drop freeze, delay, and ramp entirely — maximum simplicity
- λ=1e-3 constant from step 1 (10× weaker than v17)

## Motivation
λ=1e-2 wins every transition. With λ=1e-3 and no freeze/delay/ramp, CE keeps gates near 0.88 (bias=+2 anchor) while recur learns. As CE decreases, λ gains relative influence and hopefully finds a selective equilibrium. With cat([e,s]) providing real weight gradients from step 1, weight can develop per-token patterns without bias dominating.

## Results
Gate collapsed to ~0 by step 3. Same failure mode as v15/v16. λ=1e-3 is enough to collapse the gate before recur has learned anything, because sigmoid saturation creates a self-reinforcing fixed point: once g→0, gradient ∝ g*(1-g) → 0, so there is almost no signal to push g back open. λ keeps pushing down; CE can't push back up.

**Root cause confirmed across all versions:** sigmoid g=0 is a near-fixed point. Any λ>0 can collapse the gate before recur is useful, and once collapsed it can't recover because the gradient through sigmoid vanishes.

---

# Gated Recursive Training — v19
Date: 2026-03-20

## Changes vs v18
- **Leaky gate:** `g = gate_min + (1 - gate_min) * sigmoid(gate_proj(cat([e, s])))`, with `gate_min=0.1`
- g ∈ [0.1, 1.0] — model can never fully opt out of recurrence
- All other settings unchanged: cat([e,s]) gate, bias=+2, weight=0, λ=1e-3 constant, no freeze/delay/ramp

## Motivation
The sigmoid g=0 fixed point is a structural flaw: once the gate closes, the gradient vanishes and nothing can reopen it. Leaky gate fixes this directly:
- g ≥ 0.1 always → recur always contributes at least 10% of its update → recur always trains
- Gradient ∂g/∂x = (1-gate_min)*sigmoid'(x) > 0 always — recovery is always possible
- λ drives g toward gate_min, not toward 0 — minimum compute, not zero compute

## Results
Killed early — leaky gate doesn't address the root cause. Step 0 is always forced, so recur already always trains regardless of gate state on steps 1+. The leaky gate just floors gate_mean at 0.1 without changing the underlying incentive: CE has no reason to keep gated steps open beyond what step 0 already provides.

**Root cause of all collapses:** CE has no incentive to keep gated steps (1+) open — step 0 already provides the recurrence benefit. The gate input signal also matters: norm(u-s) is structurally limited to a scalar convergence measure and cannot distinguish token difficulty, which is why v13 found a global bias equilibrium (all tokens same gate value) rather than per-token selectivity.

---

# Gated Recursive Training — v20
Date: 2026-03-20

## Changes vs v19
- **Gate input: cat([u, s])** — "given proposed update and current state, should I recurse?"
  - Richer than norm(u-s): gate sees full vector space of u vs s, not just scalar magnitude
  - Can learn directional patterns that distinguish token difficulty (hard reasoning vs simple tokens)
  - norm(u-s) was structurally limited to global equilibrium; cat([u,s]) enables per-token selectivity
- **bias=0 init**: gates start at sigmoid(0)=0.5 during freeze period
  - Model co-adapts to partial recurrence (not nearly-full like bias=+2)
  - On unfreeze CE doesn't strongly pull toward 1.0 (v12 failure mode)
  - λ=1e-2 starts from 0.5, needs less distance to find equilibrium
- **Restore freeze** (gate_delay_ratio=0.2): gate_proj frozen while recur learns
- **Restore ramp** (gate_ramp_ratio=0.2): λ ramps gradually after freeze lifts
- **λ=1e-2**: same as v13 which found a partial equilibrium (0.667); now with better gate input

## Results
gate_mean stayed at 0 throughout base_train, mid_train, and SFT. Same collapse as every other from-scratch run. Killed.

**Key realization from reviewing W&B history:** Three consecutive pre-v11 runs all showed gate_mean ~0.10 in mid-training — a robust, consistent attractor. Most of these were mid-training runs starting from the pretrained `Trelis/nanochat-recursive` checkpoint (not from scratch). v9b (gated pre-train from scratch at ratio=5) also showed ~0.09-0.10. The v11 "fix" (norm(u-s)) broke what was working.

**What changed in v11 that broke the ~0.10 equilibrium:** norm(u-s) bounded the gate input, preventing the saturation that was creating per-token heterogeneity. With raw unscaled u-s, after the first Muon step on inject, u-s became a large token-specific vector → sigmoid saturated differently per token → natural ~0.10 average. Norming it destroyed this mechanism.

**v9b re-evaluation:** The original notes called ~0.09-0.10 a failure ("did not become selective"). This may have been wrong — ~0.10 could be the correct equilibrium for this λ calibration. The runs were not allowed to run long enough to see whether per-task differentiation emerged.

## Launch command
```bash
uv run runpod/launch_pretrain.py --version v20 --lambda-gate 1e-2 --name nanochat-gated-v20
```

---

# Gated Recursive Training — v21
Date: 2026-03-20

## Changes vs v20
- **Gate input: cat([u, s])** — same as v20 (keep)
- **bias=+2 init**: gates start at sigmoid(2)≈0.88 (near-open), not 0.5
  - Pretrained model co-adapted to full recurrence (fixed_k=4 throughout pre-training)
  - bias=+2 is closer to what pretrained model expects, causes less disruption
  - λ drives toward selective equilibrium from open, not from 0.5
- **weight=0 init**: no token-specific signal at init, weight learns once recur produces meaningful u
- **Freeze gate_proj during delay (gate_delay_ratio=0.2)**: λ=0 and gate frozen for first 20%
  - Prevents CE from collapsing gate before recur has learned to produce useful u
- **gate_ramp_ratio=0.2**: λ ramps from 0→1e-3 over next 20%
- **λ=1e-3 constant**: gentler than v20's 1e-2; targets ~0.10 equilibrium seen in pre-v11 runs
- **MID+SFT ONLY from pretrained checkpoint**: `Trelis/nanochat-recursive` base
  - All pre-v11 successes (~0.10 equilibrium) were mid+SFT from pretrained checkpoint
  - From-scratch pre-training always collapses gate before recur learns: u≈s at init (c_proj=0), CE indifferent, λ wins
  - Pretrained model has meaningful u≠s from step 1 → gate gets real per-token signal immediately

## Motivation
Three consecutive pre-v11 runs showed gate_mean ~0.10 in mid-training — a robust attractor. All from mid-training of `Trelis/nanochat-recursive`. v11 "fix" broke this. v21 replicates those conditions with cat([u,s]) instead of norm(u-s) and adds bias=+2/freeze/ramp.

## Pod restart bug fix (v21)
RUNPOD_API_KEY was not forwarded to pod in launch_mid_sft.py → self-terminate curl failed silently → pod restarted and re-ran training. Fixed: RUNPOD_API_KEY now forwarded in env.

## Launch command
```bash
uv run runpod/launch_mid_sft.py --version v21 --name nanochat-gated-v21
```

---

# Architecture Improvement Ideas

Two candidates for step 11:

## Option A: Freeze gate_proj during λ warmup

Don't allow gate_proj to update until λ kicks in. This keeps gates open during warmup so recur blocks learn against a consistent open-gate input distribution, before gating pressure is applied. Pending v10 results — if gate_mean stays stuck at ~0.09 all through warmup, this is the fix.

## Option B: Gate conditioned on u-s instead of s ✅ Implemented in v11

**Was:** `g = sigmoid(gate_proj(s))` — gate computed before recurrence from current state.

**Now:** always run recur step 0 with a full forced update (`s = u`), then for steps 1+ compute `g = sigmoid(gate_proj(u - s))`. Small `u-s` means state barely changed → converged → close gate.

Benefits:
- Recur blocks always get full gradients for step 0 (g=1 forced) — breaks the chicken-and-egg
- Gate signal is more natural: "how much did this step want to change things?"
- At inference: always pay one recur cost, then gate decides whether to continue

**gate_mean normalisation:** averaged over steps 1..(K-1) only (step 0 is always open, not counted).

## Option C: Forced full first recurrence ✅ Implemented in v11

For step 0, `s = u` always (no gate). For steps 1+, `g = sigmoid(gate_proj(u - s))`. This ensures:
1. Recur blocks always receive full-strength gradients on the first pass
2. The gate only needs to learn "is another recurrence worth it?" — a simpler task
3. The chicken-and-egg problem (gates close before recurrence becomes useful) is eliminated for step 0

This is now the default implementation (combined with Option B).

---

# Key Design Insights (v11 discussion)

## Gate mechanics summary

- `g` is a per-token scalar in (0,1), recomputed fresh at each recurrence step — not one value shared across all steps
- Step 0: always `s = u` (forced full update, g=1). Steps 1+: `g = sigmoid(gate_proj(u - s))`
- At **inference**: step 0 always runs; after each subsequent step, if `g.max() < gate_threshold` exit early
- At **training**: no early exit (gate_threshold has zero effect); all K steps always run; gradient to recur blocks scaled by `g`
- `gate_threshold=0.01` is inference-only, tunable post-training as speed/quality tradeoff
- `gate_mean` is normalised over steps 1..(K-1) only (step 0 not counted, always open)

## Why gradient scaling matters

Even with forced step 0, steps 1-3 receive gradients scaled by `g`. If `g ≈ 0.09`, recur blocks learn at ~9% gradient magnitude for those steps. The forced step 0 breaks the chicken-and-egg for the first recurrence but steps 1+ still face weaker gradients until recurrence becomes demonstrably useful.

## Baseline: recursive (from scratch, no gating)

All recursive results are from-scratch training, not continued fine-tuning:

| Metric | d20 | r=2 | r=4 | r=8 | r=16 |
|--------|-----|-----|-----|-----|------|
| ARC-Easy | 0.4630 | 0.4141 | 0.4306 | 0.4423 | 0.4381 |
| ARC-Challenge | 0.3234 | 0.3063 | 0.3114 | 0.3106 | 0.3123 |
| MMLU | 0.3222 | 0.3119 | 0.3158 | 0.3185 | 0.3179 |
| GSM8K | 0.0508 | 0.0356 | 0.0614 | 0.0599 | 0.0644 |
| HumanEval | 0.1220 | 0.0793 | 0.0793 | 0.0915 | 0.0793 |
| SpellingBee | 0.9883 | 0.9844 | 0.9883 | 0.9883 | 0.9844 |
| ChatCORE | 0.2732 | 0.2459 | 0.2566 | 0.2614 | 0.2588 |

Recursion (uncontrolled, r=4) hurts on most tasks but helps on GSM8K — shared weights have lower representational capacity than unique layers, except for multi-step reasoning where recurrence adds value.

## Right success metric for gated model

The goal is **not** to beat d20 overall (fewer unique weights makes that unlikely). The goal is to show **gate_mean correlates with task difficulty**:
- Hard reasoning tasks (GSM8K, HumanEval) → higher gate_mean (model wants more recurrences)
- Simple tasks (SpellingBee, easy ARC) → lower gate_mean (model exits early)

If this signal is present, the architecture is demonstrating dynamic compute allocation — a meaningful research result independent of aggregate benchmark scores. Follow-on: log gate_mean **per eval task** rather than just globally.

