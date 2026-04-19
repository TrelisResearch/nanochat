# GSM8K — Prefill / Decode Recurrence Benchmark

Branch: `prefill-recur-bench`
Base model: `Trelis/nanochat-recursive` — `sft/d20`, step 700 (328.3M params, P=2, R=4, C=2, fixed_k=4)
Hardware: single H100 (Modal `dev-ronan`)
Dataset: GSM8K (`main/test`), 128–256 problems per config (fixed seed shuffle)
Decoding: greedy (T=0.0), max_new_tokens=512, calculator tool enabled
Dates: 2026-04-19

## TL;DR

1. **Decode-time recurrence drives GSM8K accuracy.** On the untouched `Trelis/nanochat-recursive` SFT checkpoint: r=1 → 0.39%, r=2 → 3.91%, r=4 → 7.42% (matches the model-card numbers within noise).
2. **Prefill-only-deep, decode-shallow doesn't transfer.** All split configs (prefill∈{2,4}, decode=1, keep∈{first,last}) land at or below the r=1 baseline on the base model. Notably `split_p4_d1_last` = 0.00%. The result is robust: it is **not** an artefact of the inference engine's `warm_start_state` (benching with `--no-warm-start` gives the same 0.00%).
3. **SFT training on the split regime does not fix this.** Three training experiments (row-by-row, batched depth_mask, and 50/50 mixed) all produced checkpoints with `split_p4_d1_last` ≤ 0.78% — indistinguishable from the baseline at this sample size. Meanwhile, the splits **catastrophically forgot** the original full-depth capability: r=2 and r=4 accuracy both collapsed to 0.00% on every trained checkpoint.
4. **The model found a "cheat" minimum.** On all trained checkpoints, r=1 and every `keep=first` config converged to identical accuracy (2.34% on d20_depth/no-warm-start). The model is **not** using the deep prefix K/V — it adapted to decode-at-depth-1-everywhere, ignoring prefill depth.

Net: without pretraining-time support for variable depth, you can't retrofit "think deep in prefill, sprint in decode" on this recursive architecture via SFT alone.

## Full results matrix

256 problems unless noted. 128-problem columns rerun with `--no-warm-start` for direct comparison.

| Config                | d20 (orig, warm) | d20 (orig, no-warm) | d20_split_v2 (row, 300 iter) | d20_depth (batched, 1000 iter) | d20_mixed (50/50, 1000 iter) |
|-----------------------|:---------------:|:-------------------:|:----------------------------:|:------------------------------:|:----------------------------:|
| full_r1               |      0.39%      |         –           |             –                |          **2.34%** ¹           |        **2.34%** ¹           |
| full_r2               |      **3.91%**  |         –           |             –                |            0.00% ⚠             |          0.00% ⚠             |
| full_r4               |      **7.42%**  |         –           |             –                |            0.00% ⚠             |          0.00% ⚠             |
| split_p2_d1_last      |      0.78%      |         –           |             –                |            0.00%               |          0.00%               |
| split_p2_d1_first     |      0.39%      |         –           |             –                |            2.34% ¹             |          2.34% ¹             |
| split_p4_d1_last      |      0.00%      |       0.00%         |           0.78% (2/256)      |            0.00%               |          0.00%               |
| split_p4_d1_first     |      0.39%      |       0.78%         |           0.39% (1/256)      |            2.34% ¹             |          2.34% ¹             |
| split_p4_d2_last      |      0.00% ²    |         –           |             –                |             –                  |            –                 |
| split_p4_d2_first     |      0.00% ²    |         –           |             –                |             –                  |            –                 |

¹ — measured at N=128 with `--no-warm-start`. The original bench was at N=256 with warm_start; on the trained checkpoints both dropped to 0.78% with warm_start (engine mismatch), climbing to 2.34% when inference matched the training regime.
² — measured at N=256 on the **untouched baseline** `d20` to test whether a decode depth of 2 (instead of 1) can recover performance under `prefill_kv_keep=last`. It cannot: 0/256 on both keep modes, and generation runs to the max_tokens cap (~800 s vs 360 s for full_r4), indicating responses are incoherent rather than just wrong.
⚠ — catastrophic forgetting. Generation often runs to the max_tokens cap.

## What we did

The high-level pipeline: bench the base model's split-recurrence behavior → verify via `--no-warm-start` that the engine isn't hiding the signal → SFT-train against the split regime → re-bench.

### Training experiment A: row-by-row two-stage (d20_split_v2)
- For each batch row, run a stage-1 prefill forward over prompt tokens at `num_recur=4` with `prefill_kv_keep=last`, then a stage-2 decode forward over response tokens at `num_recur=1`. Loss on response tokens only.
- 300 iterations × 16 rows per step = 4.8k examples at `init_lr_frac=0.02`.
- Val loss: 1.2659 → 1.2653 (essentially flat). `split_p4_d1_last` moved from 0.00% → 0.78% (2/256) — one-problem noise-level bump.

### Training experiment B: batched single-forward via `depth_mask` (d20_depth)
- Added a `depth_mask` kwarg to `GPT.forward` expressing per-position recurrence depth. Tests (`test_depth_mask.py`) verify mathematical equivalence to the two-stage KVCache path.
- 1000 iterations × 32 rows per step = 32k examples at `init_lr_frac=0.02`. 10 min on H100.
- Val loss: split=1.3026 → 1.2975, full=1.1234 → 1.3066 (full regime broke by step 100).
- GSM8K: `split_p4_d1_last` still 0.00%. Full r=2/r=4 destroyed.

### Training experiment C: 50/50 mixed regime (d20_mixed)
- Same as B, but each row randomly picks "split" or "uniform-depth" schedule with probability 0.5. Hope: uniform batches preserve full-depth capability while split batches teach the split regime.
- 1000 iterations × 32 rows per step, same hyperparameters.
- Val loss: split=1.3026 → 1.2978, full=1.1234 → 1.3028 (full regime still broke, just as fast).
- GSM8K: identical failure mode to experiment B.

### Diagnostic: did we find a cheat minimum?
On both trained checkpoints, `full_r1`, `split_p2_d1_first`, and `split_p4_d1_first` all converged to exactly 2.34% (3/128). Since these three configs differ in how much prefill compute is spent but **share the property that decode only sees near-iter-0 prefix K/V**, their identical accuracy is evidence that the model learned to ignore the deep-prefill K/V entirely and lean on its own iter-1 decode.

Compare to `split_p2_d1_last` / `split_p4_d1_last`, which force decode to attend to iter-3 prefix K/V — both remained at 0.00%. The model *cannot* make use of that K/V after SFT.

## Sharpened finding: cross-depth attention only works when decode *catches up* to prompt depth

A follow-up sanity check: run the untouched baseline at prefill=4, decode=**2** (instead of decode=1). If "decode=1 is simply too shallow for math" were the whole story, decode=2 should recover some accuracy — full_r2 manages 3.91% with decode depth 2. But on the baseline:

- `split_p4_d2_last` → 0.00% (0/256), 807 s
- `split_p4_d2_first` → 0.00% (0/256), 801 s
- `full_r2` (same decode depth but prompt K/V also at iter-1) → 3.91%, 220 s

`split_p4_d2` and `full_r2` have **identical coda input** in terms of depth — response's `s_iter_1`. The only difference is the prompt K/V the response attends to during its 2 iters: iter-3 in the split, iter-1 in full_r2. Feeding the response iter-0/iter-1 queries *more-processed* prompt K/V actively hurts — 0% vs 3.91%.

This revises an earlier claim. I said "cross-depth attention works at inference for free" because `full_r4` decode has iter-0 queries attending to iter-3 K/V and performs well. What's actually true: this cross-depth pairing is only harmless when the decode *continues iterating* to match the prompt's depth before coda reads `s`. Iter-0 queries against iter-3 keys are a transient intermediate state on the way to iter-3 queries against iter-3 keys. Stop early and coda reads an `s` that was evolved under a depth mismatch it was never trained to terminate at — and that `s` is unusable for generation (outputs run to the max_tokens cap).

So the specific mechanism "use later-iter K/V of earlier tokens to shortcut decode" *is* what we were hoping would be free, but empirically on this checkpoint it isn't. The baseline cannot decode coherently at any depth < prompt_k/v_depth.

## Why the SFT attempts fail — a hypothesis

The base checkpoint was trained (base + mid + SFT) with Poisson-sampled but **uniform-depth-per-sequence** recurrence. At no point during pretraining did any iter-1 query ever attend to iter-k≥2 K/V. The attention-space mapping from "iter-k K/V" to "useful information" was learned *per iter k*, with the implicit assumption that queries at iter k see K/V at iter k.

When SFT tries to couple iter-1 queries with iter-3 K/V, the model has two roads to reduce loss:
- **Hard road (the one we want):** learn a cross-iteration K/V geometry — make iter-3 K/V readable by iter-1 queries. Requires shifting many weights.
- **Easy road (the one we got):** make iter-1 decode work well *on its own* regardless of the K/V in cache. The decoder's lm_head and coda layers can be nudged to produce reasonable next tokens from just the current token's iter-1 state, ignoring prefill depth.

The easy road costs fewer bits of capacity to find and is what SFT gradient descent converges to. Evidence: trained r=1 ≈ all keep=first configs ≈ all keep=last configs on the *decode* side (they all land at 2.34%); the K/V depth simply doesn't matter to the trained model.

Catastrophic forgetting (r=2, r=4 going to 0%) is the corollary: the coda/lm_head's new iter-1-centric mapping is incompatible with the iter-k-centric one the base model had.

## Implementation notes

Code on branch `prefill-recur-bench`:

- `nanochat/gpt.py` — `forward` now supports:
  - `prefill_kv_keep ∈ {last, first}` — which recurrence's K/V to persist in the cache (inference/bench).
  - `depth_mask: (B, T) long` — per-position recurrence depth for a single-forward batched training path.
  - Selective cache writes during the recurrence loop so autograd doesn't trip over in-place K/V overwrites when training through a KVCache.
- `nanochat/engine.py` — `Engine.generate` accepts `num_recur_prefill`, `num_recur_decode`, `prefill_kv_keep`, `use_warm_start`. `KVCache.insert_kv` clones returned K/V views when `torch.is_grad_enabled()` so backward can't be invalidated by later in-place writes.
- `scripts/bench_gsm8k.py` — single-GPU bench driver with the matrix in this report. `--no-warm-start` for training-regime-matched inference.
- `scripts/chat_sft_split.py` — row-by-row two-stage SFT (experiment A).
- `scripts/chat_sft_depth.py` — batched single-forward SFT with `depth_mask` (experiments B and C). `--split-prob` controls mixed regime.
- `modal_bench_gsm8k.py`, `modal_train_split_sft.py`, `modal_train_depth_sft.py` — Modal runners (dev-ronan env, H100).
- Tests: `test_prefill_kv_keep.py`, `test_split_train_grad.py`, `test_depth_mask.py` (35 tests pass on CPU).

## Suggested next steps

The sharpened finding (baseline can't decode at depth < prompt depth, at decode=1 *or* decode=2) suggests the SFT-scale interventions would have to teach something the checkpoint is structurally unprepared to do. Things worth trying if someone wants to push this further, roughly ordered by cost-effectiveness:

1. **Call it.** Best evidence we have is that this is a pretraining-era decision for this checkpoint. Further SFT tuning on this model is likely to keep producing "decoder works at iter-1-everywhere" solutions (which is what we got) rather than "decoder leverages deep prompt K/V at shallow decode" (which it can't be coerced into on this budget).
2. **Continued pretraining on large data with curriculum (McLeish recipe).** 10–50B tokens on FineWeb-Edu + math, Poisson-Lognormal depth with curriculum from low mean → 16 or 32. This is the established path to depth-robust recurrent models; they explicitly show it works from pretrained initialisations. Expensive but the clean answer.
3. **Pretrain variable-depth from scratch.** Either per-sequence (as McLeish / Geiping do) or per-position depth randomisation. Gives the model "any depth works at any depth" semantics built in. Most expensive, most powerful.
4. **Freeze lm_head / coda during split SFT.** Force the adaptation onto the recur-block K/V projections and stop the decoder from finding the iter-1-everywhere shortcut. Combined with mixed regime this might close the cheat door at small scale.
5. **Auxiliary contrastive loss.** Penalise the model when decode logits are invariant to keep=first vs keep=last. Directly punishes the cheat pattern. Could be bolted onto any of the above.

## Files

- `bench_results/REPORT.md` — this document
- `bench_results/bench_gsm8k_*.json` — raw bench JSONs (initial matrix)
- `bench_results/bench_d20_depth_full.log` — full matrix on depth-trained checkpoint
- `bench_results/bench_d20_mixed.log` — mixed-regime checkpoint bench
- `bench_results/full_train_*.log` — training logs for experiments A / B / C
- Modal run dashboard: https://modal.com/apps/trelisresearch/dev-ronan
