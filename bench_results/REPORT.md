# GSM8K — Prefill / Decode Recurrence Benchmark

Branch: `prefill-recur-bench`
Model: `Trelis/nanochat-recursive` — `sft/d20`, step 700 (328.3M params, P=2, R=4, C=2, fixed_k=4)
Hardware: single H100 (Modal `dev-ronan`)
Dataset: GSM8K (`main/test`), 256 problems per config (fixed seed shuffle)
Decoding: greedy (T=0.0), max_new_tokens=512, calculator tool enabled
Date: 2026-04-19

## TL;DR

1. **Decode-time recurrence is what drives GSM8K accuracy.** r=1 → 0.39%, r=2 → 3.91%, r=4 → 7.42%. This matches the published `Trelis/nanochat-recursive` model-card numbers (r=2: 3.56%, r=4: 6.14%) within sampling noise on 256 vs 1319 problems.
2. **Running more recurrences only during prefill (decode=1) does not help and can hurt.** All four split configs (prefill∈{2,4} × keep∈{first,last}) land at or below the r=1 baseline of 0.39%. In particular, `prefill=4, keep=last` drops to 0.00% — the iter-4 K/V cache is *actively misleading* for iter-1 decode queries.
3. **The wall-clock savings from cheap decode are modest** because decode time already dominates only moderately for this model size / completion length. Fastest accurate config remains full r=2 (220 s / 256 problems, 3.91%). Full r=4 costs 360 s for 7.42%.

## Results

| Config                   | Accuracy | N pass | Wall-clock (s) | s / problem |
|--------------------------|---------:|-------:|---------------:|------------:|
| **full_r1**              |    0.39% |  1/256 |          258.6 |        1.01 |
| **full_r2**              |    3.91% | 10/256 |          220.3 |        0.86 |
| **full_r4**              |    7.42% | 19/256 |          360.2 |        1.41 |
| split_p2_d1_**last**     |    0.78% |  2/256 |          318.2 |        1.24 |
| split_p2_d1_**first**    |    0.39% |  1/256 |          253.9 |        0.99 |
| split_p4_d1_**last**     |    0.00% |  0/256 |          339.0 |        1.32 |
| split_p4_d1_**first**    |    0.39% |  1/256 |          256.2 |        1.00 |

(*`split_pK_d1_keep` = prefill uses K recurrences, decode uses 1, recur-layer K/V left in cache from {first, last} prefill iteration.*)

### Sanity vs. model card (full-recur configs)

| r | This run (N=256) | Card (N=1319) |
|---|-----------------:|--------------:|
| 2 |             3.91% |          3.56% |
| 4 |             7.42% |          6.14% |

Within expected noise for 256-problem subsample — validates the eval plumbing.

## Interpretation

### Why split collapses to r=1

Each recur iteration produces different K/V for the recur layers (because the block input `u` changes as the recurrent state `s` evolves). The model was trained with a **fixed** recurrence count (`fixed_k=4`, or a Poisson sampled mean of 4 in pretraining). Its decoder queries at iteration-1 have learned to attend to K/V that correspond to **their own iteration depth**. When you drop decode to 1 recurrence but leave higher-iter K/V in cache:

- **keep=first** ≈ pure r=1 (0.39%). The cache is overwritten with iter-0 K/V, so decode sees what it would have seen if prefill had only done 1 recurrence. Consistent → matches r=1.
- **keep=last** is a cross-iteration mismatch: iter-1 queries attending to iter-P K/V. Sometimes marginally better (p=2, 0.78%), sometimes actively worse (p=4, 0.00%). No signal that the "extra processing" in prefill's deeper iterations benefits single-iter decode.

### Why wall-clock savings are small

Rough cost per problem ≈ prefill_cost(P) + decode_cost(D) × n_tokens.
Prefill is a single pass over ~100-200 prompt tokens; decode runs ~50-150 steps for correct answers, more for wrong/rambly answers. So decode dominates, and total time tracks decode recurrence count much more than prefill recurrence count.

Observed: full_r2 was actually the **fastest** config at 220 s (faster than full_r1's 259 s!) because r=2 generates correct answers sooner and stops at `<|assistant_end|>` — r=1 keeps rambling to the max_tokens cap on most problems.

### One-line summary
This recursive model uses decode-time iteration as its load-bearing compute; pushing that compute to prefill-only doesn't transfer. A "cheap decode via prefill offload" regime would require training the model to tolerate mismatched prefill/decode depths (e.g. depth-agnostic training, or explicit depth dropout) — something the current checkpoint was never exposed to.

## Implementation notes

Code changes (see `prefill-recur-bench` branch):

- `nanochat/gpt.py` — `forward` gets a `prefill_kv_keep="last"|"first"` kwarg. With `"first"` and `num_recur>1` and a cache present, the recur-layer K/V slice and `s` are snapshotted after iteration 0 and restored after the loop. No behavioural change when `kv_cache is None` or `num_recur==1`. Note that `gate_min=0.0` for this checkpoint (non-gated recursive), so the gating path is effectively disabled.
- `nanochat/engine.py` — `Engine.generate` gains `num_recur_prefill`, `num_recur_decode`, and `prefill_kv_keep`; the prefill forward and the decode forward now take independent recurrence counts. Backwards compatible via the existing `num_recur` kwarg.
- `scripts/bench_gsm8k.py` — single-GPU benchmark script. Sweeps the full × split matrix and records accuracy and wall-clock per config.
- `modal_bench_gsm8k.py` — Modal runner targeting `dev-ronan`, single H100, pulls Trelis/nanochat-recursive on first run and caches the model on a named Volume.
- `tests/test_prefill_kv_keep.py` — three unit tests verifying the K/V snapshot/restore invariants. All pass on CPU in <2 s.

## Files

- Raw JSON: `bench_results/bench_gsm8k_20260419_011901.json`
- Full log: `bench_results/matrix_256.log`
- Modal run URL: https://modal.com/apps/trelisresearch/dev-ronan/ap-M321NC1UeFkM5EoIQjtjj4

## Suggested next steps

1. **Train a depth-tolerant variant.** Randomly mask recur steps during SFT (e.g. randomly drop to k∈{1,2,4} per batch) to teach the model to produce K/V that behaves under mismatched decode depth. Then retry the split matrix — if it closes the gap, the "prefill-only deep think" idea becomes viable.
2. **Measure prefill-only vs decode-only cost directly.** Current timing lumps both together. A micro-benchmark on the two forward calls (T=200 prefill vs 100 × T=1 decode calls) would isolate where the budget actually goes and tell us the ceiling for split-mode speedup.
3. **Expand the full-recur sweep with r=8, r=16.** The card shows r=8 ≈ r=4 and r=16 barely moves — would be interesting to reproduce on a newer checkpoint, but not critical for the split question.
