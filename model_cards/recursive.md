---
license: apache-2.0
language:
- en
---
# Recursive NanoChat d20 (i.e. 20 layers effective depth, ~$100 model)

A recursive tranformer implementation of NanoChat.

See [https://github.com/TrelisResearch/nanochat]() for details on how to download and run.

## Design Choices
```yaml
- (P, R, C) = (2, 4, 2) → 8 unique layer weights # 2 prelude layers, 4 recursive and 2 coda
- train_recur_mean = 4.0 → effective depth 20 (matches original depth=20) # 2 + 4*4 + 2 = 20
- train_recur_max = 16 # Number of recurrences during training are sampled from a Poisson log-normal distribution (σ=0.5) with a mean of train_recur_mean = 4.0
- bptt_k = 4 → gradient flows through max 16 recur layers # So that the mean case has full back prop but higher level of recurrence are truncated
- inject_mode = "concat_linear" (learned adapter, identity-initialized) # the recycle stream is concatenated with inputs and passed through a shrinking linear layer; identity-init ensures gradients flow
- recur_warm_start = True # the recycle stream is zero initialised for the first token generated, but the next token borrows the last state from the previous token, accelerating inference
- kv_cache_recur_budget = 1 (cache only final recurrence) # the final recurrence state is always used for later tokens, saving memory and assisting accuracy
- Sampling: Poisson log-normal distribution (σ=0.5) # this was used for pre-, mid- and sft- training. Although perhaps it would have been better to use a constant number of recursions for the mid and post training, which would also allow torch compile to be used for mid-training.
```

## Results

### SFT
**NanoChat d20:**
ARC-Easy: 0.4630
ARC-Challenge: 0.3234
MMLU: 0.3222
GSM8K: 0.0508
HumanEval: 0.1220
SpellingBee: 0.9883
ChatCORE metric: 0.2732

**NanoChat Recursive - num_recur=2:**


**NanoChat Recursive - num_recur=4:**


**NanoChat Recursive - num_recur=8:**


**NanoChat Recursive - num_recur=16:**