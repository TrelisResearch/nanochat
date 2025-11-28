import math
import random

import torch
import torch.nn.functional as F

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from contextlib import nullcontext


class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.encode_special("<|mask|>")
        self.assistant_end = tokenizer.encode_special("<|assistant_end|>")

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        num_samples=1,
        response_length=256,
        diffusion_steps=32,
        temperature=0.0,
        top_k=None,
        seed=None,
    ):
        assert num_samples == 1, "Diffusion sampler currently supports num_samples=1"
        response = self._diffusion_sample(
            tokens,
            response_length=response_length,
            diffusion_steps=diffusion_steps,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
        for token in response:
            yield [token], [1]

    def _diffusion_sample(self, prompt_tokens, response_length, diffusion_steps, temperature, top_k, seed):
        device = self.model.get_device()
        prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        response = torch.full((1, response_length), self.mask_token_id, dtype=torch.long, device=device)
        full = torch.cat([prompt, response], dim=1)
        # mask indicates which positions remain masked during the reverse process
        mask = torch.zeros_like(full, dtype=torch.bool)
        resp_start = prompt.size(1)
        resp_end = resp_start + response_length
        resp_slice = slice(resp_start, resp_end)
        mask[:, resp_slice] = True
        total_slots = response_length
        rng = torch.Generator(device=device)
        if seed is None:
            seed = random.randrange(0, 2**63 - 1)
        rng.manual_seed(seed)
        for step in range(diffusion_steps, 0, -1):
            logits = self.model(full)
            resp_logits = logits[:, resp_slice, :]
            if temperature > 0:
                logits_mod = resp_logits / temperature
                if top_k is not None:
                    k = min(top_k, logits_mod.size(-1))
                    top_vals, _ = torch.topk(logits_mod, k, dim=-1)
                    cutoff = top_vals[..., -1, None]
                    logits_mod = logits_mod.masked_fill(logits_mod < cutoff, float("-inf"))
                probs = F.softmax(logits_mod, dim=-1)
                preds = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1, generator=rng)
                preds = preds.view(1, total_slots)
            else:
                preds = torch.argmax(resp_logits, dim=-1)
                probs = F.softmax(resp_logits, dim=-1)
            resp_mask = mask[:, resp_slice]
            full[:, resp_slice] = torch.where(resp_mask, preds, full[:, resp_slice])
            conf = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)
            desired_mask = math.floor((step - 1) / max(diffusion_steps, 1) * total_slots)
            if desired_mask <= 0:
                mask[:, resp_slice] = False
            else:
                values, indices = torch.sort(conf[0])
                selected = indices[:desired_mask] + resp_start
                mask.zero_()
                mask[0, selected] = True
        response_tokens = full[0, resp_slice].tolist()
        if self.assistant_end in response_tokens:
            cutoff = response_tokens.index(self.assistant_end)
            response_tokens = response_tokens[:cutoff + 1]
        return response_tokens

    def generate_batch(self, tokens, num_samples=1, seed=None, **kwargs):
        results = []
        result_masks = []
        for sample_idx in range(num_samples):
            seq = tokens.copy()
            mask_seq = [0] * len(tokens)
            # Advance the RNG seed for each completion so sampling paths stay independent.
            sample_seed = seed + sample_idx if seed is not None else random.randrange(0, 2**63 - 1)
            for token_column, token_masks in self.generate(tokens, num_samples=1, seed=sample_seed, **kwargs):
                seq.append(token_column[0])
                mask_seq.append(token_masks[0])
            results.append(seq)
            result_masks.append(mask_seq)
        return results, result_masks


if __name__ == "__main__":
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = autodetect_device_type()
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    engine = Engine(model, tokenizer)
    for token_column, _ in engine.generate(prompt_tokens, response_length=64, diffusion_steps=16):
        chunk = tokenizer.decode(token_column)
        print(chunk, end="", flush=True)
    print()
